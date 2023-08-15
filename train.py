import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import BCEWithLogitsLoss
from ultralytics.nn.tasks import ClassificationModel
import yaml
#custom
from config.fundusdataset import Fundusdataset
from config.models import SPNet, ResGCNet, AsymmetricLossOptimized, initialize_model

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, dest='mname', default='yolov8', help='deep learning model will be used')
    parser.add_argument("--optim", type=str, default='AdamW', help='optimizer')
    parser.add_argument("--in_channel", type=int, default=3, dest="inch",help="the number of input channel of model")
    parser.add_argument("--img_size", type=int, default=512,help="image size")
    parser.add_argument("--nclass", type=int, default=5,help="the number of class for classification task")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--epoches", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=36, help="Batch size during training")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for optimizer")
    parser.add_argument("--threshold", type=float, default=0.5, dest="thresh",help="the threshold of that predicting if belong the class")
    parser.add_argument("--weight_path", type=str,dest='wpath', default='.\\best.pth', help="path of model we trained best")

    return parser.parse_args()

def main():
    #設置Logging架構
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    log_filename = 'log.txt'
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    hparam = get_args()

    #資料隨機分選訓練、測試集
    pipe = transforms.Compose([transforms.Resize((hparam.img_size,hparam.img_size))])
    trainset = Fundusdataset("train",transforms=pipe)
    testset = Fundusdataset("test",transforms=pipe)
    print(trainset.class2ndx)
    print(testset.class2ndx)

    # model = get_model(model_name=hparam.mname,in_ch=hparam.inch,img_shape=(hparam.img_size,hparam.img_size),num_class=hparam.nclass)
    model = initialize_model(hparam.mname,num_classes=hparam.nclass,use_pretrained=True)
    
    logging.info(model)
    optimizer = get_optim(optim_name=hparam.optim,model=model,lr=hparam.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criteria = BCEWithLogitsLoss()
    # criteria = AsymmetricLossOptimized()

    #training
    history = training(model,trainset, testset, criteria, optimizer,lr_scheduler, device, hparam)
    #plotting
    plot_history(history['train_history'],history['validationn_history'],saved=True)

    fig, ax = plt.subplots(1,1)
    ax.plot(torch.arange(1,hparam.epoches+1,dtype=torch.int64),history['acc_history'])
    ax.grid(visible=True)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"accuracy History")
    fig.savefig("accuracy")

    # evaluate performance of testset
    # best_model = get_model(model_name=hparam.mname,in_ch=hparam.inch,num_class=hparam.nclass)
    # best_model.load_state_dict(torch.load(hparam.wpath))

    return

def evaluate(model,dataset, loss_fn,device,hparam):
    model.eval()
    model = model.to(device)
    total_loss = 0
    total_acc = []
    dataloader = DataLoader(dataset=dataset,batch_size=hparam.batch_size,shuffle=False)
    for img_data,labels in dataloader:
        img_data = img_data.to(device)
        labels = labels.to(device).squeeze() #變為一軸
        logits = model(img_data)
        loss = loss_fn(logits,labels)
        total_loss += loss.item()
        predict = torch.sigmoid(logits.detach()).squeeze()
        mask = (predict > hparam.thresh).to(torch.int64) # mask 中值為True即為預測值
        labels = labels.to(torch.int64)
        batch_acc = (torch.sum(mask & labels).item()) / (torch.sum(mask | labels).item())
        # print(f"after sigmoid: \n{predict}")
        # print(f"predicts: \n{mask}")
        # print(f"label: \n{labels}")
        # print(f"predicts & labels: \n{mask & labels}")
        # print(f"mask | labels: \n{mask | labels}")
        # print(f"batch_acc: {batch_acc}")
        # print(f"="*40)
        total_acc.append(batch_acc)

    mean_loss = total_loss/ ((len(dataset)//hparam.batch_size)+1)
    acc = round(sum(total_acc) / len(total_acc), 4)

    return mean_loss, acc

def training(model, trainset, testset, loss_fn, optimizer,lr_scheduler, device, hparam):
    model = model.to(device)
    dataloader = DataLoader(trainset,batch_size=hparam.batch_size,shuffle=True,)

    logging.info(f'''Starting training:
        Model:          {hparam.mname}
        Optimizer:      {hparam.optim}
        Epochs:         {hparam.epoches}
        Batch size:     {hparam.batch_size}
        Training size:  {len(trainset)}
        Testing size:   {len(testset)}
        Image size:     {hparam.img_size}
        Device:         {device.type}
        Initial learning rate:  {hparam.lr}
        Predict class threshold:{hparam.thresh}
    ''')

    train_history = []
    validationn_history = []
    acc_history = []

    for epoch in range(1,hparam.epoches+1):
        model.train()
        epoch_loss = 0
        for img_data, labels in tqdm(dataloader):
            img_data = img_data.to(device)
            labels = labels.to(device).squeeze()
            logits = model(img_data).squeeze()
            loss = loss_fn(logits,labels)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f'train Loss for epoch {epoch}: {epoch_loss/((len(trainset)//hparam.batch_size)+1):.4f}')
        train_history.append(epoch_loss/((len(trainset)//hparam.batch_size)+1))

        test_mean_loss, overallacc = evaluate(model=model,dataset=testset,loss_fn=loss_fn,device=device,hparam=hparam)
        every_class_acc = acc_every_class(dataset=testset,model=model,thresh=hparam.thresh)
        logging.info(f"every class accuracy{every_class_acc}")
        lr_scheduler.step(test_mean_loss) # lr_scheduler 參照 test_mean_loss

        validationn_history.append(test_mean_loss)
        acc_history.append(overallacc)
        
        logging.info(f'test_mean_loss: {test_mean_loss:.4f}')
        logging.info(f'testset accuracy: {overallacc:.4f}')
        #儲存最佳的模型
        if epoch == 1:
            criterion = test_mean_loss
            torch.save(model.state_dict(), hparam.wpath)
            logging.info(f'at epoch {epoch}, BESTMODEL.pth saved!')
        elif(test_mean_loss < criterion):
            criterion = test_mean_loss
            torch.save(model.state_dict(),hparam.wpath)
            logging.info(f'at epoch {epoch}, BESTMODEL.pth saved!')

    return dict(train_history=train_history, validationn_history=validationn_history, acc_history=acc_history)

def plot_history(trainingloss:list,validationloss:list, saved:bool=False,figname:str='history'):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (12, 6))
    ax.plot(torch.arange(1,len(trainingloss)+1,dtype=torch.int64), trainingloss, marker=".")
    ax.plot(torch.arange(1,len(validationloss)+1,dtype=torch.int64), validationloss, marker=".")
    ax.grid(visible=True)
    ax.legend(['TrainingLoss', 'ValidationLoss'])
    ax.set_title(f"Train History")

    fig.tight_layout()
    fig.show()
    if saved:
        fig.savefig(figname)

    return

def get_model(model_name:str, in_ch=2,filters=32,num_class=5,img_shape:tuple=(2048,2048)):
    '''model_name 必須是 spnet 或 resgcnet'''
    if model_name == 'spnet':
        return SPNet(in_ch=in_ch, num_class=num_class,img_shape=img_shape,filters=filters)
    elif model_name == 'resgcnet':
        return ResGCNet(in_ch=in_ch, num_class=num_class,img_shape=img_shape,filters=filters)
    else:
        print(f'Don\'t find the model: {model_name} .')

def get_optim(optim_name:str, model, lr:float):
    """optim_name = [adam | adamw | sgd | rmsprop | adagrad"""
    optim_name = optim_name.lower()
    if optim_name == "adam":
        return torch.optim.Adam(model.parameters(),lr = lr)
    elif optim_name == "adamw":
        return torch.optim.AdamW(model.parameters(),lr = lr)
    elif optim_name == "sgd":
        return torch.optim.SGD(model.parameters(),lr = lr)
    elif optim_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(),lr = lr)
    elif optim_name == "adagrad":
        return torch.optim.Adagrad(model.parameters(),lr = lrr)
    else:
        print(f'Don\'t find the model: {optim_name} . default optimizer is adam')
        return torch.optim.Adam(model.parameters(),lr = hparam.lr)

def acc_every_class(dataset:[str|Fundusdataset],model, thresh:float=0.5, transforms=None,)->dict:
    
    if isinstance (dataset,str):
        ds = Fundusdataset(dataset,transforms=transforms)
    else:
        ds = dataset
    class2ndx = ds.class2ndx
    ndx2class = {v:k for k,v in class2ndx.items()}
    acc_dict = {}
    count_t = torch.zeros(len(class2ndx),dtype=torch.int64)
    for i in range(len(ds)):
        img = ds[i][0].unsqueeze(0) #加入批次軸為了預測
        img = img.cuda()
        prob = torch.sigmoid(model(img)).squeeze() #去掉batch軸
        predict = (prob > thresh).to(torch.int64)
        label = ds[i][1]
        label = label.cuda()
        for j in range(len(label)):
            if predict[j] == label[j]:
                count_t[j] += 1

    count_t = count_t.to(torch.float32) / len(ds)
    for i in range(len(count_t)):
        acc_dict[ndx2class[i]] = round(count_t[i].item(),4)

    return acc_dict
    

if __name__ == '__main__':
    main()