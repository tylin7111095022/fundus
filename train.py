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
#custom
from config.fundusdataset import Fundusdataset, split_dataset
from config.models import SPNet, ResGCNet

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, dest='mname', default='resgcnet', help='deep learning model will be used')
    parser.add_argument("--in_channel", type=int, default=3, dest="inch",help="the number of input channel of model")
    parser.add_argument("--img_size", type=int, default=512,help="image size")
    parser.add_argument("--nclass", type=int, default=5,help="the number of class for classification task")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--epoches", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size during training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
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
    DATASET = Fundusdataset("fundus_dataset_multilabel_0812",transforms=pipe)
    trainset, testset = split_dataset(DATASET,test_ratio=0.2,seed=20230813)
    
    model = get_model(model_name=hparam.mname,in_ch=hparam.inch,img_shape=(hparam.img_size,hparam.img_size),num_class=hparam.nclass)
    logging.info(model)
    optimizer = torch.optim.Adam(model.parameters(),lr = hparam.lr, weight_decay=1e-8,)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    critera = BCEWithLogitsLoss()

    #training and evaluate
    history = training(model,trainset, testset, critera, optimizer,lr_scheduler, device, hparam)
    plot_history(history['train_history'],history['validationn_history'],saved=True)

    # evaluate performance of testset
    # best_model = get_model(model_name=hparam.mname,in_ch=hparam.inch,num_class=hparam.nclass)
    # best_model.load_state_dict(torch.load(hparam.wpath))
    # show_score(model=best_model, dataset=testset, hparam=hparam)

    return

def evaluate(model,dataset, loss_fn,predict_threshold:float,device,hparam):
    model.eval()
    model = model.to(device)
    total_loss = 0
    tp = tn = fp = fn = 0
    dataloader = DataLoader(dataset=dataset,batch_size=hparam.batch_size,shuffle=False)
    for img_data,labels in dataloader:
        img_data = img_data.to(device)
        labels = labels.to(device).squeeze() #變為一軸
        logits = model(img_data)
        loss = loss_fn(logits,labels)
        total_loss += loss.item()
        predict = torch.sigmoid(logits.detach()).squeeze()
        mask = predict > predict_threshold # mask 中值為True即為預測值
        #底下需要再改進，先測試程式能不能正常運行

        mean_loss = total_loss/ ((len(dataset)//hparam.batch_size)+1)

    return mean_loss

def training(model, trainset, testset, loss_fn, optimizer,lr_scheduler, device, hparam):
    model = model.to(device)
    dataloader = DataLoader(trainset,batch_size=hparam.batch_size,shuffle=True,)

    logging.info(f'''Starting training:
        Model:          {hparam.mname}
        Epochs:         {hparam.epoches}
        Batch size:     {hparam.batch_size}
        Training size:  {len(trainset)}
        Testing size:   {len(testset)}
        Image size:     {hparam.img_size}
        Device:         {device.type}
        Initial learning rate:  {hparam.lr}
    ''')
    train_history = []
    validationn_history = []

    for epoch in range(1,hparam.epoches+1):
        model.train()
        epoch_loss = 0
        for img_data, labels in tqdm(dataloader):
            img_data = img_data.to(device)
            labels = labels.to(device).squeeze()
            logits = model(img_data)
            loss = loss_fn(logits,labels)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f'train Loss for epoch {epoch}: {epoch_loss/((len(trainset)//hparam.batch_size)+1):.4f}')
        train_history.append(epoch_loss/((len(trainset)//hparam.batch_size)+1))

        test_mean_loss = evaluate(model=model,dataset=testset,loss_fn=loss_fn,predict_threshold=0.5,device=device,hparam=hparam)
        lr_scheduler.step(test_mean_loss) # lr_scheduler 參照 f1 score

        validationn_history.append(test_mean_loss)
        
        logging.info(f'test_mean_loss: {test_mean_loss:.4f}')
        #儲存最佳的模型
        if epoch == 1:
            criterion = test_mean_loss
            torch.save(model.state_dict(), hparam.wpath)
            logging.info(f'at epoch {epoch}, BESTMODEL.pth saved!')
        elif(test_mean_loss < criterion):
            criterion = test_mean_loss
            torch.save(model.state_dict(),hparam.wpath)
            logging.info(f'at epoch {epoch}, BESTMODEL.pth saved!')

    return dict(train_history=train_history, validationn_history=validationn_history)

def show_score(model, dataset, hparam):
    DELTA = 1e-20 #防止分母為0
    model.eval()
    model.to(device='cpu')
    tp = tn = fp = fn = 0
    dataloader = DataLoader(dataset=dataset,batch_size=hparam.batch_size,shuffle=False)
    for data, labels in tqdm(dataloader):
        labels = labels.squeeze() #變為一軸
        logits = model(data)
        probs = torch.softmax(logits.detach(), dim=1).squeeze()
        predict = torch.argmax(probs, dim=1)
        for pred,target in zip(predict,labels):
            if pred == 1 and pred == target:
                tp += 1
            elif pred == 1 and pred != target:
                fp += 1
            elif pred == 0 and pred == target:
                tn += 1
            else:
                fn += 1

    logging.info(f'''
    =================
     TP:{tp}   TN:{tn}

     FP:{fp}   FN:{fn}
    =================     
    ''')
    acc = (tp+tn)/(tp+tn+fp+fn)
    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    f1score = 2 * (precision*recall) / (recall+precision+DELTA)

    logging.info(f'''
    accuracy : {acc*100:4.2f} %
    recall   : {recall*100:4.2f} %
    precision: {precision*100:4.2f} %
    F1_score : {f1score*100:4.2f} %
    ''')
    return acc,recall,precision,f1score

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


if __name__ == '__main__':
    main()
    # gradcam
    # hparam = get_args()
    # df = pd.read_csv('labels\ph1_label.csv')
    # model = get_model(model_name=hparam.mname)
    # model.load_state_dict(torch.load('PH1_BESTMODEL.pth'))
    # gradcam = GradCam(model=model)
    # datafolder = 'ph1_data'

    # for ndx,i in enumerate(df['Filename']):
    #     gradcam.plot_heatmap(os.path.join(datafolder,i),saved_folder='gradCAM_image')
    #     if ndx % 10 == 0:
    #         plt.close('all')