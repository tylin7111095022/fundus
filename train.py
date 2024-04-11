# 當創建新環境時，需要將ultralytics/nn/modules內的Classify class做更動
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

import yaml
from typing import Optional
#custom
from config.fundusdataset import Fundusdataset, split_dataset
from models import initialize_model, get_optim
from models.loss import get_loss_fn

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, dest='mname', default='mldecoder', help='deep learning model will be used')
    parser.add_argument("--loss", choices=["asl", "bce"],default="asl", help='loss function when training')
    parser.add_argument("--optim", type=str, default='AdamW', help='optimizer')
    parser.add_argument("--in_channel", type=int, default=3, dest="inch",help="the number of input channel of model")
    parser.add_argument("--img_size", type=int, default=96,help="image size")
    parser.add_argument("--nclass", type=int, default=3,help="the number of class for classification task")
    parser.add_argument("--num_workers", type=int, default=0, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--epoches", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size during training")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate for optimizer")
    parser.add_argument("--threshold", type=float, default=0.5, dest="thresh",help="the threshold of that predicting if belong the class")
    parser.add_argument("--weight_path", type=str,dest='wpath', default='./best.pth', help="path of model we trained best")
    parser.add_argument("--is_parallel", type=bool, default=False, dest="paral",help="parallel calculation at multiple gpus")
    parser.add_argument("--device", type=str, default='cpu', help='device trainging deep learning')
    # parser.add_argument("--weighted_loss", type=bool,dest="wl", default=False, help='balance the loss between pos and neg')
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
    logging.info(hparam)
    # TRANSPIPE = transforms.Compose([transforms.Resize((hparam.img_size,hparam.img_size))])

    #資料隨機分選訓練、測試集
    fundus_dataset = Fundusdataset("./data/train",transforms=None, imgsize=hparam.img_size)
    trainset, testset = split_dataset(fundus_dataset,test_ratio=0.2,seed=20230823)
    logging.info(fundus_dataset.class2ndx)
    # 計算每個標籤的正負樣本比
    # if hparam.wl:
    #     pos_count = torch.zeros(hparam.nclass)
    #     all_count = torch.zeros(hparam.nclass)
    #     for i in range(len(trainset)):
    #         pos_count += trainset[i][1]
    #         all_count += torch.ones(hparam.nclass)
    #     neg_count = all_count - pos_count
    #     pos_weights = (neg_count / pos_count).to(hparam.device)
    # else:
    #     pos_weights = torch.ones(hparam.nclass).to(hparam.device)
    # logging.info(f"pos_weights: {pos_weights}")

    model = initialize_model(hparam.mname,num_classes=hparam.nclass,use_pretrained=True,use_custom_clf=True)
    if torch.cuda.device_count() > 1 and hparam.paral:
          logging.info(f"use {torch.cuda.device_count()} GPUs!")
          model = torch.nn.DataParallel(model)
    
    logging.info(model)
    optimizer = get_optim(optim_name=hparam.optim,model=model,lr=hparam.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                              mode='min',
                                                              factor=0.5,
                                                              patience=3,
                                                              min_lr=1e-6,
                                                              verbose =True)
    device = torch.device( hparam.device if torch.cuda.is_available() else 'cpu')
    criteria = get_loss_fn(hparam.loss)

    #training
    history = training(model,trainset, testset, criteria, optimizer,lr_scheduler, device, hparam)
    #plotting
    plot_history(history['train_history'],history['validationn_history'],saved=True)

    fig, ax = plt.subplots(2,1, figsize = (15, 8))
    ax[0].plot(torch.arange(1,hparam.epoches+1,dtype=torch.int64),history['acc_history'])
    ax[0].grid(visible=True)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].set_title(f"accuracy History")
    ax[1].plot(torch.arange(1,hparam.epoches+1,dtype=torch.int64),history['inter_union_history'])
    ax[1].grid(visible=True)
    ax[1].set_ylim(0.0, 1.0)
    ax[1].set_title(f"inter_over_union")
    plt.tight_layout()
    fig.savefig("accuracy")

    # evaluate performance of testset
    # best_model = get_model(model_name=hparam.mname,in_ch=hparam.inch,num_class=hparam.nclass)
    # best_model.load_state_dict(torch.load(hparam.wpath))
    return

def evaluate(model,dataset, loss_fn,device,hparam):
    model.eval()
    model = model.to(device)
    total_loss = 0
    total_iou = []
    dataloader = DataLoader(dataset=dataset,batch_size=hparam.batch_size,shuffle=False)
    for img_data,labels in dataloader:
        img_data = img_data.to(device)
        labels = labels.to(device).squeeze() #變為一軸
        logits = model(img_data).squeeze()
        loss = loss_fn(logits,labels)
        total_loss += loss.item()
        predict = torch.sigmoid(logits.detach()).squeeze()
        mask = (predict > hparam.thresh).to(torch.int64) # mask 中值為True即為預測值
        labels = labels.to(torch.int64)
        batch_iou = (torch.sum(mask & labels).item()) / (torch.sum(mask | labels).item())
        print(f"after sigmoid: \n{predict}")
        # print(f"predicts: \n{mask}")
        # print(f"label: \n{labels}")
        # print(f"predicts & labels: \n{mask & labels}")
        # print(f"mask | labels: \n{mask | labels}")
        # print(f"batch_acc: {batch_iou}")
        # print(f"="*40)
        total_iou.append(batch_iou)

    mean_loss = total_loss/ ((len(dataset)//hparam.batch_size)+1)
    iou = round(sum(total_iou) / len(total_iou), 4)

    return mean_loss, iou

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
    inter_union_history = []

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

        test_mean_loss, intersect_union = evaluate(model=model,dataset=testset,loss_fn=loss_fn,device=device,hparam=hparam)
        every_class_acc, overallacc = acc_every_class(dataset=testset,model=model,thresh=hparam.thresh,device=hparam.device)
        logging.info(f"every class accuracy{every_class_acc}")
        lr_scheduler.step(test_mean_loss) # lr_scheduler 參照 test_mean_loss

        validationn_history.append(test_mean_loss)
        acc_history.append(overallacc)
        inter_union_history.append(intersect_union)
        
        logging.info(f'test_mean_loss: {test_mean_loss:.4f}')
        logging.info(f'testset accuracy: {overallacc:.4f}')
        logging.info(f'intersection over union: {intersect_union:.4f}')
        
        #儲存最佳的模型
        if epoch == 1:
            criterion = intersect_union
            torch.save(model.state_dict(), hparam.wpath)
            logging.info(f'at epoch {epoch}, BESTMODEL.pth saved!')
        elif(intersect_union > criterion):
            criterion = intersect_union
            torch.save(model.state_dict(),hparam.wpath)
            logging.info(f'at epoch {epoch}, BESTMODEL.pth saved!')
            
        torch.save(model.state_dict(),"./last.pth")

    return dict(train_history=train_history,
                validationn_history=validationn_history,
                acc_history=acc_history,
                inter_union_history=inter_union_history)

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

def acc_every_class(dataset:Fundusdataset,model, device, thresh:float=0.5, transforms=None,)->dict:
    model = model.to(device) 
    if isinstance (dataset,str):
        ds = Fundusdataset(dataset,transforms=transforms)
    else:
        ds = dataset
    class2ndx = ds.dataset.class2ndx
    ndx2class = {v:k for k,v in class2ndx.items()}
    acc_dict = {}
    count_t = torch.zeros(len(class2ndx),dtype=torch.int64)
    for i in range(len(ds)):
        img = ds[i][0].unsqueeze(0) #加入批次軸為了預測
        img = img.to(device=device)
        prob = torch.sigmoid(model(img)).squeeze() #去掉batch軸
        predict = (prob > thresh).to(torch.int64)
        label = ds[i][1]
        label = label.to(device=device)
        for j in range(len(label)):
            if predict[j] == label[j]:
                count_t[j] += 1

    count_t = count_t.to(torch.float32) / len(ds)
    overall_acc = (torch.sum(count_t) / len(count_t)).item()
    for i in range(len(count_t)):
        acc_dict[ndx2class[i]] = round(count_t[i].item(),4)

    return acc_dict, overall_acc
    

if __name__ == '__main__':
    main()
