import os
from torch.utils.data import Dataset,Subset
import cv2
import torch
from random import random
from pathlib import Path
import json
from torchvision import transforms
from .utils import Resize, Padding

def get_image_multilabel(root):
    class_name = os.listdir(root)
    class2ndx = {name:idx for idx,name in enumerate(class_name)}
    num_class = len(class_name)

    json_file = {}
    json_file["img_label"] = {}
    json_file["img_path"] = {}
    for dirname, dirs, files in os.walk(root):
        for file in files:
            if not file in json_file["img_label"].keys():
                # json_file["img_name"][file] = torch.zeros(num_class)
                json_file["img_label"][file] = [0 for i in range(num_class)]
                json_file["img_path"][file] = os.path.join(dirname, file) #當img_name第一次被讀取時建立該image的路徑，避免圖片被重複讀取
            ndx = class2ndx[os.path.basename(dirname)]
            json_file["img_label"][file][ndx] = 1
    
    return json_file, class2ndx

class Fundusdataset(Dataset):
    def __init__(self, dataset_root, imgsize:int,transforms=None):
        super(Fundusdataset,self).__init__()
        self.json_file, self.class2ndx = get_image_multilabel(dataset_root)
        self.ndx2img = {ndx:key for ndx,key in enumerate(self.json_file["img_label"].keys())}
        self.transforms = transforms
        self.imgsize = imgsize
    def __getitem__(self, ndx):
        key = self.ndx2img[ndx]
        imgpath = self.json_file["img_path"][key]
        img = cv2.imread(imgpath)
        resize = Resize(base_size=self.imgsize,fix_ratio=True)
        padding = Padding(padding_value=0)
        img = resize(img)
        img = img / 255 #limit the value to [0, 1]
        img = padding(img)
        img_t = torch.from_numpy(img).to(dtype=torch.float32).permute(2,0,1) # (h,w,ch) -> (ch,h,w)
        if self.transforms:
            img_t = self.transforms(img_t)

        imglabel = torch.tensor(self.json_file["img_label"][key],dtype=torch.float32)

        return img_t, imglabel
    
    def __len__(self):
        return len(self.json_file["img_label"])

def split_dataset(dataset:Dataset,test_ratio:float= 0.2,seed:int=20230813):
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    ndx = torch.randperm(len(dataset),generator=g_cpu).tolist()
    test_ndx = ndx[:int(test_ratio*len(dataset))]
    train_ndx = ndx[int(test_ratio*len(dataset)):]
    testset = Subset(dataset,test_ndx)
    trainset = Subset(dataset,train_ndx)

    return trainset, testset

if __name__ == '__main__':
    labels, _ = get_image_multilabel("../multiLabel_base")

    print(len(labels["img_label"]))
    print(len(labels["img_path"]))
    with open("label.json", "w") as outfile:
        json.dump(labels, outfile, indent = 4)

    # fundus_dataset = Fundusdataset("fundus_dataset_multilabel",transforms=None, imgsize=512)
    # trainset, testset = split_dataset(fundus_dataset,test_ratio=0.2,seed=20230823)
    
    # for i in range(len(trainset)):
    #     print(trainset[0][0].shape)
    # print(fundus_dataset.class2ndx)
    
