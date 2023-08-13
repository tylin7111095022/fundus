import os
from torch.utils.data import Dataset,Subset, DataLoader
import cv2
import torch
from random import random
from pathlib import Path
import json
from torchvision import transforms

def get_image_multilabel(root) -> dict:
    class_name = os.listdir(root)
    class_ndx_map = {name:idx for idx,name in enumerate(class_name)}
    print(class_ndx_map)
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
            ndx = class_ndx_map[os.path.basename(dirname)]
            json_file["img_label"][file][ndx] = 1     
    
    return json_file

class Fundusdataset(Dataset):
    def __init__(self, dataset_root,transforms=None):
        super(Fundusdataset,self).__init__()
        self.json_file = get_image_multilabel("fundus_dataset_multilabel_0812")
        self.ndx_key_map = {ndx:key for ndx,key in enumerate(self.json_file["img_label"].keys())}
        self.transforms = transforms
    def __getitem__(self, ndx):
        key = self.ndx_key_map[ndx]
        imgpath = self.json_file["img_path"][key]
        img = cv2.imread(imgpath)
        img = img / 255 #limit the value to [0, 1]
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
    labels = get_image_multilabel("fundus_dataset_multilabel_0812")

    # print(len(labels["img_label"]))
    # print(len(labels["img_path"]))
    with open("label.json", "w") as outfile:
        json.dump(labels, outfile, indent = 4)
    pipe = transforms.Compose([transforms.Resize((2048,2048))])

    dataset = Fundusdataset("fundus_dataset_multilabel_0812",transforms=pipe)
    trainset, testset = split_dataset(dataset,test_ratio=0.2)
    print(trainset[0][0])
    print(trainset[0][0].shape)
    print(trainset[0][1])
    
