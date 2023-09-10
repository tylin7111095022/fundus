from config.models import initialize_model
import time
from PIL import Image
from config.utils import Resize, Padding
import cv2
import torch
import argparse
from torchvision import transforms
import os
import threading
import warnings
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, dest='mname', default='yolov8', help='deep learning model will be used')
    parser.add_argument("--in_channel", type=int, default=3, dest="inch",help="the number of input channel of model")
    parser.add_argument("-img","--image_path", type=str, dest="img", default='./demoimg.JPG', help='deep learning model will be used')
    parser.add_argument("--img_size", type=int, default=512,help="image size")
    parser.add_argument("--nclass", type=int, default=6,help="the number of class for classification task")
    parser.add_argument("--threshold", type=float, default=0.7, dest="thresh",help="the threshold of that predicting if belong the class")
    parser.add_argument("--weight_path", type=str,dest='wpath', default='./log/train16_final_asl_except_normal/best.pth', help="path of model we want to load")
    parser.add_argument("--show", action="store_true", default=False,help="decide whether show image or not")
    parser.add_argument("--ensemble",'-e', action="store_true", default=False, help="decide whether using ensemble predict or not")

    return parser.parse_args()

def main():
    start = time.time()
    hparam = get_args()
    # TRANSPIPE = transforms.Compose([transforms.Resize((hparam.img_size,hparam.img_size))])
    if  not hparam.ensemble:
        # 使用單個weight 進行 predict
        prob, ans = predict(img_path=hparam.img,show_img=hparam.show, transforms=None)
    else:
        # 使用多個weight進行ensemble predict
        print(f"using ensemble predict")
        weight_path = ["log/train16_final_asl_except_normal/best.pth", "log/train17_final_bce_except_normal/last.pth"]
        prob, ans = ensemble_predict(num_weight = 2,weight_path = weight_path)
        
    end = time.time()
    dur = round(end-start,4)
    print(f"image is {hparam.img}")
    print(f"prob is {prob}")
    print(f"threshold is {hparam.thresh}")
    print(f"ans is {ans}")
    print(f"predict spend {dur} seconds")

def predict(img_path,transforms=None,show_img:bool = False):
    hparam = get_args()
    model = initialize_model(hparam.mname,num_classes=hparam.nclass,use_pretrained=True)
    model.load_state_dict(torch.load(hparam.wpath))
    model = model.eval()
    img = cv2.imread(img_path)
    resize = Resize(base_size=hparam.img_size,fix_ratio=True)
    padding = Padding(padding_value=0)
    img = resize(img)
    img = img / 255 #limit the value to [0, 1]
    img = padding(img)
    img_t = torch.from_numpy(img).to(dtype=torch.float32).permute(2,0,1) # (h,w,ch) -> (ch,h,w)
    if transforms:
        img_t = transforms(img_t)
    
    if show_img:
        im = Image.open(hparam.img)
        im.show()
    
    img_t = img_t.unsqueeze(0) #加入批次軸
    
    logit = model(img_t)
    prob = torch.sigmoid(logit.detach()).squeeze()
    ans = (prob > hparam.thresh).to(torch.int64) # mask 中值為True即為預測值
    
    return prob, ans

def ensemble_predict(num_weight:int, weight_path:list):
    assert num_weight == len(weight_path) , f"len(weight_path) must be equal to num_weight, but got {len(weight_path)}"
    hparam = get_args()
    infoes = [{} for i in range(num_weight)]
    threads = []
    for i in range(len(infoes)):
        t = threading.Thread(target=predict_for_multiweight, args=(hparam.img,weight_path[i],infoes[i],None,))
        threads.append(t)
        threads[i].start()
    
    for t in threads:
        t.join()
    
    prob_sum = infoes[0]["prob"]
    for i in range(1,len(infoes)):
        prob_sum += infoes[i]["prob"]
        
    mean_prob = prob_sum / len(infoes)
    ans = (mean_prob > hparam.thresh).to(torch.int64) # mask 中值為True即為預測值
    
    return mean_prob, ans
    
def predict_for_multiweight(img_path,model_weight:str,saved_info:dict, transforms=None,show_img:bool = False):
    """本函式用來幫助function ensemble_predict執行multi-thread運算"""
    hparam = get_args()
    model = initialize_model(hparam.mname,num_classes=hparam.nclass,use_pretrained=True)
    model.load_state_dict(torch.load(model_weight))
    model = model.eval()
    img = cv2.imread(img_path)
    resize = Resize(base_size=hparam.img_size,fix_ratio=True)
    padding = Padding(padding_value=0)
    img = resize(img)
    img = img / 255 #limit the value to [0, 1]
    img = padding(img)
    img_t = torch.from_numpy(img).to(dtype=torch.float32).permute(2,0,1) # (h,w,ch) -> (ch,h,w)
    if transforms:
        img_t = transforms(img_t)
    if show_img:
        im = Image.open(hparam.img)
        im.show()
    
    img_t = img_t.unsqueeze(0) #加入批次軸
    logit = model(img_t)
    prob = torch.sigmoid(logit.detach()).squeeze()
    ans = (prob > hparam.thresh).to(torch.int64) # mask 中值為True即為預測值
    saved_info["prob"] = prob
    saved_info["ans"] = ans
    return 

if __name__ == "__main__":
    main()