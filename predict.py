from config.models import SPNet, ResGCNet, initialize_model
from PIL import Image
from config.utils import Resize, Padding
import cv2
import sys
import torch
import argparse
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, dest='mname', default='yolov8', help='deep learning model will be used')
    parser.add_argument("--in_channel", type=int, default=3, dest="inch",help="the number of input channel of model")
    parser.add_argument("-img","--image_path", type=str, dest="img", default='./demoimg.JPG', help='deep learning model will be used')
    parser.add_argument("--img_size", type=int, default=512,help="image size")
    parser.add_argument("--nclass", type=int, default=6,help="the number of class for classification task")
    parser.add_argument("--threshold", type=float, default=0.5, dest="thresh",help="the threshold of that predicting if belong the class")
    parser.add_argument("--weight_path", type=str,dest='wpath', default='./log/train4/best.pth', help="path of model we want to load")
    parser.add_argument("--show", type=bool, default=False, dest="show",help="decide whether show image or not")

    return parser.parse_args()

def main():
    hparam = get_args()
    # TRANSPIPE = transforms.Compose([transforms.Resize((hparam.img_size,hparam.img_size))])
    prob, ans = predict(hparam.img,show_img=hparam.show, transforms=None)
    print(f"image is {hparam.img}")
    print(f"prob is {prob}")
    print(f"threshold is {hparam.thresh}")
    print(f"ans is {ans}")

def predict(img_path,transforms=None,show_img:bool = False):
    hparam = get_args()
    model = initialize_model(hparam.mname,num_classes=hparam.nclass,use_pretrained=True)
    model.load_state_dict(torch.load(hparam.wpath))
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

if __name__ == "__main__":
    main()
    
