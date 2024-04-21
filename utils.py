import cv2
import torch
import os
import numpy as np
import torch.nn as nn
from typing import Optional

class Resize(object):
    def __init__(self,base_size:int, fix_ratio:bool=False):
        self.base_size = base_size
        self.fix_ratio = fix_ratio
        
    def __call__(self, img):
        orig_h,orig_w,c = img.shape
        if self.fix_ratio:
            if orig_h >= orig_w:
                ratio = orig_h / orig_w
                new_h = self.base_size
                new_w = int(new_h / ratio)
            else:
                ratio = orig_h / orig_w
                new_w = self.base_size
                new_h = int(new_w * ratio)
            img = cv2.resize(img,(new_w, new_h))
            # print(f"orig_h:{orig_h}, orig_w:{orig_w}")
            # print(f"new_h:{new_h}, new_w:{new_w}")
        else:
            img = cv2.resize(img,(self.base_size, self.base_size))
            
        return img

class Padding(object):
    def __init__(self, padding_value:int=0):
        self.padding_val = padding_value
    def __call__(self, img):
        h,w,c = img.shape
        pad = self.padding_val
        diff = h - w
        if diff > 0:
            top, bottom = 0, 0
            left = abs(diff) // 2
            right = abs(diff) - left
        else:
            left, right = 0, 0
            top = abs(diff) // 2
            bottom = abs(diff) - top
                    
        pad_img = cv2.copyMakeBorder(img,top=top,bottom=bottom,left=left,right=right,borderType=cv2.BORDER_CONSTANT,value=(pad,pad,pad))
        
        return pad_img
    
class GradCam(object):
    def __init__(self, model:nn.Module, gradCamLayer:Optional[list]=None):
        self.model = model
        self.targetLayers = gradCamLayer
        self.activationAndGrads = ActivationandGradients(model,gradCamLayer)
        self.device = next(self.model.parameters()).device

    def forwardandBackward(self, x, class_ndx:int):
        self.model.zero_grad()
        logits= self.activationAndGrads(x)
        prob = torch.sigmoid(logits)
        prob[:,class_ndx].backward()

        return

    def mode(self,mode:str):
        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()
        else:
            print("Error, mode should be train or eval !!!, mode is eval now.")
            self.model.eval()

    def to(self, device:str):
        self.model.to(device=device)
        self.device = next(self.model.parameters()).device

    def load_weights(self, weights):
        model_dict = self.model.state_dict()
        pretrained_w_dict = torch.load(weights)
        weights_dict = {k:v for k, v in pretrained_w_dict.items() if k in model_dict }
        self.model.load_state_dict(weights_dict)
    
    def get_cam_images(self,x, class_ndx:int):
        x = x.to(self.device,dtype=torch.float32)
        self.forwardandBackward(x,class_ndx)
        target_size = (x.shape[-2], x.shape[-1]) # h, w
        cam_per_target_layer = []
        weights = []
        cams = []
        # get cam weights
        for grad in self.activationAndGrads.gradients:
            # print(grad.shape)
            weight = torch.mean(grad, dim=(2,3)) # transformer  的話向量為 b, L(pathch數量), C
            weights.append(weight)
        # get cam activations
        activations = self.activationAndGrads.activations
        # print(f"activations: {activations}")
        for w, a in zip(weights, activations):
            weighted_activations = w[:, :, None, None] * a
            cam = torch.sum(weighted_activations,dim=1)
            cam = torch.where(cam > 0., cam, 0.)  # relu, cam shape: [b,h,w]
            cams.append(cam)
        for cam in cams:
            scaled_cam = self._scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled_cam[:,None,:,:]) # insert channel axis shape: [b,1,h,w]

        return cam_per_target_layer

    def _scale_cam_image(self, cams, target_size=None):
        result = []
        for img in cams: # img shape [h,w]
            img = img - torch.min(img)
            img = img / (1e-7 + torch.max(img)) # normalize
            img_a = img.numpy()
            if target_size is not None:
                img = cv2.resize(img_a, (target_size[1],target_size[0])) # taeget size should be w, h
            result.append(img)
        result = np.stack(result,axis=0)
        result = np.float32(result) # result shape [b,h,w]

        return result
    
    def plot_cams(self, imgpath, class_ndx:int, imgsize):
        name = os.path.basename(imgpath).split(".")[0]
        img_a = load_img(imgpath, imgsize)
        img_t = torch.from_numpy(img_a)[None, :,:, :] # 1,h,w, 3
        img_t = img_t.permute(0,3,1,2)
        cams_per_target = self.get_cam_images(img_t,class_ndx)
        for i, cam in enumerate(cams_per_target):
            cam = cam.squeeze() # remove one dimension
            heatmap = np.uint8(255 * cam)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap * 0.4 + img_a
            cv2.imwrite(f'{name}_class{class_ndx}_{i}_gradcam.jpg', superimposed_img)

    
# ref: https://github.com/jacobgil/pytorch-grad-cam/tree/master
class ActivationandGradients(object):
    def __init__(self, model, targetLayer:list):
        self.model = model
        self.targetLayer = targetLayer
        self.gradients = []
        self.activations = []
        self.handles = []

        for target in targetLayer:
            h_a = target.register_forward_hook(self.get_activations)
            h_g = target.register_forward_hook(self.get_gradients)
            self.handles.append(h_a)
            self.handles.append(h_g)

    def get_activations(self, module, input, output):
        activation = output
        self.activations.append(activation.cpu().detach())

    def get_gradients(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return
        # Gradients are computed in reverse order
        def _store_grad(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        # initial gradients and activations
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

def adjust_lr(optimizer, lr):
    for param in optimizer.param_groups:
        param["lr"] = lr

def cosine_decay_with_warmup(current_iter:int, total_iter:int, warmup_iter:int, base_lr:float):
    assert current_iter <= total_iter
    assert warmup_iter < total_iter

    if current_iter > warmup_iter:
        lr = 0.5 * base_lr * (1 + (np.cos(np.pi*(current_iter-warmup_iter)/(total_iter-warmup_iter))))
    else:
        slope = float(base_lr / warmup_iter)
        lr = slope * current_iter
    return lr

def load_img(imgpath, imgsize):
    img = cv2.imread(imgpath)
    resize = Resize(base_size=imgsize,fix_ratio=True)
    padding = Padding(padding_value=0)
    img = resize(img)
    img = padding(img)
    return img
    
if __name__ == "__main__":
    resize = Resize(base_size=512,fix_ratio=True)
    nofix = Resize(base_size=512,fix_ratio=False)
    padding = Padding(padding_value=0)
    img = cv2.imread("../test2.JPG")
    # cv2.imwrite("orig.jpg",img)
    after = resize(img)
    after = padding(after)
    nofixresize = nofix(img)
    # cv2.imwrite("after.jpg",after)
    cv2.imwrite("nofix.jpg",nofixresize)
            