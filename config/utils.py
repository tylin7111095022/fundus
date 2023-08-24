import cv2

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
    
if __name__ == "__main__":
    resize = Resize(base_size=1024,fix_ratio=True)
    padding = Padding(padding_value=0)
    img = cv2.imread("test2.JPG")
    cv2.imwrite("orig.jpg",img)
    after = resize(img)
    after = padding(after)
    cv2.imwrite("after.jpg",after)
            