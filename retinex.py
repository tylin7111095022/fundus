import numpy as np
import cv2
import time

def singleScaleRetinex(img,variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex

def SSR(imgpath:str, variance):
    start = time.time()
    img_a = cv2.imread(imgpath)
    img_a = np.float64(img_a) + 1.0 # +1防止log(0)
    img_retinex = singleScaleRetinex(img_a, variance)
    max_value = np.max(img_retinex,axis=(0,1),keepdims=True)
    min_value = np.min(img_retinex,axis=(0,1),keepdims=True)
    print(f"max_value shape{max_value.shape}")
    print(f"min_value shape{min_value.shape}")
    img_retinex = ((img_retinex - min_value) / (max_value - min_value))*255
    img_retinex = img_retinex.astype(np.uint8)
    end = time.time()
    print(f"using the {end-start} s")
    return img_retinex

if __name__ == "__main__":
    variance = 1000
    img_ssr=SSR("demoimg.JPG", variance=variance)
    cv2.imwrite(f'var{variance}_SSR.jpg', img_ssr)
    cv2.imshow('SSR', img_ssr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()