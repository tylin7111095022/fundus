import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import albumentations as A

def main():
    start = time.time()
    # gamma correction
    # root = "multi_labels_Final_0911_clean_1000_gamma16_reverse"
    # for folder, dir_list, files in os.walk(root):
    #     for f in files:
    #         img_path = os.path.join(folder, f)
    #         gamma_correction(imgpath=img_path, gamma=1/(1.6))

    transform = A.Compose([
            A.augmentations.geometric.rotate.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,value=0, p=1.0, crop_border=True),
            A.HorizontalFlip(p=0.7),
            A.RandomBrightnessContrast(brightness_limit=[-0.1, 0.3], contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=1.0),
    ])
    augment(root="multi_labels_Final_0911_clean_1000_gamma16\\RVO", transform=transform)

    end = time.time()

    print(f"spend {end-start} second")

    return 0

def augment(root, transform):
    """transform must use albumentations module"""
    # transform = A.Compose([
    #         A.augmentations.geometric.rotate.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,value=0, p=1.0, crop_border=True),
    #         A.HorizontalFlip(p=0.7),
    #         A.RandomBrightnessContrast(brightness_limit=[-0.1, 0.3], contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=1.0),
    # ])
    for folder, dir_list, files in os.walk(root):
        for f in files:
            print(f)
            img_path = os.path.join(folder, f)
            img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
            transformed_image = transform(image = img)["image"]
            newname = f"aug_{f}"
            cv2.imwrite(os.path.join(folder, newname), transformed_image)    
    return
    
def img_transform():
    if len(sys.argv) < 2 :
        print("give the root of dataset which you want to preprocess")
        sys.exit()
    elif len(sys.argv) > 2:
        print("the number of parameter must be one(root of dataset)")
        sys.exit()

    dataset_path = sys.argv[1]

    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for f in filenames:
            if filenames == []:
                print("filenames_list is empty")
                continue
            filepath = os.path.join(dirpath, f)
            img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
            img = cv2.medianBlur(img, 5)
            laplacian_img = cv2.Laplacian(img, -1, ksize=5, scale=1)
            #cv2.imshow(filepath,after_img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            green_img = cv2.imread(filepath)[:,:,1] # green channel
            clahe_img = clahe(filepath)

            after_img = np.stack([laplacian_img,green_img,clahe_img],axis=2)
            cv2.imwrite(filepath, after_img)

    print("images have transformed")
    
    return 0


def cal_hist(img,img_ch:int=0,if_plot:bool=False):
    """img: numpy.array
        calculate the number of pixel values between 0~255"""
    if len(img.shape) == 2:
        img = np.expand_dims(img,axis=2)
    total_pixel = img.shape[0]*img.shape[1]
    saved = np.zeros(256) #index from 0 to 255
    index = np.arange(256)
    img = img[:,:,img_ch]

    for row_idx in range(img.shape[0]):
        for col_idx in range(img.shape[1]):
            saved[int(img[row_idx][col_idx])] += 1
    
    saved = saved / total_pixel #normalize for easy to observe
    if if_plot:
        plt.plot(index, saved)
        plt.title('Image Histogram')
        plt.show()

    return saved

def clahe(imgpath:str,clipLimit:float=2.5, tileGridSize:tuple=(16,16),show_pic:bool = False) -> np.array:
    """clahe"""    
    img = cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
    if len(img.shape) == 2:
        img = np.expand_dims(img,axis=2)

    clahe_object = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize = tileGridSize )
    if show_pic:
        cv2.imshow(f'{imgpath}', img)
        cv2.waitKey(0)
    img = np.squeeze(img,axis=2) #去掉通道軸以利clahe object可使用
    after_img = clahe_object.apply(img)
    #after_img = np.expand_dims(after_img, axis=2) #把通道軸加回來
    if show_pic:
        cv2.imshow(f'clahe', after_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return after_img

def gamma_correction(imgpath:str, gamma:float):
    img = cv2.imread(imgpath,cv2.IMREAD_UNCHANGED)
    img = img / 255 #normalize to [0,1]
    
    img = np.power(img, gamma)
    img = np.round(img*255)

    cv2.imwrite(f"{imgpath}", img)
    return

if __name__ == "__main__":
    main()
