import os
from img_preprocessing import cal_hist
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


if __name__ == "__main__":
    start = time.time()
    if len(sys.argv) < 2 :
        print("give the root of folder which you want to look up distribution.")
        sys.exit()
    elif len(sys.argv) > 2:
        print("the number of parameter must be one(folder name)")
        sys.exit()

    foldername = sys.argv[1]
    tb = np.zeros(256,dtype=np.float32)
    tg = np.zeros(256,dtype=np.float32)
    tr= np.zeros(256,dtype=np.float32)
    file_count = 0
    for dir, subdirs, files in os.walk(foldername):
        for f in files:
            img_path = os.path.join(dir, f)
            img = cv2.imread(img_path)
            tb += cal_hist(img,img_ch=0, if_plot=False)
            tg += cal_hist(img,img_ch=1, if_plot=False)
            tr += cal_hist(img,img_ch=2, if_plot=False)
            file_count += 1
    
    tb /= file_count
    tg /= file_count
    tr /= file_count  

    index = np.arange(256)
    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.plot(index, tb, label="blue",color="blue")
    ax.plot(index, tg, label="green", color="green")
    ax.plot(index, tr, label="red", color="red")
    ax.grid()
    ax.legend()
    fig.savefig(f'{os.path.basename(foldername)}_Histogram.jpg')
    end = time.time()
    
    print(f"spend the {end-start} second.")