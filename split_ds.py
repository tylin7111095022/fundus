import os
import random
import shutil
import sys

random.seed(20230814)
if len(sys.argv) < 2 :
    print("give the root of dataset which you want to preprocess")
    sys.exit()
elif len(sys.argv) > 2:
    print("the number of parameter must be one(root of dataset)")
    sys.exit()

dataset_path = sys.argv[1]

train_dir = os.path.join(dataset_path,"train")
val_dir =  os.path.join(dataset_path,"val")
test_dir =  os.path.join(dataset_path,"test")
train_ratio = 0.8
val_ratio = 0.0
test_ratio = 0.2

class_dict = {}
for dirname, dirs, files in os.walk(dataset_path):
    if not class_dict.get(dirname, 0):
        class_dict[dirname] = []
    for file in files:
        path = os.path.join(dirname, file)
        class_dict[dirname].append(path)

try:
    del class_dict[dataset_path] # delete the key of current dirextory
except:
    print(f"{dataset_path} don\'t exist.")

# 看每個目錄底下各自有多少檔案
for dir in class_dict.keys():
    desease = os.path.basename(dir)
    print(dir, len(class_dict[dir]))
    ndx = [i for i in range(len(class_dict[dir]))]
    random.shuffle(ndx) #shuffle the ndx list
    num_train = int(len(class_dict[dir]) * train_ratio)
    num_val = int(len(class_dict[dir]) * val_ratio)
    num_test = int(len(class_dict[dir]) - num_train - num_val)

    target_dir_train = os.path.join(train_dir, desease)
    if not os.path.exists(target_dir_train):
        os.makedirs(target_dir_train)

    target_dir_val = os.path.join(val_dir, desease)
    if not os.path.exists(target_dir_val):
        os.makedirs(target_dir_val)
    target_dir_test = os.path.join(test_dir, desease)
    if not os.path.exists(target_dir_test):
        os.makedirs(target_dir_test)

    for i in range(len(ndx)):
        if i < num_train:
            shutil.copy(class_dict[dir][i],target_dir_train)
        elif i < (num_train+num_val):
            shutil.copy(class_dict[dir][i], target_dir_val)
        elif i < (num_train+num_val+num_test):
            shutil.copy(class_dict[dir][i], target_dir_test)

    print(f"files in {dir} have moved")
