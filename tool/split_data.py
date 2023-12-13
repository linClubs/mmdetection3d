# 将图片和标注数据按比例切分为 训练集和测试集
import shutil
import random
import os
import argparse
 
 
# 检查文件夹是否存在
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
 
def main(dataroot, train_percent, val_percent, test_percent):
    ImageSets_dir = os.path.join(dataroot, "ImageSets")
    mkdir(ImageSets_dir)
 
    lidar_dir = os.path.join(dataroot, "velodyne")
 
    if (train_percent + val_percent + test_percent) != 1:
        print("split percent is error")
        return 0
 
    total_file = os.listdir(lidar_dir)

    num = len(total_file)
    list_all = range(num)  # 范围 range(0, num)

    num_train = int(num * train_percent)
    num_val = int(num * val_percent)
    num_test = num - num_train - num_val
    
    
    train = random.sample(list_all, num_train)
   
    # 在全部数据集中取出train
    val_test = [i for i in list_all if not i in train]

    # 再从val_test取出num_val个元素，val_test剩下的元素就是test
    val = random.sample(val_test, num_val)
    test = [i for i in val_test if not i in val]

    print("训练集数目：{}, 验证集数目：{},测试集数目：{}".format(len(train), len(val), len(val_test) - len(val)))
    
    train_txt_path = os.path.join(ImageSets_dir, "train.txt")
    val_txt_path = os.path.join(ImageSets_dir, "val.txt")
    test_txt_path = os.path.join(ImageSets_dir, "test.txt")
    
    f_train = open(train_txt_path, "w")
    f_val = open(val_txt_path, "w")
    f_test = open(test_txt_path, "w")


    for i, name in enumerate(total_file):
        prefix = name.split('.')[0]
    
        if i in train:
            f_train.write(str(prefix) + '\n') 
        elif i in val:
            f_val.write(str(prefix) + '\n')
        else:
            f_test.write(str(prefix) + '\n') 

 
if __name__ == '__main__':
    dataroot = "/home/lin/code/mmdetection3d/data/custom2"
    # 数据集划分比例，训练集75%，验证集15%，测试集15%，按需修改
    train_percent = 0.8
    val_percent = 0.1
    test_percent = 0.1
    
    main(dataroot, train_percent, val_percent, test_percent)