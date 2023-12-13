import os
import mmengine
from pyquaternion import Quaternion
import json
import numpy as np

# 种类
class_names = ['car', 'truck', 'trailer', 'bus', 'construction',
                'bicycle', 'motorcycle', 'pedestrian', 'trafficcone', 'barrier']

def get_infos(dataroot, ids):
    
    infos = []

    for id in ids:
        info = {}
        idx_info = {}
        annos_info = {}
        point_info = {}
        calib_info = {}
        image_info ={}
        
        lidar_path = os.path.join(dataroot, 'points', id + '.bin')
        label_path = os.path.join(dataroot, 'labels', id + '.txt')
        calib_path = os.path.join(dataroot, 'calibs', id + '.txt')

        # print(lidar_path)
        # print(label_path)
        # print(calib_path)
        
        idx_info['idx_info'] = int(id)

        point_info['num_pts_feats'] = 4
        point_info['lidar_path'] = id + '.bin'
        point_info['T_lidar_cam'] = None

        instance = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                instance_info = {}
                line = [x for x in line.strip().split(' ')]
                box3d = [float(x) for x in line[:6]]

                class_id = class_names.index(line[-1])

                instance_info['bbox_label_3d'] = class_id
                instance_info['bbox_3d'] = box3d
                instance_info['label_path'] = id + '.txt'
                instance.append(instance_info)
        
        # 'image', 'point_cloud', 'calib', 'annos'

        info["sample_idx"] = idx_info
        info['point_cloud'] = point_info
        info['annos'] = instance
        # info['image'] = image_info
        infos.append(info)

    # info['calib'] = calib_info
    return infos



# 根据前缀和路径调用create_my_infos函数，
def create_my_infos(root_path, info_prefix):
    """
    处理数据集的激光雷达数据
    """
    # 类别信息
    metainfo = dict(categories=class_names)

    imageset_dir = os.path.join(root_path, "ImageSets") 
    if not os.path.exists(imageset_dir):
        print("Don't exist ImageSets dir")
        return None,None


     # _read_imageset_file返回的顺序列表  得到文件名前缀
    train_ids = _read_imageset_file(os.path.join(imageset_dir, "train.txt"))
    val_ids = _read_imageset_file(os.path.join(imageset_dir, "val.txt"))
    test_ids = _read_imageset_file(os.path.join(imageset_dir, "test.txt"))

    train_infos = get_infos(root_path, train_ids)
    val_infos = get_infos(root_path, val_ids)
    test_infos = get_infos(root_path, test_ids)


    # 使用 mmengine.dump 保存成 pkl 文件
    if train_infos is not None:
        # 2个数据一个datalist一个是类别
        data = dict(data_list=train_infos, metainfo=metainfo)
        
        info_path = os.path.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)
    
    
    if val_infos is not None:
        data['data_list'] = val_infos
        info_val_path = os.path.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmengine.dump(data, info_val_path)
    
    if test_infos is not None:
        data['data_list'] = test_infos
        info_test_path = os.path.join(root_path,
                                 '{}_infos_test.pkl'.format(info_prefix))
        mmengine.dump(data, info_test_path)

def _read_imageset_file(path):
    ids = []
    with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                ids.append(line)
    return ids

def _fill_trainval_infos(root_path):
    """
    划分训练集和验证集
    """
    train_infos = []
    val_infos = []

    """
    这部分自己写，步骤：
    1. 读取自定义数据集的 ann info 文件
    2. 将自定义的 ann info 信息处理成 dict
    3. 将 dict 存入 train_infos 和 val_infos 列表
    """

    # 图像设置路径
    imageset_dir = os.path.join(root_path, "ImageSets") 
    if not os.path.exists(imageset_dir):
        print("Don't exist ImageSets dir")
        return None,None
    
    # _read_imageset_file返回的顺序列表  得到文件名前缀
    train_ids = _read_imageset_file(os.path.join(imageset_dir, "train.txt"))
    val_ids = _read_imageset_file(os.path.join(imageset_dir, "val.txt"))
    test_ids = _read_imageset_file(os.path.join(imageset_dir, "test.txt"))
    

    # 点云路径和标签路径
    lidar_dir = os.path.join(root_path, 'points')
    label_dir = os.path.join(root_path, "labels")
    
    # 标注文件
    label_names = os.listdir(label_dir)  # 所有标注文件名
    frame_idx = 0  # 计数器
    
    ann_info = {}
    for label_name in label_names:
        label_path = os.path.join(label_dir, label_name)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = [x for x in line.strip().split(' ')]
                # print(line)
                box = np.array(line[:6], dtype=np.float32)
                cate_name = line[-1]
                print(box, cate_name)              
        
        lidar_name = label_name.split('.')[0] + '.bin'
        lidar_path = os.path.join(lidar_dir, lidar_name)
        print(lidar_name)
        ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            
    
    info = {}

    train_infos.append(info)
    frame_idx += 1

    return train_infos, val_infos