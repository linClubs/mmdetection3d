# 1 生成数据集

提供了 3 种不同的方式：
+ 支持新的数据格式、
+ 将新数据的格式转换为现有数据的格式、
+ 将新数据集的格式转换为一种当前可支持的中间格式

本质都是 4 步：

1. 将数据集预处理成 pkl 文件
2. 创建自定义数据集类，并在数据集中读取并解析 pkl 文件
3. 创建训练的配置文件
4. 创建评价类 Metric

---

~~~python
python tools/create_data.py custom --root-path ./data/custom3 --out-dir ./data/custom3 --extra-tag custom
~~~

1. 以kitti格式pkl介绍

`create_data.py`的主函数调用`kitti_data_prep`函数
`kitti_data_prep`函数中主要看下面2个函数
~~~python
# 1. create_kitti_info_file改函数把标注信息读如字典
kitti.create_kitti_info_file(root_path, info_prefix, with_plane)


# 2. update_pkl_infos 把上面的字典转成标准的mmdet3d支持字典格式
update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_train_path)
~~~

2. 标准`mmdet3d`的字典里面含有一下字符

~~~python
info = {
        'sample_idx': i,
        'timestamp': time_stamp,
        'lidar_points': dict(),
        'images': dict(),
        'instances': [],
        'cam_instances': dict(),
    }
~~~


3. 根据上面的流程自制的数据集也需要读入标注数据然后转成标准的mmdet3d格式。

+ 在`create_data.py`的主函数增加一个`custom_data_prep`函数
+ 在`tools/dataset_converters/`增加一个`custom_converter.py`文件


~~~python
# 在tools/dataset_converters/增加一个custom_converter.py文件
def custom_data_prep(root_path,
                  info_prefix,
                  dataset_name,
                  out_dir):
    custom_converter.create_custom_dataset_infos(root_path, info_prefix)

if __name__ == '__main__':
    ...
    elif args.dataset == 'custom':
            custom_data_prep(
                root_path=args.root_path,          # 数据集路径
                info_prefix=args.extra_tag,        # 生成文件名前缀
                dataset_name='customDataset',
                out_dir=args.out_dir)
~~~

+ `custom_converter.py`文件内容

需要更改类别数和索引号和`_fill_trainval_infos`函数中原始数据得各个路径

~~~python
# custom_converter.py
import os
from os import path as osp
import mmengine
from pyquaternion import Quaternion
import json
import numpy as np
import open3d as o3d
import math

# 类别
pcdClass_names = ['car', 'truck', 'trailer', 'bus', 'construction',
                'bicycle', 'motorcycle', 'pedestrian', 'trafficcone', 'barrier']
# id号
pcdClass_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

pcdCategories = dict(zip(pcdClass_names, pcdClass_order))
# imgClass_order = []
# imgCatagores = dict(zip())

def create_custom_dataset_infos(root_path, info_prefix):
    
    # 划分训练和测试的并返回了info 
    train_infos, val_infos = _fill_trainval_infos(root_path)
    metainfo = {
        'categories': pcdCategories,
        'dataset': 'custom_dataset', 
        'info_version': 1.0,
    }
    
    # 如果训练的info存在
    if train_infos is not None:
        # 创建新的词典 2个key['data_list', 'metainfo']
        data = dict(data_list=train_infos, metainfo=metainfo)
        # 文件名字
        info_path = osp.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        # 写入词典
        mmengine.dump(data, info_path)

    # 如果测试的info存在
    if val_infos is not None:
        data['data_list'] = val_infos
        info_val_path = osp.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmengine.dump(data, info_val_path)

# 根据数据集内容使用
def add_difficulty_to_annos(bbox, occlusion, truncation):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    height = bbox[3] - bbox[1]
    diff = 0
    easy_mask = np.ones((1, ), dtype=bool)
    moderate_mask = np.ones((1, ), dtype=bool)
    hard_mask = np.ones((1, ), dtype=bool)

    if occlusion > max_occlusion[0] or truncation > max_trunc[0]:
        easy_mask = False
    if occlusion > max_occlusion[1] or truncation > max_trunc[1]:
        moderate_mask = False
    if occlusion > max_occlusion[2] or truncation > max_trunc[2]:
        hard_mask = False

        
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)
    
    if is_easy:
        diff = 0
    elif is_moderate:
        diff = 1
    elif is_hard:
        diff = 2
    else:
        diff = -1

    return diff

def _fill_trainval_infos(root_path):

    train_infos = []
    val_infos = []
    use_camera = True

    trainSet = root_path + '/ImageSets/train.txt'
    valSet = root_path + '/ImageSets/val.txt'
    train_dict  , val_dict = set(), set()

    # 读取作为训练集的名称前缀
    with open(trainSet, 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')
            train_dict.add(ann)
    
    # 读取作为测试集的名称前缀
    with open(valSet, 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')
            val_dict.add(ann)

    # 点云文件名
    totalPoints = os.listdir(root_path + '/points')
    for i in range(len(totalPoints)):
        
        # 文件名字 点云去掉'.bin' 4位数
        file_name = totalPoints[i][:-4]
        
        lidar_path = root_path + '/points/' + file_name + '.bin'
        img_path = root_path + '/images/' + file_name + '.png'
        label_path = root_path + '/labels/' + file_name + '.txt'
        
        # 检测文件是否存在
        mmengine.check_file_exist(lidar_path)
        mmengine.check_file_exist(img_path)
        mmengine.check_file_exist(label_path)
        
        # 时间戳
        # time_stamp_list = file_name.split('_')
        # time_stamp = int(time_stamp_list[0][-4:]) + int(time_stamp_list[1]) / (10 * len(time_stamp_list[1]))
        time_stamp = str(i).zfill(10)

        # 标注信息的词典
        info = {
            'sample_idx': i,
            'timestamp': time_stamp,
            'lidar_points': dict(),
            'images': dict(),
            'instances': [],
            'cam_instances': dict(),
        }
        
        # lidar_points 相关参数  lidar路径，点云维度，lidar2cam4x4
        info['lidar_points']['lidar_path'] = lidar_path
        info['lidar_points']['num_pts_feats'] = 4
        info['lidar_points']['Tr_velo_cam'] = np.array([
                                                        [0.79807554, 0.60254895, 0.00319398, 0.18529999999999987],
                                                        [ 0.2647308, -0.34586413, -0.90016421, 0.12779,],
                                                        [-0.54128832, 0.71924458, -0.43553896, -0.12140999999999998],
                                                        [0, 0, 0, 1]
                                                    ])
        # imu的外参为None
        info['lidar_points']['Tr_imu_to_velo'] = None

        # 相机信息
        cameras = [
            'cam62',
            # 'cam63',
            # 'cam64',
        ]


        # image 相关参数
        for cam_name in cameras:
            if cam_name not in info['images']:
                info['images'][cam_name] = dict()
                info['cam_instances'][cam_name] = []
            cam_path = img_path

            info['images'][cam_name]['img_path'] = cam_path
            info['images'][cam_name]['height'] = 1080
            info['images'][cam_name]['width'] = 1920
            info['images'][cam_name]['cam2img'] = np.array([
                                                        [1158.52, 0, 964.76, 0],
                                                        [0, 1153.0, 545.86, 0],
                                                        [0, 0, 1.0, 0],
                                                        [0, 0, 0, 1]
                                                    ])

            info['images'][cam_name]['lidar2cam'] = np.array([
                                                        [0.79807554, 0.60254895, 0.00319398, 0.18529999999999987],
                                                        [ 0.2647308, -0.34586413, -0.90016421, 0.12779,],
                                                        [-0.54128832, 0.71924458, -0.43553896, -0.12140999999999998],
                                                        [0, 0, 0, 1]
                                                    ])
            info['images'][cam_name]['lidar2img'] = info['images'][cam_name]['cam2img'] @ info['images'][cam_name]['lidar2cam']
            
            # print(info['images'][cam_name])
        info['images']['R0_rect'] = np.array([
                                            [1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]
                                        ])


        with open(label_path, 'r', encoding='utf-8') as f:
            # i = 0
            for ann in f.readlines():
                ann = ann.strip('\n')
                ann = ann.split()
                if len(ann):
                    # instances
                    info['instances'].append(dict())
                    # 2b框不能用
                    info['instances'][-1]['bbox'] = [float(ann[4]), float(ann[5]), float(ann[6]), float(ann[7])]
                    info['instances'][-1]['bbox_label'] = pcdCategories[ann[0]]
                    
                    # 3d框 xyz hwl 注意kitti的顺序 mmdet3d的顺序为 xyzlwh
                    info['instances'][-1]['bbox_3d'] = [float(ann[11]), float(ann[12]), float(ann[13]), float(ann[10]), float(ann[8]), float(ann[9]), float(ann[14])]
                    info['instances'][-1]['bbox_label_3d'] = pcdCategories[ann[0]]
                    
                    info['instances'][-1]['num_lidar_pts'] = None # 如果没有需要使用的地方，可以用None代替
                    info['instances'][-1]['alpha'] = float(ann[3])
                    info['instances'][-1]['occluded'] = int(float(ann[2]))
                    info['instances'][-1]['truncated'] = int(float(ann[1]))
                    info['instances'][-1]['difficulty'] = int(float(ann[2])) if int(float(ann[2])) < 2 else 2
                    info['instances'][-1]['depth'] = None # 如果没有需要使用的地方，可以用None代替
                    info['instances'][-1]['center_2d'] = None # 如果没有需要使用的地方，可以用None代替
                    info['instances'][-1]['group_id'] = None # 如果没有需要使用的地方，可以用None代替
                    info['instances'][-1]['index'] = None # 如果没有需要使用的地方，可以用None代替
                    
                    # cam_instances
                    info['cam_instances']['cam62'].append(dict())
                    info['cam_instances']['cam62'][-1]['bbox'] = [float(ann[4]), float(ann[5]), float(ann[6]), float(ann[7])]
                    info['cam_instances']['cam62'][-1]['bbox_label'] = pcdCategories[ann[0]]
                    info['cam_instances']['cam62'][-1]['bbox_3d'] = [float(ann[11]), float(ann[12]), float(ann[13]), float(ann[10]), float(ann[8]), float(ann[9]), float(ann[14])]
                    info['cam_instances']['cam62'][-1]['bbox_label_3d'] = pcdCategories[ann[0]]
                    info['cam_instances']['cam62'][-1]['bbox_3d_isvalid'] = True
                    info['cam_instances']['cam62'][-1]['velocity'] = None # 如果没有需要使用的地方，可以用None代替
                    info['cam_instances']['cam62'][-1]['center_2d'] = None # 如果没有需要使用的地方，可以用None代替
                    info['cam_instances']['cam62'][-1]['depth'] = None # 如果没有需要使用的地方，可以用None代替
                    
                            # i += 1
        
        if file_name in train_dict:
            train_infos.append(info)
        else:
            val_infos.append(info)
                
    return train_infos, val_infos
~~~

# 3 查看pkl内容

~~~python
import pickle

# 读取 pickle 文件
pkl_path = 'data/custom3/custom_infos_train.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# 打印数据
print(data.keys())

print(data)
~~~

# 4 创建自定义数据集类

+ 自定义数据集类的作用是从路径中读取数据pkl文件，并处理成 mmdet3d 模型的格式。

因为读取数据部分放在 pipeline 中，所以只需要解析 ann info 并 return 即可。mmdet3d 的包围框 gt_bboxes_3d 有固定的格式：LiDARInstance3DBoxes 。创建数据集的代码如下：

+ 在 mmdet3d/datasets/下增加`1custom_dataset.py`内容如下

~~~python
from typing import Callable, List, Union

import numpy as np
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.datasets.det3d_dataset import Det3DDataset

@DATASETS.register_module()
class customDataset(Det3DDataset):

    # 替换成自定义 pkl 信息文件里的所有类别
    METAINFO = {
        'classes':
        ('car', 'truck', 'trailer', 'bus', 'construction',
                  'bicycle', 'motorcycle', 'pedestrian', 'trafficcone', 'barrier')
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = 'cam62',
                 load_type: str = 'frame_based',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 1],
                 **kwargs) -> None:

        self.pcd_limit_range = pcd_limit_range
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera')
        
    def parse_ann_info(self, info):
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # 空实例
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # 过滤掉没有在训练中使用的类别
        # lidar2cam = np.array(info['images']['CAM2']['lidar2cam'])
        ann_info = self._remove_dontcare(ann_info)
        # gt_bboxes_3d = LiDARInstance3DBoxes(
        #     ann_info['gt_bboxes_3d'],
        #     box_dim=ann_info['gt_bboxes_3d'].shape[-1],
        #     origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        # 因为已经转成kitti的格式了，LiDARInstance3DBoxes给默认参数即可
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
~~~



训练的时候，首先进入 parse_data_info 函数，然后调用 parse_ann_info 函数。其中的参数 info 是 读取的 pkl 格式。其中，METAINFO 根据自定义数据集的实际类别情况修改。下面 gt_bboxes_3d 也根据实际情况自行编写：

~~~python
'''
 1 在./mmdet3d/datasets/transforms/loading.py中末尾
# 添加一个customLoadPointsFromFile函数 
'''
@TRANSFORMS.register_module()
class customLoadPointsFromFile(LoadPointsFromFile):

    def _load_pcd_points(self, pts_filename: str) -> np.ndarray:
        """读取 bin 文件， 得到 np.ndarray(N, 4)
        """
        points = np.fromfile(pts_filename).astype(np.float32).reshape([-1, 4])
        return points

    def transform(self, results: dict) -> dict:
        """
            生成 LiDARPoints 格式
        """
        pts_file_path = results['lidar_path']
        points = self._load_pcd_points(pts_file_path)  # (N, 4)

        points = points[:, self.use_dim]

        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1])
        results['points'] = points

        return results

# 2 在mmdet3d/datasets/transforms/__init__.py导入上面customLoadPointsFromFile
from .loading import customLoadPointsFromFile
# 在__all__ = 加入'customLoadPointsFromFile'
__all__ = [ 
     # ...
    'customLoadPointsFromFile',
    ]

# 在mmdet3d/datasets/__init__.py加入下面语句
from .custom_dataset import customDataset 
from .transforms import customLoadPointsFromFile
__all__ = [
        # ...
        'customDataset',
        'customLoadPointsFromFile'
    ]
~~~
