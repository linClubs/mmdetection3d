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
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
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
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info