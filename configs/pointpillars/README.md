# PointPillars: Fast Encoders for Object Detection from Point Clouds

> [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)

<!-- [ALGORITHM] -->

## Abstract

Object detection in point clouds is an important aspect of many robotics applications such as autonomous driving. In this paper we consider the problem of encoding a point cloud into a format appropriate for a downstream detection pipeline. Recent literature suggests two types of encoders; fixed encoders tend to be fast but sacrifice accuracy, while encoders that are learned from data are more accurate, but slower. In this work we propose PointPillars, a novel encoder which utilizes PointNets to learn a representation of point clouds organized in vertical columns (pillars). While the encoded features can be used with any standard 2D convolutional detection architecture, we further propose a lean downstream network. Extensive experimentation shows that PointPillars outperforms previous encoders with respect to both speed and accuracy by a large margin. Despite only using lidar, our full detection pipeline significantly outperforms the state of the art, even among fusion methods, with respect to both the 3D and bird's eye view KITTI benchmarks. This detection performance is achieved while running at 62 Hz: a 2 - 4 fold runtime improvement. A faster version of our method matches the state of the art at 105 Hz. These benchmarks suggest that PointPillars is an appropriate encoding for object detection in point clouds.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644370/143885905-aab6ffcf-7727-495e-90ca-edb8dd5e324b.png" width="800"/>
</div>

## Introduction

We implement PointPillars and provide the results and checkpoints on KITTI, nuScenes, Lyft and Waymo datasets.

## Results and models

### KITTI

|                            Backbone                             |  Class  |   Lr schd   | Mem (GB) | Inf time (fps) |  AP   |                                                                                                                                                                                                         Download                                                                                                                                                                                                         |
| :-------------------------------------------------------------: | :-----: | :---------: | :------: | :------------: | :---: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  [SECFPN](./pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py)   |   Car   | cyclic 160e |   5.4    |                | 77.6  |       [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606.log.json)       |
| [SECFPN](./pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py) | 3 Class | cyclic 160e |   5.5    |                | 64.07 | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306.log.json) |

### nuScenes

|                                Backbone                                 | Lr schd | Mem (GB) | Inf time (fps) |  mAP  |  NDS  |                                                                                                                                                                                                     Download                                                                                                                                                                                                     |
| :---------------------------------------------------------------------: | :-----: | :------: | :------------: | :---: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      [SECFPN](./pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py)       |   2x    |   16.4   |                | 34.33 | 49.1  |   [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857.log.json)   |
| [SECFPN (FP16)](./pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py) |   2x    |   8.37   |                | 35.19 | 50.27 | [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626.log.json) |
|         [FPN](./pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py)          |   2x    |   16.3   |                | 39.7  | 53.2  |         [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936.log.json)         |
|    [FPN (FP16)](./pointpillars_hv_fpn_sbn-all_8xb2-amp-2x_nus-3d.py)    |   2x    |   8.40   |                | 39.26 | 53.26 |       [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719.log.json)       |

### Lyft

|                           Backbone                            | Lr schd | Mem (GB) | Inf time (fps) | Private Score | Public Score |                                                                                                                                                                                                     Download                                                                                                                                                                                                     |
| :-----------------------------------------------------------: | :-----: | :------: | :------------: | :-----------: | :----------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [SECFPN](./pointpillars_hv_secfpn_sbn-all_8xb2-2x_lyft-3d.py) |   2x    |   12.2   |                |     13.8      |     14.1     | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d_20210829_100455-82b81c39.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d_20210829_100455.log.json) |
|    [FPN](./pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d.py)    |   2x    |   9.2    |                |     14.8      |     15.0     |       [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d_20210822_095429-0b3d6196.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d_20210822_095429.log.json)       |

### Waymo

|                                 Backbone                                 | Load Interval |  Class  | Lr schd | Mem (GB) | Inf time (fps) | mAP@L1 | mAPH@L1 | mAP@L2 | **mAPH@L2** |                                                                                                                                                                                                                   Download                                                                                                                                                                                                                   |
| :----------------------------------------------------------------------: | :-----------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :----: | :---------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  [SECFPN](./pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-car.py)   |       5       |   Car   |   2x    |   7.76   |                |  70.2  |  69.6   |  62.6  |    62.1     |       [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car_20200901_204315-302fc3e7.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car_20200901_204315.log.json)       |
| [SECFPN](./pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class.py) |       5       | 3 Class |   2x    |   8.12   |                |  64.7  |  57.6   |  58.4  |    52.1     | [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class_20200831_204144-d1a706b1.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class_20200831_204144.log.json) |
|                               above @ Car                                |               |         |   2x    |   8.12   |                |  68.5  |  67.9   |  60.1  |    59.6     |                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|                            above @ Pedestrian                            |               |         |   2x    |   8.12   |                |  67.8  |  50.6   |  59.6  |    44.3     |                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|                             above @ Cyclist                              |               |         |   2x    |   8.12   |                |  57.7  |  54.4   |  55.5  |    52.4     |                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|   [SECFPN](./pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car.py)    |       1       |   Car   |   2x    |   7.76   |                |  72.1  |  71.5   |  63.6  |    63.1     |                                                                                                                           [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-car/hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-car.log.json)                                                                                                                            |
|  [SECFPN](./pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-3class.py)  |       1       | 3 Class |   2x    |   8.12   |                |  68.8  |  63.3   |  62.6  |    57.6     |                                                                                                                        [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-3class/hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-3class.log.json)                                                                                                                         |
|                               above @ Car                                |               |         |   2x    |   8.12   |                |  71.6  |  71.0   |  63.1  |    62.5     |                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|                            above @ Pedestrian                            |               |         |   2x    |   8.12   |                |  70.6  |  56.7   |  62.9  |    50.2     |                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|                             above @ Cyclist                              |               |         |   2x    |   8.12   |                |  64.4  |  62.3   |  61.9  |    59.9     |                                                                                                                                                                                                                                                                                                                                                                                                                                              |

#### Note:

- **Metric**: For model trained with 3 classes, the average APH@L2 (mAPH@L2) of all the categories is reported and used to rank the model. For model trained with only 1 class, the APH@L2 is reported and used to rank the model.
- **Data Split**: Here we provide several baselines for waymo dataset, among which D5 means that we divide the dataset into 5 folds and only use one fold for efficient experiments. Using the complete dataset can boost the performance a lot, especially for the detection of cyclist and pedestrian, where more than 5 mAP or mAPH improvement can be expected.
- **Implementation Details**: We basically follow the implementation in the [paper](https://arxiv.org/pdf/1912.04838.pdf) in terms of the network architecture (having a
  stride of 1 for the first convolutional block). Different settings of voxelization, data augmentation and hyper parameters make these baselines outperform those in the paper by about 7 mAP for car and 4 mAP for pedestrian with only a subset of the whole dataset. All of these results are achieved without bells-and-whistles, e.g. ensemble, multi-scale training and test augmentation.
- **License Aggrement**: To comply the [license agreement of Waymo dataset](https://waymo.com/open/terms/), the pre-trained models on Waymo dataset are not released. We still release the training log as a reference to ease the future research.
- `FP16` means Mixed Precision (FP16) is adopted in training. With mixed precision training, we can train PointPillars with nuScenes dataset on 8 Titan XP GPUS with batch size of 2. This will cause OOM error without mixed precision training. The loss scale for PointPillars on nuScenes dataset is specifically tuned to avoid the loss to be Nan. We find 32 is more stable than 512, though loss scale 32 still cause Nan sometimes.

## Citation

```latex
@inproceedings{lang2019pointpillars,
  title={Pointpillars: Fast encoders for object detection from point clouds},
  author={Lang, Alex H and Vora, Sourabh and Caesar, Holger and Zhou, Lubing and Yang, Jiong and Beijbom, Oscar},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={12697--12705},
  year={2019}
}
```