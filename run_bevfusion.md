# 基于mmdetection3d框架配置BEVFusion

+ 由于`mit-han-lab/bevfusion`中的配置文件是yaml格式，比较绕, 
+ 目前`mmdetection3d`也支持了`BEVFusion`，可以直接使用`mmdet3d`项目验证`BEVFusion`

# 1 ENV

**ubuntu20.04, cuda-11.3, cudnn-8.6,torch-1.10.0, mmdet3d-1.3.0**

~~~python
# 0 安装依赖
sudo apt install wget git g++ cmake ffmpeg libsm6 libxext6

# 1 创建虚拟环境
conda create -n mmdet3d python=3.8

# 2 激活虚拟环境
conda activate mmdet3d

# 3 安装torch
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# 4 配置安装mmdetection3d
pip install openmim
# 下载的是mmdet3d是v1.3.0版本
git clone https://github.com/open-mmlab/mmdetection3d.git -b v1.3.0
cd mmdetection3d
# 使用mim可以自动配置mmcv,mmdet,mmengine
mim install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。


# 5 安装 cumm-cuxxx spconv-cuxxx
pip install cumm-cu113
pip install spconv-cu113


# 6 配置 mmdet3d中的BEVFusion
python projects/BEVFusion/setup.py develop
# 或者运行下面2句
# cd projects/BEVFusion
# pip install -v -e .

# 7 查看相关库的版本
## 7.1 openlab相关库版本
mim list
# 终端显示如下
mmcv       2.1.0      https://github.com/open-mmlab/mmcv
mmdet      3.2.0      https://github.com/open-mmlab/mmdetection
mmdet3d    1.3.0      /root/share/code/mmdetection3d
mmengine   0.10.1     https://github.com/open-mmlab/mmengine

## 7.2 torch相关库版本
pip list | grep torch
# 终端显示如下
torch                     1.10.0+cu113
torchaudio                0.10.0+rocm4.1
torchvision               0.11.0+cu113
~~~

# 2 生成数据集

+ nuscenes数据集下载具体下载细节参考[Fast-BEV代码复现实践](https://blog.csdn.net/h904798869/article/details/130317240?spm=1001.2014.3001.5502)的第2小节数据集准备内容

1. 如果是`nuscenes-mini`数据集,需要修改文件

+ `mmdet3d/datasets/nuscenes_dataset.py`文件中的`v1.0-trainval`改成`v1.0-mini`即可

+ `nuscenes-full`无需修改

2 生成pkl格式的数据集
~~~python
python tools/create_data.py nuscenes --root-path ./data/nuscenes --version v1.0-mini --out-dir ./data/nuscenes --extra-tag nuscenes 
~~~

运行完后 data/nuscenes目录如下所示：
~~~python
nuscenes
    ├── maps
    ├── nuscenes_dbinfos_train.pkl   # 新生成的文件
    ├── nuscenes_gt_database         # 新生成的目录
    ├── nuscenes_infos_train.pkl     # 新生成的文件
    ├── nuscenes_infos_val.pkl       # 新生成的文件
    ├── samples
    ├── sweeps
    └── v1.0-mini
~~~


# 3 训练

复制一份配置文件`projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`重命名为`only_lidar.py`

1. 训练
~~~python
# 配置文件中的max_epochs=2, batch_size=1, num_workers=0
# 上面三个参数按需更改，前期测试环境是否正常,可以按上面数字设置

# 1. 只训练lidar数据集
bash tools/dist_train.sh projects/BEVFusion/configs/only_lidar.py 1

# 2. lidar和相机数据共同训练
## 2.1 预训练权重
# 因为图像特征提取层配置的swin-transform需要下载预训练权重, 如果网络出问题, 可以加上代码下载即可
# 在cam_lidar.py配置文件全局搜索https://github.com,并在该地址前面加上https://mirror.ghproxy.com/即可

## 2.2 训练
# cam_lidar.py配置文件是继承only_lidar.py所以batch_size,num_workers需要在only_lidar.py中修改

### 2.2.1分布式训练
bash tools/dist_train.sh projects/BEVFusion/configs/cam_lidar.py 1

### 2.2.2 单步训练 加载数据时比较慢，不是卡住了,只要报错和卡住就等着
python tools/train.py projects/BEVFusion/configs/cam_lidar.py

'''
正常训练时，终端会打印信息如下：
...
12/13 17:56:37 - mmengine - INFO - Epoch(train) [1][150/408]  lr: 1.0551e-04  eta: 0:38:26  time: 0.9984  data_time: 0.0137  memory: 21795  grad_norm: 16.1443  loss: 12.3859  loss_heatmap: 2.6139  layer_-1_loss_cls: 3.9985  layer_-1_loss_bbox: 5.7735  matched_ious: 0.0343
...
'''
~~~

+ 训练完成结果(权重，配置文件，log，vis_data)后会保存在`work_dirs`目录下

+ 官方提供了训练好的权重, [参考](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion)


## 3.1 图像backbone改resnet
~~~python
# 将projects/BEVFusion/configs/cam_lidar.py中img_backbone与img_neck改成如下：
# mmdet.ResNet模式  ResNet50: depth=50   ResNet101: depth=101 
# pretrained权重路径对应上, 权重下载地址
'''
mkdir pretrained && cd pretrained
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
'''

img_backbone=dict(
            pretrained=pretrained_path + 'resnet50-0676ba61.pth',
            type='mmdet.ResNet',
            depth=50,
            num_stages=4,
            out_indices=[1, 2, 3],
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            with_cp=True,
            style='pytorch'
        ),

img_neck=dict(
            type='GeneralizedLSSFPN',
            in_channels=[512, 1024, 2048],
            out_channels=256,
            start_level=0,
            num_outs=3,
            norm_cfg=dict(type='BN2d', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
            upsample_cfg=dict(mode='bilinear', align_corners=False)
        ),
~~~

+ 训练
~~~python
python tools/train.py projects/BEVFusion/configs/cam_lidar.py
~~~


# 4 测试
~~~python
bash tools/dist_test.sh work_dirs/cam_lidar/cam_lidar.py work_dirs/cam_lidar/epoch_2.pth 1
~~~


# 5 可视化demo
~~~python
python projects/BEVFusion/demo/multi_modality_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl  work_dirs/cam_lidar/cam_lidar.py work_dirs/cam_lidar/epoch_6.pth --cam-type all --score-thr 0.2 --show
~~~

mini数据集只train了6个周期效果比较差：

<p align="center">
<img src="./docs/1.png" width="320" height="180"/> <img src="./docs/2.png" width="320" height="180"/>
</p> 


---

[参考](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion)
[更多bev算法部署](https://blog.csdn.net/h904798869/article/details/133279972)

---


