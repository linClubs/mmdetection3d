import torch
import numpy as np
import os
from mmengine import load

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes  # 默认底部中心， 设置origin=(0.5, 0.5, 0.5)


from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import CameraInstance3DBoxes

def vis_3dbox(points, box):
    visualizer = Det3DLocalVisualizer()
    # set point cloud in visualizer
    visualizer.set_points(points)

    # bottom center底部中心
    bboxes_3d = LiDARInstance3DBoxes(
        tensor=box,
        origin=(0.5, 0.5, 0.5)
        )
    
    # Draw 3D bboxes
    visualizer.draw_bboxes_3d(bboxes_3d)
    visualizer.show()



def vis_bev_box(bboxes_3d):
    # info_file = load('demo/data/kitti/000008.pkl')
    # bboxes_3d = []
    # for instance in info_file['data_list'][0]['instances']:
    #     bboxes_3d.append(instance['bbox_3d'])
    gt_bboxes_3d = np.array(bboxes_3d, dtype=np.float32)
    
    gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d)

    visualizer = Det3DLocalVisualizer()
    # set bev image in visualizer
    visualizer.set_bev_image()
    # draw bev bboxes
    visualizer.draw_bev_bboxes(gt_bboxes_3d, edge_colors='orange')
    visualizer.show()



dataroot = "/home/lin/code/mmdetection3d/data/custom"
lidar_dir = os.path.join(dataroot, "points")
label_dir = os.path.join(dataroot, "labels")

lidar_names = os.listdir(lidar_dir)


for i in range(len(lidar_names)):
    lidar_name = str(i).zfill(6) + ".bin"
    label_name = lidar_name.replace('bin', 'txt')
    
    lidar_path = os.path.join(lidar_dir, lidar_name)
    label_path = os.path.join(label_dir, label_name)
    # print(label_path)
    boxes = []
    with open(label_path) as f:
        lines = f.readlines()
        for line in lines:
            line = [x for x in line.strip().split(' ')]
            box = np.array(line[: 7]).astype(np.float32) 
            # box[2] = box[2] - box[5] / 2
            boxes.append(box)
    
    # print(lidar_path)
    points = np.fromfile(lidar_path, dtype=np.float32)
    points = points.reshape([-1, 4])
    
    box = torch.tensor(boxes)
    
    vis_3dbox(points, box)
    # vis_bev_box(boxes)





    break