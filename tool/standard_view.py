import torch
import numpy as np
import os
import open3d as o3d



dataroot = "/home/lin/code/mmdetection3d/data/custom"
lidar_dir = os.path.join(dataroot, "points")
label_dir = os.path.join(dataroot, "labels")

lidar_names = os.listdir(lidar_dir)



colormap = [[255,  0, 0],  [127, 0, 0], [255, 255, 0], [255, 127, 0], 
            [255, 255, 127], [127, 255, 0], [127, 127, 0], [127, 255, 255], 
            [127, 255, 127], [0, 127, 0], [0, 127, 255], [0, 127, 127]]

nus_categories = ['car', 'truck', 'trailer', 'bus', 'construction',
                'bicycle', 'motorcycle', 'pedestrian', 'trafficcone', 'barrier']

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
            # box = np.array(line[: 7]).astype(np.float32) 
            # box[2] = box[2] - box[5] / 2
            boxes.append(line)
    
    # print(lidar_path)
    points = np.fromfile(lidar_path, dtype=np.float32)
    points = points.reshape([-1, 4])
    
    point_o3d = o3d.geometry.PointCloud()
    point_o3d.points = o3d.utility.Vector3dVector(points[:, :3])

    # point_o3d.colors = o3d.utility.Vector3dVector([0,0,1])


    vis = o3d.visualization.Visualizer()
    # vis.create_window()
    vis.create_window(window_name='可视化', width=1600, height=900)
    vis.add_geometry(point_o3d)

    # 添加box
    for i, box in enumerate(boxes):
        # 创建一个open3d的盒子类
        b = o3d.geometry.OrientedBoundingBox()
        b.center = box[:3]     # 中心点
        b.extent = box[3:6]    # dxdydz
        R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((0, 0, box[6])) # yaw角
        b.rotate(R, b.center)  # 中心点旋转yaw
        
        id = nus_categories.index(box[-1])
        b.color =colormap[id]   # 盒子的颜色
        
        # 创建坐标系几何体
        directions = R
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=b.center)
        coord_frame.rotate(R=directions, center=b.center)

        vis.add_geometry(coord_frame)
        vis.add_geometry(b) 

    # 创建坐标轴几何体
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    # 将坐标轴几何体添加到可视化窗口
    vis.add_geometry(coordinate_frame)


    vis.get_render_option().point_size = 1  # 点云的大小
    vis.get_render_option().background_color = np.asarray([0, 0, 0])  # 背景色 you can set the bg color
    vis.run()                # 窗口显示
    vis.destroy_window()     # 窗口销毁

    # 可视化保存为图像
    # vis.capture_screen_image("temp_%04d.jpg" % i)


    break