import open3d as o3d
import numpy as np
import os

import cv2
import matplotlib.pyplot as plt

from common import read_calib_file
from common import get_rgb_by_distance, get_color_jet
from common import  rot_y
from PIL import Image

custom_colormap = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)


# dataroot = "/home/lin/code/OpenPCDet/data/kitti/training"
dataroot = "/home/lin/code/mmdetection3d/data/custom2"



lidar_dir = os.path.join(dataroot, "velodyne")
camera_dir= os.path.join(dataroot, "image_2")
calib_dir = os.path.join(dataroot, "calib")
label_dir = os.path.join(dataroot, "label_2")

for i in range(len(os.listdir(lidar_dir))):
    lidar_name = str(i).zfill(6) + '.bin'
    lidar_path = os.path.join(lidar_dir, lidar_name)
    
    points = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 4])

    # points_o3d = o3d.geometry.PointCloud()
    # points_o3d.points = o3d.utility.Vector3dVector(points[:, :3])  # 取numpy的前3维度
    # o3d.visualization.draw_geometries([points_o3d])

    camera_name = str(i).zfill(6) + '.png'
    image_path = os.path.join(camera_dir, camera_name)
    image = cv2.imread(image_path)
    # cv2.imshow("view", image)
    # cv2.waitKey()

    calib_name = str(i).zfill(6) + '.txt'
    calib_path = os.path.join(calib_dir, calib_name)
    calib_data = read_calib_file(calib_path)


    T_cam0_lidar = np.eye(4)
    T_cam0_lidar[: 3, :] = np.reshape(calib_data['Tr_velo_to_cam'], (3, 4))

    # K0 = np.eye(4)
    # K0[: 3, :] = np.reshape(calib_data['P0'], (3, 4))

    # K2 cam0 -> cam2 的点变换 T_cam2_cam0
    K2 = np.eye(4)
    K2[:3, :] = np.reshape(calib_data['P2'], (3, 4))

    # 相机旋转
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = np.reshape(calib_data['R0_rect'], (3, 3))


    # P2 * R0_rect * Tr_velo_to_cam
    # img = P2 @ R0_rect @ Tr_velo_to_cam @ velo
    # lidar -> img
    T_img2_lidar = K2 @ R0_rect @ T_cam0_lidar
    # print(T_img2_lidar)

    # lidar ->cam
    T_cam2_lidar = R0_rect @ T_cam0_lidar
    print(T_cam2_lidar)
    # 齐次坐标
    points_homo = np.concatenate([points[:, :3], np.ones([len(points), 1])], axis=1)

    # 删除激光后方的点云 所有points_cam的第一列是激光的x轴（车体正前方）  np.delete(axis=1)按列删除
    # points_homo = np.delete(points_homo, np.where(points_homo[0, :] < 0), axis=1)

    # 点云变换到像素平面下
    points_cam = T_img2_lidar @ points_homo.T

    # 归一化像素平面
    points_cam[:2, :] /= points_cam[[2], :]

    # 删除相机归一化坐标 z < 0 的点
    points_cam = np.delete(points_cam, np.where(points_cam[2, :] <= 0), axis=1)
    
    # 归一化距离L
    dist_points = np.linalg.norm(points, axis=1).astype(float)
    dist_max = np.max(dist_points).astype(float)
    dist_min = np.min(dist_points).astype(float)

    # 遍历点  把点云投影到图像上
    for j in range(points_cam.shape[1]):
        uv = points_cam[:2, j].astype(int)

        thre = get_rgb_by_distance(points_cam[3, j])
        

        color_id = int(get_color_jet(dist_points[j], dist_max, dist_min))
        # custom_colormap[color_id][0]
        color_list = custom_colormap[color_id][0].tolist()
        # print(color_list)
        # b, g, r =  int(custom_colormap[color_id][0][0]), int(custom_colormap[color_id][0][1]), int(custom_colormap[color_id][0][2])
        # print(b, g, r)
        cv2.circle(image, uv, 1, color_list, -1, 16)

    # label处理   type  truncated  occluded  alpha  bbox_2d(lt-rd,pixel)  dimensions-3d(m), location_3d, rotation_y_3d, score
    
    label_name = str(i).zfill(6) + '.txt'
    label_path = os.path.join(label_dir, label_name)
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()

            name = line[0]
            bbox = [line[4], line[5], line[6], line[7]]
            bbox = np.array([float(x) for x in bbox]).astype(np.int16)

            dimensions = [line[8], line[9], line[10]]
            dimensions = np.array([ float(x) for x in dimensions])

            location = [line[11], line[12], line[13]]
            location = np.array([float(x) for x in location])
            
            rotation_y = float(line[14])


            # print(name, bbox, dimensions, location, rotation_y)
            # cv2.rectangle(image, bbox[0: 2], bbox[2: 4], [0, 255, 0], 2)

            R = rot_y(rotation_y)
            h, w, l = dimensions[0], dimensions[1], dimensions[2]  #  根据人身高尺寸 hwl 的到顺序 高宽长

            # 得到该物体的坐标以底面为原点中心所在的物体坐标系下各个点的坐标
            #     7 -------- 4
            #    /|         /|
            #   6 -------- 5 .
            #   | |        | |
            #   . 3 -------- 0
            #   |/ z *- - -|/ - - -> (x)  水平向右
            #   2 ---|----- 1
            #        |
            #        | (y)  垂直于底向下
            #        v

            #   z   向指向里面             相机坐标系

            # 8个点的坐标   xyz对应lhw
            x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]

            # 根据y可以确认 location是底面的中心
            y = [0, 0, 0, 0, -h, -h, -h, -h]

            z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

            # xyz : lhw

            # 12 根线的下标
            index_x = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
            index_y = [1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7]

            # 将xyz转化成3*8的矩阵
            corner_3d = np.vstack([x, y, z])
            # R * X  只有旋转3x3就行 将框选择yaw
            corner_3d = R @ corner_3d   # 将以地面中心为原点,12个点旋转yaw(转向角)

            # print(R)
            # print(rotation_y)
            # print('\n')

            
            # 将框移动到xyz位置，将该物体移动到相机坐标系下的原点处（涉及到坐标的移动，直接相加就行）
            corner_3d[0, :] += location[0]
            corner_3d[1, :] += location[1]
            corner_3d[2, :] += location[2]   

            # corner_3d是相机系下坐标 ， 需要转换到图像系
            # 将3d的bbox转换到2d坐标系中（需要用到内参矩阵)
            # 因为K2是4x4的矩阵，这里需要给corner_3d变成齐次坐标
            corner_3d = np.vstack((corner_3d, np.zeros((1, corner_3d.shape[-1]))))
            corner_2d = K2 @ corner_3d

            # 在像素坐标系下，横坐标x = corner_2d[0, :] /= corner_2d[2, :]
            # 纵坐标的值以此类推  # 归一化 得到 uv值
            corner_2d[0, :] /= corner_2d[2, :]
            corner_2d[1, :] /= corner_2d[2, :]

            # 转成int
            corner_2d = np.array(corner_2d).astype(np.int16)

            color = [0, 255, 0]

            corner_2d = corner_2d.T  # 转置后是8*4的维度
            # print(corner_2d)


            uv0 =  corner_2d[0]
            uv1 =  corner_2d[1]
            uv2 =  corner_2d[2]
            uv3 =  corner_2d[3]
            
            u1 = int((uv2[0] + uv3[0]) / 2)
            v1 = int((uv2[1] + uv3[1]) / 2)
            
            u2 = int((uv0[0] + uv1[0]) / 2)
            v2 = int((uv0[1] + uv1[1]) / 2)
            
            u0 = int((u1 + u2) / 2)
            v0 = int((v1 +v2) / 2)

            # cv2.line(image, (uv0[:2]), (uv3[:2]), [0, 255, 0], 2, 16)
            # cv2.line(image, (uv0[:2]), (uv1[:2]), [0, 255, 0], 2, 16)
            # cv2.line(image, (uv1[:2]), (uv2[:2]), [0, 255, 0], 2, 16)

            
            # cv2.line(image, (u0, v0), (u2, v2), [0, 0, 255], 2, 16)

            for p0, p1 in zip(corner_2d[index_x], corner_2d[index_y]):
                # 排除z小于0的点
                if p0[2] <= 0 or p1[2] <= 0:
                    continue

                cv2.line(image, (p0[0], p0[1]), (p1[0], p1[1]), [0, 255, 0], 2, 16)
                
                
                

    cv2.imshow("image", image)

    if cv2.waitKey() == ord('q'):
        break
