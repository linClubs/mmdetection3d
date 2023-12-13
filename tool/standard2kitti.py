import os
import cv2
import numpy as np
import shutil

def creatDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

data_root = "/home/lin/code/mmdetection3d/data/custom"
save_path = "/home/lin/code/mmdetection3d/data/custom2" 

cameras = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
count = 404

lidar_dir = os.path.join(data_root, "points")
lidar_names = os.listdir(lidar_dir)

image_dir = lidar_dir.replace('points', 'images/image1') # nus的前视角
calib_dir = lidar_dir.replace('points', 'calibs')
label_dir = lidar_dir.replace('points', 'labels')

save_calib_dir = os.path.join(save_path, "calib")
save_lidar_dir = os.path.join(save_path, "velodyne")
save_image_dir = os.path.join(save_path, "image_2")
save_label_dir = os.path.join(save_path, "label_2")

creatDir(save_path)
creatDir(save_calib_dir)
creatDir(save_lidar_dir)
creatDir(save_image_dir)
creatDir(save_label_dir)

for j, lidar_name in enumerate(lidar_names):
    # 1 lidar数据
    lidar_path = os.path.join(lidar_dir, lidar_name)
    # print(lidar_path)
    points = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 4])
    
    # 1.1 保存点云数据
    shutil.copy(lidar_path, save_lidar_dir)

    # 2 图像数据
    image_name = lidar_name.replace('bin', 'png')
    image_path = os.path.join(image_dir, image_name)
    # print(image_path)
    img = cv2.imread(image_path)

    ## 保存图像数据
    shutil.copy(image_path, save_image_dir)

    # 3 标定参数
    calib_name = lidar_name.replace('bin', 'txt')
    calib_path = os.path.join(calib_dir, calib_name)
    
    K0 = None
    Tr_velo_to_cam = None
    R0_rect = np.eye(3).reshape(-1)

    with open(calib_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            if line[0] == 'P1':
                K0 = [float(x) for x in line if x != line[0]]
                K0 = np.array(K0).reshape([3, 3])
            if line[0] == 'lidarcam1':
                Tr_velo_to_cam = [float(x) for x in line if x != line[0]]
                Tr_velo_to_cam = np.array(Tr_velo_to_cam).reshape([4, 4])
    f.close()

    T_cam_lidar = Tr_velo_to_cam 
    

    # 3.2 保存标注数据
    P0 = np.zeros([3, 4])
    P0[:, :3] = K0
    P0 = np.array(P0).reshape([-1])
    
    ss = 'P0:'
    for i in P0:
        ss += ' ' + str(i) 

    ss += '\nP1:'
    for i in P0:
        ss += ' ' + str(i) 

    ss += '\nP2:'
    for i in P0:
        ss += ' ' + str(i) 
    
    ss += '\nP3:'
    for i in P0:
        ss += ' ' + str(i) 
    
    ss += '\nR0_rect:'
    for i in R0_rect:
        ss += ' ' + str(i) 

    Tr_velo_to_cam = Tr_velo_to_cam[:3,:].reshape(-1)
    ss += "\nTr_velo_to_cam:"
    for i in Tr_velo_to_cam:
        ss += " " + str(i) 

    ss += "\nTr_velo_to_imu:"
    for i in Tr_velo_to_cam:
        ss += " " + str(i) 


    # print(ss)
    
    # print(Tr_velo_to_cam)

    # break

    with open(save_calib_dir + '/' + calib_name, 'w') as f2:
        f2.write(ss)

    # 4 标注数据3dbox
    label_name = lidar_name.replace('bin', 'txt')
    label_path = os.path.join(label_dir, label_name)


    ## 4.1 读取数据 并保存label
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip().split(' ')
            xyz = [float(x) for x in line[0: 3]]
            dxdydz = [float(x) for x in line[3: 6]]
            yaw = float(line[6])
            class_name = line[-1]
            
            xyz_hom = np.ones(4)
            xyz_hom[:3] = xyz
            xyz_hom = np.array(xyz_hom).reshape([4, -1])
            
            xyz_cam = (T_cam_lidar @ xyz_hom).reshape(-1)


            kitti = {}
            kitti["objectType"] = class_name
            kitti["truncated"] = "1.0"
            kitti["occluded"] = "0"
            kitti["alpha"] = "0.0"
            kitti["bbox"] = [0.00, 0.00, 50.00, 50.00]  # should be higher than 50
            
            # hwl  zyx
            kitti["diamensions"] = [dxdydz[2], dxdydz[1], dxdydz[0]] #height, width, length
            
            # xyz
            kitti["location"] = [xyz_cam[0], xyz_cam[1] + float(dxdydz[2])/2 , xyz_cam[2]] # camera coordinate
            kitti["rotation_y"] = -1.57-yaw
            if kitti["rotation_y"] > 1.57:
                kitti["rotation_y"] -= 3.14
            if kitti["rotation_y"] < -1.57:
                kitti["rotation_y"] += 3.14
            
            # print(save_label_dir + "/"  + label_name)
            with open(save_label_dir + "/"  + label_name, 'a+') as f1:
                # 写值
                for item in kitti.values():
                    if isinstance(item, list):
                        for temp in item:
                            f1.writelines(str(temp) + " ")
                    else:      
                        f1.writelines(str(item)+ " ")
                f1.writelines("\n")
        
        f.close()

    # 5 保存calib


    print( "save the %s frame successfully" %j)
    if j > count:
        break
        
    
