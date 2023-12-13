import numpy as np
from pypcd import pypcd
import os

# 4维度点云
def pcd2bin(pcd_dir, bin_dir):
    # 创建bin文件的保存路径
    if not os.path.exists(bin_dir):
        os.mkdir(pcd_dir)
    # 获得pcd文件列表
    pcd_names = os.listdir(pcd_dir)

    # 遍历pcd文件并保存为同名的bin文件
    for pcd_name in pcd_names:
        pcd_path = os.path.join(pcd_dir, pcd_name)
        pcd_data = pypcd.PointCloud.from_path(pcd_path)
        # 建立一个全为0的空的点云维度 n * 4
        points = np.zeros([pcd_data.width, 4], dtype=np.float32)  
        
        # 把pcd的xyz维度拷贝到points上
        points[:, 0] = pcd_data.pc_data['x'].copy()
        points[:, 1] = pcd_data.pc_data['y'].copy()
        points[:, 2] = pcd_data.pc_data['z'].copy()
        points[:, 3] = pcd_data.pc_data['intensity'].copy().astype(np.float32)
        
        bin_path = os.path.join(bin_dir, pcd_name.replace("pcd", "bin"))
        
        with open(bin_path, 'wb') as f:
            f.write(points.tobytes())

def main(arg=None):
    # pcd_dir = ""
    # bin_dir =  
    # pcd2bin(pcd_dir, bin_dir)