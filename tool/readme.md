 1. nus2kitti.py
 2. split_data.py 产生ImageSets
 
 
 
 3. create_data.py 产生pkl文件

# 1 点云保存，距离颜色值

 ~~~python
# 1 kitti命名 补位0填充
str_name = str(i).zfill(6)

# 2 距离颜色值
# 2.1 cv::COLORMAP_JET
custom_colormap = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)

# 2.2 映射到0-255
def get_color_jet(pixel, max, min):
    colormap_id = (pixel - min) / (max - min) * 255
    return colormap_id

## 2.3 从colormap取值 必须转成list
color_list = custom_colormap[color_id][0].tolist()

# 3 点云保存
# 3.1 保存pcd  # 点云4维的量 n*4
'''
FIELDS x y z intensity  # 定义点云中每个点包含的字段（属性）名称
SIZE 4 4 4 4            # 每个字段的字节数，这里都是 4 字节（32 位浮点数）
TYPE F F F F            # 每个字段的数据类型，这里都是浮点数（F 表示浮点数）
COUNT 1 1 1 1           # 每个字段的元素数量，这里都是 1，表示每个字段是标量
WIDTH 1000              # 点云数据的宽度，即点的数量
HEIGHT 1                # 点云数据的高度，通常是 1，表示点云是一维的
VIEWPOINT 0 0 0 1 0 0 0 # 视角信息
POINTS 1000             # 点的总数量，与 WIDTH 相同
DATA ascii / binary     # 数据的存储格式，这里是 ASCII
'''

# 1 创建一个点云

# random_arr = np.random.random((rows, columns))  # 0-1 均匀分布
# random_arr = np.random.randint(low=1, high=10, size=(rows, columns)) # 1-10的整数
# random_arr = np.random.uniform(0, 100, size=(rows, columns))  # 均匀分布浮点型
# points = np.zeros([1000, 4], dtype=np.float32)

points = np.random.uniform(-100, 100, (1000, 4))

## 3.1 保存pcd格式   open3d只能保存xyz类型的pcd格
meta_data = {
    'version': '0.7',
    'fields': ['x', 'y', 'z', 'intensity'],
    'size': [4, 4, 4, 4],
    'type': ['F', 'F', 'F', 'F'],
    'count': [1, 1, 1, 1],
    'width': points.shape[0],  # 使用点云数据的行数作为宽度
    'height': 1,
    'viewpoint': [0, 0, 0, 1, 0, 0, 0],
    'points': points.shape[0],  # 使用点云数据的行数作为点数
    'data': 'ascii'
}

pcd_save = pypcd.PointCloud(meta_data, points)
pcd_save.save_pcd("1.pcd", compression='ascii')

# open3d保存为xyz的pcd格式
points_o3d = o3d.geometry.PointCloud()
points_o3d.points = o3d.utility.Vector3dVector(points[:, :3])
o3d.io.write_point_cloud("2.pcd", points_o3d)

## 3.2 保存bin格式
with open("1.bin", 'wb') as f:
        f.write(points.tobytes())
    f.close()

## 保存bin格式  
points[:, :4].tofile("2.bin")


## 3.3 保存npy格式  
np.save("1.npy", points[:, :4])
 ~~~

# 2 python文件操作
~~~python

import os
import shutil

# 1 目录拼接
file_path = os.path.join(root, name)

# 2 目录存在和目录创建
if not os.path.exists(image_path):
        os.mkdir(image_path)

# 3 复制
shutil.copy(src_file_path, target_file_path)

# 4 剪切，移动
shutil.copy(src_file_path, target_file_path)
~~~


# 3 字符串处理

~~~python
# 1 kitti命名 补位0填充
str_name = str(i).zfill(6)   # 000000

# 2 读取txt文件
# line.strip()去除字符串首末的空格和回车，line.split('\')按'\'内容划分
# [x for x in line] 遍历line，生成一个新的list
with open(label_path) as f:
        lines = f.readlines()   # 读取所有内容
        for line in lines:      # 按行遍历
            line = [x for x in line.strip().split(' ')]

# 3 读取json文件
with open('file.json') as f:
    data = json.load(f)
~~~