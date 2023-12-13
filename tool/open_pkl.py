import pickle

# 读取 pickle 文件
# pkl_path = 'data/custom3/custom_infos_train.pkl'
pkl_path = "/home/lin/code/mmdetection3d/data/kitti/kitti_infos_train.pkl"
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# 打印数据
print(data.keys())
print(data['metainfo'])
print(len(data['data_list']))

for sample in data['data_list']:
    print(sample.keys())
    print(sample['lidar_points'].keys())
    print(sample['images'].keys())
    print(sample['instances'])
    break
