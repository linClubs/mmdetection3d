import numpy as np
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':')
            except ValueError:
                # key, value = line.split(' ')
                pass
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def get_rgb_by_distance(p, min_val=0, max_val=50):
    z = np.sqrt(np.square(p))
    if z > max_val:
        z = 50
    thre = (z - min_val) / (max_val - min_val) * 255
    thre = int(thre)
    return thre

# cv::COLORMAP_JET
def get_color_jet(pixel, max, min):
    # print(pixel, max, min)
    colormap_id = (pixel - min) / (max - min) * 255
    return colormap_id
    
    

class Object3d:
    def __init__(self, content):
        super(Object3d, self).__init__()
        # content 就是一个字符串，根据空格分隔开来
        lines = content.split()

        # 去掉空字符
        lines = list(filter(lambda x: len(x), lines))

        self.name, self.truncated, self.occluded, self.alpha = lines[0], float(lines[1]), float(lines[2]), float(lines[3])

        self.bbox = [lines[4], lines[5], lines[6], lines[7]]
        self.bbox = np.array([float(x) for x in self.bbox])
        self.dimensions = [lines[8], lines[9], lines[10]]
        self.dimensions = np.array([float(x) for x in self.dimensions])
        self.location = [lines[11], lines[12], lines[13]]
        self.location = np.array([float(x) for x in self.location])
        self.rotation_y = float(lines[14])
        # 这一行是模型训练后的label通常最后一行是阈值，可以同个这个过滤掉概率低的object
        # 如果只要显示kitti本身则不需要这一行
        # self.ioc = float(lines[15])

# 根据偏航角计算旋转矩阵（逆时针旋转）
def rot_y(rotation_y):
    cos = np.cos(rotation_y)
    sin = np.sin(rotation_y)
    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return R