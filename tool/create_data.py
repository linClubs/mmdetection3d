import argparse
import my_converter 
from mmdet3d.utils import register_all_modules

# 数据准备
def custom_data_prep(root_path,
                  info_prefix,
                  dataset_name,
                  out_dir):
    my_converter.create_my_infos(root_path, info_prefix)

# 参数配置
parser = argparse.ArgumentParser(description='Data converter arg parser')

# 数据名称
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')

# 数据路径
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/custom',
    help='specify the root path of dataset')

# 输出路径
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/custom',
    required=False,
    help='name of info pkl')

# 拓展版本 最后生成pkl文件名前缀
parser.add_argument('--extra-tag', type=str, default='custom')

parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')

# 参数加载 解析参数
args = parser.parse_args()

if __name__ == '__main__':
    
    # 注册
    register_all_modules()

    print(args.dataset)

    if args.dataset == 'custom':
        custom_data_prep(
            root_path=args.root_path,          # 数据集路径
            info_prefix=args.extra_tag,        # 生成文件名前缀
            dataset_name='customDataset',
            out_dir=args.out_dir)
        
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')