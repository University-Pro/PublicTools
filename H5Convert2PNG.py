import os
import argparse
import h5py
import numpy as np
import cv2
from tqdm import tqdm

def h5_to_images(h5_path, output_dir, dataset_key='image', 
                 format='png', start_slice=0, end_slice=None,
                 normalize=True, prefix='', suffix=''):
    """
    将HDF5文件中的3D图像数据转换为2D图像序列
    参数：
        h5_path: 输入的HDF5文件路径
        output_dir: 输出目录
        dataset_key: HDF5中数据集的键名（默认：'image'）
        format: 输出格式，支持png/jpg（默认：png）
        start_slice: 起始切片编号（默认：0）
        end_slice: 结束切片编号（默认：None表示全部）
        normalize: 是否自动归一化到0-255（默认：True）
        prefix: 输出文件名前缀（默认：''）
        suffix: 输出文件名后缀（默认：''）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # 验证数据集存在
            if dataset_key not in f:
                available_keys = list(f.keys())
                raise ValueError(f"数据集键 '{dataset_key}' 不存在，可用键：{available_keys}")

            data = f[dataset_key][:]
            print(f"原始数据形状：{data.shape} | 数据类型：{data.dtype} | 数值范围：[{data.min():.2f}, {data.max():.2f}]")

            # 处理不同维度数据
            if data.ndim == 2:
                data = data[np.newaxis, ...]  # 统一为3D处理
            elif data.ndim == 3:
                pass  # 已经是3D数据
            else:
                raise ValueError(f"不支持{data.ndim}维数据，仅支持2D/3D")

            # 自动确定切片范围
            total_slices = data.shape[0]
            end_slice = end_slice or total_slices
            if end_slice > total_slices:
                print(f"警告：结束切片超过数据范围，自动调整为{total_slices}")
                end_slice = total_slices

            # 处理归一化
            if normalize:
                if data.dtype != np.uint8:
                    data_min = data.min()
                    data_max = data.max()
                    if data_max - data_min > 1e-6:  # 避免除以零
                        data = (data - data_min) / (data_max - data_min) * 255
                    else:
                        data = data * 255
                    data = data.astype(np.uint8)
                print(f"归一化后范围：[{data.min()}, {data.max()}]")
            else:
                if data.dtype != np.uint8:
                    print("警告：未启用归一化，但数据类型不是uint8，可能影响输出质量")

            # 设置压缩参数（仅对jpg有效）
            encode_params = []
            if format.lower() == 'jpg':
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]  # 90%质量

            # 主转换循环
            for slice_idx in tqdm(range(start_slice, end_slice), 
                                desc=f"转换 {os.path.basename(h5_path)}"):
                slice_data = data[slice_idx]
                
                # 处理二维数据
                if slice_data.ndim == 3 and slice_data.shape[-1] in [3,4]:  # RGB/RGBA
                    pass  # 保持颜色通道
                elif slice_data.ndim == 2:  # 灰度
                    slice_data = cv2.cvtColor(slice_data, cv2.COLOR_GRAY2BGR)
                else:
                    raise ValueError(f"非常规图像维度：{slice_data.shape}")

                # 生成文件名
                filename = f"{prefix}slice_{slice_idx:04d}{suffix}.{format}"
                output_path = os.path.join(output_dir, filename)
                
                # 保存图像
                if not cv2.imwrite(output_path, slice_data, encode_params):
                    raise RuntimeError(f"无法写入文件：{output_path}")

        print(f"转换完成！保存至：{output_dir}")

    except Exception as e:
        print(f"转换失败：{str(e)}")
        raise

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='将HDF5文件转换为图像序列')
    parser.add_argument('-i', '--input', required=True, 
                       help='输入HDF5文件路径')
    parser.add_argument('-o', '--output', required=True,
                       help='输出目录路径')
    parser.add_argument('--dataset_key', default='image',
                       help='HDF5中的数据集键名（默认：image）')
    parser.add_argument('--format', choices=['png', 'jpg'], default='png',
                       help='输出图像格式（默认：png）')
    parser.add_argument('--start', type=int, default=0,
                       help='起始切片编号（默认：0）')
    parser.add_argument('--end', type=int, default=None,
                       help='结束切片编号（默认：全部）')
    parser.add_argument('--no-normalize', action='store_false', dest='normalize',
                       help='禁用自动归一化到0-255范围')
    parser.add_argument('--prefix', default='',
                       help='输出文件名前缀（默认：空）')
    parser.add_argument('--suffix', default='',
                       help='输出文件名后缀（默认：空）')

    args = parser.parse_args()

    # 执行转换
    h5_to_images(
        h5_path=args.input,
        output_dir=args.output,
        dataset_key=args.dataset_key,
        format=args.format,
        start_slice=args.start,
        end_slice=args.end,
        normalize=args.normalize,
        prefix=args.prefix,
        suffix=args.suffix
    )