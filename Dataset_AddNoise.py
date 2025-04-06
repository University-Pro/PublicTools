"""
给数据集添加不同程度的高斯噪声
保存到指定位置
用于Synapse数据集
"""

import numpy as np
import random
import torch
import os
from scipy import ndimage
from tqdm import tqdm
from torch.utils.data import Dataset
import h5py
import nibabel as nib  # 新增nibabel库
from scipy.ndimage import zoom
import cv2

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

# 噪声函数保持不变
def add_gaussian_noise(image, std):
    """添加高斯噪声，支持2D/3D数据"""
    noise = np.random.normal(0, std, image.shape)
    return np.clip(image + noise, 0, 1)

def add_pepper_noise(image, amount):
    """添加椒盐噪声，支持2D/3D数据"""
    noisy_image = np.copy(image)
    # 计算需要添加噪声的像素总数
    num_pixels = int(amount * image.size)
    # 生成随机坐标（适应不同维度）
    coords = tuple([np.random.randint(0, dim, num_pixels) for dim in image.shape])
    noisy_image[coords] = 0
    return noisy_image

class H5NoiseGenerator:
    """HDF5噪声数据集生成器（支持批量处理）"""
    def __init__(self, output_dir, noise_config):
        self.output_dir = output_dir
        self.noise_config = noise_config
        os.makedirs(output_dir, exist_ok=True)
    
    def batch_process(self, file_list, input_base_dir):
        """批量处理H5文件"""
        processed_files = []
        for vol_name in tqdm(file_list, desc="处理H5文件"):
            orig_path = os.path.join(input_base_dir, f"{vol_name}.npy.h5")
            output_path = self.process_volume(orig_path, vol_name)
            processed_files.append(output_path)
        return processed_files
    
    def process_volume(self, original_path, vol_name):
        """处理单个H5文件（增加异常处理）"""
        try:
            with h5py.File(original_path, 'r') as f:
                image = f['image'][:]
                label = f['label'][:]
            
            # 应用噪声
            if self.noise_config['type'] == 'gaussian':
                noisy_image = add_gaussian_noise(image, self.noise_config['std'])
            elif self.noise_config['type'] == 'pepper':
                noisy_image = add_pepper_noise(image, self.noise_config['amount'])
            
            # 保存文件
            output_path = os.path.join(self.output_dir, f"{vol_name}.npy.h5")
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('image', data=noisy_image, dtype=np.float32)
                f.create_dataset('label', data=label, dtype=np.uint8)
            return output_path
        except Exception as e:
            print(f"处理文件 {vol_name} 失败: {str(e)}")
            return None

class RandomGenerator(object):
    """修改后的数据增强类"""
    def __init__(self, output_size, apply_aug=True, noise_config=None):
        """
        output_size: 目标尺寸 (height, width)
        apply_aug: 是否应用数据增强
        noise_config: 噪声配置，格式: {'type': 'gaussian', 'std': 0.1}
        """
        self.output_size = output_size
        self.apply_aug = apply_aug
        self.noise_config = noise_config or {}

    def _spatial_augmentation(self, image, label):
        """空间数据增强"""
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        return image, label

    def _resize_volume(self, image, label):
        """调整三维数据尺寸"""
        z, h, w = image.shape
        if (h, w) != self.output_size:
            # 保持通道/切片维度不变
            image = zoom(image, (1, self.output_size[0]/h, self.output_size[1]/w), order=3)
            label = zoom(label, (1, self.output_size[0]/h, self.output_size[1]/w), order=0)
        return image, label

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # 应用数据增强
        if self.apply_aug and len(image.shape) == 2:  # 仅对2D数据增强
            image, label = self._spatial_augmentation(image, label)
        
        # 处理三维数据
        if len(image.shape) == 3:
            image, label = self._resize_volume(image, label)
        else:  # 处理二维数据
            h, w = image.shape
            if (h, w) != self.output_size:
                image = zoom(image, (self.output_size[0]/h, self.output_size[1]/w), order=3)
                label = zoom(label, (self.output_size[0]/h, self.output_size[1]/w), order=0)

        # 添加噪声
        if self.noise_config:
            if self.noise_config.get('type') == 'gaussian':
                image = add_gaussian_noise(image, self.noise_config['std'])
            elif self.noise_config.get('type') == 'pepper':
                image = add_pepper_noise(image, self.noise_config['amount'])

        # 转换为Tensor
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # 添加通道维度
        label = torch.from_numpy(label.astype(np.float32))
        return {'image': image, 'label': label.long()}

class SynapseDataset(Dataset):
    """增强后的数据集类（支持批量转换）"""
    def __init__(self, base_dir, list_dir, split, transform=None, 
                 save_h5=False, h5_noise_config=None, 
                 preprocess_all=False):  # 新增预处理参数
        self.base_dir = base_dir
        self.list_dir = list_dir
        self.split = split
        self.transform = transform
        self.sample_list = self._load_sample_list()
        
        # H5处理配置
        self.save_h5 = save_h5
        if self.save_h5:
            self.h5_generator = H5NoiseGenerator(
                output_dir=os.path.join(base_dir, "test_noise_gaussian_20"),
                noise_config=h5_noise_config or {}
            )
            # 新增批量预处理
            if preprocess_all and split == "test":
                self._preprocess_all()

    def _preprocess_all(self):
        """预处理所有H5文件"""
        print(f"开始批量预处理{len(self.sample_list)}个文件...")
        input_dir = os.path.join(self.base_dir, "test")
        self.h5_generator.batch_process(self.sample_list, input_dir)

    def _load_sample_list(self):
        with open(os.path.join(self.list_dir, f"{self.split}.txt")) as f:
            return [line.strip() for line in f.readlines()]

    def _process_h5(self, idx):
        """处理H5文件并保存"""
        vol_name = self.sample_list[idx]
        orig_path = os.path.join(self.base_dir, "test", f"{vol_name}.npy.h5")
        return self.h5_generator.process_volume(orig_path, vol_name)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # 预处理H5文件（如果需要）
        if self.save_h5 and self.split == "test":
            self._process_h5(idx)

        # 常规数据加载
        if self.split == "train":
            slice_name = self.sample_list[idx]
            data_path = os.path.join(self.base_dir, "train", f"{slice_name}.npz")
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx]
            h5_path = os.path.join(self.base_dir, "test", f"{vol_name}.npy.h5")
            with h5py.File(h5_path, 'r') as f:
                image, label = f['image'][:], f['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx]
        return sample

# 测试用例
if __name__ == "__main__":
    # 配置参数
    h5_noise_config = {
        'type': 'gaussian',
        'std': 0.2
    }
    
    # 初始化数据集并预处理所有文件
    db_test = SynapseDataset(
        base_dir="./datasets/Synapse/data",
        list_dir="./datasets/Synapse/list",
        split="test",
        transform=RandomGenerator((224, 224)),
        save_h5=True,
        h5_noise_config=h5_noise_config,
        preprocess_all=True  # 启用批量预处理
    )
    
    # 验证处理结果
    print(f"处理完成，共转换{len(db_test)}个文件")
