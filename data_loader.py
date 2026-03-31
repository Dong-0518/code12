"""
数据加载和预处理模块 (双流架构升级版)
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import glob

class PlantDataset(Dataset):
    """植物图像数据集（用于特征提取和验证测试）"""
    
    def __init__(self, image_paths, labels, transform_g=None, transform_l=None):
        """
        Args:
            image_paths: 图像路径列表
            labels: 标签列表
            transform_g: 全局流图像变换 (Global)
            transform_l: 局部流图像变换 (Local)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform_g = transform_g
        self.transform_l = transform_l
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 同时应用全局和局部变换
            image_g = self.transform_g(image) if self.transform_g else image
            image_l = self.transform_l(image) if self.transform_l else image
            
            # 返回4个元素：(全局图, 局部图, 标签, 路径)
            return image_g, image_l, label, image_path
            
        except Exception as e:
            print(f"加载图像失败: {image_path}, 错误: {e}")
            # 返回安全的黑色图像作为替补
            image = Image.new('RGB', (1024, 1024), (0, 0, 0))
            image_g = self.transform_g(image) if self.transform_g else image
            image_l = self.transform_l(image) if self.transform_l else image
            return image_g, image_l, label, image_path

class TripletDataset(Dataset):
    """用于Triplet Loss的数据集（双流架构专用）"""
    
    def __init__(self, image_paths, labels, transform_g=None, transform_l=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform_g = transform_g
        self.transform_l = transform_l
        
        # 构建标签到索引的映射
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]
        
        # 选择positive（同一物种）
        positive_indices = self.label_to_indices[anchor_label]
        positive_idx = np.random.choice(positive_indices)
        while positive_idx == idx and len(positive_indices) > 1:  # 确保不是同一个图像
            positive_idx = np.random.choice(positive_indices)
        
        # 选择negative（不同物种）
        negative_label = np.random.choice(
            [l for l in self.label_to_indices.keys() if l != anchor_label]
        )
        negative_idx = np.random.choice(self.label_to_indices[negative_label])
        
        # 加载三张原始图像
        anchor_img = self._load_image(anchor_path)
        pos_img = self._load_image(self.image_paths[positive_idx])
        neg_img = self._load_image(self.image_paths[negative_idx])
        
        # 将这三张图，全部分裂成 Global(全局) 和 Local(局部) 两个版本
        anchor_g = self.transform_g(anchor_img) if self.transform_g else anchor_img
        anchor_l = self.transform_l(anchor_img) if self.transform_l else anchor_img
        
        pos_g = self.transform_g(pos_img) if self.transform_g else pos_img
        pos_l = self.transform_l(pos_img) if self.transform_l else pos_img
        
        neg_g = self.transform_g(neg_img) if self.transform_g else neg_img
        neg_l = self.transform_l(neg_img) if self.transform_l else neg_img
        
        # 核心：一共返回 7 个元素
        return anchor_g, anchor_l, pos_g, pos_l, neg_g, neg_l, anchor_label
    
    def _load_image(self, path):
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败: {path}, 错误: {e}")
            return Image.new('RGB', (1024, 1024), (0, 0, 0))

def get_dual_transforms(mode='train', image_size_g=224, image_size_l=384):
    """
    双流专用预处理：同时返回全局和局部的 Transform
    
    Args:
        mode: 'train' 或 'test'
        image_size_g: 全局图像尺寸 (默认 224，喂给 ViT)
        image_size_l: 局部高清裁剪尺寸 (默认 384，喂给 CNN)
    """
    # 1. 全局流 (Global)：直接将原图压缩到 224x224，提取宏观轮廓
    transform_g = transforms.Compose([
        transforms.Resize((image_size_g, image_size_g)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. 局部流 (Local)：先温和缩小，再高清截取 384x384，用于捕捉微观星毛
    if mode == 'train':
        transform_l = transforms.Compose([
            transforms.Resize(1024),  # 保留极高分辨率
            transforms.RandomCrop(image_size_l), # 随机像放大镜一样切一块
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # 测试提取时，直接在高清图中心截取
        transform_l = transforms.Compose([
            transforms.Resize(1024),
            transforms.CenterCrop(image_size_l),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    return transform_g, transform_l

def load_dataset(data_path, image_type='specimen'):
    """加载数据集 (不变)"""
    image_paths = []
    labels = []
    species_names = []
    
    # 获取所有物种文件夹
    if os.path.isdir(data_path):
        species_dirs = [d for d in os.listdir(data_path) 
                       if os.path.isdir(os.path.join(data_path, d))]
    else:
        print(f"数据路径不存在: {data_path}")
        return [], [], []
    
    # 支持的图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
    
    # 遍历每个物种文件夹
    for species_idx, species_dir in enumerate(sorted(species_dirs)):
        species_path = os.path.join(data_path, species_dir)
        
        # 获取该物种的所有图像
        species_images = []
        for ext in image_extensions:
            species_images.extend(glob.glob(os.path.join(species_path, ext)))
        
        if len(species_images) == 0:
            print(f"警告: {species_dir} 文件夹中没有找到图像")
            continue
        
        # 添加到列表
        image_paths.extend(species_images)
        labels.extend([species_idx] * len(species_images))
        species_names.append(species_dir)
        
        print(f"加载物种 {species_dir}: {len(species_images)} 张图像")
    
    print(f"\n总共加载 {len(image_paths)} 张图像，{len(species_names)} 个物种")
    return image_paths, labels, species_names

def create_dataloaders(image_paths, labels, batch_size=32, train_ratio=0.7, 
                      val_ratio=0.2, test_ratio=0.1, use_triplet=False, 
                      image_size_g=224, image_size_l=384, num_workers=4, model_type='hybrid'):
    """
    创建数据加载器（升级版，支持双尺寸传入）
    """
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=(1 - train_ratio), 
        stratify=labels, random_state=42
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size), 
        stratify=y_temp, random_state=42
    )
    
    print(f"训练集: {len(X_train)} 张图像")
    print(f"验证集: {len(X_val)} 张图像")
    print(f"测试集: {len(X_test)} 张图像")
    
    # 获取双流专属 Transforms
    trans_train_g, trans_train_l = get_dual_transforms('train', image_size_g, image_size_l)
    trans_test_g, trans_test_l = get_dual_transforms('test', image_size_g, image_size_l)
    
    # 创建数据集
    if use_triplet:
        train_dataset = TripletDataset(X_train, y_train, transform_g=trans_train_g, transform_l=trans_train_l)
        val_dataset = TripletDataset(X_val, y_val, transform_g=trans_test_g, transform_l=trans_test_l)
        # 测试集始终使用 PlantDataset
        test_dataset = PlantDataset(X_test, y_test, transform_g=trans_test_g, transform_l=trans_test_l)
    else:
        train_dataset = PlantDataset(X_train, y_train, transform_g=trans_train_g, transform_l=trans_train_l)
        val_dataset = PlantDataset(X_val, y_val, transform_g=trans_test_g, transform_l=trans_test_l)
        test_dataset = PlantDataset(X_test, y_test, transform_g=trans_test_g, transform_l=trans_test_l)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader