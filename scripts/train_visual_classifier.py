#!/usr/bin/env python3
"""
Script to train a ResNet18 classifier on the Plaintextdataset using visual_anchor.jpg images.
Structure: Plaintextdataset/<ClassName>/<EpisodeID>/visual_anchor.jpg
"""

import os
import glob
import random
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 物理属性定义 (Physical Properties Ground Truth)
# =============================================================================
PHYSICAL_PROPERTIES = {
    # Item Name: (mass, stiffness, material)
    'WoodBlock_Native':         {'mass': 'medium',   'stiffness': 'rigid',     'material': 'wood'},
    'WoodBlock_Foil':           {'mass': 'medium',   'stiffness': 'rigid',     'material': 'wood'},
    'WoodBlock_Red':            {'mass': 'medium',   'stiffness': 'rigid',     'material': 'wood'},
    'YogaBrick_Native':         {'mass': 'low',      'stiffness': 'medium',    'material': 'foam'},
    'YogaBrick_Blue':           {'mass': 'low',      'stiffness': 'medium',    'material': 'foam'},
    'CardboardBox_Hollow':      {'mass': 'very_low', 'stiffness': 'soft',      'material': 'hollow_container'},
    'CardboardBox_SpongeFilled':{'mass': 'very_low', 'stiffness': 'soft',      'material': 'filled_container'},
    'CardboardBox_RockFilled':  {'mass': 'high',     'stiffness': 'rigid',     'material': 'filled_container'},
    'CardboardBox_RockFilled_Red':{'mass': 'high',   'stiffness': 'rigid',     'material': 'filled_container'},
    'Sponge_Blue':              {'mass': 'very_low', 'stiffness': 'very_soft', 'material': 'sponge'},
}

# 将属性值映射为数值标签（用于多任务学习）
MASS_TO_IDX = {'very_low': 0, 'low': 1, 'medium': 2, 'high': 3}
STIFFNESS_TO_IDX = {'very_soft': 0, 'soft': 1, 'medium': 2, 'rigid': 3}
MATERIAL_TO_IDX = {'sponge': 0, 'foam': 1, 'wood': 2, 'hollow_container': 3, 'filled_container': 4}

# 反向映射（用于推理时显示）
IDX_TO_MASS = {v: k for k, v in MASS_TO_IDX.items()}
IDX_TO_STIFFNESS = {v: k for k, v in STIFFNESS_TO_IDX.items()}
IDX_TO_MATERIAL = {v: k for k, v in MATERIAL_TO_IDX.items()}


# =============================================================================
# 多任务模型定义 (Multi-Task Model)
# =============================================================================
class MultiTaskResNet(nn.Module):
    """多任务 ResNet-18：同时预测类别 + 质量 + 硬度 + 材料"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        
        # 加载预训练 ResNet-18 作为主干
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # 获取特征维度
        num_features = self.backbone.fc.in_features  # 512
        
        # 移除原始分类头
        self.backbone.fc = nn.Identity()
        
        # 多任务分类头
        self.head_class = nn.Linear(num_features, num_classes)      # 10 类
        self.head_mass = nn.Linear(num_features, len(MASS_TO_IDX))  # 4 类: very_low, low, medium, high
        self.head_stiffness = nn.Linear(num_features, len(STIFFNESS_TO_IDX))  # 4 类
        self.head_material = nn.Linear(num_features, len(MATERIAL_TO_IDX))    # 5 类
        
    def forward(self, x):
        # 提取共享特征
        features = self.backbone(x)  # [batch, 512]
        
        # 多任务输出
        out_class = self.head_class(features)
        out_mass = self.head_mass(features)
        out_stiffness = self.head_stiffness(features)
        out_material = self.head_material(features)
        
        return {
            'class': out_class,
            'mass': out_mass,
            'stiffness': out_stiffness,
            'material': out_material
        }


def print_property_summary():
    """打印所有物品的物理属性表"""
    logger.info("\n" + "="*70)
    logger.info("Physical Properties Ground Truth:")
    logger.info("-"*70)
    logger.info(f"{'Item Name':<30} {'Mass':<10} {'Stiffness':<12} {'Material'}")
    logger.info("-"*70)
    for item, props in PHYSICAL_PROPERTIES.items():
        logger.info(f"{item:<30} {props['mass']:<10} {props['stiffness']:<12} {props['material']}")
    logger.info("="*70 + "\n")


class LivePlotter:
    """实时绘制训练曲线的类（单任务模式）"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        # 启用交互模式
        plt.ion()
        
        # 创建图表
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 5))
        self.fig.suptitle('Training Progress', fontsize=14)
        
        # 初始化线条
        self.line_train_loss, = self.ax1.plot([], [], 'b-o', label='Train Loss', markersize=4)
        self.line_val_loss, = self.ax1.plot([], [], 'r-o', label='Val Loss', markersize=4)
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)
        
        self.line_train_acc, = self.ax2.plot([], [], 'b-o', label='Train Acc', markersize=4)
        self.line_val_acc, = self.ax2.plot([], [], 'r-o', label='Val Acc', markersize=4)
        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend(loc='lower right')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show(block=False)
    
    def update(self, train_loss: float, val_loss: float, train_acc: float, val_acc: float):
        """更新图表数据"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        
        epochs = list(range(1, len(self.history['train_loss']) + 1))
        
        # 更新 Loss 图
        self.line_train_loss.set_data(epochs, self.history['train_loss'])
        self.line_val_loss.set_data(epochs, self.history['val_loss'])
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # 更新 Accuracy 图
        self.line_train_acc.set_data(epochs, self.history['train_acc'])
        self.line_val_acc.set_data(epochs, self.history['val_acc'])
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax2.set_ylim(0, 1.05)
        
        # 刷新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)
    
    def save(self):
        """保存最终图表"""
        if self.output_dir:
            self.fig.savefig(self.output_dir / 'training_metrics.png', dpi=150)
            logger.info(f"Saved training metrics plot to {self.output_dir / 'training_metrics.png'}")
    
    def close(self):
        """关闭图表"""
        plt.ioff()
        plt.close(self.fig)


class MultiTaskLivePlotter:
    """实时绘制多任务训练曲线的类"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc_class': [], 'val_acc_class': [],
            'train_acc_mass': [], 'val_acc_mass': [],
            'train_acc_stiffness': [], 'val_acc_stiffness': [],
            'train_acc_material': [], 'val_acc_material': [],
        }
        
        plt.ion()
        
        # 创建 2x3 的图表布局
        self.fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Multi-Task Training Progress', fontsize=14)
        
        # Loss 图
        self.ax_loss = axes[0, 0]
        self.line_train_loss, = self.ax_loss.plot([], [], 'b-o', label='Train', markersize=4)
        self.line_val_loss, = self.ax_loss.plot([], [], 'r-o', label='Val', markersize=4)
        self.ax_loss.set_title('Total Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.legend(loc='upper right')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Class Accuracy 图
        self.ax_class = axes[0, 1]
        self.line_train_class, = self.ax_class.plot([], [], 'b-o', label='Train', markersize=4)
        self.line_val_class, = self.ax_class.plot([], [], 'r-o', label='Val', markersize=4)
        self.ax_class.set_title('Class Accuracy')
        self.ax_class.set_xlabel('Epoch')
        self.ax_class.set_ylim(0, 1.05)
        self.ax_class.legend(loc='lower right')
        self.ax_class.grid(True, alpha=0.3)
        
        # Mass Accuracy 图
        self.ax_mass = axes[0, 2]
        self.line_train_mass, = self.ax_mass.plot([], [], 'b-o', label='Train', markersize=4)
        self.line_val_mass, = self.ax_mass.plot([], [], 'r-o', label='Val', markersize=4)
        self.ax_mass.set_title('Mass Accuracy')
        self.ax_mass.set_xlabel('Epoch')
        self.ax_mass.set_ylim(0, 1.05)
        self.ax_mass.legend(loc='lower right')
        self.ax_mass.grid(True, alpha=0.3)
        
        # Stiffness Accuracy 图
        self.ax_stiffness = axes[1, 0]
        self.line_train_stiffness, = self.ax_stiffness.plot([], [], 'b-o', label='Train', markersize=4)
        self.line_val_stiffness, = self.ax_stiffness.plot([], [], 'r-o', label='Val', markersize=4)
        self.ax_stiffness.set_title('Stiffness Accuracy')
        self.ax_stiffness.set_xlabel('Epoch')
        self.ax_stiffness.set_ylim(0, 1.05)
        self.ax_stiffness.legend(loc='lower right')
        self.ax_stiffness.grid(True, alpha=0.3)
        
        # Material Accuracy 图
        self.ax_material = axes[1, 1]
        self.line_train_material, = self.ax_material.plot([], [], 'b-o', label='Train', markersize=4)
        self.line_val_material, = self.ax_material.plot([], [], 'r-o', label='Val', markersize=4)
        self.ax_material.set_title('Material Accuracy')
        self.ax_material.set_xlabel('Epoch')
        self.ax_material.set_ylim(0, 1.05)
        self.ax_material.legend(loc='lower right')
        self.ax_material.grid(True, alpha=0.3)
        
        # 综合对比图（所有 Val Accuracy）
        self.ax_compare = axes[1, 2]
        self.line_cmp_class, = self.ax_compare.plot([], [], 'o-', label='Class', markersize=4, color='#1f77b4')
        self.line_cmp_mass, = self.ax_compare.plot([], [], 's-', label='Mass', markersize=4, color='#ff7f0e')
        self.line_cmp_stiffness, = self.ax_compare.plot([], [], '^-', label='Stiffness', markersize=4, color='#2ca02c')
        self.line_cmp_material, = self.ax_compare.plot([], [], 'd-', label='Material', markersize=4, color='#d62728')
        self.ax_compare.set_title('Val Accuracy Comparison')
        self.ax_compare.set_xlabel('Epoch')
        self.ax_compare.set_ylim(0, 1.05)
        self.ax_compare.legend(loc='lower right')
        self.ax_compare.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show(block=False)
    
    def update(self, train_loss, val_loss, 
               train_acc_class, val_acc_class,
               train_acc_mass, val_acc_mass,
               train_acc_stiffness, val_acc_stiffness,
               train_acc_material, val_acc_material):
        """更新所有图表"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc_class'].append(train_acc_class)
        self.history['val_acc_class'].append(val_acc_class)
        self.history['train_acc_mass'].append(train_acc_mass)
        self.history['val_acc_mass'].append(val_acc_mass)
        self.history['train_acc_stiffness'].append(train_acc_stiffness)
        self.history['val_acc_stiffness'].append(val_acc_stiffness)
        self.history['train_acc_material'].append(train_acc_material)
        self.history['val_acc_material'].append(val_acc_material)
        
        epochs = list(range(1, len(self.history['train_loss']) + 1))
        
        # 更新 Loss
        self.line_train_loss.set_data(epochs, self.history['train_loss'])
        self.line_val_loss.set_data(epochs, self.history['val_loss'])
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        
        # 更新 Class Acc
        self.line_train_class.set_data(epochs, self.history['train_acc_class'])
        self.line_val_class.set_data(epochs, self.history['val_acc_class'])
        self.ax_class.relim()
        self.ax_class.autoscale_view()
        self.ax_class.set_ylim(0, 1.05)
        
        # 更新 Mass Acc
        self.line_train_mass.set_data(epochs, self.history['train_acc_mass'])
        self.line_val_mass.set_data(epochs, self.history['val_acc_mass'])
        self.ax_mass.relim()
        self.ax_mass.autoscale_view()
        self.ax_mass.set_ylim(0, 1.05)
        
        # 更新 Stiffness Acc
        self.line_train_stiffness.set_data(epochs, self.history['train_acc_stiffness'])
        self.line_val_stiffness.set_data(epochs, self.history['val_acc_stiffness'])
        self.ax_stiffness.relim()
        self.ax_stiffness.autoscale_view()
        self.ax_stiffness.set_ylim(0, 1.05)
        
        # 更新 Material Acc
        self.line_train_material.set_data(epochs, self.history['train_acc_material'])
        self.line_val_material.set_data(epochs, self.history['val_acc_material'])
        self.ax_material.relim()
        self.ax_material.autoscale_view()
        self.ax_material.set_ylim(0, 1.05)
        
        # 更新对比图
        self.line_cmp_class.set_data(epochs, self.history['val_acc_class'])
        self.line_cmp_mass.set_data(epochs, self.history['val_acc_mass'])
        self.line_cmp_stiffness.set_data(epochs, self.history['val_acc_stiffness'])
        self.line_cmp_material.set_data(epochs, self.history['val_acc_material'])
        self.ax_compare.relim()
        self.ax_compare.autoscale_view()
        self.ax_compare.set_ylim(0, 1.05)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)
    
    def save(self):
        """保存图表"""
        if self.output_dir:
            self.fig.savefig(self.output_dir / 'multitask_training_metrics.png', dpi=150)
            logger.info(f"Saved multi-task training metrics to {self.output_dir / 'multitask_training_metrics.png'}")
    
    def close(self):
        plt.ioff()
        plt.close(self.fig)

class PlaintextImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, return_properties: bool = False):
        """
        Args:
            root_dir (str): Directory with all the class folders (Plaintextdataset).
            transform (callable, optional): Optional transform to be applied on a sample.
            return_properties (bool): If True, also return physical properties (mass, stiffness, material).
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_properties = return_properties
        self.samples: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        self._find_classes_and_samples()

    def _find_classes_and_samples(self):
        # List all subdirectories in root_dir as classes
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")

        # Get class names (sorted for consistency)
        self.classes = sorted([
            d.name for d in self.root_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        if not self.classes:
            raise ValueError(f"No class directories found in {self.root_dir}")

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        logger.info(f"Found {len(self.classes)} classes: {self.classes}")

        # Collect all visual_anchor.jpg files
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            cls_idx = self.class_to_idx[cls_name]
            
            # Assuming structure: root/class/episode/visual_anchor.jpg
            # Find all episode directories
            episode_dirs = [d for d in cls_dir.iterdir() if d.is_dir()]
            
            for ep_dir in episode_dirs:
                img_path = ep_dir / "visual_anchor.jpg"
                if img_path.exists():
                    self.samples.append((img_path, cls_idx))
        
        logger.info(f"Found total {len(self.samples)} images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load image and convert to RGB (in case of RGBA or grayscale)
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            if self.return_properties:
                # 获取类名并查找物理属性
                class_name = self.classes[label]
                props = PHYSICAL_PROPERTIES.get(class_name, {})
                
                # 转换为数值标签
                mass_idx = MASS_TO_IDX.get(props.get('mass', 'medium'), 2)
                stiffness_idx = STIFFNESS_TO_IDX.get(props.get('stiffness', 'medium'), 2)
                material_idx = MATERIAL_TO_IDX.get(props.get('material', 'wood'), 2)
                
                return image, label, mass_idx, stiffness_idx, material_idx
            
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise e

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu', save_path=None, output_dir=None):
    best_acc = 0.0
    
    # 创建实时绘图器
    plotter = LivePlotter(output_dir=output_dir)
    
    # 临时存储每个 epoch 的 train/val 指标
    epoch_metrics = {}
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            # Iterate over data
            pbar = tqdm(dataloader, desc=f"{phase} Phase")
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics - 累加每个 batch
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
                pbar.set_postfix({'loss': loss.item()})

            epoch_loss = running_loss / total_samples
            epoch_acc = (running_corrects.double() / total_samples).item()

            logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 保存当前 phase 的指标
            epoch_metrics[f'{phase}_loss'] = epoch_loss
            epoch_metrics[f'{phase}_acc'] = epoch_acc

            # Deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    if save_path:
                        torch.save(model.state_dict(), save_path)
                        logger.info(f"Saved new best model with Acc: {best_acc:.4f} to {save_path}")
                
                # 更新实时图表
                plotter.update(
                    train_loss=epoch_metrics['train_loss'],
                    val_loss=epoch_metrics['val_loss'],
                    train_acc=epoch_metrics['train_acc'],
                    val_acc=epoch_metrics['val_acc']
                )
                
                # Detailed evaluation metrics (only on last epoch to save time)
                if epoch == num_epochs - 1:
                    try:
                        from sklearn.metrics import classification_report
                        all_preds = []
                        all_labels = []
                        with torch.no_grad():
                            for inputs, labels in dataloader:
                                inputs = inputs.to(device)
                                outputs = model(inputs)
                                _, preds = torch.max(outputs, 1)
                                all_preds.extend(preds.cpu().numpy())
                                all_labels.extend(labels.cpu().numpy())
                        
                        report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
                        logger.info(f"\nClassification Report ({phase}):\n{report}")
                    except ImportError:
                        logger.warning("scikit-learn not installed, skipping detailed report.")

    logger.info(f'Best val Acc: {best_acc:.4f}')
    
    # 保存并关闭图表
    plotter.save()
    plotter.close()
    
    return model


def train_multitask_model(model, train_loader, val_loader, criterion, optimizer, 
                          num_epochs=10, device='cpu', save_path=None, output_dir=None):
    """多任务训练循环"""
    best_acc = 0.0
    
    # 创建多任务实时绘图器
    plotter = MultiTaskLivePlotter(output_dir=output_dir)
    epoch_metrics = {}
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_loss_class = 0.0
            running_loss_mass = 0.0
            running_loss_stiffness = 0.0
            running_loss_material = 0.0
            
            running_corrects_class = 0
            running_corrects_mass = 0
            running_corrects_stiffness = 0
            running_corrects_material = 0
            total_samples = 0
            
            pbar = tqdm(dataloader, desc=f"{phase} Phase")
            for batch in pbar:
                inputs, labels_class, labels_mass, labels_stiffness, labels_material = batch
                inputs = inputs.to(device)
                labels_class = labels_class.to(device)
                labels_mass = labels_mass.to(device)
                labels_stiffness = labels_stiffness.to(device)
                labels_material = labels_material.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    # 计算各任务损失
                    loss_class = criterion(outputs['class'], labels_class)
                    loss_mass = criterion(outputs['mass'], labels_mass)
                    loss_stiffness = criterion(outputs['stiffness'], labels_stiffness)
                    loss_material = criterion(outputs['material'], labels_material)
                    
                    # 总损失（可以加权，这里先用等权重）
                    loss = loss_class + 0.5 * (loss_mass + loss_stiffness + loss_material)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_loss_class += loss_class.item() * batch_size
                running_loss_mass += loss_mass.item() * batch_size
                running_loss_stiffness += loss_stiffness.item() * batch_size
                running_loss_material += loss_material.item() * batch_size
                
                _, preds_class = torch.max(outputs['class'], 1)
                _, preds_mass = torch.max(outputs['mass'], 1)
                _, preds_stiffness = torch.max(outputs['stiffness'], 1)
                _, preds_material = torch.max(outputs['material'], 1)
                
                running_corrects_class += torch.sum(preds_class == labels_class.data)
                running_corrects_mass += torch.sum(preds_mass == labels_mass.data)
                running_corrects_stiffness += torch.sum(preds_stiffness == labels_stiffness.data)
                running_corrects_material += torch.sum(preds_material == labels_material.data)
                total_samples += batch_size
                
                pbar.set_postfix({'loss': loss.item()})

            epoch_loss = running_loss / total_samples
            epoch_acc_class = (running_corrects_class.double() / total_samples).item()
            epoch_acc_mass = (running_corrects_mass.double() / total_samples).item()
            epoch_acc_stiffness = (running_corrects_stiffness.double() / total_samples).item()
            epoch_acc_material = (running_corrects_material.double() / total_samples).item()

            logger.info(f'{phase} Loss: {epoch_loss:.4f}')
            logger.info(f'  Class Acc: {epoch_acc_class:.4f} | Mass Acc: {epoch_acc_mass:.4f} | '
                       f'Stiffness Acc: {epoch_acc_stiffness:.4f} | Material Acc: {epoch_acc_material:.4f}')
            
            epoch_metrics[f'{phase}_loss'] = epoch_loss
            epoch_metrics[f'{phase}_acc_class'] = epoch_acc_class
            epoch_metrics[f'{phase}_acc_mass'] = epoch_acc_mass
            epoch_metrics[f'{phase}_acc_stiffness'] = epoch_acc_stiffness
            epoch_metrics[f'{phase}_acc_material'] = epoch_acc_material

            if phase == 'val':
                if epoch_acc_class > best_acc:
                    best_acc = epoch_acc_class
                    if save_path:
                        torch.save(model.state_dict(), save_path)
                        logger.info(f"Saved new best model with Class Acc: {best_acc:.4f}")
                
                plotter.update(
                    train_loss=epoch_metrics['train_loss'],
                    val_loss=epoch_metrics['val_loss'],
                    train_acc_class=epoch_metrics['train_acc_class'],
                    val_acc_class=epoch_metrics['val_acc_class'],
                    train_acc_mass=epoch_metrics['train_acc_mass'],
                    val_acc_mass=epoch_metrics['val_acc_mass'],
                    train_acc_stiffness=epoch_metrics['train_acc_stiffness'],
                    val_acc_stiffness=epoch_metrics['val_acc_stiffness'],
                    train_acc_material=epoch_metrics['train_acc_material'],
                    val_acc_material=epoch_metrics['val_acc_material'],
                )
                
                # 最后一个 epoch 输出详细分类报告
                if epoch == num_epochs - 1:
                    logger.info("\n" + "="*60)
                    logger.info("Final Multi-Task Performance Summary:")
                    logger.info(f"  Class Accuracy:     {epoch_acc_class:.4f}")
                    logger.info(f"  Mass Accuracy:      {epoch_acc_mass:.4f}")
                    logger.info(f"  Stiffness Accuracy: {epoch_acc_stiffness:.4f}")
                    logger.info(f"  Material Accuracy:  {epoch_acc_material:.4f}")
                    logger.info("="*60)
                    
                    # 生成各任务的分类报告
                    try:
                        from sklearn.metrics import classification_report
                        
                        all_preds = {'class': [], 'mass': [], 'stiffness': [], 'material': []}
                        all_labels = {'class': [], 'mass': [], 'stiffness': [], 'material': []}
                        
                        model.eval()
                        with torch.no_grad():
                            for batch in dataloader:
                                inputs, labels_class, labels_mass, labels_stiffness, labels_material = batch
                                inputs = inputs.to(device)
                                outputs = model(inputs)
                                
                                all_preds['class'].extend(outputs['class'].argmax(1).cpu().numpy())
                                all_preds['mass'].extend(outputs['mass'].argmax(1).cpu().numpy())
                                all_preds['stiffness'].extend(outputs['stiffness'].argmax(1).cpu().numpy())
                                all_preds['material'].extend(outputs['material'].argmax(1).cpu().numpy())
                                
                                all_labels['class'].extend(labels_class.numpy())
                                all_labels['mass'].extend(labels_mass.numpy())
                                all_labels['stiffness'].extend(labels_stiffness.numpy())
                                all_labels['material'].extend(labels_material.numpy())
                        
                        # Class 分类报告
                        logger.info("\n📊 Classification Report - CLASS:")
                        logger.info(classification_report(all_labels['class'], all_preds['class'], 
                                                          digits=4, zero_division=0))
                        
                        # Mass 分类报告
                        mass_names = list(MASS_TO_IDX.keys())
                        logger.info("\n📊 Classification Report - MASS:")
                        logger.info(classification_report(all_labels['mass'], all_preds['mass'],
                                                          target_names=mass_names, digits=4, zero_division=0))
                        
                        # Stiffness 分类报告
                        stiffness_names = list(STIFFNESS_TO_IDX.keys())
                        logger.info("\n📊 Classification Report - STIFFNESS:")
                        logger.info(classification_report(all_labels['stiffness'], all_preds['stiffness'],
                                                          target_names=stiffness_names, digits=4, zero_division=0))
                        
                        # Material 分类报告
                        material_names = list(MATERIAL_TO_IDX.keys())
                        logger.info("\n📊 Classification Report - MATERIAL:")
                        logger.info(classification_report(all_labels['material'], all_preds['material'],
                                                          target_names=material_names, digits=4, zero_division=0))
                        
                    except ImportError:
                        logger.warning("scikit-learn not installed, skipping detailed reports.")

    logger.info(f'Best val Class Acc: {best_acc:.4f}')
    
    plotter.save()
    plotter.close()
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 on Plaintextdataset visual anchors")
    parser.add_argument('--dataset_root', type=str, default='Plaintextdataset', help='Path to Plaintextdataset root')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='learn_PyBullet/outputs', help='Directory to save model')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--no-augment', action='store_true', help='Disable data augmentation (for overfitting baseline)')
    parser.add_argument('--multitask', action='store_true', help='Enable multi-task learning (class + mass + stiffness + material)')
    
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists(): # Handle relative path check if needed, assume running from repo root
         # fallback check
         if not dataset_root.exists():
             logger.error(f"Dataset root {dataset_root} does not exist.")
             return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / 'resnet18_best.pth'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data augmentation and normalization for training
    # Just normalization for validation
    if args.no_augment:
        logger.info("⚠️  Data augmentation DISABLED (overfitting baseline mode)")
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    else:
        logger.info("✅ Data augmentation ENABLED (RandomResizedCrop + ColorJitter)")
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.9, 1.1)),  # 随机裁剪并缩放
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # 颜色抖动
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    logger.info("Initializing dataset...")
    print_property_summary()  # 打印物理属性表
    full_dataset = PlaintextImageDataset(dataset_root, transform=None) # Transform applied later? No, we need subset specific transforms.
    
    # Actually, random_split splits the dataset but keeps the same underlying dataset object (and thus transform).
    # To handle different transforms for train/val cleanly with random_split, we can wrap the subsets.
    
    class SubsetWrapper(Dataset):
        def __init__(self, subset, transform=None, return_properties=False, class_names=None):
            self.subset = subset
            self.transform = transform
            self.return_properties = return_properties
            self.class_names = class_names
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            # Note: The underlying dataset returns (PIL Image, label) because we passed transform=None to it
            if self.transform:
                x = self.transform(x)
            
            if self.return_properties and self.class_names:
                class_name = self.class_names[y]
                props = PHYSICAL_PROPERTIES.get(class_name, {})
                mass_idx = MASS_TO_IDX.get(props.get('mass', 'medium'), 2)
                stiffness_idx = STIFFNESS_TO_IDX.get(props.get('stiffness', 'medium'), 2)
                material_idx = MATERIAL_TO_IDX.get(props.get('material', 'wood'), 2)
                return x, y, mass_idx, stiffness_idx, material_idx
            
            return x, y
        
        def __len__(self):
            return len(self.subset)

    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dataset = SubsetWrapper(train_subset, transform=data_transforms['train'], 
                                   return_properties=args.multitask, class_names=full_dataset.classes)
    val_dataset = SubsetWrapper(val_subset, transform=data_transforms['val'],
                                 return_properties=args.multitask, class_names=full_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Setup Model
    num_classes = len(full_dataset.classes)
    logger.info(f"Number of classes: {num_classes} -> {full_dataset.classes}")
    
    if args.multitask:
        logger.info("🎯 Setting up Multi-Task ResNet18 model...")
        logger.info(f"  Heads: class({num_classes}) + mass({len(MASS_TO_IDX)}) + stiffness({len(STIFFNESS_TO_IDX)}) + material({len(MATERIAL_TO_IDX)})")
        model_ft = MultiTaskResNet(num_classes=num_classes, pretrained=True)
    else:
        logger.info("Setting up Single-Task ResNet18 model...")
        model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.learning_rate)

    # Train
    if args.multitask:
        train_multitask_model(model_ft, train_loader, val_loader, criterion, optimizer_ft, 
                              num_epochs=args.epochs, device=device, save_path=save_path, output_dir=output_dir)
    else:
        train_model(model_ft, train_loader, val_loader, criterion, optimizer_ft, 
                    num_epochs=args.epochs, device=device, save_path=save_path, output_dir=output_dir)

    # Save class names for inference
    class_names_path = output_dir / 'class_names.txt'
    with open(class_names_path, 'w') as f:
        for cls in full_dataset.classes:
            f.write(f"{cls}\n")
    logger.info(f"Saved class names to {class_names_path}")
    
    # Save physical properties mapping for inference
    import json
    properties_path = output_dir / 'physical_properties.json'
    properties_data = {
        'properties': PHYSICAL_PROPERTIES,
        'mass_to_idx': MASS_TO_IDX,
        'stiffness_to_idx': STIFFNESS_TO_IDX,
        'material_to_idx': MATERIAL_TO_IDX,
        'idx_to_mass': IDX_TO_MASS,
        'idx_to_stiffness': IDX_TO_STIFFNESS,
        'idx_to_material': IDX_TO_MATERIAL,
    }
    with open(properties_path, 'w') as f:
        json.dump(properties_data, f, indent=2)
    logger.info(f"Saved physical properties to {properties_path}")

if __name__ == '__main__':
    main()
