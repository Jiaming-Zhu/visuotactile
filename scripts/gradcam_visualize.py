#!/usr/bin/env python3
"""
Grad-CAM 可视化脚本：查看 ResNet18 模型在分类时关注图像的哪些区域。

使用方法:
    python learn_PyBullet/scripts/gradcam_visualize.py --model_path learn_PyBullet/outputs/resnet18_best.pth
    python learn_PyBullet/scripts/gradcam_visualize.py --image_path Plaintextdataset/WoodBlock_Red/episode_xxx/visual_anchor.jpg
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """Grad-CAM 实现类"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """保存前向传播的特征图"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """保存反向传播的梯度"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        生成 Grad-CAM 热力图
        
        Args:
            input_tensor: 输入图像张量 (1, C, H, W)
            target_class: 目标类别索引，None 则使用预测类别
            
        Returns:
            cam: 热力图 numpy 数组 (H, W)
        """
        self.model.eval()
        
        # 前向传播
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 清零梯度
        self.model.zero_grad()
        
        # 反向传播目标类别的分数
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算权重：对梯度进行全局平均池化
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # 加权求和特征图
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU 激活（只保留正向影响）
        cam = F.relu(cam)
        
        # 归一化到 [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class, output.softmax(dim=1)[0, target_class].item()


def load_model(model_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    """加载训练好的模型"""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_class_names(class_names_path: Path) -> List[str]:
    """加载类别名称"""
    with open(class_names_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def preprocess_image(image_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image.resize((224, 224)))
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor, original_image


def overlay_cam_on_image(image: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """将 CAM 热力图叠加到原图上"""
    # 将 CAM 上采样到图像大小
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (image.shape[1], image.shape[0]), Image.BILINEAR)) / 255.0
    
    # 应用颜色映射
    heatmap = cm.jet(cam_resized)[:, :, :3]  # 去掉 alpha 通道
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 叠加
    overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
    
    return overlay, heatmap, cam_resized


def find_sample_images(dataset_root: Path, num_samples: int = 10, target_classes: List[str] = None) -> List[Path]:
    """从数据集中随机选择样本图像
    
    Args:
        dataset_root: 数据集根目录
        num_samples: 每个类别的样本数量（如果指定了target_classes）或总样本数
        target_classes: 指定要可视化的类别列表，None则从所有类别随机选择
    """
    all_images = []
    
    if target_classes:
        # 从指定类别中选择
        for cls_name in target_classes:
            cls_dir = dataset_root / cls_name
            if cls_dir.exists() and cls_dir.is_dir():
                episodes = [d for d in cls_dir.iterdir() if d.is_dir()]
                random.shuffle(episodes)
                for ep_dir in episodes[:num_samples]:
                    img_path = ep_dir / "visual_anchor.jpg"
                    if img_path.exists():
                        all_images.append(img_path)
    else:
        # 从所有类别随机选择
        for class_dir in dataset_root.iterdir():
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                for episode_dir in class_dir.iterdir():
                    if episode_dir.is_dir():
                        img_path = episode_dir / "visual_anchor.jpg"
                        if img_path.exists():
                            all_images.append(img_path)
        
        random.shuffle(all_images)
        all_images = all_images[:num_samples]
    
    return all_images


def visualize_gradcam(
    model_path: Path,
    class_names_path: Path,
    image_paths: List[Path],
    output_dir: Path,
    device: torch.device
):
    """可视化 Grad-CAM"""
    
    # 加载类别名称和模型
    class_names = load_class_names(class_names_path)
    num_classes = len(class_names)
    model = load_model(model_path, num_classes, device)
    
    # 获取 ResNet18 的最后一个卷积层 (layer4)
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算网格布局
    num_images = len(image_paths)
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols * 3, figsize=(5 * cols * 3, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, image_path in enumerate(image_paths):
        row = idx // cols
        col_base = (idx % cols) * 3
        
        # 预处理图像
        input_tensor, original_image = preprocess_image(image_path)
        input_tensor = input_tensor.to(device)
        
        # 生成 Grad-CAM
        cam, pred_class, confidence = gradcam.generate_cam(input_tensor)
        
        # 叠加热力图
        overlay, heatmap, cam_resized = overlay_cam_on_image(original_image, cam)
        
        # 获取真实类别（从路径提取）
        true_class = image_path.parent.parent.name
        pred_class_name = class_names[pred_class]
        
        # 绘制原图
        axes[row, col_base].imshow(original_image)
        axes[row, col_base].set_title(f'Original\nTrue: {true_class}', fontsize=9)
        axes[row, col_base].axis('off')
        
        # 绘制热力图
        axes[row, col_base + 1].imshow(cam_resized, cmap='jet')
        axes[row, col_base + 1].set_title(f'Grad-CAM Heatmap', fontsize=9)
        axes[row, col_base + 1].axis('off')
        
        # 绘制叠加图
        axes[row, col_base + 2].imshow(overlay)
        correct = '✓' if true_class == pred_class_name else '✗'
        axes[row, col_base + 2].set_title(
            f'Overlay {correct}\nPred: {pred_class_name}\nConf: {confidence:.2%}', 
            fontsize=9,
            color='green' if true_class == pred_class_name else 'red'
        )
        axes[row, col_base + 2].axis('off')
    
    # 隐藏多余的子图
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col_base = (idx % cols) * 3
        for i in range(3):
            axes[row, col_base + i].axis('off')
    
    plt.suptitle('Grad-CAM Visualization: Where is the model looking?', fontsize=14)
    plt.tight_layout()
    
    save_path = output_dir / 'gradcam_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved Grad-CAM visualization to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for ResNet18")
    parser.add_argument('--model_path', type=str, default='learn_PyBullet/outputs/resnet18_best.pth',
                        help='Path to trained model weights')
    parser.add_argument('--class_names_path', type=str, default='learn_PyBullet/outputs/class_names.txt',
                        help='Path to class names file')
    parser.add_argument('--dataset_root', type=str, default='Plaintextdataset',
                        help='Path to dataset root (for random sampling)')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Specific image path to visualize (optional)')
    parser.add_argument('--num_samples', type=int, default=6,
                        help='Number of random samples to visualize (per class if --classes specified)')
    parser.add_argument('--output_dir', type=str, default='learn_PyBullet/outputs',
                        help='Directory to save visualization')
    parser.add_argument('--classes', type=str, nargs='+', default=None,
                        help='Specific class names to visualize (e.g., --classes CardboardBox_Hollow CardboardBox_RockFilled)')
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = Path(args.model_path)
    class_names_path = Path(args.class_names_path)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    if not class_names_path.exists():
        print(f"Error: Class names file not found at {class_names_path}")
        return
    
    # 确定要可视化的图像
    if args.image_path:
        image_paths = [Path(args.image_path)]
    else:
        image_paths = find_sample_images(dataset_root, args.num_samples, target_classes=args.classes)
    
    if not image_paths:
        print("Error: No images found to visualize")
        return
    
    print(f"Visualizing {len(image_paths)} images...")
    
    visualize_gradcam(model_path, class_names_path, image_paths, output_dir, device)


if __name__ == '__main__':
    main()

