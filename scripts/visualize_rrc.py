import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

# 找一些样本图片
dataset_root = Path('Plaintextdataset')
all_images = []
for class_dir in dataset_root.iterdir():
    if class_dir.is_dir():
        for ep_dir in class_dir.iterdir():
            if ep_dir.is_dir():
                img_path = ep_dir / 'visual_anchor.jpg'
                if img_path.exists():
                    all_images.append((img_path, class_dir.name))

random.shuffle(all_images)
samples = all_images[:4]  # 取4张图

# 定义训练和验证的 transform
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.3, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

# 创建可视化
fig, axes = plt.subplots(4, 6, figsize=(18, 12))
fig.suptitle('Data Augmentation Comparison\\nOriginal | Val (Resize only) | Train Augmented x4', fontsize=14)

for row, (img_path, class_name) in enumerate(samples):
    img = Image.open(img_path).convert('RGB')
    
    # 原图
    axes[row, 0].imshow(img)
    axes[row, 0].set_title(f'Original\\n{class_name}', fontsize=9)
    axes[row, 0].axis('off')
    
    # 验证集处理（只 resize）
    val_img = val_transform(img)
    axes[row, 1].imshow(val_img)
    axes[row, 1].set_title('Val Transform\\n(Resize only)', fontsize=9)
    axes[row, 1].axis('off')
    
    # 训练集处理（随机裁剪 + 增强）- 展示4个不同的随机结果
    for i in range(4):
        train_img = train_transform(img)
        axes[row, 2+i].imshow(train_img)
        axes[row, 2+i].set_title(f'Train Aug #{i+1}', fontsize=9)
        axes[row, 2+i].axis('off')

plt.tight_layout()
plt.savefig('learn_PyBullet/outputs/augmentation_preview.png', dpi=150, bbox_inches='tight')
print('Saved to learn_PyBullet/outputs/augmentation_preview.png')
plt.show()