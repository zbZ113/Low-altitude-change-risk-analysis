# dataset.py
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# RGB → 类别映射（含未定义/忽略 = 6）
COLOR2LABEL = {
    (0, 128, 0): 0,       # Low vegetation
    (128, 128, 128): 1,   # Non-veg surface
    (0, 255, 0): 2,       # Tree
    (0, 0, 255): 3,       # Water
    (128, 0, 0): 4,       # Building
    (255, 0, 0): 5,       # Playground
    (255, 255, 255): 6    # Unchanged / ignore
}

def build_transform(img_size, is_train: bool):
    """
    - 图像使用线性插值，mask 使用最近邻；
    - 边界常数填充：图像 value=0，mask mask_value=6（忽略类）；
    - 通过 additional_targets 保证 (im2, label2, change) 与 im1 同步。
    """
    if is_train:
        return A.Compose(
            [
                A.RandomCrop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.2),

                # 纯旋转（替代旧 Affine 的不兼容参数）
                A.ShiftScaleRotate(
                    shift_limit=0.0, scale_limit=0.0, rotate_limit=30,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,               # 图像边界填充值
                    mask_value=6,          # 掩码边界填“忽略类=6”
                    p=0.5
                ),

                # 位移 + 缩放 + 轻度旋转
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=15,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=6,
                    p=0.5
                ),

                A.GaussianBlur(p=0.3),

                # 归一化到 [-1, 1]
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ],
            additional_targets={
                'image2': 'image',
                'mask2': 'mask',
                'mask3': 'mask'
            }
        )
    else:
        return A.Compose(
            [
                A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ],
            additional_targets={
                'image2': 'image',
                'mask2': 'mask',
                'mask3': 'mask'
            }
        )

class SECOND_Dataset(Dataset):
    """
    目录结构：
      root/
        train|val|test/
          im1/*.png
          im2/*.png
          label1/*.png     # RGB 标签
          label2/*.png
    """
    def __init__(self, root, split='train', img_size=512):
        super().__init__()
        self.root = root
        self.split = split
        self.img_size = int(img_size)
        self.transform = build_transform(self.img_size, is_train=(split == 'train'))

        im1_dir = os.path.join(root, split, 'im1')
        self.image_list = sorted(os.listdir(im1_dir))

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def _rgb_to_label(rgb: np.ndarray) -> np.ndarray:
        """将 RGB 彩色标签映射到类别 id；未匹配像素置 6（忽略）"""
        h, w, _ = rgb.shape
        label = np.full((h, w), 6, dtype=np.uint8)
        for color, cls in COLOR2LABEL.items():
            mask = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
            label[mask] = cls
        return label

    @staticmethod
    def _generate_change_mask(label1: np.ndarray, label2: np.ndarray) -> np.ndarray:
        """仅在 (label1!=6 & label2!=6) 且 label1!=label2 的位置记为 1"""
        valid = (label1 != 6) & (label2 != 6)
        change = (label1 != label2) & valid
        return change.astype(np.uint8)

    def __getitem__(self, idx):
        name = self.image_list[idx]
        im1_path  = os.path.join(self.root, self.split, 'im1', name)
        im2_path  = os.path.join(self.root, self.split, 'im2', name)
        lab1_path = os.path.join(self.root, self.split, 'label1', name)
        lab2_path = os.path.join(self.root, self.split, 'label2', name)

        # 读取图像（RGB）
        im1 = np.array(Image.open(im1_path).convert('RGB'))
        im2 = np.array(Image.open(im2_path).convert('RGB'))

        # 读取标签（RGB→类 id）
        label1_rgb = np.array(Image.open(lab1_path).convert('RGB'))
        label2_rgb = np.array(Image.open(lab2_path).convert('RGB'))
        label1 = self._rgb_to_label(label1_rgb)
        label2 = self._rgb_to_label(label2_rgb)

        # 生成变化掩码（0/1）
        change_mask = self._generate_change_mask(label1, label2)

        # 同步增广
        transformed = self.transform(
            image=im1, image2=im2,
            mask=label1, mask2=label2, mask3=change_mask
        )
        im1_t = transformed['image']    # float32, [0,1]后经Normalize变[-1,1]
        im2_t = transformed['image2']
        label1_t = transformed['mask']   # tensor
        label2_t = transformed['mask2']
        change_mask_t = transformed['mask3']

        # ---------------- 强制清洗（关键防 NaN） ----------------
        # 语义：裁剪到 [0..5]，其余置 6（忽略）
        six = torch.tensor(6, dtype=torch.long, device=label1_t.device)
        label1_t = label1_t.long()
        label2_t = label2_t.long()
        label1_t = torch.where((label1_t >= 0) & (label1_t <= 5), label1_t, six)
        label2_t = torch.where((label2_t >= 0) & (label2_t <= 5), label2_t, six)

        # 变化：严格二值化到 {0,1}，float
        change_mask_t = (change_mask_t > 0.5).float()

        return {
            'im1': im1_t,
            'im2': im2_t,
            'label1': label1_t,
            'label2': label2_t,
            'change_mask': change_mask_t,
            'name': name
        }
