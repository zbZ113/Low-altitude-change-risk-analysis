import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import os
import torch
import torch.nn.functional as F

# 类别颜色（0~5），6 表示未变化区域
CLASS_COLORS = {
    0: [0, 128, 0],       # Low vegetation
    1: [128, 128, 128],   # Non-veg surface
    2: [0, 255, 0],       # Tree
    3: [0, 0, 255],       # Water
    4: [128, 0, 0],       # Building
    5: [255, 0, 0],       # Playground
    6: [255, 255, 255]    # Unchanged class (non-change)
}

def visualize_sample(model, dataloader, device, epoch, save_root='outputs/vis', num_samples=1):
    """对指定数量的样本进行九宫格可视化"""
    model.eval()
    os.makedirs(save_root, exist_ok=True)

    for idx, batch in enumerate(dataloader):
        if idx >= num_samples:
            break

        im1 = batch['im1'].to(device)
        im2 = batch['im2'].to(device)
        gt_change = batch['change_mask'][0].cpu().numpy()
        gt_sem1 = batch['label1'][0].cpu().numpy()
        gt_sem2 = batch['label2'][0].cpu().numpy()
        name = batch['name'][0]

        with torch.no_grad():
            pred_change, pred_sem1, pred_sem2 = model(im1, im2)

            # 强制 resize 输出为 512×512
            pred_change = F.interpolate(pred_change, size=(512, 512), mode='bilinear', align_corners=False)
            pred_sem1 = F.interpolate(pred_sem1, size=(512, 512), mode='bilinear', align_corners=False)
            pred_sem2 = F.interpolate(pred_sem2, size=(512, 512), mode='bilinear', align_corners=False)

            pred_change = (torch.sigmoid(pred_change) > 0.5).float()
            pred_change = pred_change[0, 0].cpu().numpy()

            pred_sem1 = torch.argmax(pred_sem1, dim=1)[0].cpu().numpy()
            pred_sem2 = torch.argmax(pred_sem2, dim=1)[0].cpu().numpy()

        # 图像反归一化
        im1_np = ((im1[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
        im2_np = ((im2[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)

        fig = plot_nine_grid(im1_np, im2_np, gt_change, pred_change,
                             gt_sem1, pred_sem1, gt_sem2, pred_sem2)

        save_path = os.path.join(save_root, f"epoch_{epoch}_sample{idx}.png")
        fig.savefig(save_path)
        plt.close()
        print(f"[可视化] 已保存预测图像至 {save_path}")

def plot_nine_grid(im1, im2, gt_change, pred_change, gt_sem1, pred_sem1, gt_sem2, pred_sem2):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    images = [
        (im1, "Time 1 Image"),
        (im2, "Time 2 Image"),
        (gt_change, "GT Change", 'gray'),
        (gt_sem1, "GT Semantics T1", 'color'),
        (gt_sem2, "GT Semantics T2", 'color'),
        (pred_change, "Pred Change", 'gray'),
        (pred_sem1, "Pred Semantics T1", 'color'),
        (pred_sem2, "Pred Semantics T2", 'color')
    ]

    for idx, (img, title, *mode) in enumerate(images):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        if mode and mode[0] == 'gray':
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        elif mode and mode[0] == 'color':
            ax.imshow(colorize_mask(img))
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    cm = compute_change_matrix(gt_sem1, gt_sem2)
    plot_change_matrix(cm, ax=axes[2, 2])

    plt.tight_layout()
    return fig

def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask

def compute_change_matrix(gt_sem1, gt_sem2, num_classes=6):
    change_matrix = np.zeros((num_classes, num_classes), dtype=int)
    valid_mask = (gt_sem1 != 6) & (gt_sem2 != 6)
    t1 = gt_sem1[valid_mask]
    t2 = gt_sem2[valid_mask]
    for i in range(num_classes):
        for j in range(num_classes):
            change_matrix[i, j] = np.sum((t1 == i) & (t2 == j))
    return change_matrix

def plot_change_matrix(matrix, ax):
    im = ax.imshow(matrix, cmap='viridis')
    ax.set_title('Change Matrix')
    ax.set_xlabel('Time 2 Class')
    ax.set_ylabel('Time 1 Class')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white')
    fig = ax.get_figure()
    fig.colorbar(im, ax=ax, label='Pixel Count')
