# -*- coding: utf-8 -*-
"""
val_sec.py —— SECOND/val 验证（修正 BN→GN 跨设备问题）
要点：
  1) ★ 在加载 ckpt 前，将模型中的 BatchNorm2d 全部替换为 GroupNorm（与训练保持一致）
     —— 先替换，再 model.to(device)，避免 GroupNorm 留在 CPU
  2) 输出修正后的指标：
     - Change（二值）：F1(best, 扫阈值) / IoU(best)；F1/IoU@0.5
     - 语义 mIoU：T1 / T2（忽略 6=未变化）
     - 官方二类 mIoU（两口径）：
         miou_official_full   （★全图口径，通常 70%+ 的对齐口径）
         miou_official_valid  （valid 口径，可与旧日志对齐）
     - SeK：sek（0~1）与 sek_x100（×100 后常见“20 多”）
"""

import os
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn

# ==== 项目内模块（模型/数据） ====
from model import MultiTaskChangeFormer                 # 三分支：change + sem1 + sem2  :contentReference[oaicite:2]{index=2}
from dataset import SECOND_Dataset                      # SECOND 读写与颜色→类ID、ignore=6       :contentReference[oaicite:3]{index=3}

# ------------------------------------------------------------------------------------
# 训练时做过的“BN→GN”替换（与 train.py 保持一致；并★对齐设备）  :contentReference[oaicite:4]{index=4}
# ------------------------------------------------------------------------------------
def _choose_groups(num_channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g //= 2
    return max(1, g)

def convert_bn_to_gn(module: nn.Module, max_groups: int = 32):
    """
    递归替换：nn.BatchNorm2d -> nn.GroupNorm；保持 affine=True，并拷贝权重/偏置。
    ★ 新建的 GroupNorm 显式放到原 BN 所在设备，避免“CUDA 输入 + CPU GN 权重”报错。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            groups = _choose_groups(num_channels, max_groups)
            gn = nn.GroupNorm(num_groups=groups, num_channels=num_channels, eps=child.eps, affine=True)
            # —— 关键：把 GN 放到与 BN 相同的设备，再拷贝参数
            gn = gn.to(child.weight.device) if child.affine else gn
            with torch.no_grad():
                if child.affine:
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, max_groups=max_groups)
    return module

# ------------------------------------------------------------------------------------
# 度量：语义 mIoU、SeK（含 ×100 展示）
# ------------------------------------------------------------------------------------
EPS = 1e-9

def _miou_from_cm(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - tp
    ious = tp / np.maximum(union, 1e-12)
    return float(np.nanmean(ious))

def _build_scd_Q(gt1: torch.Tensor, gt2: torch.Tensor,
                 pd1: torch.Tensor, pd2: torch.Tensor,
                 K: int = 6) -> np.ndarray:
    """
    构造 SCD 混淆矩阵 Q（C x C），C = 1 + K*(K-1)
    - 0 索引 = unchanged（a==b）
    - 1..  = 所有 a->b（a!=b），按行优先压缩掉对角线
    忽略任何出现 ignore 类(=6)的像素。
    """
    C = 1 + K * (K - 1)
    Q = np.zeros((C, C), dtype=np.int64)

    a = gt1.detach().cpu().numpy()
    b = gt2.detach().cpu().numpy()
    c = pd1.detach().cpu().numpy()
    d = pd2.detach().cpu().numpy()

    valid = (a >= 0) & (a < K) & (b >= 0) & (b < K) & \
            (c >= 0) & (c < K) & (d >= 0) & (d < K)
    if not np.any(valid):
        return Q

    a = a[valid].astype(np.int64); b = b[valid].astype(np.int64)
    c = c[valid].astype(np.int64); d = d[valid].astype(np.int64)

    gi = np.where(a == b, 0, 1 + a * (K - 1) + (b - (b > a)))
    pi = np.where(c == d, 0, 1 + c * (K - 1) + (d - (d > c)))

    idx = gi * C + pi
    binc = np.bincount(idx, minlength=C*C)
    Q += binc.reshape(C, C)
    return Q

def _sek_from_Q(Q: np.ndarray) -> float:
    """
    Separated Kappa（返回 0~1）；如需“20 多”的口径，乘以 100 显示。
    """
    Q = Q.astype(np.float64, copy=False)
    total = Q.sum()
    if total <= 0:
        return 0.0
    q11 = Q[0, 0]
    total_wo = total - q11
    if total_wo <= 0:
        return 0.0

    iou2 = Q[1:, 1:].sum() / total_wo
    rho  = np.trace(Q[1:, 1:]) / total_wo
    Q_hat = Q.copy(); Q_hat[0, 0] = 0.0
    row, col = Q_hat.sum(1), Q_hat.sum(0)
    eta = (row @ col) / (total_wo**2 + 1e-12)

    return float(np.exp(iou2 - 1.0) * (rho - eta) / (1.0 - eta + 1e-12))

# ------------------------------------------------------------------------------------
# 验证主流程
# ------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_second(ckpt_path: str,
                    val_dir: str,
                    batch_size: int = 2,
                    img_size: int = 512,
                    num_workers: int = 0,
                    thr_min: float = 0.1, thr_max: float = 0.9, thr_points: int = 17):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # --- 解析 SECOND 根目录 ---
    val_dir = os.path.normpath(val_dir)
    second_root = os.path.dirname(val_dir) if os.path.basename(val_dir).lower() == "val" else val_dir

    # --- 数据 ---
    val_set = SECOND_Dataset(second_root, split='val', img_size=img_size)   # ignore_index=6 的定义在此  :contentReference[oaicite:5]{index=5}
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )

    # --- 模型（★先 BN→GN，再整体搬到 device；与训练结构一致） ---
    # MultiTaskChangeFormer = Siamese 编码器 + 原 ChangeFormer 解码器 + 语义头  :contentReference[oaicite:6]{index=6}
    model = MultiTaskChangeFormer(n_classes_sem=7, decoder_embed_dim=512)   # 先在 CPU
    convert_bn_to_gn(model, max_groups=32)  # ★ 替换发生在 CPU，避免 GN 留在 CPU 的风险  :contentReference[oaicite:7]{index=7}
    model = model.to(device)                # ★ 再整体搬到 CUDA/CPU

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到权重：{ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)

    # 兼容 DataParallel 存在的 'module.' 前缀
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        print("[WARN] 严格加载失败，尝试 strict=False：\n", str(e))
        model.load_state_dict(sd, strict=False)
    model.eval()

    # --- 阈值并行向量 ---
    T = torch.linspace(thr_min, thr_max, thr_points, device=device)

    # --- 累计器 ---
    tot_tp = torch.zeros(thr_points, device=device, dtype=torch.long)
    tot_fp = torch.zeros_like(tot_tp)
    tot_fn = torch.zeros_like(tot_tp)
    tp05 = torch.zeros((), device=device, dtype=torch.long)
    fp05 = torch.zeros((), device=device, dtype=torch.long)
    fn05 = torch.zeros((), device=device, dtype=torch.long)

    sem_K = 6
    cm1 = torch.zeros((sem_K, sem_K), device=device, dtype=torch.long)
    cm2 = torch.zeros((sem_K, sem_K), device=device, dtype=torch.long)

    # mIoU（全图/valid）
    inter_chg_full = inter_non_full = 0
    union_chg_full = union_non_full = 0
    inter_chg_valid = inter_non_valid = 0
    union_chg_valid = union_non_valid = 0

    # SeK
    C_scd = 1 + 6 * 5
    Q = np.zeros((C_scd, C_scd), dtype=np.int64)

    amp_enabled = use_amp
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
        for batch in val_loader:
            im1 = batch['im1'].to(device, non_blocking=True)
            im2 = batch['im2'].to(device, non_blocking=True)
            label1 = batch['label1'].to(device, non_blocking=True)   # [B,H,W] in {0..6}
            label2 = batch['label2'].to(device, non_blocking=True)
            gt_change = batch['change_mask'].to(device, non_blocking=True)  # [B,H,W] or [B,1,H,W]

            # 前向
            pred_change, pred_label1, pred_label2 = model(im1, im2)

            # 对齐尺寸
            target_hw = label1.shape[-2:]
            if pred_change.shape[-2:] != target_hw:
                pred_change = F.interpolate(pred_change, size=target_hw, mode='bilinear', align_corners=False)
            if pred_label1.shape[-2:] != target_hw:
                pred_label1 = F.interpolate(pred_label1, size=target_hw, mode='bilinear', align_corners=False)
            if pred_label2.shape[-2:] != target_hw:
                pred_label2 = F.interpolate(pred_label2, size=target_hw, mode='bilinear', align_corners=False)

            # ---------- Change（二值） ----------
            prob = torch.sigmoid(pred_change)            # [B,1,H,W]
            gt = gt_change
            if gt.ndim == 3:
                gt = gt.unsqueeze(1)
            gt = (gt > 0).to(torch.int32)

            pred05 = (prob > 0.5).to(torch.int32)
            tp05 += ((pred05 == 1) & (gt == 1)).sum()
            fp05 += ((pred05 == 1) & (gt == 0)).sum()
            fn05 += ((pred05 == 0) & (gt == 1)).sum()

            pred_multi = (prob.unsqueeze(-1) > T).to(torch.int32)     # [B,1,H,W,T]
            tot_tp += ((pred_multi == 1) & (gt.unsqueeze(-1) == 1)).sum(dim=(0,1,2,3))
            tot_fp += ((pred_multi == 1) & (gt.unsqueeze(-1) == 0)).sum(dim=(0,1,2,3))
            tot_fn += ((pred_multi == 0) & (gt.unsqueeze(-1) == 1)).sum(dim=(0,1,2,3))

            # ---------- 语义（忽略 6） ----------
            pred_sem1 = torch.argmax(pred_label1, dim=1)  # [B,H,W]
            pred_sem2 = torch.argmax(pred_label2, dim=1)

            mask1 = (label1 != 6)
            mask2 = (label2 != 6)
            if mask1.any():
                gt1 = label1[mask1]; pd1 = pred_sem1[mask1]
                v1 = (gt1 >= 0) & (gt1 < sem_K) & (pd1 >= 0) & (pd1 < sem_K)
                if v1.any():
                    idx1 = (gt1[v1] * sem_K + pd1[v1]).to(torch.long)
                    cm1 += torch.bincount(idx1, minlength=sem_K*sem_K).reshape(sem_K, sem_K)
            if mask2.any():
                gt2 = label2[mask2]; pd2 = pred_sem2[mask2]
                v2 = (gt2 >= 0) & (gt2 < sem_K) & (pd2 >= 0) & (pd2 < sem_K)
                if v2.any():
                    idx2 = (gt2[v2] * sem_K + pd2[v2]).to(torch.long)
                    cm2 += torch.bincount(idx2, minlength=sem_K*sem_K).reshape(sem_K, sem_K)

            # ---------- 官方二类 mIoU ----------
            # A) 全图口径（与你看到的 70%+ 常见口径一致）
            gt_changed_full   = (label1 != 6) & (label2 != 6) & (label1 != label2)
            gt_nonchange_full = ~gt_changed_full
            pd_changed_full   = (pred_sem1 != pred_sem2)
            pd_nonchange_full = ~pd_changed_full

            inter_chg_full += int((pd_changed_full & gt_changed_full).sum().item())
            union_chg_full += int(((pd_changed_full | gt_changed_full)).sum().item())
            inter_non_full += int((pd_nonchange_full & gt_nonchange_full).sum().item())
            union_non_full += int(((pd_nonchange_full | gt_nonchange_full)).sum().item())

            # B) valid 口径（仅在两时相都≠6处统计，等价于你旧日志）
            valid = (label1 != 6) & (label2 != 6)
            gt_changed_valid   = valid & (label1 != label2)
            gt_nonchange_valid = valid & (label1 == label2)
            pd_changed_valid   = valid & (pred_sem1 != pred_sem2)
            pd_nonchange_valid = valid & (pred_sem1 == pred_sem2)

            inter_chg_valid += int((pd_changed_valid & gt_changed_valid).sum().item())
            union_chg_valid += int(((pd_changed_valid | gt_changed_valid) & valid).sum().item())
            inter_non_valid += int((pd_nonchange_valid & gt_nonchange_valid).sum().item())
            union_non_valid += int(((pd_nonchange_valid | gt_nonchange_valid) & valid).sum().item())

            # ---------- SeK ----------
            Q += _build_scd_Q(label1, label2, pred_sem1, pred_sem2, K=6)

    # --- 聚合：change（二值） ---
    denom_curve = (2 * tot_tp + tot_fp + tot_fn).to(torch.float32)
    f1_curve = (2 * tot_tp.to(torch.float32)) / torch.clamp(denom_curve, min=1.0)
    iou_curve = tot_tp.to(torch.float32) / torch.clamp((tot_tp + tot_fp + tot_fn).to(torch.float32), min=1.0)
    best_idx = int(torch.argmax(f1_curve).item())
    T = torch.linspace(thr_min, thr_max, thr_points, device=device)  # 重新构造以便取值
    best_thr = float(T[best_idx].item())
    f1_best  = float(f1_curve[best_idx].item())
    iou_best = float(iou_curve[best_idx].item())

    denom05 = (2 * tp05 + fp05 + fn05).item()
    f1_05  = float((2 * tp05.item()) / (denom05 + EPS)) if denom05 > 0 else 0.0
    iou_05 = float(tp05.item() / (tp05.item() + fp05.item() + fn05.item() + EPS)) if denom05 > 0 else 0.0

    # --- 语义 mIoU（忽略 6） ---
    miou1 = _miou_from_cm(cm1.detach().cpu().numpy())
    miou2 = _miou_from_cm(cm2.detach().cpu().numpy())

    # --- 二类 mIoU（两口径） ---
    iou_nc_full = inter_non_full / max(1, union_non_full)
    iou_cg_full = inter_chg_full / max(1, union_chg_full)
    miou_official_full = 0.5 * (iou_nc_full + iou_cg_full)

    iou_nc_valid = inter_non_valid / max(1, union_non_valid)
    iou_cg_valid = inter_chg_valid / max(1, union_chg_valid)
    miou_official_valid = 0.5 * (iou_nc_valid + iou_cg_valid)

    # --- SeK ---
    sek = _sek_from_Q(Q)
    sek_x100 = sek * 100.0

    out = {
        "best_thr": round(best_thr, 5),
        "f1_best": f1_best, "iou_best": iou_best,
        "f1@0.5": f1_05, "iou@0.5": iou_05,

        "miou1_T1(ignore6)": miou1,
        "miou2_T2(ignore6)": miou2,

        "iou_nonchange_full": iou_nc_full,
        "iou_change_full": iou_cg_full,
        "miou_official_full": miou_official_full,     # ★ 全图口径（常见 70%+）

        "iou_nonchange_valid": iou_nc_valid,
        "iou_change_valid": iou_cg_valid,
        "miou_official_valid": miou_official_valid,   # 仅 valid 口径（旧日志）

        "sek": sek, "sek_x100": sek_x100              # ★ sek×100 方便与你对齐“20 多”
    }
    return out

# ------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
        default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--val", type=str,
        default="data/Second")
    parser.add_argument("--batch", type=int, default=2)          # 8G 显存建议 1~2
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)        # Windows 用 0~2
    args = parser.parse_args()

    os.makedirs("outputs/validation", exist_ok=True)
    metrics = evaluate_second(args.ckpt, args.val, batch_size=args.batch,
                              img_size=args.img_size, num_workers=args.workers)

    # 打印
    print("\n================= VALIDATION (SECOND/val) =================")
    for k, v in metrics.items():
        print(f"{k:>22s}: {v:.6f}" if isinstance(v, float) else f"{k:>22s}: {v}")
    print("===========================================================\n")

    # 保存
    with open(os.path.join("outputs/validation", "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join("outputs/validation", "val_metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k:>22s}: {v:.6f}\n" if isinstance(v, float) else f"{k:>22s}: {v}\n")

    print("[OK] 指标已保存到 outputs/validation/")

if __name__ == "__main__":
    main()
