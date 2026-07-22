# evaluate.py (fast & aligned + robust guards)
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from visualize import colorize_mask

# ---------- global guards ----------
EPS = 1e-8
# 在多次调用 evaluate_all 之间记住上一轮最优阈值（用于回退）
LAST_BEST_THR = 0.5

# ---------- helpers ----------
def _miou_from_cm(cm_np: np.ndarray) -> float:
    tp = np.diag(cm_np)
    union = cm_np.sum(0) + cm_np.sum(1) - tp
    ious = tp / np.maximum(union, 1e-9)
    return float(np.nanmean(ious))

def _sek_from_Q(Q: np.ndarray) -> float:
    Q = Q.astype(np.float64, copy=False)
    total = Q.sum()
    if total <= 0:
        return 0.0
    q11 = Q[0, 0]
    total_wo = total - q11
    if total_wo <= 0:
        return 0.0
    # IOU2：把所有“变化类型”视为一个大类后的 IoU（交/并）
    iou2 = Q[1:, 1:].sum() / total_wo
    # 观测一致率（变化类型的对角线命中率）
    rho = np.trace(Q[1:, 1:]) / total_wo
    # 机会一致率（移去 q11 后）
    Q_hat = Q.copy()
    Q_hat[0, 0] = 0.0
    row, col = Q_hat.sum(1), Q_hat.sum(0)
    eta = (row @ col) / (total_wo ** 2 + 1e-12)
    return float(np.exp(iou2 - 1.0) * (rho - eta) / (1.0 - eta + 1e-12))

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
    # 搬到 CPU / numpy
    a = gt1.detach().cpu().numpy()
    b = gt2.detach().cpu().numpy()
    c = pd1.detach().cpu().numpy()
    d = pd2.detach().cpu().numpy()

    # 有效：四个张量都在 [0..K-1]
    valid = (a >= 0) & (a < K) & (b >= 0) & (b < K) & \
            (c >= 0) & (c < K) & (d >= 0) & (d < K)
    if not np.any(valid):
        return Q

    a = a[valid].astype(np.int64); b = b[valid].astype(np.int64)
    c = c[valid].astype(np.int64); d = d[valid].astype(np.int64)

    # 索引映射：unchanged -> 0；a!=b -> 1 + a*(K-1) + (b - (b > a))
    gi = np.where(a == b, 0, 1 + a * (K - 1) + (b - (b > a)))
    pi = np.where(c == d, 0, 1 + c * (K - 1) + (d - (d > c)))

    idx = gi * C + pi
    binc = np.bincount(idx, minlength=C * C)
    Q += binc.reshape(C, C)
    return Q

@torch.no_grad()
def evaluate_all(model, dataloader, device,
                 num_classes=7,              # 传 6 或 7 都可；语义 IoU 实际按 K=6 计算
                 save_dir=None, criterion=None,
                 thr_min=0.1, thr_max=0.9, thr_points=17,
                 save_images=False, save_max=2):
    """
    对齐 SECOND 的验证（加入稳健性保护）：
      - Change 分支：F1/IoU 扫阈值 & 固定阈值 0.5（稳定除法，空掩码跳过，阈值回退）
      - 语义分割：mIoU@T1 / mIoU@T2（忽略类=6），在 GPU 上累计混淆矩阵
      - 官方口径 mIOU = 0.5 * (IOU_nonchange + IOU_change)（只在有效像素上统计）
      - SeK（Separated Kappa）：基于 SCD 混淆矩阵 Q（unchanged + 全部变化类型）
    """
    global LAST_BEST_THR

    model.eval()

    # === 阈值向量（GPU 并行）
    T = torch.linspace(thr_min, thr_max, thr_points, device=device)

    # === 累计器（change）
    tot_tp = torch.zeros(thr_points, device=device, dtype=torch.long)
    tot_fp = torch.zeros_like(tot_tp)
    tot_fn = torch.zeros_like(tot_tp)
    tp05 = torch.zeros((), device=device, dtype=torch.long)
    fp05 = torch.zeros((), device=device, dtype=torch.long)
    fn05 = torch.zeros((), device=device, dtype=torch.long)

    # === 语义类别数（排除未变化类6）
    sem_K = 6 if num_classes >= 6 else num_classes  # 0..5 有效类
    cm1 = torch.zeros((sem_K, sem_K), device=device, dtype=torch.long)
    cm2 = torch.zeros((sem_K, sem_K), device=device, dtype=torch.long)

    # === 官方 mIOU & SeK 累计
    inter_chg = 0
    union_chg = 0
    inter_non = 0
    union_non = 0
    C_scd = 1 + 6 * 5
    Q = np.zeros((C_scd, C_scd), dtype=np.int64)

    # === 损失累计
    total_loss = total_chg = total_sem1 = total_sem2 = total_kl = 0.0
    total_batches = 0

    saved = 0  # 控制保存数量

    amp_enabled = (device.type == 'cuda')
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=amp_enabled):
        for idx, batch in enumerate(dataloader):
            im1 = batch['im1'].to(device, non_blocking=True)
            im2 = batch['im2'].to(device, non_blocking=True)
            label_change = batch['change_mask'].to(device, non_blocking=True)  # [B,1,H,W] or [B,H,W]
            label1 = batch['label1'].to(device, non_blocking=True)
            label2 = batch['label2'].to(device, non_blocking=True)

            # 前向
            pred_change, pred_label1, pred_label2 = model(im1, im2)

            # NaN/Inf 守卫：任一输出异常就跳过该 batch
            if (torch.isnan(pred_change).any() or torch.isinf(pred_change).any() or
                torch.isnan(pred_label1).any() or torch.isinf(pred_label1).any() or
                torch.isnan(pred_label2).any() or torch.isinf(pred_label2).any()):
                print("[Eval-Guard] NaN/Inf in logits, skip this batch.")
                continue

            # === 对齐尺寸
            target_size = label1.shape[-2:]
            if pred_change.shape[-2:] != target_size:
                pred_change = F.interpolate(pred_change, size=target_size, mode='bilinear', align_corners=False)
            if pred_label1.shape[-2:] != target_size:
                pred_label1 = F.interpolate(pred_label1, size=target_size, mode='bilinear', align_corners=False)
            if pred_label2.shape[-2:] != target_size:
                pred_label2 = F.interpolate(pred_label2, size=target_size, mode='bilinear', align_corners=False)

            # === 损失（如需），数值稳健
            if criterion is not None:
                loss, loss_dict = criterion((pred_change, pred_label1, pred_label2),
                                            (label_change, label1, label2))
                # 将可能的 NaN/Inf 清洗为有限值再累计
                total_loss += float(np.nan_to_num(float(loss), nan=0.0, posinf=0.0, neginf=0.0))
                total_chg  += float(np.nan_to_num(float(loss_dict['change']), nan=0.0, posinf=0.0, neginf=0.0))
                total_sem1 += float(np.nan_to_num(float(loss_dict['sem1']),   nan=0.0, posinf=0.0, neginf=0.0))
                total_sem2 += float(np.nan_to_num(float(loss_dict['sem2']),   nan=0.0, posinf=0.0, neginf=0.0))
                total_kl   += float(np.nan_to_num(float(loss_dict['kl']),     nan=0.0, posinf=0.0, neginf=0.0))
                total_batches += 1

            # === Change：阈值扫描 & 固定 0.5（稳健）
            # 概率前做 NaN→0 清洗
            prob = torch.sigmoid(torch.nan_to_num(pred_change, nan=0.0, posinf=0.0, neginf=0.0))  # [B,1,H,W]
            gt = label_change
            if gt.ndim == 3:
                gt = gt.unsqueeze(1)
            # 将非常见取值（如 255）规范到 {0,1}
            gt = (gt > 0).to(torch.int32)

            # 有效掩码（change 分支默认全有效）
            valid_chg = torch.ones_like(gt, dtype=torch.bool)

            if not valid_chg.any():
                print("[Eval-Guard] empty valid mask for change, skip batch.")
                continue

            # 固定 0.5（先算；加掩码）
            pred05 = (prob > 0.5).to(torch.int32)
            tp05 += ((pred05 == 1) & (gt == 1) & valid_chg).sum()
            fp05 += ((pred05 == 1) & (gt == 0) & valid_chg).sum()
            fn05 += ((pred05 == 0) & (gt == 1) & valid_chg).sum()

            # 扫描（[B,1,H,W,T]）
            pred_multi = (prob.unsqueeze(-1) > T).to(torch.int32)
            tot_tp += ((pred_multi == 1) & (gt.unsqueeze(-1) == 1) & valid_chg.unsqueeze(-1)).sum(dim=(0,1,2,3))
            tot_fp += ((pred_multi == 1) & (gt.unsqueeze(-1) == 0) & valid_chg.unsqueeze(-1)).sum(dim=(0,1,2,3))
            tot_fn += ((pred_multi == 0) & (gt.unsqueeze(-1) == 1) & valid_chg.unsqueeze(-1)).sum(dim=(0,1,2,3))

            # === Semantic：两时相各自混淆矩阵（忽略类6）
            pred_sem1 = torch.argmax(pred_label1, dim=1)  # [B,H,W]
            pred_sem2 = torch.argmax(pred_label2, dim=1)

            mask1 = (label1 != 6)
            mask2 = (label2 != 6)

            if mask1.any():
                gt1 = label1[mask1]
                pd1 = pred_sem1[mask1]
                valid1 = (gt1 >= 0) & (gt1 < sem_K) & (pd1 >= 0) & (pd1 < sem_K)
                if valid1.any():
                    idx1 = (gt1[valid1] * sem_K + pd1[valid1]).to(torch.long)
                    cm1 += torch.bincount(idx1, minlength=sem_K * sem_K).reshape(sem_K, sem_K)

            if mask2.any():
                gt2 = label2[mask2]
                pd2 = pred_sem2[mask2]
                valid2 = (gt2 >= 0) & (gt2 < sem_K) & (pd2 >= 0) & (pd2 < sem_K)
                if valid2.any():
                    idx2 = (gt2[valid2] * sem_K + pd2[valid2]).to(torch.long)
                    cm2 += torch.bincount(idx2, minlength=sem_K * sem_K).reshape(sem_K, sem_K)

            # === 官方 mIOU（仅在有效像素统计）
            valid = (label1 != 6) & (label2 != 6)
            gt_changed   = valid & (label1 != label2)
            gt_nonchange = valid & (label1 == label2)
            pd_changed   = valid & (pred_sem1 != pred_sem2)
            pd_nonchange = valid & (pred_sem1 == pred_sem2)

            inter_chg += int(((pd_changed & gt_changed)).sum().item())
            union_chg += int((((pd_changed | gt_changed) & valid)).sum().item())
            inter_non += int(((pd_nonchange & gt_nonchange)).sum().item())
            union_non += int((((pd_nonchange | gt_nonchange) & valid)).sum().item())

            # === SeK：累计 SCD 混淆矩阵 Q（CPU 汇总）
            Q += _build_scd_Q(label1, label2, pred_sem1, pred_sem2, K=6)

            # === 可选保存（极少量，避免 IO 拖慢）
            if save_dir and save_images and saved < save_max:
                os.makedirs(save_dir, exist_ok=True)
                bin_change = (prob[:, 0] > 0.5).to(torch.uint8) * 255
                for b in range(bin_change.shape[0]):
                    sample_id = f"sample_{idx}_{b}"
                    Image.fromarray(bin_change[b].detach().cpu().numpy()).save(
                        os.path.join(save_dir, f"{sample_id}_pred_change.png"))
                    Image.fromarray(colorize_mask(pred_sem1[b].detach().cpu().numpy())).save(
                        os.path.join(save_dir, f"{sample_id}_pred_label1.png"))
                    Image.fromarray(colorize_mask(pred_sem2[b].detach().cpu().numpy())).save(
                        os.path.join(save_dir, f"{sample_id}_pred_label2.png"))
                    saved += 1
                    if saved >= save_max:
                        break

    # 若整轮都被跳过，给出空结果但不报错
    if (tot_tp + tot_fp + tot_fn).sum().item() == 0 and (tp05 + fp05 + fn05).item() == 0:
        metrics = {
            'f1_score': 0.0, 'iou': 0.0, 'best_thr': float(LAST_BEST_THR),
            'f1@0.5': 0.0, 'iou@0.5': 0.0,
            'miou1': 0.0, 'miou2': 0.0,
            'iou_nonchange': 0.0, 'iou_change': 0.0,
            'miou_official': 0.0, 'sek': 0.0
        }
        if criterion is not None and total_batches > 0:
            metrics.update({
                'val_loss': total_loss / total_batches,
                'val_chg':  total_chg / total_batches,
                'val_sem1': total_sem1 / total_batches,
                'val_sem2': total_sem2 / total_batches,
                'val_kl':   total_kl / total_batches
            })
        return metrics

    # === 聚合：change 分支（加入有效性判断与回退）
    denom_curve = (2 * tot_tp + tot_fp + tot_fn).to(torch.float32)
    valid_thr_mask = denom_curve > 0

    # 先构建曲线（避免除零用 EPS）
    f1_curve = 2 * tot_tp.to(torch.float32) / (denom_curve + EPS)
    iou_curve = tot_tp.to(torch.float32) / (tot_tp + tot_fp + tot_fn + EPS)

    if valid_thr_mask.any():
        # 在有效阈值里找最优
        valid_idx = torch.nonzero(valid_thr_mask, as_tuple=False).squeeze(1)
        best_local = valid_idx[torch.argmax(f1_curve[valid_idx])]
        best_idx = int(best_local.item())
        best_thr = float(T[best_idx].item())
        best_f1  = float(f1_curve[best_idx].item())
        best_iou = float(iou_curve[best_idx].item())
    else:
        # 全无效：沿用上一轮阈值
        best_thr = float(LAST_BEST_THR)
        best_f1, best_iou = 0.0, 0.0

    # 0.5 固定阈值
    denom05 = (2 * tp05 + fp05 + fn05)
    if denom05.item() > 0:
        f1_05  = float((2 * tp05.item()) / (denom05.item() + EPS))
        iou_05 = float(tp05.item() / (tp05.item() + fp05.item() + fn05.item() + EPS))
    else:
        f1_05, iou_05 = 0.0, 0.0

    # === 语义 mIoU（T1/T2）
    cm1_np = cm1.detach().cpu().numpy().astype(np.float64)
    cm2_np = cm2.detach().cpu().numpy().astype(np.float64)
    miou1 = _miou_from_cm(cm1_np)
    miou2 = _miou_from_cm(cm2_np)

    # === 官方 mIOU / SeK
    iou_nc = inter_non / max(1, union_non)
    iou_cg = inter_chg / max(1, union_chg)
    miou_official = 0.5 * (iou_nc + iou_cg)
    sek = _sek_from_Q(Q)

    # 记录本轮最佳阈值（供下一轮回退用）
    LAST_BEST_THR = best_thr

    # === 汇总输出（保持原有键名 + 新增键）
    metrics = {
        'f1_score': best_f1,
        'iou': best_iou,
        'best_thr': best_thr,
        'f1@0.5': f1_05,
        'iou@0.5': iou_05,
        'miou1': miou1,
        'miou2': miou2,
        'iou_nonchange': iou_nc,
        'iou_change': iou_cg,
        'miou_official': miou_official,
        'sek': sek,
    }

    if criterion is not None and total_batches > 0:
        metrics.update({
            'val_loss': total_loss / total_batches,
            'val_chg':  total_chg / total_batches,
            'val_sem1': total_sem1 / total_batches,
            'val_sem2': total_sem2 / total_batches,
            'val_kl':   total_kl / total_batches
        })

    return metrics
