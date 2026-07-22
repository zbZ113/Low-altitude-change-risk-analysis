# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Focal Loss（带 ignore_index、类权重）--------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input, target):
        target = target.long()

        if input.dim() == 4:
            # [B,C,H,W]
            logpt = F.log_softmax(input, dim=1)
            pt = logpt.exp()
            valid = (target != self.ignore_index)
            if not valid.any():
                return torch.tensor(0.0, device=input.device)
            logpt = logpt.permute(0, 2, 3, 1)[valid]   # [N,C]
            pt    = pt.permute(0, 2, 3, 1)[valid]      # [N,C]
            target = target[valid]                      # [N]
        elif input.dim() == 2:
            # [N,C]
            logpt = F.log_softmax(input, dim=1)
            pt = logpt.exp()
            valid = (target != self.ignore_index)
            if not valid.any():
                return torch.tensor(0.0, device=input.device)
            logpt = logpt[valid]
            pt    = pt[valid]
            target = target[valid]
        else:
            raise ValueError(f"Unsupported input shape: {input.shape}")

        idx = torch.arange(target.numel(), device=input.device)
        logpt = logpt[idx, target]
        pt    = pt[idx, target]

        if self.alpha is not None:
            alpha = self.alpha.to(input.device) if isinstance(self.alpha, torch.Tensor) \
                    else torch.tensor(self.alpha, device=input.device)
            if alpha.dim() == 0:
                alpha = alpha.expand_as(pt)
            else:
                alpha = alpha[target]
        else:
            alpha = 1.0

        loss = -alpha * (1 - pt) ** self.gamma * logpt
        return loss.mean()


# -------- 二值 Dice（配合 BCE）--------
def binary_dice_loss(logits, target, eps=1e-6):
    """
    更稳健版本：
      - 目标强制裁剪到 [0,1]
      - 概率裁剪到 [eps, 1-eps]，避免极端 0/1 下的数值不稳定
    """
    target = torch.clamp(target, 0.0, 1.0)
    prob = torch.sigmoid(logits)
    prob = torch.clamp(prob, eps, 1.0 - eps)

    num = 2 * (prob * target).sum() + eps
    den = (prob.pow(2).sum() + target.sum() + eps)
    return 1.0 - num / den


# -------- 多任务总损失 --------
class MultiTaskLoss(nn.Module):
    """
    total = α·L_change + β·L_sem1 + γ·L_sem2 + λ·KL(change区) + λ_unch·KL(未变化区)
    - 默认开启 supervise_all_sem：在所有有效像素(≠ignore_index)上计算语义 CE
    - 若关闭则回退到：变化区强监督 + 未变化区弱监督(η)
    """
    def __init__(self,
                 alpha=1.0, beta=1.0, gamma=1.0, lambda_kl=0.005,
                 change_weight=None, sem_weight=None, use_focal=False,
                 ignore_index=6, label_smoothing=0.0,
                 supervise_all_sem: bool = True,
                 eta_unch: float = 0.6,
                 lambda_kl_unch: float = 0.002):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_kl = lambda_kl
        self.ignore_index = int(ignore_index)
        self.supervise_all_sem = bool(supervise_all_sem)
        self.eta_unch = float(eta_unch)
        self.lambda_kl_unch = float(lambda_kl_unch)

        # ---- 二值变化：BCE + Dice（带正负不均衡权重）----
        if change_weight is None:
            pos_w = torch.tensor(4.6, dtype=torch.float32)  # 可替换为数据集统计值
        else:
            pos_w = torch.as_tensor(change_weight, dtype=torch.float32)
            if pos_w.numel() != 1:
                raise ValueError("change_weight must be a scalar.")
        self.register_buffer('pos_weight', pos_w)

        # ---- 语义 CE（或 Focal）----
        weight = None
        if sem_weight is not None:
            weight = torch.as_tensor(sem_weight, dtype=torch.float32)
        if use_focal:
            self.sem_loss_t1 = FocalLoss(alpha=weight, ignore_index=self.ignore_index)
            self.sem_loss_t2 = FocalLoss(alpha=weight, ignore_index=self.ignore_index)
        else:
            self.sem_loss_t1 = nn.CrossEntropyLoss(weight=weight, ignore_index=self.ignore_index,
                                                   label_smoothing=label_smoothing)
            self.sem_loss_t2 = nn.CrossEntropyLoss(weight=weight, ignore_index=self.ignore_index,
                                                   label_smoothing=label_smoothing)

        # ---- KL 一致性（温度可调）----
        self.temperature = 0.5

    def kl_divergence(self, p_logits, q_logits, mask=None):
        T = self.temperature
        p_log = F.log_softmax(p_logits / T, dim=1)
        q_soft = torch.clamp(F.softmax(q_logits / T, dim=1), 1e-7, 1 - 1e-7)
        kl = F.kl_div(p_log, q_soft, reduction='none').sum(dim=1)  # [B,H,W]
        if mask is not None:
            kl = kl * mask.float()
        if kl.numel() == 0 or kl.sum() == 0:
            return torch.tensor(0.0, device=p_logits.device)
        # 不额外乘 T*T，保持与你现有数值尺度一致
        return torch.nan_to_num(kl.mean().clamp(0.0, 5.0))

    def forward(self, outputs, targets):
        """
        outputs: (pred_change[B,1,H,W], pred_sem1[B,C,Hs,Ws], pred_sem2[B,C,Hs,Ws])
        targets: (gt_change[B,1,H,W] or [B,H,W], gt_sem1[B,H,W], gt_sem2[B,H,W])
        """
        pred_change, pred_sem1, pred_sem2 = outputs
        gt_change, gt_sem1, gt_sem2 = targets

        # ---- 形状对齐（务必在任何 mask/CE 前进行）----
        if gt_change.ndim == 3:
            gt_change = gt_change.unsqueeze(1)  # -> [B,1,H,W]
        if pred_change.shape[-2:] != gt_change.shape[-2:]:
            pred_change = F.interpolate(pred_change, size=gt_change.shape[-2:], mode='bilinear', align_corners=False)

        if pred_sem1.shape[-2:] != gt_sem1.shape[-2:]:
            pred_sem1 = F.interpolate(pred_sem1, size=gt_sem1.shape[-2:], mode='bilinear', align_corners=False)
        if pred_sem2.shape[-2:] != gt_sem2.shape[-2:]:
            pred_sem2 = F.interpolate(pred_sem2, size=gt_sem2.shape[-2:], mode='bilinear', align_corners=False)

        # ---- 变化分支：BCE + Dice（带安全截断，防止 NaN）----
        gt_change_safe = torch.clamp(gt_change.float(), 0.0, 1.0)   # <<< 关键：目标裁剪到 [0,1]
        bce  = F.binary_cross_entropy_with_logits(
            pred_change, gt_change_safe, pos_weight=self.pos_weight.to(pred_change.device)
        )
        dice = binary_dice_loss(pred_change, gt_change_safe)
        loss_change = 0.3 * bce + 0.7 * dice

        # ---- 构造语义掩码 ----
        valid1 = (gt_sem1 != self.ignore_index)
        valid2 = (gt_sem2 != self.ignore_index)
        # 变化/未变化掩码（且语义标签有效）
        change_mask    = (gt_change == 1).squeeze(1) & valid1 & valid2
        unchanged_mask = (gt_change == 0).squeeze(1) & valid1 & valid2 & (gt_sem1 == gt_sem2)

        # ---- 语义损失 ----
        if self.supervise_all_sem:
            # 全像素监督（忽略=ignore_index）
            loss_sem1_total = torch.tensor(0.0, device=pred_sem1.device)
            loss_sem2_total = torch.tensor(0.0, device=pred_sem2.device)
            if valid1.any():
                p1 = pred_sem1.permute(0, 2, 3, 1)[valid1]   # [N,C]
                g1 = gt_sem1[valid1]                          # [N]
                loss_sem1_total = self.sem_loss_t1(p1, g1)
            if valid2.any():
                p2 = pred_sem2.permute(0, 2, 3, 1)[valid2]
                g2 = gt_sem2[valid2]
                loss_sem2_total = self.sem_loss_t2(p2, g2)
        else:
            # 旧策略：变化区强监督 + 未变化区弱监督(η)
            loss_sem1 = torch.tensor(0.0, device=pred_sem1.device)
            loss_sem2 = torch.tensor(0.0, device=pred_sem2.device)
            if change_mask.any():
                p1 = pred_sem1.permute(0, 2, 3, 1)[change_mask]; g1 = gt_sem1[change_mask]
                p2 = pred_sem2.permute(0, 2, 3, 1)[change_mask]; g2 = gt_sem2[change_mask]
                loss_sem1 = self.sem_loss_t1(p1, g1)
                loss_sem2 = self.sem_loss_t2(p2, g2)

            loss_sem1_unch = torch.tensor(0.0, device=pred_sem1.device)
            loss_sem2_unch = torch.tensor(0.0, device=pred_sem2.device)
            if unchanged_mask.any() and self.eta_unch > 0:
                p1u = pred_sem1.permute(0, 2, 3, 1)[unchanged_mask]
                p2u = pred_sem2.permute(0, 2, 3, 1)[unchanged_mask]
                gu  = gt_sem1[unchanged_mask]
                loss_sem1_unch = self.sem_loss_t1(p1u, gu)
                loss_sem2_unch = self.sem_loss_t2(p2u, gu)

            loss_sem1_total = loss_sem1 + self.eta_unch * loss_sem1_unch
            loss_sem2_total = loss_sem2 + self.eta_unch * loss_sem2_unch

        # ---- KL 一致性（变化区 + 未变化区）----
        kl_chg = self.kl_divergence(pred_sem1, pred_sem2.detach(), mask=change_mask) + \
                 self.kl_divergence(pred_sem2, pred_sem1.detach(), mask=change_mask)

        kl_unch = torch.tensor(0.0, device=pred_sem1.device)
        if self.lambda_kl_unch > 0:
            kl_unch = self.kl_divergence(pred_sem1, pred_sem2.detach(), mask=unchanged_mask) + \
                      self.kl_divergence(pred_sem2, pred_sem1.detach(), mask=unchanged_mask)

        # ---- 总损失 ----
        total_loss = (self.alpha * loss_change +
                      self.beta  * loss_sem1_total +
                      self.gamma * loss_sem2_total +
                      self.lambda_kl      * kl_chg +
                      self.lambda_kl_unch * kl_unch)

        # 统一返回键，兼容现有打印/日志
        return total_loss, {
            'change': float(loss_change.detach().cpu()),
            'sem1':   float((loss_sem1_total if 'loss_sem1_total' in locals() else 0.0)),
            'sem2':   float((loss_sem2_total if 'loss_sem2_total' in locals() else 0.0)),
            'kl':     float((kl_chg + self.lambda_kl_unch * kl_unch).detach().cpu())
        }
