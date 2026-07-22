# train.py
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import MultiTaskChangeFormer
from dataset import SECOND_Dataset
from losses import MultiTaskLoss
from utils import create_optimizer, setup_ema
from evaluate import evaluate_all
from visualize import visualize_sample

# ========= 配置参数 ========= #
config = {
    'batch_size': 8,
    'lr': 1e-4,                 # 起始 LR
    'epochs': 100,
    'amp': False,               # 稳定起见先关 AMP
    'ema_decay': 0.999,
    'img_size': 512,
    'loss_weights': [0.3, 0.6, 0.5, 0.001],  # [chg, sem1, sem2, kl]
    'change_weight': 2.0,
    'save_dir': 'outputs/checkpoints',
    'data_root': 'data/Second/',
    'csv_log': 'outputs/train_log.csv',
    'use_focal': False,
    'ignore_index': 6,
    'sem_weight': None,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,
    'label_smoothing': 0.05,
    'num_classes': 6,
    'num_workers': 0 if os.name == 'nt' else 8,
    'persistent_workers': False,

    # ===== 学习率更快下降的关键参数 =====
    'cosine_fraction': 0.6,     # 用总步数的 30% 完成余弦衰减
    'eta_min_factor': 100,      # 最低 LR = lr / 100 （可调大到 200 以更低）

    # ===== 断电续训：从“最后一个模型”恢复 =====
    # 从 latest 权重恢复参数（仅权重 & EMA），优化器与调度器重建以使用新策略
    'resume': 'latest',                 # 'latest' | 'best' | 'auto' | 'none'
    'reset_optimizer_on_resume': True,  # 应用新 LR 策略，建议 True
    'reset_scheduler_on_resume': True,  # 应用新 LR 策略，建议 True
}

# ====== 将全网 BatchNorm2d 动态替换为 GroupNorm（小 batch 更稳） ======
def _choose_groups(num_channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g //= 2
    return max(1, g)

def convert_bn_to_gn(module: nn.Module, max_groups: int = 32):
    """
    递归替换：nn.BatchNorm2d -> nn.GroupNorm；保持 affine=True。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            groups = _choose_groups(num_channels, max_groups)
            gn = nn.GroupNorm(num_groups=groups, num_channels=num_channels, eps=child.eps, affine=True)
            with torch.no_grad():
                if child.affine:
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, max_groups=max_groups)
    return module

# ===== NaN 守门 + 批内检查 =====
def _is_finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()

def _dump_batch_stats(batch):
    def _summ(name, t):
        t_cpu = t.detach().cpu()
        if t_cpu.dtype.is_floating_point:
            vmin = float(torch.min(t_cpu))
            vmax = float(torch.max(t_cpu))
            mean = float(torch.mean(t_cpu))
            print(f"  {name}: shape={tuple(t_cpu.shape)}, dtype={t_cpu.dtype}, "
                  f"min={vmin:.6f}, max={vmax:.6f}, mean={mean:.6f}")
        else:
            vmin = int(torch.min(t_cpu))
            vmax = int(torch.max(t_cpu))
            uniq = torch.unique(t_cpu)
            uniq_list = [int(u) for u in uniq[:20].tolist()]
            print(f"  {name}: shape={tuple(t_cpu.shape)}, dtype={t_cpu.dtype}, "
                  f"min={vmin}, max={vmax}, uniq[:20]={uniq_list}")
    print("\n[NaN-Guard] Batch stats:")
    _summ('im1', batch['im1'])
    _summ('im2', batch['im2'])
    _summ('label1', batch['label1'])
    _summ('label2', batch['label2'])
    _summ('change_mask', batch['change_mask'])

def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_enabled = bool(config['amp'] and (device.type == 'cuda'))

    # ===== 模型 ===== #
    model = MultiTaskChangeFormer(n_classes_sem=7, decoder_embed_dim=512).to(device)
    model = convert_bn_to_gn(model, max_groups=32).to(device)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # ===== 数据 ===== #
    train_set = SECOND_Dataset(config['data_root'], split='train', img_size=config['img_size'])
    val_set   = SECOND_Dataset(config['data_root'], split='val',   img_size=config['img_size'])
    test_set  = SECOND_Dataset(config['data_root'], split='test',  img_size=config['img_size'])

    train_loader = DataLoader(
        train_set, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=(device.type == 'cuda'),
        persistent_workers=config['persistent_workers'] and config['num_workers'] > 0
    )
    val_loader = DataLoader(
        val_set, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=(device.type == 'cuda'),
        persistent_workers=config['persistent_workers'] and config['num_workers'] > 0
    )
    test_loader = DataLoader(
        test_set, batch_size=min(16, config['batch_size']), shuffle=False,
        num_workers=config['num_workers'], pin_memory=(device.type == 'cuda'),
        persistent_workers=config['persistent_workers'] and config['num_workers'] > 0
    )

    # ===== 优化器 & 调度器（更快下降版余弦退火） ===== #
    optimizer = create_optimizer(model, config['lr'], config['weight_decay'])

    total_steps  = config['epochs'] * max(1, len(train_loader))
    decay_steps  = max(1, int(total_steps * float(config.get('cosine_fraction', 1.0))))
    eta_min      = config['lr'] / float(config.get('eta_min_factor', 100))

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=decay_steps,   # 用更短的周期完成衰减
        eta_min=eta_min
    )

    # ===== 损失 ===== #
    criterion = MultiTaskLoss(
        alpha=config['loss_weights'][0],
        beta=config['loss_weights'][1],
        gamma=config['loss_weights'][2],
        lambda_kl=config['loss_weights'][3],
        change_weight=config.get('change_weight', None),
        sem_weight=config['sem_weight'],
        use_focal=config['use_focal'],
        ignore_index=config['ignore_index'],
        label_smoothing=config['label_smoothing'],
        supervise_all_sem=True,
        eta_unch=1.0,
        lambda_kl_unch=0.005
    )

    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    ema_model = setup_ema(model, config['ema_decay'])

    # ====== 断点恢复（从 latest 权重开始） ====== #
    start_epoch = 1
    best_f1 = 0.0
    global_step = 0  # 用于控制 scheduler 的步进窗口

    best_ckpt   = os.path.join(config['save_dir'], 'best_model.pth')
    latest_ckpt = os.path.join(config['save_dir'], 'latest_checkpoint.pth')

    def _load_ckpt(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        # 使用保存的模型参数（通常是 EMA 模型）
        model.load_state_dict(ckpt['state_dict'], strict=True)
        # 恢复 EMA
        if 'ema_state' in ckpt:
            ema_model.load_state_dict(ckpt['ema_state'])
        return ckpt

    resume_mode = (config.get('resume') or 'none').lower()
    if resume_mode == 'latest' and os.path.exists(latest_ckpt):
        ckpt = _load_ckpt(latest_ckpt)
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_f1 = float(ckpt.get('best_f1', 0.0))
        # 若你想完全延续旧优化器/调度器，把下面两项设为 False 即可
        if not config.get('reset_optimizer_on_resume', True) and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if not config.get('reset_scheduler_on_resume', True) and 'scheduler' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler'])
            except Exception:
                pass
        # 继续使用保存的步数；若重置调度器，则也可将其置 0 以按新策略重新退火
        global_step = int(ckpt.get('global_step', 0))
        if config.get('reset_scheduler_on_resume', True):
            global_step = 0  # 应用新的更快衰减策略，从头退火
        print(f"[Resume] from LATEST @ epoch {start_epoch-1} (best_f1={best_f1:.4f})")

    elif resume_mode == 'best' and os.path.exists(best_ckpt):
        ckpt = _load_ckpt(best_ckpt)
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_f1 = float(ckpt.get('best_f1', 0.0))
        global_step = int(ckpt.get('global_step', 0))
        if config.get('reset_scheduler_on_resume', True):
            global_step = 0
        print(f"[Resume] from BEST @ epoch {start_epoch-1} (best_f1={best_f1:.4f})")
    else:
        print("[Resume] start fresh.")

    # ===== CSV 日志 ===== #
    os.makedirs(os.path.dirname(config['csv_log']), exist_ok=True)
    if not os.path.exists(config['csv_log']):
        with open(config['csv_log'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss',
                'val_loss', 'val_chg', 'val_sem1', 'val_sem2', 'val_kl',
                'f1_best', 'iou_best', 'f1@0.5', 'iou@0.5',
                'miou1', 'miou2', 'miou_official', 'sek',
                'iou_nonchange', 'iou_change', 'best_thr'
            ])

    # ========== 训练循环 ========== #
    for epoch in range(start_epoch, config['epochs'] + 1):
        model.train()
        total_loss = 0.0

        for i, batch in enumerate(train_loader):
            im1 = batch['im1'].to(device, non_blocking=True)
            im2 = batch['im2'].to(device, non_blocking=True)
            targets = (
                batch['change_mask'].to(device, non_blocking=True),
                batch['label1'].to(device, non_blocking=True),
                batch['label2'].to(device, non_blocking=True)
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                outputs = model(im1, im2)

                # 前向输出 NaN 检查（logits）
                chg_logit, sem1_logit, sem2_logit = outputs
                if torch.isnan(chg_logit).any() or torch.isnan(sem1_logit).any() or torch.isnan(sem2_logit).any():
                    print("\n[NaN-Guard] NaN found in model outputs (logits). Skip this batch.")
                    _dump_batch_stats(batch)
                    continue  # 跳过坏 batch

                loss, loss_dict = criterion(outputs, targets)

                # Loss NaN 检查
                if (not _is_finite_tensor(loss)) or \
                   (not np.isfinite(loss_dict['change'])) or \
                   (not np.isfinite(loss_dict['sem1'])) or \
                   (not np.isfinite(loss_dict['sem2'])):
                    print("\n[NaN-Guard] Non-finite loss detected! Skip this batch.")
                    print(f"  loss: {float(loss.detach().cpu()) if torch.isfinite(loss) else 'NaN/Inf'}")
                    print(f"  change: {loss_dict['change']}, sem1: {loss_dict['sem1']}, sem2: {loss_dict['sem2']}, kl: {loss_dict['kl']}")
                    _dump_batch_stats(batch)
                    continue  # 跳过坏 batch

            # —— 梯度缩放 + 裁剪 ——
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if config['grad_clip'] and config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(optimizer)

            # ★ 只在衰减区间内 step；达到最低 lr 后保持不变
            global_step += 1
            if global_step <= decay_steps:
                scheduler.step()

            scaler.update()
            ema_model.update(model)

            total_loss += float(loss.detach().cpu())
            print(f"[Epoch {epoch} | Batch {i+1}/{len(train_loader)}] "
                  f"Loss: {total_loss/(i+1):.4f} (cur {float(loss):.4f}) | "
                  f"chg: {loss_dict['change']:.4f} | "
                  f"sem1: {loss_dict['sem1']:.4f} | "
                  f"sem2: {loss_dict['sem2']:.4f} | kl: {loss_dict['kl']:.4f}")

        avg_loss = total_loss / max(1, len(train_loader))
        # 打印所有 param_group 的 lr
        all_lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"[Epoch {epoch}] Avg Train Loss: {avg_loss:.4f} | lrs: " +
              ", ".join(f"{x:.6e}" for x in all_lrs))

        # ========== 验证 ========== #
        val_metrics = evaluate_all(
            ema_model.model, val_loader, device,
            num_classes=config['num_classes'],
            save_dir=None, criterion=criterion
        )

        print(("[Val @Epoch {ep}] "
               "F1(best)={f1:.4f} | IoU(best)={iou:.4f} | "
               "F1@0.5={f105:.4f} | IoU@0.5={iou05:.4f} | "
               "mIoU@T1={m1:.4f} | mIoU@T2={m2:.4f} | "
               "mIOU_official={moff:.4f} | SeK={sek:.4f} | "
               "Loss={vl:.4f}").format(
                   ep=epoch,
                   f1=val_metrics['f1_score'],
                   iou=val_metrics['iou'],
                   f105=val_metrics.get('f1@0.5', 0.0),
                   iou05=val_metrics.get('iou@0.5', 0.0),
                   m1=val_metrics['miou1'],
                   m2=val_metrics['miou2'],
                   moff=val_metrics.get('miou_official', 0.0),
                   sek=val_metrics.get('sek', 0.0),
                   vl=val_metrics.get('val_loss', 0.0)
               ))

        # === 写入 CSV === #
        with open(config['csv_log'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, avg_loss,
                val_metrics.get('val_loss', 0.0),
                val_metrics.get('val_chg', 0.0), val_metrics.get('val_sem1', 0.0),
                val_metrics.get('val_sem2', 0.0), val_metrics.get('val_kl', 0.0),
                val_metrics['f1_score'], val_metrics['iou'],
                val_metrics.get('f1@0.5', 0.0), val_metrics.get('iou@0.5', 0.0),
                val_metrics['miou1'], val_metrics['miou2'],
                val_metrics.get('miou_official', 0.0), val_metrics.get('sek', 0.0),
                val_metrics.get('iou_nonchange', 0.0), val_metrics.get('iou_change', 0.0),
                val_metrics.get('best_thr', 0.0)
            ])

        # === 保存 best（按 F1） === #
        best_so_far = max(best_f1, val_metrics['f1_score'])
        if val_metrics['f1_score'] >= best_so_far:
            best_f1 = best_so_far
            save_checkpoint({
                'epoch': epoch,
                'state_dict': ema_model.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ema_state': ema_model.state_dict(),
                'best_f1': best_f1,
                'global_step': global_step,
                'decay_steps': decay_steps,
                'total_steps': total_steps
            }, filename=os.path.join(config['save_dir'], 'best_model.pth'))

            visualize_sample(ema_model.model, test_loader, device, epoch,
                             save_root='outputs/best_sample_vis', num_samples=1)

        # === 每轮保存 latest checkpoint === #
        save_checkpoint({
            'epoch': epoch,
            'state_dict': ema_model.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'ema_state': ema_model.state_dict(),
            'best_f1': best_f1,
            'global_step': global_step,
            'decay_steps': decay_steps,
            'total_steps': total_steps
        }, filename=os.path.join(config['save_dir'], 'latest_checkpoint.pth'))

if __name__ == "__main__":
    main()
