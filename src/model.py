# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ChangeFormer import ChangeFormerV6
from timm.layers import DropPath

def _make_gn(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """
    选择一个能整除 num_channels 的 group 数，优先 32/16/8/...，至少为 1。
    """
    g = min(max_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, num_channels)

class MultiTaskChangeFormer(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, n_classes_sem=7,
                 encoder_name='mit_b2', decoder_embed_dim=512):
        super().__init__()

        # ===== 共享编码器（Siamese） =====
        cf = ChangeFormerV6(
            input_nc=input_nc,
            output_nc=output_nc,
            embed_dim=decoder_embed_dim
        )
        self.encoder = cf.encoder          # EncoderTransformer_v3 (LN-based)
        self.change_decoder = cf.decoder   # DecoderTransformer_v3

        # 编码器每层通道
        self.embed_dims = list(self.encoder.embed_dims)  # [64, 128, 320, 512]
        self.last_ch = self.embed_dims[-1]               # 512

        # 语义解码（轻量 FPN，吃多尺度特征）
        self.sem_decoder_t1 = SemHeadFPN(self.embed_dims, out_ch=128, n_classes=n_classes_sem)
        self.sem_decoder_t2 = SemHeadFPN(self.embed_dims, out_ch=128, n_classes=n_classes_sem)

        # ===== 跨时相增强 & 残差融合（最后尺度）=====
        self.cross_attn = CrossAttentionModule(self.last_ch)
        self.residual_fusion = ResidualFusionModule(self.last_ch)

        # ===== 差异图引导（最后尺度）=====
        mid = max(16, self.last_ch // 2)
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(input_nc, mid, kernel_size=3, padding=1),
            _make_gn(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, self.last_ch, kernel_size=3, padding=1),
            _make_gn(self.last_ch),
            nn.ReLU(inplace=True)
        )

        # ===== 多尺度特征融合（逐层 1×1，对位融合）=====
        self.feature_fusion_conv = nn.ModuleList([
            nn.Conv2d(2 * c, c, kernel_size=1) for c in self.embed_dims
        ])

    def forward(self, im1, im2):
        # Step 1: 原图差异 → 最后尺度辅助
        diff = torch.abs(im1 - im2)                 # [B,3,H,W]
        diff_feat = self.diff_encoder(diff)         # [B,512,h32,w32]

        # Step 2: 编码器多尺度特征
        f1_all = self.encoder(im1)  # [C1@1/4, C2@1/8, C3@1/16, C4@1/32]
        f2_all = self.encoder(im2)

        f1_last = f1_all[-1]  # [B,512,h32,w32]
        f2_last = f2_all[-1]

        # Step 3: 交叉注意力增强 + 残差融合（最后尺度）
        f1_enhanced, f2_passthrough = self.cross_attn(f1_last, f2_last)  # 仅增强 f1
        fused_last = self.residual_fusion(f1_enhanced, f2_passthrough)

        # Step 4: 差异引导融合（最后尺度）
        diff_feat = F.interpolate(diff_feat, size=fused_last.shape[-2:], mode='bilinear', align_corners=False)
        fused_last = fused_last + diff_feat

        # Step 5: 多尺度对位融合（各尺度拼接压缩 + 在最后尺度注入 fused_last）
        f1_all_new, f2_all_new = [], []
        for i, (f1_i, f2_i) in enumerate(zip(f1_all, f2_all)):
            x_cat = torch.cat([f1_i, f2_i], dim=1)            # [B,2*C_i,H_i,W_i]
            fused_i = self.feature_fusion_conv[i](x_cat)      # [B,C_i,H_i,W_i]
            if i == len(self.embed_dims) - 1:                 # 最后一层再加 fused_last
                fused_i = fused_i + fused_last
            f1_all_new.append(f1_i + fused_i)                 # 残差注入，保留差异
            f2_all_new.append(f2_i + fused_i)

        # Step 6: 变化检测（原 ChangeFormer 解码器）
        change_pred_list = self.change_decoder(f1_all_new, f2_all_new)
        change_pred = change_pred_list[-1]  # [B,1,H,W] (logits)

        # Step 7: 语义分割（FPN）
        sem_pred_t1 = self.sem_decoder_t1(f1_all_new)  # [B,7,H,W]
        sem_pred_t2 = self.sem_decoder_t2(f2_all_new)  # [B,7,H,W]

        return change_pred, sem_pred_t1, sem_pred_t2


# ===== 轻量 FPN 语义头 =====
class SemHeadFPN(nn.Module):
    def __init__(self, in_channels_list, out_ch=128, n_classes=7):
        super().__init__()
        C1, C2, C3, C4 = in_channels_list
        # 1x1 侧向
        self.lat1 = nn.Conv2d(C1, out_ch, 1)
        self.lat2 = nn.Conv2d(C2, out_ch, 1)
        self.lat3 = nn.Conv2d(C3, out_ch, 1)
        self.lat4 = nn.Conv2d(C4, out_ch, 1)
        # 平滑3x3（加 GN）
        self.s3 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1), _make_gn(out_ch), nn.ReLU(inplace=True))
        self.s2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1), _make_gn(out_ch), nn.ReLU(inplace=True))
        self.s1 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1), _make_gn(out_ch), nn.ReLU(inplace=True))
        self.cls = nn.Conv2d(out_ch, n_classes, 1)

    def forward(self, feats):  # feats = [C1(1/4), C2(1/8), C3(1/16), C4(1/32)]
        c1, c2, c3, c4 = feats
        p4 = self.lat4(c4)
        p3 = self.s3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False))
        p2 = self.s2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False))
        p1 = self.s1(self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False))
        # C1 / P1 处为 1/4 分辨率，恢复到原图需 ×4
        logits = self.cls(F.interpolate(p1, scale_factor=4, mode='bilinear', align_corners=False))
        return logits


# ===== 改进的交叉注意力（空间 + 通道）=====
class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 空间注意力：用 x2 引导 x1
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 8, 1, 1),
            nn.Sigmoid()
        )
        # 通道注意力：用 x2 引导 x1
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 8, embed_dim, 1),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.drop_path = DropPath(0.05)  # 更稳

    def forward(self, x1, x2):
        spatial_mask = self.spatial_attn(x2)
        channel_mask = self.channel_attn(x2)
        out = spatial_mask * x1 + channel_mask * x1
        enhanced_x1 = x1 + self.drop_path(self.gamma * out)
        return enhanced_x1, x2


# ===== 改进的残差融合（门控 + 差分）=====
class ResidualFusionModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1),
            _make_gn(embed_dim), nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            _make_gn(embed_dim), nn.ReLU(inplace=True)
        )
        self.diff_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            _make_gn(embed_dim), nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            _make_gn(embed_dim), nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        fused = self.fusion_conv(torch.cat([x1, x2], dim=1))   # 语义融合
        diff  = self.diff_conv(torch.abs(x1 - x2))             # 显式差分
        out   = self.out_conv(torch.cat([fused, diff], dim=1))
        return out
