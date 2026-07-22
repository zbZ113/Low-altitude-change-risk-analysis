# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import math
import traceback
import textwrap
import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# —— 让 Matplotlib 图片支持中文（尽量不出方块）
plt.rcParams["font.sans-serif"] = [
    "SimHei", "Source Han Sans CN", "Noto Sans CJK SC", "Microsoft YaHei",
    "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

from PyQt5 import QtCore, QtGui, QtWidgets

# === 项目内模块 ===
from model import MultiTaskChangeFormer
from visualize import colorize_mask, compute_change_matrix, plot_change_matrix

# -----------------------------
# 形态学依赖与工具
# -----------------------------
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
try:
    from scipy import ndimage as ndi
    _HAS_NDI = True
except Exception:
    _HAS_NDI = False

def _apply_morph(binary_np: np.ndarray, mode: str, ksize: int, iters: int) -> np.ndarray:
    """
    binary_np: uint8 {0,1}
    mode: 'none' | 'open' | 'close' | 'open_close' | 'close_open'
    ksize: 奇数（>=1），iters: >=1
    """
    if mode == 'none' or ksize < 1 or iters < 1:
        return (binary_np > 0).astype(np.uint8)

    b = (binary_np > 0).astype(np.uint8)

    if _HAS_CV2:
        if ksize % 2 == 0: ksize += 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        def _open(x):  return cv2.morphologyEx(x, cv2.MORPH_OPEN,  ker, iterations=iters)
        def _close(x): return cv2.morphologyEx(x, cv2.MORPH_CLOSE, ker, iterations=iters)
    elif _HAS_NDI:
        rad = ksize // 2
        se = np.ones((2*rad+1, 2*rad+1), np.uint8)
        def _open(x):  return ndi.binary_opening(x, structure=se, iterations=iters).astype(np.uint8)
        def _close(x): return ndi.binary_closing(x, structure=se, iterations=iters).astype(np.uint8)
    else:
        # 兜底近似
        from numpy.lib.stride_tricks import as_strided
        r = ksize // 2
        pad = np.pad(b, r, mode='edge')
        H, W = pad.shape
        sh = (H-ksize+1, ksize, W-ksize+1, ksize)
        st = pad.strides + pad.strides
        def pool(x, op):
            windows = as_strided(x, shape=sh, strides=st)
            return (windows.max(axis=(1,3)) if op=='max' else windows.min(axis=(1,3)))
        def _erode(x):
            y = pad
            for _ in range(iters): y = np.pad(pool(y, 'min'), r, mode='edge')
            return y[r:-r, r:-r]
        def _dilate(x):
            y = pad
            for _ in range(iters): y = np.pad(pool(y, 'max'), r, mode='edge')
            return y[r:-r, r:-r]
        def _open(x):  return _dilate(_erode(x))
        def _close(x): return _erode(_dilate(x))

    if mode == 'open':
        b2 = _open(b)
    elif mode == 'close':
        b2 = _close(b)
    elif mode == 'open_close':
        b2 = _close(_open(b))
    elif mode == 'close_open':
        b2 = _open(_close(b))
    else:
        b2 = b
    return (b2 > 0).astype(np.uint8)

# -----------------------------
# BN→GN 兼容：与训练一致的递归替换
# -----------------------------
def _choose_groups(num_channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g //= 2
    return max(1, g)

def convert_bn_to_gn(module: nn.Module, max_groups: int = 32):
    """
    递归替换：nn.BatchNorm2d -> nn.GroupNorm；保持 affine=True。
    与训练脚本的实现一致，保证权重兼容。 见 train.py。"""
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

# -----------------------------
# 工具与数据结构
# -----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def now_tag():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

@dataclass
class InferenceOutputs:
    # 原始可视化（letterbox到512的）
    im1_vis: np.ndarray         # HxWx3 uint8
    im2_vis: np.ndarray         # HxWx3 uint8
    # 预测缓存（512对齐）
    chg_prob: np.ndarray        # HxW float32 in [0,1]
    sem1_logits: np.ndarray     # HxWxC float32
    sem2_logits: np.ndarray     # HxWxC float32
    # 元数据
    best_thr: float
    valid_mask: Optional[np.ndarray] = None  # HxW uint8 {0,1}
    class_names: Tuple[str, ...] = ("Low veg", "Non-veg", "Tree", "Water", "Building", "Playground")  # 0..5

def normalize_ai_text(markdown: str) -> str:
    """
    将大模型返回的 Markdown/列表样式转换为更适合 PDF/人读的纯文本。
    """
    s = markdown or ""
    s = re.sub(r"```.*?```", "", s, flags=re.S)  # 代码块
    s = s.replace("`", "")                        # 行内代码
    s = re.sub(r"^\s*#{1,6}\s*", "", s, flags=re.M)  # 标题 '#'
    s = s.replace("**", "").replace("__", "").replace("*", "")  # 粗斜体
    s = re.sub(r"^(\s*)(\d+)\.\s+", r"\1\2) ", s, flags=re.M)   # 有序列表
    s = re.sub(r"^(\s*)[-•·]\s+", r"\1· ", s, flags=re.M)       # 无序列表
    lines = [ln.rstrip() for ln in s.splitlines()]
    out, prev_blank = [], False
    for ln in lines:
        if not ln.strip():
            if not prev_blank: out.append("")
            prev_blank = True
        else:
            out.append(ln); prev_blank = False
    return "\n".join(out).strip()

# -----------------------------
# 影像 letterbox 到 512
# -----------------------------
def letterbox_to_square(img_np: np.ndarray, target=512, pad_value=255) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = img_np.shape[:2]
    scale = min(target / h, target / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = np.array(Image.fromarray(img_np).resize((new_w, new_h), Image.BILINEAR), dtype=np.uint8)
    canvas = np.full((target, target, 3), pad_value, dtype=np.uint8)
    top = (target - new_h) // 2
    left = (target - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    meta = dict(scale=scale, top=top, left=left, new_h=new_h, new_w=new_w)
    return canvas, meta

def to_tensor_normalized(img_np_letterboxed: np.ndarray) -> torch.Tensor:
    x = img_np_letterboxed.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    return x

def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    x = t[0].permute(1, 2, 0).detach().cpu().numpy()
    x = ((x + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return x

def apply_colormap_to_prob(prob: np.ndarray, cmap_name="jet") -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(prob)[:, :, :3]
    return (colored * 255).astype(np.uint8)

def overlay(base_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha=0.5) -> np.ndarray:
    base = base_rgb.astype(np.float32)
    over = overlay_rgb.astype(np.float32)
    out = (1 - alpha) * base + alpha * over
    return out.clip(0, 255).astype(np.uint8)

def save_image(path: str, img: np.ndarray):
    Image.fromarray(img).save(path)

def logits_argmax_numpy(logits_hw_c: np.ndarray) -> np.ndarray:
    return np.argmax(logits_hw_c, axis=-1).astype(np.int32)

def mask_unaffected_to_class6(sem_pred: np.ndarray, chg_bin: np.ndarray) -> np.ndarray:
    out = sem_pred.copy()
    out[chg_bin == 0] = 6
    return out

def compute_transition_matrix_from_preds(a_hw: np.ndarray, b_hw: np.ndarray, chg_bin: np.ndarray) -> np.ndarray:
    valid = (chg_bin == 1) & (a_hw != 6) & (b_hw != 6)
    a = a_hw[valid].ravel()
    b = b_hw[valid].ravel()
    M = np.zeros((6, 6), dtype=np.int64)
    if a.size > 0:
        np.add.at(M, (a, b), 1)
    return M

def make_change_category_map(a_hw: np.ndarray, b_hw: np.ndarray, chg_bin: np.ndarray) -> np.ndarray:
    H, W = a_hw.shape
    code = np.full((H, W), 255, dtype=np.uint8)
    valid = (chg_bin == 1) & (a_hw != 6) & (b_hw != 6) & (a_hw != b_hw)
    code[valid] = (a_hw[valid] * 6 + b_hw[valid]).astype(np.uint8)
    return code

def colorize_change_category_map(code_map: np.ndarray) -> np.ndarray:
    H, W = code_map.shape
    base_colors = [
        (166,206,227),(31,120,180),(178,223,138),(51,160,44),(251,154,153),(227,26,28),
        (253,191,111),(255,127,0),(202,178,214),(106,61,154),(255,255,153),(177,89,40),
        (141,211,199),(255,255,179),(190,186,218),(251,128,114),(128,177,211),(253,180,98),
        (179,222,105),(252,205,229),(217,217,217),(188,128,189),(204,235,197),(255,237,111),
        (128,0,0),(0,128,128),(0,0,128),(128,128,0),(0,128,0),(128,0,128),
        (255,0,255),(0,255,255),(255,165,0),(154,205,50),(70,130,180),(199,21,133)
    ]
    palette = np.array(base_colors, dtype=np.uint8)
    out = np.full((H, W, 3), 255, dtype=np.uint8)
    mask = code_map != 255
    if np.any(mask):
        out[mask] = palette[code_map[mask] % 36]
    return out

# -----------------------------
# 模型推理线程（含 BN→GN 兼容加载）
# -----------------------------
class ModelRunner(QtCore.QThread):
    finished = QtCore.pyqtSignal(object, object)  # (InferenceOutputs or None, Exception or None)

    def __init__(self, im1_path: str, im2_path: str, ckpt_path: str, parent=None):
        super().__init__(parent)
        self.im1_path = im1_path
        self.im2_path = im2_path
        self.ckpt_path = ckpt_path

    def _load_checkpoint_with_bn_gn_compat(self, model: nn.Module, ckpt_path: str, device: torch.device) -> float:
        """
        以 strict=True 加载；若发现 BN/GN 不匹配（典型是缺失 running_mean/var），则先在 CPU 上 BN→GN，
        再 strict=True 加载；最后统一 model.to(device)。
        返回：best_thr（若权重未保存该字段，回退 0.5）
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"未找到权重文件：{ckpt_path}")

        # —— 为了兼容最大，权重先加载到 CPU —— 
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get("state_dict", ckpt)

        def _try_strict():
            model.load_state_dict(state_dict, strict=True)

        try:
            _try_strict()
        except Exception as e:
            msg = str(e)
            print(f"[Checkpoint] strict=True 加载失败：{repr(e)}")
            # 常见于：模型里是 BN，但权重里是 GN（或相反），缺失 running_mean/var
            if "running_mean" in msg or "running_var" in msg or "GroupNorm" in msg or "BatchNorm" in msg:
                # 在 CPU 上递归 BN→GN
                convert_bn_to_gn(model, max_groups=32)
                # 再次 strict=True 加载
                _try_strict()
                print("已进行 BN→GN 转换，并以 strict=True 加载成功。")
            else:
                raise

        # **统一把模型移动到指定 device（尤其确保新替换的 GN 层不遗留在 CPU）**
        model.to(device)

        # 读取 best_thr（没有就回退 0.5）
        return float(ckpt.get("best_thr", 0.5))

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            use_amp = (device.type == 'cuda')

            # 读取图像
            im1_np = np.array(Image.open(self.im1_path).convert("RGB"))
            im2_np = np.array(Image.open(self.im2_path).convert("RGB"))

            # letterbox to 512
            im1_lb, meta1 = letterbox_to_square(im1_np, target=512, pad_value=255)
            im2_lb, meta2 = letterbox_to_square(im2_np, target=512, pad_value=255)

            # 归一化
            t1 = to_tensor_normalized(im1_lb)
            t2 = to_tensor_normalized(im2_lb)

            # 构建模型（先在 CPU 上，以便做 BN→GN 转换）
            model = MultiTaskChangeFormer(
                input_nc=3, output_nc=1, n_classes_sem=7, decoder_embed_dim=512
            )  # 先不 .to(device)

            # —— 兼容加载（含 BN→GN）——
            best_thr = self._load_checkpoint_with_bn_gn_compat(model, self.ckpt_path, device)

            model.eval()
            t1 = t1.to(device, non_blocking=True)
            t2 = t2.to(device, non_blocking=True)

            # 推理
            with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
                chg_logits, sem1_logits, sem2_logits = model(t1, t2)

            # 上采样到 512×512（一般已是512，这里稳妥处理）
            chg_prob = torch.sigmoid(
                F.interpolate(chg_logits, size=(512, 512), mode='bilinear', align_corners=False)
            )[0, 0].detach().cpu().numpy().astype(np.float32)
            sem1_logits_up = F.interpolate(sem1_logits, size=(512, 512), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
            sem2_logits_up = F.interpolate(sem2_logits, size=(512, 512), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

            # 反归一化可视图
            im1_vis = tensor_to_uint8_img(t1)
            im2_vis = tensor_to_uint8_img(t2)

            # 有效域（两时相 letterbox 的交集）
            def _meta_mask(meta):
                m = np.zeros((512, 512), np.uint8)
                t, l, h, w = meta["top"], meta["left"], meta["new_h"], meta["new_w"]
                m[t:t+h, l:l+w] = 1
                return m
            mask1 = _meta_mask(meta1)
            mask2 = _meta_mask(meta2)
            valid_mask = (mask1 & mask2).astype(np.uint8)

            outputs = InferenceOutputs(
                im1_vis=im1_vis,
                im2_vis=im2_vis,
                chg_prob=chg_prob,
                sem1_logits=sem1_logits_up,
                sem2_logits=sem2_logits_up,
                best_thr=best_thr,
                valid_mask=valid_mask
            )
            self.finished.emit(outputs, None)
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            self.finished.emit(None, e)

# -----------------------------
# LLM 异步调用线程（OpenAI 兼容 / SiliconFlow）
# -----------------------------
class LLMRunner(QtCore.QThread):
    finished = QtCore.pyqtSignal(str, object)  # (text, err)

    def __init__(self, base_url: str, model: str, api_key: str,
                 system_prompt: str, user_prompt: str,
                 temperature: float = 0.2, max_tokens: int = 800, parent=None):
        super().__init__(parent)
        self.base_url = base_url.strip() or "https://api.openai.com/v1"
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url.rstrip("/") + "/v1"
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def run(self):
        try:
            text = self._call_llm()
            self.finished.emit(text, None)
        except Exception as e:
            self.finished.emit("", e)

    def _call_llm(self) -> str:
        endpoint = self.base_url + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt}
            ]
        }
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM API 调用失败：HTTP {resp.status_code} - {resp.text[:300]}")
        data = resp.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        text = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()  # SiliconFlow 支持
        if reasoning:
            text = "【推理过程】\n" + reasoning + "\n\n【结论】\n" + text
        return text if text else json.dumps(data, ensure_ascii=False)

# -----------------------------
# PDF 报告构建（内嵌中文字体）
# -----------------------------
class PdfReportBuilder:
    def __init__(self, out_pdf_path: str, font_path: Optional[str] = None):
        self.out_pdf_path = out_pdf_path
        self.font_path = font_path

        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas

        self.pdfmetrics = pdfmetrics
        self.TTFont = TTFont
        self.UnicodeCIDFont = UnicodeCIDFont
        self.A4 = A4
        self.mm = mm
        self.canvas_cls = canvas

        self.font_name = "CNFont"
        ok = False

        def try_register_ttc(ttc_path: str, max_try: int = 8) -> bool:
            for idx in range(max_try):
                try:
                    pdfmetrics.registerFont(TTFont(self.font_name, ttc_path, subfontIndex=idx))
                    return True
                except Exception:
                    continue
            return False

        if font_path and os.path.exists(font_path):
            try:
                ext = os.path.splitext(font_path)[1].lower()
                if ext == ".ttc":
                    ok = try_register_ttc(font_path)
                elif ext in (".ttf", ".otf"):
                    pdfmetrics.registerFont(TTFont(self.font_name, font_path))
                    ok = True
            except Exception as e:
                print(f"[PDF] 指定字体注册失败：{e}")

        if not ok:
            try:
                pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
                self.font_name = "STSong-Light"
                ok = True
                print("[PDF] 使用内建 CID 字体 STSong-Light 作为中文字体。")
            except Exception as e:
                print(f"[PDF] CID 字体注册失败：{e}，将回退 Helvetica（中文将无法显示）")
                self.font_name = "Helvetica"

    def build(self, pages: Dict[str, str], info: Dict[str, Any]):
        c = self.canvas_cls.Canvas(self.out_pdf_path, pagesize=self.A4)
        W, H = self.A4
        c.setAuthor("Change Detection GUI")
        c.setTitle("地表变化检测分析报告")

        def draw_title(text):
            c.setFont(self.font_name, 18)
            c.drawString(20 * self.mm, (H - 25 * self.mm), text)

        def draw_text(text, x_mm, y_mm, size=11):
            c.setFont(self.font_name, size)
            c.drawString(x_mm * self.mm, y_mm * self.mm, text)

        def draw_img_center(img_path, top_y_mm, width_mm=170):
            if not img_path or not os.path.exists(img_path):
                return
            img = Image.open(img_path)
            iw, ih = img.size
            target_w = width_mm * self.mm
            scale = target_w / iw
            target_h = ih * scale
            x = (W - target_w) / 2
            y = (top_y_mm * self.mm) - target_h
            c.drawInlineImage(img_path, x, y, width=target_w, height=target_h)

        # --- 封面 ---
        draw_title("地表变化检测与变化类别分析报告")
        draw_text(f"项目：MultiTaskChangeFormer", 20, 260)
        draw_text(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", 20, 250)
        draw_text(f"权重文件：{os.path.basename(info.get('ckpt_path',''))}", 20, 240)
        draw_text(f"best_thr：{info.get('best_thr', 0.5):.3f}", 20, 230)
        draw_text(f"像元分辨率(m/px)：{info.get('gsp_m_per_px','未设置')}", 20, 220)
        draw_img_center(pages.get("cover"), top_y_mm=200, width_mm=170)
        c.showPage()

        # --- 输入页面 ---
        draw_title("输入影像")
        draw_img_center(pages.get("inputs"), top_y_mm=260, width_mm=170)
        c.showPage()

        # --- 变化图与热力图 ---
        draw_title("变化检测结果（概率图/阈值图/热力图叠加）")
        draw_img_center(pages.get("change"), top_y_mm=260, width_mm=170)
        c.showPage()

        # --- 语义图（T1/T2）与变化类别图 ---
        draw_title("语义分割与变化类别图")
        draw_img_center(pages.get("semantics"), top_y_mm=260, width_mm=170)
        c.showPage()

        # --- 转移矩阵与统计 ---
        draw_title("变化转移矩阵与统计")
        draw_img_center(pages.get("matrix"), top_y_mm=260, width_mm=170)
        draw_img_center(pages.get("bars"), top_y_mm=120, width_mm=170)
        c.showPage()

        # --- AI 分析（若提供） ---
        ai_text = normalize_ai_text((info.get("ai_analysis") or "").strip())
        if ai_text:
            draw_title("AI 变化趋势与归因分析（自动生成）")
            y = 260
            lines = ai_text.splitlines()
            for ln in lines:
                for seg in textwrap.wrap(ln, width=42):
                    draw_text(seg, 20, y, size=11)
                    y -= 7.6
                    if y < 30:
                        c.showPage()
                        draw_title("AI 变化趋势与归因分析（续）")
                        y = 260
            c.showPage()

        # --- 元数据页 ---
        draw_title("实验参数与元数据")
        y = 260
        for k, v in info.items():
            if k == "ai_analysis":
                continue
            draw_text(f"{k}: {v}", 20, y)
            y -= 8
            if y < 30:
                c.showPage()
                draw_title("实验参数与元数据（续）")
                y = 260

        c.save()

# -----------------------------
# 主窗口
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("变化检测与变化类别分析 GUI")
        self.session_dir = None
        self.outputs: Optional[InferenceOutputs] = None
        self.ckpt_path = "outputs/checkpoints/best_model.pth"
        self.font_path = "fonts/SourceHanSansCN-Regular.otf"
        self.gsp_m_per_px = ""  # 像元分辨率
        self.class_names_zh: Tuple[str, ...] = ("低矮植被", "非植被", "乔木", "水体", "建筑", "操场")
        self._last_render_meta: Dict[str, Any] = {}
        self._label_imgs: Dict[QtWidgets.QLabel, np.ndarray] = {}

        # LLM 默认（SiliconFlow 预置）
        self.llm_base_url = "https://api.siliconflow.cn/v1"
        self.llm_model = "deepseek-ai/DeepSeek-V3"
        self.llm_api_key = ""
        self.llm_temperature = 0.2
        self.llm_max_tokens = 1000
        self.ai_analysis_text = ""

        self.setStyleSheet("""
            QWidget{font-size:14px;}
            QGroupBox{font-size:15px; font-weight:bold;}
            QPushButton{font-size:14px;}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {font-size:14px;}
            QTextEdit{font-size:14px;}
            QTabBar::tab {font-size:14px; padding:6px 12px;}
        """)

        self._build_ui()

    # --- UI 构建 ---
    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # 左侧控制面板
        ctrl_widget = QtWidgets.QWidget(self)
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_widget)

        self.im1_edit = QtWidgets.QLineEdit(self)
        self.im2_edit = QtWidgets.QLineEdit(self)
        self.ckpt_edit = QtWidgets.QLineEdit(self)
        self.ckpt_edit.setText(self.ckpt_path)
        self.font_edit = QtWidgets.QLineEdit(self)
        self.font_edit.setText(self.font_path)
        self.gsp_edit = QtWidgets.QLineEdit(self)
        self.gsp_edit.setPlaceholderText("像元分辨率（m/px），可选")

        btn_im1 = QtWidgets.QPushButton("选择时相1图像")
        btn_im2 = QtWidgets.QPushButton("选择时相2图像")
        btn_ckpt = QtWidgets.QPushButton("选择模型权重")
        btn_font = QtWidgets.QPushButton("选择中文字体")

        btn_run = QtWidgets.QPushButton("运行推理")
        btn_pdf = QtWidgets.QPushButton("导出PDF报告")

        # —— 阈值（0~1）+ 数字框 + 恢复按钮
        thr_layout = QtWidgets.QHBoxLayout()
        thr_label = QtWidgets.QLabel("变化阈值：")
        self.thr_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.thr_slider.setMinimum(0)
        self.thr_slider.setMaximum(1000)
        self.thr_slider.setSingleStep(1)
        self.thr_slider.setPageStep(10)
        self.thr_slider.setValue(500)
        self.thr_slider.setMinimumWidth(260)
        self.thr_slider.setTickInterval(50)
        self.thr_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)

        self.thr_spin = QtWidgets.QDoubleSpinBox()
        self.thr_spin.setDecimals(3)
        self.thr_spin.setRange(0.000, 1.000)
        self.thr_spin.setSingleStep(0.001)
        self.thr_spin.setValue(0.500)

        btn_thr_best = QtWidgets.QPushButton("恢复 best_thr")

        thr_layout.addWidget(thr_label)
        thr_layout.addWidget(self.thr_slider, 1)
        thr_layout.addWidget(self.thr_spin)
        thr_layout.addWidget(btn_thr_best)

        # —— 形态学控件
        morph_group = QtWidgets.QGroupBox("形态学后处理")
        morph_v = QtWidgets.QVBoxLayout(morph_group)
        row1 = QtWidgets.QHBoxLayout()
        row2 = QtWidgets.QHBoxLayout()

        row1.addWidget(QtWidgets.QLabel("操作："))
        self.morph_mode = QtWidgets.QComboBox()
        self.morph_mode.addItems(["无", "开运算", "闭运算", "先开后闭", "先闭后开"])
        self.morph_mode.setCurrentIndex(0)
        row1.addWidget(self.morph_mode, 1)

        row1.addWidget(QtWidgets.QLabel("核大小："))
        self.morph_ks = QtWidgets.QSpinBox()
        self.morph_ks.setRange(1, 51)
        self.morph_ks.setSingleStep(2)
        self.morph_ks.setValue(5)
        row1.addWidget(self.morph_ks)

        row2.addWidget(QtWidgets.QLabel("迭代："))
        self.morph_iter = QtWidgets.QSpinBox()
        self.morph_iter.setRange(1, 5)
        self.morph_iter.setValue(1)
        row2.addWidget(self.morph_iter)
        row2.addStretch(1)

        morph_v.addLayout(row1)
        morph_v.addLayout(row2)

        # 状态栏信息
        self.status_box = QtWidgets.QTextEdit(self)
        self.status_box.setReadOnly(True)
        self.status_box.setFixedHeight(120)

        # 左侧布局
        for w in [
            QtWidgets.QLabel("时相1路径："), self.im1_edit, btn_im1,
            QtWidgets.QLabel("时相2路径："), self.im2_edit, btn_im2,
            QtWidgets.QLabel("模型权重："), self.ckpt_edit, btn_ckpt,
            QtWidgets.QLabel("中文字体："), self.font_edit, btn_font,
            QtWidgets.QLabel("像元分辨率(m/px)："), self.gsp_edit,
            btn_run, btn_pdf,
        ]:
            ctrl_layout.addWidget(w)
        ctrl_layout.addLayout(thr_layout)
        ctrl_layout.addWidget(morph_group)
        ctrl_layout.addWidget(self.status_box)
        ctrl_layout.addStretch(1)

        # 右侧展示区：Tab
        tabs = QtWidgets.QTabWidget(self)
        self.tab_vis = QtWidgets.QWidget(self)
        self.tab_stat = QtWidgets.QWidget(self)
        self.tab_ai = QtWidgets.QWidget(self)

        # 可视化页：九宫格
        grid = QtWidgets.QGridLayout(self.tab_vis)
        self.lbl_im1 = QtWidgets.QLabel(); self._prep_label(self.lbl_im1)
        self.lbl_im2 = QtWidgets.QLabel(); self._prep_label(self.lbl_im2)
        self.lbl_chg_prob = QtWidgets.QLabel(); self._prep_label(self.lbl_chg_prob)
        self.lbl_chg_bin = QtWidgets.QLabel(); self._prep_label(self.lbl_chg_bin)
        self.lbl_sem1 = QtWidgets.QLabel(); self._prep_label(self.lbl_sem1)
        self.lbl_sem2 = QtWidgets.QLabel(); self._prep_label(self.lbl_sem2)
        self.lbl_catmap = QtWidgets.QLabel(); self._prep_label(self.lbl_catmap)
        self.lbl_matrix = QtWidgets.QLabel(); self._prep_label(self.lbl_matrix)
        self.lbl_heat = QtWidgets.QLabel(); self._prep_label(self.lbl_heat)

        grid.addWidget(self._titled(self.lbl_im1, "时相1图像"), 0, 0)
        grid.addWidget(self._titled(self.lbl_im2, "时相2图像"), 0, 1)
        grid.addWidget(self._titled(self.lbl_chg_prob, "变化概率图"), 0, 2)

        grid.addWidget(self._titled(self.lbl_sem1, "语义T1（变化区）"), 1, 0)
        grid.addWidget(self._titled(self.lbl_sem2, "语义T2（变化区）"), 1, 1)
        grid.addWidget(self._titled(self.lbl_chg_bin, "变化阈值图"), 1, 2)

        grid.addWidget(self._titled(self.lbl_catmap, "变化类别图"), 2, 0)
        grid.addWidget(self._titled(self.lbl_matrix, "变化转移矩阵"), 2, 1)
        grid.addWidget(self._titled(self.lbl_heat, "热力图叠加"), 2, 2)

        # 统计页：文本 + 插图（柱状图）
        vbox_stat = QtWidgets.QVBoxLayout(self.tab_stat)
        self.top_changes_box = QtWidgets.QTextEdit(); self.top_changes_box.setReadOnly(True)
        self.lbl_bars = QtWidgets.QLabel(); self._prep_label(self.lbl_bars)
        vbox_stat.addWidget(QtWidgets.QLabel("变化统计（Top 转移对/面积/占比）："))
        vbox_stat.addWidget(self.top_changes_box)
        vbox_stat.addWidget(self._titled(self.lbl_bars, "净增净减柱状图"))
        vbox_stat.addStretch(1)

        # —— AI分析页
        ai_layout = QtWidgets.QVBoxLayout(self.tab_ai)
        form = QtWidgets.QFormLayout()
        self.cmb_provider = QtWidgets.QComboBox()
        self.cmb_provider.addItems(["SiliconFlow", "OpenAI", "DeepSeek", "Custom"])
        self.cmb_provider.setCurrentText("SiliconFlow")

        self.edit_base_url = QtWidgets.QLineEdit(self.llm_base_url)
        self.edit_model = QtWidgets.QLineEdit(self.llm_model)
        self.edit_api_key = QtWidgets.QLineEdit()
        self.edit_api_key.setEchoMode(QtWidgets.QLineEdit.Password)
        self.spin_temp = QtWidgets.QDoubleSpinBox(); self.spin_temp.setRange(0.0, 2.0); self.spin_temp.setDecimals(2); self.spin_temp.setValue(self.llm_temperature)
        self.spin_max_tokens = QtWidgets.QSpinBox(); self.spin_max_tokens.setRange(128, 8192); self.spin_max_tokens.setValue(self.llm_max_tokens)

        form.addRow("提供商：", self.cmb_provider)
        form.addRow("Base URL：", self.edit_base_url)
        form.addRow("模型名：", self.edit_model)
        form.addRow("API Key：", self.edit_api_key)
        form.addRow("温度：", self.spin_temp)
        form.addRow("最大字数：", self.spin_max_tokens)

        btn_ai_gen = QtWidgets.QPushButton("生成AI分析")
        btn_ai_copy = QtWidgets.QPushButton("复制到剪贴板")

        self.txt_ai = QtWidgets.QTextEdit(); self.txt_ai.setReadOnly(False)
        self.txt_ai.setPlaceholderText("点击“生成AI分析”后，这里会出现大模型写的趋势与归因分析…")

        ai_layout.addLayout(form)
        row_btn = QtWidgets.QHBoxLayout()
        row_btn.addWidget(btn_ai_gen); row_btn.addWidget(btn_ai_copy); row_btn.addStretch(1)
        ai_layout.addLayout(row_btn)
        ai_layout.addWidget(self.txt_ai, 1)

        # 装入Tab
        tabs.addTab(self.tab_vis, "可视化")
        tabs.addTab(self.tab_stat, "统计分析")
        tabs.addTab(self.tab_ai, "AI分析")

        # ==== 用滚动区 + 分割条 ====
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(ctrl_widget)

        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        right_container = QtWidgets.QWidget()
        right_v = QtWidgets.QVBoxLayout(right_container)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.addWidget(tabs)
        right_scroll.setWidget(right_container)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 900])

        main_layout.addWidget(splitter)

        # 连接信号槽
        btn_im1.clicked.connect(self._choose_im1)
        btn_im2.clicked.connect(self._choose_im2)
        btn_ckpt.clicked.connect(self._choose_ckpt)
        btn_font.clicked.connect(self._choose_font)
        btn_run.clicked.connect(self._run_inference)
        btn_pdf.clicked.connect(self._export_pdf)

        # 阈值双向绑定
        self.thr_slider.valueChanged.connect(self._on_thr_slider_changed)
        self.thr_spin.valueChanged.connect(self._on_thr_spin_changed)
        btn_thr_best.clicked.connect(self._restore_best_thr)

        # 形态学变更即时重绘
        self.morph_mode.currentIndexChanged.connect(lambda *_: self._render_all(self._current_thr()))
        self.morph_ks.valueChanged.connect(lambda *_: self._render_all(self._current_thr()))
        self.morph_iter.valueChanged.connect(lambda *_: self._render_all(self._current_thr()))

        # AI
        btn_ai_gen.clicked.connect(self._on_ai_generate)
        btn_ai_copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.txt_ai.toPlainText()))
        self.cmb_provider.currentTextChanged.connect(self._on_provider_changed)

        self._log("准备就绪。请选择两幅图像与权重后点击【运行推理】。")

    def _prep_label(self, lbl: QtWidgets.QLabel):
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setMinimumSize(180, 180)
        lbl.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        lbl.setStyleSheet("QLabel { background: #222; color: #ddd; }")

    def _titled(self, widget: QtWidgets.QWidget, title: str) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _choose_im1(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择时相1图像", "", "Images (*.png *.jpg *.jpeg *.tif)")
        if p: self.im1_edit.setText(p)

    def _choose_im2(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择时相2图像", "", "Images (*.png *.jpg *.jpeg *.tif)")
        if p: self.im2_edit.setText(p)

    def _choose_ckpt(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择模型权重", "", "PyTorch (*.pth *.pt)")
        if p: self.ckpt_edit.setText(p)

    def _choose_font(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择中文字体", "", "Fonts (*.ttf *.otf *.ttc)")
        if p: self.font_edit.setText(p)

    # 阈值联动
    def _current_thr(self) -> float:
        return float(self.thr_spin.value())

    def _on_thr_slider_changed(self, v: int):
        thr = v / 1000.0
        if abs(self.thr_spin.value() - thr) >= 1e-6:
            self.thr_spin.blockSignals(True); self.thr_spin.setValue(thr); self.thr_spin.blockSignals(False)
        if self.outputs is not None:
            self._render_all(thr)

    def _on_thr_spin_changed(self, val: float):
        v = int(round(val * 1000))
        if self.thr_slider.value() != v:
            self.thr_slider.blockSignals(True); self.thr_slider.setValue(v); self.thr_slider.blockSignals(False)
        if self.outputs is not None:
            self._render_all(val)

    def _restore_best_thr(self):
        if self.outputs is not None:
            thr = float(self.outputs.best_thr)
            thr = max(0.0, min(1.0, thr))
            self.thr_slider.setValue(int(round(thr * 1000)))
            self.thr_spin.setValue(thr)
            self._log(f"阈值已恢复为 best_thr={thr:.3f}")

    def _run_inference(self):
        im1 = self.im1_edit.text().strip()
        im2 = self.im2_edit.text().strip()
        ckpt = self.ckpt_edit.text().strip()
        self.font_path = self.font_edit.text().strip()
        self.gsp_m_per_px = self.gsp_edit.text().strip()

        if not im1 or not os.path.exists(im1):
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择合法的时相1图像文件。"); return
        if not im2 or not os.path.exists(im2):
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择合法的时相2图像文件。"); return
        if not ckpt or not os.path.exists(ckpt):
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择合法的模型权重文件。"); return

        self.session_dir = ensure_dir(os.path.join("outputs", "gui_session", now_tag()))
        self._log(f"Session 目录：{self.session_dir}")

        self.runner = ModelRunner(im1, im2, ckpt, self)
        self.runner.finished.connect(self._on_inference_finished)
        self._log("开始推理（将自动选择 CUDA/CPU，并按需启用 AMP）……")
        self.runner.start()

    def _on_inference_finished(self, outputs: Optional[InferenceOutputs], err: Optional[Exception]):
        if err is not None:
            QtWidgets.QMessageBox.critical(self, "错误", f"推理失败：\n{str(err)}")
            self._log(f"[ERROR] {str(err)}")
            return
        self.outputs = outputs
        thr = float(outputs.best_thr)
        self.thr_slider.setValue(int(round(thr * 1000)))
        self.thr_spin.setValue(thr)
        self._log(f"推理完成。best_thr={thr:.3f}。拖动阈值滑条可即时查看不同阈值效果。")
        self._render_all(thr)

    # --- 渲染与保存一套图 ---
    def _render_all(self, thr: float):
        o = self.outputs
        if o is None: return

        chg_prob = o.chg_prob
        chg_bin = (chg_prob > thr).astype(np.uint8)

        # 形态学后处理
        mode_idx = self.morph_mode.currentIndex()
        mode_map = {0:'none', 1:'open', 2:'close', 3:'open_close', 4:'close_open'}
        mode = mode_map.get(mode_idx, 'none')
        ks = int(self.morph_ks.value())
        iters = int(self.morph_iter.value())
        chg_bin = _apply_morph(chg_bin, mode, ks, iters)

        sem1_pred = logits_argmax_numpy(o.sem1_logits)
        sem2_pred = logits_argmax_numpy(o.sem2_logits)

        sem1_vismask = mask_unaffected_to_class6(sem1_pred, chg_bin)
        sem2_vismask = mask_unaffected_to_class6(sem2_pred, chg_bin)

        cat_code = make_change_category_map(sem1_pred, sem2_pred, chg_bin)
        cat_rgb = colorize_change_category_map(cat_code)

        heat_rgb = apply_colormap_to_prob(chg_prob, cmap_name="jet")
        heat_over_im2 = overlay(o.im2_vis, heat_rgb, alpha=0.45)

        M_pred = compute_transition_matrix_from_preds(sem1_pred, sem2_pred, chg_bin)

        # === 保存图 ===
        inputs_path = os.path.join(self.session_dir, "inputs.png")
        self._save_grid([[o.im1_vis, o.im2_vis]], inputs_path, titles=[["Time 1", "Time 2"]])

        change_path = os.path.join(self.session_dir, "change.png")
        prob_gray = (chg_prob * 255).astype(np.uint8)
        prob_gray_rgb = np.stack([prob_gray]*3, axis=-1)
        bin_rgb = np.stack([chg_bin*255]*3, axis=-1)
        self._save_grid(
            [[prob_gray_rgb, bin_rgb, heat_over_im2]],
            change_path,
            titles=[["Change Prob", f"Change Bin (thr={thr:.2f})", "Heatmap on T2"]]
        )

        sem1_color = colorize_mask(sem1_vismask)
        sem2_color = colorize_mask(sem2_vismask)
        cats_path = os.path.join(self.session_dir, "semantics.png")
        self._save_grid([[sem1_color, sem2_color, cat_rgb]], cats_path,
                        titles=[["Sem T1 (chg only)", "Sem T2 (chg only)", "Change Category"]])

        matrix_path = os.path.join(self.session_dir, "matrix.png")
        self._save_matrix_figure(M_pred, matrix_path, self.class_names_zh)

        bars_path, top_text = self._save_bars_and_text(M_pred, chg_bin, self.gsp_m_per_px, self.class_names_zh)

        cover_path = os.path.join(self.session_dir, "cover.png")
        self._save_grid([
            [o.im1_vis, o.im2_vis, prob_gray_rgb],
            [sem1_color, sem2_color, bin_rgb],
            [cat_rgb, self._load_img_safe(matrix_path), heat_over_im2],
        ], cover_path, titles=[
            ["Time 1", "Time 2", "Change Prob"],
            ["Sem T1 (chg only)", "Sem T2 (chg only)", f"Change Bin (thr={thr:.2f})"],
            ["Change Category", "Transition Matrix", "Heatmap on T2"]
        ])

        # --- 刷新 UI 预览 ---
        self._set_pixmap(self.lbl_im1, o.im1_vis)
        self._set_pixmap(self.lbl_im2, o.im2_vis)
        self._set_pixmap(self.lbl_chg_prob, prob_gray_rgb)
        self._set_pixmap(self.lbl_chg_bin, bin_rgb)
        self._set_pixmap(self.lbl_sem1, sem1_color)
        self._set_pixmap(self.lbl_sem2, sem2_color)
        self._set_pixmap(self.lbl_catmap, cat_rgb)
        self._set_pixmap(self.lbl_heat, heat_over_im2)
        self._set_pixmap(self.lbl_matrix, self._load_img_safe(matrix_path))

        self.top_changes_box.setPlainText(top_text)
        self._set_pixmap(self.lbl_bars, self._load_img_safe(bars_path))

        # 统计信息（供 AI 使用）
        valid_mask = o.valid_mask if (o.valid_mask is not None and o.valid_mask.shape == chg_bin.shape) else np.ones_like(chg_bin, np.uint8)
        valid_px = int(valid_mask.sum())
        chg_px_valid = int((chg_bin & valid_mask).sum())
        chg_ratio_valid = (chg_px_valid / valid_px) if valid_px > 0 else 0.0

        thr_lo = max(0.0, thr - 0.05)
        thr_hi = min(1.0, thr + 0.05)
        bin_lo = ((o.chg_prob > thr_lo).astype(np.uint8) & valid_mask)
        bin_hi = ((o.chg_prob > thr_hi).astype(np.uint8) & valid_mask)
        inter = np.logical_and(bin_lo, bin_hi).sum()
        union = np.logical_or(bin_lo, bin_hi).sum()
        stability = (inter / union) if union > 0 else 1.0

        self._last_render_meta = dict(
            cover=cover_path, inputs=inputs_path, change=change_path,
            semantics=cats_path, matrix=matrix_path, bars=bars_path, thr=f"{thr:.3f}",
            valid_px=valid_px, chg_px_valid=chg_px_valid,
            chg_ratio_valid=f"{chg_ratio_valid*100:.2f}%",
            stability=f"{stability:.3f}"
        )

    def _save_matrix_figure(self, M: np.ndarray, out_path: str,
                            class_names: Optional[Tuple[str, ...]] = None):
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        plot_change_matrix(M, ax=ax)
        if class_names and len(class_names) >= 6:
            ax.set_xticks(np.arange(6)); ax.set_yticks(np.arange(6))
            ax.set_xticklabels(class_names, rotation=0)
            ax.set_yticklabels(class_names)
            ax.set_xlabel("Time 2 类别")
            ax.set_ylabel("Time 1 类别")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def _save_bars_and_text(self, M: np.ndarray, chg_bin: np.ndarray, gsp_m_per_px_str: str,
                            class_names: Optional[Tuple[str, ...]] = None) -> Tuple[str, str]:
        total_chg_px = int(chg_bin.sum())
        m_per_px = None
        if gsp_m_per_px_str:
            try:
                m_per_px = float(gsp_m_per_px_str)
            except:
                m_per_px = None

        names = class_names if (class_names and len(class_names) >= 6) else tuple(str(i) for i in range(6))

        gain = (M.sum(axis=0) - np.diag(M))
        loss = (M.sum(axis=1) - np.diag(M))
        net  = gain - loss

        pairs = []
        for i in range(6):
            for j in range(6):
                if i == j: continue
                if M[i, j] > 0:
                    pairs.append((i, j, int(M[i, j])))
        pairs.sort(key=lambda x: x[2], reverse=True)
        top5 = pairs[:5]

        lines = []
        lines.append(f"变化像素总数：{total_chg_px}")
        if m_per_px:
            area_m2 = total_chg_px * (m_per_px ** 2)
            lines.append(f"变化面积（m^2）：{area_m2:.2f}")
        lines.append("Top5 变化转移对（原类→新类：像素数）：")
        for i, j, v in top5:
            lines.append(f"  {names[i]} → {names[j]} : {v}（{i}→{j}）")

        out_path = os.path.join(self.session_dir, "bars.png")
        fig = plt.figure(figsize=(7.2, 4.2))
        ax = fig.add_subplot(111)
        x = np.arange(6)
        ax.bar(x - 0.25, gain, width=0.25, label="净转入(Gain)")
        ax.bar(x,        loss, width=0.25, label="净转出(Loss)")
        ax.bar(x + 0.25, net,  width=0.25, label="净增(Net)")
        ax.set_xticks(x)
        xtick_lbls = [f"{i}:{names[i]}" for i in range(6)]
        ax.set_xticklabels(xtick_lbls, rotation=0)
        ax.set_title("各类别净转入/净转出/净增统计")
        ax.set_xlabel("类别")
        ax.set_ylabel("像素数")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        return out_path, "\n".join(lines)

    def _save_grid(self, rows_imgs, out_path, titles=None):
        r = len(rows_imgs); c = max(len(row) for row in rows_imgs)
        fig, axes = plt.subplots(r, c, figsize=(4*c, 4*r))
        if r == 1 and c == 1:
            axes = np.array([[axes]])
        elif r == 1:
            axes = np.array([axes])
        elif c == 1:
            axes = axes.reshape(-1, 1)
        for i in range(r):
            for j in range(c):
                ax = axes[i, j]
                if j < len(rows_imgs[i]) and rows_imgs[i][j] is not None:
                    ax.imshow(rows_imgs[i][j])
                ax.axis("off")
                if titles and i < len(titles) and j < len(titles[i]):
                    ax.set_title(titles[i][j])
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)

    def _load_img_safe(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            return np.zeros((512, 512, 3), dtype=np.uint8)
        return np.array(Image.open(path).convert("RGB"))

    def _set_pixmap(self, label: QtWidgets.QLabel, img: np.ndarray):
        self._label_imgs[label] = img
        h, w = img.shape[:2]
        qimg = QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        label.setPixmap(pix.scaled(label.width(), label.height(),
                                   QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event: QtGui.QResizeEvent):
        try:
            for lbl, img in self._label_imgs.items():
                if isinstance(lbl, QtWidgets.QLabel) and img is not None:
                    h, w = img.shape[:2]
                    qimg = QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
                    pix = QtGui.QPixmap.fromImage(qimg)
                    lbl.setPixmap(pix.scaled(lbl.width(), lbl.height(),
                                             QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        except Exception:
            pass
        super().resizeEvent(event)

    def _make_ai_prompts(self) -> Tuple[str, str]:
        if self.outputs is None:
            raise RuntimeError("请先完成一次推理，再生成AI分析。")

        top_txt = self.top_changes_box.toPlainText().strip()
        thr = self._last_render_meta.get("thr", "0.5")
        morph_mode = getattr(self.morph_mode, "currentText", lambda: "")()
        morph_ks = getattr(self.morph_ks, "value", lambda: 0)()
        morph_iter = getattr(self.morph_iter, "value", lambda: 0)()

        valid_px = self._last_render_meta.get("valid_px", None)
        chg_px_valid = self._last_render_meta.get("chg_px_valid", None)
        chg_ratio_valid = self._last_render_meta.get("chg_ratio_valid", None)
        stability = self._last_render_meta.get("stability", None)

        o = self.outputs
        vm = o.valid_mask if o.valid_mask is not None else np.ones_like(o.chg_prob, np.uint8)
        sem1_pred = logits_argmax_numpy(o.sem1_logits)
        sem2_pred = logits_argmax_numpy(o.sem2_logits)
        mask_valid_cls1 = (vm == 1) & (sem1_pred != 6)
        mask_valid_cls2 = (vm == 1) & (sem2_pred != 6)
        t1_counts = np.bincount(sem1_pred[mask_valid_cls1].ravel(), minlength=7)[:6]
        t2_counts = np.bincount(sem2_pred[mask_valid_cls2].ravel(), minlength=7)[:6]

        names = list(self.class_names_zh)
        t1_line = "；".join([f"{i}:{names[i]}={int(t1_counts[i])}" for i in range(6)])
        t2_line = "；".join([f"{i}:{names[i]}={int(t2_counts[i])}" for i in range(6)])

        system_prompt = (
            "你是“遥感变化检测→低空运行安全预警”助手。"
            "任务：仅依据用户给出的统计量（变化像素/占比、Top 转移对、T1/T2 类面积、阈值与形态学、稳定性），"
            "判断本次两时相像素变化会对无人机航线走廊与起降点带来哪些安全隐患，并给出简明可执行提示。"
            "\n严格要求："
            "\n- 只讨论本次变化相关的安全问题：净空受限（→建筑）、硬化/反光（0/2→1）、涉水扩张/回填（1↔3），以及其他显著 A→B；"
            "\n- 结论必须回引具体数字（像素/占比/稳定性/阈值/形态学），避免空泛描述；"
            "\n- 不得引用外部数据、政策、坐标或常识臆测；不得输出 JSON；"
            "\n- 若变化占比 < 1% 且 Top 转移对均 < 100 像素，请判为低风险并给出“持续监测”的最小成本建议；"
            "\n- 严格按以下 5 个标题输出，每节不超过 120 字："
            "\n# 安全隐患清单（走廊/起降点）"
            "\n# 预警等级与判断依据"
            "\n# 证据与变化解释"
            "\n# 建议处置（最小成本优先）"
            "\n# 不确定性与复核建议"
            "\n类别索引：0 低矮植被，1 非植被，2 乔木，3 水体，4 建筑，5 操场。"
            "\n关键转移对参考（仅作规则）：0/2→4、0/2→1、1↔3、1/0/2→4；涉及 5（操场）变化从严评估。"
        )

        user_prompt = f"""
【统计汇总（仅可依据以下数据作出判断）】
推理分辨率：512×512
有效域像素：{valid_px}；有效域变化：{chg_px_valid}（占比：{chg_ratio_valid}）
稳定性（thr±0.05 交并比）：{stability}
阈值/形态学：thr={thr}；操作={morph_mode}；核={morph_ks}；迭代={morph_iter}
类别映射(0..5)：{names}
T1 类面积（有效域）：{t1_line}
T2 类面积（有效域）：{t2_line}
Top5 转移对（原类→新类，含像素/占比的中文行）：
{top_txt}

【输出要求】
- 只输出以下 5 个标题与正文，简洁明确，不得添加其他段落、表格或 JSON；
- 所有结论需引用上面的具体数字；
- 重点围绕：净空受限（→建筑）、硬化/反光（0/2→1）、涉水扩张/回填（1↔3）；若无显著证据，请明确说明。

# 安全隐患清单（走廊/起降点）
# 预警等级与判断依据
# 证据与变化解释
# 建议处置（最小成本优先）
# 不确定性与复核建议
""".strip()

        return system_prompt, user_prompt

    def _on_ai_generate(self):
        if self.outputs is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先完成一次推理再生成AI分析。")
            return

        provider = self.cmb_provider.currentText().strip()
        base_url = self.edit_base_url.text().strip()
        model = self.edit_model.text().strip()
        api_key = self.edit_api_key.text().strip()
        temp = float(self.spin_temp.value())
        max_tokens = int(self.spin_max_tokens.value())

        if not api_key:
            QtWidgets.QMessageBox.warning(self, "提示", "请填写 API Key。")
            return
        if provider == "OpenAI" and not base_url:
            base_url = "https://api.openai.com"
        if provider == "DeepSeek" and not base_url:
            base_url = "https://api.deepseek.com"
        if provider == "SiliconFlow" and not base_url:
            base_url = "https://api.siliconflow.cn/v1"

        system_prompt, user_prompt = self._make_ai_prompts()
        self.txt_ai.append("正在调用大模型生成分析，请稍候…")

        self.llm_thread = LLMRunner(base_url, model, api_key, system_prompt, user_prompt, temp, max_tokens, self)
        self.llm_thread.finished.connect(self._on_ai_finished)
        self.llm_thread.start()

    def _on_ai_finished(self, text: str, err: Optional[Exception]):
        if err:
            QtWidgets.QMessageBox.critical(self, "错误", f"AI分析失败：\n{err}")
            return
        pretty = normalize_ai_text(text)
        self.ai_analysis_text = pretty
        self.txt_ai.setPlainText(pretty)
        self._log("AI 分析已生成（已转换为纯文本以便阅读与导出）。")

    def _on_provider_changed(self, name: str):
        cur_model = self.edit_model.text().strip()
        if name == "OpenAI":
            self.edit_base_url.setText("https://api.openai.com")
            if not cur_model:
                self.edit_model.setText("gpt-4o-mini")
        elif name == "DeepSeek":
            self.edit_base_url.setText("https://api.deepseek.com")
            if not cur_model:
                self.edit_model.setText("deepseek-chat")
        elif name == "SiliconFlow":
            self.edit_base_url.setText("https://api.siliconflow.cn/v1")
            if not cur_model:
                self.edit_model.setText("deepseek-ai/DeepSeek-R1")
        else:
            pass

    def _export_pdf(self):
        if self.outputs is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先完成一次推理。")
            return

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存PDF报告", os.path.join(self.session_dir or ".", "Report.pdf"), "PDF (*.pdf)")
        if not save_path:
            return

        pages = dict(
            cover=self._last_render_meta.get("cover"),
            inputs=self._last_render_meta.get("inputs"),
            change=self._last_render_meta.get("change"),
            semantics=self._last_render_meta.get("semantics"),
            matrix=self._last_render_meta.get("matrix"),
            bars=self._last_render_meta.get("bars"),
        )
        info = dict(
            ckpt_path=self.ckpt_edit.text().strip(),
            best_thr=self.outputs.best_thr,
            thr_used=self._last_render_meta.get("thr"),
            gsp_m_per_px=self.gsp_m_per_px,
            im1=os.path.basename(self.im1_edit.text().strip()),
            im2=os.path.basename(self.im2_edit.text().strip()),
            session_dir=self.session_dir,
            morph_mode=self.morph_mode.currentText(),
            morph_kernel=self.morph_ks.value(),
            morph_iter=self.morph_iter.value(),
            class_names="，".join(self.class_names_zh),
            valid_px=self._last_render_meta.get("valid_px"),
            chg_px_valid=self._last_render_meta.get("chg_px_valid"),
            chg_ratio_valid=self._last_render_meta.get("chg_ratio_valid"),
            stability=self._last_render_meta.get("stability"),
            ai_analysis=self.txt_ai.toPlainText().strip()
        )

        builder = PdfReportBuilder(save_path, font_path=self.font_edit.text().strip())
        try:
            builder.build(pages, info)
            QtWidgets.QMessageBox.information(self, "成功", f"PDF 已导出：\n{save_path}")
            self._log(f"PDF 已导出：{save_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"PDF 生成失败：\n{str(e)}")
            self._log(f"[ERROR] PDF 生成失败：{str(e)}")

    def _log(self, msg: str):
        self.status_box.append(msg)

# -----------------------------
# 入口
# -----------------------------
def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(QtCore.QCoreApplication, "setHighDpiScaleFactorRoundingPolicy"):
        QtCore.QCoreApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    app = QtWidgets.QApplication(sys.argv)

    base_pt = 12
    try:
        dpi = app.primaryScreen().logicalDotsPerInch()
        if dpi >= 192:       # 200%
            base_pt = 14
        elif dpi >= 144:     # 150%
            base_pt = 13
        else:                # 100~125%
            base_pt = 12
    except Exception:
        pass
    f = QtGui.QFont("Microsoft YaHei", base_pt)
    if hasattr(QtGui.QFont, "PreferAntialias"):
        f.setStyleStrategy(QtGui.QFont.PreferAntialias)
    app.setFont(f)

    w = MainWindow()
    ag = app.primaryScreen().availableGeometry()
    init_w = min(1200, int(ag.width() * 0.78))
    init_h = min(840,  int(ag.height() * 0.78))
    w.resize(init_w, init_h)

    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
