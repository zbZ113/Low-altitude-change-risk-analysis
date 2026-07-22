# 基于双时相地表变化检测的低空飞行风险分析系统

系统面向低空航线与起降区域的环境变化核验，将双时相遥感图像转换为变化检测结果、地物语义转移矩阵和低空运行风险分析报告。

[查看项目文档](docs/project-document.pdf) · [查看获奖证明](#获奖与证据)

## 项目概览

- 任务：变化检测 + T1/T2 双时相语义分割 + 地物转移分析
- 主干：基于 ChangeFormerV6 的 Siamese Transformer 编码器
- 工程：PyTorch、Albumentations、PyQt5、Matplotlib、ReportLab
- 语言模型：可接入 SiliconFlow、DeepSeek 或自定义服务
- 数据：SECOND 语义变化检测数据集

当前归档中的验证结果：最佳 F1 0.6759、变化 IoU 0.5104、T1 mIoU 0.4636、T2 mIoU 0.5379、SeK 0.5459。

## 模型结构

`src/model.py` 中的 `MultiTaskChangeFormer` 复用 ChangeFormerV6 编码器和变化解码器，并增加：

1. T1/T2 两个语义 FPN 头，输出 7 通道语义 logits。
2. C4 跨时相空间与通道注意力增强。
3. `abs(im1 - im2)` 差异编码器和 C4 残差融合。
4. 四级特征对齐、共享融合与双时相残差回写。

## 目录

```text
assets/       页面使用的模型图、效果图和证书预览
docs/         项目文档、答辩 PDF、报告样例、获奖证书
src/          最终集成版核心源码
third_party/  ChangeFormer 上游许可证
```

## 本地运行

源码依赖 Python 3.10 左右环境。先安装 `requirements.txt`，再把 SECOND 数据集放到 `data/Second/`，把训练权重放到 `outputs/checkpoints/`。

```powershell
python -m pip install -r requirements.txt
python src/train.py
python src/val_sec.py --help
python src/gui_ai.py
```

GUI 默认读取 `outputs/checkpoints/best_model.pth`，也支持在界面中重新选择图像、权重和中文字体。LLM 分析需要用户在界面中提供 API Key。

## 获奖与证据

- “华为杯”第七届中国研究生人工智能创新大赛全国二等奖：[`docs/award-national.png`](docs/award-national.png)
- 青岛市赛题专项二等奖：[`docs/award-qingdao.png`](docs/award-qingdao.png)
- 决赛项目文档：[`docs/project-document.pdf`](docs/project-document.pdf)

## 第三方说明

`src/ChangeFormer.py` 和 `src/ChangeFormerBaseNetworks.py` 沿用 ChangeFormer 上游实现，许可证保存在 [`third_party/ChangeFormer-LICENSE`](third_party/ChangeFormer-LICENSE)。相关论文和原始项目链接见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
