# 织物瑕疵检测：模型训练全流程讲义

## 读这份讲义的收获

读完并动手复现，你将能够：

- 说清一个织物瑕疵检测项目的**数据从哪来、怎么洗、怎么进模型**
- 解释 **RT-DETR 相比 YOLO 为什么不需要 NMS**，以及微调时哪些层被训练
- 手写**七维可微奖励**的每一个公式，并解释为什么要用 softmax 代替 argmax
- 复现两个真实踩过的坑：梯度裁剪过松导致 NaN、fp16 导致 BN 层`running_var` 归零使 mAP 崩为 0
- 对实验结果做出**诚实的能力边界判断**，区分「模型不行」与「数据不行」

## 前置要求与阅读方式

**前置知识**（按依赖顺序）：

1. 卷积神经网络基本原理（卷积核、池化、感受野）
2. Softmax 与交叉熵损失
3. 反向传播与计算图，理解「梯度从哪来」
4. 边界框表示（`xyxy` 与 `cxcywh`）、IoU 计算
5. 训练/验证/测试集划分与过拟合

**技术栈版本**：

| 组件 | 版本 |
|------|------|
| Python | 3.13（训练环境为 anaconda3 base）|
| PyTorch | 2.8.0+cu126 |
| Ultralytics | 8.4.91 |
| GPU | RTX 4090 D 24GB |
| 数据集 | 天池 `smartdiagnosisofclothflaw` |

**阅读建议**：全文约 25 分钟。若时间紧，优先读「第三章 模型怎么训练的」与「第四章 微调了哪些部分」，这是课堂提问的密集区。

---

## 第一章 项目全景：一条完整的工业视觉链路

### 1.1 系统架构总览

本项目不是单一模型，而是一套 **27027 行**的工业视觉系统，涵盖数据工程、模型训练、推理服务、实时展示四大阶段，共11 个关键步骤。

| 阶段 | 核心产出 | 关键脚本 |
|------|---------|---------|
| 数据工程 | 8619 张训练图 / 8570 个标注框 | `scripts/convert_tianchi_to_yolo.py` |
| 模型训练 | mAP50 = 0.541（18 类）| `scripts/train_rtdetr.py` |
| 推理服务 | `(1, 300, 6)` 张量输出 | `backend/inference/` |
| 实时展示 | 中文看板 + WebSocket 告警 | `backend/dashboard.html` |

### 1.2 代码分层架构

```
前端层零构建 dashboard.html + simulator.html（Canvas + Chart.js）
    ↓ REST + WebSocket
接入层 routers/ 只做三件事：Pydantic 校验 → 调 service → 包 APIResponse
    ↓
业务层 services/ 与模型层 inference/ 并行调用
    ↓
数据层 models/ + database.py（SQLAlchemy 2.0 async）
```

贯穿全栈的主线是**单一事实来源**：`backend/schemas/defect.py` 定义 18 类枚举与严重度，训练、推理、告警、前端全部引用它。

### 1.3 缺陷分类体系

原始 20 类中有两对视觉上几乎无法区分的类别，经实证诊断后合并为 18 类：

| 合并项 | 原类别 | 合并原因 |
|--------|--------|---------|
| 断经 ↔ 磨痕 | `broken_warp` / `surface_mark` | 双向混淆，标签噪声 |
| 纬缩 ↔ 粗纬 | `weft_shrink` / `coarse_weft` | 双向混淆，标签噪声 |

合并后关键类召回显著回升：纬缩+粗纬 **0.40 → 0.88**，断经+磨痕 **0.52 → 0.79**。

---

## 第二章 数据从哪来、怎么洗、怎么进模型

### 2.1 数据采集：为什么用公开竞赛数据集

数据来自阿里天池公开竞赛「智能织物瑕疵诊断赛」，赛题代号 `smartdiagnosisofclothflaw`，提供 5 个压缩包。

选择公开数据集而非自采的三个理由：

| 维度 | 自采数据 | 天池数据（项目选择）|
|------|---------|---------------------|
| 标注成本 | 一张图一个框，人工成本极高 | 官方已标好 8570 个框 |
| 可比性 | 无法与他人结果对比 | 赛题统一口径，可比公开方法 |
| 版权 | 产线数据多涉商业机密 | 公开可商用 |

### 2.2 原始数据结构

三个带标注的训练集加两个无标注测试集：

| 数据集 | 缺陷图数 | 说明 |
|--------|---------|------|
| train1 partA | 2387 张 | 广东产线，带标注 |
| train1 partB | 2387 张 | **与 partA 内容重复**，见2.3 |
| train2 | 1139 张 | 另一批产线，带标注 |
| testA / testB | 无标注 | 仅推理用 |

原始数据采用**两分类目录结构**，这是天池数据集的特色：

```
defect_Images/     缺陷图（有标注）
normal_Images/     正常图（无标注）
Annotations/anno_train.json   全部标注
```

标注 JSON 格式：

```json
{"name": "图片文件名.jpg", "defect_name": "断经", "bbox": [xmin, ymin, xmax, ymax]}
```

### 2.3 清洗办法一：哈希级去重

**这是一个真实存在的坑。**用 md5 校验发现 partA 与 partB 的标注文件字节级完全相同：

```
8ca9efa45e3f4ade805a12019a5880a3  partA/Annotations/anno_train.json
8ca9efa45e3f4ade805a12019a5880a3  partB/Annotations/anno_train.json
```

若直接遍历三个目录累加，同一张图会被写入两遍，**GT 框数量直接翻倍**，训练时的正负样本比完全失真。

```python
# scripts/convert_tianchi_to_yolo.py · convert_train_dir()
dst_img = os.path.join(out_images, fname)
if os.path.exists(dst_img):
    print(f"  [warn] duplicate image name skipped: {fname}")
    continue    # 按图像名去重，而非按标注条数累加
shutil.copy(src, dst_img)
```

> **教学点**：数据清洗中的「去重」必须明确**按什么键去重**。图像按文件名去重，标注按 `(name, bbox)` 元组去重，两者策略不同。用错键是数据污染的主要来源。

**顺带一提**：`__MACOSX/` 目录里还有 AppleDouble 元数据文件（`._anno_train.json`），解压 macOS 压缩包时的残留物。这类垃圾文件必须排除，否则可能被误当作标注读取。

### 2.4 清洗办法二：格式转换与坐标归一化

天池用 `xyxy` 像素坐标，YOLO 格式用 `cxcywh` 归一化坐标，转换函数如下：

```python
# scripts/convert_tianchi_to_yolo.py
def xyxy_to_yolo(bbox, w, h):
    """Convert xyxy pixel bbox to normalized (cx, cy, bw, bh)."""
    xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    cx = (xmin + xmax) / 2.0 / w        # 中心点 x
    cy = (ymin + ymax) / 2.0 / h        # 中心点 y
    bw = max(0.0, (xmax - xmin)) / w    # 宽度，max 防负
    bh = max(0.0, (ymax - ymin)) / h    # 高度，max 防负
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)   # 双向钳制
    return cx, cy, bw, bh
```

**为什么要归一化到0~1**：让标注与图像分辨率解耦。无论原图是 2446×1000 还是 1280×1280，标签格式统一，模型才能泛化到不同尺寸。

**三处防御值得单独讲**：

1. `max(0.0, ...)` —— 防止负宽度
2. `min(max(cx, 0.0), 1.0)` —— 双向钳制，防止越界坐标
3. 写入时保留六位小数 —— `f"{cx:.6f}"`。**浮点精度不足会让退化框（如宽度 0.0001）被截断为 0，直接丢框**

### 2.5 清洗办法三：负样本保留与别名归并

**负样本**（`normal_Images` 中的正常图）写成**空的 .txt 文件**：

```python
label_path = os.path.join(out_labels, os.path.splitext(fname)[0] + ".txt")
with open(label_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + ("\n" if lines else ""))   # lines 为空 → 空文件
```

> **为什么空文件也算数据**：检测任务必须见过「什么都没有」的图，否则模型会在完好织物上乱框。实测验证集 957 张里有 **374 张是负样本，占 39%**——这就是真实产线的先验分布。

**别名归并与错别字容错**：

```python
NAME_TO_CATEGORY: Dict[str, int] = {
    "破洞": 1,
    "水渍": 2, "油渍": 2, "污渍": 2,        # 三个同义名归并为一类
    # ...
    "粗纬": 12,
    "粗维": 12,  # 粗维是官方 README 的错别字
    "星跳": 16,
    "跳花": 16,   # 同义缺陷名归并
    # ...
}
```

> **教学点**：竞赛数据的常态就是别名混杂 + 官方文档笔误。别名归并是最容易被忽略、却最影响最终指标的一环。

### 2.6 划分顺序：先去重，再划分

```python
# scripts/convert_tianchi_to_yolo.py · split_train_val()
def split_train_val(root, val_ratio, seed):
    all_images = sorted(os.listdir(staging_img))
    random.seed(seed)              # 固定种子，结果可复现
    random.shuffle(all_images)
    n_val = int(len(all_images) * val_ratio)
    val_set = set(all_images[:n_val])
```

> ⚠️ **注意顺序**：先去重 → 再划分。如果先分别划分再合并，partA 的图可能进train 而 partB 的同名图进 val，**验证集就泄漏进了训练集**，指标虚高且不可复现。

清洗后的最终规模（实测统计）：

| 集合 | 图像数 | 标注框数 | 备注 |
|------|-------|---------|------|
| 训练集 train | 8619 | 8570 | 18 类，长尾明显 |
| 验证集 val | 957 | 953 | 含 374 张负样本，seed 42 |

### 2.7 图像如何输入模型

**这是最容易讲错的一环**——项目里有两套预处理，用错一套推理结果就是错的。

RT-DETR 的专用预处理（`backend/inference/rtdetr_onnx.py`）：

```python
def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR (H, W, 3) uint8 image to a (1, 3, imgsz, imgsz) tensor."""
    import cv2
    resized = cv2.resize(image_bgr, (self.imgsz, self.imgsz))  # 拉伸成正方形
    rgb = resized[:, :, ::-1]                                  # BGR → RGB
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0 # 只除 255
    return np.ascontiguousarray(tensor[None, ...])
```

**完整的五步链路**：

1. **原始图像** `H×W×3 uint8 BGR`，典型尺寸 2446×1000
2. **拉伸缩放** `cv2.resize(img, (1280, 1280))`，不等比拉伸、无灰边填充
3. **通道转换** `resized[:, :, ::-1]`，切片负步长，零内存拷贝
4. **归一化** 转CHW 后除以 255.0，**不做** ImageNet 均值方差
5. **网络输入** `(1, 3, 1280, 1280) float32`，取值区间 0.0 ~ 1.0

**两套预处理的差异对照**（答辩高频考点）：

| 项目 | 通用预处理 | RT-DETR 预处理 |
|------|-----------|---------------|
| 缩放方式 | letterbox（等比 + 填充黑边）| **scale_fill（不等比拉伸）** |
| 归一化 | ImageNet mean/std | **只除 255.0** |

> **为什么必须用拉伸而不是填充**：`letterbox` 会引入黑边，模型训练时没见过黑边，推理时突然出现会造成分布偏移。更重要的是——因为是不等比拉伸，**归一化坐标反变换回原图时只需乘以原始宽高**，无需还原padding 偏移量。

模型输出是单个 `(1, 300, 6)` 张量，格式 `[cx, cy, w, h, score, class]`，**没有 objectness 通道，且不需要 NMS**。

---

## 第三章 模型是怎么训练的

### 3.1 起点：COCO 预训练权重

```python
# scripts/train_rtdetr.py
model = YOLO(args.model)      # 默认 rtdetr-l.pt，COCO 80 类预训练
model.train(
    data=args.data,           # data/tianchi/data.yaml
    epochs=args.epochs,       # 60
    imgsz=args.imgsz,         # 1280
    batch=args.batch,         # 4
    amp=args.amp,             # ⚠️ 必须 False，理由见 5.2
    seed=42,
)
```

这是**迁移学习**的标准做法：用 COCO（自然图像，80 类通用物体）预训练权重初始化，再在织物瑕疵数据上微调。

理由很实际：织物纹理的边缘、纤维走向与自然图像差别很大，但底层特征（边缘、纹理、形状）可以复用。

### 3.2 最佳配方及其实验依据

```python
# scripts/train_mosaic1280.py（实测最优，官方口径 mean mAP = 0.6349）
model.train(
    data="data/tianchi/data.yaml",
    imgsz=1280, epochs=60, patience=20, batch=4, workers=0,
    rect=True,                  # 矩形批次：按图片真实长宽比组批
    mosaic=1.0,                 # Mosaic 四图拼接增强
    optimizer="auto",           # MuSGD，注意：不要换成 AdamW
    amp=True,                   # 已monkey patch 为 bf16
    seed=42, lr0=0.0005, lrf=0.01,
)
```

每一项都有实验依据，来自**因子解耦实验**（单变量对照）：

| 实验臂 | 变化 | mean mAP | 增量 |
|--------|------|---------|------|
| 基线 | 1280 方形，无 mosaic | 0.5719 | — |
| 加 rect | 矩形批次 | 0.6080 | **+0.036** |
| 再加 mosaic | 四图增强 | **0.6349** | **+0.027** |
| 换 1536 | 更高分辨率 | 0.6077 | **−0.027（无净增益）** |

> ⚠️ **反直觉结论，值得单独讲**：分辨率从 1280 提到 1536 反而变差。因为原图只有 2446×1000，无限拉伸到更高分辨率只是放大噪声，还要付出 25% 的时间代价。「更高分辨率一定更好」是常见的直觉错误。

`rect=True` 的额外收益：按图片真实长宽比组批，避免大量填充浪费算力，**节省 59%** 计算量。

---

## 第四章 微调了哪些部分

### 4.1 结论先行：全参数微调，唯一重建的是分类头

| 模块 | 初始化方式 | 是否冻结 |
|------|-----------|---------|
| Backbone（ResNet + 混合注意力）| COCO 预训练权重 | ❌ 全参数微调 |
| Neck（FPN 多尺度融合）| COCO 预训练权重 | ❌ 全参数微调 |
| Decoder（可变形注意力，6 层）| COCO 预训练权重 | ❌ 全参数微调 |
| **分类回归头**（focal/vfl）| **80类 → 18 类，权重重新初始化** | ❌ 从头学习 |

**分类头为什么必须重建**：`rtdetr-l.pt` 最后一层为 COCO 的 80 个物体设计，输出维度是 80。本项目是 18 类，维度不匹配，所以 ultralytics 加载 `data.yaml` 时自动重建该头，**权重随机初始化、从零学**。而前面三层学到的边缘与纹理知识完整保留。

**注意：本项目没有冻结任何层。** 项目中唯一用到的「冻结」出现在另一个实验——3 族粗分类器的**冻结线性探针**（`--frozen` 模式，冻结骨干只训分类头），用于对比表征质量。结果：非冻结微调 top-1 = 0.9119，冻结线性探针 = 0.8835，相差 0.028。

### 4.2 真正的创新：损失函数层面的注入

从「改了哪些权重」看，这就是标准全参数微调。**项目的创新不在权重，在损失函数**。

**实现方式：子类化 ultralytics 的 criterion，追加一个损失项。**

```python
# backend/training/reward_rtdetr.py
class RewardRTDETRDetectionLoss(RTDETRDetectionLoss):
    """RT-DETR criterion that appends a differentiable scalar-reward loss term."""

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        # 先算原生三项损失
        loss = super().forward(preds, batch,
                               dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta)
        dec_bboxes, dec_scores = preds
        # 再挂一个奖励项
        loss["loss_reward"] = self._reward_loss(dec_bboxes[-1], dec_scores[-1], batch)
        return loss
```

> ⚠️ **为什么这样写很聪明**：ultralytics 的 `RTDETRDetectionModel.loss()` 会把 criterion 返回字典里的**每一项都加起来**。所以只要往字典里塞一个 `loss_reward` 键，奖励就自动被折进监督目标，**完全不用改动 ultralytics 的训练循环**。这是「最小侵入式扩展」——升级框架版本时这段代码依然有效。

最终的损失函数组成：

```
L_total = L_giou + L_class(VFL) + L_bbox(L1) + β × (−R_scalar)
          └────── 原生三项 ──────┘└──── 本项目注入 ────┘
```

`β` 是超参数：`β=0` 即退化为纯监督训练，作为 A/B 对照基线。

### 4.3 奖励算在哪一步：复用匈牙利匹配

```python
def _reward_loss(self, pred_bboxes, pred_scores, batch):
    # 关键：调用的是同一个 matcher、同一批输入
    match_indices = self.matcher(
        pred_bboxes, pred_scores, batch["bboxes"], batch["cls"], batch["gt_groups"]
    )
```

注意这里复用了**监督损失已经算过一次的那次匹配**。

> **含义很重要**：奖励只评价监督信号已经选中的那些检测对，不会另起一套标准。奖励的唯一职责，是补充 CE + GIoU + denoising 无法表达的关键类召回信号。

### 4.4 最有价值的一处细节：如何处理「漏检」

```python
# 关键认知：RT-DETR 的匈牙利匹配把每个 GT 都分配给一个 query（因为 N >= num_gt）
# 所以「漏检」不是没有匹配的框，而是「匹配上的那个 query 对关键类置信度太低」
if self._p_crit(pred_scores[i][gt_to_query[g]], class_ids) >= self.miss_threshold:
    continue
spare = self._best_spare_query(i, class_ids, matched, pred_scores)  # 找个空闲 query
pred_boxes.append(pred_bboxes[i][spare])
targets.append({"boxes": [batch["bboxes"][g].tolist()], "labels": [label]})
```

这是一个**真实的认知纠错**：传统检测器（YOLO）中漏检 = 没有匹配的框；但 DETR 系因为查询数 ≥ GT 数，**每个 GT 一定有query 被分配**。

代码的做法是：拿一个**空闲的 spare query** 去指向这个漏检的 GT，让奖励项去惩罚它。这等于强行给一个空闲 query 施加召回压力——**而不是让漏检变成零梯度**。

---

## 第五章 七维可微奖励：算法内核

### 5.1 为什么必须「可微」

传统强化学习里最著名的问题是 **REINFORCE 梯度消失**：用 `argmax` 做决策时，梯度为0，模型学不动。

解决办法是把硬决策替换为软指示：

| 硬决策 | 替换为 | 是否可微 |
|--------|--------|---------|
| `argmax` | `softmax(logits)` 概率质量 | ✅ |
| 硬 IoU 阈值比较 | `sigmoid((θ − IoU) / τ)` | ✅ |
| Python `max()` | `torch.maximum()` | ✅ |

### 5.2 七个维度定义

| 维度 | 公式 | 取值范围 | 对应硬决策 |
|------|------|---------|-----------|
| D01 定位 | `2·IoU − 1` | [−1, 1] | 硬 IoU |
| D02 分类 | `2·p_true − 1` | [−1, 1] | `argmax` |
| D03 校准 | `−\|conf − p_true\|` | [−1, 0] | 置信度比较 |
| D04漏检 | `−λ·sigmoid((θ−IoU)/τ)` | [−λ, 0] | 阈值判断 |
| D05 误检 | `−λ·conf·(1−matched)` | [−λ, 0] | 阈值判断 |
| D06 断经敏感 | `3·p_crit − 2` | [−2, 1] | 预测类是否属关键集 |
| D07 跳花敏感 | `3·p_crit − 2` | [−2, 1] | 预测类是否属关键集 |

### 5.3 可微 IoU 的实现

```python
# backend/training/dimension_rewards.py
def differentiable_iou(pred_boxes, gt_boxes, eps=1e-6):
    """Vectorized, differentiable IoU for boxes in [x, y, w, h] format."""
    px1, py1, pw, ph = pred_boxes.unbind(1)     # unbind 而非切片，保持梯度
    gx1, gy1, gw, gh = gt_boxes.unbind(1)
    px2, py2 = px1 + pw, py1 + ph
    gx2, gy2 = gx1 + gw, gy1 + gh

    ix1 = torch.maximum(px1, gx1)   # ← torch 版本才有梯度
    iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2)
    iy2 = torch.minimum(py2, gy2)

    inter = torch.clamp(ix2 - ix1, min=0.0) * torch.clamp(iy2 - iy1, min=0.0)
    union = pw * ph + gw * gh - inter
    return inter / (union + eps)
```

> ⚠️ **关键点**：必须用 `torch.maximum` / `torch.minimum`（可微算子），**不能**用 Python 内置的 `max()` / `min()`。后者会直接切断计算图，`loss.backward()` 抛异常。
>
> `clamp(min=0.0)` 是 IoU 的常规保护——两框不相交时交集面积为负，必须截断到 0。

### 5.4 D02：用 softmax 代替 argmax

```python
def compute_cls_reward(self, predictions, targets):
    """R_cls = mean(2 * p_true_i - 1)"""
    logits, labels = self._aligned_logits_labels(predictions, targets)
    probs = self._class_probs(logits)                          # softmax
    p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)   # GT 类上的概率质量
    return (2.0 * p_true - 1.0).mean()
```

`p_true` 是模型在真实类别上的概率质量。即使模型预测错了，**`p_true` 依然携带梯度信息**：从 0.01 涨到 0.05 也是进步，梯度照样回传。

> **对照梯度公式**：softmax + 交叉熵的梯度是 `∂loss/∂z = p − onehot(y)`。当真实类概率被压到 0 时，梯度不是 0 而是接近 −1，模型仍能学习。而 argmax 硬决策直接让梯度归零。**这就是可微设计的全部意义。**

### 5.5 D06/D07：关键类集合的软敏感度

```python
def _sensitivity_reward(self, predictions, targets, class_ids):
    ids = torch.as_tensor(sorted(class_ids), dtype=torch.long, device=logits.device)
    probs = self._class_probs(logits)
    if self.head_type == "sigmoid":
        p_crit = probs[:, ids].max(dim=1).values    # 逐类头：取最大值
    else:
        p_crit = probs[:, ids].sum(dim=1)         # softmax 头：求概率质量
    is_crit = (labels.unsqueeze(1) == ids.unsqueeze(0)).any(dim=1)
    reward = torch.where(is_crit, 3.0 * p_crit - 2.0, torch.zeros_like(p_crit))
    return reward.mean()
```

三个设计要点：

1. **没有硬编码单个类别**，而是把 `{9, 15}` 当作集合算概率质量
2. **外积判定** `labels.unsqueeze(1) == ids.unsqueeze(0)` 判断样本是否属于该集合，非关键类样本贡献 0
3. **兼容两种检测头**：`head_type` 区分 softmax（单标签头）与 sigmoid（RT-DETR 的逐类 focal/vfl 头）

### 5.6 可学习维度权重

```python
# backend/training/hybrid_reward_trainer.py
self.dim_weights = nn.Parameter(
    torch.ones(NUM_DIMENSIONS, device=self.device) / NUM_DIMENSIONS
)
self.optimizer = optim.AdamW(
    [*list(model.parameters()), self.dim_weights],   # ← 权重与模型同一个优化器
    lr=learning_rate, weight_decay=weight_decay,
)
self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
```

> **这是「可学习权重」而非「手工调权重」**：七维的重要性不写死在超参里，而是当成参数让梯度自己去调，同时与模型参数一起被优化器更新。

### 5.7 损失函数实现

```python
# backend/training/loss_functions.py
def detection_loss(predictions, targets):
    """Standard detection loss: classification CE + bounding box regression."""
    # 分类项
    loss = loss + F.cross_entropy(logits, target_tensor)
    # 回归项
    loss = loss + F.smooth_l1_loss(p_box, t_box)
    return loss

def scalar_reward_loss(dim_rewards, dim_weights, consistency_coeff=0.3):
    """L_scalar = -R_scalar（最大化奖励，故取负）"""
    reward_tensor = torch.stack([dim_rewards[name] for name in DIMENSION_NAMES])
    return -(dim_weights * reward_tensor).sum() + ...
```

---

## 第六章 两个真实踩过的坑

### 6.1 坑一：梯度裁剪过松导致 NaN

ultralytics 在训练器里**硬编码**了 `clip_grad_norm_(max_norm=10.0)`，对 RT-DETR 来说太宽松，是 18 类训练 NaN 发散的元凶之一。

```python
# scripts/train_rtdetr.py —— monkey patch 运行时替换
_orig_clip_grad_norm_ = torch.nn.utils.clip_grad_norm_

def _clip_grad_norm_(parameters, max_norm, *a, **kw):
    if max_norm > args.clip_grad:     # 默认从 10.0 降到 1.0
        max_norm = args.clip_grad
    return _orig_clip_grad_norm_(parameters, max_norm, *a, **kw)

torch.nn.utils.clip_grad_norm_ = _clip_grad_norm_
```

> **教学点**：这是 **monkey patch（猴子补丁）**——运行时替换库的函数，无需修改库源码。前提是**你得读源码**。不翻ultralytics 源码，根本不知道这个 10.0 是硬编码的。

### 6.2 坑二：fp16 导致 BN 层污染（最具隐蔽性）

**现象**：hi1536 v1 实验（fp16）官方口径 mean mAP 只有 0.272，而同配方基线是 0.572。

**机理**：

1. fp16 最大值是 65504
2. neck 中 39 个 BN 层的激活值平方后（200～300 平方）逼近上限
3. cuDNN 的 BN 方差统计溢出，导致 `running_var` 被清零
4. 推理时 eval 模式要用 `sqrt(running_var + eps)`，归零后被 `eps` 放大 100～400 倍
5. **mAP 直接崩为 0**

**检测手段**——训练时挂 BN 哨兵：

```python
# scripts/train_mosaic1280.py
def bn_watchdog(trainer):
    zeroed_all = []
    for mdl in (trainer.model, getattr(trainer.ema, "ema", None)):
        if mdl is None:
            continue
        sd = mdl.state_dict()
        zeroed = [k for k, v in sd.items()
                  if "running_var" in k and v.abs().max().item() == 0]
        zeroed_all += zeroed
    with open(HEALTH, "a") as f:
        f.write(f"epoch {trainer.epoch}: running_var归零层数={len(zeroed_all)}\n")
    if zeroed_all:
        MARKER.write_text(f"epoch {trainer.epoch}: {len(zeroed_all)}层归零\n")
        raise RuntimeError(f"BN污染: {len(zeroed_all)}层 running_var 归零, 中止训练")

model = RTDETR(INIT)
model.add_callback("on_train_epoch_end", bn_watchdog)
```

> ⚠️ **这个坑最阴险的地方**：要**9 个 epoch 以上**才显现。1～2 epoch的冒烟测试根本测不出来——所以早期曾得出「AMP 无罪」的错误结论，直到长跑才暴露。

**修复方案：改用 bf16**

```python
import ultralytics.utils.torch_utils as TU
import ultralytics.engine.trainer as TR

def _bf16_autocast(enabled=True, device="cuda"):
    return torch.amp.autocast(device, dtype=torch.bfloat16, enabled=bool(enabled))

TU.autocast = _bf16_autocast
TR.autocast = _bf16_autocast
TR.check_amp = lambda model: True    # 关键：check_amp 会把 bf16 误判为不支持并静默退回 fp32
```

**为什么 bf16 能解决**：bf16 的指数位与 fp32 相同（8 位），动态范围与 fp32 一致，不会像 fp16（指数位 5 位，最大 65504）那样溢出。

**为什么还要 patch `check_amp`**：ultralytics 的 `check_amp` 用 `atol=0.5` 做数值对比测试，会把 bf16 判定为「不支持」然后**静默禁用 AMP**，实际跑成 fp32。必须连它一起patch。

---

## 第七章 实验结果与诚实的结论

### 7.1 官方指标口径

天池官方指标 = **mean(mAP@IoU0.1, mAP@IoU0.3, mAP@IoU0.5)**，初赛总分 = `0.2 × ACC + 0.8 × mAP`。

> **易错点**：只看 mAP@0.5 会低估宽松口径的表现。IoU 0.1 和 0.3 更宽松，对定位精度要求低，分数会明显更高。

### 7.2 核心结果

| 配置 | mAP50 | mean mAP |
|------|-------|---------|
| 20 类基线 | 0.596 | 0.641 |
| 18 类基线 | 0.541 | 0.572 |
| 18 类 + rect + mosaic（最佳）| 0.592 | **0.6349** |

与同类公开方法对比（Tianchi 20 类口径）：

| 方法 | mAP50 |
|------|-------|
| SPFFNet（YOLOv11）| 65.8% |
| 改进 RT-DETR（浙大学报）| 60.0% |
| Fab-ME（YOLOv8s）| 59.4% |
| **本项目 RT-DETR-L** | **59.4%** |

### 7.3 七维奖励的实测效果

`β=0.5` 与 `β=0` 对照：

| 指标 | 变化 |
|------|------|
| 全局召回 | **+0.030**（0.523 → 0.553）|
| 精确率 | −0.022 |
| mAP50 | 基本持平（+0.004）|

> **为什么召回涨了而 mAP 没涨**：模型把边界框「多报」了一些来提召回，代价是精确率下降。这恰好印证 D04/D05（漏检 vs 误检）这对维度的设计意图——**它在两个目标之间做了一次显式的交易，而不是像加权求和那样糊在一起**。

### 7.4 必须诚实说明的负面结论

在更强的基线上，七种训练信号干预**全部输给均匀微调**（七负消融）：

- 改权重类干预（R2 / BoxWeighted）、改度量类（NWD）、成本敏感、GRPO 均未转正
- 系统性呈现**召回与精度的零和关系**（mean −0.015 ~ −0.029 换 crit_recall +0.10 ~ +0.18）
- 唯一明确正结果：**objectness 线性探针**（假阳性率降到 1/2 ~ 1/6）
- Layer-1 可微奖励是唯一零和被打破者，但未过 +0.005 的判定线

### 7.5 最有价值的发现：标签噪声天花板

**断经（`broken_warp`）召回只有约 0.45。**项目做了一整轮排查：奖励注入、GRPO、置信度校准、过采样精调、混淆矩阵 +裁剪图人工复核。

**最终结论**：断经↔磨痕、纬缩↔粗纬在图像上视觉几乎相同、双向混淆，属于**标签噪声，不是模型可训练的差距**。任何训练干预都无法实质提升。合并视觉相同类后召回显著回升（纬缩+粗纬 0.40 → 0.88，断经+磨痕 0.52 → 0.79）。

> **这个案例的教学价值极高**：它区分了「模型不行」和「数据不行」。工业界花大价钱收集标注、修复 label noise，往往比换模型收益更大。
>
> 由此项目还在**告警层**做了兜底：磨痕（minor）视觉上可能就是断经（critical），告警时按 `effective_alert_severity()` 升级到 critical 触发停机。存储的 `severity` 保持规范值，升级级别以 `escalated_severity` 独立落库，可审计。

### 7.6 数据驱动的类别聚类

项目对 18 类做了层次聚类 + bootstrap 稳定性检验（`scripts/cluster_defect_families.py`），结论：

- **k=3 族稳定**（bootstrap support = 1.00）：线性结构族、斑渍族、点状族
- **k=6 族不被数据支持**（稳定性仅 0.286），退化为 3 组 + 3 单例

训练集类别分布为长尾形态，最大类（三丝）1779 个框，最小类（花板跳）117 个框，相差 **15.2 倍**：

| 结构族 | 成员类别 |
|--------|---------|
| 线性结构族 | 三丝、吊经、松经、粗经、星跳、粗纬 |
| 斑渍族 | 污渍、浆斑、稀密档、百脚、花板跳 |
| 点状族 | 破洞、结头、毛粒、纬缩、整经结、断氨纶 |

线性结构族内部的跨族互混占全部错误的 **70%**——细线状缺陷的分类天花板由此定位。

---

## 第八章 常见问题 FAQ

**Q：为什么不用现成的 YOLO？**

RT-DETR 端到端免 NMS，实时性更适合产线；且项目重点在奖励机制，换检测器不影响七维框架——这正是 `InferenceEngine` 抽象基类的价值所在（`backend/inference/engine.py` 定义抽象接口，`create_engine()` 工厂函数在 dummy / onnx / pytorch / rtdetr 四种后端间切换）。

**Q：为什么不用自定义 Dataset，直接用 ultralytics 的？**

项目同时保留了两套：`backend/training/dataset.py` 的 `FabricDataset` 供自研训练循环使用，`scripts/convert_tianchi_to_yolo.py` 产出的 YOLO 格式供 ultralytics API 使用。原因是 ultralytics 的验证器与训练器耦合较深，直接复用比重写更可靠。

**Q：七维奖励为什么不直接加权求和？**

加权求和会把 [−1, 1] 和 [−2, 1] 不同量纲混在一起，训练不稳定。代码保留了七维原始信号（`compute_all` 返回字典），再由标量项做归一化聚合，并把维度权重做成可学习参数。

**Q：可微奖励真的有效吗？**

诚实回答：**部分有效，且有明确的边界**。在弱基线上，奖励注入使全局召回 +0.030；换到强基线上，七种干预全部输给均匀微调。**把失败实验讲清楚，比只讲成功更能说明什么叫严谨的科研。**

**Q：worker 数为什么设成 0？**

Windows 平台下多进程 DataLoader 会触发 `WinError 1455`（内存不足），项目统一设 `workers=0`。

**Q：为什么 optimizer 用 `auto` 而不是指定 AdamW？**

项目实测：MuSGD 稳定，换 AdamW 会塌陷。`optimizer="auto"` 时 ultralytics 会自行计算学习率，会忽略传入的 `lr0`；若要严格使用 `lr0` 则需显式指定 `AdamW`。

---

## 附录 一分钟讲稿总结

> 数据从天池竞赛平台获取（3 个带标注训练集 + 2 个测试集，共 8570 个标注框）。清洗阶段做了三件事：**哈希级去重**（partA 与 partB 的标注文件字节完全相同，不去重会让 GT 翻倍）、**格式转换与归一化**（xyxy 转 cxcywh，含三处防御性钳制）、**负样本保留**（957 张验证图里 374 张是空的，正是真实产线的先验）。图像以**拉伸缩放到 1280×1280、仅除 255 不做 ImageNet 归一化**的方式输入 RT-DETR。
>
> 训练采用 COCO 预训练的 RT-DETR-L 做**全参数微调**，Backbone、Neck、Decoder 全部参与梯度更新，只有 18 类分类头因输出维度不匹配被重建。本项目的创新不在权重而在**损失函数**：通过子类化 ultralytics 的 criterion，在返回值里追加一个基于匈牙利匹配的可微七维奖励项，实现了对第三方框架的零侵入式扩展——只需往损失字典里塞一个键，训练循环自动把它加进去。
>
> 过程中踩了两个真实的坑：梯度裁剪过松导致 NaN（通过 monkey patch 把硬编码的 10.0 降到 1.0），以及 fp16 导致 neck 中 39 个 BN 层的 `running_var` 归零、推理时被 `eps` 放大百倍使 mAP 崩为 0（改用 bf16 修复，且必须同时 patch `check_amp` 否则会静默退回 fp32）。后者需要 9 个 epoch 以上才显现，具有很强隐蔽性。
>
> 最终官方口径 mean mAP 达到 0.6349。更重要的是，通过一整轮排查我们证明了：**断经召回低是标签噪声造成的天花板，不是模型能力问题**——合并视觉上不可区分的类别后，召回从 0.52 回升到 0.79。这个结论提醒我们，数据质量往往比模型选择更能决定效果上限。
