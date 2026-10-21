# 论文骨架 · Paper Skeleton

> **状态**：骨架 v0.6（v0.5 + 粗分类器验证：非冻结微调 3 族 top-1 = 0.9119 [0.894, 0.929]，冻结线性探针 0.8835；A↔C 跨族互混是主要残余错误） · **语言**：中文正文 + 英文标题/摘要/图表标注
> **篇幅**：本骨架 3–5 页，扩写目标 8–12 页（期刊式）
> **生成方式**：ARS `academic-paper` 方法论 · **引用状态**：3 篇同类对比已通过 arXiv/Crossref 核验（Semantic Scholar API 限流），其余待核验（IRON RULE：禁伪造文献）
> **诚实边界**：所有数字均来自仓库内 `runs/rtdetr*/*/results.csv` 与记忆库 A/B 记录，未达 0.90 目标即如实呈现，不以数据集替换虚报。

---

## 标题 Title

- **中文**：七维混合奖惩驱动的织物瑕疵智能检测：RT-DETR 基线、标签噪声与低置信度漏检的双层天花板
- **English**：Seven-Dimensional Hybrid Reward for Fabric Defect Detection: An RT-DETR Baseline and the Dual Ceiling of Label Noise and Low-Confidence Misses

---

## 摘要 Abstract

### English Abstract

**Background** — Fine-grained fabric defect detection requires recognizing 18–20 visually confusable defect classes from long-tailed, weakly-contrastive texture images. Existing detectors optimize a single detection loss and ignore the asymmetric industrial cost between a *miss* (defective fabric shipped) and a *false alarm* (good fabric discarded).

**Purpose** — We aim to steer detection toward critical-defect recall via a reward signal, and to characterize the empirical ceiling imposed by label noise.

**Method** — We train an RT-DETR-L baseline on the Tianchi fabric-defect benchmark (18 merged classes), then inject a differentiable seven-dimensional hybrid reward (D01 localization, D02 classification, D03 calibration, D04 miss, D05 false-positive, D06 broken-yarn sensitivity, D07 skip/weave sensitivity) into the detection loss. We further apply a label-noise ceiling diagnosis — confusion-matrix analysis, bidirectionality test, and cropped-image review — that merges visually identical class pairs.

**Findings** — The baseline reaches mAP50 = 0.596 (20-class) / 0.541 (18-class); under the official contest metric (mean of mAP at IoU 0.1/0.3/0.5, verified from the contest page) this is 0.641 / 0.572. A two-level diagnosis exposes the ceiling. *Level 1 — label noise*: merging confusable class pairs recovers recall sharply (weft_shrink+coarse_weft 0.40→0.88, broken_warp+surface_mark 0.52→0.79). *Level 2 — low-confidence misses*: after the merge, the critical classes (hole / broken_warp / broken_spandex) are *missed below threshold*, not misclassified (broken_spandex: 1 misclassified vs. 8 missed out of 54 GT instances, 5 of the 8 with a sub-threshold detection; only 7 of 166 critical-class instances misclassified at conf=0.25); lowering the confidence threshold lifts critical recall 0.57→0.91 but collapses precision (broken_warp 0.56→0.08), so true and false detections are inseparable by confidence. Seven training-signal interventions — differentiable reward injection, GRPO, classification severity weighting (4 arms), box-loss weighting, and a Normalized-Wasserstein-Distance loss geometry — all fail to beat a uniform fine-tune control on critical recall. A *calibration-layer* intervention, by contrast, is the first to move the binary defect/non-defect gate: a cost-sensitive objectness head (a linear probe on the frozen decoder) cuts cross-domain false-positive rate 2–6× at matched recall (image recall 0.48 vs. 0.01 at a 4% FPR), though its frozen-feature ceiling leaves 0.90-recall FPR at 0.67.

**Implications** — The bottleneck is not the training signal but the feature-level discriminability of thin defects: their low confidence faithfully reflects hard localization, not mis-calibration (consistent with the DETR "specialist strategy" of one-to-one matching). The diagnosis — confusion matrix → threshold calibration → systematic ablation — is a transferable methodology for cost-sensitive, fine-grained recognition.

**Keywords**: fabric defect detection; RT-DETR; hybrid reward; label noise; recall–precision trade-off; GRPO

### 中文摘要

**研究背景** — 织物瑕疵检测需在细粒度、长尾、弱对比的纹理图像中识别 18–20 个视觉高度易混淆的瑕疵类别；现有检测器以单一检测损失为唯一优化目标，忽视了"漏检（带病布匹出厂）"与"误检（良品误判报废）"之间不对称的工业代价。

**研究目的** — 通过奖励信号将检测训练导向关键类召回，并刻画标签噪声所施加的经验天花板。

**研究方法** — 在 Tianchi 布匹疵点基准（合并后 18 类）上训练 RT-DETR-L 基线，将可微的七维混合奖惩（D01 定位、D02 分类、D03 校准、D04 漏检、D05 误检、D06 断经/断氨纶敏感度、D07 星跳/死皱敏感度）注入检测损失；进一步采用标签噪声天花板诊断（混淆矩阵 + 双向性判定 + 裁剪图复核）合并视觉同源类。

**研究发现** — 基线达到 mAP50 = 0.596（20 类）/ 0.541（18 类）；按天池官方指标（IoU 0.1/0.3/0.5 三阈值 mAP 均值，已经赛题页核验）为 0.641 / 0.572。双层诊断揭示天花板：第一层为标签噪声——合并视觉同源类后召回显著回升（纬缩+粗纬 0.40→0.88、断经+磨痕 0.52→0.79）；第二层为低置信度漏检——合并后关键类（破洞/断经/断氨纶）是"低于阈值漏检"而非"错分"（conf=0.25 下断氨纶 54 个 GT 实例仅 1 例错分、8 例漏检且其中 5 例有低于阈值的检出；三类错分合计仅 7/166），降低置信度阈值可将关键类召回 0.57→0.91，但精确率崩塌（断经 0.56→0.08），真检与误检无法按置信度分离。七种训练信号干预——可微奖励注入、GRPO、分类严重度加权（4 臂）、box 损失加权、以及 NWD 损失几何——均未能超越均匀微调对照；而**标定层**干预首次推动二分类门限：代价敏感 objectness 头（冻结解码器上的线性探针）在匹配召回下将跨域误报率降低 2–6 倍（4% 误报率下图像召回 0.48 vs 0.01），但其冻结特征天花板使 0.90 召回下误报率仍达 0.67。

**研究意义** — 证明瓶颈不在训练信号，而在细缺陷的特征级判别力：低置信度是困难定位的忠实反映而非校准失误（与一对一匹配的"专家策略"一致）；"混淆矩阵→阈值标定→系统消融"诊断流程对成本敏感型细粒度识别任务具有可迁移的方法论价值。

**关键词**：织物瑕疵检测；RT-DETR；混合奖惩；标签噪声；召回—精度权衡；GRPO

---

## 论文结构 Outline

### 1 引言 Introduction

- **核心论点**：织物瑕疵检测的工业价值在于关键类（断经/断氨纶/破洞）召回，而非全局平均精度；现有单一损失训练范式无法体现漏检/误检的不对称代价。
- **要点**：
  - 范式演进：Gabor/LBP/GLCM 手工特征 → Faster R-CNN/SSD/YOLO → DETR/RT-DETR（全局注意力、一对一匹配、免 NMS）。
  - 问题一：训练目标单一，未纳入漏检/误检工业成本差异 → 关键类召回难以针对性提升。
  - 问题二：细粒度长尾 + 类别高度混淆，Transformer 检测器在该场景仍处探索期。
  - **主要创新点**：
    1. **可微七维混合奖惩框架（核心创新）**——把定位 / 分类 / 校准 / 漏检 / 误检 / 关键类敏感度（断经·断氨纶、星跳·死皱）七个维度构造成对 logits 与边界框**可微**的奖励信号，首次将"漏检 vs 误检"的不对称工业代价显式注入 RT-DETR 端到端训练。**对效果的贡献**：使检测器训练目标由"单一损失"升级为"代价感知"，可直接牵引关键类召回（实测整体召回 +0.030）。
    2. **标签噪声天花板诊断方法论**——混淆矩阵 + 双向性判定 + 裁剪图复核三步诊断，合并视觉同源类（20→18 类），召回率 0.40→0.88、0.52→0.79。**对效果的贡献**：把提升努力从"换更大模型 / 更强损失"重新导向"修数据"这一真正瓶颈。
    3. **GRPO 奖励引导的检测器训练**——查询噪声 REINFORCE + 组内相对优势 + KL 锚定，免 critic、抗 reward hacking。**对效果的贡献**：为"以工业代价为目标的端到端检测训练"提供可扩展的下一层框架。
    4. **训练稳定性诊断**——定位 deformable attention（`grid_sample`）fp16 溢出为 NaN 发散根因，fp32 修复后全程 0 NaN。**对效果的贡献**：使 18 类模型首次稳定收敛。
- **图 1　技术路线图（五阶段 pipeline）**

  ![图 1 技术路线图](figures/fig_pipeline.png)

  *Figure 1. Pipeline overview: data → RT-DETR baseline → seven-dimensional reward injection → label-noise diagnosis → class merge & evaluation (GRPO as the next layer).*

### 2 相关工作 Related Work

- **要点**：
  - 通用检测：DETR、RT-DETR、YOLO 系列（各自机制与优势）。
  - 织物瑕疵检测：天池"布匹疵点智能识别"、AITEX、ZJU-Leaper 等公开基准；现有 Tianchi 公开方案 mAP50 普遍 0.59–0.66（Fab-ME 59.4%、改进 RT-DETR 60.0%、SPFFNet 65.8%）。
  - 奖励/RLHF 训练：RLHF、PPO、GRPO（组内相对优势、免 critic），及其在检测器训练中的迁移。
- **定位句**：本项目区别于"换更强骨干"路线，聚焦"以工业代价为目标的端到端检测训练"与"标签噪声天花板"两个未被充分研究的问题。

### 3 方法 Method

#### 3.1 RT-DETR 检测基线
- 采用 ultralytics RT-DETR-L，输入 1280，fp32（`--amp False`，见 4.2 稳定性说明）。
- 20 类 → 合并 18 类（见 3.3）。

#### 3.2 七维混合奖惩（D01–D07）
- 将七个维度构造成对 logits 与边界框**可微**的奖励信号（soft IoU / smooth accuracy / calibration gap / sigmoid soft miss & fp / softmax 关键类质量）。
- 注入方式：`loss_reward = -β·mean(R)` 并入检测损失，β 为奖励强度；再以 GRPO 组内相对优势 + KL 锚定探索"奖励引导的训练"。
- **对模型效果的贡献（逐维机制 + 实证）**：
  - D01 定位（soft IoU）／D02 分类（smooth accuracy）／D03 校准（calibration gap）→ 牵引框质量、类别判别与置信度校准。
  - D04 漏检／D05 误检（sigmoid soft miss & fp）→ 直接编码"漏检代价 ≫ 误检代价"的工业不对称。
  - D06 断经·断氨纶／D07 星跳·死皱（softmax 关键类质量）→ 显式牵引关键类召回。
  - **实证（Layer-1 A/B，5 epoch，20 类模型）**：漏检纳入奖励后整体召回 **+0.030（0.523→0.553）**，关键类 broken_warp +0.032 / star_skip +0.057 / weave_defect +0.128；代价是精确率 −0.022、broken_spandex −0.100、mAP50 +0.004（持平）。
  - **深层贡献**：该 A/B 精确暴露"召回—精度"零和边界，证明当前天花板在标签层而非训练信号——这是七维奖惩最具价值的方法论输出（把后续努力重新导向数据，而非堆叠损失）。
- **表 1　七维混合奖惩 D01–D07 维度定义与对效果的贡献**：

  | 维度 | 名称 | 可微构造 | 工业语义 | 对效果的贡献 |
  |------|------|----------|----------|--------------|
  | D01 | 定位 | soft IoU | 框回归质量 | 牵引边界框精度 |
  | D02 | 分类 | smooth accuracy | 类别判别正确性 | 牵引类别判别 |
  | D03 | 校准 | calibration gap | 置信度可信度 | 置信度校准 |
  | D04 | 漏检 | sigmoid soft miss | 带病出厂代价最高 | 直接惩罚关键漏检 |
  | D05 | 误检 | sigmoid soft fp | 良品误判报废 | 惩罚误报 |
  | D06 | 断经·断氨纶敏感度 | softmax 关键类质量 | 高危关键类 | 显式牵引关键类召回 |
  | D07 | 星跳·死皱敏感度 | softmax 关键类质量 | 高影响关键类 | 显式牵引关键类召回 |

#### 3.3 标签噪声天花板诊断
- 三步流程：混淆矩阵 → 双向性判定 → 裁剪图复核。
- 合并视觉同源类：断经↔磨痕、纬缩↔粗纬（20→18 类）。

#### 3.4 关键类召回导向的 GRPO（Layer-3，扩展）
- 查询噪声 REINFORCE + 组内相对优势 + KL；似然比得分修复（`log_std` 梯度非零）。
- 定位：当前为骨架/冒烟验证，作为"奖励引导"的下一层（见 6 展望）。

### 4 实验 Experiments

#### 4.1 数据集与实现
- Tianchi 布匹疵点智能识别：训练集约 8620 张、验证集 957 张，18 类（合并后）。
- 实现：RT-DETR-L、imgsz 1280、batch 4、auto 优化器、60 epochs。
- **表 2　18 类（合并后）名称 / 中文名 / 严重度**（样本量待统计；`broken_warp` 吸收 `surface_mark`、`weft_shrink` 吸收 `coarse_weft`）：

  | id | 类名 | 中文 | 严重度 |
  |----|------|------|--------|
  | 0 | hole | 破洞 | critical |
  | 1 | stain | 污渍 | medium |
  | 2 | three_silk | 三丝 | minor |
  | 3 | knot | 结头 | minor |
  | 4 | flower_board | 花板跳 | major |
  | 5 | hundred_feet | 百脚 | major |
  | 6 | hair_particle | 毛粒 | minor |
  | 7 | coarse_warp | 粗经 | minor |
  | 8 | loose_warp | 松经 | minor |
  | 9 | broken_warp | 断经 | critical |
  | 10 | hanging_warp | 吊经 | major |
  | 11 | weft_shrink | 纬缩 | minor |
  | 12 | size_stain | 浆斑 | medium |
  | 13 | warping_knot | 整经结 | minor |
  | 14 | star_skip | 星跳 | major |
  | 15 | broken_spandex | 断氨纶 | critical |
  | 16 | dense_section | 稀密档 | major |
  | 17 | weave_defect | 死皱 | major |

#### 4.2 基线性能
- 20 类：mAP50 = **0.596**、mAP50-95 = 0.334（50 epochs）。
- 18 类（合并后）：best mAP50 = **0.541** @ e56，末轮 0.537，mAP50-95 = 0.295，P 0.536，R 0.519（60 epochs，0 NaN）。
- 稳定性贡献：定位并修复 fp16 AMP 溢出（deformable attention `grid_sample` 为 fp16 不稳定点），`amp=False` 全程无 NaN —— 作为工程/训练稳定性小节。
- **图 2　训练收敛曲线（RT-DETR-L · 18 类 · 60 轮 · 全程 0 NaN）**

  ![图 2 训练曲线](figures/fig_training_curves.png)

  *Figure 2. Training curves: mAP50 / mAP50-95 (top) and Precision / Recall (bottom) vs. epoch; best mAP50 0.541 @ epoch 56.*

#### 4.3 奖励注入 A/B（Layer-1）
- 5 epoch A/B（同代码、同 val；⚠️ 该实验在 **20 类**模型上进行，与 4.2 的 18 类 from-scratch 基线为两条独立证据）：漏检纳入奖励后整体召回 **+0.030（0.523→0.553）**，精确率 **−0.022**，mAP50 持平（+0.004，噪声）。
- 关键类召回分化：broken_warp +0.032、star_skip +0.057、weave_defect +0.128，但 broken_spandex **−0.100**（最常见关键类）。
- 结论：Layer-1 奖励注入能移动召回，但以精确率为代价，且回退主导关键类——刻画"召回—精度"零和边界，指明瓶颈在标签层。
- **图 3　奖励注入 A/B 对比（匹配奖励 vs 漏检注入）**

  ![图 3 奖励 A/B](figures/fig_reward_ab.png)

  *Figure 3. Reward-injection A/B: overall Recall / Precision / mAP50 (left) and critical-class recall changes (right); include-misses raises overall recall +0.030 but regresses broken_spandex −0.100.*

#### 4.4 混淆类合并（标签噪声天花板验证）
- 合并后召回率显著回升：纬缩+粗纬 0.40→0.88、断经+磨痕 0.52→0.79。
- **表 3　混淆类合并前后关键类召回**：

  | 合并组 | 类 A 召回 | 类 B 召回 | 合并后召回 |
  |--------|-----------|-----------|------------|
  | 纬缩 ∪ 粗纬 | 0.40（纬缩） | 0.79（粗纬） | **0.88** |
  | 断经 ∪ 磨痕 | 0.52（断经） | 0.83（磨痕） | **0.79** |

#### 4.5 低置信度漏检诊断（混淆矩阵 + 阈值标定）

合并后关键类召回仍停滞在 0.48–0.72（hole 0.718 / broken_warp 0.521 / broken_spandex 0.484）。为定位根因，对 18 类基线做两层诊断：

**（一）混淆分解——漏检而非错分。** val 957 张、conf=0.25、IoU 0.5，GT 实例级分解（协议=官方 val 方形 letterbox；"正确"判据与 ultralytics r_curve 类感知匹配同口径，实例级召回与阈值扫描 CSV 逐类核对一致；脚本 `scripts/eval_confusion_decomp.py`，存档 `runs/confusion/decomp_tianchi18_tau{0.25,0.473,0.001}.json`）：

- **表 4　关键类 GT 实例级混淆分解（conf=0.25，IoU 0.5）**

  | 类 | GT 实例 | 正确 | 错分 | 漏检 | 漏检中有 <τ 检出 | 主要错分去向 |
  |---|---|---|---|---|---|---|
  | hole | 39 | 33 | 4 | 2 | 2 | knot←2、star_skip←1、hair_particle←1 |
  | broken_warp | 73 | 51 | 2 | 20 | 14 | dense_section←1、loose_warp←1 |
  | broken_spandex | 54 | 45 | 1 | 8 | 5 | three_silk←1 |

- **阈值敏感性**（正确/错分/漏检）：conf=0.473 → 28/5/6、37/3/33、25/1/28；conf=0.25 → 33/4/2、51/2/20、45/1/8；conf=0.001 → 39/0/0、65/2/6、51/0/3。
- 关键结论：**错分是极少数（三类合计 7/166），漏检是主体且多为"低于阈值的检出"**——断氨纶 54 个 GT 仅 1 例错分、8 例漏检中 5 例低于阈值；阈值降到 conf=0.001 后真漏检仅剩 broken_warp 6 / broken_spandex 3 / hole 0。样本量排除稀有度假设（hole=273 / broken_warp=681 / broken_spandex=492 为中频，真正的长尾是 flower_board=117、hundred_feet=145）。
- 与 `box.r` 的对账：conf=0.25 召回 0.846/0.708/0.833 已高于 max-F1 的 0.718/0.521/0.484，但精确率同步崩塌（表 5）——降阈值把低于阈值的检出放回来，同时放进成倍误报；"漏检 = 低于任何可用全局阈值"是**校准/置信度**问题，不是检测/分类问题。

**（二）阈值标定——召回可抬但精确率崩塌。** 在逐类 P/R 曲线上扫置信度：

- **表 5　成本敏感关键类召回（三种阈值口径，均值 hole/broken_warp/broken_spandex）**

  | 口径 | 阈值 | crit_recall | broken_warp P | broken_spandex P |
  |---|---|---|---|---|
  | 全局 max-F1 | conf=0.473 | 0.573 | 0.56 | 0.49 |
  | 逐类 max-F1 | 各类自优 | 0.593 | — | — |
  | 召回优先 | conf=0.10 | **0.910** | **0.08** | **0.10** |

- 关键结论：**不存在「高召回 + 可用精确率」的阈值**——降阈值可把关键类召回 0.57→0.91，但 broken_warp 精确率 0.56→0.08、broken_spandex 0.49→0.10，真低置信检测与误报无法按置信度分离。
- 根因：RT-DETR 的 VarifocalLoss `gt_score` == 预测 IoU，故"低置信度" == "细缺陷定位自评弱"（断经/断氨纶纱线、小洞等细长弱对比目标）。

#### 4.6 训练信号干预的系统消融（七负）

针对上述根因，对**训练信号**做了七种干预（奖励注入、奖励塑形、分类加权、定位加权、定位度量），全部以严格 A/B 对照验证：

- **表 6　训练信号干预系统消融（关键类召回 @ max-F1）**

  | # | 干预 | 机制层 | 关键指标 | 对照均匀微调 | 判定 |
  |---|---|---|---|---|---|
  | 1 | Layer-1 可微奖励注入（β=0.5） | 奖励 | 整体召回 +0.030，broken_spandex −0.100 | — | 负 |
  | 2 | Layer-3 GRPO（β=0.5，3ep） | 奖励 | crit_recall 0.574→0.344 | — | 负 |
  | 3 | Direction B 分类严重度加权（4 臂） | 分类损失 | full 0.541 / positive-only 0.591 | < 0.606 | 负 |
  | 4 | R2 box 加权（critical 2.0×） | 定位损失权重 | crit_recall 0.598，mAP50 0.522 | < 0.606 | 负 |
  | 5 | R3 NWD 损失几何 | 定位损失度量 | crit_recall 0.598，mAP50 0.518 | < 0.606 | 负 |

- 逐条结论：
  1. **Layer-1**：漏检纳入奖励使整体召回 +0.030，但最常见关键类 broken_spandex −0.100、mAP50 持平（20 类模型，见 4.3）。
  2. **Layer-3 GRPO**：β=0.5 三 epoch 使 crit_recall 0.574→0.344（std_reward≈1e-4 使 REINFORCE 失效、log_std→−4.78、均值梯度劣化模型）。
  3. **Direction B**：分类项严重度加权 4 臂中，full-term 0.541、positive-only 0.591 **均不敌**均匀微调 0.606（positive-only 只是部分修复 full-term 的误检过抑制）。
  4. **R2**：把 box 臂（L1+GIoU）按被匹配 query 的目标类严重度加权，crit_recall 0.598 仍 < 均匀 0.606。
  5. **R3**：把 box 损失从 IoU/GIoU **换成** NWD（高斯化 Wasserstein，理论上最适细目标），并把 varifocal `gt_score` 的 IoU 一并换为 NWD，crit_recall 0.598 仍 < 均匀 0.606。
- **阈值标定复查（R2/R3 的最终判据）**：R2/R3 训练后的关键类在降阈值时精确率**原封不动崩塌**——broken_warp P@conf0.10 = 0.081（R2）/ 0.082（R3）vs 基线 0.08，broken_spandex P@0.10 = 0.080（R2）/ 0.053（R3）vs 基线 0.10。即**改权重、改度量都未能把真检测的置信度相对误报抬起来**。
- **统一解释（接口错位）**：七维奖惩中 D01 定位 / D03 校准 / D04 漏检三个维度**命中**了故障，但它们的接口形式（聚合 box 指标 / 分类损失梯度）与真实故障层（query 触发 + 定位自评）**不对齐**，故梯度无从着力。更深一层：低置信度是**忠实的低判别力**，与 DETR 一对一匹配的"专家策略"一致——每个目标只有一个"主预测" query，其置信度被训练为自身预测 IoU，而细缺陷的 IoU 天然低且无第二个 query 兜底。

#### 4.7 方向 3：代价敏感 objectness 头（标定层干预）

前两节已证低置信度漏检是**背景分缺失**问题（4.5），且七种**训练信号**干预均无法修复（4.6）。方向 3 换一个**架构/标定层**杠杆：在解码器最终隐状态上新增专用 objectness 头 `Linear(256,1)`，以代价敏感 BCE（关键类 w_crit=3.0、背景 w_bg=0.1）独立训练一个"该 query 是否命中真实目标"的前景/背景分，与 18 路类别 logits 解耦，专治 DETR"无标定背景分"。

两阶段训练：**Stage-1 线性探针**（冻结所有既有参数、仅训 257 个头部参数、3 epoch，保证不损伤检测器）→ Stage-2 联合微调（低 LR，本骨架未跑）。

- **回归检查（探针冻结有效）**：crit_recall 0.5743→0.5776、mAP50 0.537（基线 0.541）、逐类召回不变——检测器权重未被触碰，objectness 头对分类分数 mAP 零影响。
- **B 口径门限对比**（957 张全瑕疵 val + 141 张 AITEX `NODefect_images` 跨域负样本）：
  - **表 7　objectness 头 vs max-class-confidence 的 B 口径（匹配召回）**

    | 口径 | FPR @ R0.90 | FPR @ R~0.61 | FPR @ R~0.54 | R @ FPR~0.04 |
    |---|---|---|---|---|
    | max-class-confidence | 0.99 | 0.47 | 0.43 | ~0.01 |
    | objectness | **0.67** | **0.15** | **0.064** | **0.48** |

- **结论（部分正）**：专用 objectness 头是**显著更好**的前景/背景打分器——匹配召回下跨域 FPR 降 2–6 倍，固定 4% 误报率下图像级召回从 ~1% 提到 ~48%，首次在 B 口径拿到真实标定改善，直接印证 4.5 的"无标定背景分"机制假设。
- **但天花板仍在特征层**：到召回 0.90 时检测器对真瑕疵置信度仍过低，两门限都须压到 τ≈0.05、FPR 依旧爆表（objectness 0.67）。线性探针只能对冻结特征做线性分离，**改不了特征本身的判别力**，与 4.6"瓶颈在特征级判别力"一致——故 Stage-2 联合微调（让特征朝 objectness 目标移动）预期低收益且大概率重演关键类召回回退，未跑。

#### 4.8 消融与评估口径（A+B 诚实口径）
- **官方指标口径（2026-10-15 核实）**：天池赛题页明确"分别在 IoU 阈值 0.1、0.3、0.5 下计算 mAP，最终 mAP 取三个值的平均值"，且初赛总分 = 0.2·ACC + 0.8·mAP。此前论文只报 mAP@0.5，**系统性低估**了官方口径成绩（20 类 0.596→0.641，18 类 0.540→0.572）。复算脚本 `scripts/eval_multIou.py`（官方 val 管线 + 仅替换 IoU 匹配向量，mAP@0.5 复算值与训练 results.csv 逐一吻合），报告存 `runs/multi_iou/*.json`。
- **A（主指标）**：关键类（断经/断氨纶/破洞）召回率 ≥ 0.85（当前基线值待补）+ 全局 mAP@0.5 诚实值 0.54–0.59；官方口径（三阈值均值）0.57–0.64。
- **B（系统级）**：有无瑕疵二分类检测（目标 ≥ 0.90）。4.7 已实测：objectness 门限在 4% 误报率下图像级召回 0.48（vs max-class-conf ~0.01），但 0.90 召回下误报率 0.67——标定层改善但非终解。

#### 4.9 与同类方法对比
- **方法学对比（本项目 vs 传统检测器 vs 朴素奖励/类加权）**：
  | 对比维度 | 本项目 | 传统 YOLO 类检测器 | 朴素奖励 / 类加权 |
  |---|---|---|---|
  | 训练目标 | 单一损失 + 可微七维奖惩 | 单一检测损失 | 非可微 / 后验加权 |
  | 关键类召回导向 | 显式 D06/D07 牵引 | 无 | 部分（无可微构造） |
  | 标签噪声诊断 | 三步系统化 + 类合并 | 少有 | 无 |
  | 端到端 | RT-DETR 免 NMS | 需 NMS | 依赖骨干 |
  | 工业代价建模 | 漏检/误检不对称显式编码 | 无 | 无 |
  | 分级告警 | severity 驱动（critical 停机） | 二值 | — |
  | 训练稳定性 | fp32 + 溢出诊断 | AMP 默认 | — |
- **量化对比（内部 A/B，同代码同 val，20 类模型 5 epoch fine-tune）**：
  | 配置 | 整体召回 | 精确率（相对） | mAP50 |
  |---|---|---|---|
  | 基线（无奖励，β=0）| 0.523 | — | 基准 |
  | 七维奖惩注入（β=0.5 + include-misses）| **0.553（+0.030）** | −0.022 | +0.004（持平） |
- **外部同类方法对比（均在同一 Tianchi 数据集；⚠️ 口径已逐篇核验：公开方案普遍只报 mAP@0.5）**：

  | 方法 | 骨干 | 类别 | mAP@0.5 | 口径核验 | 出处 |
  |------|------|------|-------|---------|------|
  | SPFFNet | YOLOv11 | **8 类子集** | **65.8%** | v3 表 I 明确 mAP@0.5，仅 8 类（Knot/Triple Wire/Coarse Pick/Broken Spandex/Warp Knot/Weft Shrink/Hole/Stain） | arXiv:2502.01445 |
  | 改进 RT-DETR（牛仔布） | RT-DETR-R18 | 20 | 60.0% | 20 类（原文口径） | 浙江大学学报 2025, 59(6):1169 |
  | Fab-ME | YOLOv8s | 20 | 59.4% | 20 类（原文口径） | arXiv:2412.03200 |
  | **本项目 RT-DETR-L（20 类基线）** | RT-DETR-L | 20 | **59.6%** | 本工作 2026-10-15 复算 0.5963 | 本工作 |
  | **本项目 RT-DETR-L（18 类合并）** | RT-DETR-L | 18 | **54.0%** | 本工作 2026-10-15 复算 0.5403 | 本工作 |

- **官方口径三阈值复算（2026-10-15；官方 val 957 张、conf=0.001、max_det=300、imgsz 1280；脚本 `scripts/eval_multIou.py`，mAP@0.5 复算值与训练 results.csv 逐一吻合）**：

  | 模型 | mAP@0.1 | mAP@0.3 | mAP@0.5 | 官方口径（三阈值均值） |
  |------|------|------|------|------|
  | 20 类基线（50ep） | 0.673 | 0.653 | 0.596 | **0.641** |
  | 18 类合并（60ep） | 0.596 | 0.579 | 0.540 | **0.572** |

- **对比结论（诚实）**：① 复算证实本项目 20 类基线 mAP@0.5 = 0.596，与 Fab-ME（59.4%）持平、贴近改进 RT-DETR（60.0%），距 SPFFNet 65.8% 约 6 点——但 SPFFNet 的 65.8% 为 **8 类子集 + mAP@0.5** 口径，与本项目全 18/20 类不可直接对比；② 官方口径（三阈值均值）下本项目 20 类基线 **0.641**——公开方案均未报官方口径，按 mAP@0.1/0.3 通常高于 mAP@0.5 的规律，其官方口径成绩应高于各自 mAP@0.5 报值，故仍为指示性对比；③ 18 类合并的显式代价：官方口径 0.641→0.572（−6.9 点），收益是合并类召回 0.40→0.88 / 0.52→0.79——"mAP 换召回"的权衡数据是标签噪声诊断方法学的核心实证；④ 公开方案均无标签噪声诊断，本项目"18 类合并 + 标签噪声诊断 + 七维奖惩"是差异化方法学贡献，而非单纯拼 mAP。
- ⚠️ **可比性边界**：外部数字虽同数据集，但类别数（20/20/8 vs 本项目 20/18）、骨干（YOLOv11/YOLOv8s/RT-DETR-R18 vs 本项目 RT-DETR-L）、指标口径（mAP@0.5 vs 官方三阈值均值）与训练/测试切分均有差异，全部对比为指示性。另：《棉纺织技术》某文声称 mAP50=90.9%，其原始出处无法打开核验，**不采信**（防虚假引用）。

#### 4.10 数据驱动缺陷族结构（级联粗分类的先验诊断）

**动机**：细粒度级联（粗→细）需要一个"族"级粗分类定义；手工分组含 catch-all，需用数据替换。

**方法**：训练集 8570 个 GT 框（18 类）裁剪图（padding 15%，224×224）过冻结 ResNet50（ImageNet V2）提 2048 维特征，L2 归一化；类级质心 → KMeans（n_init=200, 20 seeds）+ Ward（cosine）对照；k=2..8 扫描 + bootstrap 稳定性（100 轮 × 80% 框重采样）；与 18 类基线 val 混淆矩阵（conf>0.25, IoU>0.45, 2375 计数）交叉验证。

**结果**：
- **k=3 是唯一的稳定结构**：KMeans 与 Ward 划分完全一致，silhouette 0.716（各 k 中最高），bootstrap 内聚 support 三族均 **1.00**：
  - **A 线性结构族**（7 类，2912 框）：三丝、粗经、松经、断经、吊经、跳花、织疵；
  - **B 斑渍族**（5 类，1268 框）：污渍、花板、百脚、浆斑、密路；
  - **C 点状族**（6 类，4390 框）：破洞、结头、毛粒、纬缩、整经结、断氨纶。
- **k=6 不被支持**：稳定性仅 0.286；强制二分每个超族均退化为「主组 + 1 个离群单例」（三丝/污渍/纬缩被剥出）——数据结构是"3 组 + 3 离群点"，非 6 族。k≥4 后稳定性持续崩塌（0.36→0.10）。
- **混淆矩阵验证**：主要易混对落在族内（松经↔断经 12/7、跳花↔织疵 12/7、粗经→吊经 5、结头→整经结 22）；最大错分对三丝↔结头（38/28）为跨族例外（细线状缺陷形似结点）。

**对级联设计的影响**：原计划"6 族→18 细"的粗分类前提被数据推翻，**改为 3 族→18 细**（结构稳定、类数均衡、与混淆结构吻合）。三丝作为族内离群点且是最大跨族错分源，细粒度阶段需单独关注。

**工程备注（坑）**：ultralytics 的混淆矩阵仅在 `plots=True` 时更新（validator `if self.args.plots` 门控），`plots=False` 会静默产出全零矩阵；矩阵索引为 `[pred, gt]`，已统一转置为 `[gt, pred]`。

#### 4.11 粗分类器验证（级联第一级，0.3）

**问题**：级联第一级（族级粗分类）能否达到 90%+？用手工预估的"88~95%"不可靠，用真实数字回答。

**设置**：与 4.10 完全相同的裁剪协议（pad 15%、224×224、ResNet50 ImageNet 预训练）；训练集 8570 个 GT 框（A 2912 / B 1268 / C 4390），val 953 个 GT 框；**非冻结微调**（3 类头 + 类加权 CE + hflip/vflip 增广——90° 旋转会互换经/纬语义故排除；AdamW 1e-4 + OneCycle，25 epoch，e21 最佳）；对照组为同一冻结 2048 维特征上的逻辑回归线性探针。

**结果**（val 953 框，bootstrap 1000 次 95% CI）：

| 模型 | top-1 准确率 | 95% CI | A 召回 | B 召回 | C 召回 |
|------|------|------|------|------|------|
| 冻结特征 + 线性探针 | 0.8835 | [0.864, 0.905] | 0.845 | 0.896 | 0.906 |
| **非冻结微调** | **0.9119** | [0.894, 0.929] | 0.854 | 0.928 | 0.946 |

混淆矩阵（行=GT）：A [281, 10, 38]、B [3, 116, 6]、C [21, 6, 472]。

**结论**：① **3 族粗分类达到 90%+（0.912）**——计划中"6 族 90%+"的目标以 3 族形式达成，且结构更稳（4.10）；② 非冻结微调比线性探针 +2.8 点，微调是必要的；③ 残余错误集中在 A↔C 跨族互混（38+21=59 框，占错误 70%）——线性结构族与点状族的细线状缺陷（三丝 vs 结头）是粗分类天花板，与 4.10 混淆矩阵发现的最大错分对一致；B 斑渍族几乎完全分离（错 9 框）。④ 训练曲线显示 e18 后饱和（~0.910±0.003），25 epoch 充分。

**对级联的意义**：第一级（3 族，91.2%）+ 检测器（mAP50 0.54–0.60）构成系统前端；第二级细分类（18 类）的天花板由检测器 per-class 召回决定（52~74%），符合预期的级联误差累积结构。

#### 4.12 高分辨率路线考察（1536 rect，负面结果）

**问题**：基线（RT-DETR-L @1280 方形 letterbox，60 epoch，mAP50 0.541）之上，提高输入分辨率（2446×1000 原图 → 1536 rect）能否把 mAP50 推向 0.60+？

**纯几何迁移诊断（关键对照）**：将基线 best.pt **不训练**直接在 1536 rect 下推理：mAP@0.1/0.3/0.5 = 0.581/0.566/0.524，官方口径 mean = 0.557（vs 1280 方形 0.572，2026-10-20 复跑复现 0.5566，存档 runs/multi_iou/tianchi18_best_at1536_probe.json）——**推理端几何迁移成本仅 −0.015**，rect letterbox 相比方形还避免了 59% 的算力浪费（原图 2.4:1，方形 1280 内容仅 1280×523）。

**训练端三次尝试（全部失败，均主动中止）**：

| # | 路线 | 结果 | 根因 |
|---|------|------|------|
| 1 | COCO 初始化 + fp16-AMP，1536 rect，60ep | epoch9 起 39 个 neck/encoder BN 层 running_var 被清零，e10 起 mAP=0，早停 e28，最佳 e8 mAP50=0.246 | fp16 方差统计溢出（激活 200–300 → 平方 4e4–9e4 逼近 65504 上限） |
| 2 | 收敛基线（0.541）微调 + lr0=0.0002 | e1–e5 mAP50 0.503→0.409 三连降，损失平台高于基线收敛值 16% | 微调动力学劣化 |
| 3 | 同 #2 但 lr0=0.0001 + warmup 5ep | e1–e6 mAP50 0.503→0.402，同一签名 | 同上，**与 lr 大小无关** |

**中间发现**：#1 的 fp16-AMP BN 污染用 bf16 修复（指数范围同 fp32），并经 #2/#3 共 11 个 epoch 的 BN 哨兵验证零污染——bf16 补丁需同时绕过 ultralytics `check_amp` 的 atol=0.5 误判（否则 AMP 被静默禁用跑成 fp32，显存贴顶 24.2GB、9.9s/步）。

**结论**：① 1536 的**推理收益存在但很小**（−0.015 成本，无净增益证据）；② "已收敛权重微调上新 letterbox 几何"在本配置（MuSGD momentum 0.937）下**稳定劣化**，两次独立 lr 实验证实；③ 唯一健康轨迹是从 COCO 从头训（#1 中毒前 e1=0.150→e8=0.246 单调升）——后续以该路线复测（mosaic=1.0，60ep）。

**复测终局（双臂 60 轮完成，因子分离）**：mosaic+从头训路线双双越过 0.60，且**分辨率因子被干净分离**（两臂仅 imgsz 不同，其余配方逐行一致）：

| 臂 | imgsz | mosaic | 官方口径 mean @1280 | mean @1536 | mAP50 | 每轮耗时 |
|---|---|---|---|---|---|---|
| 基线 tianchi18_full | 1280 | ✗（且方形） | 0.5719 | — | 0.541 | ~19 min |
| v4 | 1536 | ✓ | 0.6077 | 0.6094 | 0.573 | ~20 min |
| **mosaic1280** | **1280** | **✓** | **0.6349** | 0.6106 | **0.592** | ~16 min |
| 无 mosaic 对照臂 | 1280 | ✓ | 0.6080 | — | 0.567 | ~15 min |

- **1280 全面占优 1536**：同配方下 @1280 口径 0.6349 vs 0.6077（+0.027），@1536 口径 0.6106 vs 0.6094（+0.001，打平），且每轮快 25%——**高分辨率在本任务无净增益**（与原图 2.4:1 宽幅、疵点细窄的几何特性一致：rect letterbox 已消除算力浪费，再加分辨率只放大背景）。
- **letterbox 几何的耗时特性（三点实测，results.csv 累计时间差分中位）**：方形无 mosaic 18.5 min/轮、**rect 无 mosaic 15.2 min/轮（最快）**、rect+mosaic 15.4 min/轮、1536 rect+mosaic 20.0 min/轮——rect letterbox 相比方形省 ~18% 墙钟时间，mosaic 基本中性，升 1536 贵 30%。
- **mosaic1280 = 当前最强模型**：官方口径 mean 0.6349（0.666/0.647/0.592），超基线 0.5719 达 **+0.063**；mAP50 0.540→0.592。轨迹：e34 即达基线终值 mAP50 0.541（基线自身 e56 才跨过，**收敛提速约 1.7 倍**），至终局仍在上升。
- **因子解耦终局（三点定位，官方口径 mean）**：基线 0.5719 → **rect 单独（无 mosaic）0.6080（+0.036，占总增益 57%）** → **rect+mosaic 0.6349（mosaic 增量 +0.027，占 43%）**。两个便宜因子量级相当、rect 略大——修正了训练中段"mosaic 主导"的曲线猜测（中段曲线差异主要来自收敛速度而非终值）。**相对基线的 +0.063 = rect 几何 +0.036 与 mosaic 增强 +0.027 的近似可加叠加**。
- **强基线成立**（协议触发条件：mean ≥ 0.60）：七维奖惩重测以 mosaic1280 best.pt 为起点权重启动（见 4.13）。

#### 4.13 强基线复测：七维奖惩的再检验（创新点诚实救赎）

**动机**：4.6 节七负消融全部在弱基线（mAP50 0.541）上完成。强基线（mosaic1280，官方口径 0.6349）上，若干预可能改变符号——必须重测，两种结果都有价值。

**协议**（`docs/plan_stage1.3_rerun.md`，判定规则预先承诺）：与历史七负完全同口径（1280 方形、amp=False、MuSGD、lr0=1e-4、5ep、seed=42），唯一变量=起点权重（0.541 → 0.635 强基线）。每臂 A/B：干预 vs 同配置均匀微调。主指标=官方口径 mean mAP；护栏=关键类 recall@max-F1 + 逐类阈值标定精确率。**转正**=干预 mean > 对照 +0.005 且 crit_recall 不降。

**进度**：全部收官（2026-10-19）。R2/B/NWD/Layer-1/GRPO 五臂 + Layer-1 三种子复测（CONFIRMED_NEGATIVE）均已完成，见 4.13 结果表。

**臂 1 结果（R2 box 严重度加权，2026-10-18 04:50 完成）**：

| 臂 | 官方口径 mean | crit_recall | broken_spandex R@maxF1 |
|---|---|---|---|
| 均匀对照（5ep 微调） | 0.6158 | 0.6260 | 0.4074 |
| severity 加权（5ep） | 0.5927（**−0.023**） | 0.7302（**+0.104**） | **0.6852（+0.278）** |

- 判定：**仍负**（主指标未过 +0.005 线），但呈极端**召回—精度零和**：关键类召回大幅拉升（尤其 broken_spandex +0.278），整体 mAP 同步下滑。
- 与弱基线历史值对比（crit_recall 0.598 vs 0.606，几乎无差别）：**强基线上干预的"着力"强度放大约一个数量级**——同样的权重干预在强模型上能显著搬移关键类指标，但搬移的是容量而非新增可分性，官方口径不为此买单。这比弱基线时期的"无效果"更有力地支持接口错位/忠实低判别力解释。
- 附带发现：均匀对照本身从 0.6349 降至 0.6158（−0.019）——1280 方形 letterbox 对 rect 训练的强模型有轻微微调毒性（两臂同起点、同配置，A/B 内部效度不受影响）。

**臂 2 结果（Direction B 分类严重度加权 positive-only，06:23 完成）**：

| 臂 | 官方口径 mean | crit_recall | broken_spandex R@maxF1 |
|---|---|---|---|
| 均匀对照（同臂 1） | 0.6158 | 0.6260 | 0.4074 |
| severity positive-only | 0.5873（**−0.029**） | 0.8102（**+0.184**） | **0.8704（+0.463）** |

- 判定：**仍负**，但零和剂量比臂 1 更大：三个关键类召回全线上移（hole +0.049 / broken_warp +0.041 / broken_spandex +0.463），mean −0.029。
- **两臂合看的强基线结论**：R2（定位权重）与 Direction B（分类权重）在强基线上均能大幅搬移关键类召回（+0.10 / +0.18），而弱基线时期几乎无效（crit_recall 0.598 / 0.591 vs 均匀 0.606）。干预有效性随基线强度放大一个数量级，但全部表现为召回—精度零和（mean −0.02 ~ −0.03）——**支持"忠实低判别力 + 接口错位"解释，否定"干预无效"的弱基线假象**。

**臂 3 结果（R3 NWD 小目标感知加权，07:53 完成）**：

| 臂 | 官方口径 mean | crit_recall | broken_spandex R@maxF1 |
|---|---|---|---|
| 均匀对照（同臂 1） | 0.6158 | 0.6260 | 0.4074 |
| NWD 加权 | 0.6007（**−0.015**） | 0.7500（**+0.124**） | **0.8148（+0.407）** |

- 判定：**仍负**（三连负），但签名第三次复现：broken_spandex 0.407→0.815（近乎翻倍），hole 0.772→0.846；代价 mean −0.015，为三臂中最小——NWD 的零和漏损比例最低。
- **三臂合看**：三种干预（定位权重 / 分类权重 / IoU 加权）在强基线上全部同一签名——关键类召回大幅上移（+0.10 ~ +0.18），mean 代价 −0.015 ~ −0.029。召回—精度零和不是某一干预的副作用，而是**系统性结构特征**。

**臂 4 结果（Layer-1 可微奖励注入 β=0.5 + 漏检纳入，10ep，14:20 完成）**：

| 臂 | 官方口径 mean | crit_recall | broken_spandex R@maxF1 |
|---|---|---|---|
| 均匀对照（10ep，epoch 匹配） | 0.6157 | 0.7010 | 0.7037 |
| Layer-1 奖励注入 | 0.6164（**+0.0007**） | 0.7541（**+0.053**） | 0.7963（**+0.093**） |

- 判定：按预先承诺规则**负**（mean +0.0007 < +0.005 线）。seed 42 单次实验中为五臂中唯一 mean 与 crit_recall 同向者——**但三种子复测确认该表观是噪声**（见下）。
- 注意对照本身的变化：10ep 均匀微调的 crit_recall（0.7010）显著高于 5ep 对照（0.6260）而 mean 持平（0.6157 vs 0.6158）——**单纯延长均匀微调即可免费获得关键类召回 +0.075**，Layer-1 的 +0.053 是在这个更强对照之上的增量。
- 科学意义：Layer-1（可微奖励、漏检纳入）是唯一接口与故障层部分对齐的干预形式（直接在检测质量上做可微塑形，而非改损失权重/度量）。

**Layer-1 三种子复测（2026-10-19 收官，CONFIRMED_NEGATIVE）**：

| 种子 | 对照 mean | 干预 mean | Δmean | 对照 crit | 干预 crit | Δcrit |
|---|---|---|---|---|---|---|
| 42 | 0.6157 | 0.6164 | +0.0007 | 0.7010 | 0.7541 | +0.0531 |
| 123 | 0.6132 | 0.5956 | −0.0176 | 0.6116 | 0.7309 | +0.1192 |
| 2024 | 0.6123 | 0.6005 | −0.0118 | 0.6339 | 0.6112 | −0.0227 |
| **均值 ± std** | — | — | **−0.0096 ± 0.0093** | — | — | **+0.0499 ± 0.0710** |

- seed 42 的“零和被打破”确认是种子噪声；可微奖励接口与前三种干预同构落入零和。
- 方法教训：单种子 ±0.005 显著性线对种子间离散度（Δmean std 0.0093）过紧——任何单臂结论必须种子复测方可采信。
- 新增脚本：`train_rtdetr_reward.py` 加 `--seed`；`scripts/l1_seed_aggregate.py`；`scripts/l1_seeds_chain.sh`。

**臂 5 结果（Layer-3 GRPO，std_reward 地板修复版，16:25 完成）**：

| 臂 | 官方口径 mean | crit_recall |
|---|---|---|
| 均匀对照（5ep 共享） | 0.6158 | 0.6260 |
| GRPO（eps 1e-4，5ep） | **0.3101（−0.306）** | 0.4320（−0.194） |

- 判定：**全战役最差**——模型被摧毁（mean 腰斩），比历史弱基线 GRPO 崩溃（crit_recall 0.574→0.344，3ep、eps=1e-2）伤害更大。
- 地板修复的反直觉结论：eps 1e-2→1e-4 后 65% 的组 std_reward 越过地板（中位数 1.7e-4），REINFORCE 从"死寂"变为"激活"——但激活的是**噪声梯度**（奖励面在策略采样噪声下本就平坦，中位 std 仅 1.7e-4，真信号占比极低）。5390 步噪声梯度累积后模型崩溃。**在这个奖励面上，降低地板不是修复而是放大伤害**；GRPO 路线应整体放弃（除非重新设计有区分度的奖励）。

**五臂总表（强基线复测，2026-10-18 收官）**：

| 臂 | 干预 | mean Δ | crit_recall Δ | 判定 |
|---|---|---|---|---|
| 1 | R2 box 严重度加权 | −0.023 | +0.104 | 负 |
| 2 | Direction B 分类加权 | −0.029 | +0.184 | 负 |
| 3 | R3 NWD 度量 | −0.015 | +0.124 | 负 |
| 4 | Layer-1 可微奖励 | **+0.0007** | **+0.053** | 负（未过线，但零和被打破） |
| 5 | GRPO（eps 修复） | −0.306 | −0.194 | 负（崩溃） |

**总结论**：强基线上七维奖惩中四种接口形式重测——三种"改权重/改度量"全部复现召回—精度零和（系统性结构特征，支撑接口错位+忠实低判别力解释）；可微奖励形式（Layer-1）seed 42 表观同向但三种子复测确认为噪声（Δmean −0.0096 ± 0.0093），零和在该接口同样复现——**四种机制互异的接口全部落入同一零和，是最稳健的负面结论**；GRPO 实证不可救药。**创新点的诚实定位**：七维奖惩作为"性能增强手段"被数据否定，但作为"故障诊断探针"极其成功——它们精确定位了故障层（query 触发+定位自评）与可行接口，这一否定-定位双重结果比单纯的正结果更有论文价值。

### 5 讨论 Discussion

- **核心论点**：天花板有两层——第一层标签噪声（可修复，类合并）；第二层细缺陷低置信度漏检（七种训练信号干预均无法修复，故障在特征级判别力）。
- **要点**：
  - 召回—精度零和边界：奖励只能"搬移"指标，不能"新增"可分性（Layer-1）。
  - 标签污染 vs 标注歧义两类混淆的区分（断经↔磨痕 = 标签污染；纬缩↔粗纬 = 标注歧义）。
  - 低置信度漏检的**接口错位**解释：七维奖惩的 D01/D03/D04 命中故障，但接口（聚合 box 指标 / 分类损失梯度）与真实故障层（query 触发 + 定位自评）不对齐；更深层是 DETR"专家策略"下的忠实低判别力——每个目标仅一个"主预测" query，其置信度被训练为自身预测 IoU，细缺陷 IoU 天然低且无第二 query 兜底。
  - 三个可写进论文的发现：① **度量不公平**（全局 vs 逐类 max-F1 阈值相差 ~0.02 召回）；② **逐类标定缺失**（各类最优阈值差异大，无统一背景分）——4.7 的 objectness 头证实专用前景/背景分可 2–6 倍降低跨域误报，但冻结特征天花板仍在；③ **漏检奖励的匹配层盲区**。
  - 对工业部署的启示：severity 驱动的分级告警（critical→停机、major→标记）+ **逐类成本敏感阈值**（召回 0.90 可达，误报洪水需人工复核，符合"漏检代价 ≫ 误检代价"）比全局 mAP 更贴近真实代价。

### 6 结论与展望 Conclusion & Future Work

- **结论**：① 七维可微奖励确能驱动关键类召回倾斜，但呈召回—精度零和；② 天花板有两层——标签噪声（类合并可修复）与细缺陷低置信度漏检（七种训练信号干预均无法修复，故障在特征级判别力而非损失/奖励）；③ 双层诊断方法论可迁移（混淆矩阵→阈值标定→系统消融）。
- **展望**：① 特征层干预（SPDConv space-to-depth / AIFF 内可变形注意力，保留细缺陷特征，文献证据 +3.5% mAP50）；② 标定层（objectness/背景头——4.7 已证线性探针有效但冻结特征受限，下一步是 Stage-2 联合微调或与特征层干预结合，同时解决 B 口径与专家策略的"主预测不可识别"）；③ 部署层逐类成本敏感阈值（当前唯一干净正结果）；④ 关键类重标注/主动学习突破标签天花板；⑤ ONNX 边缘部署 + 分级告警系统化评估。

---

## 图表清单 Figure / Table Inventory

| 编号 | 类型 | 内容 | 状态 |
|------|------|------|------|
| 图 1 | 技术路线图 | 五阶段 pipeline 总览 | 已出图 |
| 图 2 | 训练曲线 | mAP50 / Precision / Recall vs epoch（tianchi18_full, 60ep） | 已出图 |
| 图 3 | A/B 对比 | 奖励注入：召回+0.030 / 精确−0.022 / mAP50 +0.004 + 关键类召回 | 已出图 |
| 图 4 | 阈值标定 | 关键类 P/R 曲线随置信度扫（召回 0.57→0.91、精确率崩塌） | 待出图 |
| 表 1 | 维度表 | D01–D07 定义/可微构造/工业语义 | 已填 |
| 表 2 | 类别表 | 18 类中英名 + 严重度 | 已填（样本量待统计） |
| 表 3 | 合并对照 | 混淆类合并前后召回 | 已填 |
| 表 4 | 混淆分解 | 关键类 GT 实例级正确/错分/漏检分解（conf 0.25，漏检非错分） | 已填（2026-10-20 重跑，scripts/eval_confusion_decomp.py） |
| 表 5 | 阈值标定 | 三种阈值口径 crit_recall + 关键类精确率 | 已填 |
| 表 6 | 七负消融 | 七种训练信号干预的 crit_recall 对照 | 已填 |
| 表 7 | objectness 头 | 方向 3 B 口径门限对比（FPR/召回，objectness vs max-conf） | 已填 |
| 表 8 | 同类对比 | 方法学对比 + 内部 A/B 量化 + 外部 3 篇核验 | 已填（外部 3 篇核验；SPFFNet 口径已核实为 8 类 mAP@0.5） |
| 表 9 | 官方口径三阈值复算 | 20/18 类基线 mAP@0.1/0.3/0.5 + 官方均值 | 已填（2026-10-15，scripts/eval_multIou.py） |
| 表 10 | 数据驱动族结构 | k=2..8 扫描（seed 一致性/稳定性/silhouette）+ 3 族定义 + k=6 退化证据 | 已填（2026-10-15，runs/family_clustering/） |
| 表 11 | 粗分类器验证 | 冻结线性探针 vs 非冻结微调：acc/CI/逐族召回/混淆 | 已填（2026-10-15，0.9119 vs 0.8835，scripts/train_family_classifier.py） |
| 图 5 | 类质心 t-SNE | 18 类质心按 3 稳定族着色（support 1.00） | 已出图（tsne_families3.png） |
| 图 6 | 族结构树图 | Ward 层次树（cosine, average linkage） | 已出图（dendrogram.png） |
| 图 7 | 混淆矩阵热力图 | 行归一化混淆（val 957, conf>0.25, IoU>0.45） | 已出图（confusion_heatmap.png） |

---

## 参考文献 References（⚠️ 待校验）

> 以下为"已知真实、待 Semantic Scholar 核验 DOI/条目"的草稿清单，正式稿将逐条过 IRON RULE 校验门。

1. Zhao Y. et al., *DETRs Beat YOLOs on Real-time Object Detection*（RT-DETR），CVPR 2024，arXiv:2304.08069.
2. Carion N. et al., *End-to-End Object Detection with Transformers*（DETR），ECCV 2020，arXiv:2005.12872.
3. Shao Z. et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*（GRPO），arXiv:2402.03300.
4. Schulman J. et al., *Proximal Policy Optimization Algorithms*，arXiv:1707.06347.
5. Ouyang L. et al., *Training Language Models to Follow Instructions with Human Feedback*（InstructGPT/RLHF），NeurIPS 2022，arXiv:2203.02155.
6. Ultralytics YOLOv8 / RT-DETR 实现，https://github.com/ultralytics/ultralytics.
7. 阿里天池"布匹疵点智能识别"竞赛数据集（2019）.
8. Silvestre-Blanes J. et al., AITEX 织物缺陷数据集.
9. Zhang C. et al., ZJU-Leaper 织物缺陷数据集.
10. ✅ Zhao P., Jia S., *SPFFNet: Strip Perception and Feature Fusion Spatial Pyramid Pooling for Fabric Defect Detection*，arXiv:2502.01445（Tianchi 20 类，mAP50 65.8%）.
11. ✅ Wang S., Kong H., Li B., Zheng F., *Fab-ME: A Vision State-Space and Attention-Enhanced Framework for Fabric Defect Detection*，arXiv:2412.03200 / Springer LNCS (ICIC 2025) 15854:103–114，DOI 10.1007/978-981-96-9901-8_9（Tianchi 20 类，mAP50 59.4%）.
12. ✅ Liang G., Han S., *Denim fabric defect detection algorithm based on improved RT-DETR*，浙江大学学报（工学版）2025, 59(6):1169–1178，DOI 10.3785/j.issn.1008-973X.2025.06.008（Tianchi，mAP50 60.0%）.
13. ✅ Park Y.-J., Sobolewski C., Azizan N., *Uncertainty Quantification in Detection Transformers: Object-Level Calibration and Image-Level Reliability*，arXiv:2412.01782（DETR 一对一匹配的"专家策略"与对象级标定误差 OCE）.
14. ⏳ Wang J., Xu C., Yang W., Yu L., *A Normalized Gaussian Wasserstein Distance for Tiny Object Detection*，AAAI 2022（细目标 IoU 敏感性的 NWD 替代，arXiv ID 待核验）.

---

## 关键数据速查 Key Numbers

| 指标 | 值 | 出处 |
|------|-----|------|
| 20 类基线 mAP50 / mAP50-95 | 0.596 / 0.334 | tianchi20, 50ep |
| 18 类基线 best mAP50 | 0.541 @ e56 | tianchi18_full, 60ep |
| 18 类末轮 mAP50 / mAP50-95 | 0.537 / 0.295 | tianchi18_full e60 |
| 18 类 P / R | 0.536 / 0.519 | tianchi18_full e60 |
| **20 类三阈值 mAP@0.1/0.3/0.5** | **0.673 / 0.653 / 0.596** | multIou 复算 2026-10-15（官方口径均值 0.641） |
| **18 类三阈值 mAP@0.1/0.3/0.5** | **0.596 / 0.579 / 0.540** | multIou 复算 2026-10-15（官方口径均值 0.572） |
| **阶段 1 最强（mosaic1280，18 类）** | **mAP@0.1/0.3/0.5 = 0.666/0.647/0.592，官方口径 mean 0.6349** | 1280+rect+mosaic 从头训 60ep，best.pt e53 |
| 因子解耦（官方口径 mean） | 基线 0.5719 → rect 单独 0.6080（+0.036）→ rect+mosaic 0.6349（+0.027） | 三点定位，两因子近似可加 |
| 分辨率对照（同配方） | @1280 0.6349 vs @1536 0.6077（mean），mAP50 0.592 vs 0.573 | 高分辨率无净增益且慢 25% |
| **域内二分类回放（B 口径，mosaic1280）** | AUC 0.9808；最佳正确率 **93.73%** @ τ=0.35（召回 95.2%/误报 8.6%）；τ=0.25 时 91.2%；基线最佳 92.06%；跨域 AITEX 误报 71.6% | scripts/eval_binary_replay.py 2026-10-20 |
| **推理侧七维报警评分回放（2026-10-20）** | 域内不敌 max-conf（主变体 0.9154 / 消融最佳 drop D3+D4 0.9269 vs 0.9383，等召回 FPR 1.5–2x）；跨域无优势（0.7447 vs 0.7163）；关键类召回 0.9586–0.9638 ✓；三档停机45.1/复核20.6/放行34.3，停机档 FPR 5.9%、复核档干净图告警率 17.4%，合并召回 0.967；D2 承重、D3/D4 有害（同训练侧签名）；线性→乘积修正（饱和度，首跑 AUC 0.386） | runs/alarm_replay/REPORT.md |
| **置信度校准审计（论文 4.13，2026-10-20）** | 运行区域（conf≥0.05）ECE 0.1055/0.1116 严重失准（conf 0.5–0.6 实际正确率仅 0.17–0.21，过信 2–3 倍）；单参数 TS 无效（恶化至 0.177/0.172，NLL 最优 T≈0.6 被 FP 质量主导）；isotonic 交叉拟合修复至 0.037/0.046（全区域 0.0005）——失准是曲线形状问题非温度偏移；任何单调重校准不改变 TP/FP 可分性（AUC 0.763/0.823 严格不变）→ 4.12"不可按置信度分离"升级为定量命题；条件校准（族/尺寸）无增益 | scripts/eval_calibration_audit.py，runs/calibration/audit_{tianchi18,mosaic1280}.json |
| **自适应损失加权 UW（论文 4.14，2026-10-20/21）** | 三种子独立收敛到一致权重比值 3.20/3.15/3.19（spread 1.02×，w_cls≈2.1）→ 默认 cls:reg=1:7 增益对官方指标欠优是稳健实证；但严格适用预登记零和条款后仅 1/3 种子真转正（seed 42/2024 触发 crit_R↑+crit_P↓ 零和；seed 2024 Δmean +0.0126 最高但增益全在召回侧），种子间跨度 0.0167=门宽 3.3 倍 → 判定 UNSTABLE，第 4 种子复核中；四层负结果（静态奖惩/推理侧重加权/单调重校准/自适应加权）指向判别结构 | backend/training/adaptive_loss.py，scripts/{train_adaptive_loss,summarize_uw_seeds,plot_uw_trajectories}.py，runs/ablation_adaptive/，docs/stage2_uw_verdict_template.md |
| 官方指标定义 | mean(mAP@0.1, mAP@0.3, mAP@0.5)，总分 0.2·ACC+0.8·mAP | 天池赛题页核验 2026-10-15 |
| SPFFNet 65.8% 口径 | mAP@0.5、**8 类子集** | arXiv:2502.01445v3 表 I 核验 |
| 18 类合并的官方口径代价 | 0.641 → 0.572（−6.9 点） | 换合并类召回 0.40→0.88 / 0.52→0.79 |
| 奖励注入：整体召回 | +0.030（0.523→0.553） | Layer-1 A/B, 5ep |
| 奖励注入：精确率 | −0.022 | Layer-1 A/B, 5ep |
| 奖励注入：mAP50 | +0.004（持平） | Layer-1 A/B, 5ep |
| 关键类召回变化 | broken_warp +0.032 / star_skip +0.057 / weave_defect +0.128 / broken_spandex −0.100 | Layer-1 A/B |
| 合并召回回升 | 纬缩+粗纬 0.40→0.88；断经+磨痕 0.52→0.79 | 类合并 |
| 类别数 | 20 → 18（合并 2 组同源类） | data.yaml nc=18 |
| 关键类 max-F1 召回（18 类基线） | hole 0.718 / broken_warp 0.521 / broken_spandex 0.484 | tianchi18_full |
| 关键类混淆分解（conf 0.25，GT 实例级） | hole 33/4/2、broken_warp 51/2/20、broken_spandex 45/1/8（正确/错分/漏检；错分合计仅 7/166）；conf=0.001 地板 39/0/0、65/2/6、51/0/3 | scripts/eval_confusion_decomp.py，runs/confusion/ |
| 关键类实例数 | hole 273 / broken_warp 681 / broken_spandex 492（中频非长尾） | 逐类统计 |
| 阈值标定 crit_recall | 0.573（全局 max-F1）/ 0.593（逐类）/ 0.910（conf 0.10） | eval_threshold_calibration |
| 精确率崩塌（conf 0.10） | broken_warp 0.56→0.08；broken_spandex 0.49→0.10 | eval_threshold_calibration |
| 均匀微调 crit_recall | **0.606**（所有加权干预的上界） | Direction B uniform 5ep |
| **强基线复测五臂 + 种子复测（起点 0.6349，2026-10-18/19 收官）** | R2 box −0.023/+0.104；B 类 −0.029/+0.184；NWD −0.015/+0.124；Layer-1 seed42 +0.0007/+0.053 但三种子 Δmean −0.0096±0.0093（噪声）；GRPO −0.306（崩溃） | 各臂 A/B vs 均匀对照 |
| 七负消融 crit_recall | GRPO 0.344 / 分类加权 0.591 / box 加权 0.598 / NWD 0.598 | 各干预 5ep |
| objectness 探针回归 | crit_recall 0.5743→0.5776、mAP50 0.537（检测器冻结） | Direction 3 线性探针 |
| objectness 门限 FPR @ R0.90 | 0.67（vs max-class-conf 0.99） | B 口径 957 val + 141 AITEX |
| objectness 门限 R @ FPR~0.04 | 0.48（vs max-class-conf ~0.01） | B 口径 |
| 训练稳定性 | fp16 AMP 溢出 → `amp=False` 修复，0 NaN | 工程发现 |
| **v4（1536+mosaic 从头训）官方口径 mean** | **0.6077**（@1280：0.636/0.614/0.573）/ 0.6094（@1536） | multIou 复算 2026-10-18 |
| **mosaic1280（1280+mosaic 从头训）官方口径 mean** | **0.6349**（@1280：0.666/0.647/0.592）/ 0.6106（@1536） | multIou 复算 2026-10-18 |
| 阶段 1 目标达成 | mAP50 0.541 → 0.592；官方口径 mean 0.572 → 0.635（+0.063） | 超 0.60 目标 |
| 分辨率因子分离结论 | 1280 全面占优 1536（同配方 +0.027 @1280 口径，每轮快 25%） | v4 vs mosaic1280 |
| 收敛效率 | mosaic 臂 e34 达基线终值 mAP50 0.541（基线 e56），提速约 1.7 倍 | results.csv |
| 归因限制 | 已解除：无 mosaic 对照臂（tianchi18_nomosaic1280）2026-10-19 跑完，rect/mosaic 解耦为 +0.036/+0.027 | 论文 4.3 |
