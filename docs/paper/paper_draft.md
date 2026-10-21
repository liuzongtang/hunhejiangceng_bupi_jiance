# 七维混合奖惩驱动的织物瑕疵智能检测：RT-DETR 基线、标签噪声与低置信度漏检的双层天花板

**Seven-Dimensional Hybrid Reward for Fabric Defect Detection: An RT-DETR Baseline and the Dual Ceiling of Label Noise and Low-Confidence Misses**

---

## 摘要

**研究背景** 织物瑕疵检测需在细粒度、长尾、弱对比的纹理图像中识别 18–20 个视觉高度易混淆的瑕疵类别；现有检测器以单一检测损失为唯一优化目标，忽视了"漏检（带病布匹出厂）"与"误检（良品误判报废）"之间不对称的工业代价。

**研究目的** 通过奖励信号将检测训练导向关键类召回，并刻画标签噪声与特征级判别力所施加的双层经验天花板。

**研究方法** 在天池"布匹疵点智能识别"基准（合并后 18 类）上训练 RT-DETR-L 基线；将可微的七维混合奖惩（D01 定位、D02 分类、D03 校准、D04 漏检、D05 误检、D06 断经/断氨纶敏感度、D07 星跳/死皱敏感度）注入检测损失；进一步采用"混淆矩阵→双向性判定→裁剪图复核"三步标签噪声诊断合并视觉同源类，并以混淆矩阵分解与阈值标定定位漏检根因。训练基础侧，以受控因子分离实验（rect letterbox × mosaic × 分辨率）把官方口径 mean mAP 从 0.572 提升至 0.6349。

**研究发现** 基线 mAP50 = 0.596（20 类）/ 0.541（18 类）；按官方指标（IoU 0.1/0.3/0.5 三阈值 mAP 均值，已经赛题页核验）为 0.641 / 0.572。双层诊断揭示天花板：第一层为标签噪声——合并视觉同源类后召回显著回升（纬缩+粗纬 0.40→0.88、断经+磨痕 0.52→0.79）；第二层为低置信度漏检——合并后关键类（破洞/断经/断氨纶）是"低于阈值漏检"而非"错分"（conf=0.25 下断氨纶 54 个 GT 实例仅 1 例错分、8 例漏检且其中 5 例有低于阈值的检出；三类错分合计仅 7/166），降低置信度阈值可将关键类召回 0.57→0.91，但精确率崩塌（断经 0.56→0.08）。图像级二分类回放进一步给出分层结论：最大置信度门限在**域内**可有效区分有/无瑕疵（957 张 val，AUC 0.9808，最佳正确率 93.73% @ τ=0.35，召回 95.2%/误报 8.6%），但在 **AITEX 跨域**负样本上误报率飙至 71.6%——真检与误检的置信度分离是域内属性，不跨域迁移（且二分类判断正确率不等于缺陷定位精度）。在弱基线与强基线上对七种训练信号干预（可微奖励注入、GRPO、分类严重度加权、box 损失加权、NWD 损失几何）做系统 A/B 消融：改权重/改度量类干预在强基线上全部复现"召回—精度零和"（mean −0.015~−0.029 换取关键类召回 +0.10~+0.18），是系统性结构特征而非个别干预的副作用；唯一的可微奖励接口（Layer-1）在 seed 42 上表观同向双臂改善（mean +0.0007、crit_recall +0.053），但三种子复测（seed 42/123/2024，Δmean −0.0096 ± 0.0093、Δcrit +0.0499 ± 0.0710）确认该表观为种子噪声——召回—精度零和在可微奖励接口上同样复现，四种机制互异的干预接口（权重/度量/可微奖励/策略优化）全部落入同一零和；GRPO 则被实证不可救药（降低 advantage 地板反而放大噪声梯度伤害，mean −0.306）。同一多维思想向推理侧报警评分的迁移亦经离线回放检验：七维可解释评分（严重度/置信度/不确定性/相似组冲突/尺寸/空间聚集）在域内不敌 max-conf 基线（消融最佳变体正确率 0.9269 vs 0.9383，主变体 0.9154；等召回误报 1.5–2 倍）、跨域无优势，但提供了逐检测归因接口与三档分级告警框架（停机+复核合并关键类召回 0.967）。标定审计给出第三层定量证据：运行区域置信度严重失准（ECE 0.106/0.112，过信 2–3 倍）且为可靠性曲线形状问题——单参数温度缩放反而恶化（0.177/0.172），isotonic 交叉拟合可修复至 0.037/0.046，但**任何单调重校准都不改变真检/误检在置信度维度上的可分性**（AUC 0.763/0.823 严格不变），"低置信度漏检不可按分数分离"由此从断言升级为结构性命题。进一步用不确定性加权（UW）让模型自动学习 cls:reg 损失权重：四种子独立收敛到几乎相同的权重比值（3.20/3.15/3.19/3.21，spread 1.02×），实证默认 `class×1, bbox×5, giou×2` 增益对官方指标欠优；但严格适用预登记零和条款后仅 1/4 种子真转正，第 4 种子把三种子的暂定不稳定判定解决为负——种子平均 mean 0.6176 低于转正门 0.6208，种子间跨度（0.0214）是判定门宽的 4.3 倍，效应量小于种子离散度。最后，贝叶斯路线的实用替代（深度集成；MC Dropout 因 decoder 无 dropout 层在本架构不可用）经两套同起点不同种子集成检验：集成分歧闸门在固定召回下不优于分数阈值，且分歧度与真检正相关（TP 簇 std 为 FP 簇的 40–65 倍）——同起点微调的成员对 90% 以上的 GT 一致触发，集成歧义不含正交信息。标定层干预首次改善跨域门限：代价敏感 objectness 头（冻结解码器上的线性探针）在匹配召回下将跨域误报率降低 2–6 倍（4% 误报率下图像召回 0.48 vs 0.01）；同协议回放进一步精确化其价值——域内图像级判定仍全面劣于 max-conf（AUC 0.888 vs 0.969），该头的优势是跨域前景/背景分的鲁棒性而非更好的域内报警门限。

**研究意义** 证明瓶颈不在训练信号，而在细缺陷的特征级判别力：低置信度是困难定位的忠实反映而非校准失误（与 DETR 一对一匹配的"专家策略"一致）。手动奖惩、推理侧重加权、单调重校准、集成歧义闸门、自适应损失加权五种机制互异的干预全部无法稳定突破"召回—精度零和"，瓶颈在判别结构本身而非分数或权重形式；UW 权重轨迹证明默认损失增益欠优却修正无益，恰说明该结构是内生的——重新平衡两臂只是把容量在召回与精度之间搬家。七维奖惩作为"性能增强手段"被数据否定，但作为"故障诊断探针"极其成功——它们精确定位了故障层（query 触发 + 定位自评）与可行接口（可微奖励、专用 objectness 头）。"混淆矩阵→阈值标定→系统消融→因子分离"的诊断流程对成本敏感型细粒度识别任务具有可迁移的方法论价值。

**关键词** 织物瑕疵检测；RT-DETR；混合奖惩；标签噪声；召回—精度权衡；因子分离；GRPO

---

### English Abstract

**Background** — Fine-grained fabric defect detection requires recognizing 18–20 visually confusable defect classes from long-tailed, weakly-contrastive texture images. Existing detectors optimize a single detection loss and ignore the asymmetric industrial cost between a *miss* (defective fabric shipped) and a *false alarm* (good fabric discarded).

**Purpose** — We aim to steer detection toward critical-defect recall via a reward signal, and to characterize the dual empirical ceiling imposed by label noise and feature-level discriminability.

**Method** — We train an RT-DETR-L baseline on the Tianchi fabric-defect benchmark (18 merged classes), inject a differentiable seven-dimensional hybrid reward (D01 localization, D02 classification, D03 calibration, D04 miss, D05 false-positive, D06 broken-yarn sensitivity, D07 skip/weave sensitivity) into the detection loss, and apply a three-step label-noise diagnosis (confusion matrix → bidirectionality test → cropped-image review) that merges visually identical class pairs. On the training side, controlled factor-separation experiments (rect letterbox × mosaic × input resolution) raise the official contest metric (mean of mAP at IoU 0.1/0.3/0.5) from 0.572 to 0.6349.

**Findings** — The baseline reaches mAP50 = 0.596 (20-class) / 0.541 (18-class); under the official metric this is 0.641 / 0.572. A two-level diagnosis exposes the ceiling. *Level 1 — label noise*: merging confusable class pairs recovers recall sharply (weft_shrink+coarse_weft 0.40→0.88, broken_warp+surface_mark 0.52→0.79). *Level 2 — low-confidence misses*: after the merge, the critical classes (hole / broken_warp / broken_spandex) are *missed below threshold*, not misclassified (broken_spandex: 1 misclassified vs. 8 missed out of 54 GT instances, 5 of the 8 with a sub-threshold detection; only 7 of 166 critical-class instances misclassified at conf=0.25); lowering the confidence threshold lifts critical recall 0.57→0.91 but collapses precision (broken_warp 0.56→0.08). An image-level binary replay adds a layered conclusion: the max-confidence gate separates defective from clean images **in-domain** (957 val images, AUC 0.9808, best accuracy 93.73% at τ=0.35, recall 95.2% / FPR 8.6%) but fails to transfer **cross-domain** (AITEX FPR 71.6%) — confidence separation is an in-domain property, and binary judgment accuracy is not localization accuracy. Systematic A/B ablations of seven training-signal interventions on both weak and strong baselines show that all weight/metric-modifying interventions reproduce a recall–precision zero-sum on the strong baseline (mean −0.015~−0.029 traded for critical recall +0.10~+0.18) — a systematic structural signature rather than an artifact of any single intervention. The only differentiable-reward interface (Layer-1) appeared to improve both metrics in the same direction on seed 42 (mean +0.0007, crit_recall +0.053), but a three-seed replication (seeds 42/123/2024: Δmean −0.0096 ± 0.0093, Δcrit +0.0499 ± 0.0710) confirms this was seed noise — the zero-sum replicates on the differentiable-reward interface as well, and four mechanistically distinct interfaces (weighting / metric / differentiable reward / policy optimization) all fall into the same zero-sum; GRPO is empirically unsalvageable (lowering the advantage floor amplifies noise-gradient damage, mean −0.306). The same multi-dimensional idea is also tested on the inference side as an alarm scorer: a seven-dimension interpretable score (severity / confidence / uncertainty / group conflict / size / clustering) does not beat the max-confidence baseline in-domain (best-ablated accuracy 0.9269 vs. 0.9383, main variant 0.9154; 1.5–2x FPR at matched recall) and shows no cross-domain advantage, but provides per-detection attribution and a three-tier alert framework (stop+review critical recall 0.967). A calibration audit supplies a third layer of quantitative evidence: confidences in the operating region are severely mis-calibrated (ECE 0.106/0.112, over-confident by 2–3×) and the fault lies in the reliability-curve shape — single-parameter temperature scaling makes it worse (0.177/0.172) while isotonic cross-fitting repairs ECE to 0.037/0.046, yet no monotonic recalibration changes TP/FP separability (AUC 0.763/0.823 bit-for-bit unchanged), upgrading "low-confidence misses are not separable by score" from an assertion to a structural statement. We further let the model learn the cls:reg loss weighting itself via uncertainty weighting (UW): four seeds independently converge to nearly identical weight ratios (3.20/3.15/3.19/3.21, spread 1.02×), empirically showing that the default `class×1, bbox×5, giou×2` gains are sub-optimal for the official metric; but under the strictly pre-registered zero-sum clause only 1/4 of the seeds count as truly positive, and the 4th seed resolves the three-seed UNSTABLE verdict to negative — the seed-average mean (0.6176) falls below the gate (0.6208) and the between-seed spread (0.0214) is 4.3× the gate width, i.e. the effect size is smaller than seed noise. Finally, the practical substitute for the Bayesian route (deep ensembles; MC Dropout is unavailable because the RT-DETR decoder has no dropout) is tested with two same-start ensembles: the disagreement gate does not beat score thresholding at fixed recall, and disagreement is positively correlated with true detections (TP-cluster std is 40–65× the FP-cluster std) — same-init fine-tunes fire consistently on 90%+ of GTs, so ensemble disagreement carries no orthogonal information. A calibration-layer intervention, by contrast, is the first to improve the cross-domain gate: a cost-sensitive objectness head (a linear probe on the frozen decoder) cuts cross-domain false-positive rate 2–6× at matched recall (image recall 0.48 vs. 0.01 at a 4% FPR); same-protocol replay sharpens its value — its in-domain image-level judgment still loses to max-conf across the board (AUC 0.888 vs. 0.969), so the head's advantage is precisely the robustness of the cross-domain foreground/background score, not a better in-domain alarm gate.

**Implications** — The bottleneck is not the training signal but the feature-level discriminability of thin defects: their low confidence faithfully reflects hard localization, not mis-calibration (consistent with the DETR "specialist strategy" of one-to-one matching). Manual reward, inference-side re-weighting, monotonic recalibration, ensemble-disagreement gating, and adaptive loss weighting — five mechanistically distinct interventions — all fail to break the recall–precision zero-sum stably: the bottleneck lies in the discriminative structure itself, not in the score or weighting form. The UW weight trajectories show the default loss gains are measurably sub-optimal yet correcting them does not help stably, which indicates the structure is endogenous: re-balancing the two arms merely moves capacity between recall and precision. The seven-dimensional reward is refuted as a *performance booster* but vindicated as a *diagnostic probe*: it localizes the failure layer (query triggering + localization self-assessment) and the viable interfaces (differentiable reward, dedicated objectness head). The diagnosis pipeline — confusion matrix → threshold calibration → systematic ablation → factor separation — is a transferable methodology for cost-sensitive, fine-grained recognition.

**Keywords**: fabric defect detection; RT-DETR; hybrid reward; label noise; recall–precision trade-off; factor separation; GRPO

---

## 1 引言

### 1.1 工业价值与问题定义

织物瑕疵检测是纺织质量控制的关键环节。与通用目标检测不同，其工业价值函数高度不对称：一匹漏检（false negative）的带病布匹流入下游造成的返工与索赔代价，远高于一匹误检（false positive）良品被停机复查的代价。因此，**关键类（断经、断氨纶、破洞等结构性缺陷）的召回率，而非全局平均精度，才是系统真正的优化目标**。然而现有检测器以单一检测损失训练，这一代价不对称从未进入优化目标。

与此同时，该场景对检测器提出了三重挑战：细粒度（18–20 个视觉高度易混淆的瑕疵类别）、长尾（类别样本量从 117 到 4390 不等）、弱对比（纱线级细缺陷在 2446×1000 宽幅图像中仅占少量像素）。

### 1.2 范式演进与两个未解问题

检测范式经历了 Gabor/LBP/GLCM 手工特征 → Faster R-CNN/SSD/YOLO → DETR/RT-DETR（全局注意力、一对一匹配、免 NMS）的演进。RT-DETR 以实时性能保持 DETR 的端到端优势，但其一对一匹配机制引入了一个本工作反复触及的结构性事实：每个目标只有一个"主预测" query，其置信度被训练为自身预测 IoU（VarifocalLoss 的 `gt_score`）——细缺陷的 IoU 天然低，且没有第二个 query 兜底。

在此背景之下，本工作聚焦两个未被充分研究的问题：

1. **训练目标单一**：能否把"漏检 ≫ 误检"的工业代价显式注入端到端训练（本文的七维混合奖惩框架）？
2. **经验天花板在哪里**：当训练信号干预全部失败时，瓶颈究竟在损失函数、标签质量还是特征表示？

### 1.3 主要贡献

1. **可微七维混合奖惩框架**——把定位/分类/校准/漏检/误检/两类关键类敏感度共七个维度构造成对 logits 与边界框可微的奖励信号，首次将漏检/误检的不对称工业代价显式注入 RT-DETR 端到端训练。**实证结论（诚实呈现）**：该框架能显著搬移召回指标（整体召回 +0.030），但呈召回—精度零和；其在弱基线与强基线上的系统消融，反而是全文最有价值的诊断产出（见 4.7/4.8）。
2. **双层天花板诊断方法论**——"混淆矩阵分解→阈值标定→系统消融"三步定位 + 视觉同源类合并（20→18 类），把提升努力从"换更大模型/更强损失"重新导向"修数据"这一真正瓶颈（合并类召回 0.40→0.88）。
3. **训练基础的受控因子分离**——以三点定位实验把相对基线的 +0.063 增益干净解耦为 rect letterbox（+0.036）与 mosaic 增强（+0.027）两个近似可加的因子，并证伪高分辨率路线（1536 无净增益且慢 25%）。
4. **强基线诚实重测与创新点再定位**——七维奖惩在官方口径 0.6349 的强基线上五臂重测，预先承诺判定规则；最终把创新点从"性能增强手段"重新定位为"故障诊断探针"。
5. **训练稳定性诊断**——定位 fp16-AMP BN 统计量溢出为 NaN 发散根因（39 个 BN 层 running_var 清零），以 bf16 autocast 修复并经 BN 哨兵验证，使高分辨率长跑训练首次稳定。
6. **干预空间的系统性排除与机理定位**——把"多维加权"思想沿四个机制层面推进并全部测到边界：训练侧静态奖惩（4.7/4.8，七负）、推理侧结构化报警评分（4.12，域内不敌 max-conf 但给出逐检测归因接口与三档分级告警）、置信度校准审计（4.13，运行区域 ECE 0.106/0.112 严重失准，isotonic 可修复但任何单调重校准不改变 TP/FP 可分性）、贝叶斯路线的实用替代（4.15，深度集成分歧闸门在固定召回下不优于分数阈值）、自适应损失加权（4.14，四种子独立收敛到一致的 cls:reg 权重比值 ~3.2，实证默认增益欠优，但干预效应小于种子离散度）。机制互异的负结果共同把瓶颈定位到特征级判别结构，并划定后续突破的必要条件：须引入与既有分数正交的信息源。

---

## 2 相关工作

### 2.1 通用目标检测

DETR（Carion et al., 2020）以 Transformer 全局注意力取代锚框与 NMS，实现端到端集合预测；RT-DETR（Zhao et al., 2024）以高效混合编码器保持实时性。本文基线采用 ultralytics 实现的 RT-DETR-L。

### 2.2 织物瑕疵检测

公开基准包括天池"布匹疵点智能识别"（本工作采用，20 类）、AITEX 与 ZJU-Leaper。现有天池公开方案 mAP50 普遍 0.59–0.66：Fab-ME（YOLOv8s，20 类，59.4%，arXiv:2412.03200）、改进 RT-DETR（RT-DETR-R18，牛仔布，20 类，60.0%，浙江大学学报 2025）、SPFFNet（YOLOv11，**8 类子集**，65.8%，arXiv:2502.01445，口径已核验）。均只报 mAP@0.5 且无标签噪声诊断，与本工作不可直接对比（详见 4.11）。

### 2.3 奖励/RFLF 训练信号

RLHF（Ouyang et al., 2022）与 PPO（Schulman et al., 2017）确立了"以奖励信号引导策略"的范式；GRPO（Shao et al., 2024）以组内相对优势免去 critic。本工作把这一思想迁移到检测器训练：Layer-1 可微奖励（直接对 logits/box 可微的奖励塑形）与 Layer-3 GRPO（查询噪声 REINFORCE + 组内相对优势 + KL 锚定）。实证表明迁移在本任务的奖励面上失败，失败机制本身（奖励面平坦性）是重要发现（见 4.7/4.8）。

### 2.4 检测器标定与细目标度量

DETR 家族的对象级标定问题已被量化（Park et al., 2024，OCE 指标）；图像分类中的温度缩放与 ECE 度量（Guo et al., 2017）是置信度后处理的标准工具。多任务损失的自动加权（Kendall et al., 2018 的同方差不确定性加权；Chen et al., 2018 的 GradNorm）提供了替代手工调权的路线。细目标检测中 NWD（Wang et al., AAAI 2022）以高斯化 Wasserstein 距离替代 IoU。本工作分别以 objectness 线性探针、NWD 损失几何、校准审计与不确定性加权介入这四个层面，结果与机制讨论见 4.9、4.7、4.13 与 4.14。

### 2.5 定位

本工作区别于"换更强骨干"路线，聚焦"以工业代价为目标的端到端检测训练"与"经验天花板诊断"两个问题；方法学输出（因子分离、阈值标定口径、A/B 协议）与负面结果本身均为主要贡献。

---

## 3 方法

### 3.1 RT-DETR 检测基线

采用 ultralytics RT-DETR-L，输入 1280，batch 4，MuSGD（optimizer="auto"），60 epochs，fp32（`amp=False`，稳定性说明见 4.2）。数据集经视觉同源类合并为 18 类（见 3.3）。

### 3.2 七维混合奖惩（D01–D07）

将七个维度构造成对 logits 与边界框**可微**的奖励信号，注入方式为 `loss_reward = −β·mean(R)` 并入检测损失（β 为奖励强度，A/B 取 0.5）。

**表 1　七维混合奖惩 D01–D07 维度定义**

| 维度 | 名称 | 可微构造 | 工业语义 |
|------|------|----------|----------|
| D01 | 定位 | soft IoU | 框回归质量 |
| D02 | 分类 | smooth accuracy | 类别判别正确性 |
| D03 | 校准 | calibration gap | 置信度可信度 |
| D04 | 漏检 | sigmoid soft miss | 带病出厂代价最高 |
| D05 | 误检 | sigmoid soft fp | 良品误判报废 |
| D06 | 断经·断氨纶敏感度 | softmax 关键类质量 | 高危关键类 |
| D07 | 星跳·死皱敏感度 | softmax 关键类质量 | 高影响关键类 |

D04/D05 直接编码"漏检代价 ≫ 误检代价"的工业不对称；D06/D07 显式牵引关键类召回。Layer-3 进一步以 GRPO（查询噪声 REINFORCE + 组内相对优势 + KL 锚定，免 critic）探索"奖励引导的训练"。

### 3.3 标签噪声天花板诊断

三步流程：**混淆矩阵**（定位错误结构）→ **双向性判定**（区分标签污染与标注歧义）→ **裁剪图复核**（人工目检确认）。合并视觉同源类：断经↔磨痕、纬缩↔粗纬（20→18 类）。

### 3.4 训练基础的因子分离协议

为把"训练基础优化"（rect letterbox、mosaic、输入分辨率）的贡献从奖励干预中干净分离，采用三点定位协议：所有臂共享同一 COCO 初始化、从头训 60 epochs、同一评测脚本（官方口径），逐因子单变量切换。协议保证 4.3 节每个数字只归因于一个因子。

---

## 4 实验

### 4.1 数据集与实现

天池布匹疵点智能识别：训练集 8619 张、官方验证集 957 张（374 无瑕疵 + 583 有瑕疵，953 GT 框），合并后 18 类。实现：RT-DETR-L、imgsz 1280、batch 4、auto 优化器、60 epochs、fp32。

**表 2　18 类（合并后）名称/中文名/严重度**（`broken_warp` 吸收 `surface_mark`，`weft_shrink` 吸收 `coarse_weft`）

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

### 4.2 基线性能与官方口径复算

20 类：mAP50 = 0.596（官方复算口径 0.5963；训练期 val 0.594）、mAP50-95 = 0.334（50 epochs）。18 类（合并后）：best mAP50 = **0.541** @ e56，末轮 0.537，mAP50-95 = 0.295，P 0.536，R 0.519（60 epochs，0 NaN）。

**官方指标口径（2026-10-15 赛题页核验）**：天池明确"分别在 IoU 阈值 0.1、0.3、0.5 下计算 mAP，最终 mAP 取三个值的平均值"，初赛总分 = 0.2·ACC + 0.8·mAP。此前只报 mAP@0.5 系统性低估官方口径成绩。复算脚本 `scripts/eval_multIou.py`（官方 val 管线 + 仅替换 IoU 匹配向量，mAP@0.5 复算值与训练 results.csv 最优 epoch 值一致，各臂偏差 ≤0.004，源于训练期 val 与复算管线的实现细节差异）：

**表 3　官方口径三阈值复算（val 957 张，conf=0.001，max_det=300，imgsz 1280）**

| 模型 | mAP@0.1 | mAP@0.3 | mAP@0.5 | 官方口径（三阈值均值） |
|------|------|------|------|------|
| 20 类基线（50ep） | 0.673 | 0.653 | 0.596 | **0.641** |
| 18 类合并（60ep） | 0.596 | 0.579 | 0.540 | **0.572** |

**训练稳定性**：fp16-AMP 长跑 ≥9 epoch 会清零 neck/encoder 39 个 BN 层 running_var（fp16 指数范围上限 65504，方差统计 E[x²] 溢出），eval 模式经 sqrt(eps) 放大 100–400× 使 mAP=0；1–2 epoch 冒烟测不出。修复为 bf16 autocast（指数范围同 fp32），需同时绕过 ultralytics `check_amp` 的 atol=0.5 误判（否则 AMP 被静默禁用跑成 fp32，显存贴顶 24.2GB、9.9s/步）。基线全程 fp32，0 NaN。

### 4.3 强基线构建：rect letterbox 与 mosaic 的因子解耦

**问题**：在 1280 方形 letterbox 基线（官方口径 0.5719）之上，rect（按长宽比聚类矩形批）、mosaic 增强与输入分辨率各自贡献多少？

**高分辨率路线证伪（负面结果）**：基线 best.pt 不训练直接在 1536 rect 下推理，官方口径 mean = 0.557（vs 1280 方形 0.572；探针存档 `runs/multi_iou/tianchi18_best_at1536_probe.json`，2026-10-20 复跑复现 0.5566）——推理端几何迁移成本仅 −0.015，无净增益证据。"已收敛权重微调上新 letterbox 几何"稳定劣化（两次独立 lr 实验同一签名：e1–e6 mAP50 三连降 0.50→0.40，损失平台高于基线 16%）——收敛权重盆地尖锐 + MuSGD 动量 0.937 + 新几何失配。唯一健康轨迹是从 COCO 从头训。

**三点定位（受控因子分离，官方口径 mean）**：

**表 4　因子解耦三点定位（18 类，从头训 60ep，官方口径 mean）**

| 臂 | imgsz | letterbox | mosaic | 官方口径 mean | mAP50 | 每轮耗时 |
|---|---|---|---|---|---|---|
| 基线 tianchi18_full | 1280 | 方形 | ✗ | 0.5719 | 0.541 | ~19 min |
| 无 mosaic 对照臂 | 1280 | rect | ✗ | 0.6080 | 0.567 | ~15 min |
| **mosaic1280（最强）** | **1280** | **rect** | **✓** | **0.6349** | **0.592** | **~16 min** |
| v4 | 1536 | rect | ✓ | 0.6077（@1280）/ 0.6094（@1536） | 0.573 | ~20 min |

- **因子解耦终局**：基线 0.5719 → **rect 单独 0.6080（+0.036，占总增益 57%）** → **rect+mosaic 0.6349（mosaic 增量 +0.027，占 43%）**。两个便宜因子量级相当、rect 略大，近似可加。修正了训练中段"mosaic 主导"的曲线猜测（中段曲线差异主要来自收敛速度而非终值：mosaic 臂 e34 即达基线终值 mAP50 0.541，基线自身 e56 才跨过该值，**收敛提速约 1.7 倍**）。
- **高分辨率无净增益**：同配方 @1280 口径 0.6349 vs 1536 臂（同 @1280 口径复算）0.6077（+0.027），且每轮快 25%；两种评测协议下 1536 训练均无优势（1536 臂 @1536 口径 0.6094，1280 臂 @1536 口径 0.6106）。与原始图像 2.4:1 的宽幅、疵点细窄的几何特性一致：rect letterbox 已消除算力浪费（方形 1280 内容仅 1280×523，59% 像素为填充），再加分辨率只放大背景。
- **letterbox 几何的耗时特性**（三点实测，results.csv 累计时间差分的每轮中位数）：方形无 mosaic 18.5 min/轮、**rect 无 mosaic 15.2 min/轮（最快）**、rect+mosaic 15.4 min/轮、1536 rect+mosaic 20.0 min/轮——rect letterbox 相比方形省 ~18% 墙钟时间（与 59% 的算力浪费消除方向一致），mosaic 基本中性，升到 1536 则贵 30%。

**阶段 1 结论**：官方口径 mean 0.572 → **0.6349**（+0.063），mAP50 0.541 → 0.592，越过 0.60 目标线。强基线（协议触发条件 mean ≥ 0.60）成立，4.8 节的七维奖惩重测以此为起点权重。

### 4.4 奖励注入 A/B（Layer-1，弱基线）

5 epoch A/B（同代码同 val；⚠️ 该实验在 **20 类**模型上进行，与 4.2 的 18 类从头训基线为两条独立证据）：漏检纳入奖励后整体召回 **+0.030（0.523→0.553）**，精确率 **−0.022**，mAP50 持平（+0.004，噪声带内）。关键类召回分化：broken_warp +0.032、star_skip +0.057、weave_defect +0.128，但 broken_spandex **−0.100**（最常见关键类回退）。

**结论**：Layer-1 奖励注入能移动召回，但以精确率为代价且主导关键类回退——精确刻画"召回—精度"零和边界，指明瓶颈在标签层而非训练信号。

### 4.5 混淆类合并：标签噪声天花板验证

合并后召回率显著回升：纬缩+粗纬 0.40→0.88、断经+磨痕 0.52→0.79。断经↔磨痕为标签污染（同类缺陷两种标注），纬缩↔粗纬为标注歧义（视觉同源）。

**表 5　混淆类合并前后关键类召回**

| 合并组 | 类 A 召回 | 类 B 召回 | 合并后召回 |
|--------|-----------|-----------|------------|
| 纬缩 ∪ 粗纬 | 0.40（纬缩） | 0.79（粗纬） | **0.88** |
| 断经 ∪ 磨痕 | 0.52（断经） | 0.83（磨痕） | **0.79** |

合并的显式代价：官方口径 0.641→0.572（−6.9 点）。"mAP 换召回"的权衡数据是标签噪声诊断方法学的核心实证。

### 4.6 低置信度漏检诊断（混淆分解 + 阈值标定）

合并后关键类召回仍停滞在 0.48–0.72（hole 0.718 / broken_warp 0.521 / broken_spandex 0.484 @max-F1，r_curve 值；实例级精确计数见表 6）。

**（一）混淆分解——漏检而非错分。** val 957 张、conf=0.25、IoU 0.5，按 GT 实例三分解（正确 / 错分 / 漏检，互斥且完备）。协议与官方 val 一致（方形 letterbox，RT-DETR val 强制 rect=False），"正确"判据与 ultralytics r_curve 的类感知匹配同口径——实例级召回与阈值扫描 CSV（`runs/threshold_calibration/threshold_sweep.csv`）逐类核对一致（hole 0.8462 = 0.8462、broken_warp 0.699 ≈ 0.708、broken_spandex 0.8333 = 0.8333，差异为曲线插值）。分解脚本 `scripts/eval_confusion_decomp.py`，存档 `runs/confusion/decomp_tianchi18_tau{0.25,0.473,0.001}.json`。

**表 6　关键类 GT 实例级混淆分解（conf=0.25，IoU 0.5）**

| 类 | GT 实例 | 正确 | 错分 | 漏检 | 漏检中有 <τ 检出 | 主要错分去向（←计数） |
|---|---|---|---|---|---|---|
| hole | 39 | 33 | 4 | 2 | 2 | knot←2、star_skip←1、hair_particle←1 |
| broken_warp | 73 | 51 | 2 | 20 | 14 | dense_section←1、loose_warp←1 |
| broken_spandex | 54 | 45 | 1 | 8 | 5 | three_silk←1 |
| **合计** | **166** | **129** | **7** | **30** | **21** | — |

**表 6b　阈值敏感性（正确 / 错分 / 漏检）**

| 置信度阈值 | hole | broken_warp | broken_spandex |
|---|---|---|---|
| 0.473（max-F1 工作点） | 28 / 5 / 6 | 37 / 3 / 33 | 25 / 1 / 28 |
| 0.25 | 33 / 4 / 2 | 51 / 2 / 20 | 45 / 1 / 8 |
| 0.001（不设阈值） | 39 / 0 / 0 | 65 / 2 / 6 | 51 / 0 / 3 |

关键结论：**错分是极少数，漏检是主体，且漏检多为"低于阈值的检出"**。conf=0.25 下三类错分合计 7 例（占 166 个 GT 实例的 4.2%），漏检 30 例中 21 例有低于阈值的检出（同类检出 19 例）。断氨纶最典型：54 个 GT 仅 1 例错分（被 three_silk 拿走）、8 例漏检中 5 例低于阈值。错分去向与 4.5 的族聚类一致——hole 的点状错分归 knot/hair_particle（点状族内互混），broken_warp 归 dense_section/loose_warp（线性结构族内互混）。把阈值降到 conf=0.001（等价于不设阈值）后，真漏检仅剩 broken_warp 6、broken_spandex 3、hole 0——**几乎所有漏检都是"检到了、但置信度低于任何可用全局阈值"**：漏检是校准/置信度问题，不是检测/分类问题。样本量排除稀有度假设（hole=273 / broken_warp=681 / broken_spandex=492 为中频；真正的长尾是 flower_board=117、hundred_feet=145）。与阈值标定（表 7）的对账：conf=0.25 时三类召回 0.846/0.708/0.833 已高于 max-F1 工作点的 0.718/0.521/0.484，但精确率同步崩塌（broken_warp 0.27、broken_spandex 0.25 @conf 0.25，vs 0.56/0.49 @0.473、0.08/0.10 @0.10）——降阈值把低于阈值的检出放回来，同时放进成倍误报；表 6b 的 max-F1 行给出同一机制的逐实例版本（该工作点上三类漏检 6/33/28 中，除 0/6/3 个真背景漏检外全部有低于阈值的检出）。注：表 6b 中 conf=0.473 的实例级召回（0.718/0.507/0.463）与表 7 曲线值（0.718/0.521/0.484）相差 ≤1 实例，源于 r_curve 在 1000 点 conf 网格上的线性插值。

**（二）阈值标定——召回可抬但精确率崩塌。**

**表 7　成本敏感关键类召回（三种阈值口径，均值 hole/broken_warp/broken_spandex）**

| 口径 | 阈值 | crit_recall | broken_warp P | broken_spandex P |
|---|---|---|---|---|
| 全局 max-F1 | conf=0.473 | 0.573 | 0.56 | 0.49 |
| 逐类 max-F1 | 各类自优 | 0.593 | — | — |
| 召回优先 | conf=0.10 | **0.910** | **0.08** | **0.10** |

**不存在"高召回 + 可用精确率"的阈值**：降阈值可把关键类召回 0.57→0.91，但 broken_warp 精确率 0.56→0.08、broken_spandex 0.49→0.10，真低置信检测与误报无法按置信度分离。根因：RT-DETR 的 VarifocalLoss `gt_score` == 预测 IoU，故"低置信度" == "细缺陷定位自评弱"。

### 4.7 训练信号干预的系统消融（弱基线，七负）

对**训练信号**做七种干预，全部以严格 A/B 对照验证（均匀微调对照 crit_recall = 0.606）：

**表 8　训练信号干预系统消融（弱基线 mAP50 0.541，关键类召回 @max-F1）**

| # | 干预 | 机制层 | 关键指标 | 对照均匀微调 | 判定 |
|---|---|---|---|---|---|
| 1 | Layer-1 可微奖励注入（β=0.5） | 奖励 | 整体召回 +0.030，broken_spandex −0.100 | — | 负 |
| 2 | Layer-3 GRPO（β=0.5，3ep） | 奖励 | crit_recall 0.574→0.344 | — | 负 |
| 3 | Direction B 分类严重度加权（4 臂） | 分类损失 | full 0.541 / positive-only 0.591 | < 0.606 | 负 |
| 4 | R2 box 加权（critical 2.0×） | 定位损失权重 | crit_recall 0.598，mAP50 0.522 | < 0.606 | 负 |
| 5 | R3 NWD 损失几何 | 定位损失度量 | crit_recall 0.598，mAP50 0.518 | < 0.606 | 负 |

**阈值标定复查（R2/R3 的最终判据）**：R2/R3 训练后的关键类在降阈值时精确率原封不动崩塌——broken_warp P@conf0.10 = 0.081（R2）/0.082（R3）vs 基线 0.08，broken_spandex P@0.10 = 0.080（R2）/0.053（R3）vs 基线 0.10。即改权重、改度量都未能把真检测的置信度相对误报抬起来。

**统一解释（接口错位）**：D01 定位/D03 校准/D04 漏检三个维度**命中**了故障，但接口形式（聚合 box 指标/分类损失梯度）与真实故障层（query 触发 + 定位自评）**不对齐**，梯度无从着力。更深一层：低置信度是**忠实的低判别力**，与 DETR 一对一匹配的"专家策略"一致。

### 4.8 强基线复测：七维奖惩的再检验

**动机**：4.7 的七负全部在弱基线上完成；强基线（mosaic1280，官方口径 0.6349）上干预可能改变符号——必须重测，两种结果都有价值。

**协议**（`docs/plan_stage1.3_rerun.md`，判定规则预先承诺）：与历史七负完全同口径（1280 方形、amp=False、lr0=1e-4、seed=42），唯一变量 = 起点权重（0.541 → 0.635）。每臂 A/B：干预 vs 同配置均匀微调。**转正 = 干预 mean > 对照 +0.005 且 crit_recall 不降**。

**表 9　强基线复测五臂（起点 mosaic1280 best.pt，官方口径 mean）**

| 臂 | 干预 | mean Δ | crit_recall Δ | 判定 |
|---|---|---|---|---|
| 1 | R2 box 严重度加权 | −0.023 | +0.104 | 负 |
| 2 | Direction B 分类加权 | −0.029 | +0.184 | 负 |
| 3 | R3 NWD 度量 | −0.015 | +0.124 | 负 |
| 4 | Layer-1 可微奖励 | +0.0007 | +0.053 | 负（未过线；表观破零和经三种子复测为噪声） |
| 5 | GRPO（eps 1e-2→1e-4） | −0.306 | −0.194 | 负（模型崩溃） |

逐臂要点：

- **臂 1（R2，04:50 完成）**：均匀对照 mean 0.6158 / crit 0.6260；severity 加权 mean 0.5927（−0.023）/ crit 0.7302（+0.104）/ broken_spandex 0.4074→0.6852（+0.278）。弱基线时期同样干预几乎无效（0.598 vs 0.606）——**强基线上干预"着力"强度放大约一个数量级**，但搬移的是容量而非新增可分性。
- **臂 2（Direction B）**：mean −0.029，crit +0.184，broken_spandex 0.8704（+0.463）。零和剂量比臂 1 更大。
- **臂 3（NWD）**：mean −0.015（三臂中代价最小），crit +0.124，broken_spandex 0.8148（+0.407）。
- **臂 4（Layer-1，10ep，epoch 匹配对照）**：对照（10ep 均匀）mean 0.6157 / crit 0.7010；干预 mean 0.6164（+0.0007）/ crit 0.7541（+0.053）/ broken_spandex +0.093。seed 42 单次实验中为五臂唯一 mean 与 crit 同向者。注意对照本身：10ep 均匀微调的 crit_recall（0.7010）显著高于 5ep 对照（0.6260）而 mean 持平——**单纯延长均匀微调即可免费获得关键类召回 +0.075**，Layer-1 的 +0.053 是更强对照之上的增量。**但多种子复测（见下）表明该同向性是种子噪声，零和在 Layer-1 上同样复现。**
- **臂 5（GRPO，std_reward 地板修复版）**：对照 mean 0.6158；GRPO mean **0.3101（−0.306）**，模型被摧毁。地板修复的反直觉结论：eps 1e-2→1e-4 后 65% 的组 std_reward 越过地板（中位数 1.7e-4），REINFORCE 从"死寂"变"激活"——但激活的是**噪声梯度**（奖励面在策略采样噪声下本就平坦）。5390 步噪声梯度累积后模型腰斩。**在这个奖励面上，降低地板不是修复而是放大伤害**；GRPO 路线应整体放弃（除非重新设计有区分度的奖励）。历史弱基线 GRPO 同样崩溃（crit 0.574→0.344），伤害有历史佐证。
- **附带发现**：均匀对照本身从 0.6349 降至 0.6158（−0.019）——1280 方形 letterbox 对 rect 训练的强模型有轻微微调毒性（两臂同起点同配置，A/B 内部效度不受影响）。

**表 9b　Layer-1 三种子复测（seed 42/123/2024，同协议 10ep A/B，唯一变量 = 种子）**

| 种子 | 对照 mean | 干预 mean | Δmean | 对照 crit | 干预 crit | Δcrit |
|---|---|---|---|---|---|---|
| 42 | 0.6157 | 0.6164 | +0.0007 | 0.7010 | 0.7541 | +0.0531 |
| 123 | 0.6132 | 0.5956 | −0.0176 | 0.6116 | 0.7309 | +0.1192 |
| 2024 | 0.6123 | 0.6005 | −0.0118 | 0.6339 | 0.6112 | −0.0227 |
| **均值 ± 标准差** | — | — | **−0.0096 ± 0.0093** | — | — | **+0.0499 ± 0.0710** |

**判定**（规则预先固定：CONFIRMED_POSITIVE / LIKELY_POSITIVE / NOISE / CONFIRMED_NEGATIVE）：Δmean 三种子均值 −0.0096 ± 0.0093（1/3 为正）、Δcrit +0.0499 ± 0.0710（2/3 为正，seed 2024 为负）——**CONFIRMED_NEGATIVE**。seed 42 的“零和被打破”确认是种子噪声；可微奖励接口与前三种干预同构地落入召回—精度零和。该复测同时说明：单种子 A/B 的 ±0.005 显著性线对本任务的种子间离散度（Δmean 标准差 0.0093）而言过紧，任何单臂结论都必须经种子复测方可采信——这本身是方法层面值得写入的教训。

**总结论**：三种“改权重/改度量”干预在强基线上全部复现召回—精度零和（mean −0.015~−0.029 换 crit +0.10~+0.18）——**零和不是某一干预的副作用，而是系统性结构特征**；唯一的可微奖励接口（Layer-1）的表观同向改善经三种子复测确认是噪声（Δmean −0.0096 ± 0.0093），零和在该接口上同样复现；GRPO 实证不可救药。**四种机制互异的干预接口（权重/度量/可微奖励/策略优化）在强基线上全部落入同一零和，是本工作最稳健的负面结论。** 创新点的诚实定位：七维奖惩作为性能增强手段被数据否定，但作为故障诊断探针极其成功。

### 4.9 方向 3：代价敏感 objectness 头（标定层干预）

4.6 已证低置信度漏检是**背景分缺失**问题，4.7/4.8 已证训练信号干预无法修复。方向 3 换**架构/标定层**杠杆：在解码器最终隐状态上新增专用 objectness 头 `Linear(256,1)`，以代价敏感 BCE（关键类 w_crit=3.0、背景 w_bg=0.1）独立训练"该 query 是否命中真实目标"的前景/背景分，与 18 路类别 logits 解耦，专治 DETR"无标定背景分"。Stage-1 线性探针（冻结所有既有参数、仅训 257 个头部参数、3 epoch，保证不损伤检测器）。

**表 10　objectness 头 vs max-class-confidence 的 B 口径（匹配召回，957 张 val + 141 张 AITEX 跨域负样本）**

| 口径 | FPR @ R0.90 | FPR @ R~0.61 | FPR @ R~0.54 | R @ FPR~0.04 |
|---|---|---|---|---|
| max-class-confidence | 0.99 | 0.47 | 0.36 | ~0.01 |
| objectness | **0.67** | **0.15** | **0.064** | **0.48** |

- **回归检查（探针冻结有效）**：crit_recall 0.5743→0.5776、mAP50 0.537（基线 0.541）、逐类召回不变——检测器权重未被触碰。
- **结论（部分正，优势维度已精确化）**：专用 objectness 头是显著更好的**跨域**前景/背景打分器——B 口径匹配召回下跨域 FPR 降 2–6 倍，固定 4% 误报率下图像级召回从 ~1% 提到 ~48%，首次在 B 口径拿到真实标定改善，直接印证 4.6 的"无标定背景分"机制假设；但同协议图像级回放（上文）显示其域内判定全面劣于 max-class-conf，故其价值应表述为"跨域门限的鲁棒性"而非"更好的报警门限"。
- **但天花板仍在特征层**：到召回 0.90 时两门限都须压到 τ≈0.05、FPR 依旧爆表（objectness 0.67）。线性探针只能对冻结特征做线性分离，改不了特征本身的判别力，与 4.7/4.8"瓶颈在特征级判别力"一致——Stage-2 联合微调预期低收益且大概率重演关键类召回回退，未跑。

**图像级二分类回放（域内，2026-10-20，`scripts/eval_binary_replay.py`）**：val 957 张 = 583 有瑕疵 + 374 无瑕疵（域内负样本），判定规则 = 300 查询最大置信度 > τ。mosaic1280 最强权重：

| 口径 | AUC | 最佳正确率（τ） | 召回 | 误报 | 备注 |
|---|---|---|---|---|---|
| 域内（天池 val） | 0.9808 | **93.73%**（0.35） | 95.2% | 8.6% | τ=0.25 时 91.2%；基线 tianchi18_full 最佳 92.06% |
| 跨域（AITEX 141 张负样本） | — | — | — | **71.6%**（τ=0.35） | 域内 8.6% → 跨域 71.6%，分离属性不迁移 |

**同协议 objectness 回放（2026-10-21，`scripts/eval_objectness_replay.py`，存档 `runs/calibration/objectness_replay.json`）**：把 objectness 头放到与上表**完全相同**的回放协议（957 val + 141 AITEX、方形 letterbox、τ 网格 0.05–0.95），且一次前向产出两个分数（同模型、同查询，唯一变量=分数口径）。注意该探针的基座是 tianchi18_full（探针训练时的 `--weights` 默认值），故同运行的 max_class_conf 列（AUC 0.9685 / 最佳正确率 91.64% @ τ=0.35）才是本表的对照：

| 分数口径（同模型同查询） | AUC | 最佳正确率（τ） | 域内 FPR @最佳 τ | 跨域 AITEX FPR @最佳 τ |
|---|---:|---:|---:|---:|
| max-class-confidence | **0.9685** | **91.64%**（0.35） | **7.75%** | 43.3% |
| objectness | 0.8875 | 80.98%（0.15） | 34.5% | **31.2%** |

**修正后的 objectness 结论（部分正，优势维度精确化）**：objectness 头的价值**不在域内图像级判定**——同协议下 AUC −0.081、正确率 −10.7 点、域内误报 4.5 倍，全面劣于 max-class-conf（与 4.12"域内不敌 max-conf"一致）；其真实优势是**跨域前景/背景分的鲁棒性**：B 口径匹配召回下跨域 FPR 降 2–6 倍（表 10），图像级同协议下跨域 FPR 31.2% vs 43.3%（−28% 相对）。即：objectness 头是更好的**跨域**前景/背景打分器，不是更好的域内报警门限——这一区分此前被"FPR 降 2–6 倍"的表述掩盖，同协议回放使其精确化。对部署的含义：域内用 max-conf（或分级告警框架），跨域场景才值得切换到 objectness 门限。

**分层结论**：① 域内最大置信度门限完全可用（较基线 +1.67 点，高阈值端 +4.7 点）；② 跨域失效与 4.9 的 objectness 结论互为印证——objectness 头的价值正在跨域门限（FPR 降 2–6 倍）；③ **二分类判断正确率 ≠ 缺陷定位精度**（关键类召回仍 0.52–0.72），不得表述为"系统整体正确率 93.73%"。

### 4.10 数据驱动缺陷族结构与级联粗分类诊断

**动机**：细粒度级联（粗→细）需要"族"级粗分类定义；手工分组含 catch-all，需用数据替换。

**方法**：训练集 8570 个 GT 框裁剪图（padding 15%，224×224）过冻结 ResNet50（ImageNet V2）提 2048 维特征，L2 归一化；类级质心 → KMeans（n_init=200, 20 seeds）+ Ward（cosine）对照；k=2..8 扫描 + bootstrap 稳定性（100 轮 × 80% 框重采样）；与 18 类基线 val 混淆矩阵（conf>0.25, IoU>0.45, 2375 计数）交叉验证。

**结果**：**k=3 是唯一的稳定结构**（KMeans 与 Ward 划分完全一致，silhouette 0.716，bootstrap 内聚 support 三族均 1.00）：A 线性结构族（7 类，2912 框：三丝、粗经、松经、断经、吊经、跳花、织疵）、B 斑渍族（5 类，1268 框：污渍、花板、百脚、浆斑、密路）、C 点状族（6 类，4390 框：破洞、结头、毛粒、纬缩、整经结、断氨纶）。**k=6 不被数据支持**（稳定性 0.286，退化为"3 组 + 3 单例"）。混淆矩阵验证：主要易混对落在族内；最大错分对三丝↔结头（38/28）为跨族例外。

**粗分类器验证（级联第一级）**：相同裁剪协议，非冻结微调（3 类头 + 类加权 CE + hflip/vflip 增广；AdamW 1e-4 + OneCycle，25 epoch）vs 冻结特征线性探针，val 953 框（bootstrap 1000 次 95% CI）：

**表 11　族级粗分类器验证（val 953 GT 框）**

| 模型 | top-1 准确率 | 95% CI | A 召回 | B 召回 | C 召回 |
|------|------|------|------|------|------|
| 冻结特征 + 线性探针 | 0.8835 | [0.864, 0.905] | 0.845 | 0.896 | 0.906 |
| **非冻结微调** | **0.9119** | [0.894, 0.929] | 0.854 | 0.928 | 0.946 |

① 3 族粗分类达到 90%+（0.912）；② 非冻结微调比线性探针 +2.8 点，微调必要；③ 残余错误集中在 A↔C 跨族互混（59 框，占错误 70%）——细线状缺陷（三丝 vs 结头）是粗分类天花板；B 斑渍族几乎完全分离。

**对级联的意义**：第一级（3 族，91.2%）+ 检测器（mAP50 0.54–0.60）构成系统前端；细粒度阶段的天花板由检测器 per-class 召回决定（48~72%），符合预期的级联误差累积结构。

### 4.11 与同类方法对比

**外部同类方法对比**（均同一 Tianchi 数据集；⚠️ 口径已逐篇核验，公开方案普遍只报 mAP@0.5）：

| 方法 | 骨干 | 类别 | mAP@0.5 | 口径核验 | 出处 |
|------|------|------|-------|---------|------|
| SPFFNet | YOLOv11 | **8 类子集** | **65.8%** | v3 表 I 明确 mAP@0.5，仅 8 类 | arXiv:2502.01445 |
| 改进 RT-DETR（牛仔布） | RT-DETR-R18 | 20 | 60.0% | 20 类（原文口径） | 浙江大学学报 2025, 59(6):1169 |
| Fab-ME | YOLOv8s | 20 | 59.4% | 20 类（原文口径） | arXiv:2412.03200 |
| **本工作 RT-DETR-L（20 类基线）** | RT-DETR-L | 20 | **59.6%** | 本工作 2026-10-15 复算 0.5963 | 本工作 |
| **本工作 RT-DETR-L（18 类合并）** | RT-DETR-L | 18 | **54.0%** | 本工作 2026-10-15 复算 0.5403 | 本工作 |
| **本工作 RT-DETR-L（阶段 1 最强）** | RT-DETR-L | 18 | **59.2%** | 官方口径 mean **0.6349** | 本工作 |

**对比结论（诚实）**：① 复算证实本工作 20 类基线 mAP@0.5 = 0.596，与 Fab-ME（59.4%）持平、贴近改进 RT-DETR（60.0%），距 SPFFNet 65.8% 约 6 点——但 SPFFNet 为 8 类子集口径，不可直接对比；② 公开方案均未报官方口径，按 mAP@0.1/0.3 通常高于 mAP@0.5 的规律，其官方口径应高于各自 mAP@0.5 报值，故为指示性对比；③ 本工作阶段 1 最强模型（59.2% mAP@0.5 / 0.6349 官方口径）在同等协议下达到公开方案区间上沿；④ 公开方案均无标签噪声诊断与代价感知训练，本工作差异化在方法学而非单纯拼 mAP。⚠️ 可比性边界：类别数、骨干、指标口径与训练/测试切分均有差异，全部对比为指示性。《棉纺织技术》某文声称 mAP50=90.9%，原始出处无法核验，**不采信**。

### 4.12 推理侧多维报警评分与安全决策

**动机**：训练侧多维奖惩呈结构性零和（4.7/4.8），但多维思想本身未必无效——问题可能出在“通过梯度改权重”这一接口。推理侧不受可微约束：同一多维框架可以直接对检测器输出打分，通过离线回放标定阈值，为工业报警提供可解释、可调的决策接口。**本节报告该迁移的完整离线验证（含阴性结果）。**

**七维定义**（推理侧，与训练侧共享方法论、不共享维度；完整表见 `runs/alarm_replay/REPORT.md`）：

| 维 | 名称 | 定义 | 信息来源 |
|---|---|---|---|
| D1 | 严重度 | critical（破洞/断经/断氨纶）1.0、次级（跳花/织疵）0.6、其余 0.3 | 训练侧严重度分类法 |
| D2 | 置信度 | top1_score | 检测器输出 |
| D3 | 不确定性 | 1−(q1−q2)/q1（q=18 类全局 softmax） | 检测器输出 |
| D4 | 相似组冲突 | 实证混淆伙伴（混淆矩阵计数 ≥5 的 8 对）组内归一化后最强 runner-up/top1 | 混淆矩阵 Top 对 |
| D5 | 目标尺寸 | min(1, bbox 面积/图像面积 0.05) | 检测器输出 |
| D6 | 跨帧稳定性 | 单图场景禁用，权重重分配 | — |
| D7 | 空间聚集 | min(1, 半径 0.15 内 conf≥0.25 邻居数/5) | 检测器输出 |

**实现要点（两处实证修正）**：① 计划原始线性形式 R=ΣwᵢDᵢ 在 300 查询输出上**饱和度**——每个检测的 Rᵢ 有 ~0.2 地板（D1≥0.3、D3/D4≥0），未加证据门时 noisy-or 聚合后退化为“有无证据”的二值判据（首跑 AUC 0.386，全部权重变体结果完全相同）；加入 conf≥0.25 证据门并改用乘积形式 R=D1×Q（Q={D2, 1−D3, 1−D4, D5, D7} 加权均值，严重度门控置信度质量）后恢复分辨力。线性形式（加证据门）保留为对照臂：其正确率 0.9154 与等权乘积变体完全一致——权重与组合规则均不影响结果，佐证评分分辨力已饱和。② noisy-or 只聚合 conf≥0.25 的证据检测。评分器为纯 numpy（`backend/inference/alarm_scorer.py`，18 个单元测试），回放数据由 monkey-patch `RTDETRDecoder.postprocess` 捕获逐查询 18 类分数（与二分类回放 max-conf 校验一致，差异 5e-6）。

**表 11b　七维报警评分 vs max-conf 基线（957 张 val 域内 + 141 张 AITEX 跨域，5 折 CV 选 τ）**

| 指标 | max-conf 基线 | 七维主变体（等权乘积 noisy-or） | 七维消融最佳（drop D3+D4） | 判定 |
|---|---:|---:|---:|:--:|
| AUC | **0.9808** | 0.9399 | 0.9463 | 负 |
| 域内最佳正确率（τ） | **0.9383**（0.34） | 0.9154（0.04） | 0.9269（0.04） | 负 |
| 5 折 CV 正确率 | **0.9352±0.0122** | 0.9112±0.0047 | 0.9248±0.0090 | 负 |
| 关键类召回 @最佳 τ | 0.9569 | 0.9638 | **0.9586** | 平 |
| 等召回对比（TPR≈0.95） | FPR **0.0829** | FPR 0.1604 | FPR 0.1230（最佳 τ 工作点 TPR 0.959） | 负（1.5–2 倍） |
| 跨域 AITEX 误报率 | **0.7163** | 0.7447 | 0.7447 | 负（无优势） |

> 注：max-conf 基线的 0.9383 @ τ=0.34 与 4.9 二分类回放的 93.73% @ τ=0.35 为同一批查询分数（校验差异 5e-6）在不同 τ 网格粒度下的扫描结果；本表七维两列分别给出预先设定的主变体与消融最佳变体，避免"挑最佳"的选择性报告。

**预先承诺的验收标准核对**：域内准确率 ≥94% ✗（主变体 0.9154，消融最佳 0.9269）｜关键类召回 ≥95% ✓（0.9586–0.9638）｜误报率 ≤6% ✗｜复核触发率 ≤15% ✗（20.6%）——**四项中一项，主结果判负**。

**消融的机制发现**：D2（置信度）是唯一承重维（去掉后 0.9049，−0.02 正确率）；D3/D4（不确定度/组冲突）加入后**损害**域内表现（等权 0.9154 → drop D3 0.9195 / drop D4 0.9206 / drop D3+D4 0.9269，去掉后即为最佳变体）——与训练侧“改度量干预复现零和”同一签名；D7（空间聚集）有适度贡献（drop 后 0.9185）；D1/D5 域内中性。逐维均值进一步印证：有瑕疵图 vs 无瑕疵图的分离主要来自 D2（0.531 vs 0.374）与 D7（0.199 vs 0.043），D3 甚至反向（干净图更"不确定"，0.775 vs 0.705）。

**三档决策框架**（等权七维，τ_stop=0.14 / τ_review=0.02）：停机 45.1% / 人工复核 20.6% / 放行 34.3%；停机档自身误报保守（FPR 5.9%，22/374 干净图，关键类召回 0.707），停机+复核合并的关键类召回 **0.9672**（580 张含关键类图中漏 19 张）。三档机制本身可用，但复核档把 17.4% 的干净图（65/374）推入人工复核——告警流量的代价不被官方口径买单。

**结论**：多维思想向推理侧的迁移同样不突破 max-conf 基线——不是组合规则问题（线性/乘积/最大聚合三种规则均试过）而是**信息增量问题**：检测器自身置信度已承载报警相关信息，多维重加权无法新增特征级不含的信息，与 4.7/4.8 训练侧结论同根（天花板在特征级判别力）。本节仍有两点贡献：① 可解释的逐检测归因接口（severity/置信度/冲突/尺寸/聚集），相对 max-conf 黑箱单值是部署侧的实质改进；② 三档分级告警框架与 96.7% 的合并召回可直接用于产线。**定位：推理侧迁移的阴性结果 + 可解释决策接口，不宣称性能增强。**

### 4.13 置信度校准审计：质量、可修复性与可分性的边界

4.12 的阴性结果留下一个未回答的问题：max-conf 基线不可战胜，是因为置信度不含信息，还是因为信息以不可分的形式存在？本节用校准理论把该问题定量化，并检验两类修复手段。

**协议**：957 张 val、300 查询/图、conf 地板 0.001，共采集 287,100 条预测（tianchi18_full：TP 901 / FP 286,199；mosaic1280：TP 915 / FP 286,185）。TP/FP 标签按 r_curve 同口径匹配（类感知、GT 主、IoU≥0.5，与 `utils/metrics.py` 的 `match_predictions` 一致），在全部 300 查询上计算、与置信度阈值无关，故任何阈值都是纯事后重标注。ECE 为 10 个等宽 bin；所有重校准按图像 5 折交叉拟合（在 4 折上拟合、留出折报告），运行区域定义为 conf≥0.05。脚本 `scripts/eval_calibration_audit.py`，存档 `runs/calibration/audit_{tianchi18,mosaic1280}.json`。

**表 11c　校准审计（ECE；运行区域口径，全区域见末行参考）**

| 模型 | 原始 | 全局 TS | 按族 TS | 按尺寸 TS | 仅运行区拟合 TS | **isotonic** |
|---|---:|---:|---:|---:|---:|---:|
| tianchi18_full（mean 0.572） | 0.1055 | 0.1774 | 0.1675 | 0.1665 | 0.2045 | **0.0371** |
| mosaic1280（mean 0.635） | 0.1116 | 0.1721 | 0.1662 | 0.1671 | 0.1847 | **0.0456** |
| 全区域参考（tianchi18 / mosaic） | 0.0219 / 0.0182 | 0.0029 / 0.0028 | 0.0030 / 0.0029 | 0.0032 / 0.0029 | — | 0.0005 / 0.0005 |

三条结论：

1. **运行区域失准真实且严重，且是曲线形状问题而非温度偏移**。tianchi18_full 的可靠性表显示 conf∈[0.5,0.6) 的预测实际正确率仅 0.171、conf∈[0.7,0.8) 仅 0.291——过信 2–3 倍，且缺口随置信度扩大（acc/conf 从 conf≈0.14 处的 0.11 扩大到 conf≈0.75 处的 0.39）。单参数温度缩放因此**无效**：全局 NLL 最优 T≈0.6（被 97.6% 的低置信度 FP 质量主导，呈锐化方向）把运行区域 ECE 从 0.1055 **恶化**到 0.1774；只在运行区域拟合同样无解（0.2045）。isotonic（PAVA 单调回归，可重塑任意单调可靠性曲线）则把运行区域 ECE 修到 0.037/0.046（全区域 0.0005）——该失准可由非参数单调重校准修复，且是部署侧零成本的查表后处理（部署时应在独立标定集上拟合）。这与 Guo et al. [15] "温度缩放足够" 的结论形成有趣对照：在检测器 300 查询、极端类别不平衡的输出上，单参数假设不成立。
2. **任何单调重校准都不改变可分性**——温度缩放前后 TP/FP 的 AUC 严格不变（0.7631 / 0.8228，实测验证了单调性）。这把 4.12"真检与误检无法按置信度分离"从断言升级为定量命题：原始 max-conf 的 TP/FP AUC 为 0.763（弱基线）/ 0.823（强基线），关键类 pooled 0.812 / 0.871，这是**任何**单调重校准的天花板；七维评分 0.9399 的 AUC 若成立，必来自结构而非校准。
3. **条件校准无增益**（按缺陷族/按框面积的温度缩放与全局基本相同）——失准不存在可按族或尺度利用的结构；逐类看 ECE 0.078–0.150 均匀分布，warping_knot 可分性最差（AUC 0.643）。真缺陷的置信度中位数仅 0.41–0.45（mosaic1280 上 hole 0.75，broken_warp 仍 0.47），与 4.6"低于阈值漏检"互相印证：失准的主体正是那些被任何全局阈值漏掉的真缺陷。

**对报警框架的含义**：isotonic 重校准后的置信度可作为概率语义输入（告警阈值可表述为"该检测为真的概率"），三档决策（4.12）因此获得可解释的概率刻度；但告警性能上界由 AUC 0.763/0.823 锁定，突破须引入与分数正交的信息（如集成分歧，见阶段 2 计划书 `docs/plan_stage2_adaptive.md`）。

---

### 4.14 自适应损失加权（UW）：自动学习 cls/reg 权重的隔离检验

4.7/4.8 的全部干预有一个共同结构：在默认损失之上**叠加**奖励或做**类条件**重加权，而 ultralytics 的基础增益 `class×1, bbox×5, giou×2`（回归臂比分类臂重 5–7 倍）从未被隔离检验。官方指标 mean(mAP@0.1/0.3/0.5) 有 2/3 权重在 IoU 0.1/0.3 档，低 IoU 下定位几乎免费、分类质量主导——若默认增益对本指标回归过重，则存在一个从未被测过的权重方向。本节用 Uncertainty Weighting（Kendall et al. [16]）检验该方向：两个可学习对数方差经解析梯度更新 `lv_i −= lr·(1−w_i·L_i)`、`w_i=e^{−lv_i}`，收敛到稳态 `w*=1/L_i`，即按损失值自动均衡两臂。

**协议**：与 4.8 强基线对照逐项一致（mosaic1280 best.pt 起点、5ep、b4、1280 方形、lr0 1e-4、amp=False、mosaic 0、deterministic），唯一变量=UW 加权。实现要点（`backend/training/adaptive_loss.py`）：EMA 平滑损失（回归组损失周期性归零会污染更新）、零损失臂冻结、**尺度保持归一化**（总损失恒等于默认，唯一变量=cls:reg 比值，"等效学习率"混杂按设计消除）。判定门在实验启动前登记于 `docs/plan_stage2_adaptive.md`：种子平均 mean > 0.6208（对照 0.6158 + 0.005）且 crit_R 不降；**零和条款**——crit_R 升 >0.02 且 crit_P 降 >0.02 判零和复现、不计转正。

**表 12　UW 四种子（官方口径；对照 strong_r2_uniform mean 0.6158 / mAP50 0.5710 / crit_R 0.6260 / crit_P 0.6371）**

| 种子 | mean | mAP50 | crit_R (Δ) | crit_P (Δ) | 权重终点 w_cls/w_reg | 零和条款 | 判定 |
|---|---:|---:|---:|---:|---|---|---|
| 42 | 0.6117 | 0.5689 | 0.7914 (+0.165) | 0.5337 (−0.103) | 2.091 / 0.654 | **触发** | 不计转正 |
| 123 | 0.6231 | 0.5795 | 0.6973 (+0.071) | 0.6507 (+0.014) | 2.074 / 0.659 | 未触发 | **转正** |
| 2024 | 0.6284 | 0.5802 | 0.7243 (+0.098) | 0.5933 (−0.044) | 2.103 / 0.659 | **触发** | 不计转正 |
| 7 | 0.6070 | 0.5658 | 0.7395 (+0.114) | 0.5588 (−0.078) | 2.087 / 0.650 | **触发** | 不计转正 |
| **平均** | **0.6176** | | 0.7381 | 0.5841 | | | |

三个结论：

1. **权重轨迹是稳健的架构层诊断**。四种子独立收敛到几乎相同的比值 **3.20 / 3.15 / 3.19 / 3.21**（spread 1.02×），即 UW 稳定地把分类臂权重推到默认值的 ~2 倍——这实证了"默认 cls:reg = 1:7 的增益结构对官方指标欠优"：指标 2/3 权重在分类主导的低 IoU 档，而默认增益把容量压在回归臂。该结论与干预成败无关，是本节最稳健的输出（轨迹图 `runs/ablation_adaptive/uw_trajectories.png`，四种子叠图）。
2. **干预效应不可与种子噪声区分，且第 4 种子把暂定判定解决为负**。三种子时严格适用零和条款后仅 1/3 真转正（seed 123 三项全升），判 UNSTABLE 并预登记补第 4 种子；seed 7 结果为 mean 0.6070（Δ=−0.0087，唯一种子均线低于对照）、crit_R +0.114 换 crit_P −0.078（零和条款触发）。四种子合计：种子平均 mean 0.6176 ≤ 转正门 0.6208，真转正计数 1/4（预登记 2/3 规则需 ≥3），种子间跨度 0.0214 是判定门宽的 4.3 倍——**效应量小于种子离散度**，按预登记决策树（种子平均 mean ≤ 门 → 零和复现记负结果）判定 **UW NEGATIVE**：自适应损失加权同样无法稳定突破零和。
3. **对机理章节的意义**：连同 4.8（静态奖惩）、4.12（推理侧重加权）、4.13（单调重校准），本节给出第四层证据——**连"按损失值自动学习的最优权重"也无法稳定突破零和**。瓶颈不在权重形式（静态/类条件/自适应都不是原因），而在 5.5 所述的判别结构本身。反之，四种子一致的权重轨迹证明默认增益欠优却修正无益，恰说明该结构是内生的：重新平衡两臂只是把容量在召回与精度之间搬家。

### 4.15 集成不确定性：贝叶斯路线的实用替代检验

4.12–4.14 的全部阴性结果留下最后一个推理侧可能：贝叶斯方法建模的认知不确定性（epistemic uncertainty）。MC Dropout（Gal & Ghahramani, 2016）在本架构不可用——RT-DETR decoder 无 dropout 层；深度集成（Lakshminarayanan et al., 2017）因此是本架构下唯一现实的认知不确定性来源。计划书（`docs/plan_stage2_adaptive.md` §2B-2）预登记判定：**若分歧闸门在固定召回下 FPR 显著低于分数阈值（如 −20% 相对），推理侧结构化优化成立**。本节用两套同起点（mosaic1280 best.pt）不同种子的集成检验：ctl3（strong_r2_uniform s42 5ep + l1_seed{123,2024}_ctl 10ep，协议注记：成员训练轮数 5/10/10）与 uw4（uw_seed{42,123,2024,7}，协议匹配：唯一变量=种子）。方法：成员在 val 上按官方方形协议推理（300 查询/图），全成员预测按"同类 + IoU≥0.5"贪婪聚类，每簇记录触发成员数、均值分与标准差（=该位置的分歧度），簇按 r_curve 同口径匹配 GT。

**表 13　深度集成不确定性审计（953 GT，conf 地板 0.001）**

| 指标 | ctl3（3 成员） | uw4（4 成员） |
|---|---:|---:|
| 成员召回 | 0.9517 / 0.9517 / 0.9538 | 0.9496 / 0.9475 / 0.9496 / 0.9528 |
| 集成召回 | **0.9570** | **0.9570** |
| 集成提升（vs 成员均值） | +0.0045 | +0.0071 |
| AUC（集成均值分，TP/FP 可分性） | 0.9699 | 0.9773 |
| AUC（分歧度 std_conf） | 0.9650 | 0.9792 |
| std 均值（TP / FP） | 0.157 / 0.004（40×） | 0.121 / 0.002（65×） |
| 全成员可见 GT 占比 | 878/953（92.1%） | 863/953（90.6%） |

**表 14　分歧闸门 vs 分数阈值（t=0.25 工作点；R=TP 簇召回，FPR=FP 簇误报率）**

| 集成 | 纯分数阈值 | + 分歧闸门（std≤0.20） | 判定 |
|---|---|---|---|
| ctl3 | R 0.2215 / FPR 0.0009 | R 0.0855 / FPR 0.0006 | 召回 −61%，FPR 仅 −33% 且绝对量已在地板 |
| uw4 | R 0.1250 / FPR 0.0002 | R 0.0647 / FPR 0.0002 | 召回 −48%，FPR **不变** |

三个结论：

1. **贝叶斯路线的实用替代不成立**。预登记标准（固定召回下 FPR 降 ≥20%）未达到：匹配召回下纯分数阈值的 FPR 不高于门控——按门控工作点召回在纯分数网格上插值，ctl3 门控点（R 0.0855 / FPR 0.0006）对应纯分数 t≈0.43 / FPR≈0.0002，uw4 门控点（R 0.0647 / FPR 0.0002）对应 t≈0.34 / FPR≈0.0001，两套集成上纯分数阈值均不劣于门控。机制原因决定性：**分歧度与 TP 正相关而非 FP**（TP 簇 std 是 FP 簇的 40–65 倍）——同起点微调的成员对真实缺陷多点触发、置信度有散布（std 高），而误检簇多为单成员突发（std≈0，"低分歧"）。低分歧闸门因此过滤掉的恰是 TP、保留的恰是 FP，与认知不确定性的直觉方向相反。
2. **同起点深度集成不构成认知不确定性来源**。90%+ 的 GT 被全体成员同时看到（表 13），成员间相关性太高——集成行为是置信度平均而非贝叶斯模型平均；其召回提升（+0.004~0.007）符合经典集成增益但幅度很小，且不改变任何可分性结论（与 4.13 的 AUC 不变互为印证）。
3. **对机理章节的意义**：连同 4.12（多维重加权）、4.13（单调重校准），推理侧的最后一条结构化路线——引入集成歧义作为正交信息源——同样无法比单一分数阈值更便宜地压低误报。这补全了 5.5 边界声明的实证：突破必须引入与既有分数**正交**的信息源，而本架构内可生成的歧义信号（深度集成）已被证明不含该信息。

---

### 4.16 阶段一离线回放：分布排序评分与产线决策层分析

论文定稿后，在不触碰 GPU、与训练链并行的前提下，对 mosaic1280 best.pt 的回放数据（与 4.12 同管道：val 957 张逐查询 18 类分数 + AITEX 141 张跨域负样本）做了五项 CPU 分析，预承诺门全部实验前登记于计划文档（`docs/plan_stage3_breakthrough.md` 同族流程），事后不移动。

**表 15　阶段一回放分析总表**

| # | 分析 | 预承诺门 | 实测 | 判定 |
|---|---|---|---|:--:|
| 1.1 | 分布排序七维评分（B2） | 正确率 ≥0.9373 | 0.9248（2 折交叉拟合） | **NEG** |
| 1.2 | 固定复核预算选择性拦截（C3） | 关键缺陷拦截率显著 > max-conf | 是（但归因于严重度先验） | POS（归因修正） |
| 1.3 | 工业成本模型（C7） | 三档总成本 < 单阈值 | 全部成本比下更贵 | **NEG** |
| 1.4 | 七维可调性曲线（C8） | 连续可调 | 两分数均平滑单调 | POS（描述性） |
| 1.5 | 域偏移分布分析（C4 前置） | 无门（描述性） | AITEX max-conf 4.71×，域可分 AUC 0.856 | 信号存在 |

**分布排序评分（1.1，负）**。将 Dr. Seg（Sun et al., arXiv:2603.00152，VLLM 推理分割后训练）的 Distribution-Ranked Reward 思想迁移到推理侧：把每个检测的置信度与 top1−top2 间隔变换为**类条件经验分位数**（2 折交叉拟合避免同源乐观），与 D1 严重度门控组成乘积形式 R=D1×Q，noisy-or 聚合规则与 4.12 主变体一致。

**表 16　分布排序变体 vs 基线（val 957，域内图像级判定）**

| 评分 | 定义 | 最佳正确率（τ） | 关键类召回 |
|---|---|---:|---:|
| max-conf 基线 | — | **0.9383**（0.34） | 0.9569 |
| 七维主变体（4.12） | D1×Q（原始分） | 0.9154（0.04） | 0.9638 |
| RANK-A | D1×q_conf | 0.9248（0.272） | **0.9703** |
| RANK-B | D1×(0.5·q_conf+0.5·q_margin) | 0.9133（0.209） | 0.9703 |
| RANK-C | D1×(0.5·q_conf+0.25·q_margin+0.25·D7) | 0.9143（0.206） | 0.9703 |

交叉拟合 0.9248 低于预注册中性带下限（0.925），判负；in-sample 版本（0.9258）同判负。**机制**：分位数变换使跨类分数可比、利于类内判别，但丢失了绝对置信度水平——noisy-or 跨类聚合时，稀有类低分检测与常见类低分检测被等同对待，而两者为真的先验不同。关键类召回升至全场最高（0.9703）说明严重度门控+分位数对关键类有正贡献，但域内整体正确率仍不敌 max-conf 1.35 个点——与 4.12/4.13 同结论：报警相关信息已由检测器自身置信度承载，分数形式的后处理无法新增可分性。

**固定复核预算拦截（1.2，正但归因修正）**。按分数降序取前 B 张送人工复核：

**表 17　关键类图/框拦截率（101 张关键类图、166 个关键类框）**

| B | max-conf | 七维 R_global | sev_weighted（严重度×conf） |
|---:|---:|---:|---:|
| 100 | 17.8% / 13.9% | 51.5% / 66.9% | **73.3% / 79.5%** |
| 200 | 37.6% / 44.0% | 75.3% / 83.7% | **89.1% / 92.2%** |
| 300 | 52.5% / 56.6% | **93.1% / 95.2%** | 91.1% / 94.0% |
| 500 | 88.1% / 91.0% | **96.0% / 97.0%** | 95.1% / 96.4% |

门核对通过（类感知评分远高于 max-conf），但**归因必须修正**：R_global 含 D1 严重度先验而 max-conf 是类无关的，直接比不公平。补最简单的类感知单分对照臂 sev_weighted 后，小预算（B≤200）下它全面优于完整七维机制（B=100：73.3% vs 51.5%），七维机制的增量只在 B≥300 才出现（+2pp）。**拦截优势来自类严重度先验，不来自七维置信度质量维度**；产线实现用严重度加权排序即可。

**工业成本模型（1.3，负）**。设 C_漏检=r·C_误停机、C_复核=0.1、人工复核发现缺陷概率 0.9，r 扫描 1–50：

**表 18　总成本（归一化，C_误停机=1）**

| r | 单阈 max-conf | 三档 R_global（论文，τ=0.14/0.02） | 三档 max-conf（双阈优化） |
|---:|---:|---:|---:|
| 1 | **59.0** | 76.1 | 42.8 |
| 10 | **175.0** | 385.7 | 89.0 |
| 50 | **293.0** | 1761.7 | 163.8 |

论文三档（R_global）在全部成本比下贵于单阈值（r=50 时贵 6 倍），判负；但三档机制挂在 max-conf 上比单阈省 28–45%——**三档决策机制本身有成本价值，但需要更强的分数**。与 1.2 合并：产线决策层的最优配置是"严重度加权/强分数 + 三档"，不是"七维 R + 三档"。

**可调性曲线（1.4，描述性通过）与域偏移（1.5）**。max-conf 与 R_global 的 TPR-FPR 曲线均平滑、FPR 随 TPR 单调不增（连续可调）；max-conf 在每个工作点占优（TPR≈0.95 时 FPR 0.080 vs 0.134）。域偏移方向符合预期：AITEX 逐检测置信度均值为 val 干净图的 2.05 倍、图像级 max-conf 为 4.71 倍，单统计量域可分 AUC 0.856——为 4.9 objectness 头的跨域价值与 C4 保守模式触发提供依据，但不足以单独作门。

**数据修正**：既有回放报告（`runs/alarm_replay/REPORT.md`）称"580 张含关键类 GT"与原始标注不符——关键类实为 101 张图 / 166 个 GT 框（953 总框不变），580 接近有瑕图总数 583，疑为计数口径错误；本节一切关键类统计以重算值为准。

---

## 5 讨论

### 5.1 双层天花板与干预的边界

天花板有两层：第一层标签噪声（可修复，类合并，召回 0.40→0.88）；第二层细缺陷低置信度漏检（七种训练信号干预在弱基线与强基线上均无法修复，故障在特征级判别力）。召回—精度零和边界（Layer-1 及其强基线重测）表明：奖励只能**搬移**指标，不能**新增**可分性。

### 5.2 接口错位与忠实低判别力

七维奖惩的 D01/D03/D04 命中故障，但接口（聚合 box 指标/分类损失梯度）与真实故障层（query 触发 + 定位自评）不对齐。强基线重测把这一解释从"干预无效"升级为"干预有效但搬移容量"——三种机制不同的干预全部复现同一零和签名，且可微奖励接口经三种子复测同样落入零和（seed 42 表观为噪声），是系统性结构特征。更深层是 DETR"专家策略"下的忠实低判别力：每个目标仅一个"主预测" query，置信度被训练为自身预测 IoU，细缺陷 IoU 天然低且无第二 query 兜底。

### 5.3 四个可写进论文的发现

① **度量不公平**：全局 vs 逐类 max-F1 阈值相差 ~0.02 召回；② **逐类标定缺失**：各类最优阈值差异大，无统一背景分——objectness 头证实专用前景/背景分可 2–6 倍降低跨域误报，但冻结特征天花板仍在；③ **漏检奖励的匹配层盲区**：D04 的 soft miss 只对匹配后的 query 生效，未匹配 query 的漏检不在梯度视野内；④ **多维线性平均在 300 查询输出上饱和度**：推理侧七维评分的计划原始线性形式 R=ΣwᵢDᵢ 因每检测存在 ~0.2 得分地板，noisy-or 聚合后退化为二值判据（首跑 AUC 0.386、全部权重变体结果完全相同），须改为“严重度门×质量分”的乘积形式（4.12）。

### 5.4 对工业部署的启示

severity 驱动的分级告警（critical→停机、major→标记）+ **逐类成本敏感阈值**（召回 0.90 可达，误报洪水需人工复核，符合"漏检代价 ≫ 误检代价"）比全局 mAP 更贴近真实代价。推理侧七维报警评分（4.12）虽在域内不敌 max-conf，但其逐检测归因与三档决策（停机 45%/复核 21%/放行 34%，合并关键类召回 0.967）可直接作为产线分级告警的解释层，阈值需按产线误报容忍度现场标定。4.16 的决策层分析进一步给出可执行结论：固定复核预算下**严重度加权排序**（sev×conf）以 100 张预算拦截 73.3% 的关键类图、200 张拦截 89.1%，是最简实现；**三档机制挂在强分数上**比单阈值省 28–45% 总成本，但挂在七维 R 上全面变贵——三档决策框架有价值，分数必须是检测器原生强分数。训练侧，rect+mosaic 的因子分离结论（+0.063，每轮还更快）可直接复用到任何宽幅纹理检测任务。

### 5.5 机理综合：四层负结果指向同一结构

本文报告了四个层面的阴性结果（涵盖五种机制互异的干预），它们共同指向一个比"某个方法不够好"更深的结论。

**第一层，训练侧静态干预（4.7/4.8）**：七种手动奖励/惩罚在弱基线上全部负于均匀微调；强基线上三种改权重/改度量的干预全部复现召回—精度零和；可微奖励接口经三种子复测落入同一零和。**第二层，推理侧结构化重加权（4.12）**：七维报警评分（线性/乘积/最大三种聚合规则）域内不敌 max-conf 单值。**第三层，置信度校准（4.13）**：运行区域失准真实存在且严重（ECE 0.106/0.112，过信 2–3 倍），可被 isotonic 修复到 0.037/0.046，但**任何单调重校准都不改变 TP/FP 可分性**（AUC 0.763/0.823 逐位不变）。推理侧的最后一条结构化路线——以深度集成分歧度作认知不确定性闸门（4.15，贝叶斯路线的实用替代）——同样不优于分数阈值：匹配召回下 FPR 不降，且分歧度与 TP 正相关（TP 簇 std 为 FP 簇的 40–65 倍），同起点成员对 90%+ 的 GT 一致触发，集成歧义不含正交信息。

四层的共同结构因此清晰：**瓶颈不在分数形式（校准/重加权/聚合规则），而在判别结构本身**。TP 与 FP 在置信度维度上的不可分是严格意义的结构性命题——单调变换族（含温度缩放、isotonic、任何阈值策略）的全部作用空间内 AUC 不变；而多维重加权无法新增特征级不含的信息（4.12 的消融显示各维度贡献为噪声级）。这与训练侧零和互为表里：零和说明干预只能搬移召回—精度容量，校准审计说明该容量在分数维度上已无剩余可分性。

对默认损失增益的实证补全了这一图景：ultralytics 默认 `class×1, bbox×5, giou×2` 使回归臂比分类臂重 5–7 倍，而官方指标 mean(mAP@0.1/0.3/0.5) 有 2/3 权重在分类主导的低 IoU 档——**损失增益与指标权重存在系统性错位**。阶段 2 用 Uncertainty Weighting 让模型自动学习该比值：四种子独立收敛到几乎相同的权重（w_cls:w_reg = 3.20/3.15/3.19/3.21，spread 1.02×），**"默认增益欠优"因此是稳健实证**；但修正它不能稳定获益——严格适用预登记零和条款后仅 1/4 种子真转正，第 4 种子（seed 7，Δmean −0.0087）把三种子时的 UNSTABLE 暂定判定解决为负：种子平均 mean 0.6176 ≤ 转正门 0.6208，种子间跨度（0.0214）是判定门宽的 4.3 倍，效应量小于种子离散度（4.14）。连同前述三层，这是第四层证据：**连"按损失值自动学习的最优权重"也无法稳定突破零和——瓶颈不在权重形式（静态/类条件/自适应），而在判别结构本身**；权重轨迹证明默认增益欠优却修正无益，恰说明该结构是内生的：重新平衡两臂只是把容量在召回与精度之间搬家。

**边界声明**：本结论不排除突破可能，但划定了突破的必要条件——须引入与现有分数**正交**的信息源（如多视图/集成歧异、专用 objectness 头、或特征级数据增广），而非在既有分数上做任何后处理。4.12 的 objectness 探针（跨域 FPR 降 2–6 倍但域内不增益）提示该正交空间存在但收益取决于域。

### 5.6 多维训练信号中的量纲失衡：一个概念性机理

本文用"高方差主导"描述多维目标优化中量纲与方差异质、高方差项垄断梯度的现象。该命名未见于以同样术语发表的文献，故作概念性表述、不署名引用；但现象本身在相邻领域有明确的技术诊断，可作机理支撑：Dr. Seg（Sun et al., arXiv:2603.00152，VLLM 推理分割的 GRPO 后训练）把"高方差目标主导标准 GRPO 的梯度"列为其分布排序奖励所针对的两个被忽视因素之一；PDCR（Yoon et al., CVPR 2026）在视觉-语言 RLVR 中把全局奖励导致的异构步骤信号失真命名为 mixture-induced signal degradation 并以簇内归一化修复；MMS-PRM（Lv et al., IJCAI 2026）进一步指出平均分奖励会掩盖单维度失败（compensation），提出非补偿式的最差维度优化。三者与本文的关系是概念迁移而非方法复现——它们面向 VLLM/开放集检测，本文是封闭集 RT-DETR 的疵点检测；其修复手段在本架构下的对应物（查询级组内归一化、训练侧簇内归一化、最差维度定向优化）已按"证据对账→预承诺门"流程排入后续实验。

本文的证据链与该机理互为表里：① 4.14 的 UW 四种子收敛到 cls:reg≈1:3.2——按损失值自动平衡实质是按方差平衡，回归臂因损失量大而被自动升权；② 4.16 的分布排序把分数变换为类内分位数后域内正确率不升反降（0.9248 vs 0.9383）——分数空间已无可分性剩余，与 5.5 的 AUC 不变命题一致；③ 本文全程以 crit_R/crit_P 三维（hole/broken_warp/broken_spandex）而非平均分做干预判定，其设计动机与 MMS-PRM 的最差维度思想同向：平均分掩盖 broken_spandex 等维度的局部失败，而预承诺门的"crit_R 不降"条款正是非补偿式判据。

---

## 6 结论与展望

**结论**：① 七维可微奖励确能驱动关键类召回倾斜，但呈召回—精度零和；强基线上三种改权重/改度量干预全部复现该零和（系统性结构特征），可微奖励接口（Layer-1）seed 42 上的表观同向经三种子复测确认为噪声（Δmean −0.0096 ± 0.0093），四种机制互异的干预接口全部落入同一零和；同一多维思想迁移到推理侧报警评分（域内离线回放）同样不敌 max-conf（消融最佳变体 0.9269、主变体 0.9154 vs 0.9383），表明约束是特征级信息增量而非接口形式；置信度校准（4.13）可把运行区域 ECE 从 0.11 修到 0.04 但不改变 TP/FP 可分性（AUC 严格不变）；贝叶斯路线的实用替代——深度集成分歧闸门（4.15）——在固定召回下同样不优于分数阈值（匹配召回 FPR 不降，且分歧度与 TP 正相关）；自适应损失加权（UW，4.14）四种子收敛到一致的权重比值（~3.2，实证默认 cls:reg=1:7 增益对官方指标欠优）但严格适用预登记零和条款后仅 1/4 种子真转正、种子平均 mean 0.6176 低于转正门 0.6208（第 4 种子把三种子的 UNSTABLE 暂定判定解决为负结果）——**静态奖惩、推理侧重加权、单调重校准、集成歧义闸门、自适应加权五种机制互异的干预全部无法稳定突破零和，瓶颈在判别结构而非分数/权重形式**；② 天花板有两层——标签噪声（类合并可修复）与细缺陷低置信度漏检（故障在特征级判别力而非损失/奖励）；③ 训练基础侧，官方口径 mean 0.572 → 0.6349 由 rect（+0.036）与 mosaic（+0.027）两个近似可加的便宜因子达成，高分辨率无净增益；④ "混淆矩阵→阈值标定→系统消融→因子分离"诊断方法论可迁移。

**展望**：① 特征层干预（SPDConv space-to-depth / 内可变形注意力，保留细缺陷特征）；② 标定层深化（objectness 头 Stage-2 联合微调或与特征层干预结合，同时解决 B 口径与专家策略的"主预测不可识别"）；③ 关键类重标注/主动学习突破标签天花板；④ 奖励面重新设计（Layer-1 多种子复测已确认零和复现、路线优先级下降；当前奖励面在策略采样噪声下过于平坦，任何 REINFORCE 类方法都无从着力）；⑤ ONNX 边缘部署 + 分级告警系统化评估。

---

## 参考文献

1. Zhao Y. et al. *DETRs Beat YOLOs on Real-time Object Detection*. CVPR 2024. arXiv:2304.08069.
2. Carion N. et al. *End-to-End Object Detection with Transformers*. ECCV 2020. arXiv:2005.12872.
3. Shao Z. et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.
4. Schulman J. et al. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
5. Ouyang L. et al. *Training Language Models to Follow Instructions with Human Feedback*. NeurIPS 2022. arXiv:2203.02155.
6. Ultralytics YOLOv8 / RT-DETR 实现. https://github.com/ultralytics/ultralytics.
7. 阿里天池"布匹疵点智能识别"竞赛数据集（2019）.
8. Silvestre-Blanes J. et al. AITEX 织物缺陷数据集.
9. Zhang C. et al. ZJU-Leaper 织物缺陷数据集.
10. Zhao P., Jia S. *SPFFNet: Strip Perception and Feature Fusion Spatial Pyramid Pooling for Fabric Defect Detection*. arXiv:2502.01445.（Tianchi 20 类，mAP50 65.8%，口径核验为 8 类子集 mAP@0.5）
11. Wang S., Kong H., Li B., Zheng F. *Fab-ME: A Vision State-Space and Attention-Enhanced Framework for Fabric Defect Detection*. arXiv:2412.03200 / Springer LNCS (ICIC 2025) 15854:103–114, DOI 10.1007/978-981-96-9901-8_9.
12. Liang G., Han S. *Denim fabric defect detection algorithm based on improved RT-DETR*. 浙江大学学报（工学版）2025, 59(6):1169–1178, DOI 10.3785/j.issn.1008-973X.2025.06.008.
13. Park Y.-J., Sobolewski C., Azizan N. *Uncertainty Quantification in Detection Transformers: Object-Level Calibration and Image-Level Reliability*. arXiv:2412.01782.
14. Wang J., Xu C., Yang W., Yu L. *A Normalized Gaussian Wasserstein Distance for Tiny Object Detection*. AAAI 2022. arXiv:2110.13389.（NWD 细目标度量）
15. Guo C., Pleiss G., Sun Y., Weinberger K. Q. *On Calibration of Modern Neural Networks*. ICML 2017. arXiv:1706.04599.（温度缩放与 ECE）
16. Kendall A., Gal Y., Cipolla R. *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*. CVPR 2018. arXiv:1705.07115.（UW 同方差不确定性加权）
17. Chen Z., Badrinarayanan V., Lee C.-Y., Rabinovich A. *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*. ICML 2018. arXiv:1711.02257.（多任务损失自适应加权对照路线）
18. Lakshminarayanan B., Pritzel A., Blundell C. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. NIPS 2017. arXiv:1612.01474.（深度集成认知不确定性）
19. Gal Y., Ghahramani Z. *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML 2016. arXiv:1506.02142.（MC Dropout；本架构 decoder 无 dropout 不可用）
20. Sun H., Wang T., Tang C., Yuan L., Lv J. *Dr. Seg: Revisiting GRPO Training for Visual Large Language Models through Perception-Oriented Design*. arXiv:2603.00152, 2026.（arXiv preprint；分布排序奖励，5.6 机理支撑）
21. Xu S., Li Y., Yang R., et al. *Mixed-R1: Unified Reward Perspective For Reasoning Capability in Multimodal Large Language Models*. arXiv:2505.24164, 2025.（arXiv preprint；多源奖励设计先例）
22. Pan C., Ren L., He W., Tu Z. *DINO-R1: Incentivizing Reasoning Capability in Vision Foundation Models*. arXiv:2505.24025, 2025.（arXiv preprint；GRQO 查询级组内归一化 + objectness KL 正则，后续实验 B3 概念来源）
23. Lv H., Zhang H., Li Q., Li L., Gao C. *Improving Multimodal Reasoning via Worst Dimension Optimization*. IJCAI 2026, pp. 166.（MMS-PRM；非补偿式最差维度优化，5.6 机理支撑）
24. Yoon H. S., Yoon E., Hong J. W., Eom S., Koo G., Hasegawa-Johnson M., Dai Q., Luo C., Yoo C. D. *PDCR: Perception-Decomposed Confidence Reward for Vision-Language Reasoning*. CVPR 2026, pp. 18881–18891.（簇内归一化，5.6 机理支撑）

> ✅ 引用核验状态（2026-10-21 二次更新，IRON RULE 校验门通过）：第 1–19 条如前（全表无伪造文献）；第 20–24 条为本轮阶段 0 新增，全部经 arXiv 官方页面 / CVF Open Access / IJCAI 官方 proceedings 逐一核验（标题、作者、年份、ID/页码一致）：MMS-PRM 为 IJCAI 2026 **正式发表**（proceedings 页码 166），PDCR 为 CVPR 2026 **正式发表**（CVF Open Access，pp. 18881–18891，dblp: conf/cvpr/YoonYHEKHDLY26 双源确认），Dr. Seg / Mixed-R1 / DINO-R1 为 **arXiv 预印本**（已在条目中标注，不冒充正式发表）。"方差霸权/高方差主导"为本文概念性表述，未署名引用（见 5.6）。核实记录存档 `docs/references_audit.md`。

---

*稿件状态：**定稿 v2.0（2026-10-21）**。全文口径复查通过：表 1–14 编号连续无缺号；表 12 四种子、4.13 校准审计、4.15 集成审计、4.9 objectness 回放数字与存档逐一核对一致（容差 5e-5）；5.5"四层负结果（五种机制互异的干预）"与结论①、摘要（中/英）、贡献 6 完全同步；prose 层修正两处（4.3"原图 2.4:1"→"原始图像 2.4:1 的宽幅"消歧；5.5 开篇补"四层/五种干预"口径）。数据截至阶段 1 与 1.3 收官 + Layer-1 三种子复测（4.8，CONFIRMED_NEGATIVE）+ 推理侧七维报警评分离线回放（4.12，阴性结果 + 可解释决策接口）+ 置信度校准审计（4.13）+ 自适应损失加权 UW 四种子（4.14，NEGATIVE）+ 深度集成不确定性审计（4.15，分歧闸门阴性）+ objectness 头同协议回放（4.9 精确化）。阶段 4 收官：P0 参考文献校验门通过（19/19 核验）；P1 全文口径复查完成（8 处修正）；v1.4 追加 4.6 表 6 权威重写；v1.5 追加 4.13 校准审计——运行区域 ECE 0.106/0.112 严重失准且为曲线形状问题（单参数 TS 恶化至 0.177/0.172，isotonic 交叉拟合修复至 0.037/0.046），任何单调重校准不改变 TP/FP 可分性（AUC 0.763/0.823 严格不变）；条件校准（族/尺寸）无增益。v1.6 追加 4.14 UW；v1.7 定稿 4.14——**四种子**收敛到一致的权重比值（3.20/3.15/3.19/3.21，spread 1.02×，实证默认 cls:reg=1:7 增益对官方指标欠优，是最稳健输出），但严格适用预登记零和条款后仅 1/4 种子真转正（seed 2024 Δmean +0.0126 最高但增益全在召回侧、crit_P −0.044 触发零和条款；seed 42/7 同型），第 4 种子（seed 7：mean 0.6070，Δ=−0.0087）把三种子的 UNSTABLE 暂定判定按预登记决策树解决为 **NEGATIVE**——种子平均 mean 0.6176 ≤ 转正门 0.6208，种子间跨度 0.0214 = 门宽 4.3 倍，效应量小于种子离散度。v1.8 追加 4.15 集成不确定性——两套同起点集成（ctl3 K=3 / uw4 K=4）均显示：集成召回提升 +0.004~0.007（经典集成增益），但分歧闸门在固定召回下 FPR 不优于分数阈值（预登记 −20% 标准未达到），且分歧度与 TP 正相关（TP 簇 std 为 FP 簇的 40–65 倍，误检簇多为单成员突发），同起点成员对 90%+ GT 一致触发——深度集成不构成认知不确定性来源，贝叶斯路线的实用替代阴性。v1.9 精确化 4.9 objectness 结论——同协议回放（`scripts/eval_objectness_replay.py`，同模型同查询双分数）：域内图像级判定 max-class-conf 全面占优（AUC 0.9685 vs 0.8875、正确率 91.64% vs 80.98%、域内 FPR 7.75% vs 34.5%），objectness 唯一优势是跨域 AITEX FPR（31.2% vs 43.3%，−28% 相对）——该头的价值从"更好的前景/背景打分器"修正为"跨域门限的鲁棒性"，摘要与 4.9 结论已同步。连同 4.8/4.12/4.13/4.14/4.15 构成完整负结果链（静态奖惩/推理侧重加权/单调重校准/集成歧义闸门/自适应加权全部无法稳定突破零和）。**UW 机制修复记录（两处实现 bug，均已定位修复并经全量数据诊断验证）**：① 权重更新 `w=exp(+lv)` 符号错误（Kendall 公式要求 `exp(−lv)`，正指数构成正反馈 runaway）导致双臂权重饱和于截断上限（7.39=e²，零重平衡、纯损失尺度通胀）；② 更新项误用归一化后的权重（而非 Kendall 梯度要求的未归一化 wu=e^{−lv}），偏差把 lv 推过自然稳态，双 lv 钉住截断 +2.0 后 wu 相等、归一化产出精确 1:1，臂静默退化为均匀对照（全量数据诊断 step 900 实证）。最终实现：更新项用未归一化 wu（稳态 wu=1/L）+ 尺度保持归一化（总损失恒等于默认，唯一变量=cls:reg 比值）+ EMA 平滑 + 零损失臂冻结 + lv 截断 [−3,+3]；全量诊断 900 步 w 从 1:1 单调收敛到 (1.91, 0.73)。**seed 7 链执行记录**：首跑于 00:30 被会话压缩后的后台清理杀死（epoch 2/5，无 traceback，GPU 空闲确认），01:01 整链重启干净完成（5ep + 官方口径评测，exit 0），部分产物存档 `runs/ablation_adaptive/uw_seed7_partial_0100`。*

---

*v2.1 增量（2026-10-21 下午，阶段 0+1 收官）*：**阶段 0 来源核实**——Dr. Seg（arXiv:2603.00152）/Mixed-R1（arXiv:2505.24164）/DINO-R1（arXiv:2505.24025）/MMS-PRM（IJCAI 2026 proceedings 166）/PDCR（CVPR 2026，pp. 18881–18891，CVF Open Access + dblp 双源）全部核实，"方差霸权"未找到出处按铁律作概念性表述（5.6，不署名）；核实记录 `docs/references_audit.md`。**阶段 1 五项 CPU 回放**（与训练链并行，未碰 GPU）：新增 4.16——分布排序评分（B2）交叉拟合 0.9248 判负（门 0.9373，关键类召回 0.9703 全场最高）；固定预算拦截（C3）为正但归因修正（优势来自严重度先验，小预算下 sev×conf 73.3% 优于七维 51.5%）；成本模型（C7）判负（论文三档 R_global 全成本比更贵，但三档+max-conf 省 28–45%）；可调性曲线（C8）描述性通过；域偏移（1.5）AITEX max-conf 4.71×、域可分 AUC 0.856。5.4 部署结论同步（严重度加权排序+三档挂强分数）；5.6 新增概念性机理节（引 20/23/24 三条已核实正式/预印文献，概念迁移边界已声明）；参考文献扩至 24 条（20–22 标注 arXiv preprint，23/24 正式发表）；数据修正一处（REPORT.md 关键类图 580→101，4.16 已注）。摘要"五种干预"口径不变（B2 归入推理侧重加权层的第三种形式）。脚本 `scripts/stage1_replay_analysis.py`、`scripts/stage1_rank_scoring.py`；数据 `runs/alarm_replay/stage1/`。
