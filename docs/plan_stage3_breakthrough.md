# 阶段 3 计划书：便宜因子路线的数据层延伸（Copy-Paste）

> 状态：**设计已定稿，判定门在实验启动前登记（2026-10-21 上午）**。
> 前置：论文定稿 v2.0（2026-10-21）。本计划书遵循项目铁律：
> 先与既有证据对账 → 设计 → **预承诺判定门** → 种子复测前不下单臂结论。

## 1. 背景：为什么是数据层

论文 5.5 边界声明（定稿 v2.0）：阶段 1–2 的五类干预（静态奖惩 / 推理侧重加权 /
单调重校准 / 集成歧义闸门 / 自适应损失加权）全部无法稳定突破召回—精度零和，
瓶颈定位在特征级判别结构。**突破的必要条件：引入与既有分数正交的信息源。**

既有证据中唯一被证明有效的路线正是"便宜因子"（阶段 1）：rect letterbox
（+0.036）与 mosaic（+0.027）近似可加，均属**数据/几何层**干预——它们不重加权
分数，而是改变模型所见的数据分布与几何。将该路线向未测的最后一个数据层因子
延伸，是当前证据支持度最高的方向。

## 2. 证据对账：候选方向与排除理由

| 候选方向 | 与"正交信息源"的关系 | 排除/入选理由 |
|---|---|---|
| **Copy-Paste 类感知粘贴** | 直接新增训练实例（稀有类 GT 被重复采样到新背景），是真正的数据分布干预 | **入选（主臂）**。Simple Copy-Paste（Ghiasi et al., arXiv:2012.07177）在实例分割/检测上被广泛验证；本项目长尾明确（knot 117、stain 145、three_silk 145、weft_shrink 176 均 <2.1%），关键类 hole/broken_spandex 为中频但漏检主体（4.6 表 6） |
| multi-scale（scale=0.5） | 几何层 | **排除：已含在 mosaic1280 臂配方中**（args.yaml scale=0.5，2026-10-21 核实），非新因子 |
| 图像级重复采样（data_oversampled.txt） | 零新信息，仅改类先验 | 排除：10166 行 vs 8619 唯一图（纯重复），与损失重加权同族；优先级低于 Copy-Paste |
| 特征层干预（SPDConv / 可变形注意力） | 特征层 | 排除：计算贵、需改模型结构（铁律不换模型的精神是先穷尽便宜因子）；且 5.2 定位的故障层是 query 触发，特征保留未必对齐 |
| 族级联（3 族粗分类 → 族内细检测） | 结构先验 | 留作阶段 3b 备选：粗分类器 0.9119 可用但推理 2× 成本、A↔C 互混（70% 错误）可能传导；若主臂阴性再启动 |
| 半监督自训练（天池 test 无标注） | 无标签数据 | 排除：周期长、伪标签噪声与 4.5 标签噪声教训叠加；列为阶段 4 |
| 关键类重标注 | 修复第一层天花板 | 需人工，非本阶段 GPU 实验范畴 |

## 3. 可证伪假设

**H1（主假设）**：在 mosaic1280 配方上增加类感知 Copy-Paste（长尾/关键类实例
跨图粘贴），官方口径 mean 超过对照 +0.005。

**H2（机制假设）**：若 H1 成立，增益主要来自稀有/小目标类的召回提升
（knot/stain/three_silk 等 <2% 类与点状族小目标），而非关键类。

**H3（零和排查）**：若 mean 升但 crit_R 降（或反之），记录零和签名，
与阶段 1–2 的负结果链对账——Copy-Paste 作为数据层干预若也落入零和，
则"零和"从分数/权重形式扩展到数据形式，是更强的机理证据。

## 4. 实现方案

### 4.1 ultralytics 原生 copy_paste 不可用（已核实）

`ultralytics/data/augment.py` CopyPaste.__call__ 首行
`if len(labels["instances"].segments) == 0 or self.p == 0: return labels`——
检测数据集无 segment，原生 `copy_paste=0.5` **静默无效**。必须自实现。

### 4.2 自实现检测版 Copy-Paste

- **挂载**：monkey-patch `ultralytics.data.augment.CopyPaste.__call__`
  （在启动器 import 阶段 patch，早于 dataloader 构建），检测 labels 分支
  走自定义逻辑；其余训练流程零改动。
- **触发**：概率 p=0.5（超参 copy_paste=0.5，copy_paste_mode="flip"），
  每样本独立。
- **donor 采样**：从训练集随机取另一张图（走同一 letterbox 前处理），
  按类频次倒数加权采样 donor 中的实例（稀有类被选中概率高），
  每次粘贴 1–3 个实例。
- **几何**：实例裁剪（像素级 bbox 区域）后尺度抖动 0.5–1.5×、
  随机水平翻转，粘贴到本图随机空位。
- **标签**：粘贴实例的 bbox/ cls 追加到本图 labels；与既有 GT 的
  IoU>0.1 区域跳过（不制造遮挡歧义）。
- **坐标系统**：在 letterbox rect 后的 numpy 图上操作（HWC uint8），
  bbox 用 ultralytics Instances 的归一化 xywh 格式同步更新。

### 4.3 训练协议（与 mosaic1280 臂逐项一致，唯一变量 = Copy-Paste）

| 项 | 值 |
|---|---|
| 起点权重 | rtdetr-l.pt（COCO 预训练，从头训） |
| epochs | 60（与对照臂相同） |
| imgsz / rect | 1280 / True |
| mosaic / close_mosaic | 1.0 / 10 |
| scale | 0.5 |
| fliplr / erasing / translate | 0.5 / 0.4 / 0.1 |
| batch / lr0 / optimizer | 4 / 5e-4 / auto（同对照） |
| amp | True（同对照；**全程 BN 哨兵监控**——fp16-AMP BN 污染坑仅在 1536 触发过，1280 两臂均健康，但仍按既有流程每 5ep 检查 running_var 归零层数） |
| seed | 42 |
| 数据 | data/tianchi/data.yaml（18 类，train 8619 图 / 8570 GT） |

对照臂 = `runs/rtdetr/tianchi18_mosaic1280`（官方口径 mean 0.6349，
mAP50 0.592，2026-10-19 收官），**不重跑**（epoch 数相同、协议逐项一致）。

## 5. 预承诺判定门（实验启动前登记，事后不移动）

以官方口径 mean(mAP@0.1/0.3/0.5) 为主指标，crit_R/crit_P 为
hole/broken_warp/broken_spandex 三类平均 R@maxF1 / P@maxF1：

| 条件 | 判定 |
|---|---|
| mean > 0.6349 + 0.005 = **0.6399** 且 crit_R 不降 | **转正**：Copy-Paste 记为正结果，进论文 4.16 |
| mean ≤ 0.6399 但 crit_R 升 >0.02 且 crit_P 降 >0.02 | **零和复现（数据层）**：更强的机理证据，进 5.5 负结果链 |
| mean ≤ 0.6399 且 crit_R 不升 | **阴性**：与负结果链对账，记录 |
| 冒烟期（前 3ep）出现 loss NaN / BN running_var 归零 / mAP50 < 0.50 | **实现 bug**：终止修 bug 后重跑，不计结果 |

冒烟期的附加体检（不构成判定，只用于区分"无效"与"坏了"）：
- 粘贴可视化：抽 4 张 batch 图存盘人工核对（实例位置/类别/无遮挡越界）
- 训练集 GT 计数：Copy-Paste 生效时长尾类实例数应显著上升

## 6. 执行计划

1. **实现 + 冒烟**（3ep，~45 min）：验证挂载、可视化、BN 健康、loss 曲线
2. **正式臂**（60ep，~17 h，后台）：官方口径评测（eval_multIou）
3. **判定**：按 §5 门判定；转正则补 seed 复测（项目铁律：单臂结论须种子复测，
   本次同样适用——但若冒烟期即阴性且实现验证完好，可缩短为 2 种子）
4. **回填**：论文 4.16 或 5.5 负结果链 + 状态行

### 6.1 冒烟期实测记录（2026-10-21，两个实现 bug 均已定位修复）

冒烟（3ep + 5% 数据）暴露并修复两个实现 bug，均已复测通过后才启动正式臂：

1. **dtype 提升 bug（RT-DETR CDN 崩溃）**：粘贴追加框用 `np.array(..., dtype=float)`
   （float64），`np.vstack` 把 float32 的 `instances.bboxes` 提升为 float64 →
   `Format` 的 `torch.from_numpy` 出 Double → collate 后 `batch["bboxes"]` 为 Double →
   RT-DETR `get_cdn_group` 的 `padding_bbox`(torch.zeros 默认 Float) ← `dn_bbox`(Double)
   dtype 不匹配崩溃（`expected scalar type Double but found Float`，ops.py L288）。
   修复：追加框/cls 的 dtype 从 `inst.bboxes.dtype` 推断（float32 保持）。
   教训：真实管道 `instances.bboxes` 为 **float32**（label 解析路径决定），
   孤立构造 Instances 的 numpy 默认 float64 会误导——dtype 假设必须端到端验证。
2. **`np.random.choice` 概率项非零数不足**：`replace=False` 时 p 的非零项必须 ≥ size，
   供体有效类数 < k 时抛 `ValueError: Fewer non-zero entries in p than size`。
   修复：`k = min(random.randint(1,3), nz)`，nz = 有效（非零概率）实例数。

冒烟通过标准（全部满足）：3ep 训练 + 验证跑通无 traceback；6 张粘贴可视化
（`runs/stage3_cp_debug/`）人工核对位置/类别合理；BN 哨兵 3/3 epoch 0 归零；
loss 曲线正常（cls_loss ~50 起步下降）。

**类频次加权机制实测**（80 样本插桩统计，`runs/stage3_cp_debug/pasted_class_hist.json`）：
粘贴框的类分布 vs 训练集基础类频次——flower_board（最稀有 1.37%）lift 3.66×、
hundred_feet（1.69%）2.96×、loose_warp（4.15%）2.41×、three_silk（11.04%）1.81×、
knot（最常见 20.76%）仅 1.32×。**1/类频次加权按设计放大了长尾类实例占比**，
"数据层引入与分数形式正交的信息源"这一干预前提成立。

## 7. 风险控制

| 风险 | 缓解 |
|---|---|
| 粘贴伪影引入新 FP（贴图表皮不自然） | 类频次加权 donor + 尺度抖动已降低；若 FPR 恶化明显，记录为负结果机制分析而非掩盖 |
| monkey-patch 与 ultralytics 版本耦合 | patch 仅在 import 期、只改 `CopyPaste.__call__`；版本号冻结 8.4.91 |
| 长跑跨会话被杀（seed 7 教训） | 后台任务 + 每 epoch results.csv 落盘；被杀后从 partial 重启（脚本支持 resume） |
| GPU 并发事故 | 严格顺序执行：本链运行期间无其他 GPU 进程 |

## 8. 参考

- Ghiasi et al. *Simple Copy-Paste is a Strong Data Augmentation Method for
  Instance Segmentation*. arXiv:2012.07177.（待补核验后入参考文献）
