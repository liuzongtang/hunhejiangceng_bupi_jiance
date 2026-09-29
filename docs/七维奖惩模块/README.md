# 七维奖惩模块 — 技术说明（供队友阅读）

> 本文件夹是「织瑕智鉴」项目中**七维混合奖惩模块**的技术说明，目的是让没有直接参与这块开发的同学也能看懂：**为什么要这么做、每一维奖励怎么算、奖励怎么参与训练、代码在哪**。
>
> 文中所有路径均相对于**项目根目录**（即 `维度模型(1)/`）。

## 一分钟读懂核心思路

传统目标检测器（YOLO / RT-DETR）的训练目标只有一个：**最小化平均分类 + 回归误差**。这隐含了一个假设——"所有错误同样严重"。

但纺织质检里不是这样：漏检一个**断经**（整匹布报废）和漏检一个**毛粒**（轻微瑕疵）的工业代价天差地别。

七维奖惩模块做的事，就是把这种"工业代价差异"显式地写进训练目标：

1. **把"我们希望模型学会什么"写成 7 个可微的奖励信号** `R_1 ... R_7`（定位 / 分类 / 校准 / 漏检 / 误检 / 断经断氨纶敏感度 / 星跳死皱敏感度）。
2. **让奖励对模型的 logits 和 box 可微**（用 softmax / sigmoid 替代硬判决），这样奖励梯度能反向传播进模型参数，而不是停留在"打分"层面。
3. **把奖励折进总损失**：

```
L_total = L_detection + β · L_scalar + γ · L_consistency
```

其中 `L_scalar = -R_scalar = -Σ(w_k · R_k)`（最大化奖励），`L_consistency` 惩罚七个维度"偏科"。

一句话总结：**检测损失管"检测对不对"，奖励项管"检测得对不对胃口（工业代价）"，一致性项管"别偏科"。**

## 一个必须知道的核心结论（也是本项目最大发现）

我们最初以为"断经召回低（mAP50 0.446）"是训练不够、奖励不够。做了完整实验后发现：

- 把漏检注入奖励（Layer-1 `include_misses`），**召回 +0.030（0.523→0.553），但精确率同步下降、mAP50 持平**——指标只是"搬移"，不是提升。
- 逐类混淆诊断揭示本质：**断经↔磨痕、纬缩↔粗纬是视觉同源的双向混淆**，裁剪图肉眼几乎分不清，这是**标签噪声天花板**，不是模型可训练差距。
- 把视觉同源类合并后，召回显著恢复：**纬缩+粗纬 0.40→0.88、断经+磨痕 0.52→0.79**。

**所以现在的代码是 18 类**（原 20 类合并了两对易混淆类：`surface_mark→broken_warp`、`coarse_weft→weft_shrink`）。这也是为什么"奖励不能解决一切"——训练端干预有上限，剩下要靠数据合并 / 告警层兜底。

## 文件导航

| 文档 | 讲什么 | 对应代码 |
|---|---|---|
| [01-核心思路](01-核心思路.md) | 为什么要七维、三大设计原则 | — |
| [02-七维定义与可微化](02-七维定义与可微化.md) | D01~D07 每个维度的公式 + 可微化技巧 | `backend/training/dimension_rewards.py` |
| [03-损失组装与训练器](03-损失组装与训练器.md) | L_total 怎么组装、可学习权重、训练循环 | `backend/training/loss_functions.py`、`hybrid_reward_trainer.py` |
| [04-奖励注入与GRPO](04-奖励注入与GRPO.md) | 两条训练路线：Layer-1 注入、Layer-3 GRPO | `backend/training/reward_rtdetr.py`、`grpo.py`、`grpo_trainer.py` |
| [05-代码地图](05-代码地图.md) | 文件→模块对照、关键符号索引、数据流 | 全部 |

## 关键超参数速查

| 符号 | 默认值 | 含义 | 位置 |
|---|---|---|---|
| `lambda_miss` | 1.5 | 漏检惩罚系数（比误检更重） | `DimensionRewardComputer` |
| `lambda_fp` | 1.0 | 误检惩罚系数 | `DimensionRewardComputer` |
| `iou_threshold` | 0.5 | 软漏检/误检的 IoU 分界 | `DimensionRewardComputer` |
| `tau` | 0.1 | 软指示器的温度（越小越接近硬判决） | `DimensionRewardComputer` |
| `β` (beta) | 0.1（单机版）/ 0.5（RT-DETR 注入） | 奖励项权重 | `total_loss` / `RewardRTDETRDetectionLoss` |
| `γ` (gamma) | 0.05 | 一致性项权重 | `total_loss` |
| `α` (alpha) | 0.3 | 标量奖励里的一致性系数 | `scalar_reward_loss` |
| `weight_update_freq` | 5 | 自适应维度权重更新周期（epoch） | `HybridRewardTrainer` |
| `num_groups` (G) | 4 | GRPO 每组采样条数 | `GRPOTrainer` |

## 术语表

- **可微奖励（differentiable reward）**：奖励值对模型参数可求导，能反向传播。用 softmax 概率 / sigmoid 软指示替代 argmax / 硬阈值实现。
- **head_type**：分类头归一化方式。`softmax`=单标签头（简易检测器）；`sigmoid`=RT-DETR 的逐类 focal/vfl 头。
- **Hungarian 匹配**：RT-DETR 用匈牙利算法做一对一的 query↔GT 匹配；奖励必须复用**同一个 matcher**，奖励的才是监督损失已经选中的那批检测。
- **关键类（critical class）**：断经、断氨纶（D06）与星跳、死皱（D07），工业上漏检代价最高的类。
- **include_misses**：把"匹配到但关键类置信度太低"的漏检 GT 用备用 query 重新喂给奖励，让 D04/D06/D07 看见漏检。
- **GRPO**：Group Relative Policy Optimization，用"同组 G 条采样的均值/方差"做 baseline，替代 PPO 的价值网络。
