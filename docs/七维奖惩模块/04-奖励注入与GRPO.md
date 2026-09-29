# 04 · 奖励注入与 GRPO（两条训练路线）

七维奖励本身只是"打分器"，要让它**参与训练**，有两条路线。它们解决的问题不同，代码也独立。

| 路线 | 文件 | 思路 | 解决的问题 |
|---|---|---|---|
| **Layer-1 奖励注入** | `backend/training/reward_rtdetr.py` | 把奖励项直接加进 RT-DETR 的监督损失 | 在有监督下给关键类召回加梯度 |
| **Layer-3 GRPO** | `backend/training/grpo.py` + `grpo_trainer.py` | 查询噪声采样 + 组内相对优势 | 探索奖励引导的强化学习 |

---

## 路线一：Layer-1 奖励注入（`reward_rtdetr.py`）

### 核心做法：给 RT-DETR 的损失"偷塞"一项

Ultralytics 的 `RTDETRDetectionModel.loss()` 会把 criterion 返回 dict 里的**每一个 key 都加和**。利用这一点，我们子类化 `RTDETRDetectionLoss`，在返回 dict 里**多塞一个 `loss_reward` key**，就零侵入地把奖励折进了监督目标（`RewardRTDETRDetectionLoss`，第 37 行）：

```
L_total = L_giou + L_class + L_bbox + β · (-R_scalar)
```

> 不需要改 Ultralytics 的训练循环——只要在训练前用 `attach_reward_criterion(model, ...)` 把 `model.criterion` 和 `model.init_criterion` 都替换掉（第 259 行），它就活过 Ultralytics 可能的惰性重初始化。

### 关键设计一：复用同一个 Hungarian matcher

`_reward_loss`（第 105 行）**不重新做匹配**，而是直接调用监督损失已经用过的 `self.matcher(...)`（第 132 行）。这样奖励看到的 query↔GT 对，与分类/回归损失选中的是**同一批**——奖励的职责就只有一个：补充"关键类召回"这个 CE+GIoU 表达不了的信号。

### 关键设计二：`include_misses`（让奖励"看见"漏检）

RT-DETR 的匈牙利匹配会把每个 GT 都分配到一个 query（N ≥ num_gt），所以**不存在"没匹配上的 GT"**——"漏检"只能表现为"匹配到的 query 在关键类上置信度很低"。

`include_misses=True` 时的处理（第 154-178 行）：

1. 对每个关键类 GT，看它被匹配到的 query 的关键类概率 `p_crit`；
2. 若 `p_crit < miss_threshold`（0.5），判定为"漏检"；
3. 找一个**未被占用的备用 query**（`_best_spare_query`，关键类概率最高的空闲 query，第 215 行），把它指向这个 GT，一并喂给奖励。

效果：漏检的关键类 GT 会通过备用 query 重新进入 D04/D06/D07 的计算，**空闲 query 拿到 recall 梯度**。这就是"奖励注入 + 漏检注入"那组 A/B 实验里的 `include_misses` 开关。

### 构造要点

`DimensionRewardComputer` 必须用 `head_type="sigmoid"`（RT-DETR 逐类 focal/vfl 头，第 154-156 行）。β 默认 0.5。

---

## 路线二：Layer-3 GRPO（`grpo.py` + `grpo_trainer.py`）

### GRPO 是什么

GRPO（Group Relative Policy Optimization）用**同一组 G 条采样的均值和方差**做 baseline，替代 PPO 里要额外维护的价值网络：

```
A_g = (R_g - mean_g) / (std_g + eps)          # 组内相对优势
L_policy = -mean_g ( A_g · log_prob_g )       # REINFORCE
```

`group_advantages`（`grpo.py` 第 57 行）、`grpo_policy_loss`（第 85 行）。`eps` 有个非平凡的下限（如 1e-2），防止组内奖励几乎相等时 std→0 把数值噪声放大成伪梯度。

### 策略：对解码器 content query 加高斯噪声

`LatentGaussianPolicy`（`grpo.py` 第 126 行）维护一个**可学习的探索温度 `log_std`**，采样：

```
z = mean + exp(log_std) · ε,  ε ~ N(0,1)
```

`RTDETRGRPOPolicy`（`grpo_trainer.py` 第 276 行）把这个噪声**注入 RT-DETR 解码器的 content query**——通过 monkey-patch `decoder._get_decoder_input`（第 312 行），在 embed 的最后 `num_queries` 行上加噪声。`sample_group` 跑 G 次带噪声的前向，得到 G 组检测。

### 本项目最细的一个坑：似然比打分 vs 重参数化

这是 GRPO 代码里注释最长的地方（`grpo.py` 模块 docstring + `LatentGaussianPolicy.sample` 第 141 行），值得单独讲清：

- **重参数化视角**：`z = mean + std·ε`，对 `log_std` 求导会得到**常数** `d log π / d log_std = -1`。因为优势是均值零的，`mean(A_g · 常数)` = 0 → **REINFORCE 梯度恒为 0**，温度永远学不动。
- **似然比视角**（正确做法）：把采好的 `z` 固定（`z.detach()`），对 log-density 求导：

  ```
  d log π / d log_std = ε² - 1      d log π / d mean = ε / std
  ```

  这两个 score 与奖励相关，**log_std 和 mean 都能拿到非零梯度**。

所以 `sample` 里写的是 `resid = (z.detach() - mean) / std`，而不是直接拿 `ε`。这是一个极其隐蔽但决定成败的细节。

### 但有个现实约束：RT-DETR 把 content query detach 了

RT-DETR 训练时**会 detach content query**，所以似然比里 `d log π / d mean` 那条梯度**传不回模型权重**。后果：

- `L_grpo = -mean(A_g·log_prob_g)` 实际上**只调探索温度 `log_std`**；
- 想让**模型均值**动起来，得靠第二条梯度路径 `L_reward = -β_reward · mean(R_g)`（`GRPOTrainer.train_step` 第 220-223 行）——这条走的是可微奖励，需要 `detach_detections=False` 才能让奖励梯度流进解码器。

两条梯度都暴露出来（`beta_kl`、`beta_reward`），是为了能单独消融"探索"和"均值"各自的贡献。

### 总损失

```
L = L_grpo + β_kl · KL(policy || ref) + β_reward · (-mean R_g)
```

- `β_kl`：KL 惩罚，锚定到一个固定噪声尺度的参考策略（`gaussian_kl`，`grpo.py` 第 103 行），防止探索温度塌缩到 0（策略变确定性、组方差消失）或爆炸。
- `β_reward`：可微奖励梯度项，0 表示关掉（纯 GRPO）。

---

## 两条路线怎么选

- **想要"有监督、可复现、直接对标 mAP"** → 用 **Layer-1 注入**。它改动最小、最稳，A/B 结论就是靠它做的。
- **想探索"奖励引导的强化学习、不依赖标注的探索"** → 用 **Layer-3 GRPO**。它是研究性路线，受 RT-DETR detach 限制，实际只调温度，均值还得靠 `β_reward` 的奖励梯度。

> 项目结论：Layer-1 的 `include_misses` 把召回 +0.030 但 mAP 持平；Layer-3 受 detach 约束，纯 GRPO 只能调温度。两者都没能突破"标签噪声天花板"——最终定位到数据/标注问题（见 [README](README.md)）。
