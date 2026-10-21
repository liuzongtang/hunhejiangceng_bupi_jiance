# 阶段 1.3 协议：七维奖惩在强基线上的重测（创新点诚实救赎）

> 制定日期：2026-10-16。触发条件：阶段 1.1（1536 bf16 重训）官方口径 mean mAP 显著超过
> 基线 0.572（目标 ≥ 0.60），即"强基线"成立。

## 1. 背景与动机

论文 4.6 节记录的**七负消融**全部在**弱基线**上完成：RT-DETR-L @1280、60 epoch、
mAP50 0.541、crit_recall（均匀微调对照）0.606。当时结论：七种训练信号干预
（Layer-1 可微奖励 ×2 臂、Layer-3 GRPO、Direction B 分类加权 ×4 臂、R2 box 加权、
R3 NWD 损失）均不敌均匀微调，接口错位假说（聚合 box 指标 vs query 触发+定位自评
故障层）被提出但**未在强基线上检验**。

用户硬约束（2026-10-15）：七维奖惩创新点保留，但必须"在榨干的基础模型上重新检验"。
若强基线上某些干预转正，则 4.6 节结论需要按新基线下调重写；若仍全负，则接口错位
假说获得强基线证据支持，论文叙事反而更完整。**两种结果都有价值，必须测。**

## 2. 强基线定义（对照组）

- 权重：**mosaic1280 best.pt**（2026-10-18 01:47 由 `scripts/stage13_orchestrator.sh` 按同口径官方 mean 自动选定：mosaic1280 0.6349 > v4 0.6077；因子分离证明 1280 全面占优 1536，详见论文 4.12）。原定 v2 best.pt 作废（v2/v3 微调路线已被证伪中止）。
- 对照运行：同一配置、同 seed=42、同数据顺序的均匀微调（R2 臂的 `--preset none`）。所有干预臂与对照的差异**仅限**干预项本身。
- 重测口径与历史七负实验完全一致：1280 方形 letterbox、amp=False、lr0=1e-4、5 epoch（因子分离实验证明 1280 不劣于 1536 且每轮快 25%，故回落到 1280 口径——这也是与历史结果可比的前提）
- 主指标：天池官方口径 mean(mAP@0.1, @0.3, @0.5)（`scripts/eval_multIou.py`）
- 护栏指标：关键类 recall @ max-F1（broken_warp=9 / broken_spandex=16 组）、逐类阈值标定后的精确率（防"降阈值骗召回"）

## 3. 重测清单（按预期收益排序，逐臂 A/B）

| 序 | 干预臂 | 脚本 | 弱基线历史 | 重测配置 |
|---|---|---|---|---|
| 1 | R2 box 严重度加权（critical 2.0×） | `train_rtdetr_boxweighted.py` | crit_recall 0.598 vs 0.606 | 从 v2 best 微调，5–10 epoch，lr0=1e-4 |
| 2 | Direction B 分类加权 positive-only | `train_rtdetr_costsensitive.py` | 0.591 vs 0.606 | 同上，仅正样本项加权 |
| 3 | R3 NWD 损失几何 | `train_rtdetr_nwd.py` | crit_recall 0.598，mAP50 0.518 | 同上 |
| 4 | Layer-1 可微奖励 β=0.5 + include-misses | `train_rtdetr_reward.py` | 召回 +0.030 但 broken_spandex −0.100 | 同上，10 epoch |
| 5 | Layer-3 GRPO（修 std_reward 地板） | `train_rtdetr_grpo.py` | crit_recall 0.574→0.344 | 仅当 1–4 全负时跑；先修 advantage-eps 地板 |

每一臂输出：官方口径 mean mAP、crit_recall@max-F1、P@conf0.10 标定表。

## 4. 判定规则（预先承诺，防事后解释）

- **转正**：干预臂官方 mean mAP > 对照 + 0.005（噪声带外）且 crit_recall 不降。
- **仍负**：写入 4.6 节"强基线上复测"小节，作为接口错位假说的第二重证据。
- **部分转正**（如召回升但 mAP 降）：如实记录，论文中作为 trade-off 讨论，
  不得选择性呈现。

## 5. 执行顺序与依赖

```
1.1 v2 训练完成 → 官方口径评测 → 判定强基线成立?
   ├─ 是 → 1.2 HP 扫描（mosaic/lr，短程）→ 1.3 本协议逐臂
   └─ 否（<0.58）→ 先查 2446×1000 或 mosaic 第二轮，不强推 1.3
```

每臂 5–10 epoch × ~20 分钟 ≈ 2–3.5 小时，五臂合计约 12–17 小时（可夜间排队）。

## 6. 产物归档

- 结果 JSON：`runs/ablation_strong/<arm>_official.json`
- 汇总表：追加到 `docs/paper/paper_skeleton.md` 4.6 节（新小节"强基线复测"）
- 记忆：`MEMORY.md` 记录最终判定（转正/仍负/部分）

---

## 7. 执行结果（2026-10-18 收官，实际归档）

**强基线定义落地**：mosaic1280 best.pt（1280+rect+mosaic 从头训 60ep，官方口径 mean 0.6349，mAP50 0.592）——由 stage13_orchestrator 按同口径 mean 自动选定（0.6349 > v4 0.6077）。

**五臂结果**（对照 = 同配置均匀微调 5ep，mean 0.6158 / crit_recall 0.6260；臂 4 用 epoch 匹配的 10ep 对照 0.6157 / 0.7010）：

| 臂 | 干预 | mean Δ | crit_recall Δ | 判定 |
|---|---|---|---|---|
| 1 | R2 box 严重度加权 2.0× | −0.023 | +0.104 | 负 |
| 2 | Direction B 分类加权 positive-only | −0.029 | +0.184 | 负 |
| 3 | R3 NWD 度量 | −0.015 | +0.124 | 负 |
| 4 | Layer-1 可微奖励 β=0.5+misses | **+0.0007** | **+0.053** | 负（未过线，唯一零和被打破） |
| 5 | GRPO（eps 1e-2→1e-4） | −0.306 | −0.194 | 负（模型崩溃） |

**结论**：七维奖惩中三种"改权重/改度量"接口在强基线上全部复现召回—精度零和（系统性结构特征）；唯一可微奖励接口（Layer-1）同向双臂改善但幅度在噪声带内；GRPO 不可救药（降 eps 地板放大噪声梯度伤害）。创新点最终定位：性能增强手段被否定，故障诊断探针被肯定。

**执行期工程备注**：GRPO 训练脚本只存 final.pt 无 best.pt，导致链上评测级联失败，已用 final.pt 手工补评（`runs/multi_iou/strong_l3_grpo.json`）。
