# M1 AP 代理损失 —— 预注册

- **臂**：`m1_aploss_s42`
- **协议**：`runs/rtdetr/tianchi18_mosaic1280/weights/best.pt` 起点 | 5 epoch | batch 4 | imgsz 1280 方形 | **lr0 5e-5**（handoff M1：R4 的一半）| amp OFF | mosaic 0 | workers 4 | seed 42
- **唯一变量**：在检测损失旁追加 Smooth-AP 成对排序项 `loss_ap = -beta * mean_b AP_b`（T=0.5，β=0.1）。GIoU / L1 / VFL / 去噪全部保持上游数值，**只加不换**。
- **动机（来自本轮实测，不是历史猜测）**：P0.4 显示 1632 个 FP 里 976 个是缺陷图上的残余乱框、416 个是错类框 —— 85% 的误报是"多出来的框/排错序"。AP 代理直接优化查询分数序，是本轮唯一对得上问题签名的手段。
- **对照**：`C* = 0.6077`（R3″，同起点同 5 epoch 同 lr0=1e-4）。注意它是**跨代码**对照（`train_adaptive_loss.py` vs 本臂的 `train_rtdetr_aploss.py`）。
- **判据（开跑前写死）**：
  1. 转正候选 = 官方 mean > **0.6127**（C\*+0.005）**且** crit_recall ≥ 0.6561（不降）；
  2. |Δ| ≤ 0.0093 → 噪声，**必须补同代码 beta=0 对照 + seed 123/2024** 才可判定；
  3. 若 mean > 0.6127 但 < 0.6208（B\*）：记为"部分收复微调损失"，不算转正，但进 confirm 名单。
- **失败转向**：
  - 签名 = mean↓ 且 crit_recall↑ → 与 R2/B/NWD 同型（PR 曲线滑动），记入负结果链，M1 关闭；
  - 签名 = 双降 → 按 handoff §11 F2，β 或 lr 减半重试一次；
  - mean 无变化（|Δ|≤0.0093）→ 补 seed 123/2024，仍噪声则判阴性。
- **种子计划**：42（screen）→ 判定需要时补 123 / 2024。
- **预计时长**：1.5–2 h。
- **sanity**：跑完必须过 `eval_multIou.py`，mAP50 与该 run `results.csv` 末轮偏差 ≤ 0.004（F6）。
