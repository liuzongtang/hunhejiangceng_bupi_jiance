# D1 低 IoU 定位奖励 —— 预注册

- **臂**：`d1_locreward_s42`
- **协议**：`runs/rtdetr/tianchi18_mosaic1280/weights/best.pt` 起点 | 5 epoch | batch 4 | imgsz 1280 方形 | lr0 1e-4 | amp OFF | mosaic 0 | workers 4 | seed 42
- **唯一变量**：追加定位奖励 `loss_loc = -beta * mean_i soft_iou_i`，其中 `soft_iou = 0.5*(min(1, IoU/0.1) + min(1, IoU/0.3))`，β=0.1。**只加不换**：不动 GIoU/L1 默认增益，不动分类项。
- **与历史 R3/NWD 的区别（必须写明）**：NWD 是**度量替换**（整体换掉损失形式，已零和 −0.015）；D1 是**只加一个奖励项**，且 IoU 口径对齐比赛指标（0.1/0.3，而非 0.5）。
- **动机**：P0.2 实测 mean(AP@0.1 − AP@0.5) = **0.0725**（历史 0.074，复现），且落差集中在薄框/线性族 → 定位侧理论上仍有 0.07 的头上界。
- **对照**：`C* = 0.6077`（同起点 5 epoch 微调）。跨代码对照，同上。
- **判据（开跑前写死）**：
  1. 转正候选 = 官方 mean > **0.6127**（C\*+0.005）且 crit_recall ≥ 0.6561；
  2. **增益必须主要来自 AP@0.3/@0.5 的薄框类**（loose_warp / size_stain / broken_warp / hanging_warp）；若增益只出现在 AP@0.1，说明只是框更松，不算定位改善；
  3. |Δ| ≤ 0.0093 → 噪声，补 seed 123/2024。
- **失败转向**：
  - 签名同 R3（mean↓ 且 crit_recall↑）→ 定位侧在该特征下也不可外移，记阴性；
  - 双降 → handoff §11 F2，β 或 lr 减半重试一次。
- **种子计划**：42 → 需要时 123 / 2024。
- **预计时长**：1.5–2 h。
- **sanity**：官方评测 mAP50 与 results.csv 末轮偏差 ≤ 0.004（F6）。
