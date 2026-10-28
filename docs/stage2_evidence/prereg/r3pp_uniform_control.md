# R3″ 均匀微调对照 —— 预注册

- **臂**：R3″ `r3pp_uniform_s42`（微调对照，不是干预臂）
- **协议**：`mosaic1280 best.pt` 起点 | 5 epoch | batch 4 | imgsz 1280 方形 letterbox | lr0 1e-4 | amp OFF | mosaic 0 | workers 0 | seed 42
- **实现**：`scripts/train_adaptive_loss.py --method uw --epochs 5 --name r3pp_uniform_s42`
  - 注意：`--method uw` 在 **step 0 时 w_cls=w_reg=1:1**，与均匀微调等价（脚本 docstring 明示"init 1:1 == uniform control at step 0"）；选它是为了复用已修 bug 的入口，不是引入自适应加权。
  - 若判定需要严格"无自适应"对照，改用 `scripts/train_rtdetr_boxweighted.py --preset none --epochs 5 --name r3pp_uniform_s42`（同样是均匀对照，且不产生可学习权重）。
- **存在的理由**：历史对照值 mean 0.6158 / crit_recall 0.6260 来自已丢失的 `runs/multi_iou/strong_r2_uniform.json`，在本机不可复现。本臂产出**新对照值**，供 A1 / D1 / M1 / S4 等所有 5-epoch 微调臂使用。
- **判据（开跑前写死，事后不移动）**：
  1. 本臂不是"转正/阴性"判定对象，只提供对照基线；
  2. 其官方 mean 记为新 `C*`，写入 `runs/stage2/RESULTS.md` §13 表；
  3. 任何后续微调臂的 Δmean 一律相对 `C*` 计算，**不得**引用 0.6158。
- **单项 sanity（必须过，否则本臂作废）**：
  - `runs/multi_iou/r3pp_uniform_s42_official.json` 的 `n_classes_present` = 18、`n_instances_total` = 953；
  - mAP@0.5 与训练目录 `results.csv` 最后一轮 mAP50 偏差 ≤ 0.004（handoff F6）。
- **失败转向**：sanity 不过 → 按 handoff §11 F6 停，查评测管道，不得用该数字做对照。
- **种子**：42（单种子即可——对照臂不用于显著性推断）。
- **预计时长**：1–2 h（4090D，5 ep @ 1280）。
- **开跑时间**：由 `run_stage2.bat` 队列在 R3′ 完成后自动启动。
