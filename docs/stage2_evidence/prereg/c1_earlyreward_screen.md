# C1 早期表征奖励（screen）—— 预注册

- **臂**：`c1_earlyreward_s42`
- **协议**：与 B\* **逐项同配方**（rtdetr-l.pt COCO 初始化 / rect / bf16 补丁 / MuSGD(auto) / lr0 5e-4 / lrf 0.01 / mosaic 1.0 / 60 epoch / patience 20 / batch 4 / imgsz 1280 / workers 4 / seed 42），**唯一差异**：前 10 个 epoch 以 β=0.5 注入七维可微奖励，之后 β 精确置 0。
- **对照**：`B* = 0.6208`（`runs/rtdetr/tianchi18_mosaic1280`，同 seed、同配方、全程 β=0）。**screen 阶段零额外对照成本。**
- **假说**：收敛模型上奖惩只能沿 PR 曲线滑动（已有 4 次独立确认）；若在曲线定型前施加，可能把曲线本身外移。这是"接口没错、时机错了"的直接检验。
- **判据（开跑前写死，事后不移动）**：
  1. **screen 阳性** = 官方 mean ≥ **0.6258**（B\*+0.005）→ 进入 confirm（seed 123 / 2024 各跑臂 + β=0 对照）；
  2. screen < 0.6258 → **记阴性，不进 confirm**（省 60 小时）；
  3. |Δ| ≤ 0.0093 也按阴性处理（不得因"接近"而进 confirm）。
- **失败转向**：
  - 阴性 → 阶段 C 关闭（C2 变体只在 screen 阳性但 confirm 阴性时才做）；
  - 奖励曲线持续上涨不收敛 → 按 handoff §11 F7 降 β。
- **额外观测（不作判据）**：前 10 epoch 的 reward 均值走势、BN 是否被毒化（哨兵）、`runs/rtdetr/tianchi18_earlyreward/reward_history.csv`。
- **种子计划**：42（screen）→ 阳性才补 123 / 2024。
- **预计时长**：10.5–11 h。
- **sanity**：官方评测 mAP50 与 results.csv 峰值偏差 ≤ 0.004（F6）。
