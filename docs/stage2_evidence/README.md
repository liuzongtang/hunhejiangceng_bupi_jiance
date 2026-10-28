# 阶段 2 证据包（2026-10-21 → 2026-10-28）

本目录是 `docs/stage2_execution_result.md` 的数字来源。全部 JSON 由 `scripts/eval_multIou.py`
（官方口径：方形 letterbox / imgsz 1280 / conf 0.001 / max_det 300 / IoU 0.1·0.3·0.5）在
`data/tianchi`（957 val 图 / 953 GT / 18 类）上产出，未经手工编辑。

## 目录

| 路径 | 内容 |
|---|---|
| `RESULTS.md` | 结果总台账：每个臂的 mean / mAP50 / crit_recall / Δ / 判定 |
| `REPORT_R3-P0.md` | 阶段报告：基线重建、探针、S0 判定、事故记录 |
| `AUDIT.md` | 开工前审计：为什么 handoff 的绝对数值不可复现、本机环境与坑 |
| `PLAN_v1.1.md` | 修订后的执行顺序（R3 → R3′ 重训） |
| `prereg/` | 六份预注册，**开跑前写死、事后未移动** |
| `official/` | 九个臂的官方评测 JSON（含 C1 三种子及其配对对照） |
| `probes/` | P0.1 混淆分解、P0.4 误报构成、A2 图像级回放、S0 选择层 |
| `queues/` | 四个队列定义（串行、GPU 闸门、PREREG 校验、断点续训） |

## 复现方式

```bash
# 1) 数据：天池 smartdiagnosisofclothflaw → YOLO（含 20→18 类合并）
python scripts/convert_tianchi_to_yolo.py --train-dirs <partA> <partB> <train2> \
    --test-dirs <testA> <testB> --out data/tianchi --class-scheme tianchi20
python scripts/merge_confusable_classes.py
python scripts/check_data_integrity.py            # 必须过 957/583/374/953/泄漏0

# 2) 基线（60 ep，约 10.3 h @4090D，workers=4）
python scripts/train_mosaic1280.py
python scripts/eval_multIou.py --weights runs/rtdetr/tianchi18_mosaic1280/weights/best.pt \
    --data data/tianchi/data.yaml --out runs/multi_iou/r3prime_official.json

# 3) 探针
python scripts/eval_confusion_decomp.py --weights <best.pt> --tau 0.001
python scripts/oracle_probe.py --weights <best.pt> --mode fp --tau 0.25
python scripts/eval_binary_replay.py --weights <best.pt> --tag r3prime
python scripts/eval_ttc_selection.py --weights <best.pt> --baseline-mean 0.6208 --device 0

# 4) 干预臂（判据见 prereg/）
python scripts/train_rtdetr_aploss.py --epochs 5 --lr0 5e-5 --beta 0.1 --name m1_aploss_s42
python scripts/train_rtdetr_locreward.py --epochs 5 --beta 0.1 --name d1_locreward_s42
python scripts/train_mosaic1280_earlyreward.py --reward-beta 0.5 --reward-epochs 10 --seed 42
```

## 三条读这份证据时必须知道的事

1. **历史数字不可拼接**：handoff 里的 0.6349 / 0.6158 / 93.73% 对应的权重已丢失，无法复现。
   本证据包内的数字全部来自本机自建基线（B\* 0.6208），不得与历史数值混做一条曲线。
2. **运行间方差远大于文档值**：实测 SD 0.016–0.036（文档记 0.0093）。因此 ±0.005 的判定门
   在 n=3 下几乎无功效（见 `RESULTS.md` §S0 与 `REPORT_R3-P0.md` §9）。
3. **C1 的 screen 阳性没有通过 confirm**：三种子配对差 +0.0282 / −0.0306 / +0.0357，
   按预注册判据（两个配对差都须 > +0.005）判 NEGATIVE。判据未移动。
