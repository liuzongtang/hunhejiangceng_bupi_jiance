# 织物瑕疵检测系统

> 纺织物瑕疵检测系统 — 计算机视觉 + 混合奖惩 + 实时监控

## 项目概述

面向纺织生产线的 AI 视觉瑕疵检测系统，支持 10 类缺陷实时检测、7 维度混合奖惩训练、中文 Web 实时监控看板。

### 核心指标

| 指标 | 目标 | 状态 |
|------|------|------|
| 检测准确率 | >= 95% | 模型就绪 |
| 检测速度 | >= 60 m/min | 3 后端可选 |
| 缺陷类型 | >= 10 类 | 10 类 |
| 系统可用性 | 7x24h | WebSocket + 健康检查 |
| 测试覆盖 | 168+ 用例 | 全部通过 |

## 快速启动

```bash
# 安装
pip install -e .

# 启动后端服务
python -m backend.main

# 浏览器访问
http://localhost:8000/simulator    # 中文产线模拟器
http://localhost:8000/dashboard    # 中文监控看板
http://localhost:8000/docs         # API 文档 (Swagger)
```

### 一键启动

```cmd
scripts\start.bat              # Windows: 安装 → 测试 → 训练
scripts\start.bat backend      # 启动 API 服务器
scripts\start.bat e2e          # 产线仿真
scripts\start.bat digit        # 数字识别训练
scripts\start.bat demo         # 混合奖惩演示
scripts\start.bat test         # 仅运行测试
```

```bash
bash scripts/start.sh          # Linux/macOS 同上
```

## 前端页面

| 地址 | 页面 | 语言 | 功能 |
|------|------|------|------|
| `/simulator` | 产线模拟器 | 中文 | Canvas 织物纹理生成、5 种缺陷标注动画、流水线可视化 |
| `/dashboard` | 监控看板 | 中文 | 4 标签、7 种图表、WebSocket 实时告警、时间范围选择 |
| `/docs` | API 文档 | 英文 | Swagger UI、19 端点交互式调试 |

## 系统架构

```
                          ┌──────────────────────┐
                          │   Web 前端             │
                          │   /simulator (模拟器)  │
                          │   /dashboard (看板)    │
                          └──────────┬───────────┘
                                     │ WebSocket
┌──────────┐    ┌──────────┐    ┌────▼────┐    ┌──────────┐
│ 工业相机 │───▶│ 预处理   │───▶│ 推理检测 │───▶│ 图像存储 │
│ (图像)   │    │ (640x640)│    │ (ONNX)  │    │ (MinIO)  │
└──────────┘    └──────────┘    └────┬────┘    └──────────┘
                                     │
                              ┌──────▼──────┐
                              │ 告警服务     │
                              │ (MQTT/WS)   │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │ PostgreSQL  │
                              └─────────────┘
```

## 项目结构

```
维度模型/
├── backend/                   # 后端服务
│   ├── main.py                # FastAPI 入口 (19 端点 + WebSocket)
│   ├── config.py / config.yaml
│   ├── database.py            # SQLAlchemy async 引擎
│   ├── storage.py             # MinIO/S3 图像存储 (本地回退)
│   ├── websocket.py           # WebSocket 实时告警推送
│   ├── models/                # 5 张 SQL 表
│   ├── schemas/               # 20+ Pydantic Schema
│   ├── services/              # 检测 / 模型 / 告警
│   ├── routers/               # detection / model / system / storage
│   ├── training/              # 7 维度混合奖惩训练器
│   ├── inference/             # ONNX / PyTorch / Dummy 推理引擎
│   ├── deploy/                # 导出 / 量化 / 边缘部署
│   ├── dashboard.html         # 中文监控看板
│   └── simulator.html         # 中文产线模拟器
├── mixed_reward/              # 混合奖惩机制 + 数字识别 CNN
├── scripts/                   # 训练 / 演示 / E2E 仿真 / 启动脚本
├── tests/                     # 168+ 单元测试
├── configs/                   # YAML 配置
└── requirements*.txt          # 依赖
```

## API 参考

### 检测

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/v1/detection/report` | 提交检测报告 |
| POST | `/api/v1/detection/infer` | 运行推理检测 |
| GET | `/api/v1/detection/reports` | 查询历史报告 |

### 模型管理

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/v1/model/update` | 注册模型更新 |
| GET | `/api/v1/model/current` | 当前活跃模型 |
| GET | `/api/v1/model/history` | 版本历史 |

### 系统监控

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/system/status` | 设备 + 系统资源 |
| GET | `/api/v1/system/stats` | 聚合统计 (1h/24h/7d/30d) |
| GET | `/api/v1/system/health` | 健康检查 |

### 图像存储

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/v1/storage/upload` | 上传图像 |
| GET | `/api/v1/storage/download/{name}` | 下载图像 |
| GET | `/api/v1/storage/list` | 列出图像 |
| DELETE | `/api/v1/storage/{name}` | 删除图像 |

### 实时推送

| 协议 | 端点 | 说明 |
|------|------|------|
| WebSocket | `/ws/alerts` | 实时告警推送 |

## 缺陷分类

| 代码 | 名称 | 英文 | 严重度 |
|------|------|------|--------|
| BY-01 | 断纱 | Broken Yarn | 严重 |
| MS-01 | 漏针 | Missing Stitch | 严重 |
| HO-01 | 破洞 | Hole | 严重 |
| MS-02 | 跳针 | Skip Stitch | 主要 |
| ST-01 | 污渍 | Stain | 中等 |
| CD-01 | 色差 | Color Diff | 中等 |
| TH-01 | 粗节 | Thick Yarn | 轻微 |
| TH-02 | 细节 | Thin Yarn | 轻微 |
| WR-01 | 折痕 | Crease | 轻微 |
| OT-01 | 其他 | Other | 信息 |

## 7 维度混合奖惩

| 维度 | 公式 | 范围 |
|------|------|------|
| D01 定位 | `2*IoU - 1` | [-1, 1] |
| D02 分类 | `±1` | {-1, 1} |
| D03 校准 | `-ECE` | (-inf, 0] |
| D04 漏检 | `-λ·miss` | (-inf, 0] |
| D05 误检 | `-λ·fp` | (-inf, 0] |
| D06 断纱 | `+1 / -2` | {-2, 1} |
| D07 漏针 | `+1 / -2` | {-2, 1} |

`L_total = L_det + β·L_scalar + γ·L_consistency`

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python + FastAPI |
| 数据库 | PostgreSQL + SQLAlchemy 2.0 async |
| 推理 | ONNX Runtime / PyTorch / TensorRT |
| 存储 | MinIO / S3 / 本地回退 |
| 前端 | Chart.js + WebSocket (纯 HTML，中文界面) |
| 训练 | PyTorch + 7 维度混合奖惩 |

## 测试

```bash
# 全部测试
pytest tests/ backend/tests/ backend/training/tests/ backend/inference/tests/ backend/deploy/tests/ -v

# 功能验证
python scripts/func_test.py
```

**168 单元 + 36 功能 = 204 测试全部通过**

## 部署

```bash
# 生成部署产物
python -c "from backend.deploy import generate_all; generate_all('./deploy')"

# Docker
docker-compose -f deploy/docker-compose.yml up -d

# 边缘节点 (systemd)
sudo cp deploy/fabric-defect.service /etc/systemd/system/
sudo systemctl enable --now fabric-defect
```
