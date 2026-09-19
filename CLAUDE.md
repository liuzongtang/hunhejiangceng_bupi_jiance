# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Textile fabric defect detection system — FastAPI backend + PyTorch/ONNX inference + 7-dimension hybrid reward training. Supports 20 defect types (Tianchi scheme) with real-time WebSocket alerts and a Chinese-language dashboard.

## Commands

```bash
# Install (editable)
pip install -e .

# Run all tests (168 unit + 36 functional)
pytest tests/ backend/tests/ backend/training/tests/ backend/inference/tests/ backend/deploy/tests/ -v --tb=short

# Run a single test file
pytest backend/tests/test_detection.py -v

# Run a single test function
pytest backend/tests/test_detection.py::test_submit_report -v

# Start API server (then visit http://localhost:8000/dashboard)
python -m backend.main

# One-click scripts (Windows)
scripts\start.bat test       # tests only
scripts\start.bat backend    # start server
scripts\start.bat e2e        # E2E simulation

# Functional validation
python scripts/func_test.py
```

No linting/formatting config exists yet (no ruff, flake8, or pyproject.toml).

## Architecture

### Configuration (Singleton Dataclass)

`backend/config.py` — `BackendConfig` is a dataclass with nested config objects (`DatabaseConfig`, `ModelConfig`, `AlertConfig`, etc.). Loaded via `get_config()` singleton that tries YAML files then falls back to defaults. Always access via `get_config()`, never instantiate directly. Tests reset it by setting `cfg._config = None`.

### Database (SQLAlchemy 2.0 Async)

`backend/database.py` — Async engine with automatic SQLite (dev) vs PostgreSQL (prod) based on `config.environment`. Session is provided via FastAPI dependency `get_session() → AsyncGenerator[AsyncSession, None]`. All models inherit from `Base` (DeclarativeBase). Tables are auto-created on startup.

**Pattern for DB queries:**
```python
result = await session.execute(select(Model).where(...))
items = result.scalars().all()
```

### API Layer Pattern

```
Router (routers/*.py) → Service (services/*.py) → Model (models/*.py)
```

- **Routers**: Thin — validate input via Pydantic, call service, wrap in `APIResponse(code=0, message="success", data=...)`.
- **Services**: Business logic — instantiated per-request with `AsyncSession`, handle DB operations and side effects.
- **Models**: SQLAlchemy ORM with `Mapped[]` type annotations, `to_dict()` methods for serialization.

### Pydantic Schemas (Two Layers)

- `backend/schemas/api.py` — External API request/response models (`DetectionReportRequest`, `APIResponse`, etc.)
- `backend/schemas/detection.py` — Internal module interfaces (`ImageCaptureOutput`, `DetectionOutput`, `DefectItem`, etc.)
- `backend/schemas/defect.py` — Shared enums: `DefectType`, `Severity`, `DefectCode`, plus `DEFECT_CODES` mapping.

### Inference Engine (ABC + Factory)

`backend/inference/engine.py` — `InferenceEngine` is an ABC defining `infer(images) → List[Dict]` and `load()`. Four implementations:
- `DummyInferenceEngine` — random detections for testing (no model needed)
- `ONNXInferenceEngine` — ONNX Runtime for production
- `PyTorchInferenceEngine` — native PyTorch for dev/debugging
- `RTDETRONNXEngine` — RT-DETR ONNX model for the 20-class Tianchi detector

Use `create_engine(backend="dummy|onnx|pytorch|rtdetr")` factory function. `backend/inference/detector.py` wraps the engine with preprocessing and postprocessing via `FabricDefectDetector` — use `get_detector()` singleton in API code.

### Training (7-Dimension Hybrid Reward)

`backend/training/hybrid_reward_trainer.py` — `HybridRewardTrainer` implements `L_total = L_detection + β·L_scalar + γ·L_consistency` across 7 dimensions (localization, classification, calibration, miss penalty, false-positive penalty, broken-yarn, skip/weave). Dimension reward computation is in `backend/training/dimension_rewards.py`.

### Mixed Reward (RLHF)

`mixed_reward/` — Standalone package for scalar+dimension hybrid reward in RLHF. `MultiDimensionalRewardManager` computes per-dimension reward vectors using pluggable scorers (`accuracy`, `safety`, `completeness`, `format`). Integrates with verl framework via `custom_reward_function.path`.

### Logging

`backend/logging.py` — Structured JSON logging via `LogManager`. Singleton loggers: `get_inference_log()`, `get_training_log()`. Log format includes `timestamp`, `level`, `service`, `trace_id`, `module`, `message`, `data`. Use `log.critical(msg, data={...})` for structured detection events.

### WebSocket

`backend/websocket.py` — WebSocket manager at `/ws/alerts` for real-time defect alert push. Singleton via `get_ws_manager()`. Client sends `"ping"`, server responds with `{"type":"pong"}`.

### Frontend

Two standalone HTML files with Chinese UI: `backend/simulator.html` (production line simulator with Canvas fabric rendering) and `backend/dashboard.html` (monitoring dashboard with Chart.js + WebSocket). Served directly by FastAPI, no build step.

## Coding Conventions

- **Imports**: `from __future__ import annotations` at top of every `.py` file.
- **Type annotations**: On all function signatures and class attributes. Use `Optional[X]` not `X | None` (Python 3.10+ compatible). Use `List`, `Dict`, `Tuple` from `typing`.
- **Async**: All DB and I/O operations are `async def`. Services take `AsyncSession` in `__init__`.
- **Docstrings**: Google-style with Args/Returns sections. Every public class and function has one.
- **API envelope**: Every endpoint returns `APIResponse(code=0, message="success", data=...)`. Errors raise `HTTPException`.
- **Singletons**: Config, database engine, detector, log managers, and WebSocket manager all use the module-level `_instance = None` + `get_xxx()` pattern.
- **Naming**: `snake_case` for files/variables/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- **Models**: ORM models use `Mapped[]` with `mapped_column()`. Every model has `to_dict()` for JSON serialization.
- **Test fixtures**: `backend/tests/conftest.py` provides `client` (sync TestClient), `clean_database` (autouse, resets config + engine + deletes SQLite file), and sample request dicts.
