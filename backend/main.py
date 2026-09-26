"""
FastAPI application entry point for the textile defect detection backend.

Start:
    python -m backend.main
    uvicorn backend.main:app --reload --port 8000

OpenAPI docs:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_config
from backend.database import close_db, init_db
from backend.routers import detection, model, system


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: initialize database
    config = get_config()
    print("[INFO] Starting Fabric Defect Detection Backend")
    print(f"[INFO] Environment: {config.environment}")
    print(f"[INFO] Server: {config.server.host}:{config.server.port}")
    await init_db()
    print("[INFO] Database initialized")
    yield
    # Shutdown
    await close_db()
    print("[INFO] Server shut down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    get_config()  # Ensure config is loaded

    app = FastAPI(
        title="Fabric Defect Detection API",
        description="""
## Textile Fabric Defect Detection System

Backend API for computer-vision-based fabric inspection on production lines.

### Features
- **Detection Reports**: Submit and query defect detection results
- **Model Management**: Register and deploy model updates
- **System Monitoring**: Device status, model versions, resource metrics

### Defect Types (18 Tianchi classes)
| Code  | Name          | Severity |
|-------|---------------|----------|
| HO-01 | Hole          | Critical |
| ST-01 | Stain         | Medium   |
| SL-01 | Three Silk    | Minor    |
| KN-01 | Knot          | Minor    |
| FL-01 | Flower Board  | Major    |
| HF-01 | Hundred Feet  | Major    |
| HP-01 | Hair Particle | Minor    |
| CW-01 | Coarse Warp   | Minor    |
| LW-01 | Loose Warp    | Minor    |
| BW-01 | Broken Warp   | Critical |
| HW-01 | Hanging Warp  | Major    |
| WS-01 | Weft Shrink   | Minor    |
| ST-02 | Size Stain    | Medium   |
| KN-02 | Warping Knot  | Minor    |
| SK-01 | Star Skip     | Major    |
| BS-01 | Broken Spandex| Critical |
| DS-01 | Dense Section | Major    |
| WD-01 | Weave Defect  | Major    |
""",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware — allow edge devices and frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files (Chart.js, etc.)
    import os as _os

    from fastapi.staticfiles import StaticFiles

    _static_dir = _os.path.join(_os.path.dirname(__file__), "static")
    _os.makedirs(_static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

    # Register routers
    app.include_router(detection.router)
    app.include_router(model.router)
    app.include_router(system.router)
    from backend.routers import simulator as simulator_router, storage as storage_router

    app.include_router(simulator_router.router)
    app.include_router(storage_router.router)

    # WebSocket endpoint for real-time alerts
    from fastapi import WebSocket, WebSocketDisconnect

    from backend.websocket import get_ws_manager

    @app.websocket("/ws/alerts")
    async def ws_alerts(websocket: WebSocket):
        manager = get_ws_manager()
        await manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                # Client can send pings; we respond with pong + latest stats
                if data == "ping":
                    await websocket.send_text('{"type":"pong"}')
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception:
            manager.disconnect(websocket)

    # Dashboard route
    import os

    from fastapi.responses import HTMLResponse

    @app.get("/simulator", response_class=HTMLResponse, include_in_schema=False)
    async def simulator():
        sim_path = os.path.join(os.path.dirname(__file__), "simulator.html")
        with open(sim_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard():
        dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    return app


# Application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "backend.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        log_level=config.server.log_level,
    )
