#!/usr/bin/env bash
# ============================================================================
#  ML Projects -- One-Click Startup (Linux / macOS)
# ============================================================================
#  Usage:
#    bash start.sh              = install + test + digit (3 epochs)
#    bash start.sh digit        = digit recognition training
#    bash start.sh demo         = mixed-reward demo
#    bash start.sh e2e          = fabric defect E2E simulation
#    bash start.sh backend      = start FastAPI backend server
#    bash start.sh test         = unit tests only
#    bash start.sh install      = install only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODE="${1:-all}"
EXTRA_ARGS="${*:2}"

# Set defaults
if [ "$MODE" = "digit" ] && [ -z "$EXTRA_ARGS" ]; then EXTRA_ARGS="--epochs 3"; fi
if [ "$MODE" = "all" ]   && [ -z "$EXTRA_ARGS" ]; then EXTRA_ARGS="--epochs 3"; fi

# Find Python
PYTHON=""
for p in python3 python; do
    if command -v "$p" &>/dev/null; then PYTHON="$p"; break; fi
done
if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python not found. Install Python 3.10+ first."
    exit 1
fi

echo ""
echo "============================================================"
echo "  ML Projects -- Fabric Defect + Mixed Reward + Digit"
echo "============================================================"
echo ""
echo "[INFO] Python: $PYTHON ($($PYTHON --version 2>&1))"
echo "[INFO] Mode:   $MODE"
echo ""

# ------------------------------------------------------------------
# INSTALL
# ------------------------------------------------------------------
NEED_INSTALL=1
case "$MODE" in demo|digit|e2e|backend|test) NEED_INSTALL=0 ;; esac

if [ $NEED_INSTALL -eq 1 ]; then
    echo ""
    echo "-----------------------------------------------------------"
    if [ "$MODE" = "all" ]; then echo " [1/3] Installing dependencies..."
    else echo " Installing dependencies..."; fi
    echo "-----------------------------------------------------------"
    echo ""
    $PYTHON -m pip install -e . --quiet -i https://pypi.org/simple/ || \
    $PYTHON -m pip install -e . --quiet || \
    echo "[WARN] pip install had issues, continuing..."
    echo "Done!"
fi
[ "$MODE" = "install" ] && { echo ""; echo "All done!"; exit 0; }

# ------------------------------------------------------------------
# TEST
# ------------------------------------------------------------------
NEED_TEST=1
case "$MODE" in demo|digit|e2e|backend) NEED_TEST=0 ;; esac

if [ $NEED_TEST -eq 1 ]; then
    echo ""
    echo "-----------------------------------------------------------"
    if [ "$MODE" = "all" ]; then echo " [2/3] Running unit tests..."
    else echo " Running unit tests..."; fi
    echo "-----------------------------------------------------------"
    echo ""
    $PYTHON -m pytest tests/ backend/tests/ backend/training/tests/ backend/inference/tests/ backend/deploy/tests/ -v --tb=short \
        && echo "" && echo "[OK] All tests passed!"
fi
[ "$MODE" = "test" ] && { echo ""; echo "All done!"; exit 0; }

# ------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------
echo ""
echo "-----------------------------------------------------------"
case "$MODE" in
    all)     echo " [3/3] Running digit recognition training..." ;;
    digit)   echo " Running digit recognition training..." ;;
    demo)    echo " Running mixed-reward demo..." ;;
    e2e)     echo " Running fabric defect E2E simulation..." ;;
    backend) echo " Starting FastAPI backend server..." ;;
esac
echo "-----------------------------------------------------------"
echo ""

case "$MODE" in
    demo)
        $PYTHON scripts/demo_dimension_state.py --steps 200 ;;
    e2e)
        # shellcheck disable=SC2086
        $PYTHON scripts/run_production_demo.py --batches 20 --interval 0.5 ;;
    backend)
        echo "  Dashboard:  http://localhost:8000/dashboard"
        echo "  Simulator:  http://localhost:8000/simulator"
        echo "  API Docs:   http://localhost:8000/docs"
        echo ""
        $PYTHON -m backend.main ;;
    *)
        # shellcheck disable=SC2086
        $PYTHON scripts/train_digit_recognizer.py $EXTRA_ARGS ;;
esac

# ------------------------------------------------------------------
# DONE
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "                     All done!"
echo "============================================================"
echo ""
echo "  Modes:"
echo "    bash start.sh              = install + test + digit (3 epochs)"
echo "    bash start.sh digit        = digit recognition training"
echo "    bash start.sh demo         = mixed-reward demo"
echo "    bash start.sh e2e          = fabric defect E2E simulation"
echo "    bash start.sh backend      = start API server"
echo "    bash start.sh test         = unit tests only"
echo "    bash start.sh install      = install only"
echo ""
echo "  Custom:"
echo "    bash start.sh digit --epochs 10 --wandb"
echo "    bash start.sh e2e --batches 50"
echo ""
echo "  Outputs:"
echo "    Dashboard   : http://localhost:8000/dashboard"
echo "    Simulator   : http://localhost:8000/simulator"
echo "    TensorBoard : tensorboard --logdir ./runs"
echo ""
