@echo off
:: One-click startup for ML Projects
:: Usage:
::   start.bat              = install + test + digit (3 epochs)
::   start.bat digit        = digit recognition training
::   start.bat demo         = mixed-reward demo
::   start.bat e2e          = fabric defect E2E simulation
::   start.bat backend      = start FastAPI backend server
::   start.bat test         = unit tests only
::   start.bat install      = install only

setlocal enabledelayedexpansion
cd /d "%~dp0.."

:: Find Python
set "PYTHON="
for %%p in (python.exe python3.exe) do (
    where %%p >nul 2>nul && for /f "delims=" %%i in ('where %%p 2^>nul') do if not defined PYTHON set "PYTHON=%%i"
)
if not defined PYTHON if exist "C:\Users\master\anaconda3\python.exe" set "PYTHON=C:\Users\master\anaconda3\python.exe"
if not defined PYTHON if exist "C:\Program Files\Python313\python.exe" set "PYTHON=C:\Program Files\Python313\python.exe"
if not defined PYTHON (
    echo [ERROR] Python not found.
    pause & exit /b 1
)

:: Parse mode
set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"

echo.
echo ============================================================
echo   ML Projects -- Fabric Defect + Mixed Reward + Digit
echo ============================================================
echo.
echo [INFO] Python: %PYTHON%
echo [INFO] Mode:   %MODE%
echo.

:: --- INSTALL ---
if "%MODE%"=="demo"    goto :skip_install
if "%MODE%"=="digit"   goto :skip_install
if "%MODE%"=="e2e"     goto :skip_install
if "%MODE%"=="backend" goto :skip_install
if "%MODE%"=="test"    goto :skip_install
echo ------------------------------------------------------------
echo  Installing dependencies...
echo ------------------------------------------------------------
echo.
%PYTHON% -m pip install -e . --quiet -i https://pypi.org/simple/
if %errorlevel% neq 0 (
    echo [WARN] Retrying without index override...
    %PYTHON% -m pip install -e . --quiet
)
echo Done!
if "%MODE%"=="install" goto :done

:skip_install

:: --- TEST ---
if "%MODE%"=="demo"    goto :skip_test
if "%MODE%"=="digit"   goto :skip_test
if "%MODE%"=="e2e"     goto :skip_test
if "%MODE%"=="backend" goto :skip_test
echo.
echo ------------------------------------------------------------
echo  Running unit tests...
echo ------------------------------------------------------------
echo.
%PYTHON% -m pytest tests/ backend/tests/ backend/training/tests/ backend/inference/tests/ backend/deploy/tests/ -v --tb=short
if %errorlevel% neq 0 (echo [WARN] Some tests failed.) else (echo [OK] All tests passed!)
if "%MODE%"=="test" goto :done

:skip_test

:: --- RUN ---
echo.
echo ------------------------------------------------------------
if "%MODE%"=="all"     echo  Running digit recognition training...
if "%MODE%"=="digit"   echo  Running digit recognition training...
if "%MODE%"=="demo"    echo  Running mixed-reward demo...
if "%MODE%"=="e2e"     echo  Running fabric defect E2E simulation...
if "%MODE%"=="backend" echo  Starting FastAPI backend server...
echo ------------------------------------------------------------
echo.

if "%MODE%"=="demo" (
    %PYTHON% scripts\demo_dimension_state.py --steps 200
    goto :done
)

if "%MODE%"=="e2e" (
    %PYTHON% scripts\run_production_demo.py --batches 20 --interval 0.5
    goto :done
)

if "%MODE%"=="backend" (
    echo  Dashboard:  http://localhost:8000/dashboard
    echo  Simulator:  http://localhost:8000/simulator
    echo  API Docs:   http://localhost:8000/docs
    echo.
    %PYTHON% -m backend.main
    goto :done
)

:: Digit recognition (default / all)
set "ARGS=--epochs 3"
if not "%~2"=="" set "ARGS=%2 %3 %4 %5 %6 %7 %8 %9"
%PYTHON% scripts\train_digit_recognizer.py %ARGS%

:done
echo.
echo ============================================================
echo                     All done!
echo ============================================================
echo.
echo   Modes:
echo     start.bat              = install + test + digit (3 epochs^)
echo     start.bat digit        = digit recognition training
echo     start.bat demo         = mixed-reward demo
echo     start.bat e2e          = fabric defect E2E simulation
echo     start.bat backend      = start API server
echo     start.bat test         = unit tests only
echo     start.bat install      = install only
echo.
echo   Custom:
echo     start.bat digit --epochs 10 --wandb
echo     start.bat e2e --batches 50
echo.
echo   Outputs:
echo     Dashboard   : http://localhost:8000/dashboard
echo     Simulator   : http://localhost:8000/simulator
echo     TensorBoard : tensorboard --logdir ./runs
echo.
pause
