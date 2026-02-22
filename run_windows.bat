@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

REM ===========================================================
REM  FateAnother RL - Windows Local Training
REM
REM  1 WC3 + 1 Inference Server + 1 Trainer
REM  Requires: Python 3.10+, PyTorch, VS2019+, War3Client.zip
REM ===========================================================

echo.
echo +=====================================================+
echo    FateAnother RL - Windows Local Training
echo    1 WC3 + 1 Inference + 1 Trainer
echo +=====================================================+
echo.

set "ROOT=%~dp0"
cd /d "%ROOT%"

REM -- 1. Prerequisites --
echo [1/7] Checking prerequisites...

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ first.
    pause & exit /b 1
)

python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo ERROR: PyTorch not found. Run: pip install torch
    pause & exit /b 1
)

python -c "import torch; exit(0 if torch.cuda.is_available^(^) else 1)" >nul 2>&1
if errorlevel 1 (
    echo WARNING: CUDA not available. Will use CPU ^(slower^).
    set DEVICE=cpu
) else (
    echo   [OK] CUDA available
    set DEVICE=cuda
)

echo   [OK] Prerequisites OK
echo.

REM -- 2. Assemble War3Client --
echo [2/7] Assembling War3Client...

if not exist "War3Client\JNLoader.exe" (
    python assemble.py --skip-map
    if errorlevel 1 (
        echo ERROR: Assembly failed!
        pause & exit /b 1
    )
) else (
    echo   [OK] War3Client already assembled
)
echo.

REM -- 3. Build Inference Server --
echo [3/7] Building C++ inference server...

if not exist "inference_server\build\Release\fate_inference_server.exe" (
    cd inference_server
    call build_win.bat
    cd /d "%ROOT%"
    if not exist "inference_server\build\Release\fate_inference_server.exe" (
        echo ERROR: Build failed!
        pause & exit /b 1
    )
)
echo   [OK] fate_inference_server.exe ready
echo.

REM -- 4. Create Data Directories --
echo [4/7] Setting up data directories...

if not exist "data\models" mkdir "data\models"
if not exist "data\rollouts" mkdir "data\rollouts"
if not exist "data\checkpoints" mkdir "data\checkpoints"
if not exist "data\runs" mkdir "data\runs"

echo   [OK] Data directories ready
echo.

REM -- 5. Initialize Models --
echo [5/7] Initializing models...

if not exist "data\models\model_latest.pt" (
    python -m fateanother_rl.scripts.init_models --model-dir data\models
    echo   [OK] Models initialized
) else (
    echo   [OK] Models already exist
)
echo.

REM -- 6. Start Services --
echo [6/7] Starting services...

REM Kill any existing processes
taskkill /f /im fate_inference_server.exe >nul 2>&1
taskkill /f /im war3.exe >nul 2>&1

REM Set WC3 speed multiplier
set WC3_SPEED_MULTIPLIER=10

REM Start inference server (background)
start "FateRL-Inference" cmd /c "cd /d "%ROOT%" && inference_server\build\Release\fate_inference_server.exe --port 7777 --action-port 7778 --device %DEVICE% --model-dir data\models --rollout-dir data\rollouts --rollout-size 2048 > data\inference.log 2>&1"

timeout /t 3 /nobreak >nul

REM Start trainer (background, uses dotted notation for CLI overrides)
start "FateRL-Trainer" cmd /c "cd /d "%ROOT%" && python -m fateanother_rl.scripts.train --training.rollout_dir "%ROOT%data\rollouts" --training.model_dir "%ROOT%data\models" --training.save_dir "%ROOT%data\checkpoints" --training.log_dir "%ROOT%data\runs" > data\trainer.log 2>&1"

REM Start TensorBoard (background)
start "FateRL-TensorBoard" cmd /c "python -m tensorboard.main --logdir "%ROOT%data\runs" --port 6006 2>nul"

echo   [OK] Inference server started (port 7777/7778)
echo   [OK] Trainer started (10x speed)
echo   [OK] TensorBoard started (port 6006)
echo.

REM -- 7. Launch WC3 --
echo [7/7] Launching Warcraft III...

cd /d "%ROOT%\War3Client"
REM set LOCAL_PLAYER_SLOT_OVERRIDE=7
set WC3_SPEED_MULTIPLIER=1
start "" JNLoader.exe -loadfile "Maps\rl\fateanother_rl.w3x" -window
REM start "" JNLoader.exe -window
cd /d "%ROOT%"

echo   [OK] WC3 launched
echo.
echo =========================================================
echo.
echo   TensorBoard:  http://localhost:6006
echo.
echo   Log files:
echo     data\inference.log  - Inference server
echo     data\trainer.log    - Python trainer
echo.
echo   To stop everything:
echo     taskkill /f /im fate_inference_server.exe
echo     taskkill /f /im war3.exe
echo     Close the trainer/TensorBoard windows
echo.
echo =========================================================
echo.
pause
