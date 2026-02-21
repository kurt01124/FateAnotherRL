@echo off
REM Build fate_inference_server on Windows with VS2019
REM Requires: PyTorch (pip install torch) or standalone libtorch

setlocal

REM --- Detect libtorch path ---
for /f "delims=" %%i in ('python -c "import torch; print(torch.utils.cmake_prefix_path)"') do set TORCH_CMAKE=%%i

if "%TORCH_CMAKE%"=="" (
    echo ERROR: PyTorch not found. Install with: pip install torch
    exit /b 1
)
echo Using libtorch from: %TORCH_CMAKE%

REM --- Create build directory ---
if not exist build mkdir build
cd build

REM --- Configure with CMake (VS2019 x64) ---
cmake -G "Visual Studio 16 2019" -A x64 ^
    -DCMAKE_PREFIX_PATH="%TORCH_CMAKE%" ^
    ..

if errorlevel 1 (
    echo CMake configure failed!
    exit /b 1
)

REM --- Build Release ---
cmake --build . --config Release -j

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo.
echo Build successful! Binary: build\Release\fate_inference_server.exe
