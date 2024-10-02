@echo off

:: Check if VULKAN_SDK environment variable is set
if "%VULKAN_SDK%"=="" (
    echo Error: VULKAN_SDK environment variable is not set. Please set it to the Vulkan SDK installation path.
    pause
    exit /b 1
)
echo %VULKAN_SDK%

:: Set GLSLC path and compile shaders
set GLSLC_PATH=%VULKAN_SDK%\Bin\glslc.exe

%GLSLC_PATH% ~/shaders/shader.vert -o vert.spv
%GLSLC_PATH% ~/shaders/shader.frag -o frag.spv

:: Pause to keep the terminal open
pause
