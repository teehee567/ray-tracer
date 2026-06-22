:: Check if VULKAN_SDK environment variable is set
if "%VULKAN_SDK%"=="" (
    echo Error: VULKAN_SDK environment variable is not set. Please set it to the Vulkan SDK installation path.
    pause
    exit /b 1
)
echo %VULKAN_SDK%

:: Set GLSLC path and compile shaders
set GLSLC_PATH=%VULKAN_SDK%\Bin\glslc.exe

%GLSLC_PATH% ./src/shaders/main.comp -o ./src/shaders/main.comp.spv || exit /b 1
%GLSLC_PATH% ./src/shaders/gui.vert -o ./src/shaders/gui.vert.spv || exit /b 1
%GLSLC_PATH% ./src/shaders/gui.frag -o ./src/shaders/gui.frag.spv || exit /b 1
%GLSLC_PATH% ./src/shaders/heatmap.vert -o ./src/shaders/heatmap.vert.spv || exit /b 1
%GLSLC_PATH% ./src/shaders/heatmap.frag -o ./src/shaders/heatmap.frag.spv || exit /b 1
%GLSLC_PATH% ./src/shaders/compositor.vert -o ./src/shaders/compositor.vert.spv || exit /b 1
%GLSLC_PATH% ./src/shaders/compositor.frag -o ./src/shaders/compositor.frag.spv || exit /b 1
%GLSLC_PATH% ./src/shaders/compositor_reduce.comp -o ./src/shaders/compositor_reduce.comp.spv || exit /b 1
