:: Check if VULKAN_SDK environment variable is set
if "%VULKAN_SDK%"=="" (
    echo Error: VULKAN_SDK environment variable is not set. Please set it to the Vulkan SDK installation path.
    pause
    exit /b 1
)
echo %VULKAN_SDK%

:: Set GLSLC path and compile shaders
set GLSLC_PATH=%VULKAN_SDK%\Bin\glslc.exe

%GLSLC_PATH% ./src/shaders/main.comp -o ./src/shaders/main.comp.spv
%GLSLC_PATH% ./src/shaders/gui.vert -o ./src/shaders/gui.vert.spv
%GLSLC_PATH% ./src/shaders/gui.frag -o ./src/shaders/gui.frag.spv

