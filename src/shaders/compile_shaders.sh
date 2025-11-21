#!/bin/bash
# Check if VULKAN_SDK environment variable is set
if [ -z "$VULKAN_SDK" ]; then
    echo "Error: VULKAN_SDK environment variable is not set. Please set it to the Vulkan SDK installation path."
    exit 1
fi

echo "$VULKAN_SDK"

# Set GLSLC path (note: on macOS, the executable is typically named "glslc" without the .exe extension)
GLSLC_PATH="$VULKAN_SDK/bin/glslc"

# Compile the shaders
"$GLSLC_PATH" ./src/shaders/main.comp -o ./src/shaders/main.comp.spv
"$GLSLC_PATH" ./src/shaders/gui.vert -o ./src/shaders/gui.vert.spv
"$GLSLC_PATH" ./src/shaders/gui.frag -o ./src/shaders/gui.frag.spv
