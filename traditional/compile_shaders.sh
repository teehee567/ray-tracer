#!/bin/bash

# Check if the VULKAN_SDK environment variable is set
if [ -z "$VULKAN_SDK" ]; then
    echo "Error: VULKAN_SDK environment variable is not set. Please set it to the Vulkan SDK installation path."
    exit 1
fi

# Determine the path to glslc
GLSLC_PATH="$VULKAN_SDK/bin/glslc"

# Run the GLSL compiler
$GLSLC_PATH shader.vert -o vert.spv
$GLSLC_PATH shader.frag -o frag.spv

# For Windows, add a pause to keep the terminal open
if [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    read -n 1 -s -r -p "Press any key to continue..."
fi
