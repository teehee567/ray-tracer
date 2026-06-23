#version 450

layout(set = 0, binding = 0, r32ui) uniform uimage2D accum;

void main() {
    imageAtomicAdd(accum, ivec2(gl_FragCoord.xy), 1u);
}
