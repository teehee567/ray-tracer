#version 450

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
} pc;

layout(location = 0) in vec3 in_pos;

void main() {
    gl_Position = pc.view_proj * vec4(in_pos, 1.0);
}
