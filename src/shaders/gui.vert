#version 450

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
} pc;

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;

void main() {
    vec2 pos = in_pos / pc.screen_size * vec2(2.0, 2.0) - vec2(1.0, 1.0);
    pos.y = -pos.y;
    gl_Position = vec4(pos, 0.0, 1.0);

    v_uv = in_uv;
    v_color = in_color;
}
