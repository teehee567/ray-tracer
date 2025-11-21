#version 450

layout(set = 0, binding = 0) uniform sampler2D font_texture;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;

layout(location = 0) out vec4 out_color;

void main() {
    vec4 tex = texture(font_texture, v_uv);
    out_color = tex * v_color;
}
