#version 450

layout(set = 0, binding = 0) uniform sampler font_sampler;
layout(set = 0, binding = 1) uniform texture2D font_texture;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;

layout(location = 0) out vec4 out_color;

void main() {
    vec4 tex = texture(sampler2D(font_texture, font_sampler), v_uv);
    out_color = tex * v_color;
}
