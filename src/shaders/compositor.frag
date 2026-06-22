#version 450

layout(set = 0, binding = 0) uniform sampler2D accum;
layout(push_constant) uniform PushConstants {
    float max_overlap;
} pc;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

const vec3 COLD = vec3(0.0, 0.0, 0.0);
const vec3 MID  = vec3(0.1, 0.3, 1.0);
const vec3 HOT  = vec3(1.0, 1.0, 1.0);

vec3 heat_color(float t) {
    if (t < 0.5) {
        return mix(COLD, MID, t * 2.0);
    } else {
        return mix(MID, HOT, (t - 0.5) * 2.0);
    }
}

void main() {
    float count = texture(accum, uv).r;

    float heat = clamp((count - 1.0) / (pc.max_overlap - 1.0), 0.0, 1.0);

    out_color = vec4(heat_color(heat), 1.0);
}
