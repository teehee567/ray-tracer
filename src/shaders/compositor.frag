#version 450

layout(set = 0, binding = 0) uniform usampler2D accum;
// per-frame peak overlap, written by compositor_reduce.comp
layout(set = 0, binding = 1) readonly buffer MaxBuf {
    uint max_count;
} mb;

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
    float count = float(texture(accum, uv).r);

    float max_overlap = float(mb.max_count);
    // map [1, max] -> [0, 1]; guard against a degenerate (<=1) peak
    float denom = max(max_overlap - 1.0, 1.0);
    float heat = clamp((count - 1.0) / denom, 0.0, 1.0);

    out_color = vec4(heat_color(heat), 1.0);
}
