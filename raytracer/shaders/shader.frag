#version 450

struct Ray {
    vec3 origin;
    vec3 direction;
};

// todo: look into inline uniform buffers for speed and small data
layout(binding = 0) uniform CameraBufferObject {
    uint width;
    uint height;
    float focal_len;

    vec3 origin; // should these two be merged
} camera;

layout(location = 0) out vec4 out_color;

Ray calculate_ray() {
    Ray ray;

    // this shouldn't impact peformance too much as there's zero branch divergence
    // at the same time, a lot of this math is identical for each pixel so it could be moved to a uniform to avoid divisions
    float ratio = float(camera.width) / float(camera.height);
    if (ratio > 1) {
        ray.direction = vec3(((gl_FragCoord.x / camera.height)*2-ratio), ((gl_FragCoord.y / camera.height)*2-1), -camera.focal_len);
    } else {
        ray.direction = vec3(((gl_FragCoord.x / camera.width)*2-1), ((gl_FragCoord.y / camera.width)*2-(1/ratio)), -camera.focal_len);
    }
    
    // this is not necessary but kept here for good measure
    ray.direction = normalize(ray.direction);

    ray.origin = camera.origin;

    return ray;
}

bool hit_sphere(vec3 center, float radius, Ray ray) {
    /*  
        line (parametric):
            x = orig_x + dir_x * t
            y = orig_y + dir_y * t   ->   p = orig + dir * t
            z = orig_z + dir_z * t

        sphere formula:
            (x - c_x)^2 + (y - c_y)^2 + (z - c_z)^2 <= radius^2    ->    (p - center).(p - center) <= radius^2

        now we plug the line in the sphere formula and solve for t:
            (orig + dir * t - center).(orig + dir * t - center) <= r^2

            [...]

            a = dir.dir
            b = 2dir.(orig - center)
            c = (center - orig).(center - orig) - radius^2

        referring to the quadratic formula, we can tell whether the ray intersected the sphere by the discriminant:
            hit_sphere = (b^2 - 4ac >= 0)
    */
    vec3 delta = (center - ray.origin);

    float a = dot(ray.direction, ray.direction);
    float b = -2 * dot(ray.direction, delta);
    float c = dot(delta, delta) - radius * radius;

    return (b*b - 4*a*c >= 0);
}

void main() {
    Ray ray = calculate_ray();
    
    bool hit = hit_sphere(vec3(0, 0, 10), 1, ray);

    out_color = hit ? vec4(1.0,0.0,0.0,1.0) : vec4(0.0,1.0,0.0,1.0);
    //out_color = rotated_dir;
}
