#version 460

//#define ANTIALIAS_SAMPLES // must be combined with 4 samples
#define SAMPLES 1200
#define BOUNCES 8

#extension GL_GOOGLE_include_directive : enable
#include "imports/ray.frag"
#include "imports/util.frag"

struct HitRecord {
    bool did_hit;
    bool emissive;
    vec3 pos;
    vec3 normal;
    vec3 color;
    float t;
};

struct Sphere {
    float radius;
    bool emissive;
    vec3 color;
    vec3 center;
};

// todo: look into inline uniform buffers for speed and small data
// todo: alternatively, look into push constants
layout(std140, set = 0, binding = 0) uniform CameraUBO {
    vec2 resolution; // todo: move this to (unsinged) integers
    vec2 uv;
    float focal_len;
    uint frame;
    vec3 origin;
} camera;

layout(set = 0, binding = 1) readonly buffer SphereSBO {
    uint count;
    Sphere spheres[];
} sphere_sbo;

layout(location = 0) out vec4 out_color;

uint seed;

Ray calculate_ray(vec2 frag_coord) {
    /*  We are starting from -uv/2 on each axys, and we need to reach +uv/2.
        Expressed as a formula:
            -uv/2 + pixel_delta * res = +uv/2
            pixel_delta = uv / res

        The Y component is inverted because as the pixels' y axys grows in value
        we want to descend in the 3D space.

        Some of this could be moved to the CPU as a uniform, but is left here
        for now for clarity and because the performance impact is minor.
    */
    vec2 pixel_delta = camera.uv/camera.resolution;
    pixel_delta *= vec2(1, -1);

    vec3 pixel_origin = vec3(-camera.uv.x/2, camera.uv.y/2, camera.focal_len) + // the top left is at x=-u but y=v (up is +)
                        camera.origin +  // to place it correctly in world-space
                        vec3(pixel_delta/2, 0);  // to center the pixel

    // get a random offset for anti-aliasing
    vec2 offset = random_square_offset(seed);
    vec3 pixel_sample = pixel_origin + 
                        vec3(pixel_delta * (frag_coord + offset), 0);

    Ray ray;
    ray.direction = normalize(pixel_sample - camera.origin); // vector from origin to pixel
    ray.origin = camera.origin;

    return ray;
}

HitRecord hit_sphere(Sphere sphere, Ray ray) {
    /*  line (parametric):
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
    vec3 delta = (sphere.center - ray.origin);

    // todo: apply the simpler formula for b multiples of 2
    float a = dot(ray.direction, ray.direction);
    float b = -2 * dot(ray.direction, delta);
    float c = dot(delta, delta) - sphere.radius * sphere.radius;

    float discriminant = b*b - 4*a*c;
    float t = (-b - sqrt(discriminant)) / 2*a;

    HitRecord hit;

    hit.t = t;
    hit.pos = ray_at(ray, t);
    hit.normal = normalize(hit.pos - sphere.center);
    hit.color = sphere.color;
    hit.emissive = sphere.emissive;

    bool did_not_hit = discriminant < 0;
    bool is_inside = dot(hit.normal, ray.direction) > 0;
    bool is_behind = t < 0;

    hit.did_hit = !(did_not_hit || is_behind || is_inside);

    return hit;
}

vec3 background_color(Ray ray) {
    float blend = 0.5 * ray.direction.y + 0.5;
    //return mix(vec3(0.6, 0.8, 1.0), vec3(0.2, 0.4, 1.0), blend);
    return vec3(0);
}


vec3 ray_color(Ray ray) {
    /*  The ray only gets light when it hits something emissive.
        The light that will be "received" from the emissive object
         will be tinted multiplicatively by all the surfaces the ray
         hits before reaching the camera.
        If the ray encounters multiple source of light, all of them
         will contribute to the light value of the pixel.

        The result is that we only add light to incoming_light when
         we hit an emissive, and we tint that light to the color that 
         the ray has assumed so far.
    */
    vec3 ray_color = vec3(1);
    vec3 incoming_light = vec3(0);

    for (int bounce = 0; bounce < BOUNCES + 1; bounce++) {
        HitRecord hit;
        hit.t = 1.0 / 0.0; // infinity
        hit.did_hit = false;

        for (int i = 0; i < sphere_sbo.count; i++) {
            HitRecord current_hit = hit_sphere(sphere_sbo.spheres[i], ray);

            if (current_hit.did_hit && current_hit.t < hit.t) {
                hit = current_hit;
            }
        }

        if (hit.did_hit) {
            ray_color *= hit.color;
            incoming_light += hit.emissive ? ray_color * 10 : vec3(0.0);
        } else {
            incoming_light += background_color(ray) * ray_color;
            break;
        }

        ray.origin = hit.pos;
        ray.direction = random_unit_in_hemisphere(seed, hit.normal);
    }

    return incoming_light;
}

void main() {
    // initialize random seed
    seed = uint(gl_FragCoord.x + camera.resolution.x * gl_FragCoord.y + camera.frame * 4321u);

    vec3 color = vec3(0);
    for (int i = 0; i < SAMPLES; i++) {
        Ray ray = calculate_ray(gl_FragCoord.xy);

        color += ray_color(ray) / float(SAMPLES);
    }

    out_color = vec4(color, 1.0);
}