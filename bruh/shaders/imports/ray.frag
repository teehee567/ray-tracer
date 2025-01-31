struct Ray {
    vec3 origin;
    vec3 direction;
};

vec3 ray_at(Ray ray, float t) {
    return ray.origin + (ray.direction * t);
}

