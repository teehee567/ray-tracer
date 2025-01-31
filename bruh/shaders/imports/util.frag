float antialias_samples[] = {
    3.0/8.0, -1.0/8.0,
    1.0/8.0, 3.0/8.0,
    -1.0/8.0, -3.0/8.0,
    -3.0/8.0, 1.0/8.0,
};

int antialias_state = 0;

float pgc_random(inout uint state) {
    #ifdef ANTIALIAS_SAMPLES
       return antialias_samples[antialias_state++ % 8] + 0.5;
    #else
        // minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
        // Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
        state = state * 747796405u + 2891336453u;
        uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return float(word) / 4294967295.0;
    #endif
}

vec2 random_square_offset(inout uint seed) {
    float x = pgc_random(seed) - 0.5;
    float y = pgc_random(seed) - 0.5;

    return vec2(x, y);
}

vec3 random_unit_vector(inout uint seed) {
    vec3 random;
    do {
        random = vec3(pgc_random(seed) * 2.0 - 1.0, pgc_random(seed) * 2.0 - 1.0, pgc_random(seed) * 2.0 - 1.0);
    } while (length(random) > 1);

    return normalize(random);
}

vec3 random_unit_in_hemisphere(inout uint seed, in vec3 normal) {
    vec3 vector = random_unit_vector(seed);

    if (dot(vector, normal) > 0.0) {
        return vector;
    } else {
        return -vector;
    }
}