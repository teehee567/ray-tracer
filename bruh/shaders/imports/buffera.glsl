
// Port of https://github.com/knightcrawler25/GLSL-PathTracer
// Copyright(c) 2019-2021 Asif Ali

// Headers

struct BsdfSampleRec
{
    vec3 L;
    vec3 f;
    float pdf;
};

struct LightSampleRec
{
    vec3 normal;
    vec3 emission;
    vec3 direction;
    float dist;
    float pdf;
};

uvec4 seed;
ivec2 pixel;

int numOfLights;

void InitRNG(vec2 p, int frame)
{
    pixel = ivec2(p);
    seed = uvec4(p, uint(frame), uint(p.x) + uint(p.y));
}

void pcg4d(inout uvec4 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v = v ^ (v >> 16u);
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
}

float rand()
{
    pcg4d(seed); return float(seed.x) / float(0xffffffffu);
}

vec3 FaceForward(vec3 a, vec3 b)
{
    return dot(a, b) < 0.0 ? -b : b;
}

float Luminance(vec3 c)
{
    return 0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z;
}

// Camera

Ray getCameraRay(vec2 offset) {
    vec3 origin = vec3(0);
    vec3 lookAt = vec3(0);
    
    getCameraPos(origin, lookAt);

    vec2 uv = (gl_FragCoord.xy + offset) / iResolution.xy - .5;
    uv.y *= iResolution.y / iResolution.x;

    vec3 iu = vec3(0., 1., 0.);

    vec3 iz = normalize( lookAt - origin );
    vec3 ix = normalize( cross(iz, iu) );
    vec3 iy = cross(ix, iz);

    vec3 direction = normalize( uv.x * ix + uv.y * iy + .85 * iz );

    return Ray(origin, direction);
}


// Sampling

float GTR1(float NDotH, float a)
{
    if (a >= 1.0)
        return INV_PI;
    float a2 = a * a;
    float t = 1.0 + (a2 - 1.0) * NDotH * NDotH;
    return (a2 - 1.0) / (PI * log(a2) * t);
}

vec3 SampleGTR1(float rgh, float r1, float r2)
{
    float a = max(0.001, rgh);
    float a2 = a * a;

    float phi = r1 * TWO_PI;

    float cosTheta = sqrt((1.0 - pow(a2, 1.0 - r1)) / (1.0 - a2));
    float sinTheta = clamp(sqrt(1.0 - (cosTheta * cosTheta)), 0.0, 1.0);
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);

    return vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

float GTR2(float NDotH, float a)
{
    float a2 = a * a;
    float t = 1.0 + (a2 - 1.0) * NDotH * NDotH;
    return a2 / (PI * t * t);
}

vec3 SampleGTR2(float rgh, float r1, float r2)
{
    float a = max(0.001, rgh);

    float phi = r1 * TWO_PI;

    float cosTheta = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
    float sinTheta = clamp(sqrt(1.0 - (cosTheta * cosTheta)), 0.0, 1.0);
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);

    return vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

vec3 SampleGGXVNDF(vec3 V, float rgh, float r1, float r2)
{
    vec3 Vh = normalize(vec3(rgh * V.x, rgh * V.y, V.z));

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) * inversesqrt(lensq) : vec3(1, 0, 0);
    vec3 T2 = cross(Vh, T1);

    float r = sqrt(r1);
    float phi = 2.0 * PI * r2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

    return normalize(vec3(rgh * Nh.x, rgh * Nh.y, max(0.0, Nh.z)));
}

float GTR2Aniso(float NDotH, float HDotX, float HDotY, float ax, float ay)
{
    float a = HDotX / ax;
    float b = HDotY / ay;
    float c = a * a + b * b + NDotH * NDotH;
    return 1.0 / (PI * ax * ay * c * c);
}

vec3 SampleGTR2Aniso(float ax, float ay, float r1, float r2)
{
    float phi = r1 * TWO_PI;

    float sinPhi = ay * sin(phi);
    float cosPhi = ax * cos(phi);
    float tanTheta = sqrt(r2 / (1.0 - r2));

    return vec3(tanTheta * cosPhi, tanTheta * sinPhi, 1.0);
}

float SmithG(float NDotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NDotV * NDotV;
    return (2.0 * NDotV) / (NDotV + sqrt(a + b - a * b));
}

float SmithGAniso(float NDotV, float VDotX, float VDotY, float ax, float ay)
{
    float a = VDotX * ax;
    float b = VDotY * ay;
    float c = NDotV;
    return 1.0 / (NDotV + sqrt(a * a + b * b + c * c));
}

float SchlickFresnel(float u)
{
    float m = clamp(1.0 - u, 0.0, 1.0);
    float m2 = m * m;
    return m2 * m2 * m;
}

float DielectricFresnel(float cosThetaI, float eta)
{
    float sinThetaTSq = eta * eta * (1.0f - cosThetaI * cosThetaI);

    // Total internal reflection
    if (sinThetaTSq > 1.0)
        return 1.0;

    float cosThetaT = sqrt(max(1.0 - sinThetaTSq, 0.0));

    float rs = (eta * cosThetaT - cosThetaI) / (eta * cosThetaT + cosThetaI);
    float rp = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);

    return 0.5f * (rs * rs + rp * rp);
}

vec3 CosineSampleHemisphere(float r1, float r2)
{
    vec3 dir;
    float r = sqrt(r1);
    float phi = TWO_PI * r2;
    dir.x = r * cos(phi);
    dir.y = r * sin(phi);
    dir.z = sqrt(max(0.0, 1.0 - dir.x * dir.x - dir.y * dir.y));
    return dir;
}

vec3 UniformSampleHemisphere(float r1, float r2)
{
    float r = sqrt(max(0.0, 1.0 - r1 * r1));
    float phi = TWO_PI * r2;
    return vec3(r * cos(phi), r * sin(phi), r1);
}

vec3 UniformSampleSphere(float r1, float r2)
{
    float z = 1.0 - 2.0 * r1;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = TWO_PI * r2;
    return vec3(r * cos(phi), r * sin(phi), z);
}

float PowerHeuristic(float a, float b)
{
    float t = a * a;
    return t / (b * b + t);
}

void Onb(in vec3 N, inout vec3 T, inout vec3 B)
{
    vec3 up = abs(N.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

void SampleSphereLight(in Light light, in vec3 surfacePos, inout LightSampleRec lightSampleRec)
{
    float r1 = rand();
    float r2 = rand();

    vec3 sphereCentertoSurface = surfacePos - light.position;
    float distToSphereCenter = length(sphereCentertoSurface);
    vec3 sampledDir;

    // TODO: Fix this. Currently assumes the light will be hit only from the outside
    sphereCentertoSurface /= distToSphereCenter;
    sampledDir = UniformSampleHemisphere(r1, r2);
    vec3 T, B;
    Onb(sphereCentertoSurface, T, B);
    sampledDir = T * sampledDir.x + B * sampledDir.y + sphereCentertoSurface * sampledDir.z;

    vec3 lightSurfacePos = light.position + sampledDir * light.radius;

    lightSampleRec.direction = lightSurfacePos - surfacePos;
    lightSampleRec.dist = length(lightSampleRec.direction);
    float distSq = lightSampleRec.dist * lightSampleRec.dist;

    lightSampleRec.direction /= lightSampleRec.dist;
    lightSampleRec.normal = normalize(lightSurfacePos - light.position);
    lightSampleRec.emission = light.emission * float(numOfLights);
    lightSampleRec.pdf = distSq / (light.area * 0.5 * abs(dot(lightSampleRec.normal, lightSampleRec.direction)));
}

void SampleRectLight(in Light light, in vec3 surfacePos, inout LightSampleRec lightSampleRec)
{
    float r1 = rand();
    float r2 = rand();

    vec3 lightSurfacePos = light.position + light.u * r1 + light.v * r2;
    lightSampleRec.direction = lightSurfacePos - surfacePos;
    lightSampleRec.dist = length(lightSampleRec.direction);
    float distSq = lightSampleRec.dist * lightSampleRec.dist;
    lightSampleRec.direction /= lightSampleRec.dist;
    lightSampleRec.normal = normalize(cross(light.u, light.v));
    lightSampleRec.emission = light.emission * float(numOfLights);
    lightSampleRec.pdf = distSq / (light.area * abs(dot(lightSampleRec.normal, lightSampleRec.direction)));
}

void SampleDistantLight(in Light light, in vec3 surfacePos, inout LightSampleRec lightSampleRec)
{
    lightSampleRec.direction = normalize(light.position - vec3(0.0));
    lightSampleRec.normal = normalize(surfacePos - light.position);
    lightSampleRec.emission = light.emission * float(numOfLights);
    lightSampleRec.dist = INF;
    lightSampleRec.pdf = 1.0;
}

void SampleOneLight(in Light light, in vec3 surfacePos, inout LightSampleRec lightSampleRec)
{
    int type = int(light.type);

    if (type == QUAD_LIGHT)
        SampleRectLight(light, surfacePos, lightSampleRec);
    else if (type == SPHERE_LIGHT)
        SampleSphereLight(light, surfacePos, lightSampleRec);
    else
        SampleDistantLight(light, surfacePos, lightSampleRec);
}


vec3 EmitterSample(in Ray r, in int depth, in LightSampleRec lightSampleRec, in BsdfSampleRec bsdfSampleRec)
{
    vec3 Le;

    if (depth == 0)
        Le = lightSampleRec.emission;
    else
        Le = PowerHeuristic(bsdfSampleRec.pdf, lightSampleRec.pdf) * lightSampleRec.emission;

    return Le;
}

// Disney

vec3 ToWorld(vec3 X, vec3 Y, vec3 Z, vec3 V)
{
    return V.x * X + V.y * Y + V.z * Z;
}

vec3 ToLocal(vec3 X, vec3 Y, vec3 Z, vec3 V)
{
    return vec3(dot(V, X), dot(V, Y), dot(V, Z));
}

float FresnelMix(Material material, float eta, float VDotH)
{
    float metallicFresnel = SchlickFresnel(VDotH);
    float dielectricFresnel = DielectricFresnel(VDotH, eta);
    return mix(dielectricFresnel, metallicFresnel, material.metallic);
}

vec3 EvalDiffuse(Material material, vec3 Csheen, vec3 V, vec3 L, vec3 H, out float pdf)
{
    pdf = 0.0;
    if (L.z <= 0.0)
        return vec3(0.0);

    // Diffuse
    float FL = SchlickFresnel(L.z);
    float FV = SchlickFresnel(V.z);
    float FH = SchlickFresnel(dot(L, H));
    float Fd90 = 0.5 + 2.0 * dot(L, H) * dot(L, H) * material.roughness;
    float Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);

    // Fake Subsurface TODO: Replace with volumetric scattering
    float Fss90 = dot(L, H) * dot(L, H) * material.roughness;
    float Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);
    float ss = 1.25 * (Fss * (1.0 / (L.z + V.z) - 0.5) + 0.5);

    // Sheen
    vec3 Fsheen = FH * material.sheen * Csheen;

    pdf = L.z * INV_PI;
    return (INV_PI * mix(Fd, ss, material.subsurface) * material.baseColor + Fsheen) * (1.0 - material.metallic) * (1.0 - material.specTrans);
}

vec3 EvalSpecReflection(Material material, float eta, vec3 specCol, vec3 V, vec3 L, vec3 H, out float pdf)
{
    pdf = 0.0;
    if (L.z <= 0.0)
        return vec3(0.0);

    float FM = FresnelMix(material, eta, dot(L, H));
    vec3 F = mix(specCol, vec3(1.0), FM);
    float D = GTR2(H.z, material.roughness);
    float G1 = SmithG(abs(V.z), material.roughness);
    float G2 = G1 * SmithG(abs(L.z), material.roughness);
    float jacobian = 1.0 / (4.0 * dot(V, H));

    pdf = G1 * max(0.0, dot(V, H)) * D * jacobian / V.z;
    return F * D * G2 / (4.0 * L.z * V.z);
}

vec3 EvalSpecRefraction(Material material, float eta, vec3 V, vec3 L, vec3 H, out float pdf)
{
    pdf = 0.0;
    if (L.z >= 0.0)
        return vec3(0.0);

    float F = DielectricFresnel(abs(dot(V, H)), eta);
    float D = GTR2(H.z, material.roughness);
    float denom = dot(L, H) + dot(V, H) * eta;
    denom *= denom;
    float G1 = SmithG(abs(V.z), material.roughness);
    float G2 = G1 * SmithG(abs(L.z), material.roughness);
    float jacobian = abs(dot(L, H)) / denom;

    pdf = G1 * max(0.0, dot(V, H)) * D * jacobian / V.z;

    vec3 specColor = pow(material.baseColor, vec3(0.5));
    return specColor * (1.0 - material.metallic) * material.specTrans * (1.0 - F) * D * G2 * abs(dot(V, H)) * abs(dot(L, H)) * eta * eta / (denom * abs(L.z) * abs(V.z));
}

vec3 EvalClearcoat(Material material, vec3 V, vec3 L, vec3 H, out float pdf)
{
    pdf = 0.0;
    if (L.z <= 0.0)
        return vec3(0.0);

    float FH = DielectricFresnel(dot(V, H), 1.0 / 1.5);
    float F = mix(0.04, 1.0, FH);
    float D = GTR1(H.z, material.clearcoatRoughness);
    float G = SmithG(L.z, 0.25)
        * SmithG(V.z, 0.25);
    float jacobian = 1.0 / (4.0 * dot(V, H));

    pdf = D * H.z * jacobian;
    return vec3(0.25) * material.clearcoat * F * D * G / (4.0 * L.z * V.z);
}

void GetSpecColor(Material material, float eta, out vec3 specCol, out vec3 sheenCol)
{
    float lum = Luminance(material.baseColor);
    vec3 ctint = lum > 0.0 ? material.baseColor / lum : vec3(1.0f);
    float F0 = (1.0 - eta) / (1.0 + eta);
    specCol = mix(F0 * F0 * mix(vec3(1.0), ctint, material.specularTint), material.baseColor, material.metallic);
    sheenCol = mix(vec3(1.0), ctint, material.sheenTint);
}

void GetLobeProbabilities(Material material, float eta, vec3 specCol, float approxFresnel, out float diffuseWt, out float specReflectWt, out float specRefractWt, out float clearcoatWt)
{
    diffuseWt = Luminance(material.baseColor) * (1.0 - material.metallic) * (1.0 - material.specTrans);
    specReflectWt = Luminance(mix(specCol, vec3(1.0), approxFresnel));
    specRefractWt = (1.0 - approxFresnel) * (1.0 - material.metallic) * material.specTrans * Luminance(material.baseColor);
    clearcoatWt = material.clearcoat * (1.0 - material.metallic);
    float totalWt = diffuseWt + specReflectWt + specRefractWt + clearcoatWt;

    diffuseWt /= totalWt;
    specReflectWt /= totalWt;
    specRefractWt /= totalWt;
    clearcoatWt /= totalWt;
}

vec3 DisneySample(HitRecord rec, vec3 V, vec3 N, out vec3 L, out float pdf)
{
    pdf = 0.0;
    vec3 f = vec3(0.0);

    float r1 = rand();
    float r2 = rand();

    vec3 T, B;
    Onb(N, T, B);
    V = ToLocal(T, B, N, V); // NDotL = L.z; NDotV = V.z; NDotH = H.z

    // Specular and sheen color
    vec3 specCol, sheenCol;
    GetSpecColor(rec.material, rec.eta, specCol, sheenCol);

    // Lobe weights
    float diffuseWt, specReflectWt, specRefractWt, clearcoatWt;
    // TODO: Recheck fresnel. Not sure if correct. VDotN produces fireflies with rough dielectric.
    // VDotH material Mitsuba and gets rid of all fireflies but H isn't available at this stage
    float approxFresnel = FresnelMix(rec.material, rec.eta, V.z);
    GetLobeProbabilities(rec.material, rec.eta, specCol, approxFresnel, diffuseWt, specReflectWt, specRefractWt, clearcoatWt);

    // CDF for picking a lobe
    float cdf[4];
    cdf[0] = diffuseWt;
    cdf[1] = cdf[0] + specReflectWt;
    cdf[2] = cdf[1] + specRefractWt;
    cdf[3] = cdf[2] + clearcoatWt;

    if (r1 < cdf[0]) // Diffuse Reflection Lobe
    {
        r1 /= cdf[0];
        L = CosineSampleHemisphere(r1, r2);

        vec3 H = normalize(L + V);

        f = EvalDiffuse(rec.material, sheenCol, V, L, H, pdf);
        pdf *= diffuseWt;
    }
    else if (r1 < cdf[1]) // Specular Reflection Lobe
    {
        r1 = (r1 - cdf[0]) / (cdf[1] - cdf[0]);
        vec3 H = SampleGGXVNDF(V, rec.material.roughness, r1, r2);

        if (H.z < 0.0)
            H = -H;

        L = normalize(reflect(-V, H));

        f = EvalSpecReflection(rec.material, rec.eta, specCol, V, L, H, pdf);
        pdf *= specReflectWt;
    }
    else if (r1 < cdf[2]) // Specular Refraction Lobe
    {
        r1 = (r1 - cdf[1]) / (cdf[2] - cdf[1]);
        vec3 H = SampleGGXVNDF(V, rec.material.roughness, r1, r2);

        if (H.z < 0.0)
            H = -H;

        L = normalize(refract(-V, H, rec.eta));

        f = EvalSpecRefraction(rec.material, rec.eta, V, L, H, pdf);
        pdf *= specRefractWt;
    }
    else // Clearcoat Lobe
    {
        r1 = (r1 - cdf[2]) / (1.0 - cdf[2]);
        vec3 H = SampleGTR1(rec.material.clearcoatRoughness, r1, r2);

        if (H.z < 0.0)
            H = -H;

        L = normalize(reflect(-V, H));

        f = EvalClearcoat(rec.material, V, L, H, pdf);
        pdf *= clearcoatWt;
    }

    L = ToWorld(T, B, N, L);
    return f * abs(dot(N, L));
}

vec3 DisneyEval(HitRecord rec, vec3 V, vec3 N, vec3 L, out float bsdfPdf)
{
    bsdfPdf = 0.0;
    vec3 f = vec3(0.0);

    vec3 T, B;
    Onb(N, T, B);
    V = ToLocal(T, B, N, V); // NDotL = L.z; NDotV = V.z; NDotH = H.z
    L = ToLocal(T, B, N, L);

    vec3 H;
    if (L.z > 0.0)
        H = normalize(L + V);
    else
        H = normalize(L + V * rec.eta);

    if (H.z < 0.0)
        H = -H;

    // Specular and sheen color
    vec3 specCol, sheenCol;
    GetSpecColor(rec.material, rec.eta, specCol, sheenCol);

    // Lobe weights
    float diffuseWt, specReflectWt, specRefractWt, clearcoatWt;
    float fresnel = FresnelMix(rec.material, rec.eta, dot(V, H));
    GetLobeProbabilities(rec.material, rec.eta, specCol, fresnel, diffuseWt, specReflectWt, specRefractWt, clearcoatWt);

    float pdf;

    // Diffuse
    if (diffuseWt > 0.0 && L.z > 0.0)
    {
        f += EvalDiffuse(rec.material, sheenCol, V, L, H, pdf);
        bsdfPdf += pdf * diffuseWt;
    }

    // Specular Reflection
    if (specReflectWt > 0.0 && L.z > 0.0 && V.z > 0.0)
    {
        f += EvalSpecReflection(rec.material, rec.eta, specCol, V, L, H, pdf);
        bsdfPdf += pdf * specReflectWt;
    }

    // Specular Refraction
    if (specRefractWt > 0.0 && L.z < 0.0)
    {
        f += EvalSpecRefraction(rec.material, rec.eta, V, L, H, pdf);
        bsdfPdf += pdf * specRefractWt;
    }

    // Clearcoat
    if (clearcoatWt > 0.0 && L.z > 0.0 && V.z > 0.0)
    {
        f += EvalClearcoat(rec.material, V, L, H, pdf);
        bsdfPdf += pdf * clearcoatWt;
    }

    return f * abs(L.z);
}


// DirectLight
vec3 DirectLight(in Ray r, in HitRecord rec)
{
    vec3 Li = vec3(0.0);
    vec3 surfacePos = rec.pos + rec.normal * EPS;

    BsdfSampleRec bsdfSampleRec;

//#define ENVMAP
    // Environment Light
#ifdef ENVMAP
#ifndef CONSTANT_BG
    {
        vec3 color;
        vec4 dirPdf = SampleEnvMap(color);
        vec3 lightDir = dirPdf.xyz;
        float lightPdf = dirPdf.w;

        Ray shadowRay = Ray(surfacePos, lightDir);
        bool inShadow = AnyHit(shadowRay, INF - EPS);

        if (!inShadow)
        {
            bsdfSampleRec.f = DisneyEval(rec, -r.direction, rec.ffnormal, lightDir, bsdfSampleRec.pdf);

            if (bsdfSampleRec.pdf > 0.0)
            {
                float misWeight = PowerHeuristic(lightPdf, bsdfSampleRec.pdf);
                if (misWeight > 0.0)
                    Li += misWeight * bsdfSampleRec.f * color / lightPdf;
            }
        }
    }
#endif
#endif

    // Analytic Lights 
    {
        LightSampleRec lightSampleRec;

        //Pick a light to sample
        int index = int(rand() * float(numOfLights));

        Light light = lights[index];

        light = Light(light.position, light.emission, light.u, light.v, light.radius, light.area, light.type);
        SampleOneLight(light, surfacePos, lightSampleRec);

        if (dot(lightSampleRec.direction, lightSampleRec.normal) < 0.0) // Required for quad lights with single sided emission
        {
            Ray shadowRay = Ray(surfacePos, lightSampleRec.direction);
            bool inShadow = getSceneHit(shadowRay, true).hit;//AnyHit(shadowRay, lightSampleRec.dist - EPS);

            if (!inShadow) {
                bsdfSampleRec.f = DisneyEval(rec, -r.direction, rec.ffnormal, lightSampleRec.direction, bsdfSampleRec.pdf);

                float weight = 1.0;
                if(light.area > 0.0) // No MIS for distant light
                    weight = PowerHeuristic(lightSampleRec.pdf, bsdfSampleRec.pdf);

                if (bsdfSampleRec.pdf > 0.0)
                    Li += weight * bsdfSampleRec.f * lightSampleRec.emission / lightSampleRec.pdf;
            }
        }
    }

    return Li;
}

// Path tracer

vec3 PathTrace(Ray r)
{
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);
    LightSampleRec lightSampleRec;
    BsdfSampleRec bsdfSampleRec;
    
    initLights();
    numOfLights = NUM_LIGHTS;

    const int maxDepth = 4;
    for (int depth = 0; depth < maxDepth; depth++)
    {
        HitRecord rec = getSceneHit(r, false);

        if (!rec.hit) {
            radiance += getBackground(r) * throughput;
        } else {        
            rec.ffnormal = dot(rec.normal, r.direction) <= 0.0 ? rec.normal : -rec.normal;
            Onb(rec.normal, rec.tangent, rec.bitangent);
            rec.material.roughness = max(rec.material.roughness, 0.001);
            rec.eta = dot(rec.normal, rec.ffnormal) > 0.0 ? (1.0 / rec.material.ior) : rec.material.ior;
        }

        radiance += rec.material.emission * throughput;

        if (any(greaterThan(rec.material.emission, vec3(EPS))))
        {
            radiance += EmitterSample(r, depth, lightSampleRec, bsdfSampleRec) * throughput;
            break;
        }

        // Calculate absorption directly if inside medium
        if (dot(rec.normal, rec.ffnormal) < 0.0 && rec.material.specTrans > 0.0) {
            throughput *= exp(-log(rec.material.baseColor) * rec.hitDist);
        }
        radiance += DirectLight(r, rec) * throughput;

        bsdfSampleRec.f = DisneySample(rec, -r.direction, rec.ffnormal, bsdfSampleRec.L, bsdfSampleRec.pdf);

        if (bsdfSampleRec.pdf > 0.0)
            throughput *= bsdfSampleRec.f / bsdfSampleRec.pdf;
        else
            break;

        if (depth >= RR_DEPTH)
        {
            float q = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001, 0.95);
            if (rand() > q)
                break;
            throughput /= q;
        }

        r.direction = bsdfSampleRec.L;
        r.origin = rec.pos + r.direction * EPS;
    }

    return radiance;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    InitRNG(fragCoord, iFrame);

    // Camera
    Ray ray = getCameraRay(vec2(rand(), rand()));

    // Pathtrace
    vec3 col = PathTrace(ray);
    col = clamp(col, 0., 5.);

    fragColor = vec4(col, 1.0);
}
