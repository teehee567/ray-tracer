
#define PI         3.14159265358979323
#define INV_PI     0.31830988618379067
#define TWO_PI     6.28318530717958648
#define INV_TWO_PI 0.15915494309189533
#define INV_4_PI   0.07957747154594766
#define EPS 0.001
#define INF 1000000.0

#define QUAD_LIGHT 0
#define SPHERE_LIGHT 1
#define DISTANT_LIGHT 2

#define LIGHTS
//#define RR
#define RR_DEPTH 2


struct Ray
{
    vec3 origin;
    vec3 direction;
};

struct Material
{
    vec3 baseColor;
    vec3 emission;
    float metallic;
    float roughness;
    float subsurface;
    float anisotropic;
    float specularTint;
    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatRoughness;
    float specTrans;
    float ior;
};

struct Light
{
    vec3 position;
    vec3 emission;
    vec3 u;
    vec3 v;
    float radius;
    float area;
    int type;
};

struct HitRecord
{    
    float eta;
    float hit_dist;

    vec3 pos;
    vec3 normal;
    vec3 ffnormal;
    vec3 tangent;
    vec3 bitangent;

    vec2 uv;
    Material material;

    bool did_hit;
};

#define NUM_LIGHTS 10
Light lights[NUM_LIGHTS];

void initLights() {

    for(int i = 0; i < NUM_LIGHTS; i +=1) {
        float f = float(i);
        
        vec3 p = vec3(-2.04973, 5., -8. + f);
  
        vec3 u = vec3(2.040, 5., -8. + f) - p;
        vec3 v = vec3(-2.04973, 5., -7.5 + f) - p;
        

        float area = length(cross(u, v));
        
        lights[i] = Light( p,         // Position
                           vec3(5),   // Emission
                           u,         // u, only for rect lights
                           v,         // v, only for rect lights
                           0.,        // Radius for sphere light, 
                           area,      // area  
                           0);        // type: 0 - rect, 1 - sphere, 2 - dist
    
    }
}

// Get the camera position and lookAt
void getCameraPos(inout vec3 origin, inout vec3 lookAt) {
    origin = vec3(15., 15, 0.);
    lookAt = vec3(0., 0., 0.);
}

// Get the scene background color
vec3 getBackground(Ray ray) {
    //float blend = 0.5 * ray.direction.y + 0.5;
    //return mix(vec3(0.6, 0.8, 1.0), vec3(0.2, 0.4, 1.0), blend);
    return vec3(0.2);

}

// Map

float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

vec2 opU(vec2 o1, vec2 o2) {
    return (o1.x < o2.x) ? o1 : o2;
}

vec2 map(vec3 p) {
    
    vec2 obj = vec2(sdBox(p - vec3(0, 0.4, 0), vec3(5, 0.3, 9)), 2.);
    
    // Silver
    obj = opU(vec2(length(p - vec3(-0.57, 2.2, 6.55)) - 1.2, 5), obj);

    // Ping
    obj = opU(vec2(length(p - vec3(2.6, 5., 3.6)) - 0.8, 4), obj);

    // Orange
    obj = opU(vec2(length(p - vec3(-0.8, 1.86, -3.59)) - 1., 3), obj);

    // Glass
    obj = opU(vec2(length(p - vec3(1.8, 1.6, 2.59)) - 1.3, 1), obj);

    // Marble
    obj = opU(vec2(length(p - vec3(3.5, 1.2, -1.5)) - 0.6, 6), obj);
    
    return obj;
}

vec3 calculateNormal(vec3 p) {
 
    vec3 epsilon = vec3(0.001, 0., 1.);
    
    vec3 n = vec3(map(p + epsilon.xyy).x - map(p - epsilon.xyy).x,
                  map(p + epsilon.yxy).x - map(p - epsilon.yxy).x,
                  map(p + epsilon.yyx).x - map(p - epsilon.yyx).x);
    
    return normalize(n);
}

// Get the scene hit record
HitRecord getSceneHit(Ray ray, bool shadowRay) {

    HitRecord rec;
    rec.material.anisotropic  = 0.0;

    rec.material.metallic     = 0.0;
    rec.material.roughness    = 0.5;
    rec.material.subsurface   = 0.0;
    rec.material.specularTint = 0.0;
            
    rec.material.sheen        = 0.0;
    rec.material.sheenTint    = 0.0;
    rec.material.clearcoat    = 0.0;
    rec.material.clearcoatRoughness = 0.0;
            
    rec.material.roughness    = 0.;
    rec.material.ior          = 1.45;


    float t = 0.001;
    
    // Analytical floor
    float groundDist = (0. - ray.origin.y) / ray.direction.y;
    float material = -1.;

    rec.did_hit = groundDist > 0. ? true : false;

    // Raymarch the rest 
    for(int i = 0; i < 120; ++i) {
        vec3 p = ray.origin + ray.direction * t;
        
        vec2 d = map(p);
        float ad = abs(d.x);

        if (ad < (0.0001)) {
            rec.did_hit = true;
            material = d.y;
            break;
         }
            
         t += ad;
         
         if (t>27.0) { break; }
    }
    
    if (rec.did_hit) {

        if ( (groundDist > 0. && groundDist < t) || material < 0.5 ) {

            // Ground
            rec.material.baseColor = vec3(1, 0, 0);
            rec.material.roughness = 0.5;
            rec.material.metallic = 0.;
                
            rec.pos = ray.origin + ray.direction * groundDist;
            rec.normal = vec3(0, 1, 0);

            // 70s Wallpaper from Shane, https://www.shadertoy.com/view/ls33DN
            vec2 p = rec.pos.xz;
            p.x *= sign(cos(length(ceil(p /= 2.))*99.));
    
            float f = clamp(cos(min(length(p = fract(p)), length(--p))*44.), 0., 1.);
            
            f = clamp(f, 0., 1.);
            
            rec.material.clearcoat = f;
            rec.material.clearcoatRoughness = f;
            rec.material.baseColor = mix(rec.material.baseColor, 
                        vec3(1.0, 0.71, 0.29), f);
                        
            rec.hitDist = groundDist;
        } else {
        
            rec.pos = ray.origin + ray.direction * t;
        
            // Glass
            if (material > 0.5 && material < 1.5) {
    rec.material.baseColor = vec3(1); // Example: Reddish glass
    // Or vec3(0.95, 0.95, 1.0) for slightly blue-tinted clear glass
    rec.material.metallic = 0.0;
    rec.material.roughness = 0.0;
    rec.material.specTrans = 1.0;  // Fully transmissive
    rec.material.ior = 7.;

            } else
            // Red
            if (material > 1.5 && material < 2.5) {
                rec.material.baseColor = vec3(1, 1, 1);
                rec.material.roughness = 1.;
                rec.material.metallic = 0.;
            } else
            // Orange
            if (material > 2.5 && material < 3.5) {
                rec.material.baseColor = vec3(1, 0.186, 0.);
                rec.material.roughness = 1.0;
                rec.material.clearcoat = 1.0;
                rec.material.clearcoatRoughness = 1.0;
            } else
            // Ping
            if (material > 3.5 && material < 4.5) {
                rec.material.baseColor = vec3(0.93, 0., 0.85);
                rec.material.roughness = 1.;
                rec.material.subsurface = 1.0;
                rec.material.emission = vec3(1);
            } else
            // Silver
            if (material > 4.5 && material < 5.5) {
                rec.material.baseColor = vec3(0.9, 0.9, 0.9);
                rec.material.roughness = 0.0;
                rec.material.metallic = 1.;

            } else
            // Marble
            if (material > 5.5 && material < 6.5) {
                rec.material.baseColor = vec3(0.099, 0.24, 0.134);
                rec.material.roughness = 0.001;
                rec.material.clearcoat = 1.0;
                rec.material.clearcoatRoughness = 1.0;
            }
            
            if (shadowRay == false) {
                rec.normal = calculateNormal(rec.pos);
            }
            
            rec.hitDist = t;
        }
        
        // Hack for enabling transparent reflections
        if (shadowRay == true && rec.material.specTrans > 0.5) rec.did_hit = false;
    }
    
    return rec;
}
