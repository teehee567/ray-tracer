
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
    float anisotropic;
    float metallic;
    float roughness;
    float subsurface;
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
    int depth;
    float eta;
    float hitDist;

    vec3 fhp;
    vec3 normal;
    vec3 ffnormal;
    vec3 tangent;
    vec3 bitangent;

    vec2 texCoord;
    int matID;
    Material mat;

    bool hit;
};

#define NUM_LIGHTS 17
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
    float blend = 0.5 * ray.direction.y + 0.5;
    return mix(vec3(0.6, 0.8, 1.0), vec3(0.2, 0.4, 1.0), blend);

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
    obj = opU(vec2(length(p - vec3(2.6, 1.4, 3.6)) - 0.8, 4), obj);

    // Orange
    obj = opU(vec2(length(p - vec3(-0.8, 1.86, -3.59)) - 1., 3), obj);

    // Glass
    obj = opU(vec2(length(p - vec3(1.8, 2.6, -6.59)) - 1.3, 1), obj);

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
    rec.mat.anisotropic  = 0.0;

    rec.mat.metallic     = 0.0;
    rec.mat.roughness    = 0.5;
    rec.mat.subsurface   = 0.0;
    rec.mat.specularTint = 0.0;
            
    rec.mat.sheen        = 0.0;
    rec.mat.sheenTint    = 0.0;
    rec.mat.clearcoat    = 0.0;
    rec.mat.clearcoatRoughness = 0.0;
            
    rec.mat.roughness    = 0.;
    rec.mat.ior          = 1.45;


    float t = 0.001;
    
    // Analytical floor
    float groundDist = (0. - ray.origin.y) / ray.direction.y;
    float matId = -1.;

    rec.hit = groundDist > 0. ? true : false;

    // Raymarch the rest 
    for(int i = 0; i < 120; ++i) {
        vec3 p = ray.origin + ray.direction * t;
        
        vec2 d = map(p);
        float ad = abs(d.x);

        if (ad < (0.0001)) {
            rec.hit = true;
            matId = d.y;
            break;
         }
            
         t += ad;
         
         if (t>27.0) { break; }
    }
    
    if (rec.hit) {

        if ( (groundDist > 0. && groundDist < t) || matId < 0.5 ) {

            // Ground
            rec.mat.baseColor = vec3(1, 0, 0);
            rec.mat.roughness = 0.5;
            rec.mat.metallic = 0.8;
                
            rec.fhp = ray.origin + ray.direction * groundDist;
            rec.normal = vec3(0, 1, 0);

            // 70s Wallpaper from Shane, https://www.shadertoy.com/view/ls33DN
            vec2 p = rec.fhp.xz;
            p.x *= sign(cos(length(ceil(p /= 2.))*99.));
    
            float f = clamp(cos(min(length(p = fract(p)), length(--p))*44.), 0., 1.);
            
            f = clamp(f, 0., 1.);
            
            rec.mat.clearcoat = f;
            rec.mat.clearcoatRoughness = f;
            rec.mat.baseColor = mix(rec.mat.baseColor, 
                        vec3(1.0, 0.71, 0.29), f);
                        
            rec.hitDist = groundDist;
        } else {
        
            rec.fhp = ray.origin + ray.direction * t;
        
            // Glass
            if (matId > 0.5 && matId < 1.5) {
                rec.mat.baseColor = vec3(1.0);  // Pure white
rec.mat.metallic = 1.0;         // Fully metallic
rec.mat.roughness = 0.0;        // Perfectly smooth
rec.mat.specTrans = 0.0;        // No transmission
rec.mat.clearcoat = 0.0;        // No clearcoat
rec.mat.subsurface = 0.0;       // No subsurface scattering

// Optional: slight tint for realism
rec.mat.baseColor = vec3(0.95, 0.95, 0.98);  // Slightly bluish


            } else
            // Red
            if (matId > 1.5 && matId < 2.5) {
                rec.mat.baseColor = vec3(1, 1, 1);
                rec.mat.roughness = 0.;
                rec.mat.metallic = 1.;
            } else
            // Orange
            if (matId > 2.5 && matId < 3.5) {
                rec.mat.baseColor = vec3(1, 0.186, 0.);
                rec.mat.roughness = 0.001;
                rec.mat.clearcoat = 1.0;
                rec.mat.clearcoatRoughness = 1.0;
            } else
            // Ping
            if (matId > 3.5 && matId < 4.5) {
                rec.mat.baseColor = vec3(0.93, 0., 0.85);
                rec.mat.roughness = 1.;
                rec.mat.subsurface = 1.0;
                rec.mat.emission = vec3(200000, 200000, 200000);
            } else
            // Silver
            if (matId > 4.5 && matId < 5.5) {
                rec.mat.baseColor = vec3(0.9, 0.9, 0.9);
                rec.mat.roughness = 0.0;
                rec.mat.metallic = 1.;

            } else
            // Marble
            if (matId > 5.5 && matId < 6.5) {
                rec.mat.baseColor = vec3(0.099, 0.24, 0.134);
                rec.mat.roughness = 0.001;
                rec.mat.clearcoat = 1.0;
                rec.mat.clearcoatRoughness = 1.0;
            }
            
            if (shadowRay == false) {
                rec.normal = calculateNormal(rec.fhp);
            }
            
            rec.hitDist = t;
        }
        
        // Hack for enabling transparent reflections
        if (shadowRay == true && rec.mat.specTrans > 0.5) rec.hit = false;
    }
    
    return rec;
}
