
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
    float atDistance;
    vec3 extinction;
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

struct State
{    
    int depth;
    float eta;
    float hitDist;

    vec3 fhp;
    vec3 normal;
    vec3 ffnormal;
    vec3 tangent;
    vec3 bitangent;

    bool isEmitter;

    vec2 texCoord;
    int matID;
    Material mat;
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
    return vec3(0);
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
bool getSceneHit(Ray ray, bool shadowRay, inout State state) {
                    
    float t = 0.001;
    
    // Analytical floor
    float groundDist = (0. - ray.origin.y) / ray.direction.y;
    float matId = -1.;

    bool hit = groundDist > 0. ? true : false;

    // Raymarch the rest 
    for(int i = 0; i < 120; ++i) {
        vec3 p = ray.origin + ray.direction * t;
        
        vec2 d = map(p);
        float ad = abs(d.x);

        if (ad < (0.0001)) {
            hit = true;
            matId = d.y;
            break;
         }
            
         t += ad;
         
         if (t>27.0) { break; }
    }
    
    if (hit) {

        if ( (groundDist > 0. && groundDist < t) || matId < 0.5 ) {

            // Ground
            state.mat.baseColor = vec3(1, 0, 0);
            state.mat.roughness = 0.5;
            state.mat.metallic = 0.2;
                
            state.fhp = ray.origin + ray.direction * groundDist;
            state.normal = vec3(0, 1, 0);

            // 70s Wallpaper from Shane, https://www.shadertoy.com/view/ls33DN
            vec2 p = state.fhp.xz;
            p.x *= sign(cos(length(ceil(p /= 2.))*99.));
    
            float f = clamp(cos(min(length(p = fract(p)), length(--p))*44.), 0., 1.);
            
            f = clamp(f, 0., 1.);
            
            state.mat.clearcoat = f;
            state.mat.clearcoatRoughness = f;
            state.mat.baseColor = mix(state.mat.baseColor, 
                        vec3(1.0, 0.71, 0.29), f);
                        
            state.hitDist = groundDist;
        } else {
        
            state.fhp = ray.origin + ray.direction * t;
        
            // Glass
            if (matId > 0.5 && matId < 1.5) {
                state.mat.baseColor = vec3(2);
                state.mat.specTrans = 1.;
                state.mat.roughness = 0.0;

            } else
            // Red
            if (matId > 1.5 && matId < 2.5) {
                state.mat.baseColor = vec3(1, 0, 0);
                state.mat.roughness = 0.5;
                state.mat.metallic = 0.2;
            } else
            // Orange
            if (matId > 2.5 && matId < 3.5) {
                state.mat.baseColor = vec3(1, 0.186, 0.);
                state.mat.roughness = 0.001;
                state.mat.clearcoat = 1.0;
                state.mat.clearcoatRoughness = 1.0;
            } else
            // Ping
            if (matId > 3.5 && matId < 4.5) {
                state.mat.baseColor = vec3(0.93, 0.89, 0.85);
                state.mat.roughness = 1.;
                state.mat.subsurface = 1.0;
            } else
            // Silver
            if (matId > 4.5 && matId < 5.5) {
                state.mat.baseColor = vec3(0.9, 0.9, 0.9);
                state.mat.roughness = 0.0;
                state.mat.metallic = 1.;
            } else
            // Marble
            if (matId > 5.5 && matId < 6.5) {
                state.mat.baseColor = vec3(0.099, 0.24, 0.134);
                state.mat.roughness = 0.001;
                state.mat.clearcoat = 1.0;
                state.mat.clearcoatRoughness = 1.0;
            }
            
            if (shadowRay == false) {
                state.normal = calculateNormal(state.fhp);
            }
            
            state.hitDist = t;
        }
        
        // Hack for enabling transparent reflections
        if (shadowRay == true && state.mat.specTrans > 0.5) hit = false;
    }
    
    return hit;
}
