// Disnub-Diamond-Stained-Glass: image pass.
//
// Traces the gem per pixel and composites it over the stained-glass backdrop.

#define CAM_Z      6.0
#define FOV        0.21
#define BG_Z      -3.0
#define BG_GAIN    0.85
#define FRONT_DIM  0.35
#define DISP_EXAG  1.5
#define GEM_A      2.3818
#define GEM_B      0.0121
#define MAX_NODES  8
#define W_MIN      0.004
#define GEPS       1e-4
#define TEPS       1e-5
const vec3 LAMBDA_UM = vec3(0.630, 0.550, 0.460);

vec4 gQ;
vec3 toWorld(vec3 v) { return quatRot(gQ, v); }
vec3 toObj(vec3 v) { return quatRot(vec4(-gQ.xyz, gQ.w), v); }
vec3 keepHue(vec3 c) { float m = max(max(c.r, c.g), c.b); return m > 1.0 ? c / m : c; }

float fresnelD(float cosi, float n1, float n2) {
    float s = n1 / n2 * sqrt(max(1.0 - cosi * cosi, 0.0));
    if (s >= 1.0) return 1.0;
    float cost = sqrt(1.0 - s * s);
    float rs = (n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost);
    float rp = (n1 * cost - n2 * cosi) / (n1 * cost + n2 * cosi);
    return 0.5 * (rs * rs + rp * rp);
}

vec2 screenSt(vec3 P) {
    vec2 uv = P.xy / ((CAM_Z - P.z) * FOV);
    return (uv * iResolution.y + iResolution.xy) * 0.5 / iResolution.xy;
}
#define GLASS_BASE   0.06
#define GLASS_LIGHT  1.4
#define LEAD_COL     0.012
#define MOVE_LIGHT   1
#define LIGHT_SIGMA  0.20
#define LIGHT_COL    vec3(1.0, 0.12, 0.08)
#define LIGHT_PEAK   1.6
#define LIGHT_SPEED  vec2(0.031, 0.022)
float h11(float x) { return fract(sin(x * 12.9898) * 43758.5453); }
vec3 bgAt(vec2 st) {
    float lod = max(log2(float(textureSize(u_backdrop, 0).x) / iResolution.x), 0.0);
    vec3 c = textureLod(u_backdrop, st, lod).rgb;
    vec4 g = textureLod(u_glass, st, lod);
    float lead = g.r, id = g.g * 255.0, edge = g.b, var = g.a * 2.0 - 1.0;
    float h1 = h11(id), h2 = h11(id + 17.0), kind = h11(id + 41.0);
    float t = iTime;
    float lvl;
    if (kind < 0.94) {
        lvl = 0.82 + 0.18 * sin(t * (0.3 + 0.6 * h2) + 6.2832 * h1);
    } else if (kind < 0.97) {
        lvl = 0.08 + 0.9 * pow(max(sin(t * (0.2 + 0.4 * h2) + 6.2832 * h1), 0.0), 12.0);
    } else {
        lvl = 0.3 + u_bass * 0.8 + u_onset * 0.4;
    }
    vec2 q = st - 0.5;
    lvl *= 0.92 + 0.08 * sin(length(q) * 22.0 - t * 0.7);
    vec3 lightMul = vec3(1.0);
#if MOVE_LIGHT
    vec2 lp = abs(fract(t * LIGHT_SPEED + vec2(0.13, 0.57)) * 2.0 - 1.0) * 0.8 + 0.1;
    vec2 dl = st - lp;
    lightMul += LIGHT_COL * (LIGHT_PEAK * exp(-dot(dl, dl) / (2.0 * LIGHT_SIGMA * LIGHT_SIGMA)));
#endif
    float tone = mix(0.7, 1.15, smoothstep(0.0, 0.6, edge)) * (1.0 + 0.35 * var);
    vec3 lit = c * (GLASS_BASE + GLASS_LIGHT * lvl * tone * lightMul);
    return mix(lit, vec3(LEAD_COL), lead);
}
vec3 envW(vec3 P, vec3 D) {
    if (D.z < -0.02) return bgAt(screenSt(P + D * ((BG_Z - P.z) / D.z)));
    vec3 Dm = vec3(D.xy, -max(abs(D.z), 0.02));
    return bgAt(screenSt(P + Dm * ((BG_Z - P.z) / Dm.z))) * FRONT_DIM;
}

bool inFace(int j, vec2 x) {
    float r2 = dot(x, x);
    if (j < FACE_SQ) {
        if (r2 > R_TRI * R_TRI) return false;
        float a = -0.5 * R_TRI;
        return x.x >= a && dot(x, vec2(-0.5, 0.8660254)) >= a && dot(x, vec2(-0.5, -0.8660254)) >= a;
    }
    if (j < FACE_PG) {
        if (r2 > R_SQ * R_SQ) return false;
        float a = 0.70710678 * R_SQ;
        return abs(dot(x, vec2(0.70710678, 0.70710678))) <= a && abs(dot(x, vec2(-0.70710678, 0.70710678))) <= a;
    }
    if (r2 > R_PG * R_PG) return false;
    float a = 0.30901699 * R_PG;
    int k = int(x.x <= a) + int(dot(x, vec2(0.30901699, 0.95105652)) <= a) + int(dot(x, vec2(-0.80901699, 0.58778525)) <= a)
          + int(dot(x, vec2(-0.80901699, -0.58778525)) <= a) + int(dot(x, vec2(0.30901699, -0.95105652)) <= a);
    return k == 4;
}
float soupHit(vec3 o, vec3 d, out int jH, out int cnt) {
    float tB = 1e9; jH = -1; cnt = 0;
    for (int j = 0; j < FACE_N; j++) {
        vec4 P = FACE_NW[j];
        float dn = dot(d, P.xyz);
        if (abs(dn) < 1e-7) continue;
        float t = (P.w - dot(o, P.xyz)) / dn;
        if (t <= GEPS || t > tB + TEPS) continue;
        vec3 q = o + d * t;
        if (!inFace(j, vec2(dot(q, FACE_U[j].xyz), dot(q, FACE_V[j].xyz)))) continue;
        if (t < tB - TEPS) { tB = t; jH = j; cnt = 1; } else cnt++;
    }
    return tB;
}

vec3 traceCh(vec3 o, vec3 d, int ch) {
    float nI = GEM_A + GEM_B * DISP_EXAG / (LAMBDA_UM[ch] * LAMBDA_UM[ch]);
    vec3 acc = vec3(0.0), w = vec3(1.0);
    bool inD = false, first = true;
    for (int n = 0; n < MAX_NODES; n++) {
        if (max(max(w.r, w.g), w.b) < W_MIN) break;
        int j, cnt;
        float t = soupHit(o, d, j, cnt);
        if (j < 0) {
            acc += w * envW(toWorld(o), toWorld(d));
            return acc;
        }
        vec3 p = o + d * t;
        if ((cnt & 1) == 0) { o = p + d * GEPS; continue; }
        vec3 nrm = FACE_NW[j].xyz;
        if (dot(d, nrm) > 0.0) nrm = -nrm;
        float n1 = inD ? nI : 1.0, n2 = inD ? 1.0 : nI;
        float cosi = -dot(d, nrm);
        float R = fresnelD(cosi, n1, n2);
        vec3 dR = reflect(d, nrm);
        if (first) { acc += w * R * envW(toWorld(p), toWorld(dR)); first = false; }
        if (R > 0.5) {
            if (inD && R < 1.0) acc += w * (1.0 - R) * envW(toWorld(p), toWorld(refract(d, nrm, n1 / n2)));
            if (R < 1.0) w *= R;
            d = dR; o = p + nrm * GEPS;
            continue;
        }
        w *= 1.0 - R;
        d = refract(d, nrm, n1 / n2); o = p - nrm * GEPS;
        inD = !inD;
    }
    acc += w * envW(toWorld(o), toWorld(d));
    return acc;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    gQ = normalize(texelFetch(iChannel0, ivec2(0, 0), 0));
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0.0, 0.0, CAM_Z);
    vec3 rd = normalize(vec3(uv * FOV, -1.0));
    vec3 bg = bgAt(fragCoord / iResolution.xy);
    float b = dot(ro, rd), h = b * b - (dot(ro, ro) - STONE_RB * STONE_RB);
    if (h <= 0.0) { fragColor = vec4(bg, 1.0); return; }
    vec3 o = toObj(ro + rd * (-b - sqrt(h) - 0.01)), d = toObj(rd);
    int j, cnt;
    soupHit(o, d, j, cnt);
    if (j < 0) { fragColor = vec4(bg, 1.0); return; }
    vec3 col;
    for (int ch = 0; ch < 3; ch++) col[ch] = traceCh(o, d, ch)[ch];
    fragColor = vec4(keepHue(max(col, 0.0)), 1.0);
}
