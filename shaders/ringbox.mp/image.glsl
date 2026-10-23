// Ringbox: image pass.
//
// A small honest physics toy: a star and three moons (REBOUND N-body on the CPU)
// inside a big box, with a ring of massless grains integrated on the GPU
// (ring.comp). Moons merge and heat up on contact, the star swallows what hits
// it. Two ships work as a team to keep the moons in their slots: lasers push by
// light pressure, nukes ablate material that becomes ring grains. The sun is
// textured from real observed maps (photosphere + 304 A chromosphere).
// Input: a DroidPad pad (contrib/pads/ringbox.json) and the mouse hand.

uniform usampler2D u_acc;
uniform sampler2D u_moonMap;
uniform sampler2D u_sunPhoto, u_sunEuv;
uniform vec3 u_ro, u_fw, u_rt, u_up;
uniform float u_fov;
uniform vec4 u_star;
uniform float u_starLum;
uniform int u_starMode;
uniform float u_flare;
uniform float u_flareAge;
uniform vec3 u_flareDir;
uniform vec4 u_m0, u_m1, u_m2, u_m3;
uniform vec4 u_mh;
uniform float u_splatK;
uniform vec4 u_sp0, u_sp1;
uniform vec4 u_sa0, u_sa1;
uniform vec4 u_shv0, u_shv1;
uniform vec4 u_bm0, u_bm1;
uniform vec4 u_nk0, u_nk1, u_nk2, u_nk3;
uniform vec4 u_fx0, u_fx1, u_fx2, u_fx3;
uniform vec4 u_fxs;

#define RING_GAIN   0.030
#define RING_COL    vec3(0.72, 0.84, 1.00)
#define BLUR_R      2
#define STAR_COL    vec3(0.94, 0.97, 1.00)
#define STAR_LIGHT  vec3(0.90, 0.95, 1.00)
#define STAR_GLOW   0.020
#define UV_COL      vec3(0.32, 0.50, 1.00)
#define UV_GLOW     0.060
#define UV_H        0.42
#define MOON_AMB    0.035

vec3 starBaseM(int m) {
    if (m == 1) return vec3(1.72, 0.78, 0.20);
    if (m == 2) return vec3(1.18, 0.095, 0.025);
    return vec3(1.48, 1.54, 1.68);
}
vec3 starActiveM(int m) {
    if (m == 1) return vec3(1.85, 1.05, 0.32);
    if (m == 2) return vec3(1.55, 0.30, 0.075);
    return vec3(1.40, 1.48, 1.62);
}
vec3 starPenM(int m) {
    if (m == 1) return vec3(1.15, 0.55, 0.15);
    if (m == 2) return vec3(0.82, 0.065, 0.015);
    return vec3(0.95, 1.05, 1.22);
}
vec3 starGlowM(int m) {
    if (m == 1) return vec3(1.00, 0.78, 0.42);
    if (m == 2) return vec3(1.00, 0.24, 0.09);
    return vec3(0.94, 0.97, 1.00);
}
vec3 starLightM(int m) {
    if (m == 1) return vec3(1.00, 0.78, 0.55);
    if (m == 2) return vec3(1.00, 0.27, 0.12);
    return vec3(0.90, 0.95, 1.00);
}
vec3 ringM(int m) {
    if (m == 1) return vec3(1.00, 0.78, 0.55);
    if (m == 2) return vec3(0.85, 0.23, 0.12);
    return vec3(0.72, 0.84, 1.00);
}
vec3 limbM(int m) {
    if (m == 1) return vec3(1.00, 0.48, 0.12);
    if (m == 2) return vec3(0.58, 0.055, 0.015);
    return vec3(0.74, 0.84, 1.00);
}
vec3 uvM(int m) {
    if (m == 1) return vec3(1.00, 0.55, 0.20);
    if (m == 2) return vec3(1.00, 0.12, 0.035);
    return vec3(0.32, 0.50, 1.00);
}
vec3 coronaM(int m) {
    if (m == 1) return vec3(1.00, 0.55, 0.18);
    if (m == 2) return vec3(1.00, 0.14, 0.04);
    return vec3(0.82, 0.90, 1.00);
}
float uvGlowM(int m) {
    return m == 0 ? 0.060 : (m == 1 ? 0.010 : 0.006);
}

float sphere(vec3 ro, vec3 rd, vec3 c, float r) {
    vec3 oc = ro - c;
    float b = dot(oc, rd), q = dot(oc, oc) - r * r, d = b * b - q;
    if (d < 0.0) return -1.0;
    float t = -b - sqrt(d);
    return t > 0.0 ? t : -1.0;
}
vec3 heatCol(float T) {
    float x = clamp((T - 0.45) / 0.75, 0.0, 1.2);
    return vec3(1.0, 0.35, 0.06) * smoothstep(0.0, 0.5, x) * 2.0 + vec3(0.0, 0.45, 0.25) * smoothstep(0.4, 1.0, x) * 2.0
         + vec3(0.0, 0.0, 0.35) * smoothstep(0.9, 1.2, x);
}
vec3 toScreen(vec3 x, vec2 res) {
    vec3 rel = x - u_ro;
    float z = dot(rel, u_fw);
    vec2 uv = vec2(dot(rel, u_rt), dot(rel, u_up)) / max(z, 1e-4);
    return vec3(uv / (2.0 * u_fov) * min(res.x, res.y) + 0.5 * res, z);
}
float segDist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}
float hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }

#define SF_SPEED    0.035
#define SF_SPIN     0.012
#define SF_GAIN     3.0
#define SF_TINT     vec3(1.30, 0.92, 0.60)
#define LIMB_U1     0.60
#define LIMB_U2     0.12
#define LIMB_COL    vec3(0.74, 0.84, 1.00)
#define CORONA_G    0.9
#define CORONA_H    0.05
float sfHash(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
float sfNoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(sfHash(i), sfHash(i + vec3(1, 0, 0)), u.x), mix(sfHash(i + vec3(0, 1, 0)), sfHash(i + vec3(1, 1, 0)), u.x), u.y),
               mix(mix(sfHash(i + vec3(0, 0, 1)), sfHash(i + vec3(1, 0, 1)), u.x), mix(sfHash(i + vec3(0, 1, 1)), sfHash(i + vec3(1, 1, 1)), u.x), u.y), u.z);
}
#define SHEAR_P     540.0
#define PULSE_S     2.2
float shearAge() { return mod(iTime, SHEAR_P); }
float sunPulse() { float a = shearAge(); float d = min(a, SHEAR_P - a); return exp(-(d / PULSE_S) * (d / PULSE_S)); }
vec3 sunFrame(vec3 n) {
    float lat = asin(clamp(n.z, -1.0, 1.0));
    float spin = mod(SF_SPIN * iTime, 6.28318530718) - SF_SPIN * 0.24 * sin(lat) * sin(lat) * shearAge();
    float cs = cos(spin), sn = sin(spin);
    return vec3(cs * n.x - sn * n.y, sn * n.x + cs * n.y, n.z);
}
vec2 sunUV(vec3 m) {
    return vec2(fract(atan(m.y, m.x) / 6.28318530718 + 0.5), acos(clamp(m.z, -1.0, 1.0)) / 3.14159265359);
}
vec3 starSurface(vec3 n, float t) {
    vec3 m = sunFrame(n);
    vec2 uv = sunUV(m);
    vec3 ph = texture(u_sunPhoto, uv).rgb;
    float euv = dot(texture(u_sunEuv, uv).rgb, vec3(0.299, 0.587, 0.114));
    float lum = dot(ph, vec3(0.299, 0.587, 0.114));
    float act = smoothstep(0.04, 0.45, euv);
    vec3 base = mix(starBaseM(u_starMode) * 0.055, starBaseM(u_starMode),
                    smoothstep(0.02, 0.62, lum));
    base = mix(base, starActiveM(u_starMode), act * 0.35);
    float spot = (1.0 - smoothstep(0.10, 0.28, lum)) * smoothstep(0.02, 0.22, euv);
    base *= 1.0 - 0.60 * spot;
    float pen = smoothstep(0.10, 0.20, lum) * (1.0 - smoothstep(0.24, 0.34, lum)) * smoothstep(0.02, 0.20, euv);
    base += starPenM(u_starMode) * pen * 0.85;
    float n2 = sfNoise(m * 9.0 + vec3(0.0, 0.0, SF_SPEED * t * 2.0));
    base *= 0.88 + 0.24 * n2;
    return max(base, 0.0) * SF_GAIN * 0.39 * u_starLum;
}

vec3 shadeMoon(vec3 p, vec3 c, float r, float T, int id) {
    vec3 n = normalize(p - c);
    vec3 L = normalize(u_star.xyz - p);
    float lam = max(dot(n, L), 0.0);
    vec2 uv = vec2(atan(n.y, n.x) / 6.2831853 + 0.5 + 0.23 * float(id), acos(clamp(n.z, -1.0, 1.0)) / 3.1415926);
    vec3 alb = pow(texture(u_moonMap, uv).rgb, vec3(2.2));
    vec3 col = alb * starLightM(u_starMode) * (lam * 1.4 * u_starLum * (1.0 + 0.8 * u_flare) / max(dot(u_star.xyz - c, u_star.xyz - c), 0.25) + MOON_AMB);
    float crust = smoothstep(0.35, 0.65, dot(alb, vec3(0.33)));
    float Tsurf = T * mix(1.0, 0.75, crust * smoothstep(1.1, 0.8, T));
    col += heatCol(Tsurf);
    return col;
}

float promLoop(vec2 p, vec2 sd, vec2 perp, float a, float w, float thick) {
    float u = dot(p, sd), v = dot(p, perp);
    float uu = a - (a - 0.93) * (v / max(w, 1e-4)) * (v / max(w, 1e-4));
    float tube = abs(u - uu);
    float band = 1.0 - smoothstep(w * 0.80, w * 1.05, abs(v));
    return exp(-pow(tube / max(thick, 1e-4), 2.0)) * band;
}

#define SHIP_DRAW   0.035
vec2 mapTinyShip(vec3 p) {
    vec3 s = p;
    s.x = abs(s.x);
    float fuseZ = max(-p.z - 1.2, p.z - 1.0);
    float fuseW = mix(0.04, 0.28, clamp((p.z + 1.2) * 0.45, 0.0, 1.0));
    float fuse = (s.x * 0.85 + abs(p.y) * 1.4) - fuseW;
    fuse = max(fuse, fuseZ);
    float wingZ = max(-s.z - 0.2, s.z - 0.95);
    float wingFlare = mix(0.1, 1.35, clamp((s.z + 0.2) * 0.85, 0.0, 1.0));
    float wingPlane = max(s.x - wingFlare, abs(s.y) - 0.025);
    float wings = max(wingPlane, wingZ);
    vec3 finP = s - vec3(0.35, 0.08, 0.6);
    finP.xy *= mat2(0.92, -0.38, 0.38, 0.92);
    float fin = max(max(abs(finP.x) - 0.015, finP.y - 0.32), abs(finP.z) - 0.25);
    fin = max(fin, -finP.y);
    float h = clamp(0.5 + 0.5 * (wings - fuse) / 0.08, 0.0, 1.0);
    float hull = mix(wings, fuse, h) - 0.08 * h * (1.0 - h);
    hull = min(hull, fin);
    vec3 aftP = s - vec3(0.18, 0.0, 1.02);
    float thruster = length(aftP) - 0.06;
    vec2 res = vec2(hull, 1.0);
    if (thruster < res.x) res = vec2(thruster, 2.0);
    return res;
}
vec2 traceShip(vec3 ro, vec3 rd, vec3 c, vec3 fwd, out mat3 B) {
    vec3 zb = -normalize(fwd);
    vec3 yb = vec3(0.0, 0.0, 1.0);
    if (abs(dot(yb, zb)) > 0.95) yb = vec3(1.0, 0.0, 0.0);
    vec3 xb = normalize(cross(yb, zb));
    yb = cross(zb, xb);
    B = mat3(xb, yb, zb);
    float R = SHIP_DRAW * 1.9;
    vec3 oc = ro - c;
    float b = dot(oc, rd), cc = dot(oc, oc) - R * R, disc = b * b - cc;
    if (disc < 0.0) return vec2(-1.0, 0.0);
    float t0 = max(-b - sqrt(disc), 0.0), t1 = -b + sqrt(disc);
    vec3 pm = transpose(B) * (ro + rd * t0 - c) / SHIP_DRAW;
    vec3 dm = transpose(B) * rd;
    float tm = 0.0, tmax = (t1 - t0) / SHIP_DRAW;
    for (int k = 0; k < 64; k++) {
        vec2 d = mapTinyShip(pm + dm * tm);
        if (d.x < 0.002) return vec2(t0 + tm * SHIP_DRAW, d.y);
        tm += max(d.x * 0.7, 0.003);
        if (tm > tmax) break;
    }
    return vec2(-1.0, 0.0);
}
vec3 shipNormal(vec3 pm) {
    const vec2 e = vec2(0.003, -0.003);
    return normalize(e.xyy * mapTinyShip(pm + e.xyy).x + e.yyx * mapTinyShip(pm + e.yyx).x +
                     e.yxy * mapTinyShip(pm + e.yxy).x + e.xxx * mapTinyShip(pm + e.xxx).x);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 res = iResolution.xy;
    float m = min(res.x, res.y);
    vec2 uv = (fragCoord - 0.5 * res) / m * 2.0 * u_fov;
    vec3 rd = normalize(u_fw + uv.x * u_rt + uv.y * u_up);
    vec3 ro = u_ro;

    vec3 col = vec3(0.0);
    vec2 g = floor(fragCoord / 3.0);
    float hs = hash12(g);
    if (hs > 0.9975) col += vec3(0.5, 0.55, 0.7) * (hs - 0.9975) * 400.0 * 0.08;

    float tBest = 1e9; int hit = -1;
    float t = sphere(ro, rd, u_star.xyz, u_star.w);
    if (t > 0.0) { tBest = t; hit = 0; }
    vec4 M[4] = vec4[4](u_m0, u_m1, u_m2, u_m3);
    for (int i = 0; i < 4; i++) {
        if (M[i].w <= 0.0) continue;
        t = sphere(ro, rd, M[i].xyz, M[i].w);
        if (t > 0.0 && t < tBest) { tBest = t; hit = i + 1; }
    }
    if (hit == 0) {
        vec3 p = ro + rd * tBest;
        vec3 n = normalize(p - u_star.xyz);
        float om = 1.0 - max(dot(n, -rd), 0.0);
        float I = max(1.0 - LIMB_U1 * om - LIMB_U2 * om * om, 0.03);
        col = starSurface(n, iTime) * I;
        col *= 1.0 + 0.55 * u_flare;
        float pz = sunPulse();
        col = mix(col, starGlowM(u_starMode) * 2.2, 0.85 * pz);
        col = mix(col, limbM(u_starMode) * length(col), smoothstep(0.35, 1.0, om) * 0.6);
        float edgeBoost = smoothstep(0.40, 0.85, om) * (1.0 - smoothstep(0.85, 0.99, om));
        col += starGlowM(u_starMode) * 1.15 * edgeBoost * pow(clamp(col.r * 0.35, 0.0, 1.0), 3.0) * 0.55;
    } else if (hit > 0) {
        vec3 p = ro + rd * tBest;
        col = shadeMoon(p, M[hit - 1].xyz, M[hit - 1].w, u_mh[hit - 1], hit - 1);
    }
    vec3 oc = u_star.xyz - ro;
    float tc = dot(oc, rd);
    float dmin = length(oc - rd * tc) / max(tc, 1e-3);
    float rs = u_star.w / max(tc, 1e-3);
    col += starGlowM(u_starMode) * STAR_GLOW * rs * rs / max(dmin * dmin, rs * rs * 0.5) * (hit == 0 ? 0.0 : 1.0) * u_starLum * (1.0 + 2.5 * sunPulse());
    float uvGlow = uvGlowM(u_starMode);
    if (tc > 0.0 && hit != 0 && dmin < rs * 3.5 && uvGlow > 0.0) {
        float uvz = max(dmin - rs, 0.0) / max(rs, 1e-6);
        col += uvM(u_starMode) * uvGlow * exp(-uvz / UV_H) * u_starLum;
    }
    if (u_flare > 0.0 && tc > 0.0 && hit != 0) {
        vec3 sc = toScreen(u_star.xyz, res);
        vec2 sd = vec2(dot(u_flareDir, u_rt), dot(u_flareDir, u_up));
        if (dot(sd, sd) < 1e-4) sd = vec2(1.0, 0.0);
        sd = normalize(sd);
        vec2 perp = vec2(-sd.y, sd.x);
        float px = max(rs * 0.5 * min(res.x, res.y) / max(u_fov, 1e-4), 0.5);
        vec2 p = (fragCoord - sc.xy) / px;
        float u = dot(p, sd), v = dot(p, perp);
        float age = u_flareAge;
        float a = 1.25 + 1.80 * age;
        float w = 0.62 + 0.45 * age;
        float loop = 0.0;
        for (int k = 0; k < 2; k++) {
            float fk = float(k);
            float ak = a + 0.18 * fk * age;
            float wk = w * (1.0 + 0.16 * fk);
            float uu = ak - (ak - 0.93) * (v / max(wk, 1e-4)) * (v / max(wk, 1e-4));
            float tube = abs(u - uu);
            float band = 1.0 - smoothstep(wk * 0.80, wk * 1.04, abs(v));
            loop += exp(-pow(tube / 0.055, 2.0)) * band * (1.0 - 0.24 * fk);
        }
        float feet = exp(-pow((u - 0.97) / 0.055, 2.0))
                   * (exp(-pow((v - w) / 0.055, 2.0)) + exp(-pow((v + w) / 0.055, 2.0)));
        float lead = exp(-pow((u - (a + 0.35 + 0.15 * age)) / 0.060, 2.0))
                   * (1.0 - smoothstep(w * 0.20, w * 1.0, abs(v)));
        vec3 hot = starGlowM(u_starMode) + vec3(0.15, 0.20, 0.28);
        float cU = u - (a + 0.55 + 1.10 * age);
        vec2 cq = vec2(cU / 0.60, v / 0.40);
        float bubble = exp(-pow((length(cq) - 0.42) / 0.16, 2.0));
        col += hot * u_flare * (loop * 0.9 + feet * 1.6 + lead * 0.8 + bubble * 0.55);
    }
    if (hit != 0 && tc > 0.0 && dmin < rs * 3.4) {
        vec3 sc = toScreen(u_star.xyz, res);
        float px = max(rs * 0.5 * min(res.x, res.y) / max(u_fov, 1e-4), 0.5);
        vec2 p = (fragCoord - sc.xy) / px;
        vec3 hot = starGlowM(u_starMode) * 0.85 + vec3(0.08, 0.11, 0.16);
        for (int k = 0; k < 12; k++) {
            float fk = float(k);
            float ang = 6.2831853 * (fk / 12.0) + 0.18 * sin(iTime * 0.035 + fk * 1.9);
            vec3 nlimb = normalize(u_rt * cos(ang) + u_up * sin(ang));
            vec2 auv = sunUV(sunFrame(nlimb));
            float euv = dot(texture(u_sunEuv, auv).rgb, vec3(0.299, 0.587, 0.114));
            float act = smoothstep(0.035, 0.30, euv);
            if (act <= 0.0) continue;
            float life = 0.5 + 0.5 * sin(iTime * 0.030 + fk * 1.7 + 0.6 * sin(iTime * 0.011 + fk));
            float a = 1.08 + 0.28 * life;
            float w = 0.30 + 0.10 * life;
            vec2 sd = vec2(cos(ang), sin(ang));
            vec2 perp = vec2(-sd.y, sd.x);
            float u = dot(p, sd), v = dot(p, perp);
            float uu = a - (a - 0.93) * (v / max(w, 1e-4)) * (v / max(w, 1e-4));
            float tube = abs(u - uu);
            float n1 = sfNoise(vec3(u * 6.0, v * 6.0, iTime * 0.05 + fk * 3.0));
            float thick = 0.034 * (0.65 + 0.70 * n1);
            float band = 1.0 - smoothstep(w * 0.80, w * 1.05, abs(v));
            float core = exp(-pow(tube / max(thick, 1e-4), 2.0));
            float halo = exp(-pow(tube / 0.13, 2.0));
            float fil = 0.55 + 0.45 * sfNoise(vec3(u * 15.0, v * 15.0, iTime * 0.07 + fk * 5.0));
            float alpha = act * smoothstep(0.05, 0.35, life) * (0.50 + 0.50 * life);
            col += hot * (core + 0.25 * halo) * band * fil * alpha * 0.55;
        }
    }
    if (hit != 0 && tc > 0.0 && dmin < rs * 1.65) {
        float hz = (dmin - rs) / max(rs, 1e-6);
        if (hz > 0.0 && (hit < 0 || tBest > tc)) {
            vec3 dir = normalize((ro + rd * tc) - u_star.xyz);
            vec3 md = sunFrame(dir);
            float euv = dot(texture(u_sunEuv, sunUV(md)).rgb, vec3(0.299, 0.587, 0.114));
            float act = smoothstep(0.04, 0.42, euv);
            float ray = 0.65 * sfNoise(md * 2.2 + iTime * 0.015) + 0.35 * sfNoise(md * 5.5 - iTime * 0.02);
            float eq = 1.0 + 0.45 * (1.0 - abs(dir.z));
            float localH = CORONA_H * eq * (0.60 + 1.05 * ray * ray) * (0.70 + 0.70 * act);
            vec3 corCol = mix(coronaM(u_starMode), uvM(u_starMode), smoothstep(0.00, 0.55, hz));
            col += corCol * CORONA_G * exp(-hz / max(localH, 1e-4))
                   * (0.70 + 0.60 * act) * u_starLum * (1.0 + 1.2 * u_flare + 1.5 * sunPulse());
        }
    }

    ivec2 ip = ivec2(fragCoord);
    float sum = 0.0;
    for (int dy = -BLUR_R; dy <= BLUR_R; dy++)
        for (int dx = -BLUR_R; dx <= BLUR_R; dx++) {
            ivec2 q = clamp(ip + ivec2(dx, dy), ivec2(0), ivec2(res) - 1);
            sum += float(texelFetch(u_acc, q, 0).r);
        }
    float den = sum / (u_splatK * float((2 * BLUR_R + 1) * (2 * BLUR_R + 1)));
    col += ringM(u_starMode) * den * RING_GAIN * 30.0 * (1.0 + 0.10 * u_flare);

    float zHit = hit >= 0 ? tBest * dot(rd, u_fw) : 1e9;
    float px = min(res.x, res.y) / 2410.0;
    vec4 SP[2] = vec4[2](u_sp0, u_sp1);
    vec4 SA[2] = vec4[2](u_sa0, u_sa1);
    vec4 BM[2] = vec4[2](u_bm0, u_bm1);
    for (int i = 0; i < 2; i++) {
        if (SP[i].w < 0.5) continue;
        vec3 sp = toScreen(SP[i].xyz, res);
        if (BM[i].w > 0.5) {
            vec3 bp = toScreen(BM[i].xyz, res);
            float d = segDist(fragCoord, sp.xy, bp.xy);
            float fl = 0.85 + 0.15 * sin(iTime * 90.0 + float(i) * 2.0);
            col += vec3(0.45, 1.0, 0.85) * (exp(-d * d / (1.2 * px * px)) * 2.2 + exp(-d / (7.0 * px)) * 0.10) * fl;
            float db = length(fragCoord - bp.xy);
            col += vec3(1.0, 0.85, 0.6) * (exp(-db * db / (16.0 * px * px)) * 2.5 + exp(-db / (14.0 * px)) * 0.25) * fl;
        }
        if (sp.z > zHit) continue;
        float d = length(fragCoord - sp.xy);
        float th = length(SA[i].xyz);
        vec3 hv = (i == 0 ? u_shv0 : u_shv1).xyz;
        if (dot(hv, hv) < 1e-8) hv = th > 0.02 ? SA[i].xyz : vec3(1.0, 0.0, 0.0);
        mat3 SB;
        vec2 sh = traceShip(ro, rd, SP[i].xyz, hv, SB);
        bool onHull = sh.x > 0.0 && sh.x < tBest;
        if (onHull) {
            vec3 pw = ro + rd * sh.x;
            vec3 nw = SB * shipNormal(transpose(SB) * (pw - SP[i].xyz) / SHIP_DRAW);
            vec3 Ls = u_star.xyz - pw;
            float dl = length(Ls);
            float lam = max(dot(nw, Ls / dl), 0.0);
            float rim = pow(1.0 - max(dot(nw, -rd), 0.0), 3.0);
            if (sh.y > 1.5) col = vec3(1.0, 0.62, 0.28) * (1.5 + 3.0 * min(th * 3.0, 1.0));
            else col = vec3(0.55, 0.58, 0.62) * starLightM(u_starMode) * (lam * 1.3 * u_starLum / max(dl * dl, 0.25) + 0.14)
                       + starLightM(u_starMode) * rim * 0.12;
        } else {
            col += vec3(0.75, 0.85, 1.0) * exp(-d / (9.0 * px)) * 0.10;
        }
        float thf = onHull ? 0.0 : th;
        if (thf > 0.02) {
            vec3 tail = SP[i].xyz - normalize(hv) * SHIP_DRAW * 1.05;
            vec3 tp = toScreen(tail, res);
            vec3 fp = toScreen(tail - normalize(hv) * 0.06 * sqrt(th), res);
            float df = segDist(fragCoord, tp.xy, fp.xy);
            col += vec3(1.0, 0.55, 0.2) * exp(-df * df / (2.5 * px * px)) * 1.4 * min(th * 3.0, 1.0);
        }
    }
    vec4 NK[4] = vec4[4](u_nk0, u_nk1, u_nk2, u_nk3);
    for (int i = 0; i < 4; i++) {
        if (NK[i].w < 0.5) continue;
        vec3 np = toScreen(NK[i].xyz, res);
        if (np.z > zHit) continue;
        float d = length(fragCoord - np.xy);
        float blink = 0.6 + 0.4 * step(0.5, fract(iTime * 6.0));
        col += vec3(1.0, 0.3, 0.15) * (exp(-d * d / (3.0 * px * px)) * 2.5 + exp(-d / (6.0 * px)) * 0.15) * blink;
    }
    vec4 FX[4] = vec4[4](u_fx0, u_fx1, u_fx2, u_fx3);
    for (int i = 0; i < 4; i++) {
        if (FX[i].w < 0.0) continue;
        vec3 fp = toScreen(FX[i].xyz, res);
        float age = FX[i].w;
        float rW = (0.03 + 0.20 * u_fxs[i]) * sqrt(age);
        float rP = rW / max(fp.z, 1e-3) / (2.0 * u_fov) * min(res.x, res.y);
        float d = length(fragCoord - fp.xy);
        float k = (1.0 - age) * (1.0 - age);
        col += mix(vec3(1.0, 0.95, 0.85), vec3(1.0, 0.45, 0.12), age) * k *
               (smoothstep(rP, rP * 0.6, d) * 3.0 + exp(-d / max(rP, 1.0)) * 0.8);
    }

    col = col / (1.0 + col * 0.6);
    fragColor = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}
