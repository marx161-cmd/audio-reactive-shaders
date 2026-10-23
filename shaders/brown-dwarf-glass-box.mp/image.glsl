// Brown-Dwarf-Glass-Box: image pass.
//
// The brown dwarf system in a box with stained-glass space walls lit by the
// dwarf. A music box: moon orbits and hit timing run on rails, meteor flight and
// the ring fluid are real physics. Two ships shoot from the star's beams; when
// moons and ring are gone a hole eats the rest and the dwarf spits the bodies
// back out with correction meteors (planned offline in a worker process).

uniform float u_kick;
uniform sampler2D u_ring;
uniform sampler2D u_phase;
uniform sampler2D u_moons;
uniform vec3 u_rockPos;
uniform mat3 u_rockRot;
uniform vec3 u_rockCom;
float rkCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0)) - r;
}
float racketSDF(vec3 p) {
    const float HX = 0.06, HA = 0.12, HB = 0.09, HT = 0.035, TX = -0.04, EX = -0.23, HR0 = 0.030, HR1 = 0.022,
                K = 0.04, BUMP = 0.007;
    float head = (length(vec3(p.x - HX, p.y, p.z) / vec3(HA, HB, HT)) - 1.0) * HT;
    float t = clamp((p.x - TX) / (EX - TX), 0.0, 1.0);
    float handle = length(vec3(p.x - clamp(p.x, EX, TX), p.y, p.z)) - mix(HR0, HR1, t);
    float hh = clamp(0.5 + 0.5 * (handle - head) / K, 0.0, 1.0);
    float d = mix(handle, head, hh) - K * hh * (1.0 - hh);
    float bump = sin(37.0 * p.x + 1.0) * sin(41.0 * p.y + 2.0) * sin(33.0 * p.z + 0.5)
               + 0.5 * sin(83.0 * p.x + 0.3) * sin(71.0 * p.y + 1.1) * sin(97.0 * p.z + 2.2);
    return d + BUMP * bump;
}
float tCapsule(vec3 p, vec3 a, vec3 b, float r) { vec3 pa = p - a, ba = b - a; return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0)) - r; }
float tSDF(vec3 p) {
    const float TA = 0.08, TB = 0.036, TC = 0.08, TR1 = 0.0176, TR2 = 0.016;
    return min(tCapsule(p, vec3(-TA, TB, 0.0), vec3(TA, TB, 0.0), TR1), tCapsule(p, vec3(0.0, TB, 0.0), vec3(0.0, -TC, 0.0), TR2));
}
#define ROCK_SDF tSDF
uniform sampler2D u_boltSlots;
uniform sampler2D u_bolts;

#define NR      1024
#define NT      4096
#define DW_R    0.1
#define R_IN    0.25
#define R_OUT   3.0
#define DR      ((R_OUT - R_IN) / float(NR))
#define DTH     (6.28318530718 / float(NT))
#define N_MOONS 9
#define TAU_K   0.50
#define TAU_K2  1.6
#define RING_ALB 0.90
#define RING_TINT vec3(0.90, 0.96, 1.00)
#define PLANET_MEAN vec3(0.764, 0.088, 0.993)
#define SECRET_WHITE 0.40
#define LIGHT_COL mix(PLANET_MEAN, vec3(dot(PLANET_MEAN, vec3(0.2126, 0.7152, 0.0722))), SECRET_WHITE)
#define RING_LIGHT vec3(dot(PLANET_MEAN, vec3(0.2126, 0.7152, 0.0722)))
#define RING_WHITE_0 vec3(0.86, 0.92, 1.00)
uniform vec2 u_dbHue;
vec3 dbHueCol(float h) { return clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0); }
vec3 nsTint(vec3 c) { return u_dbHue.y > 0.002 ? mix(c, dbHueCol(u_dbHue.y) * max(c.r, max(c.g, c.b)), 0.75) : c; }
#define RING_WHITE (u_dbHue.x > 0.002 ? mix(RING_WHITE_0, dbHueCol(u_dbHue.x), 0.65) : RING_WHITE_0)
vec3 dbHueRot(vec3 c, float h) {
    float a = 6.28318530718 * h, ca = cos(a), sa = sin(a);
    vec3 k = vec3(0.57735026919);
    return c * ca + cross(k, c) * sa + k * dot(k, c) * (1.0 - ca);
}
#define RING_FLUOR 2.8
#define UV_HALO_COL nsTint(vec3(0.62, 0.80, 1.00))
#define UV_HALO_G   0.55
#define UV_HALO_H   0.16
uniform vec2 u_nsStar;
uniform vec4 u_ns2;
#define NS2_R   (0.9 * DW_R)
#define LIGHT_K (5.5 * (1.0 - u_nsStar.x))
#define DUST_FWD 0.10
#define DUST_G   0.70
#define MOON_ALB 0.35
#define ROCK_ON 1
#define ROCK_ON_OLD 0
#define T_ALB    0.85
#define T_COL    vec3(1.00, 0.76, 0.33)
#define ROCK_ALB 0.60
#define RK_BOUND 0.16
#define MULTI_SC 0.85

#define SPIN_TILT  0.0
#define T_ZONE     1350.0
#define T_BELT     1900.0
#define T_STORM    2400.0
#define BANDS      7.0
#define BAND_WOB   0.16
#define JET_V      0.035
#define FLOW_P     40.0
#define TURB       0.85
#define N_STORMS   7
#define STORM_R    0.16
#define L_GAIN     1.6
#define HAZE_H     0.035

uniform vec3 u_pad_look;
uniform vec3 u_pad_move;
uniform vec4 u_pad_cam_q;
uniform vec3 u_pad_boop;

#define CAM_D      (3.2 / FOV + 3.2)
#define CAM_EL     0.5236
#define CAM_ORBIT  0.0
#define AIM_OFF    0.0
#define CAM_ROLL   0.7854
#define FOV        0.12

#define STAR_DENS  0.05

vec3 bbColour(float T) {
    float t = T / 100.0;
    vec3 c;
    c.r = t <= 66.0 ? 1.0 : clamp(1.2929 * pow(t - 60.0, -0.1332), 0.0, 1.0);
    c.g = t <= 66.0 ? clamp((99.4708 * log(t) - 161.1196) / 255.0, 0.0, 1.0) : clamp(1.1299 * pow(t - 60.0, -0.0755), 0.0, 1.0);
    c.b = t >= 66.0 ? 1.0 : (t <= 19.0 ? 0.0 : clamp((138.5177 * log(t - 10.0) - 305.0448) / 255.0, 0.0, 1.0));
    return c;
}
vec3 dwarfHue(float T) {
    float x = clamp((T - T_ZONE) / (T_STORM - T_ZONE), 0.0, 1.2);
    vec3 c = mix(vec3(0.42, 0.03, 0.26), vec3(1.00, 0.16, 0.36), smoothstep(0.0, 0.65, x));
    return mix(c, vec3(1.00, 0.62, 0.78), smoothstep(0.6, 1.1, x));
}
vec3 radiance(float T) { return dwarfHue(T) * pow(T / 1900.0, 3.0) * L_GAIN; }

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
vec3 hash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}
float vnoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), f.x), mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x), mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float fbm(vec3 p) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = p * 2.03 + 17.1; a *= 0.5; }
    return s;
}
vec3 rotY(vec3 p, float a) { float c = cos(a), s = sin(a); return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z); }

const float EDGES[12] = float[12](-1.02, -0.78, -0.55, -0.36, -0.17, -0.05, 0.08, 0.24, 0.40, 0.58, 0.74, 0.95);
#define EDGE_W     0.012
#define MEANDER    0.030
#define BILLOW_A   0.020
#define BILLOW_K   14.0
#define EDGE_TURB  0.045
#define EDGE_REACH 0.06
#define STREAK     0.22
#define LIMB_U1    0.60
#define LIMB_U2    0.12
#define LIMB_PURP  vec3(0.30, 0.03, 0.34)

float bandOf(float lat, out float edgeD) {
    float belt = 0.0, sgn = 1.0;
    edgeD = 9.0;
    for (int i = 0; i < 12; i++) {
        belt += sgn * smoothstep(EDGES[i] - EDGE_W, EDGES[i] + EDGE_W, lat);
        sgn = -sgn;
        edgeD = min(edgeD, abs(lat - EDGES[i]));
    }
    return clamp(belt, 0.0, 1.0);
}

float jet(float lat) {
    float e; float belt = bandOf(lat, e);
    return JET_V * ((2.0 * belt - 1.0) * (0.45 + 0.55 * cos(lat)) + 0.8 * exp(-(lat / 0.12) * (lat / 0.12)));
}

float streaks(vec3 n, float tp, float cs) {
    float lat0 = asin(clamp(n.y, -1.0, 1.0));
    vec3 q = rotY(n, jet(lat0) * tp);
    return fbm(q * vec3(3.0, 70.0, 3.0) + cs * 5.0) - 0.5;
}

float edgeSpeed(float lat, out float edgeLat) {
    float best = 9.0; edgeLat = 0.0;
    for (int i = 0; i < 12; i++) { float d = abs(lat - EDGES[i]); if (d < best) { best = d; edgeLat = EDGES[i]; } }
    return 0.5 * (jet(edgeLat - 0.03) + jet(edgeLat + 0.03));
}

float cloudT(vec3 n, float t) {
    float lat = asin(clamp(n.y, -1.0, 1.0)), lon = atan(n.z, n.x);
    float eLat;
    float lonE = lon - edgeSpeed(lat, eLat) * t;
    vec3 qE = vec3(cos(lat) * cos(lonE), sin(lat), cos(lat) * sin(lonE));
    float e0 = abs(lat - eLat);
    float near = exp(-(e0 / EDGE_REACH) * (e0 / EDGE_REACH));
    float wide = exp(-(e0 / 0.09) * (e0 / 0.09));
    float latW = lat + wide * MEANDER * (fbm(qE * 2.2 + eLat * 13.0) - 0.5) * 2.0;
    float psi = lonE * BILLOW_K + 1.5 * vnoise(vec3(lonE * 6.0, eLat * 20.0, 1.0));
    float bA = near * BILLOW_A * (0.15 + 0.85 * smoothstep(0.3, 0.8, vnoise(vec3(lonE * 3.0, eLat * 8.0, 2.0))));
    latW += bA * sin(psi);
    float lonW = lonE + bA * 1.6 * cos(psi);
    vec3 qw = vec3(cos(latW) * cos(lonW), sin(latW), cos(latW) * sin(lonW));
    latW += near * EDGE_TURB * (fbm(qw * vec3(14.0, 30.0, 14.0)) - 0.5) * 2.0;
    float eD;
    float belt = bandOf(latW, eD);
    float T = mix(T_ZONE, T_BELT, belt);
    float ph = t / FLOW_P;
    float a = fract(ph), b = fract(ph + 0.5);
    float st = mix(streaks(n, a * FLOW_P, floor(ph)), streaks(n, b * FLOW_P, floor(ph + 0.5) + 50.0), abs(2.0 * a - 1.0));
    T += (T_BELT - T_ZONE) * STREAK * st * 2.0 * (1.0 - wide);
    float pol = smoothstep(0.90, 1.05, abs(latW));
    T = mix(T, mix(T_ZONE, T_BELT, 0.25 + 0.6 * fbm(qE * 11.0)), pol);
    T += (T_BELT - T_ZONE) * 0.35 * near * smoothstep(0.55, 0.85, fbm(qw * vec3(9.0, 18.0, 9.0))) * (1.0 - belt);
    for (int i = 0; i < N_STORMS; i++) {
        vec3 h = hash33(vec3(float(i), 3.0, 7.0));
        bool big = i == 0;
        float sl = big ? -0.36 : (h.x - 0.5) * 1.5;
        float sr = big ? 0.24 : STORM_R * (0.35 + 0.4 * h.z);
        float lon = h.y * 6.2832 + jet(sl) * t;
        vec3 sc = vec3(cos(sl) * cos(lon), sin(sl), cos(sl) * sin(lon));
        vec3 tx = normalize(cross(vec3(0, 1, 0), sc)), ty = cross(sc, tx);
        vec2 lc = vec2(dot(n, tx), dot(n, ty)) / sr;
        lc.x *= big ? 0.55 : 0.8;
        float d = length(lc);
        if (dot(n, sc) < 0.0 || d > 2.5) continue;
        float ang = atan(lc.y, lc.x);
        float spiral = 0.5 + 0.5 * sin(ang * 2.0 + d * (big ? 10.0 : 7.0) - t * 0.25);
        T = mix(T, T_STORM, exp(-d * d * 2.2) * (big ? 0.85 : 1.0));
        T += 220.0 * spiral * exp(-(d - 1.0) * (d - 1.0) * 4.0);
    }
    return T;
}

#define DWARF_STYLE 1
#define CF_SCALE    2.0
#define CF_STRETCH  2.5
#define OS_GAIN     1.5
#define OS_COVER    0.50
#define CF_SPEED    0.035
uniform float u_churn;
#define CF_SPIN     0.012
#define CF_GAIN     3.2
#define UV_COL    vec3(0.90, 0.05, 0.75)
#define UV_GLOW   0.40
#define UV_KNEE   0.55
#define ACC_COL      vec3(1.00, 0.50, 0.85)
#define ACC_GAIN     2.5
#define ACC_LO       1.15
#define ACC_HI       1.55
#define PS_JETS      5.0
#define PS_SHEAR     0.9
#define PS_SCALE     1.0
#define PS_OCT       4
#define PS_SPREAD    3.0
#define CF_BREATH 0.15
#define N_BOLT      6
#define BD_SW       512.0
#define BD_SH       160.0
#define BD_X0       0.08
#define BD_X1       0.92
#define BD_COLS     8
#define BD_ROWS     8
#define BOLT_COL    vec3(0.62, 0.92, 1.00)
#define BOLT_GAIN   6.0
#define BOLT_GLOW   vec3(0.45, 0.55, 1.00)
#define GLOW_GAIN   0.9
#define GLOW_R      0.06
#define LEADER_T    0.07
#define BOLT_MINW   0.0022
float boltEnv(float a) {
    if (a < 0.0 || a > 2.0) return 0.0;
    float lead = smoothstep(0.0, 0.015, a) * (a < LEADER_T ? 0.35 : 1.0);
    float main = a < LEADER_T ? 1.0 : exp(-(a - LEADER_T) / 0.09);
    float re = 0.65 * exp(-pow((a - 0.24) / 0.035, 2.0)) + 0.45 * exp(-pow((a - 0.41) / 0.035, 2.0));
    return lead * main + re + 0.12 * exp(-a / 0.7);
}
vec3 lightning(vec3 m, float t) {
    vec3 acc = vec3(0.0);
    for (int i = 0; i < N_BOLT; i++) {
        vec4 s0 = texelFetch(u_boltSlots, ivec2(i, 0), 0), s1 = texelFetch(u_boltSlots, ivec2(i, 1), 0);
        float a = t - s0.w;
        float env = boltEnv(a);
        if (env <= 0.001) continue;
        float strength = texelFetch(u_boltSlots, ivec2(i, 2), 0).x;
        vec3 S = s0.xyz, E = s1.xyz;
        float arc = acos(clamp(dot(S, E), -1.0, 1.0));
        vec3 D = normalize(E - S * dot(S, E));
        vec3 B = cross(S, D);
        float along = atan(dot(m, D), dot(m, S)) / arc;
        float across = asin(clamp(dot(m, B), -1.0, 1.0)) / arc;
        if (dot(m, S) < 0.0) continue;
        float midD = acos(clamp(dot(m, normalize(S + E)), -1.0, 1.0));
        acc += BOLT_GLOW * GLOW_GAIN * strength * env * exp(-(midD * midD) / (GLOW_R * GLOW_R + arc * arc * 0.25));
        float reveal = clamp(a / LEADER_T, 0.0, 1.0);
        if (along < -0.1 || along > reveal + 0.02) continue;
        float sx = BD_X0 + along * (BD_X1 - BD_X0);
        float wide = max(1.0, BOLT_MINW / (arc * 0.019));
        float sy = 0.5 + across / wide * (BD_X1 - BD_X0) * BD_SW / BD_SH;
        if (sx < 0.0 || sx > 1.0 || sy < 0.0 || sy > 1.0) continue;
        int k = int(s1.w);
        vec2 uvA = (vec2(float(k % BD_COLS), float(k / BD_COLS)) + vec2(sx, sy)) / vec2(float(BD_COLS), float(BD_ROWS));
        float b = textureLod(u_bolts, uvA, 0.0).r;
        acc += BOLT_COL * BOLT_GAIN * strength * env * b;
    }
    return acc;
}

uniform sampler2D u_notes;
uniform sampler2D u_spark;
uniform sampler2D u_sparkHdr;
uniform sampler2D u_tube;
uniform sampler2D u_tubeHdr;
const int PC_MOON[12] = int[12](2, 0, 3, 7, 4, 8, 5, 1, 6, 2, 7, 3);
#define SPK_W     96
#define SPK_H     32
#define SPK_NSX   6
#define SPK_ARC   0.30
#define SPK_GAIN  2.6
#define SPK_HALO  0.5
#define SPK_LEAD  0.05
#define TB_W      384
#define TB_H      64
#define TB_SEG    20
#define TB_GAIN   3.0
#define TB_HALO   0.6
#define CH_BRANCH 0.30
#define AUR_GREEN vec3(0.20, 1.00, 0.35)
#define AUR_RED   vec3(1.00, 0.10, 0.08)
float sparkEnv(float a) {
    if (a < 0.0 || a > 0.8) return 0.0;
    float lead = a < SPK_LEAD ? 0.45 : 1.0;
    float main = a < SPK_LEAD ? 1.0 : exp(-(a - SPK_LEAD) / 0.07);
    return lead * main + 0.5 * exp(-pow((a - 0.17) / 0.03, 2.0)) + 0.08 * exp(-a / 0.25);
}
vec2 dbmLook(sampler2D cells, ivec2 o, ivec2 size, vec2 q, float maxOrder) {
    vec2 r = vec2(0.0);
    ivec2 c0 = ivec2(floor(q));
    for (int j = -3; j <= 3; j++) for (int i = -3; i <= 3; i++) {
        ivec2 c = c0 + ivec2(i, j);
        if (c.x < 0 || c.y < 0 || c.x >= size.x || c.y >= size.y) continue;
        vec4 v = texelFetch(cells, o + c, 0);
        if (v.y < 0.5 || v.z > maxOrder) continue;
        float w = v.y > 1.5 ? 1.0 : CH_BRANCH;
        vec2 d = vec2(c) + 0.5 - q;
        float dd = dot(d, d);
        r.x = max(r.x, w * exp(-dd / 0.45));
        r.y += w * exp(-dd / 5.0) * 0.08;
    }
    return r;
}
vec3 ftPoint(float L, float phi, float lat) {
    float r = L * cos(lat) * cos(lat);
    return vec3(r * cos(lat) * cos(phi), r * sin(lat), r * cos(lat) * sin(phi));
}
#define STK_D0 0.07
#define STK_D1 0.08
vec3 strikePoint(vec3 M, float fc, int pc) {
    float h1 = fract(sin(fc * 12.9898 + float(pc) * 78.233) * 43758.5453);
    float h2 = fract(sin(fc * 39.3468 + float(pc) * 11.135) * 24634.6345);
    vec2 er = normalize(M.xz), ep = vec2(-er.y, er.x);
    float ang = h1 * 6.28318530718, d = STK_D0 + STK_D1 * h2;
    vec2 S = M.xz + d * (cos(ang) * er + sin(ang) * ep);
    S = normalize(S) * clamp(length(S), 0.34, 2.86);
    return vec3(S.x, 0.0, S.y);
}
vec3 arcPoint(vec3 A, vec3 S, float u) {
    return mix(A, S, u) + vec3(0.0, SPK_ARC * length(S - A) * 4.0 * u * (1.0 - u), 0.0);
}
vec3 noteBolts(vec3 ro, vec3 rd, float tPl, float tRing, float trRing, float t) {
    vec3 acc = vec3(0.0);
    for (int pc = 0; pc < 12; pc++) {
        vec4 nt = texelFetch(u_notes, ivec2(pc, 0), 0);
        float a = t - nt.z, env = sparkEnv(a);
        if (env <= 0.001) continue;
        float fc = texelFetch(u_notes, ivec2(pc, 1), 0).x;
        vec4 Mm = texelFetch(u_moons, ivec2(PC_MOON[pc], 0), 0);
        if (Mm.w <= 0.0) continue;
        vec3 S = strikePoint(Mm.xyz, fc, pc);
        vec3 A = Mm.xyz + normalize(S - Mm.xyz) * Mm.w;
        float len = length(S - A);
        float cell = len / (float(SPK_W) - 4.5), halfW = cell * 0.5 * float(SPK_H);
        vec3 C = 0.5 * (A + S) + vec3(0.0, SPK_ARC * len * 0.5, 0.0);
        vec3 oc = ro - C; float bb = dot(oc, rd), cc = dot(oc, oc) - pow(0.6 * len + halfW, 2.0);
        if (bb * bb - cc < 0.0) continue;
        float best = 1e9, uB = 0.0, vB = 0.0, tB = 0.0, run = 0.0;
        vec3 prev = A;
        for (int k = 1; k <= 12; k++) {
            vec3 P = arcPoint(A, S, float(k) / 12.0);
            vec3 d1 = P - prev, w0 = ro - prev;
            float aa = dot(d1, d1), b2 = dot(rd, d1), dd = dot(rd, w0), ee = dot(d1, w0);
            float den = aa - b2 * b2;
            float sc = den > 1e-9 ? clamp((ee - b2 * dd) / den, 0.0, 1.0) : 0.0;
            vec3 Q = prev + d1 * sc;
            float tc = max(dot(rd, Q - ro), 0.0);
            vec3 off = ro + rd * tc - Q;
            float dist = length(off);
            if (dist < best) { best = dist; tB = tc; uB = run + sc * length(d1); vB = dot(off, normalize(cross(d1, rd))); }
            run += length(d1);
            prev = P;
        }
        if (best > halfW || tB > tPl) continue;
        vec2 q = vec2(1.5 + uB / run * (float(SPK_W) - 4.5), 0.5 * float(SPK_H) + vB / cell);
        int slot = pc * 2 + int(mod(fc, 2.0));
        ivec2 o = ivec2((slot % SPK_NSX) * SPK_W, (slot / SPK_NSX) * SPK_H);
        float steps = texelFetch(u_sparkHdr, ivec2(slot, 1), 0).x;
        float maxOrder = clamp(a / SPK_LEAD, 0.0, 1.0) * steps + 0.5;
        vec2 L = dbmLook(u_spark, o, ivec2(SPK_W, SPK_H), q, maxOrder);
        float vis = tB > tRing ? trRing : 1.0;
        acc += (BOLT_COL * SPK_GAIN * L.x + BOLT_GLOW * SPK_HALO * L.y) * env * nt.w * vis;
    }
    return acc;
}
uniform sampler2D u_aurF;
uniform sampler2D u_aurPhase;
#define AU_R0   1.40
#define AU_R1   2.42
#define AU_FT   2048
#define AU_FR   384
#define AU_H    0.12
#define AU_STEPS 36
#define AU_GAIN 40.0
#define AU_PINK vec3(0.95, 0.25, 0.55)
vec4 aurAt(float r, float th) {
    float jf = clamp((r - AU_R0) / (AU_R1 - AU_R0) * float(AU_FR) - 0.5, 0.0, float(AU_FR - 1));
    int j0 = int(floor(jf)); float fj = jf - float(j0);
    vec4 res = vec4(0.0);
    for (int q = 0; q < 2; q++) {
        int j = min(j0 + q, AU_FR - 1);
        float rel = mod(th - texelFetch(u_aurPhase, ivec2(j, 0), 0).x, 6.28318530718);
        float kf = rel / (6.28318530718 / float(AU_FT)) - 0.5;
        int k0 = int(floor(kf)); float fk = kf - float(k0);
        vec4 a = texelFetch(u_aurF, ivec2((k0 % AU_FT + AU_FT) % AU_FT, j), 0);
        vec4 b = texelFetch(u_aurF, ivec2(((k0 + 1) % AU_FT + AU_FT) % AU_FT, j), 0);
        res += mix(a, b, fk) * (q == 0 ? 1.0 - fj : fj);
    }
    return res;
}
vec3 auroraCurtains(vec3 ro, vec3 rd, float tEnd, float trRing, vec2 fragCoord) {
    if (abs(rd.y) < 1e-4) return vec3(0.0);
    float ta = (AU_H - ro.y) / rd.y, tb = (-AU_H - ro.y) / rd.y;
    float t0 = max(min(ta, tb), 0.0), t1 = min(max(ta, tb), tEnd);
    if (t1 <= t0) return vec3(0.0);
    float ds = (t1 - t0) / float(AU_STEPS);
    float t = t0 + ds * fract(sin(dot(fragCoord, vec2(12.9898, 78.233))) * 43758.5453);
    vec3 acc = vec3(0.0);
    for (int i = 0; i < AU_STEPS; i++, t += ds) {
        vec3 p = ro + rd * t;
        float r = length(p.xz);
        float rho2 = r * r + p.y * p.y;
        float L = rho2 * sqrt(rho2) / max(r * r, 1e-4);
        if (L < AU_R0 || L > AU_R1) continue;
        vec4 f = aurAt(L, atan(p.z, p.x));
        float e = f.x * (f.y + f.z);
        if (e < 1e-4) continue;
        float ah = abs(p.y) / AU_H;
        float pk = exp(-pow((ah - 0.05) / 0.05, 2.0));
        float gr = smoothstep(0.03, 0.10, ah) * exp(-max(ah - 0.20, 0.0) / 0.25);
        float rd_ = smoothstep(0.35, 0.65, ah) * (1.0 - smoothstep(0.85, 1.0, ah));
        vec3 em = f.x * (f.y * (AUR_GREEN * gr + AU_PINK * 0.6 * pk) + f.z * AUR_RED * rd_);
        if (p.y * ro.y < 0.0) em *= trRing;
        acc += em * ds;
    }
    return acc * AU_GAIN;
}

vec3 planetTube(vec3 ro, vec3 rd, float tPl, float tRing, float trRing, float t) {
    vec4 p0 = texelFetch(u_notes, ivec2(12, 0), 0), p1 = texelFetch(u_notes, ivec2(12, 1), 0);
    float a = t - p0.z, env = boltEnv(a);
    if (env <= 0.001) return vec3(0.0);
    int mi = PC_MOON[int(p1.y)];
    if (texelFetch(u_moons, ivec2(mi, 0), 0).w <= 0.0) return vec3(0.0);
    vec3 M = texelFetch(u_moons, ivec2(mi, 0), 0).xyz;
    float Lm = max(length(M), 1.05 * DW_R), phi = atan(M.z, M.x), lat0 = acos(sqrt(DW_R / Lm));
    int pair = int(mod(p1.x, 2.0));
    vec3 acc = vec3(0.0);
    for (int h = 0; h < 2; h++) {
        int slot = pair * 2 + h;
        float latF = h == 0 ? lat0 : -lat0;
        float arc = 0.0;
        vec3 prev = ftPoint(Lm, phi, latF);
        for (int k = 1; k <= TB_SEG; k++) { vec3 P = ftPoint(Lm, phi, mix(latF, 0.0, float(k) / float(TB_SEG))); arc += length(P - prev); prev = P; }
        float cell = arc / (float(TB_W) - 4.5);
        float halfW = cell * 0.5 * float(TB_H);
        float best = 1e9, uBest = 0.0, vBest = 0.0, tBest = 0.0, run = 0.0;
        prev = ftPoint(Lm, phi, latF);
        for (int k = 1; k <= TB_SEG; k++) {
            vec3 P = ftPoint(Lm, phi, mix(latF, 0.0, float(k) / float(TB_SEG)));
            vec3 d1 = P - prev, w0 = ro - prev;
            float aa = dot(d1, d1), bb = dot(rd, d1), dd = dot(rd, w0), ee = dot(d1, w0);
            float den = aa - bb * bb;
            float sc = den > 1e-9 ? clamp((ee - bb * dd) / den, 0.0, 1.0) : 0.0;
            vec3 Q = prev + d1 * sc;
            float tc = max(dot(rd, Q - ro), 0.0);
            vec3 off = ro + rd * tc - Q;
            float dist = length(off);
            if (dist < best) {
                best = dist; tBest = tc;
                uBest = (run + sc * length(d1)) / arc;
                vBest = dot(off, normalize(cross(d1, rd)));
            }
            run += length(d1);
            prev = P;
        }
        if (best > halfW || tBest > tPl) continue;
        vec2 q = vec2(1.5 + uBest * (float(TB_W) - 4.5), 0.5 * float(TB_H) + vBest / cell);
        ivec2 o = ivec2(0, slot * TB_H);
        float steps = texelFetch(u_tubeHdr, ivec2(slot, 1), 0).x;
        float maxOrder = clamp(a / LEADER_T, 0.0, 1.0) * steps + 0.5;
        vec2 L = dbmLook(u_tube, o, ivec2(TB_W, TB_H), q, maxOrder);
        float vis = tBest > tRing ? trRing : 1.0;
        acc += (BOLT_COL * TB_GAIN * L.x + BOLT_GLOW * TB_HALO * L.y) * env * p0.w * vis;
    }
    return acc;
}

uniform sampler2D u_spokes;
#define SPOKE_N     32
#define SPOKE_RISE  0.12
#define SPOKE_LIFE  4.0
#define SPOKE_SIMR  0.5
#define SPOKE_BRIGHT 0.9
#define SPOKE_DARK  0.45
float ringSpokes(float r, float th, float t) {
    float acc = 0.0;
    for (int i = 0; i < SPOKE_N; i++) {
        vec4 a = texelFetch(u_spokes, ivec2(i, 0), 0);
        float age = t - a.z;
        if (age < 0.0 || age > SPOKE_LIFE + SPOKE_RISE) continue;
        vec4 b = texelFetch(u_spokes, ivec2(i, 1), 0);
        float dr = r - a.x;
        if (abs(dr) > 0.5 * b.x + 0.02) continue;
        float ths = a.y + pow(r, -1.5) * age * SPOKE_SIMR;
        float dth = mod(th - ths + 3.14159265, 6.28318530718) - 3.14159265;
        float across = r * dth / (b.y * (1.0 + 0.6 * age));
        float along = 1.0 - smoothstep(0.35 * b.x, 0.5 * b.x, abs(dr));
        float env = smoothstep(0.0, SPOKE_RISE, age) * exp(-max(age - SPOKE_RISE, 0.0) / (0.35 * SPOKE_LIFE));
        acc += a.w * env * along * exp(-across * across);
    }
    return min(acc, 1.5);
}

uniform sampler2D u_met;
#define MT_N        3
#define MT_HIST     48
#define MT_COL_HEAD vec3(1.00, 0.86, 0.62)
#define MT_COL_BODY vec3(1.00, 0.34, 0.10)
#define MT_COL_DEEP vec3(0.55, 0.09, 0.12)
#define MT_CORE_R   0.010
#define MT_GLOW_R   0.024
#define MT_TAIL_S   0.55
#define MT_TRAIL_R0 0.003
#define MT_TRAIL_RW 0.014
vec3 meteors(vec3 ro, vec3 rd, float tVis, float tRing, float trRing, float t) {
    vec3 acc = vec3(0.0);
    for (int i = 0; i < MT_N; i++) {
        vec4 inf = texelFetch(u_met, ivec2(MT_HIST, i), 0);
        int nh = int(inf.z);
        float ia0 = t - inf.y;
        if (nh == 0 && !(ia0 >= 0.0 && ia0 < 1.5)) continue;
        if (nh > 0) {
            vec3 A = texelFetch(u_met, ivec2(0, i), 0).xyz, B = texelFetch(u_met, ivec2(nh - 1, i), 0).xyz;
            vec3 Cb = 0.5 * (A + B);
            float rb = 0.5 * length(A - B) + 0.15;
            vec3 oc = ro - Cb;
            float bq = dot(oc, rd);
            if (dot(oc, oc) - bq * bq > rb * rb && !(ia0 >= 0.0 && ia0 < 1.5)) continue;
        }
        if (inf.x > 0.5 && nh > 0) {
            vec3 H = texelFetch(u_met, ivec2(0, i), 0).xyz;
            float tH = dot(H - ro, rd);
            if (tH > 0.0 && tH < tVis) {
                vec3 o = ro + rd * tH - H;
                float d2 = dot(o, o);
                float heat = clamp((inf.w - 1.0) / 1.5, 0.0, 1.0);
                float vis = (tH > tRing && H.y * ro.y < 0.0) ? trRing : 1.0;
                acc += MT_COL_HEAD * vis * (exp(-d2 / (MT_CORE_R * MT_CORE_R)) * (1.6 + 2.4 * heat)
                                          + exp(-d2 / (MT_GLOW_R * MT_GLOW_R)) * (0.20 + 0.7 * heat));
            }
        }
        for (int k = 0; k < MT_HIST - 2; k += 2) {
            if (k + 2 >= nh) break;
            vec4 a = texelFetch(u_met, ivec2(k, i), 0), b = texelFetch(u_met, ivec2(k + 2, i), 0);
            float age = (t - a.w) / MT_TAIL_S;
            if (age > 1.0) break;
            vec3 d1 = b.xyz - a.xyz, w0 = ro - a.xyz;
            float aa = dot(d1, d1), bb = dot(rd, d1), dd = dot(rd, w0), ee = dot(d1, w0);
            float den = aa - bb * bb;
            float sc = den > 1e-12 ? clamp((ee - bb * dd) / den, 0.0, 1.0) : 0.0;
            vec3 C = a.xyz + d1 * sc;
            float tc = dot(C - ro, rd);
            if (tc <= 0.0 || tc >= tVis) continue;
            float ag = clamp((t - mix(a.w, b.w, sc)) / MT_TAIL_S, 0.0, 1.0);
            float rr = MT_TRAIL_R0 + MT_TRAIL_RW * ag;
            vec3 off = ro + rd * tc - C;
            float dens = (1.0 - ag) * (1.0 - ag);
            float vis = (tc > tRing && C.y * ro.y < 0.0) ? trRing : 1.0;
            acc += mix(MT_COL_DEEP, mix(MT_COL_BODY, MT_COL_HEAD, dens), 1.0 - ag) * 0.9 * dens * vis
                 * exp(-dot(off, off) / (rr * rr)) * (MT_TRAIL_R0 / rr);
        }
        float ia = t - inf.y;
        if (ia >= 0.0 && ia < 1.5) {
            vec4 ip = texelFetch(u_met, ivec2(MT_HIST + 1, i), 0);
            float tI = dot(ip.xyz - ro, rd);
            if (tI > 0.0 && tI < tVis + 0.05) {
                vec3 o = ro + rd * tI - ip.xyz;
                float rg = 0.01 + 0.06 * sqrt(ia) * max(ip.w, 0.05) / 0.05 * 0.5;
                acc += mix(MT_COL_BODY, MT_COL_HEAD, exp(-ia / 0.08)) * exp(-ia / 0.35) * 3.0
                     * exp(-dot(o, o) / (rg * rg));
            }
        }
    }
    return acc;
}

uniform sampler2D u_chase;
uniform float u_chaseOn;
uniform vec3 u_chaseNrm;
#define CH_N       48
#define CH_SW      5
uniform vec2 u_chaseHue;
#define LEAD_COL0  nsTint(vec3(0.78, 0.90, 1.00))
#define SHIP_COL0  nsTint(vec3(0.62, 0.80, 1.00))
#define LEAD_COL   (u_chaseHue.x > 0.002 ? max(dbHueRot(LEAD_COL0, u_chaseHue.x), 0.0) : LEAD_COL0)
#define LEAD_TAILS 4
float squidW(float u, float L) {
    float m = clamp((L - u) / (1.5 * L), 0.0, 1.0);
    return 0.30 * L * pow(sin(3.14159265 * m), 0.7) + 0.48 * L * exp(-pow((u + 0.22 * L) / (0.22 * L), 2.0)) * step(-0.5 * L, u);
}
#define SHIP_COL   (u_chaseHue.y > 0.002 ? max(dbHueRot(SHIP_COL0, u_chaseHue.y), 0.0) : SHIP_COL0)
vec3 chaseShip(int i, vec3 ro, vec3 rd, float tVis, float tRing, float trRing, float t, float L, float trS, float trR, vec3 col, bool squid) {
    vec3 acc = vec3(0.0);
    vec4 inf = texelFetch(u_chase, ivec2(CH_N, i), 0);
    int nh = int(inf.z);
    if (nh == 0) return acc;
    vec3 P = texelFetch(u_chase, ivec2(0, i), 0).xyz;
    int nb = 0;
    for (int k = 0; k < CH_N; k++) { if (k >= nh || (t - texelFetch(u_chase, ivec2(k, i), 0).w) > trS) break; nb = k; }
    vec3 B = texelFetch(u_chase, ivec2(nb, i), 0).xyz;
    vec3 Cb = 0.5 * (P + B), oc = ro - Cb;
    float rb = 0.5 * length(P - B) + 4.0 * L + 0.05, bq = dot(oc, rd);
    if (dot(oc, oc) - bq * bq > rb * rb) return acc;
    vec3 D = texelFetch(u_chase, ivec2(CH_N + 1, i), 0).xyz;
    vec3 S = normalize(cross(D, u_chaseNrm) + vec3(1e-6)), N = cross(S, D);
    float dn = dot(rd, N);
    if (abs(dn) > 1e-4) {
        float tp = dot(P - ro, N) / dn;
        if (tp > 0.0 && tp < tVis) {
            vec3 q = ro + rd * tp - P;
            float u = dot(q, D), v = abs(dot(q, S));
            if (squid) {
                float w = squidW(u, L);
                if (u > -0.5 * L && u < L && v < w) {
                    float rim = 1.0 - smoothstep(0.0, 0.16 * L, w - v);
                    float spine = exp(-pow(v / (0.06 * L), 2.0)) * smoothstep(-0.5 * L, 0.6 * L, u);
                    acc += col * (0.30 + 2.4 * rim + 0.8 * spine) + vec3(0.6) * (rim + 0.4 * spine);
                }
            } else {
                float w = 0.55 * L * (L - u) / (1.6 * L);
                if (u > -0.6 * L && u < L && v < w) {
                    float rim = 1.0 - smoothstep(0.0, 0.2 * L, min(w - v, u + 0.6 * L));
                    acc += col * (0.35 + 2.6 * rim) + vec3(0.6) * rim;
                }
            }
        }
    }
    vec3 E = P - D * 0.6 * L;
    float tE = dot(E - ro, rd);
    if (tE > 0.0 && tE < tVis) {
        vec3 o = ro + rd * tE - E;
        float d2 = dot(o, o), fl = inf.x, r2 = L * L;
        acc += mix(col, vec3(1.0, 0.95, 0.85), 0.6) * (exp(-d2 / (0.10 * r2)) * (0.6 + 3.0 * fl) + exp(-d2 / (2.0 * r2)) * (0.08 + 0.6 * fl));
    }
    int nT = squid ? LEAD_TAILS : 1;
    for (int j = 0; j < 4; j++) {
    if (j >= nT) break;
    float vo = 0.0;
    if (squid) vo = (LEAD_TAILS == 2) ? (j == 0 ? -0.70 : 0.70) : (-0.36 + 0.24 * float(j));
    for (int k = 0; k < CH_N - 1; k++) {
        if (k + 1 >= nh) break;
        vec4 a = texelFetch(u_chase, ivec2(k, i), 0), b = texelFetch(u_chase, ivec2(k + 1, i), 0);
        if ((t - a.w) / trS > 1.0) break;
        if (k == 0) a.xyz = squid ? P - D * (LEAD_TAILS == 2 ? 0.22 : 0.5) * L : E;
        if (squid) {
            vec3 sa = normalize(cross(b.xyz - a.xyz + vec3(1e-6), u_chaseNrm));
            float ga = clamp((t - a.w) / trS, 0.0, 1.0), gb = clamp((t - b.w) / trS, 0.0, 1.0);
            float sw = (LEAD_TAILS == 2) ? 0.0 : 0.25;
            a.xyz += sa * L * (vo * (1.0 + 0.8 * ga) + sw * ga * sin(7.0 * t - 9.0 * ga + 1.7 * float(j)));
            b.xyz += sa * L * (vo * (1.0 + 0.8 * gb) + sw * gb * sin(7.0 * t - 9.0 * gb + 1.7 * float(j)));
        }
        vec3 d1 = b.xyz - a.xyz, w0 = ro - a.xyz;
        float aa = dot(d1, d1), bb = dot(rd, d1), dd = dot(rd, w0), ee = dot(d1, w0);
        float den = aa - bb * bb;
        float sc = den > 1e-12 ? clamp((ee - bb * dd) / den, 0.0, 1.0) : 0.0;
        vec3 C = a.xyz + d1 * sc;
        float tc = dot(C - ro, rd);
        if (tc <= 0.0 || tc >= tVis) continue;
        float ag = clamp((t - mix(a.w, b.w, sc)) / trS, 0.0, 1.0);
        float rr = trR * (1.0 + 1.5 * ag);
        vec3 off = ro + rd * tc - C;
        float vis = (tc > tRing && C.y * ro.y < 0.0) ? trRing : 1.0;
        float e2 = dot(off, off) / (rr * rr);
        acc += (col * 0.9 + vec3(0.5) * exp(-4.0 * e2)) * (1.0 - ag) * (1.0 - ag) * vis * exp(-e2) * (trR / rr) * (squid ? 0.7 : 1.0);
    }
    }
    return acc;
}
uniform vec4 u_pulsar;
uniform vec2 u_pulsarW;
uniform vec4 u_pulsarR;
#define PS_BEAM_K   0.18
#define PS_FAN_K    0.11
#define PS_RAY_K    0.5
#define PS_SCAN     55.0
#define PS_BEAM_COL nsTint(vec3(0.62, 0.80, 1.00))
float holo(vec3 ro, vec3 rd, vec3 P, float t) {
    if (u_pulsar.z <= 0.0) return 1.0;
    float rr = length(ro + rd * dot(P - ro, rd));
    float sc = 0.5 + 0.5 * sin(rr * PS_SCAN * 6.2832 - t * 9.0);
    float fl = 1.0 - 0.35 * u_pulsar.w * fract(sin(floor(t * 40.0) * 91.7) * 4375.5);
    return (0.45 + 0.75 * smoothstep(0.2, 0.8, sc)) * fl;
}
float holoRay(vec3 ro, vec3 rd, vec3 P, float tVis, float t) {
    float aa = dot(P, P);
    if (aa < 1e-4) return 0.0;
    float bb = dot(rd, P), dd = dot(rd, ro), ee = dot(P, ro), den = aa - bb * bb;
    float sc = den > 1e-9 ? clamp((ee - bb * dd) / den, 0.0, 1.0) : 0.0;
    vec3 C = P * sc;
    float tc = dot(C - ro, rd);
    if (tc <= 0.0 || tc >= tVis) return 0.0;
    float w = 0.003 + 0.004 * sc, d = length(ro + rd * tc - C);
    float fl = 0.75 + 0.25 * sin(t * 23.0 + sc * 40.0);
    return exp(-d * d / (w * w)) * smoothstep(0.08, 0.3, sc * sqrt(aa)) * (0.3 + 0.7 * sc) * fl * PS_RAY_K;
}
vec3 pulsarBeams(vec3 ro, vec3 rd, float tVis, float tRing, float trRing) {
    vec3 acc = vec3(0.0);
    for (int k = 0; k < 2; k++) {
        vec3 b = (k == 0 ? 1.0 : -1.0) * vec3(u_pulsar.x, 0.0, u_pulsar.y);
        float R = k == 0 ? u_pulsarR.x : u_pulsarR.y;
        float tw = tan(k == 0 ? u_pulsarW.x : u_pulsarW.y);
        float bb = dot(rd, b), dd = dot(rd, ro), ee = dot(b, ro);
        float den = 1.0 - bb * bb;
        if (den < 1e-5) continue;
        float sb = (ee - bb * dd) / den, tc = (ee * bb - dd) / den;
        if (sb < 0.1 || sb > R || tc <= 0.0 || tc >= tVis) continue;
        float dist = length(ro + rd * tc - b * sb), cw = max(sb * tw, 0.02);
        float x2 = dist * dist / (cw * cw);
        float g = (exp(-2.0 * x2) + 1.5 * exp(-40.0 * x2)) * smoothstep(0.1, 0.35, sb) * (1.0 - smoothstep(0.8 * R, R, sb));
        float lit = (k == 0) ? 1.0 + 1.5 * (1.0 - smoothstep(u_pulsarR.z - 0.1, u_pulsarR.z + 0.25, sb)) : 1.0;
        float vis = tc > tRing ? sqrt(trRing) : 1.0;
        acc += mix(PS_BEAM_COL, vec3(1.0), 0.35 * exp(-40.0 * x2)) * g * lit * vis * (0.04 / cw + 0.3) * PS_BEAM_K * (1.0 + 2.0 * u_pulsar.w);
    }
    if (abs(rd.y) > 1e-3) {
        float t0 = -ro.y / rd.y, L = min(2.2 / abs(rd.y), 6.0), dt = 2.0 * L / 12.0, tw2 = tan(u_pulsar.z);
        float ab = atan(u_pulsar.y, u_pulsar.x);
        vec3 sh = vec3(0.0);
        for (int k = 0; k < 12; k++) {
            float tk = t0 - L + (float(k) + 0.5) * dt;
            if (tk <= 0.0 || tk >= tVis) continue;
            vec3 Q = ro + rd * tk;
            float rho = length(Q.xz);
            if (rho < 0.2 || rho > u_pulsarR.x) continue;
            float az = abs(mod(atan(Q.z, Q.x) - ab + 3.14159265, 6.2831853) - 3.14159265);
            float el = abs(Q.y) / (rho * tw2);
            float dn = exp(-pow(az / u_pulsarR.w, 4.0)) * exp(-pow(el, 4.0)) * smoothstep(0.25, 0.7, rho)
                     * (1.0 - smoothstep(0.8 * u_pulsarR.x, u_pulsarR.x, rho)) / max(rho, 0.5);
            sh += PS_BEAM_COL * dn * ((tk > tRing) ? sqrt(trRing) : 1.0);
        }
        acc += sh * dt * PS_FAN_K * (1.0 + 1.0 * u_pulsar.w);
    }
    return acc;
}
uniform sampler2D u_shots;
uniform float u_shotN;
#define SHOT_T 24
#define NUKE_R 0.026
#define LASER_R 0.0032
vec3 shipShots(vec3 ro, vec3 rd, float tVis, float tRing, float trRing, float t) {
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        if (float(i) >= u_shotN) break;
        vec4 inf = texelFetch(u_shots, ivec2(SHOT_T, i), 0);
        int n = int(inf.y);
        if (n < 1) continue;
        bool nuke = inf.x < 0.5;
        vec3 col = nuke ? LEAD_COL : SHIP_COL;
        vec3 P = texelFetch(u_shots, ivec2(0, i), 0).xyz;
        float tH = dot(P - ro, rd);
        if (tH > 0.0 && tH < tVis) {
            vec3 o = ro + rd * tH - P;
            float d2 = dot(o, o);
            float vis = (tH > tRing && P.y * ro.y < 0.0) ? trRing : 1.0;
            if (nuke) {
                float pu = 0.85 + 0.15 * sin(t * 30.0 + float(i));
                acc += (mix(col, vec3(1.0), exp(-d2 / (0.12 * NUKE_R * NUKE_R))) * 2.4 * exp(-d2 / (NUKE_R * NUKE_R))
                        + col * 0.45 * exp(-d2 / (9.0 * NUKE_R * NUKE_R))) * pu * vis;
            } else {
                acc += (mix(col, vec3(1.0), 0.6) * 3.0 * exp(-d2 / (4.0 * LASER_R * LASER_R))
                        + col * 0.35 * exp(-d2 / (64.0 * LASER_R * LASER_R))) * vis;
            }
        }
        int nT = nuke ? min(n - 1, 10) : n - 1;
        for (int k = 0; k < SHOT_T - 1; k++) {
            if (k >= nT) break;
            vec3 a = texelFetch(u_shots, ivec2(k, i), 0).xyz, b = texelFetch(u_shots, ivec2(k + 1, i), 0).xyz;
            vec3 d1 = b - a, w0 = ro - a;
            float aa = dot(d1, d1), bb = dot(rd, d1), dd = dot(rd, w0), ee = dot(d1, w0);
            float den = aa - bb * bb;
            float sc = den > 1e-12 ? clamp((ee - bb * dd) / den, 0.0, 1.0) : 0.0;
            vec3 C = a + d1 * sc;
            float tc = dot(C - ro, rd);
            if (tc <= 0.0 || tc >= tVis) continue;
            float ag = (float(k) + sc) / float(max(nT, 1));
            float rr = nuke ? NUKE_R * (0.7 - 0.5 * ag) : LASER_R;
            vec3 off = ro + rd * tc - C;
            float e2 = dot(off, off) / (rr * rr);
            float vis = (tc > tRing && C.y * ro.y < 0.0) ? trRing : 1.0;
            if (nuke) acc += (col * 1.1 + vec3(0.4) * exp(-3.0 * e2)) * exp(-e2) * (1.0 - ag) * (1.0 - ag) * vis;
            else acc += (mix(col, vec3(1.0), 0.5) * 1.8 * exp(-e2) + col * 0.25 * exp(-e2 / 16.0)) * pow(1.0 - ag, 0.5) * vis;
        }
    }
    return acc;
}
vec3 chase(vec3 ro, vec3 rd, float tVis, float tRing, float trRing, float t) {
    vec3 acc = chaseShip(0, ro, rd, tVis, tRing, trRing, t, 0.085, 1.1, 0.011, LEAD_COL, true)
        * holo(ro, rd, texelFetch(u_chase, ivec2(0, 0), 0).xyz, t);
    for (int i = 1; i <= CH_SW; i++) acc += chaseShip(i, ro, rd, tVis, tRing, trRing, t, 0.035, 0.40, 0.007, SHIP_COL, false)
        * holo(ro, rd, texelFetch(u_chase, ivec2(0, i), 0).xyz, t + 0.3 * float(i));
    if (u_pulsar.z > 0.0) {
        acc += pulsarBeams(ro, rd, tVis, tRing, trRing);
        for (int i = 0; i <= CH_SW; i++)
            acc += holoRay(ro, rd, texelFetch(u_chase, ivec2(0, i), 0).xyz, tVis, t + 0.7 * float(i)) * (i == 0 ? LEAD_COL : SHIP_COL) * (i == 0 ? 1.0 : 0.45);
    }
    return acc;
}

#define FF_GRAIN_LIT 1.0
#define FF_GRAIN_CON 12.0
// @lib flowfluid begin (generated by tools/lib_sync.py from shaders/_lib/flowfluid.glsl -- edit that)
// flowfluid -- a fluid-looking surface texture (clean-room, written 2026-10-23 for audio-reactive-shaders,
// GPL-3.0-or-later). Canonical copy: shaders/_lib/flowfluid.glsl. Consumers carry a pasted copy between the
// "@lib flowfluid begin/end" markers; tools/lib_sync.py refreshes them -- edit THIS file, then run the tool.
//
// Pieces (all written from scratch here):
//   ffHash     integer hash: 3 coordinates folded through the "lowbias32" mixer (Chris Wellons, public domain)
//   ffNoise    3D gradient noise, quintic fade; ffNoiseD also returns its analytic gradient
//   ffFbm      octave sum with a fixed rotation between octaves (axis (1,2,3), 0.71 rad) and lacunarity FF_LAC
//   ffAdvect   moves the lookup point along a divergence-free flow v = grad(a) x grad(b) of two slow noise
//              potentials, FF_STEPS steps: the pattern swirls and stretches like a fluid instead of just warping
//   flowFluid  two-deck emission/absorption: a hot deep layer (temperature field) seen through a cloud deck whose
//              optical depth is the advected pattern -> col = hot * exp(-tau) + deck glow * (1 - exp(-tau)).
//              Thin deck = bright breakthroughs, thick deck = dark filaments (stronger where darkMask = 1).
// Knobs (#define before the paste to override): FF_LAC, FF_FLOW_F, FF_FLOW_K, FF_STEPS, FF_TAU, FF_DECK_LO,
// FF_DECK_HI, FF_DARK_TAU, FF_DECK_DARK, FF_TEMP_F, FF_TEMP_K, FF_GAIN, FF_GRAIN, FF_GRAIN_F, FF_GRAIN_OCT,
// FF_GRAIN_LIT, FF_GRAIN_FLOW, FF_GRAIN_MID, FF_GRAIN_CON.
// Texture (2026-10-23 v2): octaves keep FF_GAIN of the previous one's strength (higher = grainier), and a fine
// RIDGED grain (1 - |noise|, sharp creases) carried by the same flow roughens the deck's optical depth.
#ifndef FF_LAC
#define FF_LAC       2.03    // frequency step per octave
#endif
#ifndef FF_GAIN
#define FF_GAIN      0.56    // amplitude kept per octave (0.5 = smooth, 0.6 = gritty)
#endif
#ifndef FF_GRAIN
#define FF_GRAIN     0.55    // ridged grain's share of the deck's optical depth (0 = off)
#endif
#ifndef FF_GRAIN_F
#define FF_GRAIN_F   9.0     // grain frequency x the pattern's
#endif
#ifndef FF_GRAIN_OCT
#define FF_GRAIN_OCT 4
#endif
#ifndef FF_GRAIN_LIT
#define FF_GRAIN_LIT 0.45    // ... and its share of the glow itself (the visible grit: hot layer + deck glow x 1 +- this)
#endif
#ifndef FF_GRAIN_FLOW
#define FF_GRAIN_FLOW 0.25   // how far the grain follows the flow (0 = painted on, 1 = fully advected)
#endif
#ifndef FF_GRAIN_MID
#define FF_GRAIN_MID 0.62    // the ridged sum's typical value ...
#endif
#ifndef FF_GRAIN_CON
#define FF_GRAIN_CON 3.5     // ... and the contrast stretch around it
#endif
#ifndef FF_FLOW_F
#define FF_FLOW_F    0.55    // flow potentials' frequency (x the pattern's): lower = bigger swirls
#endif
#ifndef FF_FLOW_K
#define FF_FLOW_K    0.30    // advection length per step (pattern units)
#endif
#ifndef FF_STEPS
#define FF_STEPS     3
#endif
#ifndef FF_TAU
#define FF_TAU       2.8     // optical depth of a fully thick deck
#endif
#ifndef FF_DECK_LO
#define FF_DECK_LO   -0.45   // pattern value where the deck starts / is full
#endif
#ifndef FF_DECK_HI
#define FF_DECK_HI   0.50
#endif
#ifndef FF_DARK_TAU
#define FF_DARK_TAU  1.2     // extra optical depth inside darkMask (the black-structure regions)
#endif
#ifndef FF_DECK_DARK
#define FF_DECK_DARK 0.08    // deck glow left where it is thickest (x cDeck)
#endif
#ifndef FF_TEMP_F
#define FF_TEMP_F    0.7     // deep layer's frequency (x the pattern's)
#endif
#ifndef FF_TEMP_K
#define FF_TEMP_K    2.2     // how strongly the flow drags the deep layer along
#endif

uint ffMix(uint x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
uint ffHash(ivec3 c) {
    return ffMix(uint(c.x) ^ ffMix(uint(c.y) + 0x9e3779b9U ^ ffMix(uint(c.z) + 0x632be5abU)));
}
vec3 ffGrad(ivec3 c) {
    uint h = ffHash(c);
    return vec3(float(h & 1023U), float((h >> 10) & 1023U), float((h >> 20) & 1023U)) * (2.0 / 1023.0) - 1.0;
}
// gradient noise, value in about [-1, 1]
float ffNoise(vec3 p) {
    ivec3 i = ivec3(floor(p));
    vec3 f = p - floor(p);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float v000 = dot(ffGrad(i),                   f);
    float v100 = dot(ffGrad(i + ivec3(1, 0, 0)), f - vec3(1, 0, 0));
    float v010 = dot(ffGrad(i + ivec3(0, 1, 0)), f - vec3(0, 1, 0));
    float v110 = dot(ffGrad(i + ivec3(1, 1, 0)), f - vec3(1, 1, 0));
    float v001 = dot(ffGrad(i + ivec3(0, 0, 1)), f - vec3(0, 0, 1));
    float v101 = dot(ffGrad(i + ivec3(1, 0, 1)), f - vec3(1, 0, 1));
    float v011 = dot(ffGrad(i + ivec3(0, 1, 1)), f - vec3(0, 1, 1));
    float v111 = dot(ffGrad(i + ivec3(1, 1, 1)), f - vec3(1, 1, 1));
    float x00 = mix(v000, v100, u.x), x10 = mix(v010, v110, u.x);
    float x01 = mix(v001, v101, u.x), x11 = mix(v011, v111, u.x);
    return 1.6 * mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
// the same noise + its analytic gradient: (value, d/dx, d/dy, d/dz)
vec4 ffNoiseD(vec3 p) {
    ivec3 i = ivec3(floor(p));
    vec3 f = p - floor(p);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec3 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    vec3 g000 = ffGrad(i),                   g100 = ffGrad(i + ivec3(1, 0, 0));
    vec3 g010 = ffGrad(i + ivec3(0, 1, 0)), g110 = ffGrad(i + ivec3(1, 1, 0));
    vec3 g001 = ffGrad(i + ivec3(0, 0, 1)), g101 = ffGrad(i + ivec3(1, 0, 1));
    vec3 g011 = ffGrad(i + ivec3(0, 1, 1)), g111 = ffGrad(i + ivec3(1, 1, 1));
    float v000 = dot(g000, f),                   v100 = dot(g100, f - vec3(1, 0, 0));
    float v010 = dot(g010, f - vec3(0, 1, 0)), v110 = dot(g110, f - vec3(1, 1, 0));
    float v001 = dot(g001, f - vec3(0, 0, 1)), v101 = dot(g101, f - vec3(1, 0, 1));
    float v011 = dot(g011, f - vec3(0, 1, 1)), v111 = dot(g111, f - vec3(1, 1, 1));
    // trilinear blend written as a polynomial in u: value = k0 + k1 ux + k2 uy + k3 uz + k4 ux uy + k5 uy uz + k6 uz ux + k7 ux uy uz
    float k0 = v000, k1 = v100 - v000, k2 = v010 - v000, k3 = v001 - v000;
    float k4 = v000 - v100 - v010 + v110, k5 = v000 - v010 - v001 + v011, k6 = v000 - v100 - v001 + v101;
    float k7 = -v000 + v100 + v010 - v110 + v001 - v101 - v011 + v111;
    vec3 h0 = g000, h1 = g100 - g000, h2 = g010 - g000, h3 = g001 - g000;
    vec3 h4 = g000 - g100 - g010 + g110, h5 = g000 - g010 - g001 + g011, h6 = g000 - g100 - g001 + g101;
    vec3 h7 = -g000 + g100 + g010 - g110 + g001 - g101 - g011 + g111;
    float val = k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z;
    vec3 grad = h0 + h1 * u.x + h2 * u.y + h3 * u.z + h4 * u.x * u.y + h5 * u.y * u.z + h6 * u.z * u.x + h7 * u.x * u.y * u.z
              + du * vec3(k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
                          k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
                          k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y);
    return 1.6 * vec4(val, grad);
}
const mat3 FF_ROT = mat3(0.77562, 0.55715, -0.29664, -0.48811, 0.82740, 0.27777, 0.40020, -0.07065, 0.91370);
// octave sum, about [-1, 1]
float ffFbm(vec3 p, int oct) {
    float v = 0.0, a = 0.5, norm = 0.0;
    for (int k = 0; k < oct; k++) {
        v += a * ffNoise(p);
        norm += a;
        p = FF_ROT * p * FF_LAC + vec3(1.7, -3.1, 2.3);
        a *= FF_GAIN;
    }
    return v / norm;
}
// ridged octave sum, 0..1: thin bright creases where the noise crosses zero
float ffRidge(vec3 p, int oct) {
    float v = 0.0, a = 0.5, norm = 0.0;
    for (int k = 0; k < oct; k++) {
        float r = 1.0 - abs(ffNoise(p));
        v += a * r * r;
        norm += a;
        p = FF_ROT * p * FF_LAC + vec3(-4.3, 1.1, 0.7);
        a *= FF_GAIN;
    }
    return v / norm;
}
// the flow: v = grad(a) x grad(b) is divergence-free (no sources or sinks, like an incompressible fluid)
// grad of a + 0.5 n(2 R q): the second octave's gradient comes back through 2 R^T (chain rule), so v stays exact
vec3 ffPotGrad(vec3 q, vec3 o1, vec3 o2) {
    vec3 g1 = ffNoiseD(q + o1).yzw;
    vec3 g2 = ffNoiseD(2.0 * (FF_ROT * q) + o2).yzw;
    return g1 + transpose(FF_ROT) * g2;          // 0.5 amplitude x 2 frequency = 1
}
vec3 ffFlow(vec3 p, float tt) {
    vec3 q = p * FF_FLOW_F;
    vec3 ga = ffPotGrad(q, vec3(0.0, 0.0, tt), vec3(tt, 4.1, 0.0));
    vec3 gb = ffPotGrad(q, vec3(5.3, -tt, 1.9), vec3(-2.2, 0.0, 7.7 + tt));
    return cross(ga, gb) * FF_FLOW_F;            // d/dp = FF_FLOW_F d/dq
}
vec3 ffAdvect(vec3 p, float tt) {
    for (int k = 0; k < FF_STEPS; k++) {
        p += FF_FLOW_K * ffFlow(p, tt + 0.37 * float(k));
    }
    return p;
}

struct FFOut {
    vec3 col;      // the shaded surface (linear, before any gain)
    float tr;      // deck transmittance exp(-tau): 1 = looking straight into the deep layer
    float temp;    // deep layer temperature 0..1
    float deck;    // deck thickness 0..1
};
// P: the lookup point (pattern units); tt: the flow clock; oct: pattern octaves
// cCool / cHot: the deep layer from cool to hot; cDeck: the deck's own glow
// deckBreath / hotBreath: -1..1 slow local modulations (the consumer's breathing); darkMask: 0..1 black-structure regions
FFOut flowFluid(vec3 P, float tt, int oct, vec3 cCool, vec3 cHot, vec3 cDeck, float deckBreath, float hotBreath, float darkMask) {
    vec3 A = ffAdvect(P, tt);
    float pat = ffFbm(A, oct);
    float deck = smoothstep(FF_DECK_LO, FF_DECK_HI, pat);
    float temp = clamp(0.5 + 0.75 * ffFbm(P * FF_TEMP_F + FF_TEMP_K * (A - P) + vec3(0.0, 0.0, 0.5 * tt), 4), 0.0, 1.0);
    // the grain sits on the SURFACE and only drifts part of the way with the flow (FF_GRAIN_FLOW): fully advected it
    // is stretched into smooth streamlines and reads as marble, not grit
    vec3 G = mix(P, A, FF_GRAIN_FLOW) * FF_GRAIN_F;
    float grain = FF_GRAIN > 0.0 ? ffRidge(G + vec3(0.0, 0.0, 0.2 * tt), FF_GRAIN_OCT) : FF_GRAIN_MID;
    grain = clamp(0.5 + (grain - FF_GRAIN_MID) * FF_GRAIN_CON, 0.0, 1.0);   // octave sums bunch up around their mean: spread it
    float tau = FF_TAU * deck * deck * (1.0 + 0.35 * deckBreath) * (1.0 + FF_DARK_TAU * darkMask)
              * (1.0 + FF_GRAIN * 2.0 * (grain - 0.5));
    float tr = exp(-tau);
    vec3 hot = mix(cCool, cHot, temp * temp) * (0.85 + 0.30 * hotBreath);
    vec3 glow = cDeck * mix(1.0, FF_DECK_DARK, smoothstep(0.55, 1.0, deck) * (0.35 + 0.65 * darkMask));
    FFOut o;
    float lit = 1.0 + FF_GRAIN_LIT * 2.0 * (grain - 0.5);
    o.col = (hot * tr + glow * (1.0 - tr)) * lit;
    o.tr = tr;
    o.temp = temp;
    o.deck = deck;
    return o;
}
// @lib flowfluid end
#define FD_COOL     vec3(0.060, 0.010, 0.110)
#define FD_HOT      vec3(0.700, 0.260, 0.950)
#define FD_DECK     vec3(0.110, 0.025, 0.170)
#define FD_FLOOR    0.045
#define FD_TIME     1.0
#define FD_OCT      10
vec3 flowDwarf(vec3 n, float t) {
    vec3 m = rotY(n, CF_SPIN * t);
    vec3 P = vec3(m.x, m.y * CF_STRETCH, m.z) * CF_SCALE;
    float tt = FD_TIME * (CF_SPEED * t + u_churn);
    float tb = CF_BREATH * t;
    float latN = asin(clamp(n.y, -1.0, 1.0));
    vec3 ms = rotY(n, CF_SPIN * t + PS_SHEAR * sin(latN * PS_JETS) * (0.6 + 0.4 * cos(latN)));
    vec3 Ps = vec3(ms.x, ms.y * CF_STRETCH, ms.z) * CF_SCALE * PS_SCALE;
    float ph1 = PS_SPREAD * (0.5 + 0.5 * ffFbm(Ps, PS_OCT));
    float ph2 = PS_SPREAD * (0.5 + 0.5 * ffFbm(Ps * 0.8 + vec3(7.0, 2.0, -3.0), PS_OCT));
    float w1 = sin(0.7 * tb + ph1), w2 = sin(0.3 * tb + ph2);
    float osMask = smoothstep(1.0 - OS_COVER - 0.08, 1.0 - OS_COVER + 0.08,
                              0.5 + 0.5 * (0.65 * ffNoise(m * 1.4 + vec3(3.1, 7.7, 1.9)) + 0.35 * ffNoise(m * 2.9 + vec3(9.4, 2.2, 5.6))));
    FFOut f = flowFluid(P, tt, FD_OCT, FD_COOL, FD_HOT, FD_DECK, w2, w1, osMask);
    vec3 color = (f.col + FD_FLOOR * FD_DECK * (1.0 - 0.8 * osMask)) * CF_GAIN;
    float peak = smoothstep(0.2, 1.0, 0.5 + 0.5 * w1);
    color += ACC_COL * ACC_GAIN * smoothstep(0.62, 0.92, f.temp) * smoothstep(0.55, 0.95, f.tr) * peak;
    float luma = dot(color, vec3(0.30, 0.45, 0.25));
    float dark = 1.0 - smoothstep(0.05, UV_KNEE, luma);
    float drift = 0.65 + 0.35 * (0.5 + 0.5 * ffNoise(m * 1.2 + vec3(0.0, 0.0, 0.02 * t)));
    vec3 uv = UV_COL * UV_GLOW * dark * drift;
    color = color + uv - color * uv / max(CF_GAIN, 1.0);
    color += lightning(m, t);
    return color;
}

#define NS_TINT    0.75
#define NS_GAIN    7.0
#define NS_COL     nsTint(vec3(0.72, 0.86, 1.00))
#define NS_GLOW_COL  nsTint(vec3(0.70, 0.85, 1.00))
#define RING_R2MIN   1.0
#define NS_GLOW_CORE 1.6
#define NS_GLOW_HALO 0.12
#define NS_SPIKES    0.10
vec3 dwarfSurface(vec3 n, float t) {
    vec3 c = flowDwarf(n, t);
    c = u_dbHue.y > 0.002 ? max(dbHueRot(c, u_dbHue.y), 0.0) : c;
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return mix(c, NS_COL * l * NS_GAIN, NS_TINT);
}

vec4 ringAt(float r, float th) {
    float jf = clamp((r - R_IN) / DR - 0.5, 0.0, float(NR - 1));
    int j0 = int(floor(jf));
    float fj = jf - float(j0);
    vec4 res = vec4(0.0);
    for (int q = 0; q < 2; q++) {
        int j = min(j0 + q, NR - 1);
        float rel = mod(th - texelFetch(u_phase, ivec2(j, 0), 0).x, 6.28318530718);
        float kf = rel / DTH - 0.5;
        int k0 = int(floor(kf));
        float fk = kf - float(k0);
        vec4 a = texelFetch(u_ring, ivec2((k0 % NT + NT) % NT, j), 0);
        vec4 b = texelFetch(u_ring, ivec2(((k0 + 1) % NT + NT) % NT, j), 0);
        res += mix(a, b, fk) * (q == 0 ? 1.0 - fj : fj);
    }
    return res;
}
float lambertPhase(float a) { return (sin(a) + (3.14159265 - a) * cos(a)) / 3.14159265; }
float hgPhase(float cosT, float g) { float q = 1.0 + g * g - 2.0 * g * cosT; return (1.0 - g * g) / (q * sqrt(q)); }

uniform samplerCube u_skycube;
uniform sampler2D u_nebula;
#define SKY64_GAIN    8.0
#define SKY_GAIN      3.20
#define SKY_STAR_GCUT 11.0
#define SKY_LOD_OPEN  2.0
#define NEB_W         16383.0
#define NEB_GAIN      1.0
#define NEB_DUST      1.0
#define SKY_EXPO      1.0
#define SKY_ZOOM 1.5
#define SKY_MAT mat3(0.559086, -0.677617, 0.477763, -0.547029, 0.131545, 0.826714, 0.623043, 0.723555, 0.297131)
vec3 skyDir(vec3 d) {
    float ca = cos(CAM_ORBIT * iTime), sa = sin(CAM_ORBIT * iTime);
    d = vec3(ca * d.x + sa * d.z, d.y, -sa * d.x + ca * d.z);
    vec3 ro0 = CAM_D * vec3(0.0, sin(CAM_EL), -cos(CAM_EL));
    vec3 D0 = normalize(-ro0), F0 = normalize(AIM_OFF * vec3(0.0, 0.0, -1.0) - ro0);
    float th = acos(clamp(dot(d, D0), -1.0, 1.0));
    vec3 tng = d - D0 * dot(d, D0);
    float tl = length(tng);
    if (tl > 1e-6) {
        const float SKY_LIN = 0.8;
        float th2;
        if (th <= SKY_LIN) th2 = th * SKY_ZOOM;
        else {
            float h = 3.14159265 - SKY_LIN, u = (th - SKY_LIN) / h;
            float u2 = u * u, u3 = u2 * u;
            th2 = (2.0*u3 - 3.0*u2 + 1.0) * (SKY_LIN * SKY_ZOOM) + (u3 - 2.0*u2 + u) * h * SKY_ZOOM
                + (-2.0*u3 + 3.0*u2) * 3.14159265 + (u3 - u2) * h * 1.0;
        }
        d = cos(th2) * D0 + sin(th2) * tng / tl;
    }
    vec3 ax = cross(D0, F0);
    float qs = length(ax), qc = dot(D0, F0);
    if (qs > 1e-6) { ax /= qs; d = d * qc + cross(ax, d) * qs + ax * dot(ax, d) * (1.0 - qc); }
    return SKY_MAT * d;
}
float skyWrapU(float u, float w) { return (0.5 + fract(u) * w) / (w + 1.0); }
vec3 skyL(vec3 dir, float lod) {
    vec3 d = skyDir(dir);
    vec3 c = textureLod(u_skycube, d, lod).rgb * SKY64_GAIN;
    float pk = max(c.r, max(c.g, c.b)) / SKY64_GAIN;
    float thr = 0.5 * 0.13 * pow(10.0, 0.148 * (11.7 - SKY_STAR_GCUT));
    c *= smoothstep(thr * 0.5, thr * 1.5, pk) * SKY_GAIN;
    vec2 nuv = vec2(atan(d.z, d.x) / 6.28318530718 + 0.5, 1.0 - acos(clamp(d.y, -1.0, 1.0)) / 3.14159265);
    vec4 nb = textureLod(u_nebula, vec2(skyWrapU(nuv.x, NEB_W), nuv.y), max(lod - 1.0, 0.0));
    c *= 1.0 - clamp(nb.a * NEB_DUST, 0.0, 1.0);
    c += max(nb.rgb, 0.0) * NEB_GAIN;
    c = 0.18 * pow(max(c * SKY_EXPO, 0.0) / 0.18 + 1e-6, vec3(1.4));
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return max(max(mix(vec3(l), c, 1.2), 0.0) - 0.02, 0.0);
}
vec3 sky(vec3 d) { return skyL(d, SKY_LOD_OPEN + log2(SKY_ZOOM)); }

const vec3 MOON_METAL[9] = vec3[9](
    vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0),
    vec3(0.97, 0.98, 1.00), vec3(1.00, 0.76, 0.33),
    vec3(0.0), vec3(0.0)
);
#define ROCK_COL   vec3(0.55, 0.53, 0.50)
#define ROCK_LIT   0.55
const int MOON_MAT[9] = int[9](2, 6, 1, 1, 5, 0, 7, 3, 4);
vec3 mirrorEnv(vec3 P, vec3 R, float t, mat3 toP);
uniform sampler2D u_moonMap;
#define GEPS       1e-4
#define TEPS       1e-5
#define FACE_N 204
#define FACE_SQ 120
#define FACE_PG 180
#define STONE_RB 0.900000
#define R_TRI 0.73484692
#define R_SQ 0.90000000
#define R_PG 0.66914646
const vec4 FACE_NW[FACE_N] = vec4[](
    vec4(0.90776942, 0.35682209, -0.22052818, 0.51961524),
    vec4(0.67441289, -0.73440089, 0.07630591, 0.51961524),
    vec4(0.67441289, 0.73440089, 0.07630591, 0.51961524),
    vec4(0.90776942, -0.35682209, -0.22052818, 0.51961524),
    vec4(0.23335653, -0.73440089, 0.63733827, 0.51961524),
    vec4(0.23335653, 0.73440089, 0.63733827, 0.51961524),
    vec4(0.67441289, 0.73440089, -0.07630591, 0.51961524),
    vec4(0.90776942, -0.35682209, 0.22052818, 0.51961524),
    vec4(-0.23335653, 0.73440089, 0.63733827, 0.51961524),
    vec4(0.95492907, 0.28051618, -0.09706262, 0.51961524),
    vec4(0.09706262, 0.95492907, 0.28051618, 0.51961524),
    vec4(-0.28051618, 0.09706262, 0.95492907, 0.51961524),
    vec4(0.81070680, 0.51387271, 0.28051618, 0.51961524),
    vec4(-0.28051618, 0.81070680, 0.51387271, 0.51961524),
    vec4(-0.51387271, -0.28051618, 0.81070680, 0.51961524),
    vec4(0.22052818, 0.90776942, 0.35682209, 0.51961524),
    vec4(0.07630591, -0.67441289, 0.73440089, 0.51961524),
    vec4(-0.63733827, 0.23335653, 0.73440089, 0.51961524),
    vec4(-0.51387271, 0.28051618, 0.81070680, 0.51961524),
    vec4(0.81070680, -0.51387271, 0.28051618, 0.51961524),
    vec4(-0.28051618, -0.81070680, 0.51387271, 0.51961524),
    vec4(0.07630591, 0.67441289, 0.73440089, 0.51961524),
    vec4(0.22052818, -0.90776942, 0.35682209, 0.51961524),
    vec4(-0.63733827, -0.23335653, 0.73440089, 0.51961524),
    vec4(0.90776942, 0.35682209, 0.22052818, 0.51961524),
    vec4(0.67441289, -0.73440089, -0.07630591, 0.51961524),
    vec4(-0.23335653, -0.73440089, 0.63733827, 0.51961524),
    vec4(-0.28051618, -0.09706262, 0.95492907, 0.51961524),
    vec4(0.95492907, -0.28051618, -0.09706262, 0.51961524),
    vec4(0.09706262, -0.95492907, 0.28051618, 0.51961524),
    vec4(0.23335653, 0.73440089, -0.63733827, 0.51961524),
    vec4(0.23335653, -0.73440089, -0.63733827, 0.51961524),
    vec4(-0.23335653, -0.73440089, -0.63733827, 0.51961524),
    vec4(-0.28051618, -0.09706262, -0.95492907, 0.51961524),
    vec4(0.95492907, -0.28051618, 0.09706262, 0.51961524),
    vec4(0.09706262, -0.95492907, -0.28051618, 0.51961524),
    vec4(-0.51387271, 0.28051618, -0.81070680, 0.51961524),
    vec4(0.81070680, -0.51387271, -0.28051618, 0.51961524),
    vec4(-0.28051618, -0.81070680, -0.51387271, 0.51961524),
    vec4(0.07630591, 0.67441289, -0.73440089, 0.51961524),
    vec4(0.22052818, -0.90776942, -0.35682209, 0.51961524),
    vec4(-0.63733827, -0.23335653, -0.73440089, 0.51961524),
    vec4(0.81070680, 0.51387271, -0.28051618, 0.51961524),
    vec4(-0.28051618, 0.81070680, -0.51387271, 0.51961524),
    vec4(-0.51387271, -0.28051618, -0.81070680, 0.51961524),
    vec4(0.22052818, 0.90776942, -0.35682209, 0.51961524),
    vec4(0.07630591, -0.67441289, -0.73440089, 0.51961524),
    vec4(-0.63733827, 0.23335653, -0.73440089, 0.51961524),
    vec4(-0.23335653, 0.73440089, -0.63733827, 0.51961524),
    vec4(0.95492907, 0.28051618, 0.09706262, 0.51961524),
    vec4(0.09706262, 0.95492907, -0.28051618, 0.51961524),
    vec4(-0.28051618, 0.09706262, -0.95492907, 0.51961524),
    vec4(-0.90776942, 0.35682209, 0.22052818, 0.51961524),
    vec4(-0.67441289, -0.73440089, -0.07630591, 0.51961524),
    vec4(-0.67441289, 0.73440089, -0.07630591, 0.51961524),
    vec4(-0.90776942, -0.35682209, 0.22052818, 0.51961524),
    vec4(-0.67441289, 0.73440089, 0.07630591, 0.51961524),
    vec4(-0.90776942, -0.35682209, -0.22052818, 0.51961524),
    vec4(-0.09706262, 0.95492907, -0.28051618, 0.51961524),
    vec4(-0.95492907, 0.28051618, 0.09706262, 0.51961524),
    vec4(0.28051618, 0.09706262, -0.95492907, 0.51961524),
    vec4(0.28051618, 0.81070680, -0.51387271, 0.51961524),
    vec4(-0.81070680, 0.51387271, -0.28051618, 0.51961524),
    vec4(0.51387271, -0.28051618, -0.81070680, 0.51961524),
    vec4(0.63733827, 0.23335653, -0.73440089, 0.51961524),
    vec4(-0.22052818, 0.90776942, -0.35682209, 0.51961524),
    vec4(-0.07630591, -0.67441289, -0.73440089, 0.51961524),
    vec4(0.51387271, 0.28051618, -0.81070680, 0.51961524),
    vec4(0.28051618, -0.81070680, -0.51387271, 0.51961524),
    vec4(-0.81070680, -0.51387271, -0.28051618, 0.51961524),
    vec4(0.63733827, -0.23335653, -0.73440089, 0.51961524),
    vec4(-0.07630591, 0.67441289, -0.73440089, 0.51961524),
    vec4(-0.22052818, -0.90776942, -0.35682209, 0.51961524),
    vec4(-0.90776942, 0.35682209, -0.22052818, 0.51961524),
    vec4(-0.67441289, -0.73440089, 0.07630591, 0.51961524),
    vec4(0.28051618, -0.09706262, -0.95492907, 0.51961524),
    vec4(-0.09706262, -0.95492907, -0.28051618, 0.51961524),
    vec4(-0.95492907, -0.28051618, 0.09706262, 0.51961524),
    vec4(0.28051618, -0.09706262, 0.95492907, 0.51961524),
    vec4(-0.09706262, -0.95492907, 0.28051618, 0.51961524),
    vec4(-0.95492907, -0.28051618, -0.09706262, 0.51961524),
    vec4(0.51387271, 0.28051618, 0.81070680, 0.51961524),
    vec4(0.28051618, -0.81070680, 0.51387271, 0.51961524),
    vec4(-0.81070680, -0.51387271, 0.28051618, 0.51961524),
    vec4(0.63733827, -0.23335653, 0.73440089, 0.51961524),
    vec4(-0.07630591, 0.67441289, 0.73440089, 0.51961524),
    vec4(-0.22052818, -0.90776942, 0.35682209, 0.51961524),
    vec4(0.28051618, 0.81070680, 0.51387271, 0.51961524),
    vec4(-0.81070680, 0.51387271, 0.28051618, 0.51961524),
    vec4(0.51387271, -0.28051618, 0.81070680, 0.51961524),
    vec4(0.63733827, 0.23335653, 0.73440089, 0.51961524),
    vec4(-0.22052818, 0.90776942, 0.35682209, 0.51961524),
    vec4(-0.07630591, -0.67441289, 0.73440089, 0.51961524),
    vec4(-0.09706262, 0.95492907, 0.28051618, 0.51961524),
    vec4(-0.95492907, 0.28051618, -0.09706262, 0.51961524),
    vec4(0.28051618, 0.09706262, 0.95492907, 0.51961524),
    vec4(-0.73440089, 0.63733827, 0.23335653, 0.51961524),
    vec4(0.73440089, 0.63733827, 0.23335653, 0.51961524),
    vec4(0.35682209, -0.22052818, 0.90776942, 0.51961524),
    vec4(-0.73440089, 0.07630591, 0.67441289, 0.51961524),
    vec4(0.73440089, 0.07630591, 0.67441289, 0.51961524),
    vec4(-0.35682209, -0.22052818, 0.90776942, 0.51961524),
    vec4(0.73440089, 0.63733827, -0.23335653, 0.51961524),
    vec4(0.73440089, -0.07630591, 0.67441289, 0.51961524),
    vec4(-0.35682209, 0.22052818, 0.90776942, 0.51961524),
    vec4(-0.73440089, 0.63733827, -0.23335653, 0.51961524),
    vec4(0.35682209, 0.22052818, 0.90776942, 0.51961524),
    vec4(-0.73440089, -0.07630591, 0.67441289, 0.51961524),
    vec4(0.35682209, 0.22052818, -0.90776942, 0.51961524),
    vec4(-0.73440089, -0.07630591, -0.67441289, 0.51961524),
    vec4(0.73440089, -0.07630591, -0.67441289, 0.51961524),
    vec4(-0.35682209, 0.22052818, -0.90776942, 0.51961524),
    vec4(-0.73440089, -0.63733827, -0.23335653, 0.51961524),
    vec4(0.73440089, -0.63733827, -0.23335653, 0.51961524),
    vec4(0.73440089, 0.07630591, -0.67441289, 0.51961524),
    vec4(-0.35682209, -0.22052818, -0.90776942, 0.51961524),
    vec4(0.73440089, -0.63733827, 0.23335653, 0.51961524),
    vec4(0.35682209, -0.22052818, -0.90776942, 0.51961524),
    vec4(-0.73440089, 0.07630591, -0.67441289, 0.51961524),
    vec4(-0.73440089, -0.63733827, 0.23335653, 0.51961524),
    vec4(0.58405869, -0.32699283, -0.74293414, 0.00000000),
    vec4(0.20209268, 0.94502682, -0.25706586, 0.00000000),
    vec4(0.20209268, -0.94502682, -0.25706586, 0.00000000),
    vec4(0.58405869, 0.32699283, -0.74293414, 0.00000000),
    vec4(0.78615138, 0.00000000, -0.61803399, 0.00000000),
    vec4(0.74293414, -0.58405869, -0.32699283, 0.00000000),
    vec4(-0.20209268, 0.94502682, -0.25706586, 0.00000000),
    vec4(0.32699283, 0.74293414, -0.58405869, 0.00000000),
    vec4(0.94502682, -0.25706586, -0.20209268, 0.00000000),
    vec4(0.61803399, -0.78615138, 0.00000000, 0.00000000),
    vec4(-0.74293414, -0.58405869, 0.32699283, 0.00000000),
    vec4(-0.20209268, -0.94502682, 0.25706586, 0.00000000),
    vec4(0.74293414, -0.58405869, -0.32699283, 0.00000000),
    vec4(0.20209268, -0.94502682, -0.25706586, 0.00000000),
    vec4(-0.61803399, -0.78615138, -0.00000000, 0.00000000),
    vec4(0.94502682, 0.25706586, -0.20209268, 0.00000000),
    vec4(-0.32699283, 0.74293414, 0.58405869, 0.00000000),
    vec4(0.20209268, 0.94502682, 0.25706586, 0.00000000),
    vec4(-0.78615138, 0.00000000, 0.61803399, 0.00000000),
    vec4(0.74293414, 0.58405869, -0.32699283, 0.00000000),
    vec4(-0.58405869, -0.32699283, -0.74293414, 0.00000000),
    vec4(-0.58405869, 0.32699283, -0.74293414, 0.00000000),
    vec4(0.78615138, -0.00000000, 0.61803399, 0.00000000),
    vec4(0.74293414, 0.58405869, 0.32699283, 0.00000000),
    vec4(0.32699283, -0.74293414, 0.58405869, 0.00000000),
    vec4(0.94502682, 0.25706586, 0.20209268, 0.00000000),
    vec4(-0.74293414, 0.58405869, -0.32699283, 0.00000000),
    vec4(-0.20209268, 0.94502682, -0.25706586, 0.00000000),
    vec4(0.61803399, 0.78615138, -0.00000000, 0.00000000),
    vec4(-0.61803399, 0.78615138, 0.00000000, 0.00000000),
    vec4(0.74293414, 0.58405869, 0.32699283, 0.00000000),
    vec4(0.20209268, 0.94502682, 0.25706586, 0.00000000),
    vec4(0.94502682, -0.25706586, 0.20209268, 0.00000000),
    vec4(-0.32699283, -0.74293414, -0.58405869, 0.00000000),
    vec4(-0.74293414, 0.58405869, -0.32699283, 0.00000000),
    vec4(-0.78615138, -0.00000000, -0.61803399, 0.00000000),
    vec4(0.94502682, -0.25706586, 0.20209268, 0.00000000),
    vec4(-0.94502682, -0.25706586, 0.20209268, 0.00000000),
    vec4(-0.32699283, 0.74293414, -0.58405869, 0.00000000),
    vec4(0.32699283, 0.74293414, -0.58405869, 0.00000000),
    vec4(0.94502682, -0.25706586, -0.20209268, 0.00000000),
    vec4(0.00000000, -0.61803399, 0.78615138, 0.00000000),
    vec4(-0.58405869, -0.32699283, 0.74293414, 0.00000000),
    vec4(-0.25706586, -0.20209268, 0.94502682, 0.00000000),
    vec4(0.25706586, -0.20209268, 0.94502682, 0.00000000),
    vec4(-0.94502682, -0.25706586, -0.20209268, 0.00000000),
    vec4(0.00000000, 0.61803399, -0.78615138, 0.00000000),
    vec4(0.58405869, -0.32699283, 0.74293414, 0.00000000),
    vec4(-0.32699283, 0.74293414, 0.58405869, 0.00000000),
    vec4(0.32699283, 0.74293414, 0.58405869, 0.00000000),
    vec4(0.58405869, -0.32699283, -0.74293414, 0.00000000),
    vec4(-0.00000000, -0.61803399, -0.78615138, 0.00000000),
    vec4(0.25706586, -0.20209268, -0.94502682, 0.00000000),
    vec4(-0.25706586, -0.20209268, -0.94502682, 0.00000000),
    vec4(0.58405869, 0.32699283, 0.74293414, 0.00000000),
    vec4(0.00000000, -0.61803399, -0.78615138, 0.00000000),
    vec4(-0.25706586, 0.20209268, -0.94502682, 0.00000000),
    vec4(-0.25706586, 0.20209268, 0.94502682, 0.00000000),
    vec4(0.25706586, 0.20209268, -0.94502682, 0.00000000),
    vec4(-0.25706586, -0.20209268, -0.94502682, 0.00000000),
    vec4(0.85065081, 0.52573111, 0.00000000, 0.60186627),
    vec4(0.85065081, -0.52573111, -0.00000000, 0.60186627),
    vec4(-0.00000000, 0.85065081, 0.52573111, 0.60186627),
    vec4(0.85065081, 0.52573111, 0.00000000, 0.60186627),
    vec4(0.00000000, 0.85065081, 0.52573111, 0.60186627),
    vec4(-0.52573111, -0.00000000, 0.85065081, 0.60186627),
    vec4(-0.52573111, -0.00000000, 0.85065081, 0.60186627),
    vec4(0.00000000, -0.85065081, 0.52573111, 0.60186627),
    vec4(0.00000000, -0.85065081, 0.52573111, 0.60186627),
    vec4(0.85065081, -0.52573111, 0.00000000, 0.60186627),
    vec4(-0.00000000, -0.85065081, -0.52573111, 0.60186627),
    vec4(0.00000000, -0.85065081, -0.52573111, 0.60186627),
    vec4(-0.52573111, -0.00000000, -0.85065081, 0.60186627),
    vec4(0.00000000, 0.85065081, -0.52573111, 0.60186627),
    vec4(-0.52573111, -0.00000000, -0.85065081, 0.60186627),
    vec4(0.00000000, 0.85065081, -0.52573111, 0.60186627),
    vec4(-0.85065081, -0.52573111, -0.00000000, 0.60186627),
    vec4(-0.85065081, 0.52573111, 0.00000000, 0.60186627),
    vec4(-0.85065081, 0.52573111, 0.00000000, 0.60186627),
    vec4(0.52573111, -0.00000000, -0.85065081, 0.60186627),
    vec4(0.52573111, -0.00000000, -0.85065081, 0.60186627),
    vec4(-0.85065081, -0.52573111, -0.00000000, 0.60186627),
    vec4(0.52573111, -0.00000000, 0.85065081, 0.60186627),
    vec4(0.52573111, -0.00000000, 0.85065081, 0.60186627)
);
const vec4 FACE_U[FACE_N] = vec4[](
    vec4(0.32094496, -0.25231132, 0.91287093, 0.00000000),
    vec4(0.48595294, 0.51929985, 0.70297753, 0.00000000),
    vec4(0.48595294, -0.51929985, 0.70297753, 0.00000000),
    vec4(0.32094496, 0.25231132, 0.91287093, 0.00000000),
    vec4(0.79782688, 0.51929985, 0.30626775, 0.00000000),
    vec4(0.79782688, -0.51929985, 0.30626775, 0.00000000),
    vec4(0.23844096, -0.11881706, 0.96386120, 0.00000000),
    vec4(0.07343298, 0.65279411, 0.75396780, 0.00000000),
    vec4(0.88033088, -0.11881706, 0.45923856, 0.00000000),
    vec4(0.04008607, 0.20212790, 0.97853841, 0.00000000),
    vec4(0.64668925, -0.27475403, 0.71154988, 0.00000000),
    vec4(0.91367778, 0.33184916, 0.23466795, 0.00000000),
    vec4(-0.25841618, -0.11585090, 0.95906186, 0.00000000),
    vec4(0.51319499, -0.32574430, 0.79405387, 0.00000000),
    vec4(0.67820297, 0.44586687, 0.58416047, 0.00000000),
    vec4(0.15890313, -0.39437793, 0.90510543, 0.00000000),
    vec4(0.26088367, 0.72439390, 0.63811690, 0.00000000),
    vec4(0.76550631, 0.08250399, 0.63811690, 0.00000000),
    vec4(0.67820297, -0.44586687, 0.58416047, 0.00000000),
    vec4(-0.25841618, 0.11585090, 0.95906186, 0.00000000),
    vec4(0.51319499, 0.32574430, 0.79405387, 0.00000000),
    vec4(0.26088367, -0.72439390, 0.63811690, 0.00000000),
    vec4(0.15890313, 0.39437793, 0.90510543, 0.00000000),
    vec4(0.76550631, -0.08250399, 0.63811690, 0.00000000),
    vec4(0.07343298, -0.65279411, 0.75396780, 0.00000000),
    vec4(0.23844096, 0.11881706, 0.96386120, 0.00000000),
    vec4(0.88033088, 0.11881706, 0.45923856, 0.00000000),
    vec4(0.91367778, -0.33184916, 0.23466795, 0.00000000),
    vec4(0.04008607, -0.20212790, 0.97853841, 0.00000000),
    vec4(0.64668925, 0.27475403, 0.71154988, 0.00000000),
    vec4(0.79782688, -0.51929985, -0.30626775, 0.00000000),
    vec4(0.79782688, 0.51929985, -0.30626775, 0.00000000),
    vec4(0.88033088, 0.11881706, -0.45923856, 0.00000000),
    vec4(0.91367778, -0.33184916, -0.23466795, 0.00000000),
    vec4(0.04008607, -0.20212790, -0.97853841, 0.00000000),
    vec4(0.64668925, 0.27475403, -0.71154988, 0.00000000),
    vec4(0.67820297, -0.44586687, -0.58416047, 0.00000000),
    vec4(-0.25841618, 0.11585090, -0.95906186, 0.00000000),
    vec4(0.51319499, 0.32574430, -0.79405387, 0.00000000),
    vec4(0.26088367, -0.72439390, -0.63811690, 0.00000000),
    vec4(0.15890313, 0.39437793, -0.90510543, 0.00000000),
    vec4(0.76550631, -0.08250399, -0.63811690, 0.00000000),
    vec4(-0.25841618, -0.11585090, -0.95906186, 0.00000000),
    vec4(0.51319499, -0.32574430, -0.79405387, 0.00000000),
    vec4(0.67820297, 0.44586687, -0.58416047, 0.00000000),
    vec4(0.15890313, -0.39437793, -0.90510543, 0.00000000),
    vec4(0.26088367, 0.72439390, -0.63811690, 0.00000000),
    vec4(0.76550631, 0.08250399, -0.63811690, 0.00000000),
    vec4(0.88033088, -0.11881706, -0.45923856, 0.00000000),
    vec4(0.04008607, 0.20212790, -0.97853841, 0.00000000),
    vec4(0.64668925, -0.27475403, -0.71154988, 0.00000000),
    vec4(0.91367778, 0.33184916, -0.23466795, 0.00000000),
    vec4(-0.32094496, -0.25231132, -0.91287093, 0.00000000),
    vec4(-0.48595294, 0.51929985, -0.70297753, 0.00000000),
    vec4(-0.48595294, -0.51929985, -0.70297753, 0.00000000),
    vec4(-0.32094496, 0.25231132, -0.91287093, 0.00000000),
    vec4(-0.23844096, -0.11881706, -0.96386120, 0.00000000),
    vec4(-0.07343298, 0.65279411, -0.75396780, 0.00000000),
    vec4(-0.64668925, -0.27475403, -0.71154988, 0.00000000),
    vec4(-0.04008607, 0.20212790, -0.97853841, 0.00000000),
    vec4(-0.91367778, 0.33184916, -0.23466795, 0.00000000),
    vec4(-0.51319499, -0.32574430, -0.79405387, 0.00000000),
    vec4(0.25841618, -0.11585090, -0.95906186, 0.00000000),
    vec4(-0.67820297, 0.44586687, -0.58416047, 0.00000000),
    vec4(-0.76550631, 0.08250399, -0.63811690, 0.00000000),
    vec4(-0.15890313, -0.39437793, -0.90510543, 0.00000000),
    vec4(-0.26088367, 0.72439390, -0.63811690, 0.00000000),
    vec4(-0.67820297, -0.44586687, -0.58416047, 0.00000000),
    vec4(-0.51319499, 0.32574430, -0.79405387, 0.00000000),
    vec4(0.25841618, 0.11585090, -0.95906186, 0.00000000),
    vec4(-0.76550631, -0.08250399, -0.63811690, 0.00000000),
    vec4(-0.26088367, -0.72439390, -0.63811690, 0.00000000),
    vec4(-0.15890313, 0.39437793, -0.90510543, 0.00000000),
    vec4(-0.07343298, -0.65279411, -0.75396780, 0.00000000),
    vec4(-0.23844096, 0.11881706, -0.96386120, 0.00000000),
    vec4(-0.91367778, -0.33184916, -0.23466795, 0.00000000),
    vec4(-0.64668925, 0.27475403, -0.71154988, 0.00000000),
    vec4(-0.04008607, -0.20212790, -0.97853841, 0.00000000),
    vec4(-0.91367778, -0.33184916, 0.23466795, 0.00000000),
    vec4(-0.64668925, 0.27475403, 0.71154988, 0.00000000),
    vec4(-0.04008607, -0.20212790, 0.97853841, 0.00000000),
    vec4(-0.67820297, -0.44586687, 0.58416047, 0.00000000),
    vec4(-0.51319499, 0.32574430, 0.79405387, 0.00000000),
    vec4(0.25841618, 0.11585090, 0.95906186, 0.00000000),
    vec4(-0.76550631, -0.08250399, 0.63811690, 0.00000000),
    vec4(-0.26088367, -0.72439390, 0.63811690, 0.00000000),
    vec4(-0.15890313, 0.39437793, 0.90510543, 0.00000000),
    vec4(-0.51319499, -0.32574430, 0.79405387, 0.00000000),
    vec4(0.25841618, -0.11585090, 0.95906186, 0.00000000),
    vec4(-0.67820297, 0.44586687, 0.58416047, 0.00000000),
    vec4(-0.76550631, 0.08250399, 0.63811690, 0.00000000),
    vec4(-0.15890313, -0.39437793, 0.90510543, 0.00000000),
    vec4(-0.26088367, 0.72439390, 0.63811690, 0.00000000),
    vec4(-0.64668925, -0.27475403, 0.71154988, 0.00000000),
    vec4(-0.04008607, 0.20212790, 0.97853841, 0.00000000),
    vec4(-0.91367778, 0.33184916, 0.23466795, 0.00000000),
    vec4(0.51929985, 0.30626775, 0.79782688, 0.00000000),
    vec4(-0.51929985, 0.30626775, 0.79782688, 0.00000000),
    vec4(-0.25231132, 0.91287093, 0.32094496, 0.00000000),
    vec4(0.51929985, 0.70297753, 0.48595294, 0.00000000),
    vec4(-0.51929985, 0.70297753, 0.48595294, 0.00000000),
    vec4(0.25231132, 0.91287093, 0.32094496, 0.00000000),
    vec4(-0.11881706, 0.45923856, 0.88033088, 0.00000000),
    vec4(-0.11881706, 0.96386120, 0.23844096, 0.00000000),
    vec4(0.65279411, 0.75396780, 0.07343298, 0.00000000),
    vec4(0.11881706, 0.45923856, 0.88033088, 0.00000000),
    vec4(-0.65279411, 0.75396780, 0.07343298, 0.00000000),
    vec4(0.11881706, 0.96386120, 0.23844096, 0.00000000),
    vec4(-0.25231132, -0.91287093, -0.32094496, 0.00000000),
    vec4(0.51929985, -0.70297753, -0.48595294, 0.00000000),
    vec4(-0.51929985, -0.70297753, -0.48595294, 0.00000000),
    vec4(0.25231132, -0.91287093, -0.32094496, 0.00000000),
    vec4(0.51929985, -0.30626775, -0.79782688, 0.00000000),
    vec4(-0.51929985, -0.30626775, -0.79782688, 0.00000000),
    vec4(-0.11881706, -0.96386120, -0.23844096, 0.00000000),
    vec4(0.65279411, -0.75396780, -0.07343298, 0.00000000),
    vec4(-0.11881706, -0.45923856, -0.88033088, 0.00000000),
    vec4(-0.65279411, -0.75396780, -0.07343298, 0.00000000),
    vec4(0.11881706, -0.96386120, -0.23844096, 0.00000000),
    vec4(0.11881706, -0.45923856, -0.88033088, 0.00000000),
    vec4(0.78615138, 0.00000000, 0.61803399, 0.00000000),
    vec4(0.78615138, 0.00000000, 0.61803399, 0.00000000),
    vec4(0.78615138, 0.00000000, 0.61803399, 0.00000000),
    vec4(0.78615138, 0.00000000, 0.61803399, 0.00000000),
    vec4(0.58405869, 0.32699283, 0.74293414, 0.00000000),
    vec4(0.58405869, 0.32699283, 0.74293414, 0.00000000),
    vec4(0.58405869, 0.32699283, 0.74293414, 0.00000000),
    vec4(0.58405869, 0.32699283, 0.74293414, 0.00000000),
    vec4(0.25706586, 0.20209268, 0.94502682, 0.00000000),
    vec4(0.25706586, 0.20209268, 0.94502682, 0.00000000),
    vec4(0.25706586, 0.20209268, 0.94502682, 0.00000000),
    vec4(0.25706586, 0.20209268, 0.94502682, 0.00000000),
    vec4(0.25706586, -0.20209268, 0.94502682, 0.00000000),
    vec4(0.25706586, -0.20209268, 0.94502682, 0.00000000),
    vec4(0.25706586, -0.20209268, 0.94502682, 0.00000000),
    vec4(0.25706586, -0.20209268, 0.94502682, 0.00000000),
    vec4(0.58405869, -0.32699283, 0.74293414, 0.00000000),
    vec4(0.58405869, -0.32699283, 0.74293414, 0.00000000),
    vec4(0.58405869, -0.32699283, 0.74293414, 0.00000000),
    vec4(0.58405869, -0.32699283, 0.74293414, 0.00000000),
    vec4(0.78615138, 0.00000000, -0.61803399, 0.00000000),
    vec4(0.78615138, 0.00000000, -0.61803399, 0.00000000),
    vec4(0.58405869, -0.32699283, -0.74293414, 0.00000000),
    vec4(0.58405869, -0.32699283, -0.74293414, 0.00000000),
    vec4(0.58405869, -0.32699283, -0.74293414, 0.00000000),
    vec4(0.25706586, -0.20209268, -0.94502682, 0.00000000),
    vec4(0.25706586, -0.20209268, -0.94502682, 0.00000000),
    vec4(0.25706586, -0.20209268, -0.94502682, 0.00000000),
    vec4(0.25706586, -0.20209268, -0.94502682, 0.00000000),
    vec4(0.25706586, 0.20209268, -0.94502682, 0.00000000),
    vec4(0.25706586, 0.20209268, -0.94502682, 0.00000000),
    vec4(0.25706586, 0.20209268, -0.94502682, 0.00000000),
    vec4(0.25706586, 0.20209268, -0.94502682, 0.00000000),
    vec4(0.58405869, 0.32699283, -0.74293414, 0.00000000),
    vec4(0.58405869, 0.32699283, -0.74293414, 0.00000000),
    vec4(0.58405869, 0.32699283, -0.74293414, 0.00000000),
    vec4(0.00000000, 0.61803399, 0.78615138, 0.00000000),
    vec4(0.00000000, 0.61803399, 0.78615138, 0.00000000),
    vec4(0.00000000, 0.61803399, 0.78615138, 0.00000000),
    vec4(0.00000000, 0.61803399, 0.78615138, 0.00000000),
    vec4(0.32699283, 0.74293414, 0.58405869, 0.00000000),
    vec4(0.32699283, 0.74293414, 0.58405869, 0.00000000),
    vec4(0.32699283, 0.74293414, 0.58405869, 0.00000000),
    vec4(0.20209268, 0.94502682, 0.25706586, 0.00000000),
    vec4(-0.20209268, 0.94502682, 0.25706586, 0.00000000),
    vec4(-0.32699283, 0.74293414, 0.58405869, 0.00000000),
    vec4(-0.32699283, 0.74293414, 0.58405869, 0.00000000),
    vec4(-0.32699283, 0.74293414, 0.58405869, 0.00000000),
    vec4(0.00000000, 0.61803399, -0.78615138, 0.00000000),
    vec4(0.00000000, 0.61803399, -0.78615138, 0.00000000),
    vec4(-0.32699283, 0.74293414, -0.58405869, 0.00000000),
    vec4(-0.32699283, 0.74293414, -0.58405869, 0.00000000),
    vec4(-0.20209268, 0.94502682, -0.25706586, 0.00000000),
    vec4(0.20209268, 0.94502682, -0.25706586, 0.00000000),
    vec4(0.32699283, 0.74293414, -0.58405869, 0.00000000),
    vec4(0.32699283, 0.74293414, -0.58405869, 0.00000000),
    vec4(0.61803399, 0.78615138, 0.00000000, 0.00000000),
    vec4(0.61803399, 0.78615138, 0.00000000, 0.00000000),
    vec4(0.74293414, 0.58405869, 0.32699283, 0.00000000),
    vec4(0.74293414, 0.58405869, -0.32699283, 0.00000000),
    vec4(0.29225023, -0.47287080, 0.83125388, 0.00000000),
    vec4(0.29225023, 0.47287080, 0.83125388, 0.00000000),
    vec4(0.78555721, -0.32531665, 0.52637340, 0.00000000),
    vec4(0.02043617, -0.03306642, 0.99924420, 0.00000000),
    vec4(0.34575282, -0.49330698, 0.79818745, 0.00000000),
    vec4(0.81862363, 0.27181406, 0.50593723, 0.00000000),
    vec4(0.81862363, -0.27181406, 0.50593723, 0.00000000),
    vec4(0.34575282, 0.49330698, 0.79818745, 0.00000000),
    vec4(0.78555721, 0.32531665, 0.52637340, 0.00000000),
    vec4(0.02043617, 0.03306642, 0.99924420, 0.00000000),
    vec4(0.78555721, 0.32531665, -0.52637340, 0.00000000),
    vec4(0.34575282, 0.49330698, -0.79818745, 0.00000000),
    vec4(0.81862363, -0.27181406, -0.50593723, 0.00000000),
    vec4(0.34575282, -0.49330698, -0.79818745, 0.00000000),
    vec4(0.81862363, 0.27181406, -0.50593723, 0.00000000),
    vec4(0.78555721, -0.32531665, -0.52637340, 0.00000000),
    vec4(-0.29225023, 0.47287080, -0.83125388, 0.00000000),
    vec4(-0.29225023, -0.47287080, -0.83125388, 0.00000000),
    vec4(-0.02043617, -0.03306642, -0.99924420, 0.00000000),
    vec4(-0.81862363, 0.27181406, -0.50593723, 0.00000000),
    vec4(-0.81862363, -0.27181406, -0.50593723, 0.00000000),
    vec4(-0.02043617, 0.03306642, -0.99924420, 0.00000000),
    vec4(-0.81862363, -0.27181406, 0.50593723, 0.00000000),
    vec4(-0.81862363, 0.27181406, 0.50593723, 0.00000000)
);
const vec4 FACE_V[FACE_N] = vec4[](
    vec4(0.27009076, -0.89945372, -0.34356075, 0.00000000),
    vec4(-0.55589297, -0.43701602, 0.70710678, 0.00000000),
    vec4(0.55589297, -0.43701602, -0.70710678, 0.00000000),
    vec4(-0.27009076, -0.89945372, 0.34356075, 0.00000000),
    vec4(-0.55589297, 0.43701602, 0.70710678, 0.00000000),
    vec4(0.55589297, 0.43701602, -0.70710678, 0.00000000),
    vec4(0.69879408, -0.66823487, -0.25524301, 0.00000000),
    vec4(-0.41299186, -0.66823487, 0.61878904, 0.00000000),
    vec4(0.41299186, 0.66823487, -0.61878904, 0.00000000),
    vec4(0.29411492, -0.93832563, 0.18177302, 0.00000000),
    vec4(0.75655261, 0.11234190, -0.64421071, 0.00000000),
    vec4(-0.29411492, 0.93832563, -0.18177302, 0.00000000),
    vec4(0.52533377, -0.85000789, 0.03887191, 0.00000000),
    vec4(0.81113598, 0.48646186, -0.32467412, 0.00000000),
    vec4(-0.52533377, 0.85000789, -0.03887191, 0.00000000),
    vec4(0.96234979, -0.14290111, -0.23121885, 0.00000000),
    vec4(-0.96234979, 0.14290111, 0.23121885, 0.00000000),
    vec4(0.08831774, 0.96888483, -0.23121885, 0.00000000),
    vec4(0.52533377, 0.85000789, 0.03887191, 0.00000000),
    vec4(-0.52533377, -0.85000789, -0.03887191, 0.00000000),
    vec4(-0.81113598, 0.48646186, 0.32467412, 0.00000000),
    vec4(0.96234979, 0.14290111, -0.23121885, 0.00000000),
    vec4(-0.96234979, -0.14290111, 0.23121885, 0.00000000),
    vec4(-0.08831774, 0.96888483, 0.23121885, 0.00000000),
    vec4(0.41299186, -0.66823487, -0.61878904, 0.00000000),
    vec4(-0.69879408, -0.66823487, 0.25524301, 0.00000000),
    vec4(-0.41299186, 0.66823487, 0.61878904, 0.00000000),
    vec4(0.29411492, 0.93832563, 0.18177302, 0.00000000),
    vec4(-0.29411492, -0.93832563, -0.18177302, 0.00000000),
    vec4(-0.75655261, 0.11234190, 0.64421071, 0.00000000),
    vec4(-0.55589297, -0.43701602, -0.70710678, 0.00000000),
    vec4(0.55589297, -0.43701602, 0.70710678, 0.00000000),
    vec4(0.41299186, -0.66823487, 0.61878904, 0.00000000),
    vec4(-0.29411492, -0.93832563, 0.18177302, 0.00000000),
    vec4(0.29411492, 0.93832563, -0.18177302, 0.00000000),
    vec4(0.75655261, -0.11234190, 0.64421071, 0.00000000),
    vec4(-0.52533377, -0.85000789, 0.03887191, 0.00000000),
    vec4(0.52533377, 0.85000789, -0.03887191, 0.00000000),
    vec4(0.81113598, -0.48646186, 0.32467412, 0.00000000),
    vec4(-0.96234979, -0.14290111, -0.23121885, 0.00000000),
    vec4(0.96234979, 0.14290111, 0.23121885, 0.00000000),
    vec4(0.08831774, -0.96888483, 0.23121885, 0.00000000),
    vec4(-0.52533377, 0.85000789, 0.03887191, 0.00000000),
    vec4(-0.81113598, -0.48646186, -0.32467412, 0.00000000),
    vec4(0.52533377, -0.85000789, -0.03887191, 0.00000000),
    vec4(-0.96234979, 0.14290111, -0.23121885, 0.00000000),
    vec4(0.96234979, -0.14290111, 0.23121885, 0.00000000),
    vec4(-0.08831774, -0.96888483, -0.23121885, 0.00000000),
    vec4(-0.41299186, -0.66823487, -0.61878904, 0.00000000),
    vec4(-0.29411492, 0.93832563, 0.18177302, 0.00000000),
    vec4(-0.75655261, -0.11234190, -0.64421071, 0.00000000),
    vec4(0.29411492, -0.93832563, -0.18177302, 0.00000000),
    vec4(-0.27009076, -0.89945372, 0.34356075, 0.00000000),
    vec4(0.55589297, -0.43701602, -0.70710678, 0.00000000),
    vec4(-0.55589297, -0.43701602, 0.70710678, 0.00000000),
    vec4(0.27009076, -0.89945372, -0.34356075, 0.00000000),
    vec4(-0.69879408, -0.66823487, 0.25524301, 0.00000000),
    vec4(0.41299186, -0.66823487, -0.61878904, 0.00000000),
    vec4(-0.75655261, 0.11234190, 0.64421071, 0.00000000),
    vec4(-0.29411492, -0.93832563, -0.18177302, 0.00000000),
    vec4(0.29411492, 0.93832563, 0.18177302, 0.00000000),
    vec4(-0.81113598, 0.48646186, 0.32467412, 0.00000000),
    vec4(-0.52533377, -0.85000789, -0.03887191, 0.00000000),
    vec4(0.52533377, 0.85000789, 0.03887191, 0.00000000),
    vec4(-0.08831774, 0.96888483, 0.23121885, 0.00000000),
    vec4(-0.96234979, -0.14290111, 0.23121885, 0.00000000),
    vec4(0.96234979, 0.14290111, -0.23121885, 0.00000000),
    vec4(-0.52533377, 0.85000789, -0.03887191, 0.00000000),
    vec4(0.81113598, 0.48646186, -0.32467412, 0.00000000),
    vec4(0.52533377, -0.85000789, 0.03887191, 0.00000000),
    vec4(0.08831774, 0.96888483, -0.23121885, 0.00000000),
    vec4(-0.96234979, 0.14290111, 0.23121885, 0.00000000),
    vec4(0.96234979, -0.14290111, -0.23121885, 0.00000000),
    vec4(-0.41299186, -0.66823487, 0.61878904, 0.00000000),
    vec4(0.69879408, -0.66823487, -0.25524301, 0.00000000),
    vec4(-0.29411492, 0.93832563, -0.18177302, 0.00000000),
    vec4(0.75655261, 0.11234190, -0.64421071, 0.00000000),
    vec4(0.29411492, -0.93832563, 0.18177302, 0.00000000),
    vec4(0.29411492, -0.93832563, -0.18177302, 0.00000000),
    vec4(-0.75655261, -0.11234190, -0.64421071, 0.00000000),
    vec4(-0.29411492, 0.93832563, 0.18177302, 0.00000000),
    vec4(0.52533377, -0.85000789, -0.03887191, 0.00000000),
    vec4(-0.81113598, -0.48646186, -0.32467412, 0.00000000),
    vec4(-0.52533377, 0.85000789, 0.03887191, 0.00000000),
    vec4(-0.08831774, -0.96888483, -0.23121885, 0.00000000),
    vec4(0.96234979, -0.14290111, 0.23121885, 0.00000000),
    vec4(-0.96234979, 0.14290111, -0.23121885, 0.00000000),
    vec4(0.81113598, -0.48646186, 0.32467412, 0.00000000),
    vec4(0.52533377, 0.85000789, -0.03887191, 0.00000000),
    vec4(-0.52533377, -0.85000789, 0.03887191, 0.00000000),
    vec4(0.08831774, -0.96888483, 0.23121885, 0.00000000),
    vec4(0.96234979, 0.14290111, 0.23121885, 0.00000000),
    vec4(-0.96234979, -0.14290111, -0.23121885, 0.00000000),
    vec4(0.75655261, -0.11234190, 0.64421071, 0.00000000),
    vec4(0.29411492, 0.93832563, -0.18177302, 0.00000000),
    vec4(-0.29411492, -0.93832563, 0.18177302, 0.00000000),
    vec4(0.43701602, 0.70710678, -0.55589297, 0.00000000),
    vec4(0.43701602, -0.70710678, 0.55589297, 0.00000000),
    vec4(-0.89945372, -0.34356075, 0.27009076, 0.00000000),
    vec4(-0.43701602, 0.70710678, -0.55589297, 0.00000000),
    vec4(-0.43701602, -0.70710678, 0.55589297, 0.00000000),
    vec4(-0.89945372, 0.34356075, -0.27009076, 0.00000000),
    vec4(0.66823487, -0.61878904, 0.41299186, 0.00000000),
    vec4(-0.66823487, -0.25524301, 0.69879408, 0.00000000),
    vec4(-0.66823487, 0.61878904, -0.41299186, 0.00000000),
    vec4(0.66823487, 0.61878904, -0.41299186, 0.00000000),
    vec4(-0.66823487, -0.61878904, 0.41299186, 0.00000000),
    vec4(-0.66823487, 0.25524301, -0.69879408, 0.00000000),
    vec4(-0.89945372, 0.34356075, -0.27009076, 0.00000000),
    vec4(-0.43701602, -0.70710678, 0.55589297, 0.00000000),
    vec4(-0.43701602, 0.70710678, -0.55589297, 0.00000000),
    vec4(-0.89945372, -0.34356075, 0.27009076, 0.00000000),
    vec4(0.43701602, -0.70710678, 0.55589297, 0.00000000),
    vec4(0.43701602, 0.70710678, -0.55589297, 0.00000000),
    vec4(-0.66823487, 0.25524301, -0.69879408, 0.00000000),
    vec4(-0.66823487, -0.61878904, 0.41299186, 0.00000000),
    vec4(0.66823487, 0.61878904, -0.41299186, 0.00000000),
    vec4(-0.66823487, 0.61878904, -0.41299186, 0.00000000),
    vec4(-0.66823487, -0.25524301, 0.69879408, 0.00000000),
    vec4(0.66823487, -0.61878904, 0.41299186, 0.00000000),
    vec4(-0.20209268, -0.94502682, 0.25706586, 0.00000000),
    vec4(0.58405869, -0.32699283, -0.74293414, 0.00000000),
    vec4(-0.58405869, -0.32699283, 0.74293414, 0.00000000),
    vec4(0.20209268, -0.94502682, -0.25706586, 0.00000000),
    vec4(0.20209268, -0.94502682, 0.25706586, 0.00000000),
    vec4(-0.32699283, -0.74293414, 0.58405869, 0.00000000),
    vec4(0.78615138, 0.00000000, -0.61803399, 0.00000000),
    vec4(0.74293414, -0.58405869, -0.32699283, 0.00000000),
    vec4(-0.20209268, -0.94502682, 0.25706586, 0.00000000),
    vec4(-0.74293414, -0.58405869, 0.32699283, 0.00000000),
    vec4(-0.61803399, 0.78615138, -0.00000000, 0.00000000),
    vec4(-0.94502682, 0.25706586, 0.20209268, 0.00000000),
    vec4(-0.61803399, -0.78615138, -0.00000000, 0.00000000),
    vec4(-0.94502682, -0.25706586, 0.20209268, 0.00000000),
    vec4(-0.74293414, 0.58405869, 0.32699283, 0.00000000),
    vec4(0.20209268, -0.94502682, -0.25706586, 0.00000000),
    vec4(0.74293414, 0.58405869, -0.32699283, 0.00000000),
    vec4(0.78615138, -0.00000000, -0.61803399, 0.00000000),
    vec4(0.20209268, 0.94502682, 0.25706586, 0.00000000),
    vec4(0.32699283, -0.74293414, -0.58405869, 0.00000000),
    vec4(0.20209268, -0.94502682, 0.25706586, 0.00000000),
    vec4(-0.20209268, -0.94502682, -0.25706586, 0.00000000),
    vec4(0.20209268, 0.94502682, -0.25706586, 0.00000000),
    vec4(-0.32699283, 0.74293414, -0.58405869, 0.00000000),
    vec4(0.74293414, 0.58405869, 0.32699283, 0.00000000),
    vec4(-0.20209268, 0.94502682, -0.25706586, 0.00000000),
    vec4(-0.61803399, -0.78615138, 0.00000000, 0.00000000),
    vec4(-0.94502682, -0.25706586, -0.20209268, 0.00000000),
    vec4(-0.74293414, 0.58405869, -0.32699283, 0.00000000),
    vec4(-0.74293414, -0.58405869, -0.32699283, 0.00000000),
    vec4(-0.61803399, 0.78615138, 0.00000000, 0.00000000),
    vec4(-0.94502682, 0.25706586, -0.20209268, 0.00000000),
    vec4(0.20209268, 0.94502682, 0.25706586, 0.00000000),
    vec4(0.74293414, -0.58405869, 0.32699283, 0.00000000),
    vec4(-0.32699283, -0.74293414, -0.58405869, 0.00000000),
    vec4(0.20209268, -0.94502682, -0.25706586, 0.00000000),
    vec4(-0.32699283, -0.74293414, 0.58405869, 0.00000000),
    vec4(-0.32699283, 0.74293414, -0.58405869, 0.00000000),
    vec4(0.94502682, 0.25706586, -0.20209268, 0.00000000),
    vec4(0.94502682, -0.25706586, 0.20209268, 0.00000000),
    vec4(0.00000000, -0.61803399, 0.78615138, 0.00000000),
    vec4(-0.94502682, 0.25706586, 0.20209268, 0.00000000),
    vec4(-0.74293414, 0.58405869, -0.32699283, 0.00000000),
    vec4(-0.94502682, 0.25706586, -0.20209268, 0.00000000),
    vec4(-0.94502682, -0.25706586, 0.20209268, 0.00000000),
    vec4(0.00000000, 0.61803399, -0.78615138, 0.00000000),
    vec4(0.94502682, 0.25706586, 0.20209268, 0.00000000),
    vec4(-0.74293414, -0.58405869, 0.32699283, 0.00000000),
    vec4(-0.94502682, -0.25706586, -0.20209268, 0.00000000),
    vec4(-0.94502682, 0.25706586, 0.20209268, 0.00000000),
    vec4(0.74293414, 0.58405869, 0.32699283, 0.00000000),
    vec4(0.94502682, 0.25706586, -0.20209268, 0.00000000),
    vec4(0.94502682, 0.25706586, 0.20209268, 0.00000000),
    vec4(0.94502682, -0.25706586, -0.20209268, 0.00000000),
    vec4(-0.74293414, 0.58405869, 0.32699283, 0.00000000),
    vec4(0.94502682, -0.25706586, 0.20209268, 0.00000000),
    vec4(0.74293414, -0.58405869, -0.32699283, 0.00000000),
    vec4(-0.74293414, 0.58405869, -0.32699283, 0.00000000),
    vec4(0.61803399, -0.78615138, 0.00000000, 0.00000000),
    vec4(0.61803399, -0.78615138, 0.00000000, 0.00000000),
    vec4(0.43701602, -0.70710678, -0.55589297, 0.00000000),
    vec4(-0.43701602, -0.70710678, 0.55589297, 0.00000000),
    vec4(0.61878904, 0.41299186, -0.66823487, 0.00000000),
    vec4(0.52533377, -0.85000789, -0.03887191, 0.00000000),
    vec4(0.93832563, 0.18177302, -0.29411492, 0.00000000),
    vec4(-0.23121885, 0.96234979, -0.14290111, 0.00000000),
    vec4(0.23121885, 0.96234979, 0.14290111, 0.00000000),
    vec4(-0.93832563, 0.18177302, 0.29411492, 0.00000000),
    vec4(-0.61878904, 0.41299186, 0.66823487, 0.00000000),
    vec4(-0.52533377, -0.85000789, 0.03887191, 0.00000000),
    vec4(0.61878904, -0.41299186, 0.66823487, 0.00000000),
    vec4(0.93832563, -0.18177302, 0.29411492, 0.00000000),
    vec4(-0.23121885, -0.96234979, 0.14290111, 0.00000000),
    vec4(-0.93832563, -0.18177302, -0.29411492, 0.00000000),
    vec4(0.23121885, -0.96234979, -0.14290111, 0.00000000),
    vec4(-0.61878904, -0.41299186, -0.66823487, 0.00000000),
    vec4(0.43701602, -0.70710678, -0.55589297, 0.00000000),
    vec4(-0.43701602, -0.70710678, 0.55589297, 0.00000000),
    vec4(-0.52533377, -0.85000789, 0.03887191, 0.00000000),
    vec4(0.23121885, 0.96234979, 0.14290111, 0.00000000),
    vec4(-0.23121885, 0.96234979, -0.14290111, 0.00000000),
    vec4(0.52533377, -0.85000789, -0.03887191, 0.00000000),
    vec4(0.23121885, -0.96234979, -0.14290111, 0.00000000),
    vec4(-0.23121885, -0.96234979, 0.14290111, 0.00000000)
);
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

#define LAVA_GLOW  3.0
#define MOON_FILL  0.06
#define DS_SPIN    0.25
#define GEM_IOR    vec3(2.38, 2.42, 2.47)
vec3 rotYv(vec3 v, float a) { float c = cos(a), s = sin(a); return vec3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z); }
vec3 rotXv(vec3 v, float a) { float c = cos(a), s = sin(a); return vec3(v.x, c * v.y - s * v.z, s * v.y + c * v.z); }
float wHash(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
float wNoise(vec3 p) {
    vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(wHash(i), wHash(i + vec3(1, 0, 0)), f.x), mix(wHash(i + vec3(0, 1, 0)), wHash(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(wHash(i + vec3(0, 0, 1)), wHash(i + vec3(1, 0, 1)), f.x), mix(wHash(i + vec3(0, 1, 1)), wHash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float cellEdge(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    float d1 = 8.0, d2 = 8.0;
    for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
        vec3 g = vec3(x, y, z), o = vec3(wHash(i + g), wHash(i + g + 7.1), wHash(i + g + 3.3));
        float d = length(g + o - f);
        if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) d2 = d;
    }
    return d2 - d1;
}
mat3 moonFrame(vec3 c) {
    vec3 ex = normalize(-c), ey = normalize(vec3(0.0, 1.0, 0.0) - ex * ex.y), ez = cross(ex, ey);
    return mat3(ex, ey, ez);
}
#define BOUNCE_K   0.9
vec3 gBX = vec3(1, 0, 0), gBY = vec3(0, 1, 0), gBZ = vec3(0, 0, 1);
vec3 boxGlass(vec3 ro, vec3 rd, vec3 bx, vec3 by, vec3 bz, vec3 skyC, float t, mat3 toP);
vec3 wallBounce(vec3 P, vec3 n, float t, mat3 toP) { return BOUNCE_K * boxGlass(P, n, gBX, gBY, gBZ, vec3(0.0), t, toP); }
vec3 dwarfLight(vec3 P, vec3 n) { return LIGHT_COL * LIGHT_K * max(dot(n, -normalize(P)), 0.0) / max(dot(P, P), 0.5); }
vec3 weirdMoon(int mat, vec3 P, vec3 n, vec3 rd, vec4 m, float heat, float wob, float t, mat3 toP) {
    float cosI = max(dot(n, -rd), 0.0);
    mat3 F = moonFrame(m.xyz);
    vec3 ln = transpose(F) * n;
    if (mat == 1 || mat == 2) {
        vec2 uv = vec2(0.5 + atan(ln.z, ln.x) / 6.2831853, 0.5 + asin(clamp(ln.y, -1.0, 1.0)) / 3.14159265);
        vec3 alb = pow(texture(u_moonMap, uv).rgb, vec3(2.2));
        if (mat == 1) return alb * (0.9 * dwarfLight(P, n) + wallBounce(P, n, t, toP)) + alb * MOON_FILL * vec3(0.60, 0.40, 1.00);
        vec3 lf = rotYv(ln, 0.35 * t);
        vec3 lb = vec3(lf.x * 2.5, lf.y * 11.0, lf.z * 2.5);
        float f1 = wNoise(lb + vec3(0.9 * t, 0.0, 0.5 * t)), f2 = wNoise(lb * 2.1 - vec3(0.6 * t, 0.0, 0.8 * t));
        float skin = smoothstep(0.60, 0.78, 0.65 * f1 + 0.35 * f2);
        float hot = 0.55 + 0.45 * wNoise(lf * 2.0 + 0.5 * t) + 0.6 * clamp(heat, 0.0, 1.0);
        vec3 molten = mix(vec3(1.0, 0.25, 0.03), vec3(1.0, 0.78, 0.30), clamp(hot - 0.3, 0.0, 1.0));
        vec3 lava = mix(molten * LAVA_GLOW * hot, vec3(0.03, 0.006, 0.0) + molten * 0.25 * (1.0 - smoothstep(0.0, 0.3, skin)), skin);
        return lava + alb * 0.10 * dwarfLight(P, n);
    }
    if (mat == 3) {
        vec3 lr = rotYv(ln, 0.8 * t);
        float lon = atan(lr.z, lr.x), lat = asin(clamp(lr.y, -1.0, 1.0));
        float NL = 14.0, cl = floor(lat / 3.14159265 * NL + 0.5 * NL), nl = max(floor(2.0 * NL * cos((cl + 0.5 - 0.5 * NL) / NL * 3.14159265)), 4.0);
        float lc = (cl + 0.5) / NL * 3.14159265 - 1.5707963, lo = (floor((lon / 6.2831853 + 0.5) * nl) + 0.5) / nl * 6.2831853 - 3.14159265;
        vec2 fr = vec2(fract(lat / 3.14159265 * NL + 0.5 * NL), fract((lon / 6.2831853 + 0.5) * nl));
        float grout = smoothstep(0.0, 0.12, min(min(fr.x, 1.0 - fr.x), min(fr.y, 1.0 - fr.y)));
        vec3 tn = vec3(cos(lc) * cos(lo), sin(lc), cos(lc) * sin(lo));
        tn = F * rotYv(tn, -0.8 * t);
        vec3 R = reflect(rd, tn);
        vec3 c = 0.85 * mirrorEnv(P + n * 1e-3, R, t, toP);
        float spark = pow(max(dot(R, -normalize(P)), 0.0), 400.0);
        return (c + vec3(1.0, 0.9, 1.0) * spark * 30.0) * grout + vec3(0.002);
    }
    if (mat == 4 || mat == 6) {
        vec3 acc = vec3(0.0);
        for (int ch = 0; ch < 3; ch++) {
            float ior = GEM_IOR[ch];
            vec3 d1 = refract(rd, n, 1.0 / ior);
            vec3 oc = P - m.xyz;
            float tt = -2.0 * dot(oc, d1);
            vec3 Q = P + d1 * max(tt, 0.0), nq = normalize(Q - m.xyz);
            vec3 d2 = refract(d1, -nq, ior);
            if (dot(d2, d2) < 0.5) d2 = reflect(d1, -nq);
            acc[ch] = mirrorEnv(Q + d2 * 1e-3, d2, t, toP)[ch];
        }
        float r0 = pow((GEM_IOR.g - 1.0) / (GEM_IOR.g + 1.0), 2.0);
        float Fr = r0 + (1.0 - r0) * pow(1.0 - cosI, 5.0);
        vec3 refl = mirrorEnv(P + n * 1e-3, reflect(rd, n), t, toP);
        vec3 tint = mat == 6 ? vec3(0.95, 0.97, 1.0) : vec3(0.80, 0.92, 1.0);
        return tint * acc * (1.0 - Fr) * 0.9 + refl * Fr + vec3(1.0) * pow(max(dot(reflect(rd, n), -normalize(P)), 0.0), 200.0) * 8.0;
    }
    if (mat == 5) {
        vec3 jc = vec3(0.25, 1.0, 0.45);
        float thick = pow(cosI, 0.6);
        vec3 through = mirrorEnv(P + rd * 2.0 * m.w, rd, t, toP);
        vec3 inside = jc * (0.25 + 0.55 * dwarfLight(P, n).g * 2.0);
        float spec = pow(max(dot(reflect(rd, n), -normalize(P)), 0.0), 60.0);
        return mix(through * jc, inside, 0.55 * thick) + jc * 0.25 * pow(1.0 - cosI, 2.0) + vec3(spec) * 2.0 + jc * 0.5 * wallBounce(P, n, t, toP);
    }
    return vec3(0.0);
}
float disnubHit(vec3 ro, vec3 rd, vec4 m, float t, out vec3 nW) {
    float sc = m.w / STONE_RB;
    vec3 o = (ro - m.xyz) / sc, d = rd;
    float a = DS_SPIN * t;
    o = rotXv(rotYv(o, a), 0.6 * a); d = rotXv(rotYv(d, a), 0.6 * a);
    int j, cnt;
    float tl = soupHit(o, d, j, cnt);
    if (j < 0) { nW = vec3(0.0); return 1e9; }
    vec3 nl = FACE_NW[j].xyz * sign(-dot(FACE_NW[j].xyz, d));
    nW = rotYv(rotXv(nl, -0.6 * a), -a);
    return tl * sc;
}
#define MOON_LIT   0.35
#define HOT_GAIN   2.5
#define METAL_R0   0.80
#define METAL_TINT vec3(0.97, 0.98, 1.00)
vec3 mirrorEnv(vec3 P, vec3 R, float t, mat3 toP) {
    vec3 base;
    float b = dot(P, R), c = dot(P, P) - DW_R * DW_R, disc = b * b - c;
    float tq = 1e9;
    if (disc > 0.0 && -b - sqrt(disc) > 0.0) {
        tq = -b - sqrt(disc);
        vec3 n = normalize(P + R * tq);
        float om = 1.0 - max(dot(n, -R), 0.0);
        float I = max(1.0 - LIMB_U1 * om - LIMB_U2 * om * om, 0.03);
#if DWARF_STYLE == 1
        base = dwarfSurface(toP * n, t) * I;
#else
        base = radiance(cloudT(toP * n, t) * pow(I, 1.0 / 3.0));
#endif
        base = mix(base, LIMB_PURP * length(base) * 0.9, smoothstep(0.35, 1.0, om) * 0.7);
    } else {
        base = skyL(R, 4.0);
        float hz = (length(cross(P, R)) - DW_R) / DW_R;
        if (hz > 0.0 && b < 0.0) base += UV_HALO_COL * UV_HALO_G * exp(-hz / UV_HALO_H);
    }
    if (abs(R.y) > 1e-4) {
        float tR = -P.y / R.y;
        vec3 Q = P + R * tR;
        float r = length(Q.xz);
        if (tR > 0.0 && tR < tq && r > R_IN && r < R_OUT) {
            float sg = ringAt(r, atan(Q.z, Q.x)).x;
            float trans = exp(-(TAU_K * sg + TAU_K2 * sg * sg) / abs(R.y));
            float beta = asin(clamp(DW_R / r, 0.0, 1.0));
            float top = (2.0 * beta * beta * beta / 3.0) / 3.14159265;
            base = base * trans + RING_LIGHT * RING_WHITE * RING_FLUOR * LIGHT_K * RING_ALB * (0.5 / max(r * r, RING_R2MIN) + top) * (1.0 - trans);
        }
    }
    return base;
}

#define BOX_H      vec3(3.2, 3.2, 3.2)
#define GLS_SQ     0.41846
#define MIRROR_R0      0.30
#define MIRROR_GAIN    1.0
#define MIRROR_TINT    vec3(0.80, 0.80, 0.86)
#define MIRROR_BASE    vec3(0.004, 0.004, 0.006)
#define MIRROR_BOUNCES 2
vec3 envBox(vec3 P, vec3 R, float t, mat3 toP, out bool miss) {
    vec3 base = vec3(0.0);
    miss = true;
    float b = dot(P, R), c = dot(P, P) - DW_R * DW_R, disc = b * b - c;
    float tq = 1e9;
    if (disc > 0.0 && -b - sqrt(disc) > 0.0) {
        tq = -b - sqrt(disc);
        vec3 n = normalize(P + R * tq);
        float om = 1.0 - max(dot(n, -R), 0.0);
        float I = max(1.0 - LIMB_U1 * om - LIMB_U2 * om * om, 0.03);
#if DWARF_STYLE == 1
        base = dwarfSurface(toP * n, t) * I;
#else
        base = radiance(cloudT(toP * n, t) * pow(I, 1.0 / 3.0));
#endif
        base = mix(base, LIMB_PURP * length(base) * 0.9, smoothstep(0.35, 1.0, om) * 0.7);
        miss = false;
    }
    for (int i = 0; i < N_MOONS; i++) {
        vec4 m = texelFetch(u_moons, ivec2(i, 0), 0);
        vec3 oc = P - m.xyz;
        float bb = dot(oc, R), cc = dot(oc, oc) - m.w * m.w, dd = bb * bb - cc;
        if (dd <= 0.0) continue;
        float tm = -bb - sqrt(dd);
        if (tm <= 0.0 || tm >= tq) continue;
        tq = tm;
        vec3 Q = P + R * tm, n = normalize(Q - m.xyz);
        base = METAL_TINT * 0.35 * LIGHT_COL * LIGHT_K * max(dot(n, -normalize(Q)), 0.0) / max(dot(Q, Q), 0.5) + vec3(0.002);
        miss = false;
    }
    if (abs(R.y) > 1e-4) {
        float tR = -P.y / R.y;
        vec3 Q = P + R * tR;
        float r = length(Q.xz);
        if (tR > 0.0 && tR < tq && r > R_IN && r < R_OUT) {
            float sg = ringAt(r, atan(Q.z, Q.x)).x;
            float trans = exp(-(TAU_K * sg + TAU_K2 * sg * sg) / abs(R.y));
            float beta = asin(clamp(DW_R / r, 0.0, 1.0));
            float top = (2.0 * beta * beta * beta / 3.0) / 3.14159265;
            base = base * trans + RING_LIGHT * RING_WHITE * RING_FLUOR * LIGHT_K * RING_ALB * (0.5 / max(r * r, RING_R2MIN) + top) * (1.0 - trans);
            if (trans < 0.5) miss = false;
        }
    }
    return base;
}
vec3 mirrorWalls(vec3 P, vec3 n, vec3 d, vec3 bx, vec3 by, vec3 bz, float t, mat3 toP) {
    vec3 acc = MIRROR_BASE, w = vec3(MIRROR_GAIN);
    for (int k = 0; k < MIRROR_BOUNCES; k++) {
        float fres = MIRROR_R0 + (1.0 - MIRROR_R0) * pow(1.0 - max(dot(n, -d), 0.0), 5.0);
        vec3 R = reflect(d, n);
        bool miss;
        acc += w * fres * envBox(P + n * 1e-3, R, t, toP, miss);
        if (!miss) break;
        w *= fres * MIRROR_TINT;
        vec3 o = vec3(dot(P, bx), dot(P, by), dot(P, bz)), dd = vec3(dot(R, bx), dot(R, by), dot(R, bz));
        vec3 inv = 1.0 / (dd + vec3(1e-7));
        vec3 tmx = max((-BOX_H - o) * inv, (BOX_H - o) * inv);
        float tF = min(min(tmx.x, tmx.y), tmx.z);
        vec3 q = o + dd * tF;
        if (tF == tmx.z && dd.z < 0.0) break;
        vec3 nq = tF == tmx.x ? vec3(-sign(q.x), 0.0, 0.0) : (tF == tmx.y ? vec3(0.0, -sign(q.y), 0.0) : vec3(0.0, 0.0, -sign(q.z)));
        P = q.x * bx + q.y * by + q.z * bz;
        n = nq.x * bx + nq.y * by + nq.z * bz;
        d = R;
    }
    return acc;
}

uniform vec4 u_view_q;
uniform float u_view_fov;
vec3 viewRot(vec4 q, vec3 v) { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }

#define BOX_ON     1
#define BOX_TINT   0.22
#define BOX_LIT    0.35
#define BOX_EDGE   0.05
#define BOX_EDGE_K 0.6
uniform sampler2D u_gBack, u_gWL, u_gWR, u_gTop, u_gBot;
#define CLEAN_BG      1
#define CLEAN_BG_COL  vec3(0.028, 0.100, 0.110)
#define CLEAN_BG_LIT  0.075
#define CLEAN_BG_EDGE 0.080
#define CLEAN_GRID     1
#define CLEAN_GRID_D   0.5
#define CLEAN_GRID_W   0.010
#define CLEAN_GRID_COL vec3(0.060, 0.170, 0.180)
#define BOX_GLASS    1
#define GLS_BASE     0.012
#define GLS_BACK     0.12
#define GLS_BACK_COL vec3(0.55, 0.45, 1.0)
#define GLS_DWARF    1.4
#define GLS_GAIN     1.0
vec3 mirrorWalls(vec3 P, vec3 n, vec3 d, vec3 bx, vec3 by, vec3 bz, float t, mat3 toP);
vec3 boxGlass(vec3 ro, vec3 rd, vec3 bx, vec3 by, vec3 bz, vec3 skyC, float t, mat3 toP) {
    vec3 o = vec3(dot(ro, bx), dot(ro, by), dot(ro, bz)), d = vec3(dot(rd, bx), dot(rd, by), dot(rd, bz));
    vec3 inv = 1.0 / (d + vec3(1e-7));
    vec3 t1 = (-BOX_H - o) * inv, t2 = (BOX_H - o) * inv;
    vec3 tmx = max(t1, t2);
    float tF = min(min(tmx.x, tmx.y), tmx.z);
    if (tF <= 0.0) return skyC;
    vec3 q = o + d * tF;
    bool front = tF == tmx.z && d.z < 0.0;
#if BOX_GLASS != 1
    if (front) return skyC;
#endif
    vec3 P = q.x * bx + q.y * by + q.z * bz;
    vec3 e = smoothstep(BOX_EDGE, 0.0, BOX_H - abs(q));
    float edge = max(max(min(e.x, e.y), min(e.y, e.z)), min(e.x, e.z));
#if CLEAN_BG
    float gy = q.y / BOX_H.y;
    float grad = 0.70 + 0.45 * smoothstep(-1.0, 1.0, gy);
    float lit = CLEAN_BG_LIT * LIGHT_K / max(dot(P, P), 0.45);
    vec3 cb = CLEAN_BG_COL * grad + LIGHT_COL * lit + CLEAN_BG_EDGE * edge;
#if CLEAN_GRID
    vec2 fq = tF == tmx.x ? q.yz : (tF == tmx.y ? q.xz : q.xy);
    vec2 dm = abs(fq - CLEAN_GRID_D * floor(fq / CLEAN_GRID_D + 0.5));
    vec2 dM = abs(fq - 2.0 * CLEAN_GRID_D * floor(fq / (2.0 * CLEAN_GRID_D) + 0.5));
    float gl = max(max(smoothstep(CLEAN_GRID_W, 0.0, dm.x), smoothstep(CLEAN_GRID_W, 0.0, dm.y)) * 0.6,
                   max(smoothstep(1.6 * CLEAN_GRID_W, 0.0, dM.x), smoothstep(1.6 * CLEAN_GRID_W, 0.0, dM.y)));
    cb += CLEAN_GRID_COL * gl * (1.0 + 4.0 * lit);
#endif
    return cb;
#elif BOX_GLASS == 2
    vec3 nq = tF == tmx.x ? vec3(-sign(q.x), 0.0, 0.0) : (tF == tmx.y ? vec3(0.0, -sign(q.y), 0.0) : vec3(0.0, 0.0, -sign(q.z)));
    return mirrorWalls(P, nq.x * bx + nq.y * by + nq.z * bz, rd, bx, by, bz, t, toP) * (1.0 - 0.4 * edge);
#elif BOX_GLASS
    vec3 u = q / BOX_H * 0.5 + 0.5;
    float uzp = 1.0 - u.z;
    vec3 c; float fk = 1.0;
    float uy = 0.5 + (u.y - 0.5) * GLS_SQ;
    if (front)            c = texture(u_gBack, vec2(uy, u.x)).rgb;
    else if (tF == tmx.z) c = texture(u_gBack, vec2(uy, 1.0 - u.x)).rgb;
    else if (tF == tmx.x) { c = q.x < 0.0 ? texture(u_gWL, vec2(uy, 1.0 - uzp)).rgb : texture(u_gWR, vec2(uy, uzp)).rgb; fk = 0.85; }
    else                  { c = q.y > 0.0 ? texture(u_gTop, vec2(u.x, 1.0 - uzp)).rgb : texture(u_gBot, vec2(u.x, uzp)).rgb; fk = 0.7; }
    c = pow(c, vec3(2.2));
    vec3 light = GLS_BACK * GLS_BACK_COL + GLS_DWARF * LIGHT_COL * LIGHT_K / max(dot(P, P), 1.0);
    return GLS_GAIN * c * (GLS_BASE + light * fk) * (1.0 - 0.5 * edge);
#else
    float r2 = max(dot(P, P), 0.3);
    vec3 lit = LIGHT_COL * LIGHT_K * BOX_LIT / r2;
    return skyC * BOX_TINT + lit * (0.25 + BOX_EDGE_K * edge * 4.0);
#endif
}
#define HOLE_ON      1
#define HOLE_R_DRAW  0.22
#define HOLE_RING    vec3(1.0, 0.55, 0.85)
uniform vec4 u_rbHole;
uniform usampler2D u_dust;
uniform float u_dustOn;
uniform usampler2D u_cmet;
uniform float u_cmetOn;
#define CMET_COL     vec3(1.0, 0.55, 0.22)
#define CMET_K       0.0030
#define DUST_COL     vec3(0.85, 0.80, 1.00)
#define DUST_K       0.0012
uniform float u_rbFlash;
vec3 holeFx(vec3 ro, vec3 rd, vec3 col) {
    vec3 H = u_rbHole.xyz;
    float tC = dot(H - ro, rd);
    if (tC <= 0.0) return col;
    float b = length(ro + rd * tC - H);
    float rH = HOLE_R_DRAW * u_rbHole.w;
    if (rH > 1e-4) {
        col *= mix(1.0, 0.45, exp(-b * b / (9.0 * rH * rH)));
        col += HOLE_RING * 1.6 * exp(-pow((b - 1.5 * rH) / (0.18 * rH), 2.0)) * u_rbHole.w;
        col *= smoothstep(0.92 * rH, 1.02 * rH, b);
    }
    col += vec3(1.0, 0.9, 1.0) * u_rbFlash * (3.0 * exp(-b * b / 0.36) + 0.15);
    return col;
}
vec3 nsSurface(vec3 n, vec3 rd, mat3 toP, float t) {
    float mu = max(dot(n, -rd), 0.0), om = 1.0 - mu;
    float I = max(1.0 - LIMB_U1 * om - LIMB_U2 * om * om, 0.03);
#if DWARF_STYLE == 1
    vec3 c = dwarfSurface(toP * n, t) * I;
#else
    float T = cloudT(toP * n, t) * pow(I, 1.0 / 3.0);
    vec3 c = radiance(T);
#endif
    return mix(c, LIMB_PURP * length(c) * 0.9, smoothstep(0.35, 1.0, om) * 0.7);
}
vec3 nsGlow(vec3 C, vec3 ro, vec3 rd, vec3 rt, vec3 up, float R, float tMo, float tRing, float trRing) {
    float tS = dot(C - ro, rd);
    if (tS <= 0.0 || R < 1e-5) return vec3(0.0);
    vec3 Cs = ro + rd * tS - C;
    float q = length(Cs) / R;
    float occ = (tMo < tS) ? 0.0 : 1.0;
    occ *= (tRing < tS) ? trRing : 1.0;
    vec2 sp = vec2(dot(Cs, rt), dot(Cs, up)) / R;
    float spikes = exp(-abs(sp.x) * 6.0) * exp(-abs(sp.y) * 0.35) + exp(-abs(sp.y) * 6.0) * exp(-abs(sp.x) * 0.35);
    float g = NS_GLOW_CORE * exp(-q * q / 2.2) + NS_GLOW_HALO / (1.0 + q * q * 0.35) + NS_SPIKES * spikes / (1.0 + 0.05 * q * q);
    return NS_GLOW_COL * g * occ;
}
uniform float u_dbCamOn;
uniform vec3 u_dbRo, u_dbRt, u_dbUp, u_dbFw;
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float t = iTime;
    float el = CAM_EL;
    float az = CAM_ORBIT * t;
    vec3 base = CAM_D * vec3(cos(el) * sin(az), sin(el), -cos(el) * cos(az));
    vec3 aim = AIM_OFF * normalize(vec3(base.x, 0.0, base.z));
    vec3 fw = normalize(aim - base);
    vec3 rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw));
    vec3 up = cross(fw, rt);
    { float cr = cos(CAM_ROLL), sr = sin(CAM_ROLL);
      vec3 r2 = cr * rt + sr * up; up = -sr * rt + cr * up; rt = r2; }
    vec3 bxA = rt, byA = up, bzA = fw;
    gBX = bxA; gBY = byA; gBZ = bzA;
    if (u_dbCamOn > 0.5) {
        base = u_dbRo; rt = u_dbRt; up = u_dbUp; fw = u_dbFw;
    }
    vec3 ro = base + u_pad_move.x * rt + u_pad_move.y * up + u_pad_move.z * fw;
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / min(iResolution.x, iResolution.y) * 2.0 * FOV;
    vec3 dc = vec3(uv * u_view_fov, 1.0);
    if (dot(u_view_q, u_view_q) > 0.5) dc = viewRot(u_view_q, dc);
    if (dot(u_pad_cam_q, u_pad_cam_q) > 0.5) dc = viewRot(u_pad_cam_q, dc);
    vec3 rd = normalize(dc.z * fw + dc.x * rt + dc.y * up);

    float ct = cos(SPIN_TILT), st = sin(SPIN_TILT);
    mat3 toP = mat3(1, 0, 0, 0, ct, -st, 0, st, ct);

#if CLEAN_BG
    vec3 col = CLEAN_BG_COL * (0.55 + 0.45 * smoothstep(-0.7, 0.7, rd.y));
#else
    vec3 col = sky(rd);
#endif
#if BOX_ON
    col = boxGlass(ro, rd, bxA, byA, bzA, col, t, toP);
#endif
    float NSR = DW_R * (1.0 - u_nsStar.x);
    float b = dot(ro, rd), c = dot(ro, ro) - NSR * NSR, disc = b * b - c;
    float bImp = length(cross(ro, rd));
    float tPl = 1e9;
    if (disc > 0.0 && NSR > 1e-5) {
        tPl = -b - sqrt(disc);
        vec3 P = ro + rd * tPl;
        vec3 n = normalize(P);
        float mu = max(dot(n, -rd), 0.0);
        float om = 1.0 - mu;
        float I = max(1.0 - LIMB_U1 * om - LIMB_U2 * om * om, 0.03);
#if DWARF_STYLE == 1
        col = dwarfSurface(toP * n, t) * I;
#else
        float T = cloudT(toP * n, t);
        T *= pow(I, 1.0 / 3.0);
        col = radiance(T);
#endif
        col = mix(col, LIMB_PURP * length(col) * 0.9, smoothstep(0.35, 1.0, om) * 0.7);
        col *= 1.0 + 2.5 * u_nsStar.y;
    }
    if (u_ns2.w > 0.5) {
        vec3 oc = ro - u_ns2.xyz;
        float b2 = dot(oc, rd), d2 = b2 * b2 - (dot(oc, oc) - NS2_R * NS2_R);
        if (d2 > 0.0) {
            float t2 = -b2 - sqrt(d2);
            if (t2 > 0.0 && t2 < tPl) { tPl = t2; col = nsSurface(normalize(oc + rd * t2), rd, toP, t + 17.0); }
        }
    }
    float hz = (bImp - DW_R) / DW_R;
    if (u_nsStar.x > 0.0) hz = 1e3;
    if (hz > -0.02) col += radiance(T_ZONE * 0.92) * 0.35 * exp(-max(hz, 0.0) / HAZE_H) * smoothstep(-0.02, 0.0, hz) * step(0.0, -b);
    if (hz > 0.0 && tPl > 1e8) col += UV_HALO_COL * UV_HALO_G * exp(-hz / UV_HALO_H) * (0.85 + 0.15 * sin(0.7 * t + 3.0 * atan(dot(rd, up), dot(rd, rt))));

    float tMo = 1e9; vec3 moCol = vec3(0.0); int moHit = -1; vec3 moNrmD = vec3(0.0);
    for (int i = 0; i < N_MOONS; i++) {
        vec4 m = texelFetch(u_moons, ivec2(i, 0), 0);
        vec3 oc = ro - m.xyz;
        float bb = dot(oc, rd), cc = dot(oc, oc) - m.w * m.w, dd = bb * bb - cc;
        if (dd <= 0.0) continue;
        float tm = -bb - sqrt(dd);
        if (MOON_MAT[i] == 7) continue;
        if (MOON_MAT[i] == 6) {
            vec3 nD;
            tm = disnubHit(ro, rd, m, t, nD);
            if (tm >= tMo) continue;
            tMo = tm; moHit = i; moNrmD = nD;
            continue;
        }
        if (tm <= 0.0 || tm >= tMo) continue;
        tMo = tm; moHit = i;
    }
    if (moHit >= 0) {
        vec4 m = texelFetch(u_moons, ivec2(moHit, 0), 0);
        vec3 P = ro + rd * tMo, n = normalize(P - m.xyz);
        float cosI = max(dot(n, -rd), 0.0);
        float F = METAL_R0 + (1.0 - METAL_R0) * pow(1.0 - cosI, 5.0);
        if (MOON_MAT[moHit] == 6) n = moNrmD;
        if (MOON_MAT[moHit] != 0) {
            vec2 hw = texelFetch(u_moons, ivec2(moHit, 1), 0).w * vec2(1.0, 0.0) + vec2(0.0, texelFetch(u_moons, ivec2(moHit, 2), 0).x);
            float wob = t - hw.y;
            if (MOON_MAT[moHit] == 5 && wob > 0.0 && wob < 3.0) {
                vec3 q = normalize(P - m.xyz);
                n = normalize(n + 0.35 * exp(-1.6 * wob) * sin(14.0 * wob) * vec3(sin(9.0 * q.y + 3.0), sin(7.0 * q.z), sin(8.0 * q.x + 1.0)));
            }
            moCol = weirdMoon(MOON_MAT[moHit], P, n, rd, m, hw.x, wob, t, toP);
        } else {
            vec3 tint = mix(MOON_METAL[moHit], vec3(1.0), pow(1.0 - cosI, 5.0));
            moCol = tint * F * mirrorEnv(P + n * 1e-3, reflect(rd, n), t, toP) + vec3(0.002);
            moCol += MOON_METAL[moHit] * MOON_LIT * (LIGHT_COL * LIGHT_K * max(dot(n, -normalize(P)), 0.0) / max(dot(P, P), 0.5) + wallBounce(P, n, t, toP));
        }
        float heat = texelFetch(u_moons, ivec2(moHit, 1), 0).w;
        if (heat > 0.01) {
            float hq = clamp(heat, 0.0, 2.0);
            vec3 hot = mix(vec3(0.35, 0.02, 0.0), vec3(1.0, 0.30, 0.04), smoothstep(0.0, 0.6, hq));
            hot = mix(hot, vec3(1.0, 0.85, 0.55), smoothstep(0.8, 1.6, hq));
            moCol = mix(moCol, hot * HOT_GAIN * hq, smoothstep(0.0, 0.25, hq));
        }
    }
#if ROCK_ON
    {
        vec3 lo = transpose(u_rockRot) * (ro - u_rockPos), ld = transpose(u_rockRot) * rd;
        float B = dot(lo, ld), C = dot(lo, lo) - RK_BOUND * RK_BOUND, dd = B * B - C;
        if (dd > 0.0) {
            float tk = max(-B - sqrt(dd), 0.0), tEnd = min(-B + sqrt(dd), tMo);
            bool hit = false;
            for (int k = 0; k < 96 && tk < tEnd; k++) {
                float d = ROCK_SDF(lo + ld * tk + u_rockCom);
                if (d < 0.0006) { hit = true; break; }
                tk += d * 0.6;
            }
            if (hit && tk < tMo) {
                vec3 pb = lo + ld * tk + u_rockCom;
                const vec2 e = vec2(0.0008, -0.0008);
                vec3 nb = normalize(e.xyy * ROCK_SDF(pb + e.xyy) + e.yyx * ROCK_SDF(pb + e.yyx)
                                  + e.yxy * ROCK_SDF(pb + e.yxy) + e.xxx * ROCK_SDF(pb + e.xxx));
                vec3 P = ro + rd * tk, n = u_rockRot * nb;
                float rr = length(P);
                tMo = tk; moHit = -1;
                float alb = T_ALB * (0.75 + 0.25 * sin(19.0 * pb.x + 2.0) * sin(23.0 * pb.y) * sin(29.0 * pb.z + 1.0));
                moCol = T_COL * alb * (LIGHT_COL * LIGHT_K * max(dot(n, -P / rr), 0.0) / (rr * rr) + wallBounce(P, n, t, toP)) + vec3(0.002);
            }
        }
    }
#endif
    float tRing = 1e9, trRing = 1.0;
    vec3 tubes = vec3(0.0), aurora = vec3(0.0);
    {
        if (abs(rd.y) > 1e-5) {
            float tR0 = -ro.y / rd.y; vec3 P0 = ro + rd * tR0; float r0 = length(P0.xz);
            if (tR0 > 0.0 && tR0 < tPl && tR0 < tMo && r0 > R_IN && r0 < R_OUT) {
                vec4 rg0 = ringAt(r0, atan(P0.z, P0.x));
                tRing = tR0; trRing = exp(-(TAU_K * rg0.x + TAU_K2 * rg0.x * rg0.x) / abs(rd.y));
            }
        }
        tubes = noteBolts(ro, rd, tPl, tRing, trRing, t) + planetTube(ro, rd, tPl, tRing, trRing, t);
        aurora = auroraCurtains(ro, rd, min(tPl, tMo), trRing, fragCoord);
        aurora += meteors(ro, rd, min(tPl, tMo), tRing, trRing, t);
        if (u_chaseOn > 0.5) aurora += chase(ro, rd, min(tPl, tMo), tRing, trRing, t);
        if (u_shotN > 0.5) aurora += shipShots(ro, rd, min(tPl, tMo), tRing, trRing, t);
    }
    if (tMo < tPl) col = moCol;

    if (abs(rd.y) > 1e-5) {
        float tR = -ro.y / rd.y;
        vec3 P = ro + rd * tR;
        float r = length(P.xz);
        if (tR > 0.0 && r > R_IN && r < R_OUT && tR < tPl) {
            float th = atan(P.z, P.x);
            float sg = ringAt(r, th).x;
            float tau = TAU_K * sg + TAU_K2 * sg * sg;
            float mu = abs(rd.y);
            float trans = exp(-tau / mu);
            vec3 L = normalize(-P), V = -rd;
            float cLV = clamp(dot(L, V), -1.0, 1.0);
            float ph = mix(lambertPhase(acos(cLV)), 0.25 * hgPhase(-cLV, DUST_G), DUST_FWD);
            float beta = asin(clamp(DW_R / r, 0.0, 1.0));
            float top = (2.0 * beta * beta * beta / 3.0) / 3.14159265;
            float phMS = mix(ph, 0.5, MULTI_SC);
            vec3 lit = RING_LIGHT * RING_WHITE * RING_FLUOR * LIGHT_K * RING_ALB * (phMS / max(r * r, RING_R2MIN) + top) * (1.0 - trans);
            float spk = ringSpokes(r, th, t);
            if (spk > 0.0) {
                float fwd = 1.0 - smoothstep(-0.8, 0.0, cLV);
                lit *= 1.0 - SPOKE_DARK * spk * (1.0 - fwd);
                lit += RING_LIGHT * RING_WHITE * LIGHT_K * SPOKE_BRIGHT * spk * (0.25 + fwd) / (r * r);
            }
            if (tR < tMo) col = col * trans + lit;
            else col = moCol;
        }
    }
    col += tubes + aurora;
    col += nsGlow(vec3(0.0), ro, rd, rt, up, DW_R, tMo, tRing, trRing) * (1.0 - u_nsStar.x) * (1.0 + 3.0 * u_nsStar.y);
    if (u_ns2.w > 0.5) col += nsGlow(u_ns2.xyz, ro, rd, rt, up, NS2_R, tMo, tRing, trRing);
    col = 1.0 - exp(-col);

    if (u_pad_boop.z > 0.5) {
        vec2 n = (fragCoord - u_pad_boop.xy) / min(iResolution.x, iResolution.y);
        float rho  = length(n);
        float ring = 1.0 - smoothstep(0.0025, 0.0060, abs(rho - 0.105));
        float onx  = 1.0 - smoothstep(0.0020, 0.0050, min(abs(n.x), abs(n.y)));
        float arms = onx * smoothstep(0.030, 0.040, rho) * (1.0 - smoothstep(0.086, 0.098, rho));
        float pip  = 1.0 - smoothstep(0.004, 0.009, rho);
        float hi   = clamp(max(arms, pip), 0.0, 1.0);
        col = mix(col, vec3(0.62, 0.94, 1.0), 0.80 * hi + 0.26 * ring);
    }
#if HOLE_ON
    if (u_dustOn > 0.5) col += DUST_COL * DUST_K * float(texelFetch(u_dust, ivec2(fragCoord), 0).r);
    if (u_cmetOn > 0.5) { float cm = CMET_K * float(texelFetch(u_cmet, ivec2(fragCoord), 0).r); col += mix(MT_COL_BODY, MT_COL_HEAD, clamp(cm, 0.0, 1.0)) * cm; }
    if (u_rbHole.w > 0.0 || u_rbFlash > 0.0) col = holeFx(ro, rd, col);
#endif
    fragColor = vec4(col, 1.0);
}
