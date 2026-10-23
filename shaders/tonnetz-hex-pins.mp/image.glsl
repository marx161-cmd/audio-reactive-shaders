// Tonnetz-Hex-Pins: image pass.
//
// The Tonnetz note lattice as ruby hex pins that rise with their notes
// (compute stage). Light walks crystal to crystal (refraction, total internal
// reflection, Beer-Lambert absorption); chord tones get glowing green top
// outlines; the drums are an outer ring of pins; two UV fairy lights fly over
// the lattice with recorded history trails.

uniform sampler2D u_pins;
uniform float u_pinS;
uniform float u_axM;

#define H_LIM      0.36
#define H_SINK     0.34
#define WHEEL_SCALE 23.72

#define STYLE      0
#define MAX_HEX    700
#define CAM_ELEV   0.65
#define CAM_DIST   3.0
#define FOCAL      1.35
#define KEY_DIR    vec3(-0.75, 0.55, 0.35)
#define KEY_COL    vec3(1.0, 0.97, 0.92)
#define KEY_GAIN   1.6
#define AMB        0.10
#define STEEL      vec3(0.30, 0.31, 0.33)
#define TOP_ALB    vec3(0.16, 0.165, 0.175)
#define SPEC_POW   40.0
#define SPEC_K     0.6
#define SEAM_W     0.0011
#define GAP_COL    vec3(0.004, 0.004, 0.005)
#define ACT_LO     1.6
#define ACT_HI     2.6
#define TOP_GLOW   1.6
#define WALL_AO    0.03
#define NEON_COL   vec3(1.00, 0.06, 0.32)
#define TEAL_COL   vec3(0.00, 0.78, 0.68)
#define CHORD_PIN  vec3(0.05, 0.90, 1.00)
uniform sampler2D u_chord;
#define OCT_C0     vec3(0.90, 0.08, 0.03)
#define OCT_C1     vec3(1.00, 0.45, 0.08)
#define OCT_C2     vec3(1.00, 0.90, 0.75)
#define OCT_C3     vec3(0.50, 0.72, 1.00)
#define SPOT_R0    0.95
#define SPOT_R1    1.30
#define EXPOSURE   1.0

#define KEY_DIM     0.0
#define ORBIT_BEATS 8.0
#define SPIN_BEATS  1.0
#define SPIN_ECC    0.7
#define ORBIT_R     1.0
#define PAIR_R      0.06
#define LIGHT_H     0.42
#define PL1_COL     vec3(0.20, 0.42, 1.00)
#define PL2_COL     vec3(0.42, 0.12, 0.95)
#define UV_VIS      0.25
#define BOTH_UV     1
#define RUBY_BODY   vec3(0.32, 0.010, 0.030)
#define RUBY_AMB    0.45
#define RUBY_FLUO   vec3(1.00, 0.04, 0.10)
#define FLUO_BLUE   1.6
#define FLUO_UV     2.4
#define FLUO_ACT    0.6
#define RUBY_F0     0.077
#define RUBY_IOR    1.77
#define RUBY_ABS    vec3(1.5, 38.0, 30.0)
#define RUBY_PATH_MAX 0.25
#define FLUO_DENS   36.0
#define CRYSTAL_NODES 8
#define FLUO_VOL    1.6
#define UV_ABS      45.0
#define BEHIND_COL  vec3(0.030, 0.006, 0.010)
#define ENV_TOP     vec3(0.05, 0.07, 0.12)
#define ENV_HOR     vec3(0.30, 0.20, 0.34)
#define RUBY_SPEC   220.0
#define UV_GAIN     1.6
#define PAINT_COL   vec3(0.22, 1.00, 0.35)
#define PAINT_COVER 0.85
#define RIM_W       0.0018
#define RIM_GAIN    7.0
#define PAINT_GAIN  5.0
#define DRUM_PIN_GLOW 0.8
#define PL_GAIN     1.1
#define PL_FALL     1.0
#define PL_CORE     0.008
#define PL_HALO     0.06
#define PL_VIS      2.5
#define DRUM_COL    vec3(0.95, 0.30, 0.10)
#define SKY_BLACK   1
#define FOG_COL     (SKY_BLACK == 1 ? vec3(0.0) : vec3(0.05, 0.10, 0.13))
#define EDGE_R0     1.15
#define EDGE_R1     1.35
#define DRUM_GAIN   0.3
#define TRAIL_TURNS 2.0
#define TRAIL_SEG   64
#define TRAIL_W     0.004
#define TRAIL_GAIN  0.9
uniform float u_beat_count;
uniform float u_kick;

const vec2 HN[3] = vec2[3](vec2(1.0, 0.0), vec2(0.5, 0.8660254), vec2(-0.5, 0.8660254));
const vec2 HA[3] = vec2[3](vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(-1.0, 1.0));

vec2 axialToWorld(vec2 ax) {
    return vec2(u_pinS * 1.7320508 * (ax.x + 0.5 * ax.y), u_pinS * 1.5 * ax.y);
}

vec2 hexRoundW(vec2 xz) {
    float r = xz.y / (1.5 * u_pinS);
    float q = xz.x / (1.7320508 * u_pinS) - 0.5 * r;
    vec3 c = vec3(q, r, -q - r);
    vec3 rc = floor(c + 0.5);
    vec3 d = abs(rc - c);
    if (d.x > d.y && d.x > d.z) rc.x = -rc.y - rc.z;
    else if (d.y > d.z)         rc.y = -rc.x - rc.z;
    return rc.xy;
}

float pinTop(vec2 ax) {
    ivec2 t = ivec2(ax + u_axM);
    ivec2 sz = textureSize(u_pins, 0);
    if (any(lessThan(t, ivec2(0))) || any(greaterThanEqual(t, sz))) return 0.0;
    return texelFetch(u_pins, t, 0).x;
}

float insidePath(vec3 P, vec3 rt, vec2 c, float top, out vec3 nExit) {
    float A = 0.8660254 * u_pinS;
    float tOut = RUBY_PATH_MAX;
    nExit = vec3(0.0, -1.0, 0.0);
    for (int k = 0; k < 3; k++) {
        float dn = dot(rt.xz, HN[k]);
        if (abs(dn) < 1e-6) continue;
        float sg = dn > 0.0 ? 1.0 : -1.0;
        float tt = (sg * A - dot(P.xz - c, HN[k])) / dn;
        if (tt > 1e-5 && tt < tOut) { tOut = tt; nExit = vec3(sg * HN[k].x, 0.0, sg * HN[k].y); }
    }
    if (rt.y > 1e-6) { float tt = (top - P.y) / rt.y; if (tt < tOut) { tOut = tt; nExit = vec3(0.0, 1.0, 0.0); } }
    if (rt.y < -1e-6) { float tt = (-H_SINK - P.y) / rt.y; if (tt < tOut) { tOut = tt; nExit = vec3(0.0, -1.0, 0.0); } }
    return max(tOut, 0.0);
}

bool hexTrace(vec3 ro, vec3 rd, float tStart, float tEnd, bool skipFirst,
              out float tHit, out vec3 nHit, out vec2 axHit, out float lowTop) {
    vec2 ax = hexRoundW((ro + rd * tStart).xz);
    float A = 0.8660254 * u_pinS;
    float tIn = tStart;
    vec3 nIn = vec3(0.0, 1.0, 0.0);
    float prevTop = 0.0;
    tHit = 0.0; nHit = nIn; axHit = ax; lowTop = 0.0;
    for (int i = 0; i < MAX_HEX; i++) {
        vec2 c = axialToWorld(ax);
        float top = pinTop(ax);
        float tOut = 1e9;
        vec2 stepAx = vec2(0.0), nOut = vec2(0.0);
        for (int k = 0; k < 3; k++) {
            float dn = dot(rd.xz, HN[k]);
            if (abs(dn) < 1e-7) continue;
            float sg = dn > 0.0 ? 1.0 : -1.0;
            float tt = (sg * A - dot(ro.xz - c, HN[k])) / dn;
            if (tt < tOut) { tOut = tt; stepAx = sg * HA[k]; nOut = sg * HN[k]; }
        }
        float yIn = ro.y + rd.y * tIn;
        if (!(skipFirst && i == 0)) {
            if (yIn <= top) { tHit = tIn; nHit = nIn; axHit = ax; lowTop = prevTop; return true; }
            if (rd.y < 0.0) {
                float tTop = (top - ro.y) / rd.y;
                if (tTop <= tOut) { tHit = max(tTop, tIn); nHit = vec3(0.0, 1.0, 0.0); axHit = ax; lowTop = top; return true; }
            }
        }
        if (tOut > tEnd) return false;
        if (rd.y > 0.0 && ro.y + rd.y * tOut > H_LIM) return false;
        prevTop = top;
        tIn = tOut;
        nIn = vec3(-nOut.x, 0.0, -nOut.y);
        ax += stepAx;
    }
    return false;
}

vec3 octCol(float oct) {
    float x = clamp(oct / 8.0, 0.0, 1.0) * 3.0;
    if (x < 1.0) return mix(OCT_C0, OCT_C1, smoothstep(0.0, 1.0, x));
    if (x < 2.0) return mix(OCT_C1, OCT_C2, smoothstep(0.0, 1.0, x - 1.0));
    return mix(OCT_C2, OCT_C3, smoothstep(0.0, 1.0, x - 2.0));
}

void lightAt(vec3 P, vec3 n, vec3 rd, bool isTop, vec3 L, float Ldist, vec3 Lcol,
             inout vec3 difSum, inout vec3 specSum) {
    float dif = max(dot(n, L), 0.0);
    if (dif <= 0.0) return;
    float tS, lS; vec3 nS; vec2 aS;
    vec3 Ps = P + n * 1e-4 + vec3(0.0, 1e-4, 0.0);
    if (hexTrace(Ps, L, 0.0, Ldist, isTop, tS, nS, aS, lS)) return;
    difSum += Lcol * dif;
    specSum += Lcol * pow(max(dot(reflect(rd, n), L), 0.0), SPEC_POW);
}

#define FW_BEATS    8.0
#define FW_A        0.55
#define FW_B        0.45
#define FW_H        0.44
#define FW_C        0.08
#define F_R1_LO     0.07
#define F_R1_HI     0.17
#define F_R2_LO     0.025
#define F_R2_HI     0.05
#define F_CURLS     3.0
#define F_RUSH      2.6
uniform float u_rms;

vec3 planeVec(vec3 nrm, float ang) {
    vec3 a = normalize(cross(nrm, abs(nrm.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    vec3 b = cross(nrm, a);
    return cos(ang) * a + sin(ang) * b;
}

#define DANCE_BEATS 4.0
#define DANCE_A     0.95
#define DANCE_B     0.80
#define DANCE_LAG   0.35
#define BAL_LEAP    0.16
#define BAL_LOW     0.30
#define BAL_SWIRL   2.4
#define BAL_SWIRL_R 0.35
#define RIB_R       0.07
#define RIB_BEATS   2.0
vec3 balletOne(float cnt, float sgn, float lag) {
    const float TAU = 6.2831853;
    float w = TAU * cnt / DANCE_BEATS - lag;
    vec2 xz = sgn * vec2(DANCE_A * sin(w), DANCE_B * sin(0.6180340 * w + 0.7));
    vec2 vel = sgn * vec2(DANCE_A * cos(w), 0.6180340 * DANCE_B * cos(0.6180340 * w + 0.7));
    float r = length(xz);
    float y = BAL_LOW + BAL_LEAP * exp(-(r * r) / 0.18) + 0.03 * sin(1.4142136 * w + 1.1);
    vec2 fwd = normalize(vel + 1e-5);
    float ra = TAU * cnt / RIB_BEATS + (sgn > 0.0 ? 0.0 : 3.14159);
    xz += fwd * RIB_R * cos(ra);
    y += RIB_R * sin(ra);
    float sw = BAL_SWIRL * exp(-(r * r) / (BAL_SWIRL_R * BAL_SWIRL_R));
    xz = mat2(cos(sw), sin(sw), -sin(sw), cos(sw)) * xz;
    return vec3(xz.x, y, xz.y);
}
#define FAIRY_MODE  1
#define BIN_T       0.4
#define BIN_CX      0.0
#define BIN_CZ      0.0
#define BIN_H       0.46
#define BIN_CY      0.0
#define BIN_SEP0    0.035
#define BIN_SEP1    0.50
#define BIN_BOB     0.07
#define BIN_KICK_SEP 0.22
#define F_SPIN_BEAT  1.6
#define F_SPIN_HI_K  1.5
uniform sampler2D u_fairyring;
vec3 rot2(vec3 v, int ax, float a) {
    float c = cos(a), s = sin(a);
    if (ax == 0) v.xy = mat2(c, -s, s, c) * v.xy; else v.yz = mat2(c, -s, s, c) * v.yz;
    return v;
}
void lightPositionsAt(float cnt, out vec3 p1, out vec3 p2) {
#if FAIRY_MODE == 0
    p1 = balletOne(cnt, 1.0, 0.0);
    p2 = balletOne(cnt, -1.0, DANCE_LAG);
#else
    float t = cnt * BIN_T;
    vec3 center = vec3(sin(t * 0.3) * BIN_CX, BIN_H + sin(t * 0.4) * BIN_CY, cos(t * 0.25) * BIN_CZ);
    float kickSep = texelFetch(u_chord, ivec2(15, 0), 0).r;
    float sep = BIN_SEP0 + BIN_SEP1 * pow(sin(t * 0.5), 4.0) + BIN_KICK_SEP * kickSep;
    float hi = texelFetch(u_fairyring, ivec2(19, 0), 0).r;
    float spin = texelFetch(u_chord, ivec2(14, 0), 0).r
               - (u_beat_count - cnt) * F_SPIN_BEAT * (1.0 + F_SPIN_HI_K * hi);
    float tumble = t * 1.2;
    vec3 off = vec3(cos(spin) * sep, sin(spin * 1.5) * BIN_BOB, sin(spin) * sep);
    off = rot2(off, 0, tumble);
    off = rot2(off, 1, tumble * 0.8);
    off.y *= 0.45;
    p1 = center + off;
    p2 = center - off;
#endif
}

uniform sampler2D u_fairyhist;
#define FH_N 256
int fhHead() { return int(texelFetch(u_fairyhist, ivec2(FH_N, 0), 0).r + 0.5); }
void lightPositions(out vec3 p1, out vec3 p2) {
    int h = fhHead();
    p1 = texelFetch(u_fairyhist, ivec2(h, 0), 0).xyz;
    p2 = texelFetch(u_fairyhist, ivec2(h, 1), 0).xyz;
}

float raySegDist(vec3 ro, vec3 rd, vec3 a, vec3 b, out float tr) {
    vec3 ab = b - a, w = ro - a;
    float aa = dot(ab, ab), bb = dot(rd, ab), dd = dot(rd, w), ee = dot(ab, w);
    float den = max(aa - bb * bb, 1e-12);
    float sseg = clamp((ee - bb * dd) / den, 0.0, 1.0);
    vec3 q = a + ab * sseg;
    tr = dot(q - ro, rd);
    return length(ro + rd * tr - q);
}

vec3 chordCol(float pc) {
    if (pc > 11.5) return NEON_COL;
    float m = texelFetch(u_chord, ivec2(int(clamp(pc, 0.0, 11.0) + 0.5), 0), 0).r;
    return mix(TEAL_COL, CHORD_PIN, clamp(m, 0.0, 1.0));
}

float pinOct(vec2 ax) {
    ivec2 t = ivec2(ax + u_axM);
    ivec2 sz = textureSize(u_pins, 0);
    if (any(lessThan(t, ivec2(0))) || any(greaterThanEqual(t, sz))) return 0.0;
    return texelFetch(u_pins, t, 0).w;
}

vec3 seeBehind(vec3 o, vec3 d, vec3 lp1, vec3 lp2) {
    if (d.y >= 0.0 && o.y > H_LIM) return ENV_TOP;
    float t1 = d.y < -1e-4 ? (-H_SINK - o.y) / d.y + 1e-3 : 3.0;
    float tH, lowTop; vec3 n; vec2 ax;
    if (!hexTrace(o, d, 0.0, t1, false, tH, n, ax, lowTop))
        return mix(ENV_HOR, ENV_TOP, clamp(d.y * 1.5, 0.0, 1.0)) * 0.6;
    vec3 P = o + d * tH;
    float top = pinTop(ax);
    vec3 d1 = lp1 - P, d2 = lp2 - P;
    float l1 = length(d1), l2 = length(d2);
    float eB = max(dot(n, d1 / l1), 0.0) * PL_GAIN / (1.0 + (l1 / PL_FALL) * (l1 / PL_FALL));
    float eU = max(dot(n, d2 / l2), 0.0) * UV_GAIN / (1.0 + (l2 / PL_FALL) * (l2 / PL_FALL));
    float act = smoothstep(ACT_LO, ACT_HI, abs(top) * WHEEL_SCALE);
    vec3 c = RUBY_FLUO * (FLUO_BLUE * eB + FLUO_UV * eU) * (1.0 + FLUO_ACT * act) * 0.06 * FLUO_DENS * 0.1;
    if (n.y > 0.5) {
        float pcv = pinOct(ax);
        float member = pcv > 11.5 ? 0.0 : texelFetch(u_chord, ivec2(int(clamp(pcv, 0.0, 11.0) + 0.5), 0), 0).r;
        c += PAINT_COL * PAINT_GAIN * eU * member * act;
    }
    return c;
}

vec3 skyCol(vec3 d) {
#if SKY_BLACK
    return vec3(0.0);
#else
    return mix(vec3(0.05, 0.10, 0.13), vec3(0.01, 0.02, 0.035), clamp(d.y, 0.0, 1.0));
#endif
}

float schlick(float cosT) { return RUBY_F0 + (1.0 - RUBY_F0) * pow(1.0 - clamp(cosT, 0.0, 1.0), 5.0); }

float irrB(vec3 X, vec3 lp1) {
#if BOTH_UV
    return 0.0;
#else
    float l = length(lp1 - X); return PL_GAIN / (1.0 + (l / PL_FALL) * (l / PL_FALL));
#endif
}
float irrU(vec3 X, vec3 lp2) { float l = length(lp2 - X); return UV_GAIN / (1.0 + (l / PL_FALL) * (l / PL_FALL)); }

float penetrate(vec3 X, vec3 lp) {
    vec3 d = lp - X;
    float dist = length(d);
    d /= max(dist, 1e-6);
    vec2 ax = hexRoundW(X.xz);
    float A = 0.8660254 * u_pinS;
    float tIn = 0.0, len = 0.0;
    for (int i = 0; i < 96; i++) {
        vec2 c = axialToWorld(ax);
        float top = pinTop(ax);
        float tOut = 1e9; vec2 stepAx = vec2(0.0);
        for (int k = 0; k < 3; k++) {
            float dn = dot(d.xz, HN[k]);
            if (abs(dn) < 1e-7) continue;
            float sg = dn > 0.0 ? 1.0 : -1.0;
            float tt = (sg * A - dot(X.xz - c, HN[k])) / dn;
            if (tt < tOut) { tOut = tt; stepAx = sg * HA[k]; }
        }
        float tEnd = min(tOut, dist);
        float yIn = X.y + d.y * tIn;
        if (yIn < top) {
            float tTop = d.y > 1e-6 ? (top - X.y) / d.y : 1e9;
            len += max(min(tTop, tEnd) - tIn, 0.0);
        }
        if (tOut >= dist || X.y + d.y * tOut > H_LIM || len * UV_ABS > 8.0) break;
        tIn = tOut;
        ax += stepAx;
    }
    return exp(-UV_ABS * len);
}

vec3 crystalWalk(vec3 P, vec3 n, vec3 d, vec2 ax, float eB0, float eU0, vec3 lp1, vec3 lp2) {
    vec3 C = vec3(0.0), T = vec3(1.0);
    float crossed = 0.0;
    vec3 pos = P;
    vec2 cell = ax;
    vec3 dir = refract(d, n, 1.0 / RUBY_IOR);
    for (int node = 0; node < CRYSTAL_NODES; node++) {
        vec2 c = axialToWorld(cell);
        float top = pinTop(cell);
        vec3 nEx;
        float L = insidePath(pos, dir, c, top, nEx);
        vec3 mid = pos + dir * (0.5 * L);
        float pen1, pen2;
        if (node == 0) { pen1 = penetrate(mid, lp1); pen2 = penetrate(mid, lp2); }
        else { pen1 = pen2 = exp(-UV_ABS * 0.5 * crossed); }
        float eB = irrB(mid, lp1) * pen1;
        float eU = irrU(mid, lp2) * pen2;
#if BOTH_UV
        eU += irrU(mid, lp1) * pen1;
#endif
        float act = smoothstep(ACT_LO, ACT_HI, abs(top) * WHEEL_SCALE);
        float exc = (FLUO_BLUE * eB + FLUO_UV * eU) * (1.0 + FLUO_ACT * act);
        C += T * RUBY_FLUO * exc * FLUO_VOL * (1.0 - exp(-RUBY_ABS.r * L)) / RUBY_ABS.r;
        T *= exp(-RUBY_ABS * L);
        crossed += L;
        vec3 pex = pos + dir * L;
        if (nEx.y < -0.5) break;
        vec3 dOut = refract(dir, -nEx, RUBY_IOR);
        if (dot(dOut, dOut) < 1e-6) {
            dir = reflect(dir, -nEx);
            pos = pex - nEx * 1e-5;
            continue;
        }
        T *= 1.0 - schlick(abs(dot(dOut, nEx)));
        vec3 o = pex + dOut * 2e-4;
        float tEnd = dOut.y < -1e-4 ? (-H_SINK - o.y) / dOut.y + 1e-3 : 3.0;
        float tH2, low2; vec3 n2; vec2 ax2;
        if (!hexTrace(o, dOut, 0.0, tEnd, false, tH2, n2, ax2, low2)) {
            C += T * skyCol(dOut);
            return C;
        }
        if (tH2 < 1e-3 && abs(nEx.y) < 0.5) n2 = -nEx;
        pos = o + dOut * tH2;
        cell = ax2;
        T *= 1.0 - schlick(max(dot(n2, -dOut), 0.0));
        dir = refract(dOut, n2, 1.0 / RUBY_IOR);
        if (max(T.r, max(T.g, T.b)) < 0.01) return C;
    }
    return C + T * BEHIND_COL;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    if (u_pinS <= 0.0) { fragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec3 ro = CAM_DIST * vec3(0.0, sin(CAM_ELEV), -cos(CAM_ELEV));
    vec3 fw = normalize(-ro);
    vec3 rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw));
    vec3 up = cross(fw, rt);
    vec3 rd = normalize(FOCAL * fw + uv.x * rt + uv.y * up);

    float drum = DRUM_GAIN * u_kick * u_kick;
    vec3 col = skyCol(rd);
#if SKY_BLACK == 0
    col += DRUM_COL * drum * exp(-max(rd.y, 0.0) * 8.0) * step(0.0, rd.y);
#endif
    vec3 lp1, lp2;
    lightPositions(lp1, lp2);
    float tFog = 1e3;
    if (rd.y < -1e-4) {
        float t0 = (H_LIM - ro.y) / rd.y;
        float t1 = (-H_SINK - ro.y) / rd.y + 1e-3;
        float tH, lowTop; vec3 n; vec2 ax;
        if (hexTrace(ro, rd, t0, t1, false, tH, n, ax, lowTop)) {
            vec3 P = ro + rd * tH;
            tFog = tH;
            vec2 c = axialToWorld(ax);
            float top = pinTop(ax);
            float r = length(c);
            float light = smoothstep(SPOT_R1, SPOT_R0, r);
            bool isTop = n.y > 0.5;
            vec3 d1 = lp1 - P, d2 = lp2 - P;
            float l1 = length(d1), l2 = length(d2);
            float eB = 0.0, eU = 0.0;
            vec3 dummyD = vec3(0.0), specB = vec3(0.0), specU = vec3(0.0);
            vec3 tB = vec3(0.0), tU = vec3(0.0);
            lightAt(P, n, rd, isTop, d1 / l1, l1, vec3(PL_GAIN / (1.0 + (l1 / PL_FALL) * (l1 / PL_FALL))), tB, specB);
            lightAt(P, n, rd, isTop, d2 / l2, l2, vec3(UV_GAIN / (1.0 + (l2 / PL_FALL) * (l2 / PL_FALL))), tU, specU);
            eB = tB.x; eU = tU.x;
#if BOTH_UV
            eU += eB * UV_GAIN / PL_GAIN;
            eB = 0.0;
#endif
            float act = smoothstep(ACT_LO, ACT_HI, abs(top) * WHEEL_SCALE);
            float pcv = pinOct(ax);
            float cosV = max(dot(n, -rd), 0.0);
            float F = RUBY_F0 + (1.0 - RUBY_F0) * pow(1.0 - cosV, 5.0);
            vec3 Lb = d1 / l1;
            float glint = pow(max(dot(reflect(rd, n), Lb), 0.0), RUBY_SPEC);
            float shadeB = eB > 0.0 ? 1.0 : 0.0;
            vec3 rr = reflect(rd, n);
            vec3 env = mix(ENV_HOR, ENV_TOP, clamp(rr.y * 1.5, 0.0, 1.0));
            vec3 surf = F * env
                      + PL1_COL * (PL_GAIN / (1.0 + (l1 / PL_FALL) * (l1 / PL_FALL))) * shadeB * glint * (F + 0.04) * 6.0;
            vec3 inside = (1.0 - F) * crystalWalk(P, n, rd, ax, eB, eU, lp1, lp2);
            vec3 ruby = surf + inside;

            if (isTop) {
                vec2 d = P.xz - c;
                float m = max(max(abs(dot(d, HN[0])), abs(dot(d, HN[1]))), abs(dot(d, HN[2])));
                float e = 0.8660254 * u_pinS - m;
                float aa = max(fwidth(e), 1e-6);
                float seam = 1.0 - smoothstep(SEAM_W - aa, SEAM_W + aa, e);
                float member = pcv > 11.5 ? 0.0 : texelFetch(u_chord, ivec2(int(clamp(pcv, 0.0, 11.0) + 0.5), 0), 0).r;
                float rim = (1.0 - smoothstep(SEAM_W + RIM_W - aa, SEAM_W + RIM_W + aa, e)) * (1.0 - seam);
                vec3 paint = PAINT_COL * RIM_GAIN * eU * member * act * rim;
                vec3 drum = pcv > 11.5 ? NEON_COL * DRUM_PIN_GLOW * act : vec3(0.0);
                col = mix(surf + inside * (1.0 - rim * member * act) + paint + drum, GAP_COL, seam);
            } else {
                col = ruby;
            }
#if SKY_BLACK
            col *= smoothstep(EDGE_R1, EDGE_R0, length(c));
#endif
        }
    }
    if (tFog < 1e3) {
        float fog = 1.0 - exp(-tFog * WHEEL_SCALE * 0.015);
        col = mix(col, FOG_COL, clamp(fog, 0.0, 1.0));
    }
    {
        float span = TRAIL_TURNS * SPIN_BEATS;
        int h = fhHead();
        float cNow = texelFetch(u_fairyhist, ivec2(h, 0), 0).w;
        float perEntry = mod(cNow - texelFetch(u_fairyhist, ivec2((h + FH_N - 1) % FH_N, 0), 0).w + 1024.0, 1024.0);
        int stride = max(1, int(ceil(span / max(perEntry, 1e-4) / float(TRAIL_SEG))));
        vec3 prev1 = lp1, prev2 = lp2;
        for (int k = 1; k <= TRAIL_SEG; k++) {
            int idx = (h - k * stride + FH_N * 64) % FH_N;
            if (k * stride >= FH_N) break;
            vec4 e1 = texelFetch(u_fairyhist, ivec2(idx, 0), 0);
            if (e1.w == 0.0 && dot(e1.xyz, e1.xyz) == 0.0) break;
            float age = mod(cNow - e1.w + 1024.0, 1024.0) / span;
            if (age > 1.0) break;
            vec3 q1 = e1.xyz;
            vec3 q2 = texelFetch(u_fairyhist, ivec2(idx, 1), 0).xyz;
            float fade = pow(1.0 - age + 1.0 / float(TRAIL_SEG), 1.5);
            for (int j = 0; j < 2; j++) {
                float tr;
                float dseg = j == 0 ? raySegDist(ro, rd, prev1, q1, tr) : raySegDist(ro, rd, prev2, q2, tr);
                if (tr <= 0.0 || tr > tFog) continue;
                float w = TRAIL_W * (1.0 - 0.6 * age);
                col += (j == 0 && BOTH_UV == 0 ? PL1_COL : PL2_COL * UV_VIS) * TRAIL_GAIN * fade
                     * (exp(-(dseg / w) * (dseg / w)) + 0.15 * exp(-dseg / (6.0 * w)));
            }
            prev1 = q1;
            prev2 = q2;
        }
    }
    for (int k = 0; k < 2; k++) {
        vec3 lp = k == 0 ? lp1 : lp2;
        float tl = dot(lp - ro, rd);
        if (tl <= 0.0 || tl > tFog) continue;
        float dl = length(ro + rd * tl - lp);
        col += (k == 0 && BOTH_UV == 0 ? PL1_COL : PL2_COL * UV_VIS) * PL_VIS * (exp(-(dl / PL_CORE) * (dl / PL_CORE)) + 0.25 * exp(-dl / PL_HALO));
    }
    col *= EXPOSURE;
    col = col / (1.0 + col);
#if STYLE != 0
    col = pow(col, vec3(1.0 / 2.2));
#endif
    fragColor = vec4(col, 1.0);
}
