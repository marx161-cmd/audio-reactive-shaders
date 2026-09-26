// Kerr-Black-Hole-Visualizer: common code, prepended to every pass.
//
// Constants, uniforms and shared functions: beat clock, orbital mechanics
// (Kepler orbits, gear phases), star/sky sampling (BC6H cube map), body
// surfaces, accretion disk emission and the Kerr-Newman geodesic walk
// (precomputed light paths in u_kerrpaths, walked as a polyline).
// Audio arrives in the `bands` texture: 120 semitone bands, 0..1.

const float PI  = 3.141592653589793;
const float TAU = 6.283185307179586;

uniform sampler2D bands;

uniform sampler2D u_dwarfmap;
uniform sampler2D u_chromomap;
uniform sampler2D u_giantmap;
uniform sampler2D u_rockmap;
uniform sampler2D u_ringprof;
uniform sampler2D u_stars;
uniform sampler2D u_cells;
uniform samplerCube u_skycube;
uniform sampler2D u_nebula;
#define NEB_W       4096.0
#define NEB_GAIN    15.0
#define NEB_SAT     1.35
#define NEB_FLOOR   0.0
#define NEB_TINT    vec3(1.00, 0.90, 0.78)
#define NEB_BASS_DRIVE 1.8

uniform float u_cam_elev;
uniform float u_cam_yaw;
uniform float u_steer;
uniform vec3  u_cam_move;

#define STATE_W 128
#define STATE_H 4

#define BODIES_PER_COPY 6
#define ROW2_BODY_BASE  3
#define MOONS_PER_COPY  6

#define MOON_BEATS_PG0   1.0
#define MOON_BEATS_PG1   2.0
#define MOON_BEATS_PG2   4.0
#define MOON_BEATS_PG3   8.0
#define MOON_BEATS_DGA0  2.0
#define MOON_BEATS_DGA1  4.0

#define MOON_SURGE      2.6
#define MOON_SURGE_MIX  0.45

#define STATE_BEAT_PX   25
#define STATE_MET_PX    26
#define STATE_BEATCNT_PX 29
#define BEAT_WRAP     1024.0

#define ROSETTE_BEATS   32.0
#define ORBIT_BEATS_PG  8.0
#define ORBIT_BEATS_DGA 16.0

float tempoBeatCount() { return texelFetch(iChannel0, ivec2(STATE_BEATCNT_PX, 0), 0).r; }
float tempoTurns(float beatsPerCycle) { return tempoBeatCount() / beatsPerCycle; }

#define MET_PERI        0.90
#define MET_ECC         1.05
#define MET_ECC_BOUND   0.78
#define MET_LOOP_BEATS  6.0
#define MET_SPAN_BEATS  2.0
#define MET_VIS_R       3.60
#define MET_TAIL_T      0.55
#define MET_TRAIL_R0    0.004
#define MET_TRAIL_RW    0.020
#define MET_TRAIL_GAIN  0.55
#define MET_CORE_R      0.010
#define MET_GLOW_R      0.022

float keplerE(float M, float e) {
    float E = M + e * sin(M);
    for (int i = 0; i < 5; i++) {
        E -= (E - e * sin(E) - M) / max(1.0 - e * cos(E), 1e-3);
    }
    return E;
}

float keplerH(float M, float e) {
    float am = abs(M);
    float H  = (am < 1.0) ? am / max(e - 1.0, 1e-3)
                          : log(2.0 * am / e + 1.8);
    for (int i = 0; i < 6; i++) {
        float sh = sinh(H), ch = cosh(H);
        H -= (e * sh - H - am) / max(e * ch - 1.0, 1e-4);
    }
    return (M < 0.0) ? -H : H;
}

vec3 meteorPos(vec4 el0, vec4 el1, vec4 el2, float t, out float rNow) {
    float tPeri = el0.x, n = max(el0.y, 1e-4), rp = el0.z, e = el0.w;
    vec3  u = el1.xyz;
    vec3  v = cross(el2.xyz, u);
    float M = n * (t - tPeri);

    if (e < 1.0) {
        float a  = rp / (1.0 - e);
        float E  = keplerE(M, e);
        float ce = cos(E), se = sin(E);
        rNow = a * (1.0 - e * ce);
        return u * (a * (ce - e)) + v * (a * sqrt(max(1.0 - e * e, 1e-6)) * se);
    }

    float a  = rp / (e - 1.0);
    float H  = keplerH(M, e);
    float ch = cosh(H), sh = sinh(H);
    rNow = a * (e * ch - 1.0);
    return u * (a * (e - ch)) + v * (a * sqrt(max(e * e - 1.0, 1e-6)) * sh);
}

void meteorAt(vec4 el0, vec4 el1, vec4 el2, float t,
              out vec3 pos, out vec3 vdir, out float rNow) {
    float rA, rB;
    pos  = meteorPos(el0, el1, el2, t, rNow);
    vec3 pB = meteorPos(el0, el1, el2, t + 0.02, rB);
    vec3 pA = meteorPos(el0, el1, el2, t - 0.02, rA);
    vdir = normalize(pB - pA + vec3(1e-9));
}

float meteorEdgeM(float e) {
    float a = MET_PERI / max(e - 1.0, 1e-3);
    float H = acosh(max((MET_VIS_R / a + 1.0) / e, 1.0));
    return e * sinh(H) - H;
}

float meteorTStart(vec4 el0, vec4 el1) {
    float n = max(el0.y, 1e-4);
    if (el0.w < 1.0) return el0.x - 3.14159265 / n;
    return el0.x - meteorEdgeM(el0.w) / n;
}
float meteorTEnd(vec4 el0, vec4 el1) {
    float n = max(el0.y, 1e-4);
    if (el0.w < 1.0) return meteorTStart(el0, el1) + el1.w * 6.28318531 / n;
    return el0.x + meteorEdgeM(el0.w) / n;
}

#define COMET_COL_HEAD  vec3(1.00, 0.86, 0.62)
#define COMET_COL_BODY  vec3(1.00, 0.34, 0.10)
#define COMET_COL_DEEP  vec3(0.55, 0.09, 0.12)

#define N_BANDS  108.0

#define SPHERE_RADIUS  8.10
#define SENSOR_POS     vec2(0.0, 0.0)

#define TRACKBALL_GAIN 2.6
#define SPIN_MAX       0.6
#define SPIN_TAU       0.55

#define GEAR_BINARY    0
#define GEAR_PGIANT    1
#define GEAR_PROCK     2
#define GEAR_DGA       3
#define GEAR_DGB       4
#define GEAR_STAR0     5
#define GEAR_STAR1     6

#define BARY_R_0      1.90
#define BARY_R_1      1.90
#define ECC_0         0.82
#define ECC_1         0.82
#define PERI_0        4.712389
#define PERI_1        1.570796
#define BARY_REF      2.20
#define BINARY_TILT   0.08

#define ORB_PGIANT    0.250
#define ORB_PROCK     0.400
#define ORB_DGB       0.280
#define ORB_DGA       0.500

#define PULSAR_R      0.0012
#define DWARF_R       0.09
#define PGIANT_R      0.070
#define DGA_R         0.080

#define TOTAL_MOONS    6

#define M0_PGIANT  0
#define M0_DGA     4

#define PGIANT_M_R0  0.122
#define PGIANT_M_DR  0.014
#define DGA_M_R0     0.172
#define DGA_M_DR     0.032

#define PGIANT_BOUND  0.187
#define DGA_BOUND     0.230

#define N_COPIES      1

#define COPY_TILT     0.5235988

#define COPY_PHASE_0  0.00
#define COPY_PHASE_1  0.50

#define DWARF_TINT_0  vec3(0.88, 0.26, 0.12)
#define DWARF_TINT_1  vec3(1.12, 0.60, 0.24)
#define DWARF_TINT_2  vec3(0.74, 0.90, 1.28)

#define GIANT_SHIFT_0  0.00
#define GIANT_SHIFT_1  0.30
#define GIANT_SHIFT_2 -0.26

#define CORONA_SHIFT_0  0.00
#define CORONA_SHIFT_1  0.20
#define CORONA_SHIFT_2 -0.18

vec3  copyDwarfTint(int k) {
    return k == 0 ? DWARF_TINT_0 : (k == 1 ? DWARF_TINT_1 : DWARF_TINT_2);
}
float copyGiantShift(int k) {
    return k == 0 ? GIANT_SHIFT_0 : (k == 1 ? GIANT_SHIFT_1 : GIANT_SHIFT_2);
}
float copyCoronaShift(int k) {
    return k == 0 ? CORONA_SHIFT_0 : (k == 1 ? CORONA_SHIFT_1 : CORONA_SHIFT_2);
}

float copyPhase(int k) { return (k == 1) ? COPY_PHASE_1 : COPY_PHASE_0; }

float cloneClock(int k) { return float(k) * 0.173; }

vec3 rotAxis(vec3 v, vec3 a, float ang) {
    float c = cos(ang), sn = sin(ang);
    return v * c + cross(a, v) * sn + a * dot(a, v) * (1.0 - c);
}

#define GEAR_STATE_BASE 17
float statePhase(int g) { return texelFetch(iChannel0, ivec2(GEAR_STATE_BASE + g, 0), 0).r; }
float rosetteSpin()     { return TAU * tempoTurns(ROSETTE_BEATS); }

void gearBasis(float idx, out vec3 uAxis, out vec3 vAxis, out vec3 nAxis);

vec3 copyPlaneNormal(int k) {
    vec3 u, v, n;
    gearBasis(float(GEAR_BINARY), u, v, n);
    float tilt = (float(k) - 0.5) * 2.0 * COPY_TILT;
    return normalize(rotAxis(n, u, tilt));
}
vec3 copyRot(vec3 val, int k) {
    return rotAxis(val, copyPlaneNormal(k), rosetteSpin() + TAU * copyPhase(k));
}

#define BH_RS         0.048
#define BH_CAPTURE    0.125

#define STAR1_COL     vec3(0.50, 0.76, 1.34)
#define STAR2_COL     vec3(1.06, 0.13, 0.04)
#define RING_ALBEDO   vec3(0.92, 0.86, 0.74)
#define RING_DISKLIGHT 1.35
#define TREBLE_DISP_MIN  0.0
#define TREBLE_DISP_GAIN 2.2
#define BH_SEC_DISP   0.025

#define CORONA_R      0.007
#define CORONA_R_OLD  0.020
#define CORONA_POW    5.6
#define CORONA_CORE   vec3(0.72, 0.86, 1.60)
#define CORONA_EDGE   vec3(0.24, 0.42, 1.30)

#define HALO_R        0.065
#define HALO_W_R      0.00086
#define HALO_W_Z      0.00059
#define HALO_BOUND    0.130
#ifdef OFFLINE
#define HALO_STEPS    96
#else
#define HALO_STEPS    24
#endif
#define HALO_GAIN     12.0
#define HALO_BREATHE  0.22
#define HALO_KNOTS    9.0
#define HALO_COCOON   0.30

float synchrotronDensity(vec3 p, vec3 mAxis, float bass, float treble, float et) {
    float zMag = dot(p, mAxis);
    float rMag = length(p - mAxis * zMag);

    float ringR = HALO_R * (1.0 + HALO_BREATHE * bass);
    float dr = rMag - ringR;
    float torus = exp(-(dr * dr) / HALO_W_R) * exp(-(zMag * zMag) / HALO_W_Z);

    vec3 e1 = normalize(cross(mAxis, vec3(0.0, 0.0, 1.0)) + vec3(1e-5, 0.0, 0.0));
    vec3 e2 = cross(mAxis, e1);
    float az = atan(dot(p, e2), dot(p, e1));
    float knots = 0.70 + 0.30 * sin(az * HALO_KNOTS - et * 2.2);
    torus *= mix(1.0, knots, clamp(treble * 1.6, 0.0, 1.0));

    float rr = length(p);
    float cocoon = exp(-(rr * rr) / (HALO_R * HALO_R * 0.55)) * HALO_COCOON;

    return torus + cocoon;
}

#define DGA_HUE       0.00
#define DGA_TINT      vec3(1.22, 1.03, 0.68)
#define DGA_BANDS     0.85

#define HUE_GOLDEN    0.6180339887
#define PG_HUE_OFF    0.50
#define PG_HUE_SAT    0.60
#define PG_HUE_BASE   vec3(1.00, 0.70, 0.45)
#define PG_ROCK_BASE  vec3(0.95, 0.93, 0.90)

float bodyLap(float beatsPer, float phaseOffset) {
    return floor(tempoTurns(beatsPer) + phaseOffset);
}
float lapHue(float beatsPer, float phaseOffset, float offset) {
    return fract(bodyLap(beatsPer, phaseOffset) * HUE_GOLDEN + offset) * TAU;
}

#define RING_IN       1.50
#define RING_OUT      1.80
#define RING_OUT_WIDE 2.40
#define RING_DENS_K2  1.30

#define SKY_GAIN      3.20
#define SKY_AUDIO     0.12
#define SKY_DRIFT     0.0
#define SKY_ROT_SENS  0.0
#define SKY_PITCH     1.267
#define SKY_YAW       0.60
#define SKY_YAW_SENS  0.10
#define SKY_ROLL_SENS 0.10
#define SKY_DRAG_R0    116.348319
#define SKY_DRAG_F0    -15.872584
#define SKY_DRAG_SENS  0.10
#define SKY_DRAG_SX    1.0
#define SKY_DRAG_SY    1.0
#define DWARF_MAP_W   2048.0
#define MAP_W         1024.0
#define GRAN_SCALE    7.5
#define DWARF_SPIN    0.045
#define DWARF_DIFF    0.000
#define DWARF_CHURN   0.280
#define DWARF_WARP    0.035
#define CHROMO_GAIN   0.55
#define ERUPT_SCALE   3.2
#define ERUPT_GATE    0.74
#define ERUPT_RATE    0.22
#define ERUPT_GAIN    4.2
#define GRAN_FINE    55.0
#define GRAN_DEPTH    0.42
#define SPOT_SCALE    5.5
#define SPOT_GATE     0.76
#define SPOT_DARK     0.30

#define LEAK_K            0.28
#define H2_K              0.20
#define H3_K              0.10
#define H4_K              0.06
#define FLOOR_K           0.42
#define TILT_DB           4.5
#define DB_MIN            -90.0
#define DB_SPAN           120.0
#define DISPLAY_DB        24.0
#define AGC_FLOOR_DB      -55.0
#define AGC_RELEASE_DBPS  6.0
#define AGC_ATTACK        0.5
#define BAND_ATTACK_TAU   0.030
#define BAND_RELEASE_TAU  0.200

#define N_TARGETS     3
#define ARC_GATE      0.20
#define ARC_TAU       0.60
#define IDLE_GATE     0.06
#define IDLE_PERIOD   6.0
#define IDLE_LEVEL    0.55

vec4 quatMul(vec4 a, vec4 b) {
    return vec4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
                a.w * b.w - dot(a.xyz, b.xyz));
}

vec2 intersectSphere(vec3 ro, vec3 rd, float r) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - r * r;
    float h = b * b - c;
    if (h < 0.0) return vec2(-1.0);
    h = sqrt(h);
    return vec2(-b - h, -b + h);
}

vec4 quatAxisAngle(vec3 axis, float ang) {
    return vec4(normalize(axis) * sin(ang * 0.5), cos(ang * 0.5));
}
vec4 quatFromTo(vec3 a, vec3 b) {
    float d = dot(a, b);
    if (d < -0.999999) {
        vec3 o = normalize(cross(a, abs(a.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0)));
        return vec4(o, 0.0);
    }
    return normalize(vec4(cross(a, b), 1.0 + d));
}

float gearTilt(float idx) {
    if (idx == float(GEAR_BINARY)) return BINARY_TILT;
    if (idx == float(GEAR_PGIANT)) return -0.612;
    if (idx == float(GEAR_PROCK))  return  0.31;
    if (idx == float(GEAR_DGA))    return  0.5236;
    if (idx == float(GEAR_DGB))    return -0.22;
    return 0.0;
}

void gearBasis(float idx, out vec3 uAxis, out vec3 vAxis, out vec3 nAxis) {
    float t = gearTilt(idx);
    if (idx == float(GEAR_DGA)) {
        uAxis = vec3(1.0, 0.0, 0.0);
        vAxis = vec3(0.0, 0.866025, 0.50);
    } else if (idx == float(GEAR_PGIANT)) {
        uAxis = vec3(0.0, 1.0, 0.0);
        vAxis = vec3(sin(t), 0.0, cos(t));
    } else if (idx == float(GEAR_PROCK)) {
        uAxis = vec3(1.0, 0.0, 0.0);
        vAxis = vec3(0.0, cos(t), sin(t));
    } else {
        uAxis = vec3(0.0, 1.0, 0.0);
        vAxis = vec3(sin(t), 0.0, cos(t));
    }
    nAxis = normalize(cross(uAxis, vAxis));
}

vec4 viewQuat(float elev, float yaw) {
    vec3 u, v, n0;
    gearBasis(float(GEAR_BINARY), u, v, n0);
    vec3 nt = vec3(0.0, sin(elev), -cos(elev));
    return normalize(quatMul(quatAxisAngle(nt, yaw), quatFromTo(normalize(n0), nt)));
}

vec3 moonJitter(float n) {
    return vec3(fract(n * 0.6180339887 + 0.371),
                fract(n * 0.7548776662 + 0.618),
                fract(n * 0.4142135624 + 0.117));
}

float wrapU(float u, float w) { return (0.5 + fract(u) * w) / (w + 1.0); }

void bodyFrame(vec3 axis, vec3 ref, out vec3 e1, out vec3 e2) {
    e1 = normalize(ref - axis * dot(ref, axis));
    e2 = cross(axis, e1);
}

vec3 hueRotate(vec3 c, float a) {
    const vec3 k = vec3(0.57735027);
    float ca = cos(a), sa = sin(a);
    return clamp(c * ca + cross(k, c) * sa + k * dot(k, c) * (1.0 - ca), 0.0, 1.0);
}

#define STAR_CW       1440.0
#define STAR_CH       720.0
#define STAR_TEX_W    2048.0
#define STAR_MAG_ZERO 6.2
#define STAR_GAIN     0.55
#define STAR_PSF_MIN  0.055

#define POND_FREQ    9.0
#define POND_RATE    0.45
#define POND_DECAY   1.6
#define POND_AMP     0.030
#define POND_DRIVE   1.4

float gBass  = 0.0;
float gPondR = 0.0;
float gOutE = 1.0;
#define EIN_OVER     0.02
#define EIN_EDGE     0.02
float gSplitM = 1.0;
float gPondM  = 0.0;

#define PULSE_DET_PX  30
#define PULSE_PX      31
#define PULSE_N       8
#define PULSE_SPEED   0.275
#define PULSE_W       0.045
#define PULSE_LIFE    2.8
#define PULSE_AMP     0.030
vec2 gPulse[PULSE_N];

#define SHEET_DEPTH  0.25
#define SHEET_W      0.50
vec3  gHoleDir = vec3(0.0, 0.0, 1.0);
vec3  gCamR = vec3(1.0, 0.0, 0.0);
vec3  gCamU = vec3(0.0, 1.0, 0.0);

vec3 skyWarp(vec3 d) {
    float push = SHEET_DEPTH * clamp(gBass, 0.0, 1.0);
    if (push < 1e-5) return d;
    float c = clamp(dot(d, gHoleDir), -1.0, 1.0);
    vec3 perp = d - c * gHoleDir;
    float s = length(perp);
    if (s < 1e-5) return d;
    float th = acos(c);
    th += push * th * exp(-(th * th) / (SHEET_W * SHEET_W));
    return cos(th) * gHoleDir + sin(th) * (perp / s);
}

vec3 starTint(float bv) {
    vec3 c = mix(vec3(0.62, 0.74, 1.00), vec3(0.95, 0.96, 1.00),
                 smoothstep(-0.40, 0.00, bv));
    c = mix(c, vec3(1.00, 0.95, 0.83), smoothstep(0.00, 0.65, bv));
    c = mix(c, vec3(1.00, 0.72, 0.45), smoothstep(0.65, 1.55, bv));
    return mix(c, vec3(1.00, 0.55, 0.32), smoothstep(1.55, 2.50, bv));
}

vec4 starTexel(float idx) {
    float y = floor(idx / STAR_TEX_W);
    return texelFetch(u_stars, ivec2(int(idx - y * STAR_TEX_W), int(y)), 0);
}

#define PSF_K     0.88
#define PSF_NORM  3.0
#define STAR_MAX_FOOT 6.0
#define STAR_WIN_MAX  2
vec3 starField(vec2 guv, vec2 foot) {
    float gx = guv.x * STAR_CW;
    float gy = guv.y * STAR_CH;
    int cx = int(floor(gx));
    int cy = int(floor(gy));
    vec2 f = clamp(foot, vec2(STAR_PSF_MIN), vec2(STAR_MAX_FOOT));
    vec2 psf = f * PSF_K;
    vec2 inv2 = 1.0 / (psf * psf);
    int rx = int(min(ceil(f.x * 0.5), float(STAR_WIN_MAX)));
    int ry = int(min(ceil(f.y * 0.5), float(STAR_WIN_MAX)));

    vec3 acc = vec3(0.0);
    for (int dy = -ry; dy <= ry; dy++) {
        int yy = cy + dy;
        if (yy < 0 || yy >= int(STAR_CH)) continue;
        for (int dx = -rx; dx <= rx; dx++) {
            int xx = cx + dx;
            xx = (xx < 0) ? xx + int(STAR_CW)
               : (xx >= int(STAR_CW)) ? xx - int(STAR_CW) : xx;
            vec2 sc = texelFetch(u_cells, ivec2(xx, yy), 0).rg;
            int n = int(sc.y);
            for (int k = 0; k < n; k++) {
                if (k >= 48) break;
                vec4 s = starTexel(sc.x + float(k));
                vec2 sp = vec2(float(xx) + s.x, float(yy) + s.y);
                vec2 del = vec2(gx, gy) - sp;
                del.x -= STAR_CW * floor(del.x / STAR_CW + 0.5);
                float d2 = dot(del * del, inv2);
                float core = 1.0 / (1.0 + d2);
                float c2 = core * core;
                float flux = exp2((STAR_MAG_ZERO - s.z) * 1.32860);
                acc += starTint(s.w) * (c2 * c2 * flux);
            }
        }
    }
    vec2 norm2 = max(f, vec2(1.0));
    vec3 s = acc * (STAR_GAIN * PSF_NORM / (norm2.x * norm2.y));

    #define STAR_KNEE 3.3
    return s / (1.0 + s * STAR_KNEE);
}

vec3 skyDirFinal(vec3 dir) {
    vec3 d = dir;
    {
        float gx = (u_cam_move.x - SKY_DRAG_R0) * SKY_DRAG_SENS * SKY_DRAG_SX;
        float gy = (u_cam_move.z - SKY_DRAG_F0) * SKY_DRAG_SENS * SKY_DRAG_SY;
        d = rotAxis(d, gCamU,  gx);
        d = rotAxis(d, gCamR, -gy);
        float skyPitch = SKY_PITCH + u_cam_move.y * SKY_ROT_SENS;
        float skyYaw   = SKY_YAW   + SKY_DRAG_R0 * SKY_YAW_SENS;
        float cp = cos(skyPitch), sp = sin(skyPitch);
        d = vec3(d.x, d.y * cp - d.z * sp, d.y * sp + d.z * cp);
        float cy = cos(skyYaw), sy = sin(skyYaw);
        d = vec3(d.x * cy - d.z * sy, d.y, d.x * sy + d.z * cy);
        float skyRoll = SKY_DRAG_F0 * SKY_ROLL_SENS;
        if (skyRoll != 0.0) {
            vec3 ax = normalize(gHoleDir);
            float cr = cos(skyRoll), sr = sin(skyRoll);
            d = d * cr + cross(ax, d) * sr + ax * dot(ax, d) * (1.0 - cr);
        }
    }
    if (SKY_DRIFT != 0.0) {
        float a = SKY_DRIFT * iTime, ca = cos(a), sa = sin(a);
        d = vec3(d.x * ca - d.z * sa, d.y, d.x * sa + d.z * ca);
    }
    return skyWarp(d);
}
vec2 skyDirUV(vec3 dir) {
    vec3 d = skyDirFinal(dir);
    return vec2(atan(d.z, d.x) / TAU + 0.5,
                1.0 - acos(clamp(d.y, -1.0, 1.0)) / PI);
}

#define NEB_SHOW 0
vec3 nebulaAt(vec3 dir, float bass) {
    if (NEB_SHOW == 0) return vec3(0.0);
    vec2 nuv = skyDirUV(dir);
    vec3 neb = texture(u_nebula, vec2(wrapU(nuv.x, NEB_W), nuv.y)).rgb;
    float nl = dot(neb, vec3(0.299, 0.587, 0.114));
    neb = mix(vec3(nl), neb, NEB_SAT) * NEB_TINT;
    float g = NEB_GAIN * (1.0 + NEB_BASS_DRIVE * clamp(bass, 0.0, 1.0));
    return max(neb - NEB_FLOOR, 0.0) * g;
}

#define SKY64_GAIN     2.0
#define SKY64_SHARP    0.5
#define SKY64_MAXGRAD  0.026
vec3 skyColorG(vec3 dir, float treble, float bImpact, float gscale) {
    vec3 d = skyDirFinal(dir);
    vec3 gx = dFdx(d) * SKY64_SHARP, gy = dFdy(d) * SKY64_SHARP;
    float lx = length(gx), ly = length(gy);
    if (lx > SKY64_MAXGRAD) gx *= SKY64_MAXGRAD / lx;
    if (ly > SKY64_MAXGRAD) gy *= SKY64_MAXGRAD / ly;
    vec3 c = textureGrad(u_skycube, d, gx * gscale, gy * gscale).rgb * SKY64_GAIN;
    c *= (SKY_GAIN + SKY_AUDIO * treble);

    float bEff = (bImpact > 1e-4) ? bImpact : 1e5;
    float lensStress = smoothstep(BH_CAPTURE * 1.05, BH_CAPTURE * 2.5, bEff);
    vec3 greyscale = vec3(dot(c, vec3(0.299, 0.587, 0.114)));
    c = mix(greyscale, c, clamp(lensStress, 0.35, 1.0));
    return c;
}
vec3 skyColor(vec3 dir, float treble, float bImpact) {
    return skyColorG(dir, treble, bImpact, 1.0);
}

#define GRADE_EXPOSURE   1.0
#define GRADE_CONTRAST   1.4
#define GRADE_SATURATION 1.2
#define GRADE_GAMMA      1.0
#define GRADE_TINT       vec3(1.0, 1.0, 1.0)
#define GRADE_BLACK     -0.02
vec3 globalGrade(vec3 c) {
    c *= GRADE_EXPOSURE * GRADE_TINT;
    c = max(c, 0.0);
    c = 0.18 * pow(c / 0.18 + 1e-6, vec3(GRADE_CONTRAST));
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = max(mix(vec3(l), c, GRADE_SATURATION), 0.0);
    c = pow(c, vec3(1.0 / GRADE_GAMMA));
    return max(c + GRADE_BLACK, 0.0);
}
#define SKY_CA        0.025
#ifdef OFFLINE
#define SPLIT_N       15
#else
#define SPLIT_N       7
#endif
#define GLOW_BLUR     8.0
#define SPLIT_SIGMA   0.45
vec3 skyRadial(vec3 d, float k) {
    float c  = clamp(dot(d, gHoleDir), -1.0, 1.0);
    vec3  t  = d - c * gHoleDir;
    float tl = length(t);
    if (tl < 1e-5) return d;
    float th = acos(c) * (1.0 + k);
    return normalize(cos(th) * gHoleDir + sin(th) * (t / tl));
}
vec3 skyColorDispersed(vec3 rd, vec3 bend, float treble, float bImpact) {
    float bEff = (bImpact > 1e-4) ? bImpact : 1e5;
    float prox = 1.0 - smoothstep(BH_CAPTURE * 1.05, BH_CAPTURE * 2.5, bEff);
    float disp = BH_SEC_DISP * prox * (TREBLE_DISP_MIN + TREBLE_DISP_GAIN * treble);
    float ca   = SKY_CA * treble;
    disp *= gSplitM; ca *= gSplitM;
    vec3 dG = normalize(rd + bend);
    if (disp < 1e-4 && ca < 1e-4) {
        return skyColor(dG, treble, bEff);
    }
    vec3 acc = vec3(0.0), wsum = vec3(0.0);
    vec3 glow = skyColorG(dG, treble, bEff, GLOW_BLUR);
    for (int k = 0; k < SPLIT_N; k++) {
        float t = 1.0 - 2.0 * float(k) / float(SPLIT_N - 1);
        vec3 w = vec3(exp(-pow((t - 0.75) / SPLIT_SIGMA, 2.0)),
                      exp(-pow( t          / SPLIT_SIGMA, 2.0)),
                      exp(-pow((t + 0.75) / SPLIT_SIGMA, 2.0)));
        vec3 dk = skyRadial(normalize(rd + bend * (1.0 + t * disp)), t * ca);
        acc  += w * (skyColorG(dk, treble, bEff, 1.0) - skyColorG(dk, treble, bEff, GLOW_BLUR));
        wsum += w;
    }
    return glow + acc / wsum;
}

float occludeBy(vec3 wp, vec3 L, float dLight, vec3 c, float r) {
    vec3 rel = c - wp;
    float t = dot(rel, L);
    if (t <= 0.0 || t >= dLight) return 1.0;
    float d = length(rel - L * t);
    return mix(0.04, 1.0, smoothstep(r * 0.86, r * 1.16, d));
}

#define SURF_HQ_LIVE  1
#if defined(OFFLINE) || SURF_HQ_LIVE
#define SURF_HQ 1
#else
#define SURF_HQ 0
#endif
#define GRANQ_S        38.0
#define GRANQ_RATE     0.35
#define GRANQ_LANE     0.18
#define GRANQ_DEPTH    0.30
#define NETQ_S         6.0
#define NETQ_GAIN      0.10
#define SPOTQ_SCALE    2.6
#define SPOTQ_U        0.70
#define SPOTQ_P        0.64
#define SPOTQ_F        0.56
#define SPOTQ_UMB      0.18
#define SPOTQ_PEN      0.62
#define SPOTQ_FIL      140.0
#define FACQ_GAIN      0.55
#define LD_U1          0.62
#define LD_U2          0.18
#define GLOWQ_P_GAIN   0.10
#define GLOWQ_P_POW    2.2
#define GLOWQ_P_R      11.4
#define SPIKE_GAIN     0.9
#define SPIKE_W        0.0012
#define SPIKE_POW      1.25
#define SPIKE_ANGLE    0.35
#define GLOWQ_D_GAIN   0.08
#define GLOWQ_D_POW    2.4
#define GLOWQ_D_R      1.6

float h31(vec3 p);
vec3 h33(vec3 p) { return vec3(h31(p), h31(p + 17.31), h31(p + 41.77)); }
float vnoise3(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(h31(i),                   h31(i + vec3(1, 0, 0)), f.x),
                   mix(h31(i + vec3(0, 1, 0)),   h31(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(h31(i + vec3(0, 0, 1)),   h31(i + vec3(1, 0, 1)), f.x),
                   mix(h31(i + vec3(0, 1, 1)),   h31(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float fbm3(vec3 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 6; i++) { s += a * vnoise3(p); p = p * 2.03 + vec3(3.1, 1.7, 5.3); a *= 0.5; }
    return s / 0.984;
}
vec2 worley3(vec3 p, float t) {
    vec3 i = floor(p), f = fract(p);
    float F1 = 8.0, F2 = 8.0;
    for (int z = -1; z <= 1; z++)
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
        vec3 g = vec3(x, y, z);
        vec3 h = h33(i + g);
        vec3 o = 0.5 + 0.42 * sin(t * (0.6 + 0.8 * h) + 6.2831 * h);
        vec3 r = g + o - f;
        float d = dot(r, r);
        if (d < F1) { F2 = F1; F1 = d; } else if (d < F2) { F2 = d; }
    }
    return sqrt(vec2(F1, F2));
}

vec3 neutronStarSurface(vec3 pRel, vec3 rd, float R, float bass, float et) {
    return vec3(1.0) * (6.0 + bass * 6.0);
}

float h21(vec2 p) {
    uvec2 q = floatBitsToUint(p);
    uint n = q.x * 1597334673u ^ q.y * 3812015801u;
    n = (n ^ (n >> 15u)) * 2246822519u;
    n = (n ^ (n >> 13u)) * 3266489917u;
    n ^= n >> 16u;
    return float(n) * (1.0 / 4294967296.0);
}
float h31(vec3 p) {
    uvec3 q = floatBitsToUint(p);
    uint n = q.x * 1597334673u ^ q.y * 3812015801u ^ q.z * 2654435761u;
    n = (n ^ (n >> 15u)) * 2246822519u;
    n = (n ^ (n >> 13u)) * 3266489917u;
    n ^= n >> 16u;
    return float(n) * (1.0 / 4294967296.0);
}

float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = h21(i),               b = h21(i + vec2(1.0, 0.0));
    float c = h21(i + vec2(0.0, 1.0)), d = h21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float granulation(vec2 p) {
    return 0.58 * vnoise(p)
         + 0.28 * vnoise(p * 2.31 + 11.7)
         + 0.14 * vnoise(p * 5.13 + 3.1);
}

vec3 redDwarfSurface(vec3 pRel, vec3 rd, float R, float mid, float et, vec3 tint) {
    vec3 n  = pRel / R;
    float mu = clamp(dot(-rd, n), 0.0, 1.0);

    vec3 axis = normalize(vec3(0.12, 1.0, 0.05));
    vec3 e1, e2; bodyFrame(axis, vec3(1.0, 0.0, 0.0), e1, e2);
    float sLat = clamp(dot(n, axis), -1.0, 1.0);
    float x = dot(n, e1), y = dot(n, e2);

    float spin = et * (DWARF_SPIN + DWARF_DIFF * (1.0 - sLat * sLat));
    float lon  = atan(y, x) + spin;
    vec2  uv   = vec2(lon / TAU, 0.5 + asin(sLat) / PI);

    float w1 = vnoise(vec2(x, y) * GRAN_SCALE
                      + vec2(et * DWARF_CHURN, -et * DWARF_CHURN * 0.6));
    float w2 = vnoise(vec2(y, x) * GRAN_SCALE * 2.7
                      - vec2(et * DWARF_CHURN * 0.8, et * DWARF_CHURN * 0.4));
    uv += (vec2(w1, w2) - 0.5) * DWARF_WARP;

    vec3 base = texture(u_dwarfmap, vec2(wrapU(uv.x, DWARF_MAP_W),
                                         clamp(uv.y, 0.002, 0.998))).rgb;
    float chromo = texture(u_chromomap, vec2(wrapU(uv.x * 1.03, MAP_W),
                                             clamp(uv.y, 0.002, 0.998))).r;

    vec3 c = base * tint;

#if SURF_HQ
    float cs = cos(spin), sn = sin(spin);
    vec3  q  = axis * sLat + (e1 * (x * cs - y * sn) + e2 * (x * sn + y * cs));
    vec2  wg   = worley3(q * GRANQ_S, et * GRANQ_RATE);
    float lane = smoothstep(0.0, GRANQ_LANE, wg.y - wg.x);
    float gran = lane * (1.0 - 0.45 * wg.x);
    float sf   = fbm3(q * SPOTQ_SCALE + vec3(0.0, 0.0, et * DWARF_CHURN * 0.02) + 11.3);
    float umb  = smoothstep(SPOTQ_U, SPOTQ_U + 0.02, sf);
    float pen  = smoothstep(SPOTQ_P, SPOTQ_P + 0.02, sf);
    float fac  = smoothstep(SPOTQ_F, SPOTQ_P, sf) * (1.0 - pen);
    float fil  = vnoise3(q * SPOTQ_FIL + sf * 40.0);
    float gd   = GRANQ_DEPTH * sqrt(mu) * (1.0 - pen);
    c *= (1.0 - gd) + gd * 1.6 * gran;
    vec2  wn   = worley3(q * NETQ_S, et * GRANQ_RATE * 0.15);
    float net  = 1.0 - smoothstep(0.0, 0.10, wn.y - wn.x);
    float limb = pow(1.0 - mu, 1.5);
    c *= 1.0 + (FACQ_GAIN * fac + NETQ_GAIN * net) * limb;
    c *= mix(1.0, SPOTQ_PEN * (0.8 + 0.4 * fil), pen);
    c *= mix(1.0, SPOTQ_UMB, umb);
    c  = mix(c, c * vec3(1.0, 0.70, 0.52), umb * 0.7);
    float ld = max(1.0 - LD_U1 * (1.0 - mu) - LD_U2 * (1.0 - mu) * (1.0 - mu), 0.0);
#else
    float gran = granulation(vec2(x, y) * GRAN_FINE
                             + vec2(et * DWARF_CHURN * 0.5, -et * DWARF_CHURN * 0.3));
    c *= (1.0 - GRAN_DEPTH) + GRAN_DEPTH * 2.0 * gran;

    vec3 sp3 = n * SPOT_SCALE
             + vec3(-et * DWARF_CHURN * 0.06, et * DWARF_CHURN * 0.04,
                     et * DWARF_CHURN * 0.05) + 19.7;
    float sp = 0.62 * h31(floor(sp3)) + 0.38 * h31(floor(sp3 * 2.37 + 5.1));
    float spot = smoothstep(SPOT_GATE, SPOT_GATE + 0.14, sp);
    c *= 1.0 - spot * (1.0 - SPOT_DARK);

    float ld = pow(mu, 0.55);
#endif
    c = mix(vec3(0.42, 0.06, 0.012) * (0.5 + 0.5 * ld), c, smoothstep(0.0, 0.32, mu));

    c += vec3(1.0, 0.30, 0.09) * pow(chromo, 2.2) * (CHROMO_GAIN + mid * 1.10);

    vec3  ep   = n * ERUPT_SCALE
               + vec3(et * DWARF_CHURN * 0.12, -et * DWARF_CHURN * 0.07,
                      et * DWARF_CHURN * 0.09) + 7.3;
    vec3  cell = floor(ep);
    float cid  = h31(cell);
    float site = step(ERUPT_GATE, cid);
    vec3  jc   = vec3(h31(cell + 17.0), h31(cell + 31.0), h31(cell + 47.0)) - 0.5;
    float br   = mix(0.16, 0.46, h31(cell + 5.0));
    site      *= smoothstep(br, br * 0.12, length(fract(ep) - 0.5 - jc * 0.8));
    float phase = fract(et * ERUPT_RATE + h31(cell + 3.0) * 7.0);
    float pulse = 4.0 * phase * exp(1.0 - 4.0 * phase);
    vec3 eruptCol = vec3(1.0, 0.55, 0.18) * site * pulse * ERUPT_GAIN
                  * (0.35 + 0.65 * mid);

    return c * ld * (1.45 + mid * 1.55) + eruptCol * mix(1.0, ld, 0.35);
}

vec3 giantAlbedo(vec3 n, vec3 axis, float et, float hue, float bandScale,
                 float spinRate, float warpAmt) {
    vec3 e1, e2; bodyFrame(axis, vec3(0.0, 0.0, 1.0), e1, e2);
    float sLat = clamp(dot(n, axis), -1.0, 1.0);
    float x = dot(n, e1), y = dot(n, e2);

    float spin = et * (spinRate + spinRate * 0.65 * (1.0 - sLat * sLat));
    float lon  = atan(y, x) + spin;
    float lat  = asin(sLat);

    float w = vnoise(vec2(x, y) * 3.2 + vec2(et * 0.05, et * 0.02)) - 0.5;
    vec2 uv = vec2(lon / TAU + w * warpAmt,
                   clamp(0.5 + (lat / PI) * bandScale, 0.004, 0.996));

    return hueRotate(texture(u_giantmap, vec2(wrapU(uv.x, MAP_W), uv.y)).rgb, hue);
}

vec3 rockAlbedo(vec3 n, vec3 axis, float et, float seed, vec3 tint) {
    vec3 e1, e2; bodyFrame(axis, vec3(1.0, 0.0, 0.0), e1, e2);
    float sLat = clamp(dot(n, axis), -1.0, 1.0);
    float lon  = atan(dot(n, e2), dot(n, e1)) + et * 0.03 + seed * TAU;
    vec2 uv = vec2(lon / TAU,
                   clamp(0.5 + (asin(sLat) / PI) * 0.85 + (seed - 0.5) * 0.30,
                         0.004, 0.996));
    return texture(u_rockmap, vec2(wrapU(uv.x, MAP_W), uv.y)).rgb * tint;
}

vec3 diskBodyLight(vec3 wp, vec3 n);

vec3 shadeBinary(vec3 n, vec3 rd, vec3 wp, vec3 pS1, vec3 pS2, vec3 albT,
                 float env, float et, float isGasGiant, float bass,
                 float sh1, float sh2) {
    vec3 dL1 = pS1 - wp;
    float d2_1 = dot(dL1, dL1);
    vec3 L1 = dL1 * inversesqrt(d2_1);
    float att1 = (7.6 + bass * 4.6) / (1.0 + 2.2 * d2_1) * sh1;
    float ndl1 = clamp((dot(n, L1) + 0.08) / 1.08, 0.0, 1.0);

    vec3 dL2 = pS2 - wp;
    float d2_2 = dot(dL2, dL2);
    vec3 L2 = dL2 * inversesqrt(d2_2);
    float att2 = 2.8 / (1.0 + 6.5 * d2_2) * sh2;
    float ndl2 = clamp((dot(n, L2) + 0.18) / 1.18, 0.0, 1.0);

    vec3 col1 = albT * ndl1 * att1 * STAR1_COL;
    vec3 h1 = normalize(L1 - rd);
    float spec1 = pow(max(0.0, dot(n, h1)), isGasGiant > 0.5 ? 11.0 : 20.0);
    float specK = isGasGiant > 0.5 ? 0.22 : 0.8;
    col1 += STAR1_COL * 1.12 * spec1 * att1 * specK;

    vec3 col2 = albT * ndl2 * att2 * STAR2_COL;
    vec3 col = col1 + col2;

    float rim = pow(1.0 - clamp(dot(-rd, n), 0.0, 1.0), 3.2);
    col += (STAR1_COL * att1 + STAR2_COL * att2) * rim * 0.60;

    if (isGasGiant > 0.5) {
        float gasRim = pow(1.0 - clamp(dot(-rd, n), 0.0, 1.0), 4.5);
        vec3 haze   = normalize(albT + 1e-4);
        vec3 gasCol = haze * mix(0.55, 1.45, ndl1) * STAR1_COL * att1
                    + haze * STAR2_COL * att2 * 1.6;
        col += gasCol * gasRim * 0.70;
    }

    col += albT * diskBodyLight(wp, n);

    vec3 ambLight = mix(vec3(0.02, 0.035, 0.06), vec3(0.07, 0.015, 0.005),
                        clamp(att2 / (att1 + att2 + 1e-5), 0.0, 1.0));
    col += albT * (ambLight + env * vec3(0.04, 0.05, 0.08) * (0.15 + 0.85 * rim));
    return col;
}

vec3 ringNormalDGA(int k) {
    vec3 u, v, n;
    gearBasis(float(GEAR_DGA), u, v, n);
    if (k == 1) n = rotAxis(n, normalize(u), -1.25);
    return n;
}
float ringDensity(float r, float R, int k) {
    float rOut = (k == 2) ? RING_OUT_WIDE : RING_OUT;
    float u = (r / R - RING_IN) / (rOut - RING_IN);
    if (u < 0.0 || u > 1.0) return 0.0;
    float d = texture(u_ringprof, vec2(clamp(u, 0.002, 0.998), 0.5)).r;
    d *= (k == 2) ? RING_DENS_K2 : 1.0;
    return clamp(d * smoothstep(0.0, 0.05, u) * smoothstep(1.0, 0.92, u), 0.0, 1.0);
}

vec3 getPulsarAxis() {
    vec3 u, v, n;
    gearBasis(float(GEAR_BINARY), u, v, n);
    return n;
}

#define JET_BACK 0.6
vec3 calcPulsarJets(vec3 ro, vec3 rd, vec3 pPulsar, vec3 pAxis,
                    float bass, float treble, float minT, float et) {
    vec3 w0 = ro - pPulsar;
    float dAlong = dot(w0, pAxis);
    float rdAlong = dot(rd, pAxis);
    vec3 rdPerp = rd - pAxis * rdAlong;

    float a = dot(rdPerp, rdPerp);
    float b = dot(w0 - pAxis * dAlong, rdPerp);
    float tClosest = clamp(-b / max(a, 1e-5), -JET_BACK, minT);

    vec3 pAt = ro + rd * tClosest - pPulsar;
    float zProj = dot(pAt, pAxis);
    float rDist = length(pAt - pAxis * zProj);
    float zDist = abs(zProj);

    float lengthFade = smoothstep(0.58, 0.075, zDist) * smoothstep(0.012, 0.048, zDist);
    if (lengthFade <= 0.0) return vec3(0.0);

    float coneR = 0.009 + 0.062 * zDist;
    float beam = exp(-pow(rDist / coneR, 2.0)) * lengthFade;

    float audioPower = pow(clamp(bass * 1.25 + treble * 0.45, 0.0, 1.6), 1.9);
    if (audioPower < 0.02) return vec3(0.0);

    float waves = 0.85 + 0.15 * sin(zDist * 60.0 - et * 8.0);
    vec3 jetCol = mix(vec3(0.12, 0.45, 1.0), vec3(0.75, 0.92, 1.0), exp(-rDist * 70.0));
    return jetCol * beam * waves * audioPower * 1.9;
}

float amp(float i) {
    if (i < 0.0 || i > N_BANDS - 1.0) return 0.0;
    return texelFetch(bands, ivec2(int(i), 0), 0).r;
}

#define GEO_N_B      4096.0
#define GEO_N_S      64.0
#define GEO_CAM_R    42.657458
#define GEO_B_CRIT   2.598076
#define GEO_B_MAX    43.162108
#define GEO_R_HORIZ  1.000000

#define GEO_KERR 1
uniform sampler2DArray u_kerrpaths;
ivec2 gPix = ivec2(0);
#undef  GEO_N_S
#define GEO_N_S      32.0
float kerrW(int layer) { return texelFetch(u_kerrpaths, ivec3(gPix, layer), 0).w; }

vec3 gRo = vec3(0.0);
vec3 gRd = vec3(0.0, 0.0, 1.0);

float geoImpactParam(vec3 ro, vec3 rd) {
    float rWorld = length(ro);
    float R = rWorld / BH_RS;
    vec3  er = ro / max(rWorld, 1e-9);
    if (dot(rd, er) > 0.0) return -1.0;
    float st = length(cross(er, rd));
    return R * st * inversesqrt(max(1e-6, 1.0 - GEO_R_HORIZ / R));
}

float geoRow(float b) {
    float t;
    if (b < GEO_B_CRIT) {
        float s = 1.0 - b / GEO_B_CRIT;
        t = 0.5 * (1.0 - pow(max(s, 0.0), 1.0 / 3.0));
    } else {
        float s = (b - GEO_B_CRIT) / (GEO_B_MAX - GEO_B_CRIT);
        t = 0.5 * (1.0 + pow(clamp(s, 0.0, 1.0), 1.0 / 3.0));
    }
    return clamp(t * (GEO_N_B - 1.0), 0.0, GEO_N_B - 1.0);
}

void geoFrame(vec3 ro, vec3 rd, out vec3 er, out vec3 et) {
    er = normalize(ro);
    vec3 tang = rd - er * dot(rd, er);
    float L = length(tang);
    et = (L > 1e-7) ? tang / L : normalize(cross(er, vec3(0.0, 1.0, 0.0)) + vec3(1e-7));
}

vec3  diskNormal();
vec3  skyColorDispersed(vec3 rd, vec3 bend, float treble, float bImpact);
float kerrDragScale();
vec3 geoPoint(float rowf, int i, vec3 er, vec3 et) {
    return texelFetch(u_kerrpaths, ivec3(gPix, i), 0).xyz * BH_RS;
}

vec3 geoExitDir(float row, vec3 er, vec3 et) {
    return normalize(vec3(kerrW(2), kerrW(3), kerrW(4)));
    vec3 a = geoPoint(row, int(GEO_N_S) - 2, er, et);
    vec3 b = geoPoint(row, int(GEO_N_S) - 1, er, et);
    vec3 d = b - a;
    float L = length(d);
    return (L > 1e-9) ? d / L : normalize(b);
}

bool geoCaptured(float b) {
    return kerrW(1) > 0.5;
}

bool geoSegSphere(vec3 A, vec3 B, vec3 c, float rad, out float tHit, out vec3 hitPos) {
    vec3  d  = B - A;
    vec3  m  = A - c;
    float aa = dot(d, d);
    if (aa < 1e-18) return false;
    float bb = dot(m, d);
    float cc = dot(m, m) - rad * rad;
    if (cc > 0.0 && bb > 0.0) return false;
    float disc = bb * bb - aa * cc;
    if (disc < 0.0) return false;
    float t = (-bb - sqrt(disc)) / aa;
    if (t > 1.0) return false;
    tHit   = max(t, 0.0);
    hitPos = A + d * tHit;
    return true;
}

float geoSegPointD2(vec3 A, vec3 B, vec3 p) {
    vec3  d = B - A;
    vec3  m = p - A;
    float t = clamp(dot(m, d) / max(dot(d, d), 1e-18), 0.0, 1.0);
    vec3  q = A + d * t - p;
    return dot(q, q);
}

float meteorTimeAtNu(vec4 el0, float nu) {
    float tPeri = el0.x, n = max(el0.y, 1e-4), e = el0.w;
    float th2   = tan(0.5 * nu);
    float M;
    if (e < 1.0) {
        float E = 2.0 * atan(sqrt(max((1.0 - e) / (1.0 + e), 0.0)) * th2);
        M = E - e * sin(E);
    } else {
        float th = sqrt(max((e - 1.0) / (e + 1.0), 0.0)) * th2;
        if (abs(th) >= 0.999) return -1e9;
        float H = 2.0 * atanh(th);
        M = e * sinh(H) - H;
    }
    return tPeri + M / n;
}

float meteorTrailDensity(vec4 el0, vec4 el1, vec4 el2, vec3 p,
                         float tNow, float tIn) {
    vec3  u = el1.xyz;
    vec3  n = el2.xyz;
    vec3  v = cross(n, u);

    float h = dot(p, n);
    if (abs(h) > MET_TRAIL_RW * 3.0) return 0.0;

    float x  = dot(p, u), y = dot(p, v);
    float rr = sqrt(x * x + y * y);
    float nu = atan(y, x);

    float e  = el0.w;
    float pl = el0.z * (1.0 + e);
    float den = 1.0 + e * cos(nu);
    if (den <= 1e-3) return 0.0;
    float dr = rr - pl / den;

    float tAt = meteorTimeAtNu(el0, nu);
    if (tAt < -1e8) return 0.0;
    if (e < 1.0) {
        float T = 6.28318531 / max(el0.y, 1e-4);
        tAt += floor((tNow - tAt) / T) * T;
    }
    if (tAt < tIn) return 0.0;

    float age = tNow - tAt;
    if (age < 0.0 || age > MET_TAIL_T) return 0.0;

    float w   = 1.0 - age / MET_TAIL_T;
    float rad = MET_TRAIL_R0 + MET_TRAIL_RW * (1.0 - w);
    return exp(-(dr * dr + h * h) / (rad * rad)) * w;
}

#define KERR_A        0.997
#define KERR_W        12.0
uniform float u_bass_accum;
#define DISK_SPIN_IDLE 0.6
#define DISK_SPIN_BASS 8.0
float diskClock() { return iTime * DISK_SPIN_IDLE + u_bass_accum * DISK_SPIN_BASS; }
#define PRING_THICK   0.003
#define PRING_ALIGN   0.98
#define PRING_GAIN    4.0
#define KERR_DRAG     1.0

float kerrIscoM(float a) {
    float z1 = 1.0 + pow(1.0 - a * a, 1.0 / 3.0)
                   * (pow(1.0 + a, 1.0 / 3.0) + pow(max(1.0 - a, 0.0), 1.0 / 3.0));
    float z2 = sqrt(3.0 * a * a + z1 * z1);
    return 3.0 + z2 - sqrt(max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0));
}

#define DISK_R_IN     (kerrIscoM(KERR_A) * 0.5 * BH_RS)
#define DISK_R_OUT    0.350
#define DISK_T_IN     11.0
#define DISK_INNER_FLOOR 0.08
#define DISK_GAIN     1.60
#define DISK_TILT     vec3(0.34, 1.0, 0.13)

#define DISKLIGHT_GAIN    0.90
#define DISKLIGHT_FALLOFF 0.80
#define LENSGLOW_GAIN     0.60
#define LENSGLOW_FALLOFF  0.45

vec3 diskNormal() { return normalize(DISK_TILT); }

float kerrDragScale() {
    return 0.0;
    return KERR_A * KERR_DRAG;
}

float kerrSpinAlign(vec3 er, vec3 et) { return clamp(dot(cross(et, er), diskNormal()), -1.0, 1.0); }

#define KERR_DFLAT    0.70
#define KERR_DROUND   1.2
#define KERR_DBULGE   0.35

float kerrDScale(vec3 er, vec3 et) {
    float s = kerrSpinAlign(er, et);
    float q    = s / KERR_DFLAT;
    float hq   = max(KERR_DROUND - abs(q - 1.0), 0.0) / KERR_DROUND;
    float up   = max(q, 1.0) + hq * hq * KERR_DROUND * 0.25;
    float down = 1.0 / (1.0 + KERR_DBULGE * KERR_A * max(-s, 0.0) * max(-s, 0.0));
    return mix(1.0, up * down, KERR_A);
}

float kerrCritB(vec3 er, vec3 et) {
    return GEO_B_CRIT / kerrDScale(er, et);
}
float kerrWarpB(float b, vec3 er, vec3 et) {
    return b;
    if (b < 0.0) return b;
    float f = exp(-max(b - 2.0 * GEO_B_CRIT, 0.0) / KERR_W);
    return b * mix(1.0, kerrDScale(er, et), f);
}

vec3 kelvinToRgb(float k) {
    k = clamp(k, 1000.0, 40000.0) * 0.01;
    float r = (k <= 66.0) ? 255.0
                          : 329.698727446 * pow(max(k - 60.0, 1e-3), -0.1332047592);
    float g = (k <= 66.0) ? 99.4708025861 * log(max(k, 1.0)) - 161.1195681661
                          : 288.1221695283 * pow(max(k - 60.0, 1e-3), -0.0755148492);
    float b = (k >= 66.0) ? 255.0
            : (k <= 19.0) ? 0.0
                          : 138.5177312231 * log(max(k - 10.0, 1e-3)) - 305.0447927307;
    return clamp(vec3(r, g, b) / 255.0, 0.0, 1.0);
}

#define DISK_THIN     1
#define DISK_UNDER_FLIP 1
#define DISK_D_SPAN   2.1
#define DISK_TINT     0.85
#define DISK_SAT      1.2
#define DISK_SHIFT    0.7
#define DISK_R_IN_VIS 1.5
#define DISK_BEAM_POW 1.5
#define DISK_EXPOSE   1.6
#define DISK_WHITE    6.0
vec3 diskToneSum(vec3 x) {
    float l = max(max(x.r, x.g), x.b) * DISK_EXPOSE;
    if (l <= 0.0) return vec3(0.0);
    float m = l * (1.0 + l / (DISK_WHITE * DISK_WHITE)) / (1.0 + l);
    return x * (DISK_EXPOSE * m / l);
}
#define DISK_COVER    0.0
#define DISK_GLOW     1.0
float gDiskTr = 1.0;
#define DISK_SHOW     1
uniform sampler2D u_kntable;
#define KN_RIN        0.572600
#define KN_ROUT       7.274607
#define DSIM_TLOOP    33.3932
#define DSIM_RIN      0.572600
#define DSIM_ROUT     7.274607
#define DISK_HR       0.1
#define DISK_TAU0     0.3
#define DISK_LOOP_S   10.0
#define DISK_SIM_GAIN 8.0
#define DISK_TONE_K   5.0

vec4 knTable(float rR) {
    float v = (log(rR) - log(KN_RIN)) / (log(KN_ROUT) - log(KN_RIN));
    return texture(u_kntable, vec2(clamp(v, 0.0, 1.0), 0.5));
}

#define STREAK_P      (DSIM_TLOOP * 0.25)
float sHash3(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}
float sNoise3(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(sHash3(i),                   sHash3(i + vec3(1, 0, 0)), f.x),
                   mix(sHash3(i + vec3(0, 1, 0)),   sHash3(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(sHash3(i + vec3(0, 0, 1)),   sHash3(i + vec3(1, 0, 1)), f.x),
                   mix(sHash3(i + vec3(0, 1, 1)),   sHash3(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float streakField(float rR, float phs, float seed) {
    float lr = log(rR / DSIM_RIN);
    vec3  q  = vec3(cos(phs), sin(phs), 0.0) * 1.6 + vec3(seed, seed * 0.7, lr * 5.0);
    float n = 0.0, amp = 0.5, tot = 0.0;
    for (int o = 0; o < 4; o++) {
        n += amp * sNoise3(q); tot += amp;
        q = q * vec3(2.03, 2.03, 2.11) + vec3(1.7, 9.2, 3.1);
        amp *= 0.55;
    }
    n /= tot;
    return mix(0.35, 1.75, smoothstep(0.2, 0.85, n));
}
float diskStreaks(float rR, float phi) {
    float tS = diskClock() / DISK_LOOP_S * DSIM_TLOOP;
    float p  = tS / STREAK_P;
    float Om = knTable(rR).x;
    float pa = fract(p), pb = fract(p + 0.5);
    float sa = fract(floor(p) * 0.6180339) * 23.0;
    float sb = fract(floor(p + 0.5) * 0.6180339 + 0.5) * 23.0;
    float wa = 1.0 - abs(2.0 * pa - 1.0), wb = 1.0 - abs(2.0 * pb - 1.0);
    float ca = streakField(rR, phi - Om * pa * STREAK_P, sa) - 1.0;
    float cb = streakField(rR, phi - Om * pb * STREAK_P, sb) - 1.0;
    return max(1.0 + (wa * ca + wb * cb) / sqrt(max(wa * wa + wb * wb, 1e-4)), 0.0);
}

vec4 diskShareShade(float rR, float phi, float colShare, float lam, float bassLvl) {
    if (rR <= 0.0 || colShare <= 0.0) return vec4(0.0, 0.0, 0.0, 1.0);
    float c   = diskStreaks(rR, phi);
    c *= 1.0 - exp(-2.5 * max(DSIM_ROUT - rR, 0.0) / (0.35 * DSIM_ROUT));
    c *= smoothstep(DISK_R_IN_VIS, DISK_R_IN_VIS * 1.02, rR);
    float tau = DISK_TAU0 * c * colShare / (2.50663 * DISK_HR * rR);
    float tr  = exp(-tau);
    vec4  tab = knTable(rR);
    float g   = 1.0 / max(tab.y * (1.0 - tab.x * lam), 1e-6);
    float F   = tab.z;
    float boost = pow(g, DISK_BEAM_POW);
    vec3 hue = vec3(1.0);
    {
        float D = 1.0 / max(1.0 - tab.x * lam, 1e-3);
        float t = clamp(log(D) / log(DISK_D_SPAN), -1.0, 1.0);
        vec3  tint = (t > 0.0) ? vec3(0.42, 0.64, 1.00) : vec3(1.00, 0.42, 0.22);
        tint /= dot(tint, vec3(0.299, 0.587, 0.114));
        hue = mix(hue, tint, abs(t) * DISK_TINT);
        float hl = dot(hue, vec3(0.299, 0.587, 0.114));
        hue = max(mix(vec3(hl), hue, DISK_SAT), 0.0);
    }
    vec3 src = hue * boost * F
             * DISK_GAIN * DISK_SIM_GAIN * (0.70 + 0.90 * bassLvl);
    if (DISK_TONE_K > 0.0 && DISK_THIN == 0) {
        float lum = max(max(src.r, src.g), src.b);
        if (lum > 0.0) src *= (log(1.0 + DISK_TONE_K * lum) / log(1.0 + DISK_TONE_K)) / lum;
    }
    return vec4(src * (1.0 - tr), tr);
}

vec3 diskBodyLight(vec3 wp, vec3 n) {
    vec3 nd = diskNormal();

    vec3  wperp = wp - dot(wp, nd) * nd;
    float rho   = length(wperp);
    vec3  rdir  = (rho > 1e-5) ? wperp / rho : vec3(0.0);
    float rNear = clamp(rho, DISK_R_IN, DISK_R_OUT);
    vec3  Xd    = rdir * rNear;
    vec3  dLd   = Xd - wp;
    float dd2   = dot(dLd, dLd) + 1e-6;
    vec3  Ld    = dLd * inversesqrt(dd2);
    float ndld  = clamp((dot(n, Ld) + 0.10) / 1.10, 0.0, 1.0);
    float x     = DISK_R_IN / max(rNear, 1e-4);
    float Td    = DISK_T_IN * pow(x, 0.75)
                * pow(max(1.0 - sqrt(x), DISK_INNER_FLOOR), 0.25);
    vec3  cd    = kelvinToRgb(Td * 1000.0);
    vec3  term1 = cd * ndld * (DISKLIGHT_GAIN / (1.0 + DISKLIGHT_FALLOFF * dd2));

    float rr    = max(length(wp), 1e-4);
    vec3  Lh    = -wp / rr;
    float ndlh  = clamp((dot(n, Lh) + 0.12) / 1.12, 0.0, 1.0);
    vec3  ch    = kelvinToRgb(DISK_T_IN * 1.05 * 1000.0);
    vec3  term2 = ch * ndlh * (LENSGLOW_GAIN / (1.0 + LENSGLOW_FALLOFF * rr * rr));

    return term1 + term2;
}

#define STAR_TILT_0   1.48353
#define STAR_TILT_1   1.48353
#define STAR_NODE_0   0.00
#define STAR_NODE_1   0.00

vec3 starPlaneNormal(int star) {
    vec3  nd   = diskNormal();
    float tilt = (star == 1) ? STAR_TILT_1 : STAR_TILT_0;
    float node = (star == 1) ? STAR_NODE_1 : STAR_NODE_0;
    vec3  a0   = normalize(cross(nd, vec3(0.0, 0.0, 1.0)) + vec3(1e-5));
    a0 = rotAxis(a0, nd, node);
    return normalize(rotAxis(nd, a0, tilt));
}

#define SWING_CAM_AZ  (-3.103562)
#define SWING_CAM_EL  (0.087266)

vec3 swingU() {
    vec3 nd = diskNormal();
    vec3 e1 = normalize(cross(nd, vec3(0.0, 0.0, 1.0)) + vec3(1e-5));
    vec3 e2 = cross(nd, e1);
    vec3 cd = cos(SWING_CAM_EL) * (cos(SWING_CAM_AZ) * e1 + sin(SWING_CAM_AZ) * e2)
            + sin(SWING_CAM_EL) * nd;
    return normalize(cd);
}

#define TOUCH_LAP         16.0
#define TOUCH_CYCLE_LAPS  8.0
#define TOUCH_SUB         1.0
#define TOUCH_R0          0.523599
#define TOUCH_JUMP        2.094395
#define TOUCH_CREEP       1.047198
#define TOUCH_WHIP_S0     1.28125
#define TOUCH_WHIP_L0     2.0
#define TOUCH_WHIP_S1     1.0
#define TOUCH_WHIP_L1     1.35
#define TOUCH_NSAMP       128

vec3 touchCamDir() { return swingU(); }

struct TOrb { vec3 u; vec3 v; vec3 n; float a; float e; float R; float ring; };

TOrb starTOrb(int star) {
    vec3  n  = starPlaneNormal(star);
    vec3  a0 = normalize(cross(n, vec3(0.0, 0.0, 1.0)) + vec3(1e-5));
    vec3  b0 = cross(n, a0);
    float pr = (star == 1) ? PERI_1 : PERI_0;
    vec3  uP = a0 * cos(pr) + b0 * sin(pr);
    return TOrb(uP, cross(n, uP), n,
                (star == 1) ? BARY_R_1 : BARY_R_0,
                (star == 1) ? ECC_1 : ECC_0,
                (star == 1) ? DWARF_R : PULSAR_R, 0.0);
}

vec3 torbBase(TOrb o, float M) {
    float E = keplerE(M, o.e);
    return o.u * (o.a * (cos(E) - o.e)) + o.v * (o.a * sqrt(max(1.0 - o.e * o.e, 1e-4)) * sin(E));
}

float touchRotation(int star, float c) {
    float NL  = TOUCH_LAP * TOUCH_CYCLE_LAPS;
    float cyc = floor(c / NL);
    float qb  = c - cyc * NL;
    float ws  = (star == 1) ? TOUCH_WHIP_S1 : TOUCH_WHIP_S0;
    float wl  = (star == 1) ? TOUCH_WHIP_L1 : TOUCH_WHIP_L0;
    float x   = smoothstep(0.0, 1.0, clamp((qb - ws) / wl, 0.0, 1.0));
    float cr  = clamp((qb - ws - wl) / (NL - ws - wl), 0.0, 1.0);
    return TOUCH_R0 + PI * cyc + TOUCH_JUMP * x + TOUCH_CREEP * cr;
}

float touchMargin(TOrb o, float M, float rotA) {
    vec3  cd  = touchCamDir();
    vec3  cam = GEO_CAM_R * BH_RS * cd;
    vec3  p   = rotAxis(torbBase(o, M), o.n, rotA);
    if (dot(p, -cd) <= 0.0) return 1e3;
    vec3  d   = p - cam;
    float dl  = length(d);
    float ang = degrees(acos(clamp(dot(d / dl, -cd), -1.0, 1.0)));
    if (o.ring > 0.5) return ang - degrees(atan(o.R / dl));
    vec3  rdb = d / dl;
    vec3  tg  = rdb - cd * dot(rdb, cd);
    float bk  = (length(tg) > 1e-7) ? kerrCritB(cd, normalize(tg)) : GEO_B_CRIT;
    float sh  = degrees(asin(clamp(bk / GEO_CAM_R * sqrt(1.0 - 1.0 / GEO_CAM_R), 0.0, 1.0)));
    return ang - sh - degrees(atan(o.R / dl));
}

vec2 touchSolve(TOrb o, float rotA) {
    float stp = TAU / float(TOUCH_NSAMP);
    float bestM = 0.0, bestV = 1e9;
    for (int i = 0; i < TOUCH_NSAMP; i++) {
        float M = -PI + float(i) * stp;
        float v = touchMargin(o, M, rotA);
        if (v < bestV) { bestV = v; bestM = M; }
    }
    if (bestV >= 0.0) return vec2(bestM, -1.0);
    float lo = bestM, hi = bestM;
    for (int i = 0; i < TOUCH_NSAMP; i++) {
        if (touchMargin(o, lo - stp, rotA) >= 0.0) break;
        lo -= stp;
    }
    for (int i = 0; i < TOUCH_NSAMP; i++) {
        if (touchMargin(o, hi + stp, rotA) >= 0.0) break;
        hi += stp;
    }
    float a = lo, ao = lo - stp, b = hi, bo = hi + stp;
    for (int i = 0; i < 14; i++) {
        float m = 0.5 * (a + ao);
        if (touchMargin(o, m, rotA) < 0.0) a = m; else ao = m;
        m = 0.5 * (b + bo);
        if (touchMargin(o, m, rotA) < 0.0) b = m; else bo = m;
    }
    return vec2(0.5 * (a + b), 0.5 * (b - a));
}

float touchAnomaly(float c, vec2 t, vec2 sa, vec2 sb, float sub) {
    float D  = t.y - t.x;
    float ph = (c - t.x) / D;
    float g  = PI * sub / D;
    float Aa = sa.y < 0.0 ? 0.0 : clamp((sa.y - g) / sin(g), -0.99, 0.99);
    float Ab = sb.y < 0.0 ? 0.0 : clamp((sb.y - g) / sin(g), -0.99, 0.99);
    float cb = sa.x + (mod(sb.x - sa.x + PI, TAU) - PI);
    float w  = smoothstep(0.3, 0.7, ph);
    return mix(sa.x, cb, w) + TAU * ph + mix(Aa, Ab, w) * sin(TAU * ph);
}

vec2 touchBracket(int star, float c) {
    float L = TOUCH_LAP, NL = TOUCH_LAP * TOUCH_CYCLE_LAPS, h = 0.5 * TOUCH_SUB;
    float cyc = floor(c / NL);
    float q   = c - cyc * NL;
    vec2 t;
    if (star == 0) {
        if (q < h) t = vec2((TOUCH_CYCLE_LAPS - 1.0) * L + h - NL, h);
        else { float i = floor((q - h) / L); t = vec2(i * L + h, i * L + h + L); }
    } else {
        float first = 1.5 * L + h, last = (TOUCH_CYCLE_LAPS - 1.5) * L + h;
        if (q < h)            t = vec2(last - NL, h);
        else if (q < first)   t = vec2(h, first);
        else if (q >= last)   t = vec2(last, NL + h);
        else { float i = floor((q - h) / L - 0.5); t = vec2((i + 0.5) * L + h, (i + 1.5) * L + h); }
    }
    return t + cyc * NL;
}

vec3 starTouchPos(int star) {
    float c = tempoBeatCount();
    vec2  t = touchBracket(star, c);
    TOrb  o = starTOrb(star);
    float M = touchAnomaly(c, t, touchSolve(o, touchRotation(star, t.x)),
                                 touchSolve(o, touchRotation(star, t.y)), TOUCH_SUB);
    return rotAxis(torbBase(o, M), o.n, touchRotation(star, c));
}

#define PT_RP_PG     0.98
#define PT_ECC_PG    0.75
#define PT_PSI_PG    0.523599
#define PT_SUB_PG    1.0
#define PT_ARR_PG    4.0
#define PT_RP_DGA    0.82
#define PT_ECC_DGA   0.80
#define PT_PSI_DGA   -0.523599
#define PT_SUB_DGA   1.0
#define PT_ARR_DGA   6.0
#define PT_ALPHA_PG  -0.0872665
#define PT_BETA_PG    0.1211818
#define PT_ALPHA_DGA  0.0872665
#define PT_BETA_DGA   0.0410501

#define ORBIT_BEATS_GG3 12.0
#define PT_RP_GG3    1.10
#define PT_ECC_GG3   0.74
#define PT_PSI_GG3   0.0
#define PT_SUB_GG3   1.0
#define PT_ARR_GG3   9.0
#define PT_ALPHA_GG3 0.0
#define PT_BETA_GG3  -0.07
#define GG3_R        0.075
#define GG3_HUE      0.52
#define GG3_TINT     vec3(0.80, 0.98, 1.25)
#define GG3_BANDS    0.70

#define PT_HUEPH(arr, sub, lap) (-((arr) + 0.5 * (sub) + 0.5 * (lap)) / (lap))

TOrb planetTOrb(float rp, float e, float psi, float R, float alpha, float beta) {
    vec3 nd  = diskNormal();
    vec3 cd  = touchCamDir();
    vec3 beh = normalize(-cd - nd * dot(-cd, nd));
    vec3 w   = normalize(cross(nd, beh));
    vec3 u   = rotAxis(beh, nd, psi);
    vec3 v   = cross(nd, u);
    vec3 n   = nd;
    u = rotAxis(u, beh, alpha); v = rotAxis(v, beh, alpha); n = rotAxis(n, beh, alpha);
    u = rotAxis(u, w, beta);    v = rotAxis(v, w, beta);    n = rotAxis(n, w, beta);
    return TOrb(u, v, n, rp / (1.0 - e), e, R, 1.0);
}

vec3 planetTouchPos(float rp, float e, float psi, float R, float alpha, float beta,
                    float lap, float sub, float arr) {
    float c  = tempoBeatCount();
    float t0 = arr + 0.5 * sub;
    float k  = floor((c - t0) / lap);
    vec2  t  = vec2(t0 + k * lap, t0 + (k + 1.0) * lap);
    TOrb  o  = planetTOrb(rp, e, psi, R, alpha, beta);
    vec2  s  = touchSolve(o, 0.0);
    return torbBase(o, touchAnomaly(c, t, s, s, sub));
}

vec3 starOrbitPos(int star) {
    return starTouchPos(star);
}
