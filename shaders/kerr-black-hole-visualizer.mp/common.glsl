// Kerr-Black-Hole-Visualizer: common code, prepended to every pass.
//
// Constants, uniforms and shared functions: beat clock, orbital mechanics
// (beat-locked Kepler orbits, the horseshoe swap), star/sky sampling (Gaia DR3
// point stars as a BC6H cube map, photo nebula), body surfaces and the
// Kerr-Newman geodesic walk (precomputed light paths in u_kerrpaths, walked as
// a polyline).
// KN_DISK: the accretion disk is the volumetric disk of the NPGS Kerr-Newman
// renderer (github.com/baopinshui/NPGS, GPL-3.0, its DiskColor), marched on
// the precomputed paths. It only exists around a star's close pass: it winds
// out of the star's side, is full at periapsis and is eaten outside-in as the
// star leaves. Both stars feed it (the red dwarf's disk weaker and warmer); at
// the swap, when both pass within a few beats, they feed one disk.
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
uniform sampler2D u_disklive;
uniform float u_disk_x0, u_disk_dx, u_disk_dphi, u_disk_rin, u_disk_rout;
uniform samplerCube u_skycube;
uniform sampler2D u_nebula;
#define NEB_W       16383.0
#define NEB_GAIN    1.0
#define NEB_SAT     1.0
#define NEB_FLOOR   0.0
#define NEB_TINT    vec3(1.0, 1.0, 1.0)
#define NEB_BASS_DRIVE 0.3

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
#define SKY_AUDIO     0.0
#define SKY_DRIFT     0.0
#define SKY_ROT_SENS  0.0
#define SKY_PITCH     1.267
#define SKY_YAW       0.60
#define SKY_YAW_SENS  0.10
#define SKY_ROLL_SENS 0.10
#define SKY_START_X    -194.749266
#define SKY_START_Z    -620.453492
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

#define SHEET_DEPTH  0.0
#define SHEET_W      0.50
vec3  gHoleDir = vec3(0.0, 0.0, 1.0);
vec3  gSheetDir = vec3(0.0, 0.0, 1.0);
vec3  gCamR = vec3(1.0, 0.0, 0.0);
vec3  gCamU = vec3(0.0, 1.0, 0.0);

vec3 skyWarp(vec3 d) {
    float push = SHEET_DEPTH * clamp(gBass, 0.0, 1.0);
    if (push < 1e-5) return d;
    float c = clamp(dot(d, gSheetDir), -1.0, 1.0);
    vec3 perp = d - c * gSheetDir;
    float s = length(perp);
    if (s < 1e-5) return d;
    float th = acos(c);
    th += push * th * exp(-(th * th) / (SHEET_W * SHEET_W));
    return cos(th) * gSheetDir + sin(th) * (perp / s);
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

uniform float u_rms_accum;
#define SKY_PARK      1
#define SKY_WOBBLE_R  0.035
#define SKY_FLOW_RATE 0.025
vec3 skyRotate(vec3 dir) {
    vec3 d;
    if (SKY_PARK == 1) {
        float ph = mod(SKY_FLOW_RATE * u_rms_accum / SKY_WOBBLE_R, TAU);
        d = rotAxis(dir, normalize(cos(ph) * gCamU + sin(ph) * gCamR), SKY_WOBBLE_R);
    } else {
        d = rotAxis(dir, gCamU, mod(SKY_FLOW_RATE * u_rms_accum, TAU));
    }
    {
        float gx = (u_cam_move.x + SKY_START_X - SKY_DRAG_R0) * SKY_DRAG_SENS * SKY_DRAG_SX;
        float gy = (u_cam_move.z + SKY_START_Z - SKY_DRAG_F0) * SKY_DRAG_SENS * SKY_DRAG_SY;
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
    return d;
}
vec3 skyDirFinal(vec3 dir) {
    return skyWarp(skyRotate(dir));
}
vec2 skyDirUV(vec3 dir) {
    vec3 d = skyDirFinal(dir);
    return vec2(atan(d.z, d.x) / TAU + 0.5,
                1.0 - acos(clamp(d.y, -1.0, 1.0)) / PI);
}

#define NEB_SHOW 1
vec3 nebulaAt(vec3 dir, float bass) {
    if (NEB_SHOW == 0) return vec3(0.0);
    vec2 nuv = skyDirUV(dir);
    vec3 neb = texture(u_nebula, vec2(wrapU(nuv.x, NEB_W), nuv.y)).rgb;
    float nl = dot(neb, vec3(0.299, 0.587, 0.114));
    neb = mix(vec3(nl), neb, NEB_SAT) * NEB_TINT;
    float g = NEB_GAIN * (1.0 + NEB_BASS_DRIVE * clamp(bass, 0.0, 1.0));
    return max(neb - NEB_FLOOR, 0.0) * g;
}
#define NEB_DUST 1.0
float nebulaExt(vec3 dir) {
    if (NEB_SHOW == 0) return 0.0;
    vec2 nuv = skyDirUV(dir);
    return clamp(texture(u_nebula, vec2(wrapU(nuv.x, NEB_W), nuv.y)).a * NEB_DUST, 0.0, 1.0);
}

#define SKY64_GAIN     8.0
#define SKY64_SHARP    0.5
#define SKY64_MAXGRAD  0.026
#define SKY_CLUSTER_FOOT 1
#define SKY_PROCEDURAL 0
#define PROC_DENSITY   30.0
#define PROC_SLOPE     5.0
#define PROC_GAIN      0.6
#define PROC_MINRAD    1.6e-4
vec3 kelvinToRgb(float k);
uvec3 procPcg(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}
vec3 procHash(ivec2 c, int f, int salt) {
    return vec3(procPcg(uvec3(uint(c.x + 65536), uint(c.y + 65536), uint(f * 131 + salt)))) * (1.0 / 4294967295.0);
}
vec3 procFaceDir(int f, vec2 q) {
    if (f == 0) return vec3( 1.0, q.y, q.x);
    if (f == 1) return vec3(-1.0, q.y, q.x);
    if (f == 2) return vec3(q.x,  1.0, q.y);
    if (f == 3) return vec3(q.x, -1.0, q.y);
    if (f == 4) return vec3(q.x, q.y,  1.0);
    return vec3(q.x, q.y, -1.0);
}
vec3 procStars(vec3 d, float rad) {
    vec3 a = abs(d); int f; vec2 q;
    if (a.x >= a.y && a.x >= a.z) { f = d.x > 0.0 ? 0 : 1; q = vec2(d.z, d.y) / a.x; }
    else if (a.y >= a.z)          { f = d.y > 0.0 ? 2 : 3; q = vec2(d.x, d.z) / a.y; }
    else                          { f = d.z > 0.0 ? 4 : 5; q = vec2(d.x, d.y) / a.z; }
    vec2 e = atan(q) * (4.0 / PI);
    float n = floor(90.0 * sqrt(PROC_DENSITY));
    vec2 g = (e * 0.5 + 0.5) * n;
    ivec2 gi = ivec2(floor(g));
    float r = max(rad, PROC_MINRAD);
    vec3 col = vec3(0.0);
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        ivec2 cI = gi + ivec2(dx, dy);
        if (cI.x < 0 || cI.y < 0 || cI.x >= int(n) || cI.y >= int(n)) continue;
        vec3 h = procHash(cI, f, 7);
        vec2 se = (vec2(cI) + h.xy) / n * 2.0 - 1.0;
        vec3 sd = normalize(procFaceDir(f, tan(se * (PI / 4.0))));
        float dist = length(d - sd);
        if (dist > r * 1.3) continue;
        float b = pow(h.z, PROC_SLOPE);
        vec3 h2 = procHash(cI, f, 11);
        vec3 tint = kelvinToRgb(mix(3200.0, 11000.0, h2.x * h2.x));
        col += tint * b * (1.0 - smoothstep(r, r * 1.3, dist));
    }
    return col * PROC_GAIN;
}
vec3 skyStarRead(vec3 dd, float L, float texel) {
    if (SKY_PROCEDURAL == 1) return procStars(dd, 0.5 * texel * exp2(L));
    return textureLod(u_skycube, dd, L).rgb * SKY64_GAIN;
}
float gFootScale = 1.0;
#define SKY_MAXMIP_CUBE 1
vec3 skyColorG(vec3 dir, float treble, float bImpact, float gscale) {
    vec3 d = skyDirFinal(dir);
    vec3 gx = dFdx(d) * (SKY64_SHARP * gFootScale), gy = dFdy(d) * (SKY64_SHARP * gFootScale);
    float ga = dot(gx, gx), gb = dot(gx, gy), gc = dot(gy, gy);
    float gtr = 0.5 * (ga + gc);
    float gMajor = sqrt(gtr + sqrt(max(gtr * gtr - (ga * gc - gb * gb), 0.0)));
    if (gMajor > SKY64_MAXGRAD) { float k = SKY64_MAXGRAD / gMajor; gx *= k; gy *= k; gMajor = SKY64_MAXGRAD; }
    #define SKY_LOD0 1
    #define SKY_LOD_OPEN 2.0
    #define SKY_LOD_RING 0.5
    vec3 c;
    if (SKY_MAXMIP_CUBE == 1 && gscale < 2.0) {
        float texel = 1.5708 / 16384.0;
        float Lt = log2(max(gMajor / SKY64_SHARP, 1e-9) / texel);
        float bE = (bImpact > 1e-4) ? bImpact : 1e5;
        float prox = 1.0 - smoothstep(BH_CAPTURE * 1.05, BH_CAPTURE * 3.0, bE);
        float L = clamp(Lt, 0.0, SKY_LOD_OPEN + SKY_LOD_RING * prox);
        c = skyStarRead(d, L, texel);
        #define SKY_STREAK_MIX   1.0
        #define SKY_STREAK_GAIN  2.0
        #define SKY_STREAK_RATIO 8.0
        #define SKY_STREAK_TAPS  5
        if (prox > 0.001 && SKY_STREAK_MIX > 0.0) {
            vec3 jx = gx / SKY64_SHARP, jy = gy / SKY64_SHARP;
            float a = dot(jx, jx), b = dot(jx, jy), cc = dot(jy, jy);
            float tr = 0.5 * (a + cc), dt = sqrt(max(tr * tr - (a * cc - b * b), 0.0));
            float l1 = tr + dt;
            vec2 v1 = (abs(b) > 1e-20) ? normalize(vec2(b, l1 - a)) : ((a >= cc) ? vec2(1, 0) : vec2(0, 1));
            vec3 ax = jx * v1.x + jy * v1.y;
            float axl = length(ax);
            if (axl > 1e-12) {
                ax /= axl;
                float half_ = 0.5 * min(sqrt(l1), texel * exp2(L) * SKY_STREAK_RATIO);
                vec3 acc = vec3(0.0); float wt = 0.0;
                for (int i = -SKY_STREAK_TAPS; i <= SKY_STREAK_TAPS; i++) {
                    float f = float(i) / float(SKY_STREAK_TAPS);
                    float w = exp(-2.5 * f * f);
                    acc += w * skyStarRead(normalize(d + ax * (f * half_)), L, texel);
                    wt  += w;
                }
                vec3 cs = acc / wt * SKY_STREAK_GAIN;
                c = mix(c, cs, smoothstep(0.05, 0.9, prox) * SKY_STREAK_MIX);
            }
        }
        #define PROC_GLOW_POW  0.35
        #define PROC_GLOW_GAIN 1.0
        if (SKY_PROCEDURAL == 1) {
            vec3 fx = gx / SKY64_SHARP, fy = gy / SKY64_SHARP;
            float lam = PROC_DENSITY * 3282.8 * length(cross(fx, fy));
            float tg = smoothstep(0.5, 2.0, lam);
            if (tg > 0.0) {
                vec3 glow = vec3(1.0, 0.93, 0.85) * PROC_GAIN / (PROC_SLOPE + 1.0)
                          * pow(max(lam, 1e-6), PROC_GLOW_POW) * PROC_GLOW_GAIN;
                c = mix(c, glow, tg);
            }
        }
    } else {
        c = (SKY_LOD0 == 1 && gscale < 2.0) ? textureLod(u_skycube, d, 0.0).rgb * SKY64_GAIN
                                            : textureGrad(u_skycube, d, gx * gscale, gy * gscale).rgb * SKY64_GAIN;
    }
    #define SKY_STAR_GCUT 11.0
    if (SKY_PROCEDURAL == 0) {
        float pk  = max(c.r, max(c.g, c.b)) / SKY64_GAIN;
        float thr = 0.5 * 0.13 * pow(10.0, 0.148 * (11.7 - SKY_STAR_GCUT));
        c *= smoothstep(thr * 0.5, thr * 1.5, pk);
    }
    c *= (SKY_GAIN + SKY_AUDIO * treble);
    #define SKY_MARK 0
    if (SKY_MARK == 2) {
        float th = acos(clamp(dot(normalize(d), gSheetDir), -1.0, 1.0));
        if (th < SHEET_W * 0.7071) c += vec3(0.5, 0.0, 0.0);
    }

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
    vec3 acc = vec3(0.0), wsum = vec3(0.0), wmax = vec3(0.0);
    #define SKY_MAXMIP 1
    vec3 glow = (SKY_MAXMIP == 1) ? vec3(0.0) : skyColorG(dG, treble, bEff, GLOW_BLUR);
    for (int k = 0; k < SPLIT_N; k++) {
        float t = 1.0 - 2.0 * float(k) / float(SPLIT_N - 1);
        vec3 w = vec3(exp(-pow((t - 0.75) / SPLIT_SIGMA, 2.0)),
                      exp(-pow( t          / SPLIT_SIGMA, 2.0)),
                      exp(-pow((t + 0.75) / SPLIT_SIGMA, 2.0)));
        vec3 dk = skyRadial(normalize(rd + bend * (1.0 + t * disp)), t * ca);
        acc  += w * (skyColorG(dk, treble, bEff, 1.0) - ((SKY_MAXMIP == 1) ? vec3(0.0) : skyColorG(dk, treble, bEff, GLOW_BLUR)));
        wsum += w;
        wmax = max(wmax, w);
    }
    float th    = acos(clamp(dot(normalize(rd), gHoleDir), -1.0, 1.0));
    float span  = max(2.0 * disp * length(bend), 2.0 * ca * th);
    float pix   = max(max(length(dFdx(dG)), length(dFdy(dG))), 1e-6);
    float sepPx = span / float(SPLIT_N - 1) / pix;
    vec3  norm  = mix(wsum, wmax, clamp(sepPx, 0.0, 1.0));
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

#define PULSAR_SPIN_TURNS 2.0
#define PULSAR_SPIN_CONE  1.5707963
#define PULSAR_SPIN_HOLE 1
vec3 gRo = vec3(0.0);
vec3 getPulsarAxisAt(vec3 pP) {
    float ph = 6.28318530718 * fract(PULSAR_SPIN_TURNS * tempoBeatCount());
    vec3 c  = normalize(gRo - pP);
    vec3 U  = normalize(gCamU - c * dot(gCamU, c));
    return normalize(cos(ph) * U + sin(ph) * c);
}
vec3 getPulsarAxis() {
    vec3 u, v, n;
    gearBasis(float(GEAR_BINARY), u, v, n);
    if (PULSAR_SPIN_TURNS == 0.0) return n;
    float ph = 6.28318530718 * fract(PULSAR_SPIN_TURNS * tempoBeatCount());
    return normalize(cos(PULSAR_SPIN_CONE) * n + sin(PULSAR_SPIN_CONE) * (cos(ph) * u + sin(ph) * v));
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

#define DROP_ON    1
#define DROP_FAR   0.80
#define DROP_NEAR  0.36
#define DROP_LEN   2.6
#define DROP_PINCH 3.0
#define DROP_SHRINK 0.25
float dropK(vec3 c) {
    return DROP_ON == 1 ? smoothstep(DROP_FAR, DROP_NEAR, length(c)) : 0.0;
}
float sdDrop(vec3 p, vec3 c, float R, float k) {
    vec3  ax = -normalize(c);
    float L  = max(k * DROP_LEN * R, 1e-6 * R);
    float r1 = R * (1.0 - DROP_SHRINK * k);
    vec3  d  = p - c;
    float s  = dot(d, ax);
    if (s <= 0.0) return length(d) - r1;
    if (s >= L)   return length(d - ax * L);
    float q  = length(d - ax * s);
    float u  = s / L;
    float om = 1.0 - u;
    float rho   = r1 * pow(om, DROP_PINCH) * (1.0 + DROP_PINCH * u);
    float slope = r1 / L * DROP_PINCH * (1.0 + DROP_PINCH) * u * pow(om, DROP_PINCH - 1.0);
    return (q - rho) / sqrt(1.0 + slope * slope);
}
vec3 dropNormal(vec3 p, vec3 c, float R, float k) {
    if (k < 1e-3) return normalize(p - c);
    float e = 0.002 * R;
    vec2  h = vec2(e, 0.0);
    return normalize(vec3(sdDrop(p + h.xyy, c, R, k) - sdDrop(p - h.xyy, c, R, k),
                          sdDrop(p + h.yxy, c, R, k) - sdDrop(p - h.yxy, c, R, k),
                          sdDrop(p + h.yyx, c, R, k) - sdDrop(p - h.yyx, c, R, k)));
}
bool geoSegDrop(vec3 A, vec3 B, vec3 c, float R, float k, out float tHit, out vec3 hitPos) {
    if (k < 1e-3) return geoSegSphere(A, B, c, R, tHit, hitPos);
    vec3  ab = B - A;
    float L  = length(ab);
    if (L < 1e-9) return false;
    vec3  dir = ab / L;
    float half_ = 0.5 * (k * DROP_LEN * R + R + R);
    vec3  mid   = c - normalize(c) * (0.5 * k * DROP_LEN * R);
    vec3  m  = A - mid;
    float bb = dot(m, dir);
    float cc = dot(m, m) - half_ * half_;
    if (cc > 0.0 && bb > 0.0) return false;
    if (bb * bb - cc < 0.0) return false;
    float t = max(0.0, -bb - sqrt(max(bb * bb - cc, 0.0)));
    for (int i = 0; i < 48; i++) {
        if (t > L) return false;
        float d = sdDrop(A + dir * t, c, R, k);
        if (d < 2e-4 * R) {
            tHit = t / L; hitPos = A + dir * t;
            return true;
        }
        t += 0.8 * d;
    }
    return false;
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
#define DISK_SHIFT    0.0
#define DISK_R_IN_VIS 2.0
#define DISK_BEAM_POW 1.5
#define DISK_EXPOSE   1.6
#define DISK_WHITE    6.0
vec3 diskToneSum(vec3 x) {
    float l = max(max(x.r, x.g), x.b) * DISK_EXPOSE;
    if (l <= 0.0) return vec3(0.0);
    float m = l * (1.0 + l / (DISK_WHITE * DISK_WHITE)) / (1.0 + l);
    return x * (DISK_EXPOSE * m / l);
}
#define DISK_COVER    0.5
#define DISK_GLOW     1.0
float gDiskTr = 1.0;
#define DISK_SHOW     0
#define FLARE_ON      1
#define FLARE_START   -1.0
#define FLARE_GROW    0.80
#define FLARE_WRAP_SOFT 0.8
#define DISK_H_SCALE  0.35
#define FLARE_SHRINK  1.40
#define FLARE_RMAX    6.0
#define FLARE_SOFT    0.15
#define DISK_ORANGE   1
#define DISK_ORANGE_COL vec3(1.00, 0.165, 0.022)
#define DISK_HOT_COL  vec3(1.00, 0.72, 0.42)
#define DISK_HOT_GAIN 2.0
#define DISK_HOT_POW  2.0
#define FLARE_PERIOD 256.0
const float FLARE_PERI[14] = float[14](25.304, 41.362, 57.414, 73.466, 89.516, 105.580, 129.650,
                                       151.480, 167.546, 183.598, 199.654, 215.712, 231.772, 255.866);
float gDiskEdge = 0.0;
float gFlareDt = 0.0;
float flareWrap(float phi, float phD) {
    if (FLARE_ON == 0) return 1.0;
    float sweep = (6.28318530718 + FLARE_WRAP_SOFT) * clamp(gFlareDt / FLARE_GROW, 0.0, 1.0);
    float dph = mod(phi - phD, 6.28318530718);
    return clamp((sweep - dph) / FLARE_WRAP_SOFT, 0.0, 1.0);
}
float flareEdge() {
    if (FLARE_ON == 0) return FLARE_RMAX;
    float cm = mod(tempoBeatCount(), FLARE_PERIOD);
    for (int k = 0; k < 14; k++) {
        float dt = mod(cm - FLARE_PERI[k] - FLARE_START, FLARE_PERIOD);
        float e = -1.0;
        if (dt >= 0.0 && dt < FLARE_GROW) {
            e = 1.0;
        } else if (dt >= FLARE_GROW && dt < FLARE_GROW + FLARE_SHRINK) {
            float x = (dt - FLARE_GROW) / FLARE_SHRINK;  e = 1.0 - x * x;
        }
        if (e >= 0.0) { gFlareDt = dt; return mix(DISK_R_IN_VIS, FLARE_RMAX, e); }
    }
    return 0.0;
}
uniform sampler2D u_kntable;
#define KN_RIN        2.0
#define KN_ROUT       20.0
#define DSIM_TLOOP    33.3932
#define DSIM_RIN      2.0
#define DSIM_ROUT     20.0
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
#define DISK_LIVE 1

float diskLiveContrast(float rR, float phi) {
    float rSim = rR;
    float x = log(clamp(rSim, u_disk_rin, u_disk_rout));
    float u = (x - u_disk_x0) / (log(u_disk_rout) - u_disk_x0);
    float v = phi / 6.28318530718;
    vec4 st = texture(u_disklive, vec2(u, v));
    float S = max(st.x, 1e-8);
    float S0 = pow(rSim, -0.5) * (1.0 - 0.7 * exp(-(rSim - u_disk_rin) / (0.3 * u_disk_rin)));
    return S / max(S0, 1e-6);
}

float diskStreaks(float rR, float phi) {
#if DISK_LIVE
    #define DISK_LIVE_GAIN 15.0
    return max(diskLiveContrast(rR, phi) * DISK_LIVE_GAIN, 0.0);
#else
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
#endif
}

vec4 diskEmitTr(float rR, float tr, float lam, float bassLvl) {
    vec4  tab = knTable(rR);
    float g   = 1.0 / max(tab.y * (1.0 - tab.x * lam), 1e-6);
    float F   = tab.z;
    float boost = pow(g, DISK_BEAM_POW);
    vec3 hue = vec3(1.0);
    float hotB = 1.0;
    if (DISK_ORANGE == 1) {
        float x   = clamp((rR - DISK_R_IN_VIS) / max(FLARE_RMAX - DISK_R_IN_VIS, 1e-3), 0.0, 1.0);
        float hot = pow(1.0 - x, DISK_HOT_POW);
        vec3  cO  = DISK_ORANGE_COL / dot(DISK_ORANGE_COL, vec3(0.299, 0.587, 0.114));
        vec3  cH  = DISK_HOT_COL    / dot(DISK_HOT_COL,    vec3(0.299, 0.587, 0.114));
        hue  = mix(cO, cH, hot);
        hotB = 1.0 + DISK_HOT_GAIN * hot;
    } else {
        float D = 1.0 / max(1.0 - tab.x * lam, 1e-3);
        float t = clamp(log(D) / log(DISK_D_SPAN), -1.0, 1.0);
        vec3  tint = (t > 0.0) ? vec3(0.42, 0.64, 1.00) : vec3(1.00, 0.42, 0.22);
        tint /= dot(tint, vec3(0.299, 0.587, 0.114));
        hue = mix(hue, tint, abs(t) * DISK_TINT);
        float hl = dot(hue, vec3(0.299, 0.587, 0.114));
        hue = max(mix(vec3(hl), hue, DISK_SAT), 0.0);
    }
    vec3 src = hue * hotB * boost * F
             * DISK_GAIN * DISK_SIM_GAIN * (0.70 + 0.90 * bassLvl);
    if (DISK_TONE_K > 0.0 && DISK_THIN == 0) {
        float lum = max(max(src.r, src.g), src.b);
        if (lum > 0.0) src *= (log(1.0 + DISK_TONE_K * lum) / log(1.0 + DISK_TONE_K)) / lum;
    }
    return vec4(src * (1.0 - tr), tr);
}

vec4 diskShareShade(float rR, float phi, float colShare, float lam, float bassLvl) {
    if (rR <= 0.0 || colShare <= 0.0) return vec4(0.0, 0.0, 0.0, 1.0);
    float c   = diskStreaks(rR, phi);
    c *= 1.0 - exp(-2.5 * max(DSIM_ROUT - rR, 0.0) / (0.35 * DSIM_ROUT));
    c *= smoothstep(DISK_R_IN_VIS, DISK_R_IN_VIS * 1.02, rR);
    float tau = DISK_TAU0 * c * colShare / (2.50663 * DISK_HR * rR);
    float tr  = exp(-tau);
    return diskEmitTr(rR, tr, lam, bassLvl);
}

uniform float u_disk_hr0;
uniform float u_disk_flare;
#define DISK_HR_MAX   0.6
#define DISK_H_CAP    2.5
#define DISK_ZMAX     3.0
#define DISK_VOL_N    16
#define DISK_VOL_GAIN 1.0
float diskHR0(float rR) { return u_disk_hr0 * pow(max(rR / u_disk_rin, 1e-3), u_disk_flare); }
vec2 diskLiveCH(float rR, float phi) {
    float x = log(clamp(rR, u_disk_rin, u_disk_rout));
    float u = (x - u_disk_x0) / (log(u_disk_rout) - u_disk_x0);
    vec4 st = texture(u_disklive, vec2(u, phi / 6.28318530718));
    float S  = max(st.x, 1e-8);
    float S0 = pow(rR, -0.5) * (1.0 - 0.7 * exp(-(rR - u_disk_rin) / (0.3 * u_disk_rin)));
    float Om = max(knTable(rR).x, 1e-6);
    float T  = (st.w > 0.0) ? st.w / S : pow(diskHR0(rR) * rR * Om, 2.0);
    float H  = min(sqrt(max(T, 0.0)) / Om, min(DISK_H_CAP * diskHR0(rR), DISK_HR_MAX) * rR);
    return vec2(S / max(S0, 1e-6), max(H * DISK_H_SCALE, 1e-4));
}
float erfA(float x) {
    float t = 1.0 / (1.0 + 0.3275911 * abs(x));
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return sign(x) * y;
}
float gaussColumn(float z0, float z1, float L, float H) {
    float dz = z1 - z0;
    if (abs(dz) < 1e-4 * H) return L * exp(-0.5 * z0 * z0 / (H * H));
    float k = 0.70710678 / H;
    return L / dz * 1.25331414 * H * (erfA(z1 * k) - erfA(z0 * k));
}
uniform float u_disk_time;
#define DISK_DETAIL      1.0
#define FLARE_RAGGED     0.20
#define FLARE_FRONT_RAG  0.9
#define DISK_DETAIL_MEAN 1.55
float knPerlin(vec3 P) {
    vec3 I = floor(P), F = fract(P);
    vec3 Sx = 3.0 * F * F - 2.0 * F * F * F;
    #define KNH(o) (2.0 * fract(sin(dot(I + o, vec3(12.9898, 78.233, 213.765))) * 43758.5453) - 1.0)
    float v000 = KNH(vec3(0,0,0)), v100 = KNH(vec3(1,0,0)), v010 = KNH(vec3(0,1,0)), v110 = KNH(vec3(1,1,0));
    float v001 = KNH(vec3(0,0,1)), v101 = KNH(vec3(1,0,1)), v011 = KNH(vec3(0,1,1)), v111 = KNH(vec3(1,1,1));
    #undef KNH
    return mix(mix(mix(v000, v100, Sx.x), mix(v010, v110, Sx.x), Sx.y),
               mix(mix(v001, v101, Sx.x), mix(v011, v111, Sx.x), Sx.y), Sx.z);
}
float knDiskNoise(vec3 P, float lo, float hi, float con) {
    float acc = 10.0;
    for (int i = int(floor(lo)); i < int(ceil(hi)); i++) {
        float w = max(0.0, min(hi, float(i) + 1.0) - max(lo, float(i)));
        if (w <= 0.0) continue;
        acc *= 1.0 + 0.1 * knPerlin(pow(3.0, float(i)) * P) * w;
    }
    return log(1.0 + pow(0.1 * acc, con));
}
float knSpiral(float r) {
    float u = sqrt(r), kc = pow(KERR_A * 0.5 * 0.70710678, 0.33333333);
    return (5.6568542 / kc) * (0.5 * log(max(1e-9, (r - kc * u + kc * kc) / max(1e-9, (u + kc) * (u + kc))))
         + 1.7320508 * (atan(2.0 * u - kc, 1.7320508 * kc) - 1.5707963));
}
float knWrap(float x) { return x - 6.28318530718 * floor((x + 3.14159265359) / 6.28318530718); }
float diskKNDetail(float r, float th, float y) {
    float t  = u_disk_time;
    float RO = DSIM_ROUT;
    float GT = 0.75 + max(0.0, (r - 3.0) * 0.24);
    float NL = max(0.0, 2.0 - 0.6 * GT);
    float Lm = 0.91 * log(1.0 + 0.06 / 0.91 * max(0.0, r - 10.0));
    float Cm = 80.0 * log(1.0 + 0.006 * max(0.0, r - 10.0));
    float cS = 0.02 * pow(RO, 0.7);
    float PT = knWrap(th + knSpiral(r));
    float m  = knDiskNoise(vec3(0.1 * (r + 0.25 / 3.0 * t), 0.1 * y, cS * PT), NL + 2.0 - Lm, NL + 4.0 - Lm, 80.0 - Cm);
    if (PT + 3.14159265 < 0.1 * 3.14159265) {
        float f = (PT + 3.14159265) / (0.1 * 3.14159265);
        m = m * f + (1.0 - f) * knDiskNoise(vec3(0.1 * (r + 0.25 / 3.0 * t), 0.1 * y, cS * (PT + 6.28318531)), NL + 2.0 - Lm, NL + 4.0 - Lm, 80.0 - Cm);
    }
    if (r > max(0.15379 * RO, 0.15379 * 64.0)) {
        float PL = knWrap(th + 2.0 * log(r));
        float TS = r * 4.65114e-6 - 0.1 / 3.0 * t;
        float sp = knDiskNoise(vec3(0.1 * (TS - 0.08 * RO * PL), 0.1 * y, cS * PL), NL + 2.0 - Lm, NL + 3.0 - Lm, 80.0 - Cm);
        float wm = 0.5 + 0.5 * max(-1.0, 1.0 - exp(-0.15 * (100.0 * r / max(RO, 64.0) - 20.0)));
        m *= mix(1.0, clamp(0.7 * sp * 1.5 - 0.5, 0.0, 3.0), wm);
    }
    return mix(1.0, m / DISK_DETAIL_MEAN, DISK_DETAIL);
}

#define KN_DISK        1
#define KN_R_IN        2.0
#define KN_R_OUT       6.0
#define KN_THIN        0.75
#define KN_HOPPER      0.24
#define KN_BRIGHT      1.4
#define KN_DARK        0.5
#define KN_REDDEN      0.3
#define KN_SAT         0.5
#define KN_BB_EXP      0.5
#define KN_RS_COL_EXP  3.0
#define KN_RS_INT_EXP  4.0
#define KN_RING_BOOST  11.0
#define KN_RING_TBOOST 2.8
#define KN_BOOST_ROT   1.0
#define KN_SHIFT_MAX   1.0
#define KN_DISK_ARG    2.071973e+20
#define KN_PEAK_T      5.853308e+04
#define KN_TIME_RATE   2.0
#define KN_STEP        1.0
#define KN_MAX_IT      400
#define KN_GAIN        1.0
#define KN_A           (KERR_A * 0.5)
#define KN_Q           (0.07 * 0.5)
#define KN_LIFE        1
#define STATE_KN_PX    40
#define KN_START_K     0.35
#define KN_END_K       0.02
#define KN_MIN_IN      0.5
#define KN_MIN_OUT     0.8
#define KN_LINGER      1.0
#define KN_INFALL_F    0.4
#define KN_NOISE_RATE  20.0
#define KN_WRAP_SOFT   0.8
#define KN_FRONT_RAG   0.9
#define KN_EDGE_SOFT   0.12
#define KN_EDGE_RAG    0.15
float gKNSince = -1.0;
float gKNPeri  = -1e9;
float gKNIn    = 1.0;
float gKNOut   = 1.0;
float gKNTau   = 0.0;
float gKNSeed  = 0.0;
bool  gKNOn    = true;
float dropK(vec3 c);
vec3  starTouchPosAt(int star, float c);
float knOmegaK(float r);
#define KN_SAMP_PX     48
#define KN_SAMP_N      80
#define KN_SAMP_H      0.1
#define KN_STARS       2
#define KN_DWARF_GAIN  0.6
#define KN_DWARF_TEMP  0.75
#define KN_MIN_LEAD    1.0
#define KN_MIN_LEAD_P  1.5
#define KN_SNAP        0.5
float knDK(int star, float c) { return dropK(starTouchPosAt(star, c)); }
vec4  knSample(int star, float c) { vec3 p = starTouchPosAt(star, c); return vec4(dropK(p), c, length(p), 0.0); }
vec4 knPassTrack(float c, int row) {
    int ib = 0; float best = 1e9;
    for (int k = 0; k < KN_SAMP_N; k++) {
        float rz = texelFetch(iChannel0, ivec2(KN_SAMP_PX + k, row), 0).z;
        if (rz < best) { best = rz; ib = k; }
    }
    vec2 pk = texelFetch(iChannel0, ivec2(KN_SAMP_PX + ib, row), 0).xy;
    if (pk.x < 0.9) return vec4(-1.0, -1e9, 1.0, 1.0);
    float tS = texelFetch(iChannel0, ivec2(KN_SAMP_PX, row), 0).y;
    float tE = texelFetch(iChannel0, ivec2(KN_SAMP_PX + KN_SAMP_N - 1, row), 0).y;
    for (int k = ib; k > 0; k--) {
        vec2 a = texelFetch(iChannel0, ivec2(KN_SAMP_PX + k - 1, row), 0).xy, b = texelFetch(iChannel0, ivec2(KN_SAMP_PX + k, row), 0).xy;
        if (a.x < KN_START_K) { tS = mix(a.y, b.y, clamp((KN_START_K - a.x) / max(b.x - a.x, 1e-6), 0.0, 1.0)); break; }
    }
    for (int k = ib; k < KN_SAMP_N - 1; k++) {
        vec2 a = texelFetch(iChannel0, ivec2(KN_SAMP_PX + k, row), 0).xy, b = texelFetch(iChannel0, ivec2(KN_SAMP_PX + k + 1, row), 0).xy;
        if (b.x < KN_END_K) { tE = mix(a.y, b.y, clamp((a.x - KN_END_K) / max(a.x - b.x, 1e-6), 0.0, 1.0)); break; }
    }
    tS = min(tS, pk.y - (row == 2 ? KN_MIN_LEAD_P : KN_MIN_LEAD));
    tS = KN_SNAP * floor(tS / KN_SNAP + 1e-4);
    return vec4(c - tS, c - pk.y, pk.y - tS, tE - pk.y);
}
int   gKNStar = 0;
float gKNGain = 1.0;
float gKNTemp = 1.0;
void knLifeFrame() {
    if (KN_LIFE == 0) { gKNOn = true; gKNTau = KN_TIME_RATE * iTime; return; }
    float c  = tempoBeatCount();
    vec4  sP = texelFetch(iChannel0, ivec2(STATE_KN_PX, 0), 0);
    vec4  sD = texelFetch(iChannel0, ivec2(STATE_KN_PX + 1, 0), 0);
    bool  vP = (KN_STARS != 1) && sP.y > -1e8;
    bool  vD = (KN_STARS != 0) && sD.y > -1e8;
    float s0P = c - sP.x, pkP = c - sP.y, enP = pkP + max(sP.w + KN_LINGER, KN_MIN_OUT);
    float s0D = c - sD.x, pkD = c - sD.y, enD = pkD + max(sD.w + KN_LINGER, KN_MIN_OUT);
    bool  merge = vP && vD && ((s0P <= s0D) ? (s0D <= enP) : (s0P <= enD));
    float tS, tP0, tP1, tE, gF, gL, kF, kL; int stF, stL;
    if (merge) {
        bool pF = pkP <= pkD;
        tS  = min(s0P, s0D);
        tP0 = pF ? pkP : pkD;  tP1 = pF ? pkD : pkP;  tE = pF ? enD : enP;
        stF = pF ? 0 : 1;      stL = pF ? 1 : 0;
    } else {
        bool useP = vP && (!vD || abs(c - pkP) <= abs(c - pkD));
        tS  = useP ? s0P : s0D;  tP0 = useP ? pkP : pkD;  tP1 = tP0;  tE = useP ? enP : enD;
        stF = useP ? 0 : 1;      stL = stF;
        if (!vP && !vD) { gKNOn = false; return; }
    }
    gF = (stF == 0) ? 1.0 : KN_DWARF_GAIN;  gL = (stL == 0) ? 1.0 : KN_DWARF_GAIN;
    kF = (stF == 0) ? 1.0 : KN_DWARF_TEMP;  kL = (stL == 0) ? 1.0 : KN_DWARF_TEMP;
    float w = (tP1 > tP0) ? smoothstep(tP0, tP1, c) : 0.0;
    gKNGain  = mix(gF, gL, w);
    gKNTemp  = mix(kF, kL, w);
    gKNStar  = (w < 0.5) ? stF : stL;
    gKNSince = c - tS;
    gKNIn    = max(tP0 - tS, KN_MIN_IN);
    gKNPeri  = c - tP1;
    gKNOut   = max(tE - tP1, 1e-3);
    gKNOn    = (gKNSince >= 0.0) && (c < tE);
    gKNSeed  = 53.0 * mod(floor(tP0 + 0.5), 97.0);
    gKNTau   = KN_NOISE_RATE * gKNSince + gKNSeed;
}
float knLife(float r, float th, float y, float phD) {
    if (KN_LIFE == 0) return 1.0;
    float since = gKNSince - KN_INFALL_F * gKNIn * (KN_R_OUT - r) / (KN_R_OUT - KN_R_IN);
    if (since <= 0.0) return 0.0;
    float rate  = (6.28318530718 + KN_WRAP_SOFT + KN_FRONT_RAG) / (knOmegaK(KN_R_OUT) * gKNIn);
    float front = knOmegaK(r) * rate * since;
    float dph = mod(th - phD + KN_FRONT_RAG * knPerlin(vec3(1.3 * r, 0.5 * y, 2.0 * gKNSince + gKNSeed)), 6.28318530718);
    float wind = clamp((front - dph) / KN_WRAP_SOFT, 0.0, 1.0);
    float x = clamp(gKNPeri / gKNOut, 0.0, 1.0);
    if (x >= 1.0) return 0.0;
    float edge = KN_R_IN * pow(KN_R_OUT / KN_R_IN, 1.0 - x);
    edge *= 1.0 + KN_EDGE_RAG * knPerlin(vec3(2.0 * cos(th), 2.0 * sin(th), 0.7 * r + 3.0 * gKNSince + gKNSeed));
    return wind * (1.0 - smoothstep(edge * (1.0 - KN_EDGE_SOFT), edge, r));
}
vec3 gKNEmit = vec3(0.0);
vec3 gKNTr   = vec3(1.0);

float knKSRadius(float rho2, float y2) {
    float a2 = KN_A * KN_A;
    float b  = rho2 + y2 - a2;
    float de = sqrt(b * b + 4.0 * a2 * y2);
    float r2 = (b >= 0.0) ? 0.5 * (b + de) : (2.0 * a2 * y2) / max(1e-20, de - b);
    return sqrt(r2);
}
float knOmegaK(float r) {
    float m = 0.5 * r - KN_Q * KN_Q;
    if (m < 0.0) return 0.0;
    float s = sqrt(m);
    return s / max(1e-6, r * r + KN_A * s);
}
vec3 knKelvin(float K) {
    if (K < 400.01) return vec3(0.0);
    float Te = (K - 6500.0) / (6500.0 * K * 2.2);
    vec3  c  = vec3(exp(2.05539304e4 * Te), exp(2.63463675e4 * Te), exp(3.30145739e4 * Te));
    float s  = 1.0 / max(max(1.5 * c.r, c.g), c.b);
    if (K < 1000.0) s *= (K - 400.0) / 600.0;
    return c * s;
}
float knShapeF(float x, float al, float be) {
    float k = pow(al + be, al + be) / (pow(al, al) * pow(be, be));
    return k * pow(max(x, 0.0), al) * pow(max(1.0 - x, 0.0), be);
}
float knSoftSat(float x) { return 1.0 - 1.0 / (max(x, 0.0) + 1.0); }
vec3 knToneMap(vec4 R) {
    float s  = R.r + R.g + R.b + 1e-6;
    vec3  f  = 3.0 * R.rgb / s;
    vec3  m  = -4.0 * log(1.0 - pow(clamp(R.rgb, 0.0, 0.999), vec3(2.2)));
    return min(m, 8.0 * f);
}
float knShellStep(vec3 A, vec3 B, float rA, float rB) {
    if (rB >= 1.6 + pow(KERR_A, 0.666666)) return 0.0;
    vec3  d  = B - A;
    float L  = length(d);
    if (L < 1e-9) return 0.0;
    float dr = rB - rA;
    float rot = clamp(1.0 + KN_BOOST_ROT * dot(-d, vec3(B.z, 0.0, -B.x)) / L / max(length(B.xz), 1e-6) * KERR_A, 0.0, 2.0);
    return L / (0.5 * rA + 0.5 * rB) / (1.0 + 1000.0 * (dr / L) * (dr / L)) * rot
         * clamp(11.0 - 10.0 * (KERR_A * KERR_A + 0.07 * 0.07), 0.0, 1.0);
}
void knDiskChord(vec3 A, vec3 B, float arc0, float lam, float shell, float phD, inout vec4 acc, inout float phase) {
    float MH = KN_THIN + max(0.0, KN_HOPPER * KN_R_OUT) + 2.0;
    if ((A.y > MH && B.y > MH) || (A.y < -MH && B.y < -MH)) return;
    vec2  P0 = A.xz, V = B.xz - A.xz;
    float L2 = dot(V, V);
    vec2  CP = P0 + V * ((L2 > 1e-8) ? clamp(-dot(P0, V) / L2, 0.0, 1.0) : 0.0);
    if (dot(CP, CP) > 1.21 * KN_R_OUT * KN_R_OUT) return;
    if (max(knKSRadius(dot(A.xz, A.xz), A.y * A.y), knKSRadius(dot(B.xz, B.xz), B.y * B.y)) < KN_R_IN * 0.9) return;
    float Tot  = length(B - A);
    if (Tot < 1e-9) return;
    float cosY = abs((B.y - A.y) / Tot);
    float trav = 0.0;
    const float PI = 3.14159265359;
    float RIO = KN_R_OUT - KN_R_IN;
    for (int it = 0; it < KN_MAX_IT; it++) {
        if (trav >= Tot || acc.a > 0.99) break;
        float D  = length(mix(A, B, trav / Tot));
        float SB = max(KN_R_OUT, 12.0);
        float St = 0.15 + 0.25 * min(max(0.0, 0.5 * (0.5 * D / max(10.0, SB) - 1.0)), 1.0);
        if (D >= 2.0 * SB) St *= D;
        else if (D >= SB) St *= ((1.0 + 0.25 * max(D - 12.0, 0.0)) * (2.0 * SB - D) + D * (D - SB)) / SB;
        else St *= min(1.0 + 0.25 * max(D - 12.0, 0.0), D);
        St = max(0.01, St) * KN_STEP;
        float nxt = min(Tot, trav + phase * St);
        if (nxt < Tot) { phase = 1.0; trav = nxt; }
        else { phase = max(0.0, phase - (Tot - trav) / St); trav = Tot; break; }

        vec3  S    = mix(A, B, trav / Tot);
        float tEm  = gKNTau - (arc0 + trav);
        float y    = S.y;
        float PosR = knKSRadius(dot(S.xz, S.xz), y * y);
        float GT   = KN_THIN + max(0.0, (length(S.xz) - 3.0) * KN_HOPPER);
        float ICB  = max(GT, KN_THIN) * max(0.0, 1.0 - 5.0 * pow((PosR - KN_R_IN) / min(RIO, 12.0), 2.0));
        if (!(abs(y) < max(GT * 1.5, ICB) && PosR < KN_R_OUT && PosR > KN_R_IN)) continue;

        float th  = atan(S.z, S.x);
        float lifeM = knLife(PosR, th, y, phD);
        if (lifeM <= 0.0) continue;
        float NL  = max(0.0, 2.0 - 0.6 * GT);
        float x   = (PosR - KN_R_IN) / max(1e-6, RIO);
        float ap  = max(1.0, RIO / 10.0);
        float ER  = (ap == 1.0) ? x : (-1.0 + sqrt(max(0.0, 1.0 + 4.0 * ap * ap * x - 4.0 * x * ap))) / (2.0 * ap - 2.0);
        float DT  = knShapeF(ER, 0.9, 1.5);
        float PL  = knWrap(th + 2.0 * log(max(1e-6, PosR)));
        float tk  = 0.4 + 0.6 * clamp(GT - 0.5, 0.0, 2.5) / 2.5;
        float PTk = max(1e-6, GT * DT * (tk + (1.0 - tk) * knSoftSat(knDiskNoise(vec3(1.5 * PL, PosR + 0.25 / 3.0 * tEm, 0.0), -0.7 + NL, 1.3 + NL, 80.0))));
        if (!(abs(y) < PTk || abs(y) < ICB)) continue;

        float Om = knOmegaK(max(KN_R_IN, PosR));
        float PT = knWrap(th + knSpiral(PosR));
        float ir = 1.0 / max(1e-6, PosR);
        float Vp = ir - KN_Q * KN_Q * ir * ir;
        float gtt = -(1.0 - Vp), gtp = -KN_A * Vp, gpp = PosR * PosR + KN_A * KN_A + KN_A * KN_A * Vp;
        float nm  = gtt + 2.0 * Om * gtp + Om * Om * gpp;
        float ut  = inversesqrt(max(0.01, -nm));
        float Eem = ut * (1.0 - Om * lam);
        float FR  = 1.0 / max(1e-6, Eem);
        float Tb  = pow(KN_DISK_ARG * ir * ir * ir * max(1.0 - sqrt(KN_R_IN * ir), 1e-6), 0.25);
        float VT  = Tb * pow(FR, KN_RS_COL_EXP);
        float BW  = (0.05 * min(KN_R_OUT / 1000.0, 1000.0 / KN_R_OUT)
                  + 0.55 / exp(5.0 * ER) * mix(0.2 + 0.8 * cosY, 1.0, clamp(GT - 0.8, 0.2, 1.0)))
                  * pow(Tb / KN_PEAK_T, KN_BB_EXP);

        float Den = DT;
        vec4  SC  = vec4(0.0);
        if (abs(y) < PTk) {
            float Lm = 0.91 * log(1.0 + (0.06 / 0.91 * max(0.0, min(1000.0, PosR) - 10.0)));
            float Cm = 80.0 * log(1.0 + (0.1 * 0.06 * max(0.0, PosR - 10.0)));
            float cS = 0.02 * pow(KN_R_OUT, 0.7);
            vec3  nP = vec3(0.1 * (PosR + 0.25 / 3.0 * tEm), 0.1 * y, cS * PT);
            SC = vec4(knDiskNoise(nP, NL + 2.0 - Lm, NL + 4.0 - Lm, 80.0 - Cm));
            if (PT + PI < 0.1 * PI) {
                float f = (PT + PI) / (0.1 * PI);
                SC = SC * f + (1.0 - f) * vec4(knDiskNoise(vec3(nP.xy, cS * (PT + 2.0 * PI)), NL + 2.0 - Lm, NL + 4.0 - Lm, 80.0 - Cm));
            }
            if (PosR > max(0.15379 * KN_R_OUT, 0.15379 * 64.0)) {
                float TS = PosR * 4.65114e-6 - 0.1 / 3.0 * tEm;
                float sp = knDiskNoise(vec3(0.1 * (TS - 0.08 * KN_R_OUT * PL), 0.1 * y, cS * PL), NL + 2.0 - Lm, NL + 3.0 - Lm, 80.0 - Cm);
                SC *= mix(1.0, clamp(0.7 * sp * 1.5 - 0.5, 0.0, 3.0), 0.5 + 0.5 * max(-1.0, 1.0 - exp(-0.15 * (100.0 * PosR / max(KN_R_OUT, 64.0) - 20.0))));
            }
            Den *= 0.7 * max(0.0, 1.0 - abs(y) / PTk);
            float yr = clamp(abs(y) / PTk, 0.0, 1.0);
            SC.xyz *= Den * 1.4 * max(0.0, 0.2 + 2.0 * sqrt(max(0.0, yr * yr + 0.001)));
            SC.a   *= Den * Den / 0.3;
        }
        float rg = clamp(0.3 * shell - 0.1, 0.0, 1.0);
        SC.xyz *= 1.0 + clamp(KN_RING_BOOST, 0.0, 10.0) * rg;
        VT     *= 1.0 + clamp(KN_RING_TBOOST, 0.0, 10.0) * rg;
        VT     *= gKNTemp;

        float cP = knOmegaK(max(3.0, KN_R_IN)) * tEm;
        if (abs(y) < ICB) {
            float DI = max(1.0 - pow(y / (GT * max(1.0 - 5.0 * pow((PosR - KN_R_IN) / min(RIO, 12.0), 2.0), 0.0001)), 2.0), 0.0);
            if (DI > 0.0) {
                float ang = knWrap(th - 0.666666 * cP);
                float n   = knDiskNoise(vec3(1.5 * fract((1.5 * ang + cP) / 2.0 / PI) * 2.0 * PI, PosR, y), 0.0, 6.0, 80.0);
                SC += 0.02 * vec4(vec3(DI * n), 0.2 * DI * n) * sqrt(max(0.0, 1.0001 - cosY * cosY));
            }
        }

        SC.xyz *= BW * knKelvin(VT) * min(pow(FR, KN_RS_INT_EXP), KN_SHIFT_MAX) * min(1.0, 1.3 * (KN_R_OUT - PosR) / RIO);
        SC.a   *= 0.125;
        float DOR = mix(min(KN_R_OUT, 25.0), KN_R_OUT, smoothstep(6.0, max(0.05 * KN_R_OUT, 12.0), PosR));
        SC *= max(vec4(5.0 / (max(KN_THIN, 0.2) + (KN_HOPPER * 0.5) * DOR)),
                  mix(vec4(100.0 / DOR), vec4(vec3(0.3 + 0.7 * 100.0 / DOR), 1.0), exp(-pow(20.0 * PosR / DOR, 2.0))));
        float IBF = mix(3.0, 2.0, clamp((KN_R_OUT - 50.0) / 50.0, 0.0, 1.0));
        float IBR = 1.0 - clamp(6.0 * (PosR - KN_R_IN) / RIO, 0.0, 1.0);
        IBR *= IBR;
        SC.xyz *= mix(1.0, max(1.0, cosY / 0.2), clamp(0.3 - 0.6 * (PTk / max(1e-6, Den) - 1.0), 0.0, 0.3))
                * (1.0 + 1.2 * max(0.0, max(0.0, min(1.0, 3.0 - 2.0 * KN_THIN)) * min(0.5, 1.0 - 5.0 * KN_HOPPER)))
                * KN_BRIGHT * (1.0 + IBF * IBR);
        SC.a   *= KN_DARK * (1.0 + (1.0 + IBF) * IBR);
        if (Eem < 0.0) SC = vec4(0.0);
        SC *= lifeM * gKNGain;

        vec4  SCs = SC * St;
        float oa  = 1.0 - acc.a;
        float wR  = pow(oa, 1.0), wG = pow(oa, 1.0 + 2.0 * KN_REDDEN), wB = pow(oa, 1.0 + 5.0 * KN_REDDEN);
        float Dn  = SCs.r * wR + SCs.g * wG + SCs.b * wB;
        if (Dn > 1e-6) {
            float Sum = (SCs.r + SCs.g + SCs.b) * wG;
            vec3  c3  = Sum * vec3(SCs.r * wR, SCs.g * wG, SCs.b * wB) / Dn;
            float cs  = c3.r + c3.g + c3.b;
            acc.rgb  += c3 * pow(max(3.0 * c3 / max(cs, 1e-12), vec3(0.0)), vec3(KN_SAT));
        }
        acc.a += SCs.a * (1.0 - acc.a);
    }
}

vec4 diskVolShade(float rR, float phi, float zMid, float col, float lam, float bassLvl) {
    if (rR <= 0.0 || col <= 0.0) return vec4(0.0, 0.0, 0.0, 1.0);
    vec2  ch = diskLiveCH(rR, phi);
    float c  = max(ch.x, 0.0) * diskKNDetail(rR, phi, zMid);
    c *= 1.0 - exp(-2.5 * max(DSIM_ROUT - rR, 0.0) / (0.35 * DSIM_ROUT));
    c *= smoothstep(DISK_R_IN_VIS, DISK_R_IN_VIS * 1.02, rR);
    {
        float wob = knPerlin(vec3(2.0 * cos(phi), 2.0 * sin(phi), 0.7 * rR + 3.0 * gFlareDt));
        float eR  = gDiskEdge * (1.0 + FLARE_RAGGED * wob);
        c *= 1.0 - smoothstep(eR * (1.0 - FLARE_SOFT), eR, rR);
    }
    float tau = DISK_TAU0 * c * col / (2.50663 * ch.y);
    vec4  e = diskEmitTr(rR, exp(-tau), lam, bassLvl);
    return vec4(e.rgb * DISK_VOL_GAIN, e.a);
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

vec3 starTouchPosAt(int star, float c) {
    vec2  t = touchBracket(star, c);
    TOrb  o = starTOrb(star);
    float M = touchAnomaly(c, t, touchSolve(o, touchRotation(star, t.x)),
                                 touchSolve(o, touchRotation(star, t.y)), TOUCH_SUB);
    return rotAxis(torbBase(o, M), o.n, touchRotation(star, c));
}
vec3 starTouchPos(int star) { return starTouchPosAt(star, tempoBeatCount()); }

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
