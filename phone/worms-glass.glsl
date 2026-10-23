// Worms Glass (ShaderEditor fork, Pixel 10 Pro): the lagoon worm pair in a
// 64^3 fluid. Tap knocks a water jet + kick, swipe slices, hold calls the pair,
// drag / throw, two fingers make a ring or globe, three-finger tap cycles colours.

#version 300 es

#define IMAGE_ctl   r32f    256x1
#define IMAGE_worm  rgba32f 8x1 swap
#define IMAGE_vel   rgba16f 512x512 swap
#define IMAGE_vtmp  rgba16f 512x512
#define IMAGE_divB  r32f    512x512
#define IMAGE_pres  r32f    512x512
#define IMAGE_vort  rgba16f 512x512
#define IMAGE_part  rgba32f 512x1024 swap
#define IMAGE_accR  r32ui   640x1428
#define IMAGE_accG  r32ui   640x1428
#define IMAGE_accB  r32ui   640x1428
#define IMAGE_glowP rgba16f 640x1428
#define IMAGE_bloom rgba16f 320x714

precision highp float;

#define FW_N       64
#define FW_H       1.6
#define EPS        1e-3
#define VEL_TAU    6.0
#define VISC_A     0.03
#define VORT_EPS   1.5
#define SOR_W      1.5
#define STIR_R     0.08
#define STIR_RATE  10.0

#define FW_PW        512
#define FW_PH        1024
#define FW_NP        524288
#define SPLAT_K      256.0
#define SP_K         64.0
#define FW_RATE      7000.0
#define FW_DENS      6250.0
#define FW_RATE_MAX  21000.0
#define HEAD_MAX_SPD (FW_RATE_MAX / FW_DENS)

#define CAM_D      3.0
#define FOCAL      1.9
#define CAM_TY     0.0
#define CAM_YAW    0.6
#define CAM_PITCH  0.35
#define FOG_K      0.55
#define FOG_D0     2.2

mat3 camRot() {
	float y = CAM_YAW, p = CAM_PITCH;
	mat3 ry = mat3(cos(y), 0.0, -sin(y), 0.0, 1.0, 0.0, sin(y), 0.0, cos(y));
	mat3 rx = mat3(1.0, 0.0, 0.0, 0.0, cos(p), sin(p), 0.0, -sin(p), cos(p));
	return rx * ry;
}
float fogAt(vec3 p, mat3 R) {
	return exp(-max((R * (p - vec3(0.0, CAM_TY, 0.0))).z + CAM_D - FOG_D0, 0.0) * FOG_K);
}
vec3 project(vec3 p, mat3 R) {
	vec3 c = R * (p - vec3(0.0, CAM_TY, 0.0));
	float s = FOCAL / max(c.z + CAM_D, 0.3);
	return vec3(c.xy * s, s);
}

#define COL_GREEN      vec3(0.00, 0.70, 0.40)
#define COL_VIOLET     vec3(0.01, 0.00, 0.30)
#define COL_RED        vec3(1.30, 0.08, 0.08)
#define COL_DARKRED    vec3(0.70, 0.02, 0.04)
#define COL_MAGENTA    vec3(0.90, 0.00, 0.25)
#define COL_CYAN       vec3(0.00, 1.10, 1.00)
#define N_CMODES   3
#define PAL_WHITE  6
const vec3 PAL_COL[7] = vec3[7](COL_GREEN, COL_VIOLET, COL_RED, COL_DARKRED, COL_MAGENTA, COL_CYAN, vec3(1.0));

uniform float time;
uniform int frame;
uniform float beatCount;
uniform float beatPhase;
uniform float beatConf;
uniform float beatBpm;

#ifdef COMPUTE
uniform vec2 resolution;
uniform float micAmplitude;
uniform float u_sub, u_bass, u_lowmid, u_mid, u_highmid, u_presence, u_brilliance, u_rms;
uniform vec2 touch;
uniform int pointerCount;
uniform vec3 pointers[10];
uniform vec2 surfaceResolution;
uniform bool screenLocked;
uniform float u_ms_source, u_ms_change, u_ms_next_in, u_ms_drop_in, u_ms_repeat;
uniform vec4 u_ms_cur_a, u_ms_cur_b, u_ms_sec_a, u_ms_sec_b, u_ms_next_a, u_ms_next_b;
uniform highp sampler2D bands;
uniform highp sampler2D vel;
uniform highp sampler2D vtmp;
uniform highp sampler2D worm;
uniform highp sampler2D ctl;
uniform highp sampler2D vort;

uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
float rnd(inout uint s) { s = pcg(s); return float(s) * (1.0 / 4294967296.0); }
float hash11(float x) { return fract(sin(x * 127.1 + 311.7) * 43758.5453); }
vec2 glassUV(vec2 px) { return (2.0 * px - surfaceResolution) / max(surfaceResolution.y, 1.0); }
vec2 planePt(vec2 px, float cz) { return glassUV(px) * ((cz + CAM_D) / FOCAL); }
vec3 glassWorld(vec2 px, float cz) { return transpose(camRot()) * vec3(planePt(px, cz), cz); }

ivec2 to2D(ivec3 p) {
	return ivec2(p.x + (p.z & 7) * 64, p.y + (p.z >> 3) * 64);
}
ivec3 to3D(ivec2 uv) {
	return ivec3(uv.x & 63, uv.y & 63, (uv.x >> 6) + ((uv.y >> 6) << 3));
}
bool outside(ivec3 p) {
	return any(lessThan(p, ivec3(0))) || any(greaterThanEqual(p, ivec3(FW_N)));
}
bool isFluidCell(ivec3 p) {
	return all(greaterThan(p, ivec3(0))) && all(lessThan(p, ivec3(FW_N - 1)));
}
vec3 cellPos(ivec3 p) {
	return ((vec3(p) + 0.5) / float(FW_N) * 2.0 - 1.0) * FW_H;
}
vec3 toUVW(vec3 x) {
	return x / (2.0 * FW_H) + 0.5;
}
const ivec3 NB[6] = ivec3[6](
	ivec3(1, 0, 0), ivec3(-1, 0, 0),
	ivec3(0, 1, 0), ivec3(0, -1, 0),
	ivec3(0, 0, 1), ivec3(0, 0, -1)
);

vec4 sample3D(highp sampler2D tex, vec3 uvw) {
	vec3 g = clamp(uvw, 0.0, 1.0) * float(FW_N) - 0.5;
	ivec3 p0 = ivec3(floor(g));
	vec3 f = g - vec3(p0);
	ivec3 c0 = clamp(p0, ivec3(0), ivec3(FW_N - 1));
	ivec3 c1 = clamp(p0 + ivec3(1), ivec3(0), ivec3(FW_N - 1));

	vec4 v000 = texelFetch(tex, to2D(ivec3(c0.x, c0.y, c0.z)), 0);
	vec4 v100 = texelFetch(tex, to2D(ivec3(c1.x, c0.y, c0.z)), 0);
	vec4 v010 = texelFetch(tex, to2D(ivec3(c0.x, c1.y, c0.z)), 0);
	vec4 v110 = texelFetch(tex, to2D(ivec3(c1.x, c1.y, c0.z)), 0);
	vec4 v001 = texelFetch(tex, to2D(ivec3(c0.x, c0.y, c1.z)), 0);
	vec4 v101 = texelFetch(tex, to2D(ivec3(c1.x, c0.y, c1.z)), 0);
	vec4 v011 = texelFetch(tex, to2D(ivec3(c0.x, c1.y, c1.z)), 0);
	vec4 v111 = texelFetch(tex, to2D(ivec3(c1.x, c1.y, c1.z)), 0);

	vec4 v00 = mix(v000, v100, f.x), v10 = mix(v010, v110, f.x);
	vec4 v01 = mix(v001, v101, f.x), v11 = mix(v011, v111, f.x);
	return mix(mix(v00, v10, f.y), mix(v01, v11, f.y), f.z);
}

#define C_TPREV    64
#define C_DT       65
#define C_KICK     66
#define C_ODFAVG   67
#define C_PRES     68
#define C_BRIL     69
#define C_DRUMHI   70
#define C_MICAVG   71
#define C_MICFLR   72
#define C_MICK     73
#define C_MICPRV   74
#define C_EMIT_ST  75
#define C_EMIT_N0  76
#define C_EMIT_N1  77
#define C_INIT     78
#define C_LOW0     80
#define C_SW_P     110
#define C_SW_DOWN  113
#define C_SW_GRIP  114
#define C_SW_W     115
#define C_CEN_Y    118
#define C_SW_ANG   119
#define C_SW_LOCK  120
#define C_TDOWN    128
#define C_TX0      129
#define C_TY0      130
#define C_TXP      131
#define C_TYP      132
#define C_TT0      133
#define C_TSWIPE   134
#define C_HOLD     135
#define C_HVX      136
#define C_HVY      137
#define C_FLX      138
#define C_FLY      139
#define C_KW       140
#define C_KX       141
#define C_KY       142
#define C_KZ       143
#define C_K2W      144
#define C_K2X      145
#define C_K2Y      146
#define C_K2Z      147
#define C_PCPREV   148
#define C_PCMAX    149
#define C_CX       150
#define C_CY       151
#define C_CVX      152
#define C_CVY      153
#define C_TAPK     154
#define C_SLW      155
#define C_SLA      156
#define C_SLB      159
#define C_SLV      162
#define C_G2MODE   167
#define C_G2W      168
#define C_G2USED   169
#define C_P0X      170
#define C_P0Y      171
#define C_P1X      172
#define C_P1Y      173
#define C_RMX      174
#define C_RMY      175
#define C_RR       176
#define C_RUX      177
#define C_RUY      178
#define C_RPH      179
#define C_GCX      180
#define C_GCY      181
#define C_GCZ      182
#define C_GR       183
#define C_CMODE    184
#define C_CBLINK   185
#define C_SPLITT   186
#define C_SPLITS   187
#define C_SLHT     188

#define SWIPE_PX     40.0
#define TAP_MAX_S    0.45
#define HOLD_S       0.30
#define CEN_PERIOD   3.0
#define CEN_DAMP     0.85
#define CEN_VMAX     0.45
#define HOLD_PERIOD  1.2
#define HOLD_VMAX    1.2
#define CEN_LIM_X    0.45
#define CEN_LIM_Y    0.75
#define FLING_SMOOTH 0.35
#define FLING_K      1.0
#define FLING_MAX    3.0
#define FLING_MIN    0.25
#define FLING_TAU    0.40
#define KNOCK_R      0.22
#define KNOCK_Z      -0.25
#define KNOCK_V      2.5
#define KNOCK_RATE   12.0
#define KNOCK_TAU    0.25
#define TAP_KICK     1.0
#define THROW_ON     1
#define G2_WINDOW    0.15
#define RING_PX      550.0
#define RING_BEATS   2.0
#define RING_MIN     0.06
#define RING_SCALE   1.7
#define RING_MAX     1.20
#define G2_TAU_IN    0.15
#define G2_TAU_OUT   0.50
#define GLOBE_MIN    0.12
#define GLOBE_MAX    1.30
#define TAP3_MAX_S   0.60
#define SWITCH_BLINK 0.25
#define SLICE_R      0.07
#define SLICE_Z      0.0
#define SLICE_VMAX   4.0
#define SLICE_RATE   14.0
#define SLICE_TAU    0.08
#define MID_R_FRAC   0.125
#define SPLIT_BEATS  4.0
#define SPLIT_Y      1.10
#define SPLIT_KEEP   0.30
#define SPLIT_IN     0.25
#define SPLIT_OUT    1.0
#define SLICE_HAWK_V   1.0
#define SLICE_HAWK_GAP 0.35

#define SWIRL_R      1.0
#define SWIRL_RATE   6.0
#define SWIRL_VMAX   2.5
#define SWIRL_OMMAX  12.0
#define SWIRL_RELEASE 0.15
#define GLOBUS_SENSE (+1.0)

#define SENSOR_UV    vec2(0.0, -0.465)
#define CEN_TAU      0.5
#define WHIRL_R      1.2
#define WHIRL_CORE   0.25
#define WHIRL_SINK   0.3

#define PLANE_ROT  1.5707963
#define SQUISH_X   0.45
#define FIG_ROT    0.0

#define FORM_DEMO_BEATS  8.0
#define FORM_MORPH_BEATS 2.0
#define PRED_BACK        6.0
#define PRED_FWD         6.0
#define SPECIAL_E        0.80
#define SPECIAL_W        0.50
#define SPECIAL_I        0.30
#define SCATTER_BEATS    1.0
#define SCATTER_IMPACT   0.5
#endif

==compute init 524288==
layout(local_size_x = 256) in;
layout(rgba32f, binding = part_UNIT) uniform writeonly highp image2D partImg;
layout(r32f, binding = ctl_UNIT) uniform readonly highp image2D ctlImg;
void main() {
	int i = int(gl_GlobalInvocationID.x);
	if (i >= FW_NP || imageLoad(ctlImg, ivec2(C_INIT, 0)).r > 0.5) return;
	ivec2 ip = ivec2(i & 511, i >> 9);
	imageStore(partImg, ip, vec4(0.0, 0.0, 0.0, -1.0));
}

==compute worm 1==
layout(local_size_x = 1) in;
layout(r32f, binding = ctl_UNIT) uniform highp image2D stImg;
layout(rgba32f, binding = worm_out_UNIT) uniform writeonly highp image2D worm_out;

#define KICK_LO        4
#define KICK_HI        27
#define KICK_GATE      0.15
#define KICK_K         1.6
#define KICK_MIN       0.10
#define KICK_TAU       0.12
#define MIC_AVG_TAU    1.0
#define MIC_ONSET      0.07
#define MIC_FLOOR_RISE 0.004
#define MIC_HEARD_LO   0.15
#define MIC_HEARD_HI   0.30
#define MIC_DYN_K      1.5

float C(int k) { return imageLoad(stImg, ivec2(k, 0)).r; }
void W(int k, float v) { imageStore(stImg, ivec2(k, 0), vec4(v)); }
float nd(float lin) { return clamp((20.0 * log(max(lin, 1e-9)) / log(10.0) + 60.0) / 60.0, 0.0, 1.0); }
float group(int lo, int hi) {
	float e = 0.0;
	for (int b = lo; b <= hi; b++) { float v = texelFetch(bands, ivec2(b, 0), 0).r; e += v * v; }
	return nd(sqrt(e));
}

#define ORBIT_BEATS   16.0
#define F_HI_TAU      0.30
#define F_SEP_ATT     0.08
#define F_SEP_REL     0.35
#define BIN_T         0.1
#define BIN_SEP0      0.035
#define BIN_SEP1      0.50
#define BIN_BOB       0.07
#define BIN_KICK_SEP  0.22
#define WORM_SCALE    1.15
#define WORM_MAX_R    1.25
#define N_MODES       5.0
#define SPLIT_E       0.85
#define SPLIT_I       0.20
#define SPLIT_EDGE    1.00
#define SPLIT_INNER   0.20
#define SPLIT_SWEEP   0.45
#define SPLIT_WARP    0.40
#define WELL_G        2.5
#define PUSH_RATE     0.5
#define PUSH_EARLY    4.0
#define KICK_WOBBLE   0.12
#define POCKET_MAX    0.6
#define POCKET_TAU    1.0
#define PIR_R         0.035
#define PIR_BEATS     2.0
#define HAT_REL       0.30
#define WINDUP_BEATS  8.0
#define WINDUP_K      0.6
#define SPONGE_NEAR   0.05
#define SPONGE_FAR    0.25
#define SPONGE_BEATS  4.0
#define SPONGE_PUSH   0.30
#define STRETCH_BEATS 8.0
#define SEP_NEAR      0.10
#define SEP_FAR_QUIET 0.70
#define SEP_FAR_LOUD  0.28
#define FIG_MIRROR_K  0.05
#define FIG_EDGE1     1.5707963
#define TILT_DEG      0.7853982
#define FIG_CAMLOCK   2.0
#define NAT_BEATS     2.0
#define NAT_CALM      1.0
#define FIG_LANE_N     4.0
#define FIG_LANE_Z0    0.175
#define FIG_LANE_DZ    0.30
#define FIG_LANE_BEATS 8.0
#define RUSH_BEATS    0.9
#define RUSH_SHORT    0.05
#define GOO_BEATS     6.0
#define GOO_R         0.035
#define GOO_W         22.0
#define GOO_Z         0.06
#define GOO_ATT       0.3
#define GOO_REL       2.0
#define STIR_BEATS    8.0
#define TORN_W        11.0
#define TORN_LEAD     3.14159
#define TORN_R0       0.05
#define TORN_R1       0.40
#define TORN_H        0.55
#define TORN_EXIT_H   2.2
#define TORN_EXIT_R   1.3
#define FLICKER_BEATS   1.0
#define FLICKER_STROBES 4.0
#define REPEL_A       0.06
#define REPEL_R0      0.18
#define REPEL_SWIRL   2.5
#define HAT_HIT       0.06
#define HAT_SPIN      0.7853982
#define SCATTER_SPD   3.0
#define HAWK_TAU      0.35
#define HAWK_KICK     0.25
#define HAWK_SPD      1.5
#define HAWK_SPIN     17.95
#define SWIM_SPLIT       0.03125
#define HELIX_H          0.62
#define HELIX_W          0.16
#define HELIX_CLIMB      0.80
#define FORM_HELIX_BEATS 5.0
#define RING_R           0.65
#define PUNCH_D          1.5
#define PUNCH_SHOT       0.28
#define SWIM_HELIX_TURNS 4.0
#define SWIM_FACE_TURNS  4.0

vec3 inside(vec3 p) { float r = length(p); return r > WORM_MAX_R ? p * (WORM_MAX_R / r) : p; }
float circDist(float a, float b) { float d = abs(a - b); return min(d, N_MODES - d); }

vec3 headOffAt(float bc, vec4 w, float wE, float pocket, vec3 sepX, float edge, float blade, float camLock, float laneZ, float loud) {
	float spin = 6.2831853 * bc / ORBIT_BEATS;
	float t = bc * BIN_T;
	float wave = 0.5 - 0.5 * cos(6.2831853 * bc / STRETCH_BEATS);
	float farR = mix(SEP_FAR_LOUD, SEP_FAR_QUIET, 1.0 - loud);
	float sep = mix(SEP_NEAR, farR, wave) + BIN_KICK_SEP * sepX.x;
	sep += SPONGE_PUSH * sepX.y;
	sep *= 1.0 - WINDUP_K * sepX.z;
	float rose = 1.0 - pocket * (0.5 - 0.5 * cos(spin * 3.0));
	vec3 offA = vec3(cos(spin) * sep, sin(spin * 1.5) * BIN_BOB, sin(spin) * sep);
	vec3 offB = vec3(cos(spin) * sep, sin(spin) * BIN_BOB, sin(2.0 * spin) * 0.5 * sep);
	float rr = sep * (1.0 - 0.6 * (0.5 - 0.5 * cos(3.0 * spin)));
	vec3 offC = vec3(cos(spin) * rr, sin(3.0 * spin) * 2.0 * BIN_BOB, sin(spin) * rr);
	float sMid = 0.5 * (SPLIT_EDGE + SPLIT_INNER), sRad = 0.5 * (SPLIT_EDGE - SPLIT_INNER);
	float sPh = spin + SPLIT_WARP * sin(spin);
	vec3 offD = vec3(sMid + sRad * cos(sPh), BIN_BOB * sin(spin * 1.5), SPLIT_SWEEP * sin(sPh));
	vec3 offE = vec3(SPLIT_SWEEP * sin(sPh), BIN_BOB * sin(spin * 1.5), sMid + sRad * cos(sPh));
	float wsum = max(w.x + w.y + w.z + w.w + wE, 1e-4);
	vec3 off = (offA * w.x + offB * w.y + offC * w.z) / wsum;
	off.xz *= rose;
	off += (offD * w.w + offE * wE) / wsum;
	vec3 nat = off;
	float tum = t * 1.2;
	nat.xy = mat2(cos(tum), -sin(tum), sin(tum), cos(tum)) * nat.xy;
	float tum2 = tum * 0.8;
	nat.yz = mat2(cos(tum2), -sin(tum2), sin(tum2), cos(tum2)) * nat.yz;
	vec2 pl = mat2(cos(PLANE_ROT), -sin(PLANE_ROT), sin(PLANE_ROT), cos(PLANE_ROT)) * vec2(off.x, off.z);
	vec3 camP = vec3(pl.x, pl.y, off.y);
	camP.z += laneZ;
	vec3 n = vec3(cos(blade), sin(blade), 0.0);
	float ce = cos(edge), se = sin(edge);
	camP = camP * ce + cross(n, camP) * se + n * dot(n, camP) * (1.0 - ce);
	vec3 locked = transpose(camRot()) * camP;
	return mix(nat, locked, clamp(camLock, 0.0, 1.0)) * WORM_SCALE;
}

float pickMode(vec4 A, vec4 B) {
	float calm   = 0.7 * (1.0 - A.x) + 0.3 * (1.0 - A.z);
	float groove = 0.6 * B.x + 0.2 * (1.0 - abs(A.x - 0.6));
	float busy   = 0.45 * A.z + 0.25 * B.y + 0.2 * A.w;
	float heavy  = 0.6 * A.y + 0.5 * max(B.w, 0.0) + 0.2 * A.x;
	float m = 0.0, best = calm;
	if (groove > best) { best = groove; m = 1.0; }
	if (busy   > best) { best = busy;   m = 2.0; }
	if (heavy  > best) { best = heavy;  m = 3.0; }
	if (m == 3.0 && A.x >= SPLIT_E && B.w >= SPLIT_I) m = 4.0;
	return m;
}

vec3 softCore(vec3 p) {
	float r = length(p);
	vec3 dir = r > 1e-4 ? p / r : vec3(1.0, 0.0, 0.0);
	vec3 tanV = normalize(cross(vec3(0.0, 1.0, 0.0), dir));
	return p + (dir + REPEL_SWIRL * tanV) * (REPEL_A * exp(-r / REPEL_R0));
}
vec3 swimSlot(int sh, float u, float f, float bc) {
	float ud = (u - SWIM_SPLIT) / (1.0 - SWIM_SPLIT);
	if (sh == 1) {
		if (u < SWIM_SPLIT) {
			float t  = u / SWIM_SPLIT;
			float y  = mix(HELIX_H, -HELIX_H, t);
			float sx = HELIX_W * (f < 0.5 ? -1.0 : 1.0);
			return vec3(sx, y, 0.0);
		}
		float uEnd = HELIX_CLIMB;
		if (u < uEnd) {
			float ud2 = (u - SWIM_SPLIT) / (uEnd - SWIM_SPLIT);
			float y  = mix(-HELIX_H, HELIX_H, ud2);
			float rr = HELIX_W * (1.0 - smoothstep(0.85, 1.0, ud2));
			float ph = 6.2831853 * SWIM_HELIX_TURNS * ud2 + (f < 0.5 ? 0.0 : 3.14159);
			return vec3(rr * cos(ph), y, rr * sin(ph));
		}
		return vec3(0.0, mix(HELIX_H, -HELIX_H, (u - uEnd) / (1.0 - uEnd)), 0.0);
	}
	if (sh == 2) {
		if (u < SWIM_SPLIT) return vec3(0.0);
		float a = 6.2831853 * (0.5 * ud + (f < 0.5 ? 0.0 : 0.5));
		return vec3(0.0, cos(a), sin(a)) * RING_R;
	}
	if (sh == 3) {
		if (u < SWIM_SPLIT) return vec3(0.0);
		float z  = mix(-RING_R, RING_R, ud);
		float ph = 6.2831853 * SWIM_FACE_TURNS * ud + (f < 0.5 ? 0.0 : 3.14159);
		return vec3(HELIX_W * cos(ph), HELIX_W * sin(ph), z);
	}
	if (sh == 4) {
		float ab = (u - SWIM_SPLIT) / (1.0 - SWIM_SPLIT);
		float tri = ab < 2.0 * PUNCH_SHOT ? (ab < PUNCH_SHOT ? ab / PUNCH_SHOT : (2.0 * PUNCH_SHOT - ab) / PUNCH_SHOT) : 0.0;
		return vec3(0.0, 0.0, (f < 0.5 ? -1.0 : 1.0) * PUNCH_D * tri);
	}
	if (sh == 5) {
		float ab = (u - SWIM_SPLIT) / (1.0 - SWIM_SPLIT);
		float tri = ab < 2.0 * PUNCH_SHOT ? (ab < PUNCH_SHOT ? ab / PUNCH_SHOT : (2.0 * PUNCH_SHOT - ab) / PUNCH_SHOT) : 1.0;
		return vec3(0.0, 0.0, (f < 0.5 ? -1.0 : 1.0) * PUNCH_D * tri);
	}
	if (sh == 6) {
		float sx = HELIX_W * (f < 0.5 ? -1.0 : 1.0);
		if (u < SWIM_SPLIT) return vec3(sx * (u / SWIM_SPLIT), 0.0, 0.0);
		if (u < 0.5) {
			float t = (u - SWIM_SPLIT) / (0.5 - SWIM_SPLIT);
			return vec3(sx, mix(0.0, -HELIX_H, t), 0.0);
		}
		float t = (u - 0.5) / 0.5;
		return vec3(sx, mix(-HELIX_H, HELIX_H, t), 0.0);
	}
	if (sh == 7) {
		float x  = mix(-RING_R, RING_R, u);
		float sy = 0.16 * (f < 0.5 ? 1.0 : -1.0);
		return vec3((f < 0.5 ? x : -x), sy, 0.0);
	}
	if (sh == 8) {
		float a = 6.2831853 * u + (f < 0.5 ? 0.0 : 3.14159);
		float r = 1.0 - 0.5 * (0.5 - 0.5 * cos(7.0 * a));
		return vec3(cos(a) * r, sin(a) * r, 0.0) * (RING_R * 0.7 * (f < 0.5 ? 1.0 : 0.82));
	}
	if (sh == 9) {
		float a  = 6.2831853 * u;
		float ph = 6.2831853 * 4.0 * u + (f < 0.5 ? 0.0 : 3.14159);
		vec3  nrm = vec3(-sin(a), cos(a), 0.0);
		return vec3(cos(a), sin(a), 0.0) * (RING_R * 0.6 * (f < 0.5 ? 1.0 : 0.82))
		     + (cos(ph) * nrm + sin(ph) * vec3(0.0, 0.0, 1.0)) * HELIX_W;
	}
	if (sh == 10) {
		float a = 6.2831853 * u + (f < 0.5 ? 0.0 : 3.14159);
		return vec3(cos(a), 0.72 * sin(a), 0.0) * (RING_R * (f < 0.5 ? 1.0 : 0.82));
	}
	if (sh == 11) {
		float a = 6.2831853 * u + (f < 0.5 ? 0.0 : 1.5707963);
		return vec3(sin(2.0 * a), sin(3.0 * a + 0.7854), 0.14 * cos(5.0 * a))
		     * (RING_R * 0.68 * (f < 0.5 ? 1.0 : 0.86));
	}
	if (sh == 12) {
		float a = 6.2831853 * u + (f < 0.5 ? 0.0 : 0.6283185);
		vec2  p = 0.58 * vec2(cos(2.0 * a), sin(2.0 * a))
		        + 0.38 * vec2(cos(3.0 * a), -sin(3.0 * a));
		return vec3(p, 0.10 * sin(5.0 * a)) * (RING_R * 0.78 * (f < 0.5 ? 1.0 : 0.85));
	}
	if (sh == 13) {
		float a = 6.2831853 * u + (f < 0.5 ? 0.0 : 3.14159);
		float r = 0.62 + 0.32 * cos(3.0 * a);
		return vec3(r * cos(2.0 * a), r * sin(2.0 * a), 0.16 * sin(3.0 * a)) * (RING_R * 0.75);
	}
	if (sh == 14) {
		float a = 6.2831853 * u + (f < 0.5 ? 0.0 : 3.14159);
		float r = 0.48 - 0.22 * cos(4.0 * a) + 0.26 * pow(sin(2.0 * a), 2.0);
		return vec3(sin(a) * r, cos(a) * (r + 0.18 * cos(2.0 * a)), 0.10 * sin(4.0 * a))
		     * (RING_R * 1.05 * (f < 0.5 ? 1.0 : 0.84));
	}
	if (sh == 15) {
		float a = 6.2831853 * u + (f < 0.5 ? 0.0 : 0.7853982);
		vec2  p = 0.68 * vec2(cos(a), sin(a)) - 0.28 * vec2(cos(5.0 * a), sin(5.0 * a));
		return vec3(p, 0.12 * cos(4.0 * a)) * (RING_R * 0.74 * (f < 0.5 ? 1.0 : 0.78));
	}
	if (u < SWIM_SPLIT) return vec3(0.0);
	float a = 6.2831853 * (0.5 * ud + (f < 0.5 ? 0.0 : 0.5));
	return vec3(cos(a), 0.0, sin(a)) * RING_R;
}

float figBeats(float id) {
	return (abs(id - 1.0) < 0.5) ? FORM_HELIX_BEATS : FORM_DEMO_BEATS;
}

int pickFigure(vec4 A, vec4 B, float R, float spd, float cur, float salt, bool fireSpecial, float repeat, float lastSp, float win) {
	if (fireSpecial) {
		float sc[8];
		sc[0] = 0.8 * A.y + 0.4 * (1.0 - A.z);
		sc[1] = 0.7 * A.z + 0.5 * A.x;
		sc[2] = 0.8 * A.y + 0.4 * B.x;
		sc[3] = 0.7 * A.z + 0.5 * A.w;
		sc[4] = 1.1 * B.w + 0.4 * A.x;
		sc[5] = 0.9 * B.w + 0.5 * B.z;
		sc[6] = 0.8 * A.y + 0.7 * B.w;
		sc[7] = 0.7 * B.x + 0.6 * A.z;
		int bestSp = 1; float bs = -1e9;
		for (int i = 0; i < 8; i++) {
			if (i == 6) continue;
			float noRep = (abs(float(i) - lastSp) < 0.5 || abs(float(i) - cur) < 0.5) ? -1.5 : 0.0;
			float v = sc[i] + 0.45 * hash11(salt * 13.7 + float(i)) + noRep;
			if (v > bs) { bs = v; bestSp = i; }
		}
		return bestSp;
	}
	float mus[8];
	mus[0] = 0.6 * A.z + 0.4 * (1.0 - A.x);
	mus[1] = 0.6 * B.x + 0.5 * A.z;
	mus[2] = 0.8 * (1.0 - A.z) + 0.4 * (1.0 - B.w);
	mus[3] = 0.7 * A.z + 0.5 * B.y;
	mus[4] = 0.7 * A.w + 0.5 * B.x;
	mus[5] = 0.6 * B.z + 0.5 * A.y;
	mus[6] = 0.6 * B.y + 0.5 * A.w;
	mus[7] = 0.6 * A.x + 0.5 * A.z;
	int best = 8; float bs = -1e9;
	for (int n = 0; n < 8; n++) {
		int id = 8 + n;
		float samePen = (abs(float(id) - cur) < 0.5) ? -1.2 : 0.0;
		float v = 1.6 * mus[n] + 0.45 * hash11(salt * 7.3 + float(n)) + samePen;
		if (v > bs) { bs = v; best = id; }
	}
	return best;
}

void main() {
	for (int k = 0; k < 8; k++) imageStore(worm_out, ivec2(k, 0), texelFetch(worm, ivec2(k, 0), 0));
	if (C(C_INIT) < 0.5) {
		W(C_INIT, 1.0);
		W(38, 8.0); W(39, 10.0);
	}
	float iTimeDelta = clamp(time - C(C_TPREV), 1e-4, 0.05);
	W(C_TPREV, time); W(C_DT, iTimeDelta);

	int figNow = 0;
	bool gestHawk = false;
	float hold = C(C_HOLD);
	float tapKick = C(C_TAPK) * exp(-iTimeDelta / KICK_TAU);
	float g2mode = C(C_G2MODE), g2w = C(C_G2W);
	{
		float dt0 = iTimeDelta;
		vec2 hv = vec2(C(C_HVX), C(C_HVY));
		vec2 fling = vec2(C(C_FLX), C(C_FLY)) * exp(-dt0 / FLING_TAU);
		float knockW = C(C_KW) * exp(-dt0 / KNOCK_TAU);
		float knock2W = C(C_K2W) * exp(-dt0 / KNOCK_TAU);
		int pc = pointerCount, pcPrev = int(C(C_PCPREV) + 0.5);
		float sliceW = C(C_SLW) * exp(-dt0 / SLICE_TAU);
		float pcMax = C(C_PCMAX), g2used = C(C_G2USED);
		if (pc >= 2 && pc > pcPrev) {
			vec3 g2 = glassWorld(pointers[min(pc, 10) - 1].xy, KNOCK_Z);
			W(C_K2X, g2.x); W(C_K2Y, g2.y); W(C_K2Z, g2.z); knock2W = 1.0; tapKick = TAP_KICK;
		}
		W(C_PCPREV, float(pc));
		if (pc > 0) {
			if (C(C_TDOWN) < 0.5) {
				W(C_TDOWN, 1.0); W(C_TX0, touch.x); W(C_TY0, touch.y); W(C_TXP, touch.x); W(C_TYP, touch.y);
				W(C_TSWIPE, 0.0); W(C_TT0, time); hold = 0.0; hv = vec2(0.0); pcMax = 0.0; g2used = 0.0;
				vec3 g = glassWorld(touch, KNOCK_Z);
				W(C_KX, g.x); W(C_KY, g.y); W(C_KZ, g.z); knockW = 1.0; tapKick = TAP_KICK;
			} else {
				vec2 gNow = planePt(touch, 0.0), gPrev = planePt(vec2(C(C_TXP), C(C_TYP)), 0.0);
				hv = mix(hv, (gNow - gPrev) / dt0, FLING_SMOOTH);
				if (C(C_TSWIPE) > 0.5 && hold < 0.5 && g2used < 0.5) {
					vec3 a = glassWorld(vec2(C(C_TXP), C(C_TYP)), SLICE_Z), b = glassWorld(touch, SLICE_Z);
					vec3 bv = (b - a) / dt0; float bl = length(bv);
					if (bl > SLICE_VMAX) bv *= SLICE_VMAX / bl;
					W(C_SLA, a.x); W(C_SLA + 1, a.y); W(C_SLA + 2, a.z);
					W(C_SLB, b.x); W(C_SLB + 1, b.y); W(C_SLB + 2, b.z);
					W(C_SLV, bv.x); W(C_SLV + 1, bv.y); W(C_SLV + 2, bv.z);
					sliceW = 1.0;
					if (bl > SLICE_HAWK_V && time - C(C_SLHT) >= SLICE_HAWK_GAP) { gestHawk = true; W(C_SLHT, time); }
				}
				W(C_TXP, touch.x); W(C_TYP, touch.y);
			}
			if (pc >= 2 && pcMax < 1.5 && hold < 0.5 && time - C(C_TT0) < G2_WINDOW) {
				float spreadPx = length(pointers[0].xy - pointers[1].xy);
				g2mode = spreadPx < RING_PX ? 1.0 : 2.0; g2used = 1.0; g2w = 0.0;
				W(C_P0X, pointers[0].x); W(C_P0Y, pointers[0].y); W(C_P1X, pointers[1].x); W(C_P1Y, pointers[1].y);
				W(C_RPH, 0.0); W(C_TSWIPE, 0.0);
			}
			pcMax = max(pcMax, float(pc));
			if (pc >= 3) { g2mode = 0.0; g2used = 1.0; }
			if (g2used > 0.5) hold = 0.0;
			else if (pcMax < 1.5) {
				if (hold < 0.5 && length(touch - vec2(C(C_TX0), C(C_TY0))) > SWIPE_PX) W(C_TSWIPE, 1.0);
				if (C(C_TSWIPE) < 0.5 && time - C(C_TT0) >= HOLD_S) hold = 1.0;
			}
		} else if (C(C_TDOWN) > 0.5) {
			W(C_TDOWN, 0.0);
			if (pcMax >= 3.5 && time - C(C_TT0) < TAP3_MAX_S) {
				W(C_CMODE, mod(floor(C(C_CMODE) + 0.5) + 1.0, float(N_CMODES)));
				W(C_CBLINK, SWITCH_BLINK);
			} else if (pcMax >= 2.5 && time - C(C_TT0) < TAP3_MAX_S) {
				figNow = 2;
			}
			if (hold < 0.5 && C(C_TSWIPE) < 0.5 && pcMax < 1.5 && g2used < 0.5) {
				vec2 sr = surfaceResolution;
				if (length(vec2(C(C_TX0), C(C_TY0)) - 0.5 * sr) < MID_R_FRAC * sr.x) { W(C_SPLITT, SPLIT_BEATS); W(C_SPLITS, 0.0); }
				else figNow = 1;
			}
			if (hold > 0.5) {
				hold = 0.0;
				float hs = length(hv) * FLING_K;
				if (THROW_ON > 0 && hs > FLING_MIN) fling = hv * FLING_K * (hs > FLING_MAX ? FLING_MAX / hs : 1.0);
			}
		}
		bool g2on = g2used > 0.5 && pc >= 2 && g2mode > 0.5;
		if (g2on) {
			vec2 a0 = planePt(pointers[0].xy, 0.0), a1 = planePt(pointers[1].xy, 0.0);
			vec2 mid = 0.5 * (a0 + a1), ax2 = a1 - a0;
			float hspan = 0.5 * length(ax2);
			W(C_RMX, mid.x); W(C_RMY, mid.y);
			if (g2mode < 1.5) {
				W(C_RR, clamp(hspan * RING_SCALE, RING_MIN, RING_MAX));
				if (dot(ax2, ax2) > 1e-6) { ax2 = normalize(ax2); W(C_RUX, ax2.x); W(C_RUY, ax2.y); }
				else if (abs(C(C_RUX)) + abs(C(C_RUY)) < 0.5) { W(C_RUX, 1.0); W(C_RUY, 0.0); }
			} else {
				vec3 gc = glassWorld(0.5 * (pointers[0].xy + pointers[1].xy), 0.0);
				W(C_GCX, gc.x); W(C_GCY, gc.y); W(C_GCZ, gc.z); W(C_GR, clamp(hspan, GLOBE_MIN, GLOBE_MAX));
			}
			g2w += (1.0 - g2w) * (1.0 - exp(-dt0 / G2_TAU_IN));
		} else g2w *= exp(-dt0 / G2_TAU_OUT);
		W(C_HOLD, hold); W(C_HVX, hv.x); W(C_HVY, hv.y); W(C_FLX, fling.x); W(C_FLY, fling.y);
		W(C_KW, knockW); W(C_K2W, knock2W); W(C_PCMAX, pcMax); W(C_TAPK, tapKick); W(C_SLW, sliceW);
		W(C_G2MODE, g2mode); W(C_G2W, g2w); W(C_G2USED, g2used);
		vec2 cenLim = vec2(CEN_LIM_X, CEN_LIM_Y);
		bool ringOn = g2on && g2mode < 1.5;
		bool rush = (hold > 0.5 || ringOn) && !screenLocked;
		vec2 tgt = !rush ? vec2(0.0) : clamp(ringOn ? vec2(C(C_RMX), C(C_RMY)) : planePt(touch, 0.0), -cenLim, cenLim);
		vec2 cen = vec2(C(C_CX), C(C_CY)), cv = vec2(C(C_CVX), C(C_CVY));
		float om = 6.2831853 / (rush ? HOLD_PERIOD : CEN_PERIOD);
		cv += (om * om * (tgt - cen) - 2.0 * CEN_DAMP * om * cv) * dt0;
		float cs = length(cv), cmax = rush ? HOLD_VMAX : CEN_VMAX;
		if (cs > cmax) cv *= cmax / cs;
		cen = clamp(cen + (cv + fling) * dt0, -cenLim, cenLim);
		W(C_CX, cen.x); W(C_CY, cen.y); W(C_CVX, cv.x); W(C_CVY, cv.y);

		{
			vec3 omg = vec3(C(C_SW_W), C(C_SW_W + 1), C(C_SW_W + 2));
			float grip = C(C_SW_GRIP);
			if (g2on && g2mode > 1.5) {
				vec2 a0 = planePt(pointers[0].xy, 0.0), a1 = planePt(pointers[1].xy, 0.0);
				vec2 b0 = planePt(vec2(C(C_P0X), C(C_P0Y)), 0.0), b1 = planePt(vec2(C(C_P1X), C(C_P1Y)), 0.0);
				vec2 tr = 0.5 * (a0 + a1) - 0.5 * (b0 + b1);
				vec2 da = a1 - a0, db = b1 - b0;
				float th = atan(da.y, da.x) - atan(db.y, db.x);
				th -= 6.2831853 * floor(th / 6.2831853 + 0.5);
				float GR = C(C_GR);
				vec3 wc = GLOBUS_SENSE * vec3(tr.y, -tr.x, 0.0) / (GR * dt0) + vec3(0.0, 0.0, th / dt0);
				float wl = length(wc);
				if (wl > SWIRL_OMMAX) wc *= SWIRL_OMMAX / wl;
				omg += (transpose(camRot()) * wc - omg) * (1.0 - exp(-dt0 / 0.05));
				grip = 1.0;
			} else grip *= exp(-dt0 / SWIRL_RELEASE);
			if (pc >= 2) { W(C_P0X, pointers[0].x); W(C_P0Y, pointers[0].y); W(C_P1X, pointers[1].x); W(C_P1Y, pointers[1].y); }
			W(C_SW_W, omg.x); W(C_SW_W + 1, omg.y); W(C_SW_W + 2, omg.z); W(C_SW_GRIP, grip);
		}
	}
	vec2 cenPlane = vec2(C(C_CX), C(C_CY));
	{
		float lockF = screenLocked ? 1.0 : 0.0;
		float cy = C(C_CEN_Y);
		cy += (SENSOR_UV.y * lockF - cy) * (1.0 - exp(-iTimeDelta / CEN_TAU));
		W(C_CEN_Y, cy);
		imageStore(worm_out, ivec2(6, 0), vec4(0.0, cy, 0.0, 1.0));
	}

	float pres = u_presence, bril = u_brilliance;
	float drumHi = max(pres, bril);
	float rms = u_rms;
	float flux = 0.0;
	for (int b = KICK_LO; b <= KICK_HI; b++) {
		float x = nd(texelFetch(bands, ivec2(b, 0), 0).r);
		flux += max(0.0, x - C(C_LOW0 + b - KICK_LO));
		W(C_LOW0 + b - KICK_LO, x);
	}
	float avg = C(C_ODFAVG);
	W(C_ODFAVG, avg + (flux - avg) * (1.0 - exp(-iTimeDelta / 1.5)));
	bool clocked = beatConf > 0.25 && beatBpm > 30.0;
	float offB = min(beatPhase, 1.0 - beatPhase);
	bool onGrid = !clocked || offB < KICK_GATE;
	float kick = C(C_KICK) * exp(-iTimeDelta / KICK_TAU);
	if (flux > max(KICK_MIN, KICK_K * avg) && onGrid) kick = 1.0;
	W(C_KICK, kick);
	float u_kick = kick;

	float jdspRms = rms;
	float micDb = nd(micAmplitude);
	float mAvg = C(C_MICAVG);
	W(C_MICAVG, mAvg + (micDb - mAvg) * (1.0 - exp(-iTimeDelta / MIC_AVG_TAU)));
	float mFloor = C(C_MICFLR);
	mFloor = (mFloor <= 0.0 || micDb < mFloor) ? micDb : mFloor + iTimeDelta * MIC_FLOOR_RISE;
	W(C_MICFLR, mFloor);
	float micHeard = smoothstep(MIC_HEARD_LO, MIC_HEARD_HI, micDb - mFloor) * (1.0 - smoothstep(0.08, 0.20, jdspRms));
	float mk = C(C_MICK) * exp(-iTimeDelta / KICK_TAU);
	if (micDb - mAvg > MIC_ONSET && micDb > C(C_MICPRV) && onGrid) mk = 1.0;
	W(C_MICK, mk); W(C_MICPRV, micDb);
	if (micHeard > 0.0) u_kick = mix(u_kick, mk, micHeard);
	u_kick = max(u_kick, tapKick);

	float heard = smoothstep(0.08, 0.20, jdspRms);
	float cached = u_ms_source > 0.5 ? 1.0 : 0.0;
	float cacheDrive = cached * (1.0 - heard);
	if (cacheDrive > 0.0) {
		float kStr = smoothstep(0.25, 0.7, u_ms_cur_a.x) * mix(0.6, 1.0, u_ms_cur_a.y);
		float kSyn = kStr * exp(-fract(beatCount) * 60.0 / max(beatBpm, 30.0) / KICK_TAU);
		u_kick = mix(u_kick, kSyn, cacheDrive * (1.0 - micHeard));
		float hiC = 0.6 * u_ms_cur_a.w + 0.4 * u_ms_cur_a.z;
		pres = mix(pres, hiC, cacheDrive);
		bril = mix(bril, hiC, cacheDrive);
		drumHi = max(pres, bril);
	}

	float u_beat_bpm = clocked ? beatBpm : 120.0;
	float u_tbeat = 60.0 / max(u_beat_bpm, 30.0);
	float u_beat_count = beatCount;

	float dt = iTimeDelta;
	float dB = dt / max(u_tbeat, 0.2);

	float sb = imageLoad(stImg, ivec2(2, 0)).r;
	float kickOnset = (u_kick > sb + HAWK_KICK) ? 1.0 : 0.0;
	sb += (u_kick - sb) * (1.0 - exp(-dt / (u_kick > sb ? F_SEP_ATT : F_SEP_REL)));
	imageStore(stImg, ivec2(2, 0), vec4(sb));

	float hawkPrev = imageLoad(stImg, ivec2(15, 0)).r;
	float hawk = (kickOnset > 0.5 || gestHawk) ? HAWK_TAU : max(0.0, hawkPrev - dt);
	imageStore(stImg, ivec2(15, 0), vec4(hawk));
	float hawkAmt = smoothstep(0.0, HAWK_TAU, hawk);

	float pdir = imageLoad(stImg, ivec2(11, 0)).r; pdir = pdir < 0.0 ? -1.0 : 1.0;
	float scat = imageLoad(stImg, ivec2(12, 0)).r;
	float sseen = imageLoad(stImg, ivec2(14, 0)).r;
	if (u_ms_change != sseen) {
		sseen = u_ms_change;
		imageStore(stImg, ivec2(14, 0), vec4(sseen));
		if (u_ms_sec_b.w > SCATTER_IMPACT || u_ms_cur_b.w > SCATTER_IMPACT) { pdir = -pdir; scat = SCATTER_BEATS; }
	}
	float spd = 1.0 + SCATTER_SPD * smoothstep(0.0, SCATTER_BEATS, scat) + HAWK_SPD * hawkAmt;
	float pb = imageLoad(stImg, ivec2(13, 0)).r + pdir * spd * dB;
	imageStore(stImg, ivec2(13, 0), vec4(pb));
	imageStore(stImg, ivec2(11, 0), vec4(pdir));
	imageStore(stImg, ivec2(12, 0), vec4(max(0.0, scat - dB)));
	float hi = imageLoad(stImg, ivec2(0, 0)).r;
	hi += (max(pres, bril) - hi) * (1.0 - exp(-dt / F_HI_TAU));
	imageStore(stImg, ivec2(0, 0), vec4(hi));
	float spin = 6.2831853 * pb / ORBIT_BEATS;

	float state  = imageLoad(stImg, ivec2(3, 0)).r;
	float seen   = imageLoad(stImg, ivec2(4, 0)).r;
	float target = imageLoad(stImg, ivec2(5, 0)).r;
	if (u_ms_change != seen) {
		seen = u_ms_change;
		target = pickMode(u_ms_sec_a, u_ms_sec_b);
	}
	if (u_ms_source > 0.5 && u_ms_next_in > 0.0 && u_ms_next_in < PUSH_EARLY && u_ms_next_a.x >= 0.0)
		target = pickMode(u_ms_next_a, u_ms_next_b);
	float d = target - state;
	d -= N_MODES * floor(d / N_MODES + 0.5);
	if (abs(d) > 0.05) state += sign(d) * min(PUSH_RATE * dB, abs(d));
	else state += (-sin(state * 6.2831853) * WELL_G / 6.2831853) * dB;
	state += KICK_WOBBLE * u_kick * u_kick * u_kick * sin(spin * 2.0) * dB;
	state = mod(state, N_MODES);
	imageStore(stImg, ivec2(3, 0), vec4(state));
	imageStore(stImg, ivec2(4, 0), vec4(seen));
	imageStore(stImg, ivec2(5, 0), vec4(target));

	float wA = max(0.0, 1.0 - circDist(state, 0.0));
	float wB = max(0.0, 1.0 - circDist(state, 1.0));
	float wC = max(0.0, 1.0 - circDist(state, 2.0));
	float wD = max(0.0, 1.0 - circDist(state, 3.0));
	float wE = max(0.0, 1.0 - circDist(state, 4.0));

	float pocket = imageLoad(stImg, ivec2(7, 0)).r;
	pocket += (POCKET_MAX * u_ms_cur_a.z - pocket) * (1.0 - exp(-dt / POCKET_TAU));
	imageStore(stImg, ivec2(7, 0), vec4(pocket));
	float hat = imageLoad(stImg, ivec2(6, 0)).r;
	float hatHit = (drumHi > hat + HAT_HIT) ? 1.0 : 0.0;
	hat = max(drumHi, hat * exp(-dt / HAT_REL));
	imageStore(stImg, ivec2(6, 0), vec4(hat));
	float roll = imageLoad(stImg, ivec2(10, 0)).r + HAT_SPIN * hatHit + HAWK_SPIN * (hawk > 0.0 ? dt : 0.0);
	imageStore(stImg, ivec2(10, 0), vec4(roll));
	imageStore(worm_out, ivec2(4, 0), vec4(roll, 0.0, 0.0, 1.0));
	float windup = (u_ms_source > 0.5 && u_ms_drop_in > 0.0)
		? 1.0 - smoothstep(0.0, WINDUP_BEATS * u_tbeat, u_ms_drop_in) : 0.0;

	float t = pb * BIN_T;
	float loud = clamp(0.5 * u_ms_cur_a.x + 0.5 * sb, 0.0, 1.0);
	float sep0 = BIN_SEP0 + BIN_SEP1 * pow(sin(t * 0.5), 4.0) + BIN_KICK_SEP * sb;
	float press = imageLoad(stImg, ivec2(1, 0)).r;
	float close = 1.0 - smoothstep(SPONGE_NEAR, SPONGE_FAR, sep0);
	press += (close - press) * (1.0 - exp(-dB / SPONGE_BEATS));
	imageStore(stImg, ivec2(1, 0), vec4(press));

	float tiltPose = mod(u_ms_change, 4.0);
	float figEdge, figBlade;
	if      (tiltPose < 0.5) { figEdge =  TILT_DEG;  figBlade = 0.0; }
	else if (tiltPose < 1.5) { figEdge = -TILT_DEG;  figBlade = 0.0; }
	else if (tiltPose < 2.5) { figEdge =  FIG_EDGE1; figBlade = 0.0; }
	else                     { figEdge =  FIG_EDGE1; figBlade = FIG_EDGE1; }
	float calmNow  = (pickMode(u_ms_sec_a, u_ms_sec_b) < 0.5) ? 1.0 : 0.0;
	float camLockT = (FIG_CAMLOCK < 0.5) ? 0.0 : ((FIG_CAMLOCK < 1.5) ? 1.0 : 1.0 - NAT_CALM * calmNow);
	float figCamLock = imageLoad(stImg, ivec2(54, 0)).r;
	figCamLock += (camLockT - figCamLock) * (1.0 - exp(-dB / NAT_BEATS));
	imageStore(stImg, ivec2(54, 0), vec4(figCamLock));
	float lane  = mod(floor(pb / FIG_LANE_BEATS), FIG_LANE_N);
	float laneZ = FIG_LANE_Z0 + lane * FIG_LANE_DZ;
	vec3 fig = headOffAt(pb, vec4(wA, wB, wC, wD), wE, pocket, vec3(sb, press, windup), 0.0, 0.0, figCamLock, laneZ, loud);
	float pir = 6.2831853 * pb / PIR_BEATS;
	vec3 po = vec3(cos(pir), sin(pir), 0.0) * PIR_R * clamp(0.6 * hat + 0.4 * u_ms_cur_a.w, 0.0, 1.0);
	float tumble = t * 1.2;
	po.xy = mat2(cos(tumble), -sin(tumble), sin(tumble), cos(tumble)) * po.xy;
	fig += po;

	vec3 baseP = softCore(fig);
	vec3 p0 = baseP;
	float teamW = 0.0;
	vec3  p1t   = p0;
	float figActive = 0.0;
	{
		float figPhase = imageLoad(stImg, ivec2(53, 0)).r;
		float swA = imageLoad(stImg, ivec2(38, 0)).r, swB = imageLoad(stImg, ivec2(39, 0)).r;
		float swM = imageLoad(stImg, ivec2(40, 0)).r;
		if (figNow > 0) figPhase = 1.0;
		figPhase += dB / figBeats(swB);
		if (figPhase >= 1.0) {
			figPhase -= 1.0;
			float figN = imageLoad(stImg, ivec2(41, 0)).r + 1.0;
			imageStore(stImg, ivec2(41, 0), vec4(figN));

			float spCool    = max(0.0, imageLoad(stImg, ivec2(50, 0)).r - 1.0);
			float spSecSeen = imageLoad(stImg, ivec2(51, 0)).r;
			float lastSp    = imageLoad(stImg, ivec2(52, 0)).r;

			bool secChanged = (u_ms_change != spSecSeen);
			if (secChanged) imageStore(stImg, ivec2(51, 0), vec4(u_ms_change));

			bool heavySec   = u_ms_sec_a.x >= SPECIAL_E && (u_ms_sec_a.y >= SPECIAL_W || u_ms_sec_b.w >= SPECIAL_I);
			bool secPunch   = secChanged && (u_ms_sec_b.w >= 0.28 || (u_ms_sec_a.x >= 0.68 && u_ms_sec_a.y >= 0.48));
			bool phrasePeak = heavySec && (mod(figN, 3.0) < 0.5);
			bool livePeak   = (u_ms_cur_a.x * max(u_ms_cur_b.w, 0.0) + 0.35 * sb) > 0.68;

			bool fireSpecial = (figNow == 2) || (figNow == 0 && (spCool < 0.5) && (secPunch || phrasePeak || livePeak));

			swA = swB;
			swB = float(pickFigure(u_ms_sec_a, u_ms_sec_b, 1.0, 0.0, swB, figN, fireSpecial, u_ms_repeat, lastSp, figN));
			if (swB < 7.5) {
				spCool = 2.0;
				imageStore(stImg, ivec2(52, 0), vec4(swB));
			}
			imageStore(stImg, ivec2(50, 0), vec4(spCool));

			swM = 0.0;
			imageStore(stImg, ivec2(38, 0), vec4(swA));
			imageStore(stImg, ivec2(39, 0), vec4(swB));
		}
		imageStore(stImg, ivec2(53, 0), vec4(figPhase));
		float uu = figPhase;
		swM = min(1.0, swM + dB / FORM_MORPH_BEATS);
		imageStore(stImg, ivec2(40, 0), vec4(swM));
		float sm = smoothstep(0.0, 1.0, swM);
		bool isTeam = (swB > 3.5 && swB < 7.5);
		if (isTeam) sm = 1.0;
		vec3 s0 = mix(swimSlot(int(swA + 0.5), uu, 0.0, u_beat_count),
		              swimSlot(int(swB + 0.5), uu, 0.0, u_beat_count), sm);
		vec3 s1 = mix(swimSlot(int(swA + 0.5), uu, 1.0, u_beat_count),
		              swimSlot(int(swB + 0.5), uu, 1.0, u_beat_count), sm);
		{
			float fc = cos(FIG_ROT), fs = sin(FIG_ROT);
			s0.xy = mat2(fc, -fs, fs, fc) * s0.xy;
			s1.xy = mat2(fc, -fs, fs, fc) * s1.xy;
		}
		{
			vec3  tn  = vec3(cos(figBlade), sin(figBlade), 0.0);
			float tce = cos(figEdge), tse = sin(figEdge);
			s0 = s0 * tce + cross(tn, s0) * tse + tn * dot(tn, s0) * (1.0 - tce);
			s1 = s1 * tce + cross(tn, s1) * tse + tn * dot(tn, s1) * (1.0 - tce);
		}
		p0   = baseP + transpose(camRot()) * s0;
		p1t  = baseP + transpose(camRot()) * s1;
		teamW = 0.0;
		figActive = 0.0;
	}

	float rushT   = imageLoad(stImg, ivec2(24, 0)).r;
	float rushSee = imageLoad(stImg, ivec2(28, 0)).r;
	if (u_ms_change != rushSee) {
		imageStore(stImg, ivec2(28, 0), vec4(u_ms_change));
		if (u_ms_sec_b.w > SCATTER_IMPACT || u_ms_cur_b.w > SCATTER_IMPACT) {
			rushT = RUSH_BEATS;
			imageStore(stImg, ivec2(25, 0), vec4(p0.x));
			imageStore(stImg, ivec2(26, 0), vec4(p0.y));
			imageStore(stImg, ivec2(27, 0), vec4(p0.z));
		}
	}
	if (rushT > 0.0) {
		float u = clamp(1.0 - rushT / RUSH_BEATS, 0.0, 1.0);
		vec3 A = vec3(imageLoad(stImg, ivec2(25, 0)).r, imageLoad(stImg, ivec2(26, 0)).r, imageLoad(stImg, ivec2(27, 0)).r);
		vec3 Bm = -A;
		vec3 near = mix(A, Bm, 1.0 - RUSH_SHORT);
		p0 = mix(mix(A, near, smoothstep(0.0, 0.5, u)),
		         mix(near, Bm, smoothstep(0.5, 1.0, u)), step(0.5, u));
		teamW = 0.0; figActive = 0.0;
		imageStore(stImg, ivec2(24, 0), vec4(max(0.0, rushT - dB)));
	}

	float gooT     = imageLoad(stImg, ivec2(29, 0)).r;
	float dropSeen = imageLoad(stImg, ivec2(30, 0)).r;
	float dropIn   = u_ms_drop_in;
	if (dropSeen > 0.0 && dropSeen < 1.0 && (dropIn < 0.0 || dropIn > dropSeen + 1.0)) {
		float dc = imageLoad(stImg, ivec2(37, 0)).r + 1.0;
		imageStore(stImg, ivec2(37, 0), vec4(dc));
		if (mod(dc, 2.0) < 1.0) gooT = GOO_BEATS;
		else imageStore(stImg, ivec2(32, 0), vec4(STIR_BEATS));
	}
	imageStore(stImg, ivec2(30, 0), vec4(dropIn));
	if (gooT > 0.0) {
		float g = smoothstep(0.0, GOO_ATT, GOO_BEATS - gooT)
		        * (1.0 - smoothstep(GOO_BEATS - GOO_REL, GOO_BEATS, gooT));
		float ang = imageLoad(stImg, ivec2(31, 0)).r + GOO_W * g * dt;
		imageStore(stImg, ivec2(31, 0), vec4(ang));
		float R = GOO_R * g;
		p0 = transpose(camRot()) * vec3(R * cos(ang), R * sin(ang), GOO_Z * g);
		teamW = 0.0; figActive = 0.0;
		imageStore(stImg, ivec2(29, 0), vec4(max(0.0, gooT - dB)));
	}

	float stirT = imageLoad(stImg, ivec2(32, 0)).r;
	if (stirT > 0.0) {
		float clk = imageLoad(stImg, ivec2(34, 0)).r + dt;
		imageStore(stImg, ivec2(34, 0), vec4(clk));
		float p    = clamp(1.0 - stirT / STIR_BEATS, 0.0, 1.0);
		float gIn  = smoothstep(0.0, 1.0, STIR_BEATS - stirT);
		float rise = smoothstep(0.0, 0.62, p);
		float exit = smoothstep(0.64, 1.0, p);
		float y    = (rise - 0.5) * 2.0 * TORN_H * gIn + exit * TORN_EXIT_H;
		float rr   = (TORN_R0 + TORN_R1 * rise) * gIn + exit * TORN_EXIT_R;
		float th   = clk * TORN_W;
		p0  = transpose(camRot()) * vec3(rr * cos(th), y, rr * sin(th));
		p1t = transpose(camRot()) * vec3(rr * cos(th + TORN_LEAD), y, rr * sin(th + TORN_LEAD));
		teamW = 1.0;
		imageStore(stImg, ivec2(32, 0), vec4(max(0.0, stirT - dB)));
	}

	{
		float mode = clamp(C(C_CMODE), 0.0, float(N_CMODES - 1));
		float blink = max(C(C_CBLINK) - dt, 0.0);
		W(C_CBLINK, blink);
		float white = (rushT > 0.0 || gooT > 0.0 || blink > 0.0) ? 1.0 : 0.0;
		imageStore(worm_out, ivec2(5, 0), vec4(2.0 * mode, white, 2.0 * mode + 1.0, 1.0));
	}

	float ringW = (g2mode < 1.5 && !screenLocked) ? g2w : 0.0;
	float ringPh = C(C_RPH) + dB * 6.2831853 / RING_BEATS;
	W(C_RPH, ringPh);
	vec2 ringU = vec2(C(C_RUX), C(C_RUY));
	float ringR = C(C_RR);

	vec3 c0 = camRot() * p0;
	float splitT = C(C_SPLITT), splitS = C(C_SPLITS);
	if (splitT > 0.0 && abs(splitS) < 0.5) { splitS = c0.y >= 0.0 ? 1.0 : -1.0; W(C_SPLITS, splitS); }
	float splitW = screenLocked ? 0.0 : smoothstep(0.0, SPLIT_IN, SPLIT_BEATS - splitT) * smoothstep(0.0, SPLIT_OUT, splitT);
	W(C_SPLITT, max(0.0, splitT - dB));
	int emitN[2];
	for (int k = 0; k < 2; k++) {
		vec4 now = texelFetch(worm, ivec2(k, 0), 0);
		vec3 p;
		{
			vec3 c = (k == 0) ? c0 : (mix(-c0, camRot() * p1t, teamW));
			c.x *= SQUISH_X;
			c.z = mix(c.z, sqrt(c.z * c.z + FIG_MIRROR_K * FIG_MIRROR_K), figCamLock * (1.0 - figActive));
			if (splitW > 0.001) {
				float sy = (k == 0 ? 1.0 : -1.0) * splitS * SPLIT_Y;
				c = mix(c, vec3(c.xy * SPLIT_KEEP + vec2(0.0, sy), c.z), splitW);
			}
			if (ringW > 0.001) {
				float ph = ringPh + 3.14159 * float(k);
				c = mix(c, vec3(ringU * (cos(ph) * ringR), sin(ph) * ringR), ringW);
			}
			c.xy += cenPlane;
			p = transpose(camRot()) * c;
		}
		{
			vec3 cw = transpose(camRot()) * vec3(cenPlane, 0.0);
			p = cw + inside(p - cw);
			p = clamp(p, vec3(-0.9 * FW_H), vec3(0.9 * FW_H));
		}
		if (now.w > 0.5) { vec3 dp = p - now.xyz; float lim = HEAD_MAX_SPD * dt, dl = length(dp); if (dl > lim) p = now.xyz + dp * (lim / dl); }
		vec4 prev = now.w > 0.5 ? now : vec4(p, 1.0);
		imageStore(worm_out, ivec2(2 + k, 0), prev);
		imageStore(worm_out, ivec2(k, 0), vec4(p, 1.0));

		float dist = length(p - prev.xyz);
		float rate = clamp(max(FW_RATE, FW_DENS * dist / max(dt, 1e-4)), FW_RATE, FW_RATE_MAX);
		emitN[k] = max(1, int(round(rate * dt)));
	}
	int emitStart = int(C(C_EMIT_ST));
	W(C_EMIT_N0, float(emitN[0]));
	W(C_EMIT_N1, float(emitN[1]));
	W(C_EMIT_ST, float((emitStart + emitN[0] + emitN[1]) % FW_NP));
}

==compute adv_fwd vel==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = vtmp_UNIT) uniform writeonly highp image2D vtmpImg;
void main() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	if (uv.x >= 512 || uv.y >= 512) return;
	ivec3 p = to3D(uv);
	if (!isFluidCell(p)) { imageStore(vtmpImg, uv, vec4(0.0)); return; }
	float u_dt = texelFetch(ctl, ivec2(C_DT, 0), 0).r;
	vec3 x = cellPos(p);
	vec3 u = texelFetch(vel, uv, 0).xyz;
	imageStore(vtmpImg, uv, sample3D(vel, toUVW(x - u_dt * u)));
}

==compute adv_corr_force vel==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = vel_out_UNIT) uniform writeonly highp image2D vel_out;
void main() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	if (uv.x >= 512 || uv.y >= 512) return;
	ivec3 p = to3D(uv);
	if (!isFluidCell(p)) { imageStore(vel_out, uv, vec4(0.0)); return; }

	float u_dt = texelFetch(ctl, ivec2(C_DT, 0), 0).r;
	vec3 x = cellPos(p);
	vec3 u = texelFetch(vel, uv, 0).xyz;
	vec4 fh = texelFetch(vtmp, uv, 0);
	vec4 back = sample3D(vtmp, toUVW(x + u_dt * u));
	vec4 res = fh + 0.5 * (texelFetch(vel, uv, 0) - back);

	vec3 g = toUVW(x - u_dt * u) * float(FW_N) - 0.5;
	ivec3 b0 = ivec3(floor(g));
	vec4 lo = vec4(1e9), hi = vec4(-1e9);
	for (int k = 0; k < 8; k++) {
		ivec3 c = clamp(b0 + ivec3(k & 1, (k >> 1) & 1, k >> 2), ivec3(0), ivec3(FW_N - 1));
		vec4 val = texelFetch(vel, to2D(c), 0);
		lo = min(lo, val);
		hi = max(hi, val);
	}
	vec4 v = clamp(res, lo, hi);
	v.w = 1.0;

	for (int k = 0; k < 2; k++) {
		vec3 a = texelFetch(worm, ivec2(2 + k, 0), 0).xyz;
		vec3 b = texelFetch(worm, ivec2(k, 0), 0).xyz;
		vec3 ab = b - a;
		float h = clamp(dot(x - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
		float d = length(x - a - ab * h);
		float gw = exp(-(d / STIR_R) * (d / STIR_R));
		vec3 wv = ab / max(u_dt, 1e-4);
		v.xyz += (wv - v.xyz) * (1.0 - exp(-u_dt * STIR_RATE * gw));
	}
	{
		float h = 2.0 * FW_H / float(FW_N);
		float wxp = isFluidCell(p + NB[0]) ? texelFetch(vort, to2D(p + NB[0]), 0).w : 0.0;
		float wxm = isFluidCell(p + NB[1]) ? texelFetch(vort, to2D(p + NB[1]), 0).w : 0.0;
		float wyp = isFluidCell(p + NB[2]) ? texelFetch(vort, to2D(p + NB[2]), 0).w : 0.0;
		float wym = isFluidCell(p + NB[3]) ? texelFetch(vort, to2D(p + NB[3]), 0).w : 0.0;
		float wzp = isFluidCell(p + NB[4]) ? texelFetch(vort, to2D(p + NB[4]), 0).w : 0.0;
		float wzm = isFluidCell(p + NB[5]) ? texelFetch(vort, to2D(p + NB[5]), 0).w : 0.0;
		vec3 g = vec3(wxp - wxm, wyp - wym, wzp - wzm) / (2.0 * h);
		vec3 N = g / (length(g) + 1e-5);
		vec3 om = texelFetch(vort, uv, 0).xyz;
		v.xyz += u_dt * VORT_EPS * h * cross(N, om);
	}
	float swGrip = texelFetch(ctl, ivec2(C_SW_GRIP, 0), 0).r;
	if (swGrip > 0.01) {
		vec3 om = vec3(texelFetch(ctl, ivec2(C_SW_W, 0), 0).r, texelFetch(ctl, ivec2(C_SW_W + 1, 0), 0).r,
		              texelFetch(ctl, ivec2(C_SW_W + 2, 0), 0).r);
		float oml = length(om);
		vec3 ax = oml > 1e-5 ? om / oml : vec3(0.0, 0.0, 1.0);
		vec3 gc = vec3(texelFetch(ctl, ivec2(C_GCX, 0), 0).r, texelFetch(ctl, ivec2(C_GCY, 0), 0).r, texelFetch(ctl, ivec2(C_GCZ, 0), 0).r);
		float GR = max(texelFetch(ctl, ivec2(C_GR, 0), 0).r, GLOBE_MIN);
		vec3 xc = x - gc;
		vec3 rp = xc - ax * dot(xc, ax);
		float rr = max(length(rp), 1e-3);
		float core = WHIRL_CORE * GR;
		float sp = oml * (rr < core ? rr : core * core / rr);
		vec3 tgt = cross(ax, rp) / rr * sp;
		tgt -= (xc / max(length(xc), 1e-3)) * (WHIRL_SINK * sp);
		float mask = 1.0 - smoothstep(0.8 * WHIRL_R * GR, WHIRL_R * GR, length(xc));
		float tl = length(tgt);
		if (tl > SWIRL_VMAX) tgt *= SWIRL_VMAX / tl;
		v.xyz += (tgt - v.xyz) * (1.0 - exp(-u_dt * SWIRL_RATE * swGrip * mask));
	}
	{
		float sw = texelFetch(ctl, ivec2(C_SLW, 0), 0).r;
		if (sw > 0.01) {
			vec3 a = vec3(texelFetch(ctl, ivec2(C_SLA, 0), 0).r, texelFetch(ctl, ivec2(C_SLA + 1, 0), 0).r, texelFetch(ctl, ivec2(C_SLA + 2, 0), 0).r);
			vec3 b = vec3(texelFetch(ctl, ivec2(C_SLB, 0), 0).r, texelFetch(ctl, ivec2(C_SLB + 1, 0), 0).r, texelFetch(ctl, ivec2(C_SLB + 2, 0), 0).r);
			vec3 bv = vec3(texelFetch(ctl, ivec2(C_SLV, 0), 0).r, texelFetch(ctl, ivec2(C_SLV + 1, 0), 0).r, texelFetch(ctl, ivec2(C_SLV + 2, 0), 0).r);
			vec3 ab = b - a;
			float h = clamp(dot(x - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
			float d = length(x - a - ab * h);
			float gw = exp(-(d / SLICE_R) * (d / SLICE_R));
			v.xyz += (bv - v.xyz) * (1.0 - exp(-u_dt * SLICE_RATE * sw * gw));
		}
	}
	{
		vec3 jet = transpose(camRot()) * vec3(0.0, 0.0, KNOCK_V);
		for (int kk = 0; kk < 2; kk++) {
			float kw = texelFetch(ctl, ivec2(kk == 0 ? C_KW : C_K2W, 0), 0).r;
			if (kw < 0.01) continue;
			int b = kk == 0 ? C_KX : C_K2X;
			vec3 d = x - vec3(texelFetch(ctl, ivec2(b, 0), 0).r, texelFetch(ctl, ivec2(b + 1, 0), 0).r, texelFetch(ctl, ivec2(b + 2, 0), 0).r);
			float gk = exp(-dot(d, d) / (KNOCK_R * KNOCK_R));
			v.xyz += (jet - v.xyz) * (1.0 - exp(-u_dt * KNOCK_RATE * kw * gk));
		}
	}
	v.xyz *= exp(-u_dt / VEL_TAU);
	imageStore(vel_out, uv, v);
}

==compute visc1 vel==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = vtmp_UNIT) uniform writeonly highp image2D vtmpImg;
void main() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	if (uv.x >= 512 || uv.y >= 512) return;
	ivec3 p = to3D(uv);
	if (!isFluidCell(p)) { imageStore(vtmpImg, uv, vec4(0.0)); return; }
	vec4 c = texelFetch(vel, uv, 0);
	vec3 s = vec3(0.0); float n = 0.0;
	for (int k = 0; k < 6; k++) {
		ivec3 q = p + NB[k];
		if (!isFluidCell(q)) continue;
		s += texelFetch(vel, to2D(q), 0).xyz;
		n += 1.0;
	}
	if (n > 0.0) c.xyz += (s / n - c.xyz) * VISC_A;
	imageStore(vtmpImg, uv, c);
}

==compute visc2 vel==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = vel_out_UNIT) uniform writeonly highp image2D vel_out;
void main() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	if (uv.x >= 512 || uv.y >= 512) return;
	ivec3 p = to3D(uv);
	if (!isFluidCell(p)) { imageStore(vel_out, uv, vec4(0.0)); return; }
	vec4 c = texelFetch(vtmp, uv, 0);
	vec3 s = vec3(0.0); float n = 0.0;
	for (int k = 0; k < 6; k++) {
		ivec3 q = p + NB[k];
		if (!isFluidCell(q)) continue;
		s += texelFetch(vtmp, to2D(q), 0).xyz;
		n += 1.0;
	}
	if (n > 0.0) c.xyz += (s / n - c.xyz) * VISC_A;
	imageStore(vel_out, uv, c);
}

==compute div vel==
layout(local_size_x = 16, local_size_y = 16) in;
layout(r32f, binding = divB_UNIT) uniform writeonly highp image2D bImg;
layout(rgba16f, binding = vort_UNIT) uniform writeonly highp image2D vortImg;
vec3 uAt(ivec3 q) {
	return isFluidCell(q) ? texelFetch(vel, to2D(q), 0).xyz : vec3(0.0);
}
void main() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	if (uv.x >= 512 || uv.y >= 512) return;
	ivec3 p = to3D(uv);
	if (!isFluidCell(p)) { imageStore(bImg, uv, vec4(0.0)); imageStore(vortImg, uv, vec4(0.0)); return; }
	vec3 xp = uAt(p + NB[0]), xm = uAt(p + NB[1]), yp = uAt(p + NB[2]), ym = uAt(p + NB[3]), zp = uAt(p + NB[4]), zm = uAt(p + NB[5]);
	float D = 0.5 * ((xp.x - xm.x) + (yp.y - ym.y) + (zp.z - zm.z));
	imageStore(bImg, uv, vec4(D));
	float h2 = 2.0 * (2.0 * FW_H / float(FW_N));
	vec3 w = vec3((yp.z - ym.z) - (zp.y - zm.y), (zp.x - zm.x) - (xp.z - xm.z), (xp.y - xm.y) - (yp.x - ym.x)) / h2;
	imageStore(vortImg, uv, vec4(w, length(w)));
}

==compute smooth pres x8==
layout(local_size_x = 16, local_size_y = 16) in;
layout(r32f, binding = pres_UNIT) uniform highp image2D qImg;
layout(r32f, binding = divB_UNIT) uniform readonly highp image2D bImg;
uniform int pass;
void main() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	if (uv.x >= 512 || uv.y >= 512) return;
	ivec3 p = to3D(uv);
	if (((p.x + p.y + p.z) & 1) != (pass & 1)) return;
	if (!isFluidCell(p)) { imageStore(qImg, uv, vec4(0.0)); return; }
	float s = 0.0, cnt = 0.0;
	for (int k = 0; k < 6; k++) {
		ivec3 q = p + NB[k];
		if (!isFluidCell(q)) continue;
		s += imageLoad(qImg, to2D(q)).r;
		cnt += 1.0;
	}
	float old = imageLoad(qImg, uv).r;
	float gs = (s - imageLoad(bImg, uv).r) / (cnt + EPS);
	imageStore(qImg, uv, vec4(mix(old, gs, SOR_W)));
}

==compute project vel==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = vel_out_UNIT) uniform writeonly highp image2D vel_out;
layout(r32f, binding = pres_UNIT) uniform readonly highp image2D qImg;
void main() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	if (uv.x >= 512 || uv.y >= 512) return;
	ivec3 p = to3D(uv);
	if (!isFluidCell(p)) { imageStore(vel_out, uv, vec4(0.0)); return; }
	float qc = imageLoad(qImg, uv).r;
	float qn[6];
	for (int k = 0; k < 6; k++) {
		ivec3 q = p + NB[k];
		qn[k] = isFluidCell(q) ? imageLoad(qImg, to2D(q)).r : qc;
	}
	vec4 v = texelFetch(vel, uv, 0);
	v.xyz -= 0.5 * vec3(qn[0] - qn[1], qn[2] - qn[3], qn[4] - qn[5]);
	imageStore(vel_out, uv, v);
}

==compute clear accR==
layout(local_size_x = 16, local_size_y = 16) in;
layout(r32ui, binding = accR_UNIT) uniform writeonly highp uimage2D accRImg;
layout(r32ui, binding = accG_UNIT) uniform writeonly highp uimage2D accGImg;
layout(r32ui, binding = accB_UNIT) uniform writeonly highp uimage2D accBImg;
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(p, imageSize(accRImg)))) return;
	imageStore(accRImg, p, uvec4(0u));
	imageStore(accGImg, p, uvec4(0u));
	imageStore(accBImg, p, uvec4(0u));
}

==compute part 524288==
layout(local_size_x = 256) in;
layout(rgba32f, binding = part_out_UNIT) uniform writeonly highp image2D part_out;
uniform highp sampler2D part;
layout(r32ui, binding = accR_UNIT) uniform highp uimage2D accRImg;
layout(r32ui, binding = accG_UNIT) uniform highp uimage2D accGImg;
layout(r32ui, binding = accB_UNIT) uniform highp uimage2D accBImg;

#define EMIT_W      0.030
#define EMIT_T      0.002
#define DIFF        1.0e-5
#define DISSOLVE_K  2.0
#define SPLAT_R_OLD 1.65
#define GLOW_HOLD   4.0
#define GLOW_TAU    6.4
#define GLOW_END0   17.0
#define GLOW_END1   19.0
#define KEEP_F      0.98
#define PERSP0      0.6333
#define BODY_FRAC   0.0
#define BODY_NOSE   0.030
#define BODY_BACK   0.018
#define BODY_W      0.022
#define BODY_T      0.004

vec3 gauss3(inout uint s) {
	float u1 = max(rnd(s), 1e-7), u2 = rnd(s), u3 = max(rnd(s), 1e-7), u4 = rnd(s);
	float r1 = sqrt(-2.0 * log(u1)), r2 = sqrt(-2.0 * log(u3));
	return vec3(r1 * cos(6.2831853 * u2), r1 * sin(6.2831853 * u2), r2 * cos(6.2831853 * u4));
}
vec3 velAt(vec3 x) { return sample3D(vel, toUVW(x)).xyz; }
void addRGB(ivec2 q, vec3 w, inout uint rs) {
	rs = pcg(rs);
	uvec3 a = uvec3(max(w + float(rs & 0xffffu) / 65536.0, 0.0));
	if (a.x > 0u) imageAtomicAdd(accRImg, q, a.x);
	if (a.y > 0u) imageAtomicAdd(accGImg, q, a.y);
	if (a.z > 0u) imageAtomicAdd(accBImg, q, a.z);
}

void main() {
	int i = int(gl_GlobalInvocationID.x);
	if (i >= FW_NP) return;
	ivec2 ip = ivec2(i & 511, i >> 9);
	uint seed = pcg(uint(i) * 1664525u + uint(frame) * 1013904223u);

	float u_dt      = texelFetch(ctl, ivec2(C_DT, 0), 0).r;
	int u_emitStart = int(texelFetch(ctl, ivec2(C_EMIT_ST, 0), 0).r);
	int u_emitN     = int(texelFetch(ctl, ivec2(C_EMIT_N0, 0), 0).r);
	int u_emitN1    = int(texelFetch(ctl, ivec2(C_EMIT_N1, 0), 0).r);

	vec4 P = texelFetch(part, ip, 0);
	imageStore(part_out, ip, P);
	float sp  = max(floor(P.w / SP_K), 0.0);
	float age = P.w - SP_K * sp;
	int rel = (i - u_emitStart + FW_NP) % FW_NP;
	if (rel < u_emitN + u_emitN1) {
		int w = rel < u_emitN ? 0 : 1;
		vec4 sl5 = texelFetch(worm, ivec2(5, 0), 0);
		sp = sl5.y > 0.5 ? float(PAL_WHITE) : (w == 0 ? sl5.x : sl5.z);
		int nw = w == 0 ? u_emitN : u_emitN1, r0 = w == 0 ? rel : rel - u_emitN;
		float fr = (float(r0) + rnd(seed)) / float(max(nw, 1));
		vec3 a = texelFetch(worm, ivec2(2 + w, 0), 0).xyz;
		vec3 b = texelFetch(worm, ivec2(w, 0), 0).xyz;
		vec3 dir = b - a;
		dir = dot(dir, dir) > 1e-12 ? normalize(dir) : vec3(0.0, 0.0, 1.0);
		vec3 s1 = normalize(cross(dir, abs(dir.y) < 0.95 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
		vec3 s2 = cross(dir, s1);
		float roll = texelFetch(worm, ivec2(4, 0), 0).x;
		float tw = roll + 3.14159 * float(w);
		vec3 side = cos(tw) * s1 + sin(tw) * s2, nrm = cross(dir, side);
		vec3 gq = gauss3(seed);
		vec3 pos;
		if (fr < BODY_FRAC) {
			float u = rnd(seed), v = rnd(seed);
			if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
			vec3 tz = nrm * (BODY_T * gq.x);
			vec3 nose  = b + dir * BODY_NOSE + tz;
			vec3 wingA = b - dir * BODY_BACK - side * BODY_W + tz;
			vec3 wingB = b - dir * BODY_BACK + side * BODY_W + tz;
			pos = nose * (1.0 - u - v) + wingA * u + wingB * v;
		} else {
			float fr2 = (fr - BODY_FRAC) / (1.0 - BODY_FRAC);
			pos = mix(a, b, fr2) + side * EMIT_W * (2.0 * rnd(seed) - 1.0) + nrm * EMIT_T * gq.x + dir * EMIT_T * gq.y;
		}
		age = (1.0 - fr) * u_dt;
		P.xyz = pos;
	} else if (age >= 0.0) {
		vec3 v = velAt(P.xyz);
		float dAge = DIFF * (1.0 + DISSOLVE_K * smoothstep(GLOW_HOLD, GLOW_END1, age));
		P.xyz += u_dt * v + sqrt(2.0 * dAge * u_dt) * gauss3(seed);
		age += u_dt;
		if (age > GLOW_END1 + 0.5) { imageStore(part_out, ip, vec4(P.xyz, -1.0)); return; }
	} else return;

	P.xyz = clamp(P.xyz, -KEEP_F * FW_H, KEEP_F * FW_H);
	P.w = age + SP_K * sp;
	imageStore(part_out, ip, P);

	float bright = exp(-max(age - GLOW_HOLD, 0.0) / GLOW_TAU) * (1.0 - smoothstep(GLOW_END0, GLOW_END1, age));
	if (bright < 0.003) return;
	mat3 R = camRot();
	vec3 sc = project(P.xyz, R);
	if ((R * P.xyz).z + CAM_D < 0.3) return;
	float persp = sc.z;
	sc.z = fogAt(P.xyz, R);

	ivec2 u_canvas = imageSize(accRImg);
	vec3 col = PAL_COL[int(sp + 0.5)];
	vec2 px = (sc.xy * float(u_canvas.y) + vec2(u_canvas)) * 0.5 - 0.5;
	ivec2 p0 = ivec2(floor(px));
	vec2 f = px - vec2(p0);
	float wgt = bright * sc.z * SPLAT_K;

	float hw = mix(1.0, SPLAT_R_OLD, smoothstep(GLOW_HOLD, GLOW_HOLD + 2.0 * GLOW_TAU, age));
	hw *= clamp(persp / PERSP0, 0.55, 1.5);
	uint rs = pcg(uint(ip.x) * 7919u + uint(ip.y) * 104729u + uint(u_emitStart));
	if (hw <= 1.001) {
		for (int k = 0; k < 4; k++) {
			ivec2 o = ivec2(k & 1, k >> 1);
			ivec2 q = p0 + o;
			if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, u_canvas))) continue;
			float bw = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y);
			addRGB(q, col * (wgt * bw), rs);
		}
	} else {
		ivec2 lo = ivec2(ceil(px - hw)), hi = ivec2(floor(px + hw));
		float wsum = 0.0;
		for (int y = lo.y; y <= hi.y; y++)
			for (int x = lo.x; x <= hi.x; x++) {
				vec2 d = abs(vec2(x, y) - px) / hw;
				wsum += max(1.0 - d.x, 0.0) * max(1.0 - d.y, 0.0);
			}
		float norm = wgt / max(wsum, 1e-6);
		for (int y = lo.y; y <= hi.y; y++)
			for (int x = lo.x; x <= hi.x; x++) {
				ivec2 q = ivec2(x, y);
				if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, u_canvas))) continue;
				vec2 d = abs(vec2(q) - px) / hw;
				float bw = max(1.0 - d.x, 0.0) * max(1.0 - d.y, 0.0);
				addRGB(q, col * (norm * bw), rs);
			}
	}
}

==compute pack glowP==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = glowP_UNIT) uniform writeonly highp image2D glowPImg;
layout(r32ui, binding = accR_UNIT) uniform readonly highp uimage2D accRImg;
layout(r32ui, binding = accG_UNIT) uniform readonly highp uimage2D accGImg;
layout(r32ui, binding = accB_UNIT) uniform readonly highp uimage2D accBImg;
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(p, imageSize(glowPImg)))) return;
	vec3 v = vec3(float(imageLoad(accRImg, p).r), float(imageLoad(accGImg, p).r), float(imageLoad(accBImg, p).r)) / SPLAT_K;
	imageStore(glowPImg, p, vec4(v, 0.0));
}

==compute bloom bloom==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = bloom_UNIT) uniform writeonly highp image2D bloomImg;
uniform highp sampler2D glowP;
vec3 glowAt(ivec2 q, ivec2 sz) {
	if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, sz))) return vec3(0.0);
	return texelFetch(glowP, q, 0).rgb;
}
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	ivec2 bs = imageSize(bloomImg);
	if (any(greaterThanEqual(p, bs))) return;
	ivec2 gs = textureSize(glowP, 0);
	ivec2 ip = ivec2((vec2(p) + 0.5) * vec2(gs) / vec2(bs));
	vec3 b = vec3(0.0); float wsum = 0.0;
	for (int ring = 1; ring <= 3; ring++) {
		float rad = float(1 << ring);
		float w = 1.0 / float(ring);
		for (int k = 0; k < 8; k++) {
			float a = 0.7853982 * (float(k) + 0.5 * float(ring & 1));
			b += w * glowAt(ip + ivec2(round(rad * vec2(cos(a), sin(a)))), gs);
		}
		wsum += 8.0 * w;
	}
	imageStore(bloomImg, p, vec4(b / wsum, 0.0));
}

==image==
uniform vec2 resolution;
uniform highp sampler2D glowP;
uniform highp sampler2D bloom;
uniform highp sampler2D worm;
out vec4 fragColor;

#define W_GLOW_GAIN  0.22
#define W_TONE_REF   18.0
#define W_DIM_LIFT   0.65
#define W_BLOOM_GAIN 0.18
#define W_BODY_LIFT  0.35
#define DEEP_COL     vec3(0.000, 0.004, 0.010)
#define UP_COL       vec3(0.006, 0.026, 0.045)
#define BODY_R       0.028
#define BODY_GAIN    2.6

vec3 fetch0(highp sampler2D tex, ivec2 p, ivec2 sz) {
	if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, sz))) return vec3(0.0);
	return texelFetch(tex, p, 0).rgb;
}
vec3 bilin(highp sampler2D tex, vec2 uvCanvas) {
	if (any(lessThan(uvCanvas, vec2(0.0))) || any(greaterThan(uvCanvas, vec2(1.0)))) return vec3(0.0);
	return texture(tex, uvCanvas).rgb;
}
vec3 wormGlow(vec2 uvCanvas) {
	vec3 x = bilin(glowP, uvCanvas) + W_BLOOM_GAIN * bilin(bloom, uvCanvas) * 6.0;
	float L = max(x.r, max(x.g, x.b));
	if (L <= 0.0) return vec3(0.0);
	float e = W_GLOW_GAIN * W_TONE_REF * pow(L / W_TONE_REF, W_DIM_LIFT);
	return 1.0 - exp(-e * (x / L));
}

vec3 waterSheen(vec2 fragCoord) {
	vec2 uv = (2.0 * fragCoord - resolution.xy) / resolution.y;
	vec3 d = normalize(transpose(camRot()) * vec3(uv, FOCAL));
	return mix(DEEP_COL, UP_COL, pow(smoothstep(-0.4, 1.0, d.y), 1.5));
}

vec3 bodyGlow(vec2 uv, vec4 sl) {
	vec3 col = vec3(0.0);
	mat3 R = camRot();
	for (int w = 0; w < 2; w++) {
		vec3 cw = R * texelFetch(worm, ivec2(w, 0), 0).xyz;
		float z = cw.z + CAM_D;
		if (z < 0.3) continue;
		float s = FOCAL / z;
		vec2 d = (uv - cw.xy * s) / max(BODY_R * s, 1e-4);
		float fog = exp(-max(z - FOG_D0, 0.0) * FOG_K);
		vec3  hue = mix(PAL_COL[int((w == 0 ? sl.x : sl.z) + 0.5)], vec3(1.0), sl.y);
		float bi  = BODY_GAIN * exp(-dot(d, d) * 1.5) * fog;
		col += mix(hue, vec3(1.0), W_BODY_LIFT) * bi;
	}
	return col;
}

void main() {
	vec2 fc = gl_FragCoord.xy - 0.5 * resolution.y * texelFetch(worm, ivec2(6, 0), 0).xy;
	vec2 uvCanvas = fc / resolution;
	vec2 uv = (2.0 * fc - resolution.xy) / resolution.y;
	vec4 sl = texelFetch(worm, ivec2(5, 0), 0);
	fragColor = vec4(waterSheen(fc) + wormGlow(uvCanvas) + bodyGlow(uv, sl), 1.0);
}
