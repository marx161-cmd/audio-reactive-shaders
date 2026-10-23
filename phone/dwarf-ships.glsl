// Dwarf Ships (ShaderEditor fork, Pixel 10 Pro): the dwarf diorama as a declared
// music box (moon orbits and hit timing on rails; meteor flight and ring fluid
// real). Two ships in the star's beams plan shots, rescue loose moons, farm the
// ring, and pincer the star to break off new moons.

#version 300 es

#define BACKDROP_dio_backC dio_backC.png
#define BACKDROP_dio_wlC   dio_wlS.png
#define BACKDROP_dio_wrC   dio_wrS.png
#define BACKDROP_dio_topC  dio_topS.png
#define BACKDROP_dio_botC  dio_botS.png
#define DIO_S       1.65
#define BOX_L       (3.0 / DIO_S)
#define RING_TILT   0.5236
#define BOX_LX      (2.5 * sin(RING_TILT) + 0.07)
#define BOX_LZ      (2.5 * cos(RING_TILT) + 0.07)
#define BOX_HALF    vec3(BOX_LX, BOX_LX * 2.231, BOX_LZ)
#define EARTH_R     2.4
#define EARTH_G     0.0
#define WALL_KX     2.01
#define WALL_KY     22.5
#define WALL_KZ     9.83
#define SHAKE_K     0.0
#define SHAKE_FREE  1.0e9
#define SHAKE_FREE_P 0.08
#define DW_DAMP     0.65
#define MOON_BOUNCE 0.90
#define DW_BOOP     0.6
#define DIO_CAM     vec3(0.0, 0.0, BOX_HALF.z + BOX_HALF.x / FOV)   /* front face = the screen (FIXED: no pinch) */
#define DIO_CAM_MAX (20.0 / DIO_S)
#define DIO_CAM_MIN (3.6 / DIO_S)

#define IMAGE_ring  rgba16f 1024x192 swap
#define IMAGE_phase r32f    640x1
#define IMAGE_moons r32f    64x4
#define IMAGE_ctl   r32f    256x1
#define IMAGE_met   r32f    128x16
#define IMAGE_ships r32f    256x2
#define CH_N        48
#define N_SHIPS     2
#define PS_BEATS    16.0
#define PS_SLEW     1.5
#define PS_W        0.45
#define PS_WA       0.6
#define PS_JET      0.36
#define PS_JET_PAD  0.15
#define PS_RMIN     0.45
#define PS_REACH    2.45
#define SHIP_R0     1.15
#define SHIP_R1     2.25
#define SHIP_H      0.26
#define SHIP_L      0.05
#define SHIP_HIST_DT 0.045
#define SHIP_TRAIL_S 0.55
#define SHIP_FALLBACK_BPM 100.0
#define RS_H        0.02
#define RS_NMAX     200
#define RS_LEAD     1.6
#define RS_TAU0     2.4
#define RS_TAU1     4.0
#define RS_TAU2     6.5
#define RS_MINFLY   0.35
#define RS_SHOTFLY  0.7
#define RS_VMAX     1.6
#define RS_STEP     0.5
#define RS_COOL2    0.5
#define RS_TOL      0.30
#define RS_BLEND    2.0
#define RS_HITR     0.10
#define RS_ERR      0.03
#define RS_ITERS    6
#define RS_EPS      0.02
#define RS_LATE     0.4
#define RS_COOL     1.0
#define FM_ON       1
#define FM_LOW      0.80
#define FM_COOL     5.0
#define FM_PUFF     1.2
#define FM_PUFF_R   0.22
#define FM_BURST_DT 0.2
#define FM_DB       0.022
#define HOLE_DB     0.012
#define HOLE_K      0.09
#define HOLE_R      0.10
#define SPIT_ON     1
#define SPIT_DELAY  4.0
#define SPIT_FLASH  0.45
#define SHOT_DWARF_KICK 0.08
#define SHIPS_ON    1

precision highp float;
precision highp int;

#define NR          192
#define NT          1024
#define R_IN        1.25
#define R_OUT       2.32
#define DW_R        0.40
#define DR          ((R_OUT - R_IN) / float(NR))
#define DTH         (6.28318530718 / float(NT))
#define TWO_PI      6.28318530718
#define PI          3.14159265359
#define DIO_LAYOUT  11.0
#define MET_N          8
#define MET_FINGER_N   4
#define MET_HIST       32
#define MET_R          0.03
#define MET_MOON_BOOST 400.0
#define MET_DWARF_G    10.0
#define MET_PUSH       0.6
#define MET_DWARF_KICK 0.25
#define MET_LIFE       25.0
#define MET_TAIL_S     0.6
#define N_MOONS     3
#define SIM_RATE    0.5

#define CAM_D       6.0
#define CAM_EL      0.65
#define CAM_ORBIT   0.006
#define AIM_OFF     0.80
#define CAM_ROLL    0.7854
#define FOV         0.125

float rowR(float j) { return R_IN + (j + 0.5) * DR; }
float omegaK(float r) { return pow(r, -1.5); }

float hash11(float p) { p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
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
float noise1(float x) {
	float i = floor(x), f = fract(x);
	f = f * f * (3.0 - 2.0 * f);
	return mix(hash11(i), hash11(i + 1.0), f);
}

float sigma0(float r) {
	float s = smoothstep(1.35, 1.41, r) * (1.0 - smoothstep(2.15, 2.20, r));
	return s * (0.88 + 0.12 * noise1(r * 40.0));
}

void getCamera(float az, out vec3 ro, out vec3 fw, out vec3 rt, out vec3 up) {
	ro = CAM_D * vec3(cos(CAM_EL) * sin(az), sin(CAM_EL), -cos(CAM_EL) * cos(az));
	vec3 aim = AIM_OFF * normalize(vec3(ro.x, 0.0, ro.z));
	fw = normalize(aim - ro);
	rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw));
	up = cross(fw, rt);
	float cr = cos(CAM_ROLL), sr = sin(CAM_ROLL);
	vec3 r2 = cr * rt + sr * up;
	up = -sr * rt + cr * up;
	rt = r2;
}

#define PARK_R   1.22
#define PARK_Y   0.06
#define PINCH_K  1.6
#define CAM_RMIN (DW_R + 0.06)
#define CAM_RMAX 4.0
void periCam(vec3 rv, vec3 ro, float scr, out vec3 fw, out vec3 rt, out vec3 up) {
	vec4 q = vec4(rv, sqrt(max(0.0, 1.0 - dot(rv, rv))));
	float x = q.x, y = q.y, z = q.z, w = q.w;
	mat3 R = mat3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w),       2.0 * (x * z - y * w),
	              2.0 * (x * y - z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w),
	              2.0 * (x * z + y * w),       2.0 * (y * z - x * w),       1.0 - 2.0 * (x * x + y * y));
	vec3 U = normalize(-ro);
	vec3 B = normalize(vec3(0.0, 1.0, 0.0) - U * U.y);
	vec3 A = cross(U, B);
	mat3 M = mat3(A, B, U);
	vec3 sx = scr > 0.5 ? vec3(0.0, -1.0, 0.0) : (scr < -0.5 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0));
	vec3 sy = scr > 0.5 ? vec3(1.0, 0.0, 0.0) : (scr < -0.5 ? vec3(-1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0));
	rt = M * (R * sx);
	up = M * (R * sy);
	fw = M * (R * vec3(0.0, 0.0, -1.0));
}

void dioCam(vec3 rv, vec2 h0, float scr, out vec3 fw, out vec3 rt, out vec3 up) {
	vec4 q = vec4(rv, sqrt(max(0.0, 1.0 - dot(rv, rv))));
	float x = q.x, y = q.y, z = q.z, w = q.w;
	mat3 R = mat3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w),       2.0 * (x * z - y * w),
	              2.0 * (x * y - z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w),
	              2.0 * (x * z + y * w),       2.0 * (y * z - x * w),       1.0 - 2.0 * (x * x + y * y));
	vec3 U = vec3(0.0, 1.0, 0.0);
	vec3 B = normalize(vec3(h0.x, 0.0, -h0.y) + vec3(1e-6, 0.0, 0.0));
	vec3 A = cross(U, B);
	mat3 M = mat3(A, B, U);
	vec3 sx = scr > 0.5 ? vec3(0.0, -1.0, 0.0) : (scr < -0.5 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0));
	vec3 sy = scr > 0.5 ? vec3(1.0, 0.0, 0.0) : (scr < -0.5 ? vec3(-1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0));
	rt = M * (R * sx);
	up = M * (R * sy);
	fw = M * (R * vec3(0.0, 0.0, -1.0));
}
#define DX_B vec3(0.0, 1.0, 0.0)
#define DY_B vec3(cos(RING_TILT), 0.0, -sin(RING_TILT))
#define DZ_B vec3(-sin(RING_TILT), 0.0, -cos(RING_TILT))
vec3 b2d(vec3 b) { return vec3(dot(b, DX_B), dot(b, DY_B), dot(b, DZ_B)); }
vec3 d2b(vec3 d) { return d.x * DX_B + d.y * DY_B + d.z * DZ_B; }
vec3 wallAcc(vec3 p) {
	vec3 dn = max(BOX_HALF + p, vec3(0.05)), dp = max(BOX_HALF - p, vec3(0.05));
	return vec3(WALL_KX, WALL_KY, WALL_KZ) * (1.0 / (dn * dn) - 1.0 / (dp * dp));
}

#define ROCK_ON  0
#define RK_BOUND 0.30
const vec3 ROCK_COM = vec3(-0.0152, 0.0, 0.0);
const vec3 ROCK_I   = vec3(0.0024, 0.0091, 0.0110);
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

uniform float time;
uniform int frame;
uniform float beatCount;
uniform float beatPhase;
uniform float beatConf;
uniform float beatBpm;

#ifdef COMPUTE
uniform vec2 resolution;
uniform vec2 touch;
uniform int pointerCount;
uniform bool screenLocked;
uniform vec3 gyroscope;
uniform vec3 gravity;
uniform vec3 linear;
uniform vec3 rotationVector;
uniform vec3 pointers[10];
uniform highp sampler2D bands;
uniform highp sampler2D ring;

#define ROS_N        6
#define ROS_A        1.92
#define ROS_GM       8.0e-6
#define MUTUAL_BOOST 250.0
#define STAB_G       0.05
#define STAB_L       0.02
#define STAB_E       0.02
#define ROS_SOFT     0.02
#define ROS_SUM      1.828427
#define ROS_N_EQ     0.376589

#define CO_A         1.30
#define CO_D         0.045
#define CO_MU        1.0e-3
#define CO_SPEED     2.5
#define CO_GM        2.0e-5
#define CO_N         0.674664
#define N_CO         2
#define SURF_A       2.22
#define SURF_E       0.012
#define SURF_GM      2.0e-5
#define PLG_A        1.75
#define PLG_E        0.55      /* (2.2 / 0.62 -> 1.75 / 0.55: apo 2.71 inside the roof/floor 2.95, peri 0.79) */
#define PLG_I        0.25
#define PLG_GM       2.0e-5

vec3 rockOmegaDot(vec3 w) {
	return vec3((ROCK_I.y - ROCK_I.z) / ROCK_I.x * w.y * w.z,
	            (ROCK_I.z - ROCK_I.x) / ROCK_I.y * w.z * w.x,
	            (ROCK_I.x - ROCK_I.y) / ROCK_I.z * w.x * w.y);
}

vec4 coAcc(vec4 p) {
	vec4 f = vec4(0.0);
	for (int i = 0; i < N_CO; i++) {
		float acc = 0.0;
		for (int j = 0; j < N_CO; j++) {
			if (i == j) continue;
			float th = mod(p[i] - p[j] + PI, TWO_PI) - PI;
			float sn = max(abs(sin(0.5 * th)), 0.02);
			acc += -3.0 * CO_N * CO_N * CO_MU * sin(th) * (1.0 - 1.0 / (8.0 * sn * sn * sn));
		}
		f[i] = acc;
	}
	return f;
}

#define ctl1(s, c) imageLoad(ctl, ivec2((s) * 4 + (c), 0)).r
#define ctl4(s) vec4(ctl1(s, 0), ctl1(s, 1), ctl1(s, 2), ctl1(s, 3))
#define ctlS(s, c, x) imageStore(ctl, ivec2((s) * 4 + (c), 0), vec4(x))
#define ctlW(s, v) do { vec4 _cv = (v); ctlS(s, 0, _cv.x); ctlS(s, 1, _cv.y); ctlS(s, 2, _cv.z); ctlS(s, 3, _cv.w); } while (false)
#define moon1(i, g, c) imageLoad(moons, ivec2((i) * 4 + (c), g)).r
#define moon4(i, g) vec4(moon1(i, g, 0), moon1(i, g, 1), moon1(i, g, 2), moon1(i, g, 3))
#define moonS(i, g, c, x) imageStore(moons, ivec2((i) * 4 + (c), g), vec4(x))
#define moonW(i, g, v) do { vec4 _mv = (v); moonS(i, g, 0, _mv.x); moonS(i, g, 1, _mv.y); moonS(i, g, 2, _mv.z); moonS(i, g, 3, _mv.w); } while (false)
#define phR(j, c) imageLoad(phase, ivec2((j) * 2 + (c), 0)).r
#define phW(j, now, prev) do { imageStore(phase, ivec2((j) * 2, 0), vec4(now)); imageStore(phase, ivec2((j) * 2 + 1, 0), vec4(prev)); } while (false)
#endif

==compute init 1024x192==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = ring_UNIT) uniform writeonly highp image2D ring_w;
layout(r32f, binding = ctl_UNIT) uniform readonly highp image2D ctl;
void main() {
	ivec2 c = ivec2(gl_GlobalInvocationID.xy);
	if (c.x >= NT || c.y >= NR) return;
	if (ctl1(0, 0) > 0.5 && abs(ctl1(27, 0) - DIO_LAYOUT) < 0.5) return;
	imageStore(ring_w, c, vec4(sigma0(rowR(float(c.y))), 0.0, 0.0, 0.0));
}

==compute ctrl 1==
layout(local_size_x = 1) in;
layout(r32f, binding = ctl_UNIT) uniform highp image2D ctl;
layout(r32f, binding = moons_UNIT) uniform highp image2D moons;
layout(r32f, binding = met_UNIT) uniform highp image2D met;
layout(r32f, binding = ships_UNIT) uniform highp image2D ships;
#define shL(k, i) imageLoad(ships, ivec2(k, i)).r
#define metL(i, c) imageLoad(met, ivec2(c, MET_N + (i))).r
#define metL4(i, c) vec4(metL(i, c), metL(i, (c) + 1), metL(i, (c) + 2), metL(i, (c) + 3))
#define metW4(i, c, v) do { vec4 _w = (v); imageStore(met, ivec2(c, MET_N + (i)), vec4(_w.x)); imageStore(met, ivec2((c) + 1, MET_N + (i)), vec4(_w.y)); imageStore(met, ivec2((c) + 2, MET_N + (i)), vec4(_w.z)); imageStore(met, ivec2((c) + 3, MET_N + (i)), vec4(_w.w)); } while (false)

#define SWIPE_PX     35.0
#define SWIPE_K      2.8
#define GYRO_K       1.0
#define YAW_TAU      1.2
#define BOOP_DV_MOON 0.30
#define BOOP_SPIN    5.0
#define SPLAT_V      0.06
#define SPLAT_T      0.15
#define FREE_GONE_R  6.0
#define TOUCH_PX     10.0
#define GRAB_CONE    (TOUCH_PX * 2.0 * FOV / min(resolution.x, resolution.y))
#define FLING_SMOOTH 0.35
#define FLING_K      1.0
#define FLING_MAX    2.5
#define TAP_S        0.25

const float ROS_DRAW[6] = float[6](0.022, 0.017, 0.026, 0.019, 0.024, 0.018);
const float CO_PHI0[4]  = float[4](0.0, 2.2, 0.0, 0.0);
const float CO_SIDE[4]  = float[4](-1.0, 1.0, 0.0, 0.0);
const float CO_DRAW[4]  = float[4](0.07, 0.06, 0.0, 0.0);
const vec4 POL0[4] = vec4[4](
	vec4(2.03, 2.87979, 0.0,       1.0),
	vec4(1.98, 3.45226, 1.570796, -1.0),
	vec4(2.10, 2.95833, 3.141593, -1.0),
	vec4(2.07, 3.35976, 4.712389,  1.0)
);
const vec2 POL1[4] = vec2[4](
	vec2(0.034, 1.5e-5), vec2(0.028, 1.2e-5), vec2(0.040, 2.0e-5), vec2(0.030, 1.3e-5)
);

void main() {
	vec4 c0 = ctl4(0), c1 = ctl4(1), c2 = ctl4(2), c3 = ctl4(3);
	vec4 cp = ctl4(8), cw = ctl4(9);
	vec3 rw = ctl4(11).xyz;
	mat3 rR = mat3(ctl4(12).xyz, ctl4(13).xyz, ctl4(14).xyz);
	vec4 c15 = ctl4(15);
	vec4 c16 = ctl4(16);
	vec4 c20 = ctl4(20);
	vec4 c21 = ctl4(21);
	vec4 c22 = ctl4(22);
	vec4 c23 = ctl4(23);
	vec4 c24 = ctl4(24);
	vec4 c25 = ctl4(25);
	vec4 c28 = ctl4(28);
	vec4 c29 = ctl4(29);
	vec4 spit = ctl4(5);
	vec4 c6 = ctl4(6);
	vec4 c33 = ctl4(33);
	vec4 c31 = ctl4(31);
#define SHP_N 16
	vec4 SHP[N_SHIPS * SHP_N];
	{
		int H0 = int(c31.z + 0.5);
		for (int i = 0; i < N_SHIPS; i++) for (int k = 0; k < SHP_N; k++) {
			int e = (H0 - k + CH_N) % CH_N;
			SHP[i * SHP_N + k] = vec4(shL(4 * e, i), shL(4 * e + 1, i), shL(4 * e + 2, i), shL(4 * e + 3, i));
		}
	}
	vec4 MA[MET_N], MB[MET_N], MX[MET_N], MV[MET_N];
	vec4 MT[MET_N], MW[MET_N];
	for (int k = 0; k < MET_N; k++) { MA[k] = metL4(k, 0); MB[k] = metL4(k, 4); MX[k] = metL4(k, 8); MV[k] = metL4(k, 12); MT[k] = metL4(k, 20); MW[k] = metL4(k, 24); }
	vec4 RSa[N_MOONS], RSb[N_MOONS], RSc[N_MOONS], RSd[N_MOONS], RSe[N_MOONS], RSf[N_MOONS];
	float phiS = ctl1(58, 0);
	float ringB = ctl1(4, 0);
	vec4 c48 = vec4(0.0, 0.0, 0.0, -100.0), c50 = vec4(0.0), c51 = vec4(0.0);
	vec4 c49 = vec4(0.0, 0.0, 0.0, -1.0);
	for (int i = 0; i < N_MOONS; i++) { RSa[i] = ctl4(40 + 6 * i); RSb[i] = ctl4(41 + 6 * i); RSc[i] = ctl4(42 + 6 * i); RSd[i] = ctl4(43 + 6 * i); RSe[i] = ctl4(44 + 6 * i); RSf[i] = ctl4(45 + 6 * i); }
	vec4 M0[N_MOONS], M1[N_MOONS], M2[N_MOONS], M3[N_MOONS];
	vec4 MP[N_MOONS];
	for (int i = 0; i < N_MOONS; i++) {
		M2[i] = moon4(i, 2);
		M3[i] = moon4(i, 3);
		MP[i] = moon4(i, 0);
	}

	bool first = c0.x < 0.5 || abs(ctl1(27, 0) - DIO_LAYOUT) > 0.5;
	float dt = first ? 0.016 : clamp(time - c0.z, 1e-4, 0.05);
	float t_sim = first ? 0.0 : c0.y;
	float itime = time;
	float h = dt * SIM_RATE;

	if (first) {
		for (int i = 0; i < N_MOONS; i++) { M2[i] = vec4(0.0); M3[i] = vec4(0.0); }
		cp = vec4(CO_PHI0[0], CO_PHI0[1], CO_PHI0[2], CO_PHI0[3]);
		cw = -1.5 * CO_N * vec4(CO_SIDE[0], CO_SIDE[1], CO_SIDE[2], CO_SIDE[3]) * (CO_D / CO_A);
		rw = vec3(0.02, 2.2, 0.01);
		rR = mat3(1.0);
		c15 = vec4(0.0, 0.0, 0.0, -100.0);
		c2 = vec4(0.0, CAM_ORBIT, 0.0, 0.0);
		c16 = vec4(DIO_CAM, 0.0);
		c23 = vec4(0.0); c24 = vec4(0.0, -0.4, 0.0, 0.0); c25 = vec4(0.0);
		c20 = vec4(0.0); c21 = vec4(0.0); c22 = vec4(0.0);
		c28 = vec4(0.0); c29 = vec4(0.0); spit = vec4(-100.0, 0.0, 0.0, 0.0); c6 = vec4(0.0);
		c33 = vec4(PS_JET, PS_JET, 0.0, -100.0); c31 = vec4(1.0, 0.0, 0.0, 0.0);
		for (int k = 0; k < MET_N; k++) { MA[k] = vec4(0.0, -100.0, 0.0, 0.0); MB[k] = vec4(0.0); MX[k] = vec4(0.0); MV[k] = vec4(0.0); MT[k] = vec4(0.0); MW[k] = vec4(0.0); }
		phiS = PI + 1.1; ringB = 1.0;
		c48 = vec4(0.0, 0.0, 0.0, -100.0); c50 = vec4(0.0); c51 = vec4(0.0);
		for (int i = 0; i < N_MOONS; i++) { RSa[i] = vec4(0.0); RSb[i] = vec4(0.0); RSc[i] = vec4(0.0); RSd[i] = vec4(0.0); RSe[i] = vec4(0.0); RSf[i] = vec4(0.0, 0.0, 0.0, -100.0); }
	}

	float yaw = c2.x, yawV = c2.y, tDown = c2.z, tx0 = c2.w;
	float txp = c3.x, tSwipe = c3.y, ty0 = c3.z, typ = c3.w;
	bool swiping = false;
	bool justTapped = false;
	bool justDown = false;
	vec2 tapXY = vec2(0.0);

	if (pointerCount > 0 && !screenLocked) {
		if (tDown < 0.5) {
			justDown = true;
			tDown = 1.0; tx0 = touch.x; txp = touch.x; ty0 = touch.y; typ = touch.y; tSwipe = 0.0;
		}
		float dx = touch.x - txp;
		txp = touch.x; typ = touch.y;
		if (length(touch - vec2(tx0, ty0)) > SWIPE_PX) tSwipe = 1.0;
		if (pointerCount >= 2) tSwipe = 1.0;
	} else {
		if (tDown > 0.5 && tSwipe < 0.5) {
			justTapped = true;
			tapXY = vec2(txp, typ);
		}
		tDown = 0.0;
	}
	yaw = 0.0; yawV = 0.0;
	vec3 camB = c16.xyz;
	if (!(dot(camB, camB) > 0.01)) camB = DIO_CAM;
	float scr = ctl1(19, 3);
	if (resolution.y >= resolution.x) scr = 0.0;
	else if (gravity.x > 2.0) scr = 1.0;
	else if (gravity.x < -2.0) scr = -1.0;
	else if (abs(scr) < 0.5) scr = 1.0;
	vec3 gB = length(gravity) > 0.5 ? -normalize(gravity) * EARTH_G : vec3(0.0, -EARTH_G, 0.0);
	if (pointerCount >= 3 && !screenLocked) camB = DIO_CAM;
	vec3 D = c24.xyz, vD = c25.xyz;
	if (!(dot(D, D) < 1e6) || !(dot(vD, vD) < 1e6)) { D = vec3(0.0, -0.4, 0.0); vD = vec3(0.0); }
	vec3 aEnv = gB - SHAKE_K * linear;
	float shakeOver = max(length(linear) - SHAKE_FREE, 0.0);
	vec3 aD = wallAcc(D) + aEnv - DW_DAMP * vD;
	vD += h * aD;
	D += h * vD;
	D = clamp(D, -BOX_HALF + vec3(DW_R + 0.05), BOX_HALF - vec3(DW_R + 0.05));
	vec3 sxB = scr > 0.5 ? vec3(0.0, -1.0, 0.0) : (scr < -0.5 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0));
	vec3 syB = scr > 0.5 ? vec3(1.0, 0.0, 0.0) : (scr < -0.5 ? vec3(-1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0));
	float pinch = 0.0;
	camB = DIO_CAM;
	vec3 fw = b2d(vec3(0.0, 0.0, -1.0)), rt = b2d(sxB), up = b2d(syB);
	vec3 ro = b2d(camB - D);
	{ float rr = length(ro); if (rr < CAM_RMIN) { ro *= CAM_RMIN / max(rr, 1e-4); camB = d2b(ro) + D; } }

	int gi = int(c20.x + 0.5) - 1;
	float grabD = c20.y, tGrab = c20.z;
	bool grabbedTouch = c20.w > 0.5 && !justDown;
	vec2 uvG = (touch - 0.5 * resolution) / min(resolution.x, resolution.y) * 2.0 * FOV;
	vec3 rdG = normalize(fw + uvG.x * rt + uvG.y * up);
	if (justDown && gi < 0 && pointerCount == 1 && !screenLocked) {
		float bestAng = GRAB_CONE;
		for (int i = 0; i < N_MOONS; i++) {
			vec4 m = MP[i];
			if (m.w <= 0.0) continue;
			vec3 rel = m.xyz - ro;
			float d = length(rel);
			float ang = acos(clamp(dot(rel / d, rdG), -1.0, 1.0)) - asin(clamp(m.w / d, 0.0, 1.0));
			if (ang < bestAng) { bestAng = ang; gi = i; }
		}
		if (gi >= 0) {
			vec3 x = MP[gi].xyz;
			grabD = max(dot(x - ro, rdG), 0.05);
			tGrab = itime;
			grabbedTouch = true;
			float st0 = M2[gi].w;
			float r = max(length(x.xz), 0.1);
			c22 = vec4((st0 > 0.5 && st0 < 1.5) ? M3[gi].xyz : sqrt(1.0 / r) * vec3(-x.z, 0.0, x.x) / r, 0.0);
			c21 = vec4(0.0);
			M2[gi] = vec4(x, 3.0);
			M3[gi] = vec4(0.0);
		}
	}
	if (gi >= 0) {
		if (pointerCount == 1 && !screenLocked) {
			vec3 tgt = ro + rdG * grabD;
			if (abs(rdG.y) > 1e-4) {
				float tp = -ro.y / rdG.y;
				vec3 Q = ro + rdG * tp;
				float rq = length(Q.xz);
				if (tp > 0.0 && rq > R_IN - 0.03 && rq < R_OUT + 0.03) { tgt = Q; grabD = tp; }
			}
			float tr = length(tgt), minR = DW_R + max(MP[gi].w, 0.02) + 0.02;
			if (tr < minR) tgt *= minR / max(tr, 1e-4);
			tgt = b2d(clamp(d2b(tgt) + D, -BOX_HALF + vec3(0.05), BOX_HALF - vec3(0.05)) - D);
			c21.xyz = mix(c21.xyz, (tgt - M2[gi].xyz) / max(h, 1e-4), FLING_SMOOTH);
			M2[gi] = vec4(tgt, 3.0);
			M3[gi] = vec4(c21.xyz, 0.0);
		} else {
			vec3 v;
			if (itime - tGrab < TAP_S && tSwipe < 0.5) v = c22.xyz + rdG * BOOP_DV_MOON;
			else {
				v = c21.xyz * FLING_K;
				float vl = length(v);
				if (vl > FLING_MAX) v *= FLING_MAX / vl;
			}
			M2[gi] = vec4(M2[gi].xyz, 1.0);
			M3[gi] = vec4(v, 0.0);
			gi = -1;
		}
	}

	float hc = h * CO_SPEED;
	cw += 0.5 * hc * coAcc(cp);
	cp = mod(cp + hc * cw, TWO_PI);
	cw += 0.5 * hc * coAcc(cp);

	t_sim += h;

	for (int i = 0; i < N_MOONS; i++) {
		vec4 m2 = M2[i];
		vec4 m3 = M3[i];
		float st = m2.w;
		vec3 pos = vec3(0.0);
		float rad = 0.025, gm = 1.5e-5, sinkOvr = 0.0;

		if (i < 2) {
			float r = CO_A - cw[i] * CO_A / (1.5 * CO_N);
			float lam = CO_N * t_sim + cp[i];
			rad = CO_DRAW[i]; gm = CO_GM;
			pos = vec3(r * cos(lam), 0.0, r * sin(lam));
		} else if (i == 2) {
			float M = pow(SURF_A, -1.5) * t_sim + phiS;
			float r = SURF_A * (1.0 - SURF_E * cos(3.0 * M));
			rad = 0.06; gm = SURF_GM;
			pos = vec3(r * cos(M), 0.0, r * sin(M));
		} else {
			float M = pow(PLG_A, -1.5) * t_sim + 1.0;
			float E = M;
			for (int it = 0; it < 6; it++) E = M + PLG_E * sin(E);
			float r = PLG_A * (1.0 - PLG_E * cos(E));
			float nu = 2.0 * atan(sqrt(1.0 + PLG_E) * sin(0.5 * E), sqrt(1.0 - PLG_E) * cos(0.5 * E));
			rad = 0.065; gm = PLG_GM;
			pos = vec3(r * cos(nu), r * sin(nu) * sin(PLG_I), r * sin(nu) * cos(PLG_I));
		}

		if (st < 0.5 && shakeOver > 0.0 && !first
		    && hash11(float(i) * 13.7 + floor(itime * 60.0) * 0.731) < min(SHAKE_FREE_P * shakeOver, 1.0)) {
			float r = max(length(pos.xz), 0.1);
			vec3 v0 = sqrt(1.0 / r) * vec3(-pos.z, 0.0, pos.x) / r;
			st = 1.0; m2 = vec4(pos, 1.0); m3 = vec4(v0 - b2d(SHAKE_K * linear) * h, 0.0);
			M2[i] = m2; M3[i] = m3;
		}
		if (st > 3.5) {
			float k = clamp((t_sim - m3.x) / RS_BLEND, 0.0, 1.0);
			pos += m2.xyz * (1.0 - k) * (1.0 - k);
			if (k >= 1.0) { st = 0.0; M2[i] = vec4(pos, 0.0); M3[i] = vec4(0.0); }
		} else if (st > 2.5) {
			pos = m2.xyz;
		} else if (st > 1.5 && SPIT_ON > 0 && c6.x > 2.5 && int(c6.z + 0.5) - 1 == i) {
			float rr = (i == 2) ? SURF_A : CO_A, rp = DW_R + rad + 0.01;
			float lam = atan(pos.z, pos.x) + PI;
			vec3 x = vec3(rp * cos(lam), 0.0, rp * sin(lam));
			float vp = sqrt(2.0 * rr / (rp * (rp + rr)));
			vec3 v = vp * vec3(-sin(lam), 0.0, cos(lam));
			st = 1.0; pos = x;
			M2[i] = vec4(x, 1.0); M3[i] = vec4(v, 0.0);
			spit = vec4(itime, lam, float(i + 1), 0.0);
			c6 = vec4(0.0);
		} else if (st > 1.5) {
			pos = vec3(0.0, -60.0, 0.0); rad = 0.0; gm = 0.0;
		} else if (st > 0.5) {
			vec3 x = m2.xyz, v = m3.xyz;
			vec3 aFr = b2d(aEnv - aD);
			float r0 = max(length(x), 0.1);
			v += 0.5 * h * (-x / (r0 * r0 * r0) + aFr);
			float yPrev = x.y;
			x += h * v;
			float r1 = max(length(x), 0.1);
			v += 0.5 * h * (-x / (r1 * r1 * r1) + aFr);
			{
				float rp = length(x.xz);
				if (yPrev * x.y < 0.0 && rp > 1.35 && rp < 2.20) c15 = vec4(x.x, x.z, 2.0 * SPLAT_V / SPLAT_T / SIM_RATE, itime);
			}
			{
				vec3 pb = d2b(x) + D, vb = d2b(v) + vD;
				vec3 limV = BOX_HALF - vec3(max(rad, 0.01));
				for (int ax = 0; ax < 3; ax++) {
					float lim = limV[ax];
					if (pb[ax] > lim && vb[ax] > 0.0) { pb[ax] = lim; vb[ax] = -MOON_BOUNCE * vb[ax]; }
					if (pb[ax] < -lim && vb[ax] < 0.0) { pb[ax] = -lim; vb[ax] = -MOON_BOUNCE * vb[ax]; }
				}
				x = b2d(pb - D); v = b2d(vb - vD);
				r1 = max(length(x), 0.1);
			}
			if (!(dot(x, x) < 1e6) || !(dot(v, v) < 1e6)) { x = vec3(1.9, 0.0, 0.0); v = vec3(0.0); }
			if (r1 < DW_R + rad) {
				st = 2.0; pos = vec3(0.0, -60.0, 0.0); rad = 0.0; gm = 0.0;
				v = vec3(itime, 0.0, 0.0);
			} else {
				pos = x;
			}
			M2[i] = vec4(x, st);
			M3[i] = vec4(v, 0.0);
		}

		float a_ref = max(length(pos), 1.4);
		float soft = 0.3 * a_ref * pow(max(gm, 1e-9) / 3.0, 0.333333) + 2e-3;
		M0[i] = vec4(pos, rad);
		M1[i] = vec4(gm, soft, sinkOvr, (st > 2.5 && st < 3.5) ? 1.0 : 0.0);
	}

	float dt4 = dt * 0.25;
	for (int s = 0; s < 4; s++) {
		vec3 k1 = rockOmegaDot(rw);
		vec3 k2 = rockOmegaDot(rw + 0.5 * dt4 * k1);
		vec3 k3 = rockOmegaDot(rw + 0.5 * dt4 * k2);
		vec3 k4 = rockOmegaDot(rw + dt4 * k3);
		rw += (dt4 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
		float th = length(rw) * dt4;
		if (th > 1e-6) {
			vec3 k = rw / (th / dt4);
			mat3 K = mat3(0.0, k.z, -k.y, -k.z, 0.0, k.x, k.y, -k.x, 0.0);
			rR = rR * (mat3(1.0) + sin(th) * K + (1.0 - cos(th)) * (K * K));
		}
	}
	vec3 u0 = normalize(rR[0]);
	vec3 u1 = normalize(rR[1] - dot(rR[1], u0) * u0);
	vec3 u2 = cross(u0, u1);
	rR = mat3(u0, u1, u2);

	vec3 bob = vec3(0.05 * sin(0.31 * itime), 0.04 * sin(0.29 * itime + 1.7), 0.035 * sin(0.23 * itime + 3.4));
	vec3 rockPos = b2d(vec3(-0.8, 2.3, 1.2) + d2b(bob) - D);   /* (inside the narrower box: x 1.32, y 2.94) */

	float pbc = c33.z, shipFlare = 0.0;
	vec3 shipP[N_SHIPS], shipD[N_SHIPS];
	vec2 beamDir;
	{
		float bcNow = (beatConf > 0.2 && beatBpm > 30.0) ? beatCount : itime * SHIP_FALLBACK_BPM / 60.0;
		if (first) pbc = bcNow;
		float bpm = (beatConf > 0.2 && beatBpm > 30.0) ? beatBpm : SHIP_FALLBACK_BPM;
		pbc += dt * bpm / 60.0;
		float err = bcNow - pbc;
		err -= PS_BEATS * floor(err / PS_BEATS + 0.5);
		pbc += clamp(err, -PS_SLEW * dt, PS_SLEW * dt);
		shipFlare = pow(1.0 - fract(pbc), 4.0);
		float ab = TWO_PI * pbc / PS_BEATS;
		beamDir = vec2(cos(ab), sin(ab));
		float u = pbc / PS_BEATS;
		float lead = PS_WA * 0.7 * sin(TWO_PI * u * 0.37 + 0.6);
		float rr = mix(SHIP_R0, SHIP_R1, 0.5 + 0.5 * sin(TWO_PI * u * 0.61 + 1.9));
		float hh = SHIP_H * sin(TWO_PI * u * 0.83 + 0.3);
		float az = ab + lead;
		vec3 P0 = vec3(rr * cos(az), hh, rr * sin(az));
		float u1 = (pbc - dt * bpm / 60.0) / PS_BEATS;
		float lead1 = PS_WA * 0.7 * sin(TWO_PI * u1 * 0.37 + 0.6);
		float rr1 = mix(SHIP_R0, SHIP_R1, 0.5 + 0.5 * sin(TWO_PI * u1 * 0.61 + 1.9));
		float hh1 = SHIP_H * sin(TWO_PI * u1 * 0.83 + 0.3);
		float az1 = TWO_PI * u1 + lead1;
		vec3 P0p = vec3(rr1 * cos(az1), hh1, rr1 * sin(az1));
		vec3 D0 = P0 - P0p;
		D0 = dot(D0, D0) > 1e-12 ? normalize(D0) : vec3(-sin(az), 0.0, cos(az));
		shipP[0] = P0; shipD[0] = D0;
		shipP[1] = -P0; shipD[1] = -D0;
		float offAx = abs(lead) + atan(abs(hh), rr);
		c33.x = max(PS_JET, offAx + PS_JET_PAD);
		c33.y = c33.x;
		c33.z = pbc;
	}
	int shipHead = int(c31.z + 0.5);
	bool shipSample = first || (itime - c33.w) >= SHIP_HIST_DT;
	if (shipSample) { shipHead = (shipHead + 1) % CH_N; c33.w = itime; }
	if (first) { shipHead = 0; }
	vec4 shipBound[N_SHIPS]; float shipNb[N_SHIPS];
	for (int i = 0; i < N_SHIPS; i++) {
		int nb = 0;
		vec3 B = shipP[i];
		for (int k = 1; k < SHP_N; k++) {
			int idx = shipSample ? k - 1 : k;
			if (idx >= SHP_N) break;
			vec4 e = first ? vec4(shipP[i], -1.0e4) : SHP[i * SHP_N + idx];
			if (itime - e.w > SHIP_TRAIL_S) break;
			nb = k; B = e.xyz;
		}
		vec3 Cb = 0.5 * (shipP[i] + B);
		shipBound[i] = vec4(Cb, 0.5 * length(shipP[i] - B) + 4.0 * SHIP_L + 0.05);
		shipNb[i] = float(nb);
	}

	int mh = int(c28.x + 0.5) - 1;
	float tMet = c28.y;
	bool metLaunch = false;
	vec3 fingerP;
	{
		vec3 dB = d2b(rdG);
		float zF = BOX_HALF.z - 0.1;
		float tf = (zF - camB.z) / min(dB.z, -1e-4);
		fingerP = b2d(camB + dB * tf - D);
	}
	if (justDown && gi < 0 && mh < 0 && pointerCount == 1 && !screenLocked) {
		int slot = 0; float oldest = 1e9;
		for (int k = MET_FINGER_N - 1; k >= 0; k--) {
			if (MA[k].x < 0.5 || MA[k].x > 2.5) slot = k;
		}
		bool anyFree = false;
		for (int k = 0; k < MET_FINGER_N; k++) if (MA[k].x < 0.5 || MA[k].x > 2.5) anyFree = true;
		if (!anyFree) for (int k = 0; k < MET_FINGER_N; k++) if (MA[k].w < oldest) { oldest = MA[k].w; slot = k; }
		mh = slot; tMet = itime;
		MA[slot] = vec4(1.0, -100.0, 0.0, itime); MX[slot] = vec4(fingerP, 0.0); MV[slot] = vec4(0.0); c29 = vec4(0.0);
	}
	if (mh >= 0) {
		if (pointerCount == 1 && !screenLocked) {
			c29.xyz = mix(c29.xyz, (fingerP - MX[mh].xyz) / max(h, 1e-4), FLING_SMOOTH);
			MX[mh] = vec4(fingerP, 0.0); MV[mh] = vec4(c29.xyz, 0.0);
		} else {
			if (itime - tMet < TAP_S && tSwipe < 0.5) MA[mh].x = 0.0;
			else {
				vec3 v = c29.xyz * FLING_K;
				float vl = length(v);
				if (vl > FLING_MAX) v *= FLING_MAX / vl;
				MA[mh].x = 2.0; MV[mh] = vec4(v, 0.0); metLaunch = true;
			}
			mh = -1;
		}
	}
	vec3 aFrM = b2d(-SHAKE_K * linear - aD);
	for (int k = 0; k < MET_N; k++) {
		if (MA[k].x < 1.5 || MA[k].x > 2.5) continue;
		vec3 x = MX[k].xyz, v = MV[k].xyz;
		float rr0 = max(length(x) - DW_R, 0.004);
		int n = clamp(int(ceil(h * length(v) / max(0.04 * rr0, 0.002))), 1, 10);
		float hs = h / float(n);
		bool dead = false;
		for (int sstep = 0; sstep < 10; sstep++) {
			if (sstep >= n || dead) break;
			for (int half_ = 0; half_ < 2; half_++) {
				vec3 a = -MET_DWARF_G * x / pow(max(length(x), 0.05), 3.0) + aFrM;
				if (k < MET_FINGER_N) for (int j = 0; j < N_MOONS; j++) {
					if (M0[j].w <= 0.0) continue;
					vec3 dj = M0[j].xyz - x;
					float q = dot(dj, dj) + 0.0001;
					a += MET_MOON_BOOST * M1[j].x * dj / (q * sqrt(q));
				}
				v += 0.5 * hs * a;
				if (half_ == 0) {
					float y0 = x.y;
					x += hs * v;
					float rp = length(vec2(x.x, x.z));
					if (y0 * x.y < 0.0 && rp > 1.35 && rp < 2.20)
						c15 = vec4(x.x, x.z, 2.0 * SPLAT_V / SPLAT_T / SIM_RATE, itime);
				}
			}
			float rx = length(x);
			if (rx < DW_R + MET_R) {
				MB[k] = vec4(x / max(rx, 1e-4) * DW_R, 0.12); dead = true;
				if (k < MET_FINGER_N) vD += d2b(v) * MET_DWARF_KICK;
				else vD += d2b(normalize(v)) * SHOT_DWARF_KICK;
				if (int(MT[k].y + 0.5) == 7) { c6.w += 1.0; MT[k] = vec4(0.0); }
				break;
			}
			if (dead) break;
			vec3 pb = d2b(x) + D;
			if (any(greaterThan(abs(pb), BOX_HALF - vec3(MET_R)))) {
				MB[k] = vec4(b2d(clamp(pb, -BOX_HALF, BOX_HALF) - D), 0.05); dead = true; break;
			}
		}
		if (!dead) {
			int tgt = int(MT[k].x + 0.5) - 1;
				for (int j = 0; j < N_MOONS; j++) {
					if (M0[j].w <= 0.0) continue;
					float hitR = (j == tgt) ? M0[j].w + RS_HITR : M0[j].w + MET_R;
					if (length(x - M0[j].xyz) > hitR) continue;
					vec3 xm = M0[j].xyz;
					if (j == tgt && int(MT[k].y + 0.5) == 6) {
						c49 = vec4(xm.x, xm.z, c49.z + FM_PUFF, float(frame)); ringB = min(ringB + FM_DB, 1.2);
						MT[k] = vec4(0.0); MB[k] = vec4(x, 0.08); dead = true; break;
					}
					if (j == tgt) {
						int kind = int(MT[k].y + 0.5);
						float rrail = (j == 2) ? SURF_A : CO_A;
						float lam = atan(xm.z, xm.x), rxz = length(xm.xz);
						if (kind == 4 && abs(rxz - rrail) < RS_TOL && abs(xm.y) < RS_TOL) {
							float rr2 = rrail;
							if (j == 2) phiS = lam - pow(SURF_A, -1.5) * t_sim;
							else {
								cp[j] = mod(lam - CO_N * t_sim, TWO_PI);
								cw[j] = clamp(1.5 * CO_N * (1.0 - rxz / CO_A), -1.5 * CO_N * 0.06 / CO_A, 1.5 * CO_N * 0.06 / CO_A);
								rr2 = CO_A - cw[j] * CO_A / (1.5 * CO_N);
							}
							vec3 railP = vec3(rr2 * cos(lam), 0.0, rr2 * sin(lam));
							M2[j] = vec4(xm - railP, 4.0); M3[j] = vec4(t_sim, 0.0, 0.0, 0.0);
							RSa[j] = vec4(0.0); RSf[j].w = -100.0;
						} else {
							M2[j] = vec4(xm, 1.0); M3[j] = vec4(MW[k].xyz, 0.0);
							if (kind == 2) { RSa[j] = vec4(3.0, RSe[j].w, 0.0, 0.0); RSd[j] = vec4(0.0); }
							else { RSa[j] = vec4(0.0); RSf[j].w = -100.0; }
						}
						MT[k] = vec4(0.0);
					} else {
						vec3 vm;
						if (M2[j].w > 0.5 && M2[j].w < 1.5) vm = M3[j].xyz;
						else { float r = max(length(xm.xz), 0.1); vm = sqrt(1.0 / r) * vec3(-xm.z, 0.0, xm.x) / r; }
						M2[j] = vec4(xm, 1.0); M3[j] = vec4(vm + MET_PUSH * v, 0.0);
					}
					MB[k] = vec4(x, 0.06); dead = true; break;
				}
		}
		if (!(dot(x, x) < 1e6) || !(dot(v, v) < 1e6)) { dead = true; MB[k] = vec4(0.0); }
		if (dead) {
			MA[k].x = 3.0; MA[k].y = itime;
			int tgt = int(MT[k].x + 0.5) - 1;
			if (tgt >= 0 && int(MT[k].y + 0.5) != 6 && int(MT[k].y + 0.5) != 7) { RSa[tgt] = vec4(0.0); RSf[tgt].w = itime; }
			MT[k] = vec4(0.0);
		}
		if (itime - MA[k].w > MET_LIFE && !dead) MA[k].x = 0.0;
		MX[k] = vec4(x, 0.0); MV[k] = vec4(v, 0.0); MA[k].z = length(v);
	}

	if (justTapped && !grabbedTouch && !metLaunch) {
		vec2 uvT = (tapXY - 0.5 * resolution) / min(resolution.x, resolution.y) * 2.0 * FOV;
		vec3 rd = normalize(fw + uvT.x * rt + uvT.y * up);
		int bestIdx = -1;
		float bestAng = GRAB_CONE;
		for (int i = 0; i < N_MOONS; i++) {
			vec4 m = M0[i];
			if (m.w <= 0.0) continue;
			vec3 rel = m.xyz - ro;
			float d = length(rel);
			float ang = acos(clamp(dot(rel / d, rd), -1.0, 1.0)) - asin(clamp(m.w / d, 0.0, 1.0));
			if (ang < bestAng) { bestAng = ang; bestIdx = i; }
		}
		vec3 relRk = rockPos - ro;
		float dRk = length(relRk);
		float angRk = acos(clamp(dot(relRk / dRk, rd), -1.0, 1.0)) - asin(clamp(0.20 / dRk, 0.0, 1.0));
#if ROCK_ON
		if (angRk < bestAng) { bestAng = angRk; bestIdx = N_MOONS; }
#endif

		if (bestIdx >= 0 && bestIdx < N_MOONS) {
			vec3 x = M0[bestIdx].xyz;
			vec3 v;
			if (M2[bestIdx].w > 0.5) {
				v = M3[bestIdx].xyz;
			} else {
				float r = max(length(x.xz), 0.1);
				v = sqrt(1.0 / r) * vec3(-x.z, 0.0, x.x) / r;
			}
			v += rd * BOOP_DV_MOON;
			M2[bestIdx] = vec4(x, 1.0);
			M3[bestIdx] = vec4(v, 0.0);
		} else if (bestIdx == N_MOONS) {
			rw += (hash33(vec3(itime, tapXY)) - 0.5) * 2.0 * BOOP_SPIN;
		} else if (dot(ro, rd) < 0.0 && dot(ro, rd) * dot(ro, rd) - (dot(ro, ro) - DW_R * DW_R) > 0.0) {
			vD += d2b(rd) * DW_BOOP;
		} else if (abs(rd.y) > 1e-4) {
			float tr = -ro.y / rd.y;
			vec3 P = ro + rd * tr;
			float r = length(P.xz);
			if (tr > 0.0 && r > R_IN && r < R_OUT) {
				c15 = vec4(P.x, P.z, SPLAT_V / SPLAT_T / SIM_RATE, itime);
			}
		}
	}

	float churn = c1.y, prevB = c1.z;
	if (!first && beatConf > 0.2 && beatBpm > 30.0) {
		float dB = beatCount - prevB;
		if (dB < -512.0) dB += 1024.0;
		if (dB > 0.0 && dB < 2.0) {
			float f0 = fract(prevB), f1 = fract(beatCount);
			float e0 = floor(prevB) + 1.0 - pow(1.0 - f0, 4.0);
			float e1 = floor(beatCount) + 1.0 - pow(1.0 - f1, 4.0);
			churn = mod(churn + 0.045 * max(e1 - e0, 0.0), 10000.0);
		}
	}

	if (c15.w == itime) ringB = max(ringB - HOLE_DB, 0.0);

	ctlW(0, vec4(1.0, t_sim, itime, dt));
	ctlW(1, vec4(h, churn, beatCount, 0.0));
	ctlW(2, vec4(yaw, yawV, tDown, tx0));
	ctlW(3, vec4(txp, tSwipe, ty0, typ));
	ctlW(8, cp);
	ctlW(9, cw);
	ctlW(10, vec4(rockPos, 0.0));
	ctlW(11, vec4(rw, 0.0));
	ctlW(12, vec4(rR[0], 0.0));
	ctlW(13, vec4(rR[1], 0.0));
	ctlW(14, vec4(rR[2], 0.0));
	ctlW(15, c15);
	ctlW(16, vec4(camB, pinch));
	ctlW(24, vec4(D, 0.0));
	ctlW(28, vec4(float(mh + 1), tMet, 0.0, 0.0));
	ctlW(29, c29);
	{
		int hk = frame % MET_HIST;
		for (int k = 0; k < MET_N; k++) {
			metW4(k, 0, MA[k]); metW4(k, 4, MB[k]); metW4(k, 8, MX[k]); metW4(k, 12, MV[k]); metW4(k, 20, MT[k]); metW4(k, 24, MW[k]);
			bool live = MA[k].x > 0.5 && MA[k].x < 2.5;
			metW4(k, 16, vec4(MX[k].xyz, MA[k].z * MET_TAIL_S + 0.12));
			vec4 hv = live ? vec4(MX[k].xyz, itime) : vec4(0.0, 0.0, 0.0, -1.0e4);
			imageStore(met, ivec2(4 * hk, k), vec4(hv.x)); imageStore(met, ivec2(4 * hk + 1, k), vec4(hv.y));
			imageStore(met, ivec2(4 * hk + 2, k), vec4(hv.z)); imageStore(met, ivec2(4 * hk + 3, k), vec4(hv.w));
		}
	}
	{
		for (int i = 0; i < N_SHIPS; i++) {
			if (shipSample) {
				imageStore(ships, ivec2(4 * shipHead, i), vec4(shipP[i].x)); imageStore(ships, ivec2(4 * shipHead + 1, i), vec4(shipP[i].y));
				imageStore(ships, ivec2(4 * shipHead + 2, i), vec4(shipP[i].z)); imageStore(ships, ivec2(4 * shipHead + 3, i), vec4(itime));
			}
			if (first) for (int k = 1; k < CH_N; k++) {
				imageStore(ships, ivec2(4 * k, i), vec4(shipP[i].x)); imageStore(ships, ivec2(4 * k + 1, i), vec4(shipP[i].y));
				imageStore(ships, ivec2(4 * k + 2, i), vec4(shipP[i].z)); imageStore(ships, ivec2(4 * k + 3, i), vec4(-1.0e4));
			}
			imageStore(ships, ivec2(4 * CH_N, i), vec4(shipFlare)); imageStore(ships, ivec2(4 * CH_N + 1, i), vec4(float(shipHead)));
			imageStore(ships, ivec2(4 * CH_N + 2, i), vec4(float(CH_N))); imageStore(ships, ivec2(4 * CH_N + 3, i), vec4(0.0));
			imageStore(ships, ivec2(4 * (CH_N + 1), i), vec4(shipD[i].x)); imageStore(ships, ivec2(4 * (CH_N + 1) + 1, i), vec4(shipD[i].y));
			imageStore(ships, ivec2(4 * (CH_N + 1) + 2, i), vec4(shipD[i].z)); imageStore(ships, ivec2(4 * (CH_N + 1) + 3, i), vec4(0.0));
		}
		ctlW(31, vec4(beamDir.x, beamDir.y, float(shipHead), shipFlare));
		ctlW(32, vec4(PS_REACH, length(shipP[0]), 0.0, PS_WA));
		ctlW(33, c33);
		ctlW(34, vec4(shipP[0], 0.0));
		ctlW(35, vec4(shipP[1], 0.0));
		ctlW(36, shipBound[0]);
		ctlW(37, shipBound[1]);
		ctlW(38, vec4(shipNb[0], shipNb[1], 0.0, 0.0));
		ctlW(58, vec4(phiS, 0.0, 0.0, 0.0));
		ctlW(61, c49);
		if (first) { ctlW(60, c48); ctlW(62, c50); ctlW(63, c51); }
		ctlW(4, vec4(ringB, 0.0, 0.0, 0.0));
		ctlW(5, spit);
		ctlW(6, c6);
		for (int i = 0; i < N_MOONS; i++) { ctlW(40 + 6 * i, RSa[i]); ctlW(41 + 6 * i, RSb[i]); ctlW(42 + 6 * i, RSc[i]); ctlW(43 + 6 * i, RSd[i]); ctlW(44 + 6 * i, RSe[i]); ctlW(45 + 6 * i, RSf[i]); }
	}
	ctlW(27, vec4(DIO_LAYOUT, 0.0, 0.0, 0.0));
	{
		float nVis = 0.0;
		for (int k = 0; k < MET_N; k++)
			if ((MA[k].x > 0.5 && MA[k].x < 2.5) || (MA[k].x > 2.5 && itime - MA[k].y < 1.5)) nVis += 1.0;
		ctlW(30, vec4(nVis, 0.0, 0.0, 0.0));
	}
	ctlW(25, vec4(vD, 0.0));
	ctlW(26, vec4(ro, 0.0));
	ctlW(20, vec4(float(gi + 1), grabD, tGrab, grabbedTouch ? 1.0 : 0.0));
	ctlW(21, c21);
	ctlW(22, c22);
	ctlW(17, vec4(fw, 0.0));
	ctlW(18, vec4(rt, 0.0));
	ctlW(19, vec4(up, scr));
	for (int i = 0; i < N_MOONS; i++) {
		moonW(i, 0, M0[i]);
		moonW(i, 1, M1[i]);
		moonW(i, 2, M2[i]);
		moonW(i, 3, M3[i]);
	}
}

==compute plan 1==
layout(local_size_x = 1) in;
layout(r32f, binding = ctl_UNIT) uniform highp image2D ctl;
layout(r32f, binding = moons_UNIT) uniform readonly highp image2D moons;
layout(r32f, binding = met_UNIT) uniform highp image2D met;
#define metL(i, c) imageLoad(met, ivec2(c, MET_N + (i))).r
#define metL4(i, c) vec4(metL(i, c), metL(i, (c) + 1), metL(i, (c) + 2), metL(i, (c) + 3))
#define metW4(i, c, v) do { vec4 _w = (v); imageStore(met, ivec2(c, MET_N + (i)), vec4(_w.x)); imageStore(met, ivec2((c) + 1, MET_N + (i)), vec4(_w.y)); imageStore(met, ivec2((c) + 2, MET_N + (i)), vec4(_w.z)); imageStore(met, ivec2((c) + 3, MET_N + (i)), vec4(_w.w)); } while (false)

void cradleStep(inout vec3 D, inout vec3 vD, vec3 aEnv, float hs, out vec3 aD) {
	aD = wallAcc(D) + aEnv - DW_DAMP * vD;
	vD += hs * aD;
	D += hs * vD;
	D = clamp(D, -BOX_HALF + vec3(DW_R + 0.05), BOX_HALF - vec3(DW_R + 0.05));
}
vec3 flyBody(int mode, vec3 x, vec3 v, float T, vec3 aEnv, vec3 aShk, vec3 D, vec3 vD, float rad, out vec3 vOut, out vec2 info) {
	int n = clamp(int(ceil(T / RS_H)), 1, RS_NMAX);
	float hs = T / float(n);
	vec3 limV = BOX_HALF - vec3(max(rad, 0.01));
	float G = mode == 0 ? 1.0 : MET_DWARF_G, rmin = mode == 0 ? 0.1 : 0.05;
	info = vec2(length(x), 0.0);
	for (int s = 0; s < n; s++) {
		vec3 aD;
		cradleStep(D, vD, aEnv, hs, aD);
		vec3 aFr = b2d((mode == 0 ? aEnv : aShk) - aD);
		float r0 = max(length(x), rmin);
		v += 0.5 * hs * (-G * x / (r0 * r0 * r0) + aFr);
		x += hs * v;
		float r1 = max(length(x), rmin);
		v += 0.5 * hs * (-G * x / (r1 * r1 * r1) + aFr);
		info.x = min(info.x, r1);
		if (mode == 0) {
			vec3 pb = d2b(x) + D, vb = d2b(v) + vD;
			for (int ax = 0; ax < 3; ax++) {
				float lim = limV[ax];
				if (pb[ax] > lim && vb[ax] > 0.0) { pb[ax] = lim; vb[ax] = -MOON_BOUNCE * vb[ax]; info.y += 1.0; }
				if (pb[ax] < -lim && vb[ax] < 0.0) { pb[ax] = -lim; vb[ax] = -MOON_BOUNCE * vb[ax]; info.y += 1.0; }
			}
			x = b2d(pb - D); v = b2d(vb - vD);
		} else if (r1 < DW_R) { vOut = v; return x + vec3(0.0, 9.0, 0.0); }
	}
	vOut = v;
	return x;
}
vec3 newtonStep(int mode, vec3 S, vec3 v, vec3 X, float T, vec3 aEnv, vec3 aShk, vec3 D, vec3 vD, float rad, out float err, out vec2 info) {
	vec3 f = vec3(0.0), vo; vec2 inf0 = vec2(0.0), infA;
	mat3 J;
	for (int a = 0; a < 4; a++) {
		vec3 dv = vec3(0.0);
		if (a < 3) dv[a] = RS_EPS;
		vec3 e = flyBody(mode, S, v + dv, T, aEnv, aShk, D, vD, rad, vo, infA) - X;
		if (a == 3) { f = e; inf0 = infA; } else J[a] = e;
	}
	for (int a = 0; a < 3; a++) J[a] = (J[a] - f) / RS_EPS;
	err = length(f); info = inf0;
	vec3 stp = inverse(J) * f;
	if (!(dot(stp, stp) < 1e6)) stp = vec3(0.0);
	float cap = mode == 0 ? RS_STEP : 4.0 * RS_STEP, sl = length(stp);
	if (sl > cap) stp *= cap / sl;
	vec3 v1 = v - stp;
	vec2 inf1;
	float err1 = length(flyBody(mode, S, v1, T, aEnv, aShk, D, vD, rad, vo, inf1) - X);
	if (err1 < err) { err = err1; info = inf1; return v1; }
	return v;
}
void railTarget(int i, vec3 X1, float tau, out vec3 X, out vec3 V) {
	float rr = (i == 2) ? SURF_A : CO_A;
	float th = atan(X1.z, X1.x) + pow(rr, -1.5) * tau;
	vec2 u = vec2(cos(th), sin(th));
	X = vec3(rr * u.x, 0.0, rr * u.y);
	V = sqrt(1.0 / rr) * vec3(-u.y, 0.0, u.x);
}
float rsTau(int c) { return c <= 0 ? RS_TAU0 : (c == 1 ? RS_TAU1 : RS_TAU2); }
float pickShip(vec3 X, vec3 S0, vec3 S1) {
	vec3 d0 = X - S0, d1 = X - S1;
	float t0 = clamp(-dot(S0, d0) / max(dot(d0, d0), 1e-6), 0.0, 1.0), t1 = clamp(-dot(S1, d1) / max(dot(d1, d1), 1e-6), 0.0, 1.0);
	bool c0 = length(S0 + d0 * t0) > DW_R + 0.12, c1 = length(S1 + d1 * t1) > DW_R + 0.12;
	if (c0 != c1) return c0 ? 0.0 : 1.0;
	return length(d0) <= length(d1) ? 0.0 : 1.0;
}
vec3 railPosAt(int i, float T, float tNow, vec4 cp, vec4 cw, float phiS) {
	if (i == 2) {
		float M = pow(SURF_A, -1.5) * T + phiS;
		float r = SURF_A * (1.0 - SURF_E * cos(3.0 * M));
		return vec3(r * cos(M), 0.0, r * sin(M));
	}
	float dT = max(T - tNow, 0.0);
	int n = clamp(int(ceil(dT / RS_H)), 1, RS_NMAX);
	float hc = dT / float(n) * CO_SPEED;
	for (int s = 0; s < RS_NMAX; s++) {
		if (s >= n) break;
		cw += 0.5 * hc * coAcc(cp);
		cp = mod(cp + hc * cw, TWO_PI);
		cw += 0.5 * hc * coAcc(cp);
	}
	float r = CO_A - cw[i] * CO_A / (1.5 * CO_N), lam = CO_N * T + cp[i];
	return vec3(r * cos(lam), 0.0, r * sin(lam));
}
int shotSlot(vec4 MA[MET_N]) {
	int slot = MET_FINGER_N; float oldest = 1e9; bool any = false;
	for (int k = MET_N - 1; k >= MET_FINGER_N; k--) if (MA[k].x < 0.5 || MA[k].x > 2.5) { slot = k; any = true; }
	if (!any) for (int k = MET_FINGER_N; k < MET_N; k++) if (MA[k].w < oldest) { oldest = MA[k].w; slot = k; }
	return slot;
}

void main() {
	vec4 c0 = ctl4(0);
	if (c0.x < 0.5) return;
	float t_sim = c0.y, itime = c0.z;
	vec4 cp = ctl4(8), cw = ctl4(9);
	vec3 D = ctl4(24).xyz, vD = ctl4(25).xyz;
	vec3 shipP[N_SHIPS];
	shipP[0] = ctl4(34).xyz; shipP[1] = ctl4(35).xyz;
	float pbc = ctl4(33).z;
	float phiS = ctl1(58, 0), ringB = ctl1(4, 0);
	vec4 c47 = ctl4(59), c48 = ctl4(60), c50 = ctl4(62), c51 = ctl4(63);
	vec4 c6 = ctl4(6), c7 = ctl4(7), c23 = ctl4(23);
	vec4 RSa[N_MOONS], RSb[N_MOONS], RSc[N_MOONS], RSd[N_MOONS], RSe[N_MOONS], RSf[N_MOONS];
	for (int i = 0; i < N_MOONS; i++) { RSa[i] = ctl4(40 + 6 * i); RSb[i] = ctl4(41 + 6 * i); RSc[i] = ctl4(42 + 6 * i); RSd[i] = ctl4(43 + 6 * i); RSe[i] = ctl4(44 + 6 * i); RSf[i] = ctl4(45 + 6 * i); }
	vec4 MA[MET_N], MB[MET_N], MX[MET_N], MV[MET_N], MT[MET_N], MW[MET_N];
	for (int k = 0; k < MET_N; k++) { MA[k] = metL4(k, 0); MB[k] = metL4(k, 4); MX[k] = metL4(k, 8); MV[k] = metL4(k, 12); MT[k] = metL4(k, 20); MW[k] = metL4(k, 24); }
	vec4 M0[N_MOONS], M2[N_MOONS], M3[N_MOONS];
	for (int i = 0; i < N_MOONS; i++) { M0[i] = moon4(i, 0); M2[i] = moon4(i, 2); M3[i] = moon4(i, 3); }
	vec3 gB = length(gravity) > 0.5 ? -normalize(gravity) * EARTH_G : vec3(0.0, -EARTH_G, 0.0);
	vec3 aEnv = gB - SHAKE_K * linear;
	vec3 aD = wallAcc(D) + aEnv - DW_DAMP * vD;

	vec3 aShk = -SHAKE_K * linear;
	vec3 aFrR = b2d(aEnv - aD), aFrS = b2d(aShk - aD);
	int jobKind = -1, jobMoon = -1, jobB = 0;
	float jobT = 1e9;
	{
		for (int i = 0; i < N_MOONS; i++) {
			float st = M2[i].w;
			int stage = int(RSa[i].x + 0.5);
			bool loose = st > 0.5 && st < 1.5;
			if (stage > 0 && !loose) { RSa[i] = vec4(0.0); continue; }
			if (stage == 0) {
				if (loose && itime - RSf[i].w >= RS_COOL && jobKind < 0) { jobKind = 4; jobMoon = i; }
				continue;
			}
			float Th = RSa[i].y;
			if (stage == 1) {
				if (Th - t_sim < RS_MINFLY) { RSa[i] = vec4(0.0); RSf[i].w = itime; continue; }
				bool window = RSc[i].w > 0.5 && Th - t_sim <= RS_SHOTFLY + 0.1;
				if (window) { if (jobKind != 1 && jobKind != 2 && jobKind != 3 || Th < jobT) { jobKind = 1; jobMoon = i; jobT = Th; } }
				else if (RSc[i].w < 0.5) { if (jobKind < 0 || jobKind == 4 || (jobKind == 0 && RSc[jobMoon].w > 0.5)) { jobKind = 0; jobMoon = i; } }
				else if (jobKind < 0 || jobKind == 4) { jobKind = 0; jobMoon = i; }
			} else if (stage == 3) {
				float T2 = RSe[i].w;
				if (T2 - t_sim < RS_MINFLY) { RSa[i] = vec4(0.0); RSf[i].w = itime; continue; }
				if (T2 - t_sim <= RS_SHOTFLY + 0.1) { if (jobKind != 1 && jobKind != 2 && jobKind != 3 || T2 < jobT) { jobKind = 2; jobMoon = i; jobT = T2; } }
			} else if (t_sim > Th + RS_LATE) {
				int slot = int(RSa[i].z + 0.5) - 1;
				if (slot >= 0) MT[slot] = vec4(0.0);
				RSa[i] = vec4(0.0); RSf[i].w = itime;
			}
		}
#if FM_ON
		{
			int fst = int(c48.x + 0.5);
			if (fst == 0) {
				if (ringB < FM_LOW && itime - c48.w > FM_COOL) {
					int mi = (c47.z <= c47.y) ? 2 : ((frame & 1) == 0 ? 0 : 1);
					if (M2[mi].w < 0.5) {
						float Tmin = t_sim + RS_SHOTFLY + 0.15 + FM_BURST_DT, Th = Tmin;
						if (beatConf > 0.2 && beatBpm > 30.0) {
							float per = 60.0 / beatBpm * SIM_RATE;
							float Tb = t_sim + (1.0 - fract(pbc)) * 60.0 / beatBpm * SIM_RATE;
							Th = Tb + per * ceil(max(Tmin - Tb, 0.0) / per);
						}
						c48 = vec4(1.0, Th, float(mi + 1), c48.w); c50 = vec4(0.0); c51 = vec4(0.0);
					}
				}
			} else if (fst == 1) {
				float Th = c48.y; int mi = int(c48.z + 0.5) - 1;
				if (M2[mi].w > 0.5 || Th - t_sim < RS_MINFLY) c48 = vec4(0.0, 0.0, 0.0, itime);
				else {
					for (int bb = 0; bb < 2; bb++) {
						float lead = RS_SHOTFLY + 0.1 + (bb == 0 ? FM_BURST_DT : 0.0);
						float stat = bb == 0 ? c50.w : c51.w;
						if (stat < 1.5 && Th - t_sim <= lead && (jobKind != 1 && jobKind != 2 && jobKind != 3 || Th < jobT)) { jobKind = 3; jobMoon = mi; jobB = bb; jobT = Th; }
					}
					if (c50.w > 1.5 && c51.w > 1.5) c48.x = 2.0;
				}
			} else if (t_sim > c48.y + RS_LATE) c48 = vec4(0.0, 0.0, 0.0, itime);
		}
#endif
#if SPIT_ON
		{
			int sst = int(c6.x + 0.5);
			if (sst == 0) {
				for (int i = 0; i < N_MOONS; i++) {
					if (M2[i].w > 1.5 && M2[i].w < 2.5 && itime - M3[i].x > SPIT_DELAY) {
						float Tmin = t_sim + RS_SHOTFLY + 0.15, Th = Tmin;
						if (beatConf > 0.2 && beatBpm > 30.0) {
							float per = 60.0 / beatBpm * SIM_RATE;
							float Tb = t_sim + (1.0 - fract(pbc)) * 60.0 / beatBpm * SIM_RATE;
							Th = Tb + per * ceil(max(Tmin - Tb, 0.0) / per);
						}
						c6 = vec4(1.0, Th, float(i + 1), 0.0); c7 = vec4(0.0); c23 = vec4(0.0);
						break;
					}
				}
			} else if (sst == 1) {
				float Th = c6.y; int mi = int(c6.z + 0.5) - 1;
				if (!(M2[mi].w > 1.5 && M2[mi].w < 2.5) || Th - t_sim < RS_MINFLY) c6 = vec4(0.0);
				else {
					for (int bb = 0; bb < 2; bb++) {
						float stat = bb == 0 ? c7.w : c23.w;
						if (stat < 1.5 && Th - t_sim <= RS_SHOTFLY + 0.1 && (jobKind != 1 && jobKind != 2 && jobKind != 3 && jobKind != 5 || Th < jobT)) { jobKind = 5; jobMoon = mi; jobB = bb; jobT = Th; }
					}
					if (c7.w > 1.5 && c23.w > 1.5) c6.x = 2.0;
				}
			} else if (sst == 2) {
				bool late = t_sim > c6.y + RS_LATE;
				if (c6.w >= 1.5 || (late && c6.w >= 0.5)) c6.x = 3.0;
				else if (late) c6 = vec4(0.0);
			}
		}
#endif
	}
	if (jobKind >= 0) {
		int i = jobMoon;
		vec3 vEnd; vec2 info; float err;
		if (jobKind == 4) {
			float T1 = t_sim + RS_LEAD;
			vec3 X1 = flyBody(0, M2[i].xyz, M3[i].xyz, RS_LEAD, aEnv, aShk, D, vD, M0[i].w, vEnd, info);
			vec3 X2, Vr;
			railTarget(i, X1, RS_TAU0, X2, Vr);
			vec3 v0 = (X2 - X1) / RS_TAU0 - 0.5 * RS_TAU0 * (-X1 / pow(max(length(X1), 0.1), 3.0) + aFrR);
			RSa[i] = vec4(1.0, T1, 0.0, 0.0); RSb[i] = vec4(X1, 0.0); RSc[i] = vec4(v0, 0.0); RSd[i] = vec4(0.0);
			RSe[i] = vec4(X2, T1 + RS_TAU0); RSf[i] = vec4(Vr, RSf[i].w);
		} else if (jobKind == 0) {
			float Th = RSa[i].y;
			vec3 X1 = flyBody(0, M2[i].xyz, M3[i].xyz, Th - t_sim, aEnv, aShk, D, vD, M0[i].w, vEnd, info);
			RSb[i].xyz = X1;
			int c = int(RSd[i].w + 0.5);
			float tau = rsTau(c);
			vec3 X2, Vr;
			railTarget(i, X1, tau, X2, Vr);
			RSe[i] = vec4(X2, Th + tau); RSf[i].xyz = Vr;
			vec3 v1 = newtonStep(0, X1, RSc[i].xyz, X2, tau, aEnv, aShk, D, vD, M0[i].w, err, info);
			RSc[i].xyz = v1;
			if (RSc[i].w < 0.5) {
				RSa[i].w += 1.0;
				if (err < RS_ERR || RSa[i].w >= float(RS_ITERS)) {
					bool ok = err < 3.0 * RS_ERR && info.x > DW_R + M0[i].w + 0.05 && info.y < 0.5 && length(v1) < RS_VMAX;
					if (ok) { RSc[i].w = 1.0; RSa[i].w = 0.0; }
					else if (c < 2) {
						c++; tau = rsTau(c);
						railTarget(i, X1, tau, X2, Vr);
						RSd[i].w = float(c); RSa[i].w = 0.0;
						RSc[i].xyz = (X2 - X1) / tau - 0.5 * tau * (-X1 / pow(max(length(X1), 0.1), 3.0) + aFrR);
						RSe[i] = vec4(X2, Th + tau); RSf[i].xyz = Vr;
					} else { RSa[i] = vec4(0.0); RSf[i].w = itime - RS_COOL + RS_COOL2; }
				}
			}
		} else if (jobKind == 1 || jobKind == 2) {
			float Th = jobKind == 1 ? RSa[i].y : RSe[i].w;
			vec3 X;
			if (jobKind == 1) { X = flyBody(0, M2[i].xyz, M3[i].xyz, Th - t_sim, aEnv, aShk, D, vD, M0[i].w, vEnd, info); RSb[i].xyz = X; }
			else X = RSe[i].xyz;
			if (RSa[i].z < 0.5) {
				RSb[i].w = pickShip(X, shipP[0], shipP[1]);
				int k = int(RSb[i].w + 0.5);
				float Tf = Th - t_sim;
				RSd[i].xyz = (X - shipP[k]) / Tf - 0.5 * Tf * (-MET_DWARF_G * shipP[k] / pow(max(length(shipP[k]), 0.1), 3.0) + aFrS);
				RSa[i].z = 1.0; RSa[i].w = 0.0;
			}
			int k = int(RSb[i].w + 0.5);
			vec3 vm = newtonStep(1, shipP[k], RSd[i].xyz, X, Th - t_sim, aEnv, aShk, D, vD, 0.0, err, info);
			RSd[i].xyz = vm; RSa[i].w += 1.0;
			if (err < RS_ERR || RSa[i].w >= float(RS_ITERS) || Th - t_sim < RS_SHOTFLY - 0.2) {
				int slot = shotSlot(MA);
				MA[slot] = vec4(2.0, -100.0, length(vm), itime); MB[slot] = vec4(0.0); MX[slot] = vec4(shipP[k], 0.0); MV[slot] = vec4(vm, 0.0);
				MT[slot] = vec4(float(i + 1), jobKind == 1 ? 2.0 : 4.0, Th, 0.0); MW[slot] = vec4(jobKind == 1 ? RSc[i].xyz : RSf[i].xyz, 0.0);
				RSa[i] = vec4(jobKind == 1 ? 2.0 : 4.0, Th, float(slot + 1), 0.0);
			}
		}
#if SPIT_ON
		else if (jobKind == 5) {
			float Th = c6.y;
			vec3 S = shipP[jobB];
			vec3 X = normalize(S) * (DW_R + MET_R + 0.04);
			vec4 g = jobB == 0 ? c7 : c23;
			float Tf = Th - t_sim;
			if (g.w < 0.5) g = vec4((X - S) / Tf - 0.5 * Tf * (-MET_DWARF_G * S / pow(max(length(S), 0.1), 3.0) + aFrS), 1.0);
			vec3 vm = newtonStep(1, S, g.xyz, X, Tf, aEnv, aShk, D, vD, 0.0, err, info);
			g.xyz = vm;
			if (err < RS_ERR || Tf < RS_SHOTFLY - 0.2) {
				int slot = shotSlot(MA);
				MA[slot] = vec4(2.0, -100.0, length(vm), itime); MB[slot] = vec4(0.0); MX[slot] = vec4(S, 0.0); MV[slot] = vec4(vm, 0.0);
				MT[slot] = vec4(float(i + 1), 7.0, Th, 0.0); MW[slot] = vec4(0.0);
				g.w = 2.0;
			}
			if (jobB == 0) c7 = g; else c23 = g;
		}
#endif
#if FM_ON
		else if (jobKind == 3) {
			float Th = c48.y;
			vec3 X = railPosAt(i, Th, t_sim, cp, cw, phiS);
			int ks = int(pickShip(X, shipP[0], shipP[1]) + 0.5);
			vec4 g = jobB == 0 ? c50 : c51;
			float lead = RS_SHOTFLY + 0.1 + (jobB == 0 ? FM_BURST_DT : 0.0), Tf = Th - t_sim;
			if (g.w < 0.5) g = vec4((X - shipP[ks]) / Tf - 0.5 * Tf * (-MET_DWARF_G * shipP[ks] / pow(max(length(shipP[ks]), 0.1), 3.0) + aFrS), 1.0);
			vec3 vm = newtonStep(1, shipP[ks], g.xyz, X, Tf, aEnv, aShk, D, vD, 0.0, err, info);
			g.xyz = vm;
			if (err < RS_ERR || Tf < lead - 0.25) {
				int slot = shotSlot(MA);
				MA[slot] = vec4(2.0, -100.0, length(vm), itime); MB[slot] = vec4(0.0); MX[slot] = vec4(shipP[ks], 0.0); MV[slot] = vec4(vm, 0.0);
				MT[slot] = vec4(float(i + 1), 6.0, Th, 0.0); MW[slot] = vec4(0.0);
				g.w = 2.0;
			}
			if (jobB == 0) c50 = g; else c51 = g;
		}
#endif
	}

	for (int i = 0; i < N_MOONS; i++) { ctlW(40 + 6 * i, RSa[i]); ctlW(41 + 6 * i, RSb[i]); ctlW(42 + 6 * i, RSc[i]); ctlW(43 + 6 * i, RSd[i]); ctlW(44 + 6 * i, RSe[i]); ctlW(45 + 6 * i, RSf[i]); }
	ctlW(60, c48); ctlW(62, c50); ctlW(63, c51);
	ctlW(6, c6); ctlW(7, c7); ctlW(23, c23);
	for (int k = MET_FINGER_N; k < MET_N; k++) {
		metW4(k, 0, MA[k]); metW4(k, 4, MB[k]); metW4(k, 8, MX[k]); metW4(k, 12, MV[k]); metW4(k, 20, MT[k]); metW4(k, 24, MW[k]);
	}
	for (int k = 0; k < MET_FINGER_N; k++) metW4(k, 20, MT[k]);
}

==compute phase 320==
layout(local_size_x = 64) in;
layout(r32f, binding = phase_UNIT) uniform highp image2D phase;
layout(r32f, binding = ctl_UNIT) uniform readonly highp image2D ctl;
void main() {
	int j = int(gl_GlobalInvocationID.x);
	if (j >= NR) return;
	float h = ctl1(1, 0);
	float prev = phR(j, 0);
	float now = mod(prev + omegaK(rowR(float(j))) * h, TWO_PI);
	phW(j, now, prev);
}

==compute advect 1024x192==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = ring_out_UNIT) uniform writeonly highp image2D ring_out;
layout(r32f, binding = phase_UNIT) uniform readonly highp image2D phase;
layout(r32f, binding = ctl_UNIT) uniform readonly highp image2D ctl;

vec4 fetchR(int j, int k) {
	return texelFetch(ring, ivec2((k % NT + NT) % NT, clamp(j, 0, NR - 1)), 0);
}
float phiRow(int j, int ch) {
	return phR(clamp(j, 0, NR - 1), ch);
}
vec4 sampleR(float r, float th, int ch) {
	float jf = clamp((r - R_IN) / DR - 0.5, 0.0, float(NR - 1));
	int j0 = int(floor(jf));
	float fj = jf - float(j0);
	vec4 res = vec4(0.0);
	for (int q = 0; q < 2; q++) {
		int j = min(j0 + q, NR - 1);
		float rel = mod(th - phiRow(j, ch), TWO_PI);
		float kf = rel / DTH - 0.5;
		float k0 = floor(kf);
		float fk = kf - k0;
		vec4 v = mix(fetchR(j, int(k0)), fetchR(j, int(k0) + 1), fk);
		res += v * (q == 0 ? 1.0 - fj : fj);
	}
	return res;
}
void main() {
	ivec2 c = ivec2(gl_GlobalInvocationID.xy);
	if (c.x >= NT || c.y >= NR) return;
	float h = ctl1(1, 0);
	float r = rowR(float(c.y));
	float th = (float(c.x) + 0.5) * DTH + phiRow(c.y, 0);
	vec4 F = fetchR(c.y, c.x);
	float rm = clamp(r - 0.5 * F.y * h, R_IN, R_OUT);
	float rd = clamp(r - F.y * h, R_IN, R_OUT);
	float thd = th - (omegaK(rm) + F.z / rm) * h;
	imageStore(ring_out, c, sampleR(rd, thd, 1));
}

==compute force 1024x192==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = ring_out_UNIT) uniform writeonly highp image2D ring_out;
layout(r32f, binding = phase_UNIT) uniform readonly highp image2D phase;
layout(r32f, binding = moons_UNIT) uniform readonly highp image2D moons;
layout(r32f, binding = ctl_UNIT) uniform readonly highp image2D ctl;

#define C_S        0.0015
#define NU         4.0e-6
#define TAU_RELAX  15.0
#define SIG_FLOOR  0.02
#define SIG_MAX    2.5
#define SINK_HILL  0.6
#define V_MAX      0.03
#define V_DAMP     2.0
#define EMPTY_DAMP 0.85
#define HIDE_GAIN  6.0
#define SEEN_GAIN  0.60
#define POOP_K     1.2
#define POOP_AREA  0.091
#define HELD_GM_K     30.0
#define HELD_SINK_K   2.5
#define HELD_SINK_MIN 0.05

vec4 fetchR(int j, int k) {
	return texelFetch(ring, ivec2((k % NT + NT) % NT, clamp(j, 0, NR - 1)), 0);
}
float phiRow0(int j) {
	return phR(clamp(j, 0, NR - 1), 0);
}
vec4 sampleR0(float r, float th) {
	float jf = clamp((r - R_IN) / DR - 0.5, 0.0, float(NR - 1));
	int j0 = int(floor(jf));
	float fj = jf - float(j0);
	vec4 res = vec4(0.0);
	for (int q = 0; q < 2; q++) {
		int j = min(j0 + q, NR - 1);
		float rel = mod(th - phiRow0(j), TWO_PI);
		float kf = rel / DTH - 0.5;
		float k0 = floor(kf);
		float fk = kf - k0;
		vec4 v = mix(fetchR(j, int(k0)), fetchR(j, int(k0) + 1), fk);
		res += v * (q == 0 ? 1.0 - fj : fj);
	}
	return res;
}
shared vec4 sM0[N_MOONS];
shared vec4 sCtl[3];
shared vec4 sM1[N_MOONS];
void main() {
	ivec2 c = ivec2(gl_GlobalInvocationID.xy);
	uint li = gl_LocalInvocationIndex;
	if (li < uint(2 * N_MOONS)) {
		int i = int(li) % N_MOONS;
		if (li < uint(N_MOONS)) sM0[i] = moon4(i, 0); else sM1[i] = moon4(i, 1);
	} else if (li == uint(2 * N_MOONS)) { sCtl[0] = ctl4(15); sCtl[1] = ctl4(61); sCtl[2] = vec4(ctl1(4, 0), ctl1(1, 0), ctl1(0, 2), 0.0); }
	memoryBarrierShared();
	barrier();
	if (c.x >= NT || c.y >= NR) return;
	float r = rowR(float(c.y));
	if (c.y == 0 || c.y == NR - 1) {
		imageStore(ring_out, c, vec4(sigma0(r), 0.0, 0.0, 0.0));
		return;
	}
	float h = sCtl[2].y;
	float itime = sCtl[2].z;
	vec4 splat = sCtl[0];

	float th = (float(c.x) + 0.5) * DTH + phiRow0(c.y);
	vec4 F  = fetchR(c.y, c.x);
	vec4 Fe = fetchR(c.y, c.x + 1), Fw = fetchR(c.y, c.x - 1);
	vec4 Fn = sampleR0(r + DR, th), Fs = sampleR0(r - DR, th);
	float S = F.x, ur = F.y, vp = F.z;
	float Se = max(S, SIG_FLOOR);
	float rdth = r * DTH;
	float dSdr = (Fn.x - Fs.x) / (2.0 * DR), dSdt = (Fe.x - Fw.x) / (2.0 * rdth);
	float durdr = (Fn.y - Fs.y) / (2.0 * DR), dvdt = (Fe.z - Fw.z) / (2.0 * rdth);
	float lapUr = (Fn.y - 2.0 * ur + Fs.y) / (DR * DR) + (Fe.y - 2.0 * ur + Fw.y) / (rdth * rdth);
	float lapV  = (Fn.z - 2.0 * vp + Fs.z) / (DR * DR) + (Fe.z - 2.0 * vp + Fw.z) / (rdth * rdth);

	vec2 er = vec2(cos(th), sin(th)), et = vec2(-er.y, er.x);
	vec3 P = vec3(r * er.x, 0.0, r * er.y);
	float ar = 0.0, at = 0.0, sink = 1.0, poop = 0.0;

	for (int i = 0; i < N_MOONS; i++) {
		vec4 m0 = sM0[i], m1 = sM1[i];
		if (m1.x <= 0.0) continue;
		vec3 d = m0.xyz - P;
		float rH = length(m0.xyz) * pow(m1.x / 3.0, 0.333333);
		float sR = m1.z > 0.0 ? m1.z : SINK_HILL * rH;
		float gk = 1.0;
		if (m1.w > 0.5) { sR = max(sR * HELD_SINK_K, HELD_SINK_MIN); gk = HELD_GM_K; }
		float dlR = length(d) / max(sR, 1e-4);
		sink *= smoothstep(0.5, 1.0, dlR);
		poop += smoothstep(1.0, 1.6, dlR) * (1.0 - smoothstep(2.4, 3.2, dlR));
		float q = dot(d, d) + m1.y * m1.y;
		vec3 a = gk * m1.x * d / (q * sqrt(q));
		ar += dot(a.xz, er);
		at += dot(a.xz, et);
	}
	if (itime - splat.w >= 0.0 && itime - splat.w < 0.15) {
		vec2 sd = P.xz - splat.xy;
		float sw = exp(-dot(sd, sd) / (0.08 * 0.08));
		vec2 sk = normalize(sd + vec2(1e-6)) * splat.z * sw;
		ar += dot(sk, er);
		at += dot(sk, et);
	}

	vec3 camPos = ctl4(26).xyz;
	vec3 dCam = P - camPos;
	float Lc = length(dCam); dCam /= Lc;
	float bC = dot(-camPos, dCam);
	float ccC = dot(camPos, camPos) - bC * bC;
	float refillGain = (ccC < DW_R * DW_R && bC > 0.0 && bC < Lc) ? HIDE_GAIN : SEEN_GAIN;
#if FM_ON
	float ringB = sCtl[2].x;
#else
	float ringB = 1.0;
#endif

	float vK = pow(r, -0.5);
	float nur = ur + h * ((2.0 * vK * vp + vp * vp) / r - C_S * C_S * dSdr / Se + ar + NU * lapUr);
	float nvp = vp + h * (-ur * vK / (2.0 * r) - ur * vp / r - C_S * C_S * dSdt / Se + at + NU * lapV);
	float div = durdr + ur / r + dvdt;
	float nS = S * exp(-clamp(div * h, -0.5, 0.5)) + (sigma0(r) * ringB - S) * (h / TAU_RELAX) * refillGain;
	nS = min(nS * mix(0.9, 1.0, sink) + POOP_K * 0.1 * POOP_AREA * sigma0(r) * poop, SIG_MAX);
#if FM_ON
	if (itime - splat.w >= 0.0 && itime - splat.w < 0.15) {
		vec2 hd = P.xz - splat.xy;
		nS *= 1.0 - HOLE_K * exp(-dot(hd, hd) / (HOLE_R * HOLE_R));
	}
	vec4 puff = sCtl[1];
	if (int(puff.w + 0.5) == frame && puff.z > 0.0) {
		vec2 pd = P.xz - puff.xy;
		nS = min(nS + puff.z * exp(-dot(pd, pd) / (FM_PUFF_R * FM_PUFF_R)), SIG_MAX);
	}
#endif
	if (nS < SIG_FLOOR) { nur *= EMPTY_DAMP; nvp *= EMPTY_DAMP; }
	nur *= exp(-h * V_DAMP); nvp *= exp(-h * V_DAMP);
	nur = clamp(nur, -V_MAX, V_MAX);
	nvp = clamp(nvp, -V_MAX, V_MAX);
	vec4 o = vec4(max(nS, 0.0), nur, nvp, 0.0);
	if (any(isnan(o)) || any(isinf(o))) o = vec4(sigma0(r), 0.0, 0.0, 0.0);
	imageStore(ring_out, c, o);
}

==compute ringsum 1 @8==
layout(local_size_x = 1) in;
layout(r32f, binding = ctl_UNIT) uniform highp image2D ctl;
void main() {
	float sIn = 0.0, sOut = 0.0, rIn = 0.0, rOut = 0.0;
	for (int j = 4; j < NR; j += 8) {
		float r = rowR(float(j)), s0 = sigma0(r), acc = 0.0;
		for (int k = 0; k < NT; k += 16) acc += texelFetch(ring, ivec2(k, j), 0).x;
		acc /= float(NT / 16);
		if (r < 0.5 * (R_IN + R_OUT)) { sIn += acc; rIn += s0; } else { sOut += acc; rOut += s0; }
	}
	ctlW(59, vec4((sIn + sOut) / max(rIn + rOut, 1e-4), sIn / max(rIn, 1e-4), sOut / max(rOut, 1e-4), float(frame)));
}

==scene 0.75==
uniform vec2 resolution;
uniform highp sampler2D ring;
uniform highp sampler2D phase;
uniform highp sampler2D moons;
uniform highp sampler2D ctl;
uniform highp sampler2D met;
uniform highp sampler2D ships;
out vec4 fragColor;

#define TCTL(s, c) texelFetch(ctl, ivec2((s) * 4 + (c), 0), 0).r
#define TCTL4(s) vec4(TCTL(s, 0), TCTL(s, 1), TCTL(s, 2), TCTL(s, 3))
#define TMOON(i, g, c) texelFetch(moons, ivec2((i) * 4 + (c), g), 0).r
#define TMOON4(i, g) vec4(TMOON(i, g, 0), TMOON(i, g, 1), TMOON(i, g, 2), TMOON(i, g, 3))
#define TPH(j, c) texelFetch(phase, ivec2((j) * 2 + (c), 0), 0).r

#define DWARF_STYLE 2
#define SKY_ON      0
#define MOON_CHROME 0
#define HALO_ON     0
#define CF_CHEAP    1

#define DIO_EXPOSURE 2.6
#define RING_TAU_K   0.40
#define RING_BRIGHT  0.50
#define MOON_FILL    0.06
#define TAU_K       0.50
#define TAU_K2      1.60
#define RING_ALB    0.90
#define PLANET_MEAN vec3(0.764, 0.088, 0.993)
#define SECRET_WHITE 0.40
#define LIGHT_COL   mix(PLANET_MEAN, vec3(dot(PLANET_MEAN, vec3(0.2126, 0.7152, 0.0722))), SECRET_WHITE)
#define RING_LIGHT  vec3(dot(PLANET_MEAN, vec3(0.2126, 0.7152, 0.0722)))
#define RING_WHITE  vec3(0.86, 0.92, 1.00)
#define RING_FLUOR  2.8
#define UV_HALO_COL vec3(0.50, 0.16, 1.00)
#define UV_HALO_G   0.55
#define UV_HALO_H   0.16
#define LIGHT_K     5.5
#define DUST_FWD    0.10
#define DUST_G      0.70
#define ROCK_ALB    0.60
#define MULTI_SC    0.85
#define METAL_R0    0.80
#define METAL_TINT  vec3(0.97, 0.98, 1.00)
#define LIMB_U1     0.60
#define LIMB_U2     0.12
#define LIMB_PURP   vec3(0.30, 0.03, 0.34)
#define HAZE_H      0.035

vec3 rotY(vec3 p, float a) { float c = cos(a), s = sin(a); return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z); }

float vnoise(vec3 p) {
	vec3 i = floor(p), f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	return mix(mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), f.x), mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y),
	           mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x), mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}

vec3 plainDwarf(vec3 n, float t) {
	float lat = asin(clamp(n.y, -1.0, 1.0));
	return PLANET_MEAN * (0.85 + 0.15 * sin(lat * 12.0 + 0.05 * t)) * 1.6;
}

float ringSigmaAt(float r, float th) {
	float jf = clamp((r - R_IN) / DR - 0.5, 0.0, float(NR - 1));
	int j0 = int(floor(jf));
	float fj = jf - float(j0);
	float res = 0.0;
	for (int q = 0; q < 2; q++) {
		int j = min(j0 + q, NR - 1);
		float rel = mod(th - TPH(j, 0), TWO_PI);
		float kf = rel / DTH - 0.5;
		int k0 = int(floor(kf));
		float fk = kf - float(k0);
		float a = texelFetch(ring, ivec2((k0 % NT + NT) % NT, j), 0).x;
		float b = texelFetch(ring, ivec2(((k0 + 1) % NT + NT) % NT, j), 0).x;
		res += mix(a, b, fk) * (q == 0 ? 1.0 - fj : fj);
	}
	return res;
}

float lambertPhase(float a) { return (sin(a) + (PI - a) * cos(a)) / PI; }
float hgPhase(float cosT, float g) { float q = 1.0 + g * g - 2.0 * g * cosT; return (1.0 - g * g) / (q * sqrt(q)); }

vec3 sky(vec3 d) {
	vec3 cell = floor(d * 260.0);
	vec3 h = hash33(cell);
	vec3 starCenter = normalize(cell + 0.2 + 0.6 * h);
	float ang = acos(clamp(dot(d, starCenter), -1.0, 1.0)) * 260.0;
	float star = smoothstep(0.32, 0.0, ang) * pow(h.x, 8.0) * 6.0;
	vec3 starCol = mix(vec3(1.0, 0.75, 0.55), vec3(0.70, 0.85, 1.0), h.y) * star;
	float neb = smoothstep(0.35, 0.85, vnoise(d * 2.5 + vec3(1.3, 4.1, 2.7)));
	return starCol + vec3(0.08, 0.02, 0.11) * neb * 0.35;
}

#if MOON_CHROME
vec3 mirrorEnv(vec3 P, vec3 R, float t, float churn) {
	vec3 base;
	float b = dot(P, R), c = dot(P, P) - 1.0, disc = b * b - c;
	float tq = 1e9;
	if (disc > 0.0 && -b - sqrt(disc) > 0.0) {
		tq = -b - sqrt(disc);
		vec3 n = normalize(P + R * tq);
		float om = 1.0 - max(dot(n, -R), 0.0);
		float I = max(1.0 - LIMB_U1 * om - LIMB_U2 * om * om, 0.03);
		base = plainDwarf(n, t) * I;
		base = mix(base, LIMB_PURP * length(base) * 0.9, smoothstep(0.35, 1.0, om) * 0.7);
	} else {
		base = sky(R);
		float hz = length(cross(P, R)) - 1.0;
		if (hz > 0.0 && b < 0.0) base += UV_HALO_COL * UV_HALO_G * exp(-hz / UV_HALO_H);
	}
	if (abs(R.y) > 1e-4) {
		float tR = -P.y / R.y;
		vec3 Q = P + R * tR;
		float r = length(Q.xz);
		if (tR > 0.0 && tR < tq && r > R_IN && r < R_OUT) {
			float sg = ringSigmaAt(r, atan(Q.z, Q.x));
			float trans = exp(-(TAU_K * sg + TAU_K2 * sg * sg) / abs(R.y));
			float beta = asin(clamp(DW_R / r, 0.0, 1.0));
			float top = (2.0 * beta * beta * beta / 3.0) / PI;
			base = base * trans + RING_LIGHT * RING_WHITE * RING_FLUOR * LIGHT_K * RING_ALB * (0.5 / (r * r) + top) * (1.0 - trans);
		}
	}
	return base;
}
#endif

#define TMET(i, c) texelFetch(met, ivec2(c, MET_N + (i)), 0).r
#define TMET4(i, c) vec4(TMET(i, c), TMET(i, (c) + 1), TMET(i, (c) + 2), TMET(i, (c) + 3))
#define THIS(i, k) vec4(texelFetch(met, ivec2(4 * (k), i), 0).r, texelFetch(met, ivec2(4 * (k) + 1, i), 0).r, texelFetch(met, ivec2(4 * (k) + 2, i), 0).r, texelFetch(met, ivec2(4 * (k) + 3, i), 0).r)
#define MT_COL_HEAD vec3(1.00, 0.86, 0.62)
#define MT_COL_BODY vec3(1.00, 0.34, 0.10)
#define MT_COL_DEEP vec3(0.55, 0.09, 0.12)
#define SHOT_COL_HEAD vec3(0.92, 0.85, 1.00)
#define SHOT_COL_BODY vec3(0.70, 0.45, 1.00)
#define SHOT_COL_DEEP vec3(0.30, 0.12, 0.60)
#define MT_CORE_R   0.020
#define MT_GLOW_R   0.050
#define MT_TRAIL_R0 0.006
#define MT_TRAIL_RW 0.025
vec3 meteorsPhone(vec3 ro, vec3 rd, float tVis, float t) {
	vec3 acc = vec3(0.0);
	for (int i = 0; i < MET_N; i++) {
		vec4 inf = TMET4(i, 0);
		float ia = t - inf.y;
		bool live = inf.x > 0.5 && inf.x < 2.5;
		bool flash = inf.x > 2.5 && ia >= 0.0 && ia < 1.5;
		if (!live && !flash) continue;
		bool shipShot = i >= MET_FINGER_N;
		vec3 cH = shipShot ? SHOT_COL_HEAD : MT_COL_HEAD, cB = shipShot ? SHOT_COL_BODY : MT_COL_BODY, cD = shipShot ? SHOT_COL_DEEP : MT_COL_DEEP;
		if (live) {
			vec4 bnd = TMET4(i, 16);
			vec3 oc = ro - bnd.xyz;
			float bq = dot(oc, rd);
			if (dot(oc, oc) - bq * bq < bnd.w * bnd.w) {
				vec3 H = TMET4(i, 8).xyz;
				float tH = dot(H - ro, rd);
				if (tH > 0.0 && tH < tVis) {
					vec3 o = ro + rd * tH - H;
					float d2 = dot(o, o);
					float heat = clamp((inf.z - 0.5) / 1.5, 0.0, 1.0);
					acc += cH * (exp(-d2 / (MT_CORE_R * MT_CORE_R)) * (1.6 + 2.4 * heat)
					                    + exp(-d2 / (MT_GLOW_R * MT_GLOW_R)) * (0.20 + 0.7 * heat));
				}
				for (int k = 0; k < MET_HIST; k++) {
					vec4 a = THIS(i, k), b = THIS(i, (k + MET_HIST - 1) % MET_HIST);
					if (a.w < t - MET_TAIL_S || b.w < t - MET_TAIL_S || a.w <= b.w || a.w - b.w > 0.1) continue;
					vec3 d1 = b.xyz - a.xyz, w0 = ro - a.xyz;
					float aa = dot(d1, d1), bb = dot(rd, d1), dd = dot(rd, w0), ee = dot(d1, w0);
					float den = aa - bb * bb;
					float sc = den > 1e-12 ? clamp((ee - bb * dd) / den, 0.0, 1.0) : 0.0;
					vec3 C = a.xyz + d1 * sc;
					float tc = dot(C - ro, rd);
					if (tc <= 0.0 || tc >= tVis) continue;
					float ag = clamp((t - mix(a.w, b.w, sc)) / MET_TAIL_S, 0.0, 1.0);
					float rr = MT_TRAIL_R0 + MT_TRAIL_RW * ag;
					vec3 off = ro + rd * tc - C;
					float dens = (1.0 - ag) * (1.0 - ag);
					acc += mix(cD, mix(cB, cH, dens), 1.0 - ag) * 0.9 * dens
					     * exp(-dot(off, off) / (rr * rr)) * (MT_TRAIL_R0 / rr);
				}
			}
		}
		if (flash) {
			vec4 ip = TMET4(i, 4);
			float tI = dot(ip.xyz - ro, rd);
			if (tI > 0.0 && tI < tVis + 0.05) {
				vec3 o = ro + rd * tI - ip.xyz;
				float rg = 0.02 + 0.12 * sqrt(ia) * max(ip.w, 0.05) / 0.05 * 0.5;
				acc += mix(MT_COL_BODY, MT_COL_HEAD, exp(-ia / 0.08)) * exp(-ia / 0.35) * 3.0 * exp(-dot(o, o) / (rg * rg));
			}
		}
	}
	return acc;
}

#define SHIP_COL    vec3(0.78, 0.60, 1.00)
#define PS_BEAM_COL vec3(0.66, 0.42, 1.00)
#define PS_BEAM_K   0.18
#define PS_FAN_K    0.11
#define PS_RAY_K    0.5
#define PS_SCAN     55.0
#define SH(k, i) texelFetch(ships, ivec2(4 * (k), i), 0).r
#define SH4(k, i) vec4(SH(k, i), texelFetch(ships, ivec2(4 * (k) + 1, i), 0).r, texelFetch(ships, ivec2(4 * (k) + 2, i), 0).r, texelFetch(ships, ivec2(4 * (k) + 3, i), 0).r)
vec4 shipHist(int i, int head, int k) { return SH4((head - k + CH_N) % CH_N, i); }
vec3 shipDraw(int i, vec3 ro, vec3 rd, float tVis, float tRing, float trRing, float t, float L, float trS, float trR, vec3 col) {
	vec3 acc = vec3(0.0);
	vec4 bnd = TCTL4(36 + i);
	vec3 oc = ro - bnd.xyz;
	float bq = dot(oc, rd);
	if (dot(oc, oc) - bq * bq > bnd.w * bnd.w) return acc;
	vec4 inf = SH4(CH_N, i);
	int head = int(inf.y + 0.5), nh = min(int(inf.z), int(TCTL(38, i)) + 1);
	if (nh == 0) return acc;
	vec3 P = shipHist(i, head, 0).xyz;
	vec3 D = SH4(CH_N + 1, i).xyz;
	vec3 S = normalize(cross(D, vec3(0.0, 1.0, 0.0)) + vec3(1e-6)), N = cross(S, D);
	float dn = dot(rd, N);
	if (abs(dn) > 1e-4) {
		float tp = dot(P - ro, N) / dn;
		if (tp > 0.0 && tp < tVis) {
			vec3 q = ro + rd * tp - P;
			float u = dot(q, D), v = abs(dot(q, S));
			float w = 0.55 * L * (L - u) / (1.6 * L);
			if (u > -0.6 * L && u < L && v < w) {
				float rim = 1.0 - smoothstep(0.0, 0.2 * L, min(w - v, u + 0.6 * L));
				acc += col * (0.35 + 2.6 * rim) + vec3(0.6) * rim;
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
	for (int k = 0; k < CH_N - 1; k++) {
		if (k + 1 >= nh) break;
		vec4 a = shipHist(i, head, k), b = shipHist(i, head, k + 1);
		if ((t - a.w) / trS > 1.0) break;
		if (k == 0) a.xyz = E;
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
		acc += (col * 0.9 + vec3(0.5) * exp(-4.0 * e2)) * (1.0 - ag) * (1.0 - ag) * vis * exp(-e2) * (trR / rr);
	}
	return acc;
}
float holo(vec3 ro, vec3 rd, vec3 P, float t, float flare) {
	float rr = length(ro + rd * dot(P - ro, rd));
	float sc = 0.5 + 0.5 * sin(rr * PS_SCAN * 6.2832 - t * 9.0);
	float fl = 1.0 - 0.35 * flare * fract(sin(floor(t * 40.0) * 91.7) * 4375.5);
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
	vec4 ps = TCTL4(31), pr = TCTL4(32), pw = TCTL4(33);
	float R = pr.x, flare = ps.w;
	for (int k = 0; k < 2; k++) {
		vec3 b = (k == 0 ? 1.0 : -1.0) * vec3(ps.x, 0.0, ps.y);
		float tw = tan(k == 0 ? pw.x : pw.y);
		float bb = dot(rd, b), dd = dot(rd, ro), ee = dot(b, ro);
		float den = 1.0 - bb * bb;
		if (den < 1e-5) continue;
		float sb = (ee - bb * dd) / den, tc = (ee * bb - dd) / den;
		if (sb < 0.1 || sb > R || tc <= 0.0 || tc >= tVis) continue;
		float dist = length(ro + rd * tc - b * sb), cw = max(sb * tw, 0.02);
		float x2 = dist * dist / (cw * cw);
		float g = (exp(-2.0 * x2) + 1.5 * exp(-40.0 * x2)) * smoothstep(0.1, 0.35, sb) * (1.0 - smoothstep(0.8 * R, R, sb));
		float lit = 1.0 + 1.5 * (1.0 - smoothstep(pr.y - 0.1, pr.y + 0.25, sb));
		float vis = tc > tRing ? sqrt(trRing) : 1.0;
		acc += mix(PS_BEAM_COL, vec3(1.0), 0.35 * exp(-40.0 * x2)) * g * lit * vis * (0.04 / cw + 0.3) * PS_BEAM_K * (1.0 + 2.0 * flare);
	}
	if (abs(rd.y) > 1e-3) {
		float t0 = -ro.y / rd.y;
		if (t0 > 0.0 && t0 < tVis) {
			vec3 Q = ro + rd * t0;
			float rho = length(Q.xz);
			if (rho > 0.2 && rho < R) {
				float ab = atan(ps.y, ps.x);
				float az = abs(mod(atan(Q.z, Q.x) - ab + 0.5 * PI, PI) - 0.5 * PI);
				float dn = exp(-pow(az / pr.w, 4.0)) * smoothstep(0.25, 0.7, rho) * (1.0 - smoothstep(0.8 * R, R, rho)) / max(rho, 0.5);
				float path = min(2.0 * rho * tan(PS_W) / abs(rd.y), 6.0) * 0.86;
				acc += PS_BEAM_COL * dn * path * PS_FAN_K * (1.0 + 1.0 * flare) * ((t0 > tRing) ? sqrt(trRing) : 1.0);
			}
		}
	}
	return acc;
}
vec3 shipsDraw(vec3 ro, vec3 rd, float tVis, float tRing, float trRing, float t) {
	vec3 acc = vec3(0.0);
	float flare = TCTL(31, 3);
	for (int i = 0; i < N_SHIPS; i++) {
		vec3 P = TCTL4(34 + i).xyz;
		acc += shipDraw(i, ro, rd, tVis, tRing, trRing, t, SHIP_L, SHIP_TRAIL_S, 0.008, SHIP_COL) * holo(ro, rd, P, t + 0.3 * float(i), flare);
		acc += holoRay(ro, rd, P, tVis, t + 0.7 * float(i)) * SHIP_COL * 0.7;
	}
	acc += pulsarBeams(ro, rd, tVis, tRing, trRing);
	return acc;
}

void main() {
	float t = time;
	float yaw = TCTL(2, 0);
	float churn = TCTL(1, 1);

	vec3 ro = TCTL4(26).xyz, fw = TCTL4(17).xyz, rt = TCTL4(18).xyz, up = TCTL4(19).xyz;
	vec2 uv = (gl_FragCoord.xy - 0.5 * resolution) / min(resolution.x, resolution.y) * 2.0 * FOV;
	vec3 rd = normalize(fw + uv.x * rt + uv.y * up);

#if SKY_ON
	vec3 col = sky(rd);
#else
	vec3 col = vec3(0.0);
#endif
	float T = 1.0;
	float b = dot(ro, rd), c = dot(ro, ro) - DW_R * DW_R, disc = b * b - c;
	float bImp = length(cross(ro, rd));
	float tPl = 1e9;
	if (disc > 0.0 && -b - sqrt(disc) > 0.0) {
		tPl = -b - sqrt(disc);
		T = 0.0;
		vec3 P = ro + rd * tPl;
		vec3 n = normalize(P);
		float om = 1.0 - max(dot(n, -rd), 0.0);
		float I = max(1.0 - LIMB_U1 * om - LIMB_U2 * om * om, 0.03);
		col = plainDwarf(n, t) * I;
		col = mix(col, LIMB_PURP * length(col) * 0.9, smoothstep(0.35, 1.0, om) * 0.7);
		{
			vec4 sp = TCTL4(5);
			float ag = (t - sp.x) / SPIT_FLASH;
			if (ag >= 0.0 && ag < 1.0) {
				float side = 0.5 + 0.5 * dot(normalize(vec2(n.x, n.z) + 1e-5), vec2(cos(sp.y), sin(sp.y)));
				col += vec3(1.0, 0.85, 1.0) * (1.0 - ag) * (1.0 - ag) * (1.5 + 4.0 * side) * LIGHT_K * 0.3;
			}
		}
	}
	float hz = bImp - DW_R;
#if HALO_ON
	if (hz > -0.02) col += vec3(0.42, 0.03, 0.26) * 0.35 * exp(-max(hz, 0.0) / HAZE_H) * smoothstep(-0.02, 0.0, hz) * step(0.0, -b);
	if (hz > 0.0 && tPl > 1e8) col += UV_HALO_COL * UV_HALO_G * exp(-hz / UV_HALO_H);
#endif

	float tMo = 1e9; vec3 moCol = vec3(0.0); int moHit = -1;
	for (int i = 0; i < N_MOONS; i++) {
		vec4 m = TMOON4(i, 0);
		if (m.w <= 0.0) continue;
		vec3 oc = ro - m.xyz;
		float bb = dot(oc, rd), cc = dot(oc, oc) - m.w * m.w, dd = bb * bb - cc;
		if (dd <= 0.0) continue;
		float tm = -bb - sqrt(dd);
		if (tm <= 0.0 || tm >= tMo) continue;
		tMo = tm; moHit = i;
	}
	if (moHit >= 0) {
		vec4 m = TMOON4(moHit, 0);
		vec3 P = ro + rd * tMo, n = normalize(P - m.xyz);
#if MOON_CHROME
		float cosI = max(dot(n, -rd), 0.0);
		float F = METAL_R0 + (1.0 - METAL_R0) * pow(1.0 - cosI, 5.0);
		moCol = METAL_TINT * F * mirrorEnv(P + n * 1e-3, reflect(rd, n), t, churn) + vec3(0.002);
#else
		float rr = length(P);
		moCol = METAL_TINT * LIGHT_COL * LIGHT_K * ROCK_ALB * max(dot(n, -P / rr), 0.0) / (rr * rr) + vec3(0.002)
		      + MOON_FILL * METAL_TINT * (0.35 + 0.65 * max(dot(n, -rd), 0.0));
#endif
	}

#if ROCK_ON
	vec3 rockPos = TCTL4(10).xyz;
	mat3 rockRot = mat3(TCTL4(12).xyz,
	                    TCTL4(13).xyz,
	                    TCTL4(14).xyz);
	vec3 lo = transpose(rockRot) * (ro - rockPos), ld = transpose(rockRot) * rd;
	float B = dot(lo, ld), C = dot(lo, lo) - RK_BOUND * RK_BOUND, dd = B * B - C;
	if (dd > 0.0) {
		float tk = max(-B - sqrt(dd), 0.0), tEnd = min(-B + sqrt(dd), tMo);
		bool hit = false;
		for (int k = 0; k < 40 && tk < tEnd; k++) {
			float d = racketSDF(lo + ld * tk + ROCK_COM);
			if (d < 0.0008) { hit = true; break; }
			tk += d * 0.65;
		}
		if (hit && tk < tMo) {
			vec3 pb = lo + ld * tk + ROCK_COM;
			const vec2 e = vec2(0.0008, -0.0008);
			vec3 nb = normalize(e.xyy * racketSDF(pb + e.xyy) + e.yyx * racketSDF(pb + e.yyx)
			                  + e.yxy * racketSDF(pb + e.yxy) + e.xxx * racketSDF(pb + e.xxx));
			vec3 P = ro + rd * tk, n = rockRot * nb;
			float rr = length(P);
			tMo = tk;
			float alb = ROCK_ALB * (0.75 + 0.25 * sin(19.0 * pb.x + 2.0) * sin(23.0 * pb.y) * sin(29.0 * pb.z + 1.0));
			moCol = LIGHT_COL * LIGHT_K * alb * max(dot(n, -P / rr), 0.0) / (rr * rr) + vec3(0.002);
		}
	}
#endif
	if (tMo < tPl) { col = moCol; T = 0.0; }

	float tRingS = 1e9, trRingS = 1.0;
	if (abs(rd.y) > 1e-5) {
		float tR = -ro.y / rd.y;
		vec3 P = ro + rd * tR;
		float r = length(P.xz);
		if (tR > 0.0 && r > R_IN && r < R_OUT && tR < tPl) {
			float sg = ringSigmaAt(r, atan(P.z, P.x));
			float tau = RING_TAU_K * (TAU_K * sg + TAU_K2 * sg * sg);
			float trans = exp(-tau / abs(rd.y));
			vec3 L = normalize(-P), V = -rd;
			float cLV = clamp(dot(L, V), -1.0, 1.0);
			float ph = mix(lambertPhase(acos(cLV)), 0.25 * hgPhase(-cLV, DUST_G), DUST_FWD);
			float beta = asin(clamp(DW_R / r, 0.0, 1.0));
			float top = (2.0 * beta * beta * beta / 3.0) / PI;
			float phMS = mix(ph, 0.5, MULTI_SC);
			vec3 lit = RING_BRIGHT * RING_LIGHT * RING_WHITE * RING_FLUOR * LIGHT_K * RING_ALB * (phMS / (r * r) + top) * (1.0 - trans);
			if (tR < tMo) { col = col * trans + lit; T *= trans; }
			else col = moCol;
			if (tR < tMo) { tRingS = tR; trRingS = trans; }
		}
	}

#if SHIPS_ON
	col += shipsDraw(ro, rd, min(tPl, tMo), tRingS, trRingS, t);
#endif
	if (TCTL(30, 0) > 0.5) col += meteorsPhone(ro, rd, min(tPl, tMo), t);
	fragColor = vec4(col, T);
}

==image==
#define WALL_STYLE 4
#define SMK_COL      vec3(0.012, 0.016, 0.040)
#define SMK_EDGE_COL vec3(0.30, 0.55, 1.00)
#define SMK_EDGE_K   0.35

uniform vec2 resolution;
uniform float battery;
uniform sampler2D scene;
uniform highp sampler2D ctl;
out vec4 fragColor;
#define TCTL(s, c) texelFetch(ctl, ivec2((s) * 4 + (c), 0), 0).r
#define TCTL4(s) vec4(TCTL(s, 0), TCTL(s, 1), TCTL(s, 2), TCTL(s, 3))
#define DIO_EXPOSURE 2.6
#define PLANET_MEAN  vec3(0.764, 0.088, 0.993)
#define SECRET_WHITE 0.40
#define LIGHT_COL    mix(PLANET_MEAN, vec3(dot(PLANET_MEAN, vec3(0.2126, 0.7152, 0.0722))), SECRET_WHITE)
#define LIGHT_K      5.5
#define GLS_BASE     0.010
#define GLS_BACK     0.12
#define GLS_BACK_COL vec3(0.55, 0.45, 1.0)
#define GLS_DWARF    1.4
#if WALL_STYLE == 0
uniform sampler2D dio_backC;
uniform sampler2D dio_wlC;
uniform sampler2D dio_wrC;
uniform sampler2D dio_topC;
uniform sampler2D dio_botC;
#endif
#define PANE_SCALE   2.6
#define PANE_GAIN    0.16
#define LEAD_W       0.05
#define GRID_STEP    0.5
#define TILE_NX      4.0
#define TILE_NY      9.0
#define TILE_NZ      7.0
#define TILE_EMIT    0.30
#define TILE_LIT     0.08
vec2 hash22(vec2 p) { return hash33(vec3(p, 7.3)).xy; }
vec3 iqPalette(float t, vec3 teal, vec3 red, vec3 dark) {
	if (t < 0.25) return mix(dark, teal, t / 0.25);
	if (t < 0.50) return mix(teal, dark, (t - 0.25) / 0.25);
	if (t < 0.75) return mix(dark, red,  (t - 0.50) / 0.25);
	return mix(red, dark, (t - 0.75) / 0.25);
}
vec3 panes(vec2 p, out float border) {
	vec2 i = floor(p), f = fract(p);
	float d1 = 8.0, d2 = 8.0; vec2 c1 = vec2(0.0);
	for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
		vec2 g = vec2(float(x), float(y));
		vec2 r = g + hash22(i + g) - f;
		float d = dot(r, r);
		if (d < d1) { d2 = d1; d1 = d; c1 = i + g; } else if (d < d2) d2 = d;
	}
	border = sqrt(d2) - sqrt(d1);
	return vec3(c1, sqrt(d1));
}
vec3 paneColour(vec2 id) {
	vec3 h = hash33(vec3(id, 3.1));
	vec3 c = h.x < 0.50 ? vec3(0.05, 0.10, 0.90) : (h.x < 0.78 ? vec3(0.45, 0.08, 0.80) : (h.x < 0.94 ? vec3(0.95, 0.18, 0.08) : vec3(0.85, 0.55, 0.95)));
	return c * PANE_GAIN * (0.45 + 1.1 * h.y * h.y);
}
vec3 wallPattern(vec2 pw, float px) {
#if WALL_STYLE == 4
	return SMK_COL;
#elif WALL_STYLE == 2
	vec2 g = abs(fract(pw / GRID_STEP) - 0.5) * GRID_STEP;
	vec2 G = abs(fract(pw / (4.0 * GRID_STEP)) - 0.5) * 4.0 * GRID_STEP;
	float minor = 1.0 - smoothstep(0.0, 1.5 * px, min(g.x, g.y) - 0.5 * px);
	float major = 1.0 - smoothstep(0.0, 2.0 * px, min(G.x, G.y) - 1.0 * px);
	return vec3(0.010, 0.020, 0.045) + vec3(0.15, 0.60, 1.00) * (0.30 * minor + 1.20 * major);
#elif WALL_STYLE == 3
	vec2 p = pw;
	float id = 0.5 + 0.5 * cos(time * 0.6 + sin(dot(floor(p + 0.5) + px * 53.0, vec2(113.1, 17.81))) * 43758.545);
	float b = clamp(battery, 0.0, 1.0);
	vec3 teal = vec3(0.0, 0.65, 0.35) * b, red = vec3(1.0, 0.0, 0.25) * (1.0 - b), dark = vec3(0.0, 0.0, 0.2);
	vec3 co = iqPalette(id, teal, red, dark);
	vec2 pa = smoothstep(0.0, 0.2, id * (0.5 + 0.5 * cos(6.2831 * p)));
	return co * pa.x * pa.y;
#else
	vec2 p = pw * PANE_SCALE;
	float border; vec3 cell = panes(p, border);
	float aa = px * PANE_SCALE * 1.5 + 1e-4;
	float glass = smoothstep(LEAD_W - aa, LEAD_W + aa, border);
	vec3 c = paneColour(cell.xy) * (1.15 - 0.45 * cell.z);
	float h = hash33(vec3(pw * 37.0, 1.7)).x;
	c *= 0.92 + 0.16 * h;
	return mix(vec3(0.004), c, glass);
#endif
}
vec4 boxGlass(vec3 ro, vec3 rd, vec3 bc, vec3 dw) {
	vec3 inv = 1.0 / (rd + vec3(1e-7));
	vec3 t0 = (bc - BOX_HALF - ro) * inv, t1 = (bc + BOX_HALF - ro) * inv;
	vec3 tmn = min(t0, t1), tmx = max(t0, t1);
	float tN = max(max(tmn.x, tmn.y), tmn.z), tF = min(min(tmx.x, tmx.y), tmx.z);
	if (tF < max(tN, 0.0)) return vec4(0.0, 0.0, 0.0, -1.0);
	vec3 P = ro + rd * tF, q = (P - bc) / BOX_HALF * 0.5 + 0.5;
	vec3 light = GLS_BACK * GLS_BACK_COL + GLS_DWARF * LIGHT_COL * LIGHT_K / max(dot(P - dw, P - dw), 1.0);
	float px = tF * 2.0 * FOV / min(resolution.x, resolution.y);
	vec3 tq = q * vec3(TILE_NX, TILE_NY, TILE_NZ) - 0.5;
	vec2 pw; vec3 c; float face = 0.0;
	vec3 ed = min(q, 1.0 - q) * 2.0 * BOX_HALF;
	float dEdge = 0.0;
	if (tF == tmx.z) {
		if (P.z - bc.z > 0.0) return vec4(0.0, 0.0, 0.0, -1.0);
		pw = P.xy; dEdge = min(ed.x, ed.y);
#if WALL_STYLE == 3
		pw = tq.xy;
#elif WALL_STYLE == 0
		c = pow(texture(dio_backC, vec2(q.y, 1.0 - q.x)).rgb, vec3(2.2));
#endif
	} else if (tF == tmx.x) {
		pw = vec2(P.z * sign(P.x - bc.x) + 11.0, P.y); dEdge = min(ed.y, ed.z);
		face = P.x - bc.x < 0.0 ? 1.0 : 2.0;
#if WALL_STYLE == 3
		pw = tq.zy;
#elif WALL_STYLE == 0
		c = P.x - bc.x < 0.0 ? pow(texture(dio_wlC, vec2(q.y, 1.0 - q.z)).rgb, vec3(2.2)) : pow(texture(dio_wrC, vec2(q.y, q.z)).rgb, vec3(2.2));
#endif
		light *= 0.85;
	} else {
		pw = vec2(P.x, P.z * sign(P.y - bc.y) + 23.0); dEdge = min(ed.x, ed.z);
		face = P.y - bc.y > 0.0 ? 3.0 : 4.0;
#if WALL_STYLE == 3
		pw = tq.xz;
#elif WALL_STYLE == 0
		c = P.y - bc.y > 0.0 ? pow(texture(dio_topC, vec2(q.x, 1.0 - q.z)).rgb, vec3(2.2)) : pow(texture(dio_botC, vec2(q.x, q.z)).rgb, vec3(2.2));
#endif
		light *= 0.7;
	}
#if WALL_STYLE == 3
	return vec4(wallPattern(pw, face) * (TILE_EMIT + TILE_LIT * light), tF);
#elif WALL_STYLE == 4
	c = wallPattern(pw, px);
	float edge = 1.0 - smoothstep(0.0, 2.5 * px, dEdge - 0.8 * px);
	return vec4(c * (GLS_BASE + light) + SMK_EDGE_COL * SMK_EDGE_K * edge, tF);
#elif WALL_STYLE != 0
	c = wallPattern(pw, px);
#endif
	return vec4(c * (GLS_BASE + light), tF);
}
void main() {
	vec3 ro = TCTL4(26).xyz, fw = TCTL4(17).xyz, rt = TCTL4(18).xyz, up = TCTL4(19).xyz;
	vec2 uv = (gl_FragCoord.xy - 0.5 * resolution) / min(resolution.x, resolution.y) * 2.0 * FOV;
	vec3 rd = normalize(fw + uv.x * rt + uv.y * up);
	vec4 s = texture(scene, gl_FragCoord.xy / resolution);
	vec3 Dd = TCTL4(24).xyz;
	vec4 gb = boxGlass(d2b(ro) + Dd, d2b(rd), vec3(0.0), Dd);
	vec3 col = s.rgb + (gb.w > 0.0 ? gb.rgb : vec3(0.0)) * s.a;
	col = 1.0 - exp(-col * DIO_EXPOSURE);
	fragColor = vec4(col, 1.0);
}
