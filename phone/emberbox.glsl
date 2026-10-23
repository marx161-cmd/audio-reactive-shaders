// Emberbox (ShaderEditor fork, Pixel 10 Pro, 120 fps): the ember flocks in a
// glass tank with stained-glass walls. Gestures: tap / swipe / hold, gravity,
// back tap. Textures: phone/backdrops/dio_*C.png as BACKDROP_ samplers.

#version 300 es

#ifdef DESKTOP
#define TUNE_N 18
uniform float u_tune[TUNE_N];
#endif
#define IMAGE_boid  rgba32f 1024x3 swap
#define IMAGE_ctl   r32f    208x1
#define IMAGE_flk   r32f    64x1
#define IMAGE_ldr   r32f    768x72
#define IMAGE_spk   r32f    8192x8
#define IMAGE_glowI r32ui   640x1428
#define IMAGE_glowH r32ui   640x1428
#define IMAGE_rad   rgba16f 640x1428
#define IMAGE_blmA  rgba16f 320x714
#define IMAGE_blmT  rgba16f 320x714
#define IMAGE_blmB  rgba16f 160x357
#define IMAGE_blmU  rgba16f 160x357
#define BACKDROP_dio_backC dio_backC.png
#define BACKDROP_dio_wlC   dio_wlC.png
#define BACKDROP_dio_wrC   dio_wrC.png
#define BACKDROP_dio_topC  dio_topC.png
#define BACKDROP_dio_botC  dio_botC.png

precision highp float;

#ifndef DESKTOP
#define N_PER    450
#else
#define N_PER    2304
#endif
#ifndef DESKTOP
#define N_ALL    900
#else
#define N_ALL    4608
#endif
#define FIX      1024.0
#define N_LEAD   48
#define N_LMAX   192
#define SPK_N    8192
#define SPK_TEXW 2048
#define PERSIST_TAU_I 0.085
#define PERSIST_TAU_H 0.028
#define EMIT_REF_FPS  60.0
#define BL_CLAMP      5.0
#ifndef DESKTOP
#define CAM_D    3.0
#else
#define CAM_D    3.0
#endif
#ifndef DESKTOP
#define FOCAL    1.3
#else
#define FOCAL    1.9
#endif
#ifndef DESKTOP
#define BOX_ZN  -1.00
#else
#define BOX_ZN  -0.25
#endif
#ifndef DESKTOP
#define BOX_ZB   1.00
#else
#define BOX_ZB   2.50
#endif
#define EYE_Y    0.00
float boxXW(float edge) { return edge * (BOX_ZN + CAM_D) / FOCAL; }
float boxYW()           { return (BOX_ZN + CAM_D) / FOCAL; }
vec3 camPos() { return vec3(0.0, EYE_Y, -CAM_D); }
vec3 project(vec3 p) { float s = FOCAL / max(p.z + CAM_D, 0.3); return vec3((p.xy - vec2(0.0, EYE_Y)) * s + vec2(0.0, EYE_Y) * FOCAL / (CAM_D + BOX_ZN), s); }
#define LIFE0         14.0
#define LIFE1         38.0
#define LIFE_STALL    1.00
#define LIFE_STALL_K  1.00
#define LIFE_STALL_HZ 0.45
#define LIFE_SAG_P    2.00
#define LIFE_DEAD_H   0.85
#define LIFE_HEAT_CURVE 0.60
#define LIFE_DEAD_I   1.00
#define LIFE_BRIGHT_CURVE 4.00
#define LIFE_BIRTH_I  1.10
#define LIFE_BIRTH_H  0.22
#define LIFE_BIRTH_T  0.30
#define LIFE_BIRTH_R  0.05
#define LIFE_BIRTH_V  0.30
#define BASE_BPM     90.0
#define SLOW_BPM     85.0
#define N_MODES      5.0
#define ORBIT_BEATS  16.0
#define L_BIN_T      0.1
#define L_SEP0       0.035
#define L_SEP1       0.50
#define L_BOB        0.07
#define L_KICK_SEP   0.22
#define SPONGE_PUSH  0.30
#define WINDUP_K     0.6
#define Y_STRETCH_K  0.75
#define Y_STRETCH_MAX 2.6
#define SQUISH_X     0.60
#define STRETCH_BEATS 8.0
#define SEP_NEAR     0.22
#define SEP_FAR_Q    0.95
#define SEP_FAR_L    0.70
#define LASH_K       0.80
#define RADIAL_SKEW  2.0
#define HEAD_MAX_R   1.15
#define SPLIT_EDGE   0.62
#define SPLIT_INNER  0.18
#define SPLIT_SWEEP  0.55
#define SPLIT_WARP   0.35
#define SPLIT2_DZ    0.70
#define SPLIT2_EASE  1.5
#define TILT_P_BANK  0.35
#define TILT_P_DIVE  0.30
#define TILT_BANK_A  0.80
#define TILT_DIVE_A  1.40
#define TILT_IN_BEATS 0.50
#define TILT_OUT_BEATS 0.15
#define TILT_E0      0.35
#define TILT_E1      0.80
#define TILT_POCKET_BEATS 2.0
#define FORM_R          0.30
#define FORM_CYC_BEATS  8.0
#define FORM_MORPH_BEATS 2.0
#ifdef DESKTOP
#define LEAD_PERIOD (u_tune[0] > 0.0 ? u_tune[0] : 0.35)
#else
#define LEAD_PERIOD     0.35
#endif
#ifdef DESKTOP
#define LEAD_DAMP (u_tune[1] > 0.0 ? u_tune[1] : 0.75)
#else
#define LEAD_DAMP       0.75
#endif
#ifdef DESKTOP
#define LEAD_VMAX (u_tune[3] > 0.0 ? u_tune[3] : 2.5)
#else
#define LEAD_VMAX       2.5
#endif
#define LEAD_VMAX_HI    5.0
#define LEAD_CATCH_V    1.25
#define LEAD_CATCH_K    6.0
#define LEAD_FOLLOW     1.0
#define PRED_BACK       6.0
#define PRED_FWD        6.0
#ifndef DESKTOP
#define PRED_N          49
#else
#define PRED_N          97
#endif
#define TURN_FIT        0.5
#define CZ_MIN          0.30
#ifdef DESKTOP
#define LEAD_FF (u_tune[2] > 0.0 ? u_tune[2] : 1.0)
#else
#define LEAD_FF         1.0
#endif
#define BAL_HOLD_BEATS  1.0
#define BAL_COOL_BEATS  8.0
#ifdef DESKTOP
#define LEAD_SOFT (u_tune[4] > 0.0 ? u_tune[4] : 0.18)
#else
#define LEAD_SOFT       0.18
#endif
#ifdef DESKTOP
#define LEAD_SPREAD (u_tune[5] > 0.0 ? u_tune[5] : 0.10)
#else
#define LEAD_SPREAD     0.10
#endif
#define SPLIT_STEP_BEATS 0.25
#define SH_RING      0
#define SH_ARROW     1
#define SH_V         2
#define SH_BALL      3
#define SH_CORK      5
#define SH_RIBBON    6
#define SH_STRIPE    9
#define SH_FUNNEL    10
#define SH_WASP      11
#define SH_DISK      12
#define SH_TORUS     14
#define SH_GALAXY    15
#define SH_JELLY     16
#define SH_COMET     17
#define SH_CROWN     18
#define SH_ATOM      19
#define SH_MANTA     20
#define SH_TREFOIL   21
#define SH_MOBIUS    22
#define SH_SLINKY    24
#define SH_BRAID     30
#define SH_BRAID2    31
#define SH_COLUMN    40
#define FB_BASE      50
#define SH_FB_WING   50
#define SH_FB_TEAR   51
#define SH_FB_VORTEX 52
#define SH_FB_PLAIT  53
#define SH_FB_MANTA  54
#define SH_FB_BELL   55
#define SH_FB_CAGE   56
#define SH_FB_KNOT   57
#define SH_FB_CADU   58
#define SH_FB_KLEIN  59
#define SH_FB_VIC    61
#define SH_FB_ECH    62
#define SH_FB_ASTERN 63
#define SH_FB_DOME   64
#define SH_FB_ABR    65
bool isFB(int sh) { return sh >= FB_BASE && sh <= FB_BASE + 15 && sh != FB_BASE + 10; }
#ifndef DESKTOP
#define BODY_SQX     0.70
#define BODY_SIZE    0.85
#else
#define BODY_SQX     0.85
#define BODY_SIZE    0.95
#endif
#define PENT_SPHERE  1
#define SPHERE_TURN_BEATS 8.0
#define SPHERE_MERID 6
#define SPHERE_FLOW_BEATS 4.0
#define FB_SCALE     1.0
#define FF_N       9
#define FF_SPREAD  0.115
#define FF_R       0.032
#define FS_LEN       0.50
#define FS_RAD       0.22
#define FS_FLAT      0.25
#define FS_BRAIDSETS 1
#define BRL_RAD      0.035
#define FS_BR_TURNS  1.6
#define FS_BR_SPIN   0.35
#define FS_SETS      2
#define FS_SET_GAP   0.22
#define FS_BR_RAD    0.08
#define FS_FILL_DIV  4
#define FS_CORE_K    0.15
#define FS_INNER_K   0.55
#define FS_FOLD      0.35
#define FS_HELIX_TURNS 2.50
#define FS_BRAID_TURNS 1.40
#define FS_SQ_MIN    0.70
#define BANK_K       3.5
#define FIG_TRAIL_LEN 1.6
#define TURN_CORNER_A    0.60
#define TURN_LEAD_BEATS  0.6
#define TURN_MORPH_BEATS 1.6
#ifdef DESKTOP
#define BODY_HOLD_MIN (u_tune[16] > 0.0 ? u_tune[16] : 12.0)
#else
#define BODY_HOLD_MIN   12.0
#endif
#define BODY_HOLD_MAX   24.0
#define BODY_HASH        0.50
#define REORDER_LEAD   0.50
#define REORDER_BEATS  1.20
#define REORDER_STEPS  1.00
#define BRAID_BEATS  2.0
#define HELD_BRAID_ON 0.50
#define HELD_RUSH     4.0
#define BRAID_P_MAX  0.35
#define BRAID_EASE   0.30
#define BRAID_BPM0   90.0
#define BRAID_BPM1   170.0
#define BRAID_E0     0.30
#define BRAID_E1     0.85
#define BRAID_D0     0.20
#define BRAID_D1     0.65
#define BC_CATCH     2.0
#define DASH_BEATS   0.25
#define DASH_SPD     1.5
#define HUNT_E0      0.40
#define HUNT_E1      0.75
#define HUNT_E_BEATS 2.0
#define ORBIT_SPEED_H 2.0
#define HIST_N       32
#define FOL_GIRTH       0.014
#define FOL_ROPE_LEN    0.70
#define FOL_WANDER      0.030
#define FOL_STRAG_SHARE 0.20
#define FOL_STRAG_MUL   3.0
#define FOL_AGE_K       4.0
#define WV_SLIDE     1.0
#define WV_S_JIT     0.10
#define WV_T0        0.25
#define WV_T1        0.85
#define WV_GRIP_MIN  0.15
#define WV_SNAP      0.55
#define WV_PER_HOT   0.12
#define WV_PER_COLD  1.40
#define WV_DAMP      0.70
#define WV_ACC       24.0
#define WV_HOP_T     0.40
#define WV_HOPS      2.5
#define ROPE_VMAX    6.0
#define ROPE_TOUCH   0.80
#define FIG_TRAIL_ON  1
#define HH_N         96
#define TIP_LUNGE       1.2
#define TIP_LUNGE_BEATS 0.25
#define TIP_LUNGE_SKIP  0.45
#define TIP_LUNGE_DECAY 8.0
#define TIP_WANDER      0.60
#define TIP_WANDER_HZ   0.35
#define TIP_SWAY_K      1.0
#define TIP_VMAX        3.0
#define MID_R_FRAC   0.125
#define FAR_TAP_BEATS 4.0
#define FAR_TAP_EASE  0.35
#define SWIPE_DASH_V 1.0
#define SWIPE_DASH_GAP 0.35
#define PENT_TURN_BEATS 16.0
#define STRIPE_SPIN  0.125
#define HELIX_R      0.16
#define HELIX_TURNS  1.5
#define HELIX_SPIN   0.0625
#define FUNNEL_R0    0.06
#define FUNNEL_R1    0.60
#define FUNNEL_LEN   2.0
#define FUNNEL_TURNS 3.0
#define FUNNEL_SPIN  0.25
#define DISK_R       0.40
#define DISK_SPIN    0.125
#define WASP_PER     12
#define WASP_R       0.05
#define WASP_ORB0    0.35
#define WASP_ORB1    0.65
#define WASP_TURN    4.0
#define WASP_SURGE   1.5
#define WASP_STRIKE  0.85
#define WASP_MORPH_BEATS 0.5
#define POSS_BEATS   8.0
#define POSS_TEAR    0.2
#define TOR_RM       0.52
#define TOR_RT       0.17
#define TOR_SPIN     0.125
#define GAL_R0       0.10
#define GAL_R1       0.86
#define GAL_TWIST    2.2
#define GAL_TILT     0.62
#define GAL_SPIN     0.0625
#define JEL_BELL     0.42
#define JEL_TENT     0.60
#define JEL_BEATS    2.0
#define COMET_R      0.30
#define COMET_LEN    2.4
#define CROWN_R      0.42
#define CROWN_SPIKE  0.26
#define CROWN_N      8.0
#define CROWN_SPIN   0.125
#define ATOM_A       0.52
#define ATOM_B       0.24
#define ATOM_TILT    1.0471976
#define ATOM_SPIN    0.0625
#define MAN_X        0.70
#define MAN_Z        0.90
#define MAN_W        0.16
#define MAN_BEATS    2.0
#define TREF_S       0.30
#define TREF_SPIN    0.0625
#define MOB_R        0.50
#define MOB_W        0.16
#define MOB_SPIN     0.0625
#define SLK_R        0.40
#define SLK_GAP      0.16
#define SLK_AMP      0.55
#define SLK_BEATS    1.0
#define COL_Y0      (-0.55)
#define COL_H        1.35
#define COL_R0       0.045
#define COL_R1       0.34
#define COL_TWIST    6.2831853
#define BRAID_RAD    0.20
#define BRAID_SPAN   0.70
#define BRAID_TURNS  1.75
#define BRAID_RIPPLE 0.30
#define BRAID_SPIN   0.125
#define AGG_WASP     0
#define AGG_ERUPT    1
#define AGG_FIGS     2
#define FIG_MORPH_BEATS 0.25
#define FIG_SPLV     1.0
#define FG_HELIX     1
#define FG_RINGV     2
#define FG_HELIXF    3
#define FG_LASH      4
#define FG_PUNCHIN   5
#define FG_SHEAR     7
#define FG_LISSA     11
#define FG_STAR      12
#define FG_TREFOIL   13
#define FG_BUTTERFLY 14
#define FG_CLOVER    15
#define FG_WALLPUNCH 20
#define FG_EYERINGS  21
#define FG_EYEARM    22
#define FIG_BEATS      8.0
#define FIG_HELIX_BEATS 5.0
#define FIG_COOL_BEATS 12.0
#ifndef DESKTOP
#define FIG_FIT_N      24
#else
#define FIG_FIT_N      64
#endif
#ifdef DESKTOP
#define PICK_HASH (u_tune[15] > 0.0 ? u_tune[15] : 0.70)
#else
#define PICK_HASH      0.70
#endif
#define PICK_PEN1      1.50
#define PICK_PEN2      1.00
#define PICK_PEN3      0.60
#define VAR_SC0        0.80
#define VAR_SC1        1.20
#define VAR_OFF_XY     0.22
#define VAR_OFF_Z0    -0.15
#define VAR_OFF_Z1     0.30
#define VAR_REV_P      0.50
#ifdef DESKTOP
#define FAR_GAP0 (u_tune[17] > 0.0 ? u_tune[17] : 10.0)
#else
#define FAR_GAP0      10.0
#endif
#define FAR_GAPJ      12.0
#define FAR_BEATS      6.0
#define FAR_EASE       1.0
#define CLIMB_EVERY    4.0
#define CLIMB_P        0.25
#define CLIMB_BOTH_P   0.67
#define CLIMB_COOL    12.0
#define CLIMB_BEATS    8.0
#define CLIMB_H0       0.45
#define CLIMB_HJ       0.40
#define FIG_EASE       0.8
#define FIG_R          0.85
#define FIG_XROOM      0.94
#define GOAL_MARGIN    0.08
#define FIG_SPLIT      0.03125
#define HELIX_H        0.62
#define HELIX_W        0.16
#define HELIX_CLIMB    0.80
#define RING_R         0.65
#define PUNCH_D        1.5
#define PUNCH_SHOT     0.28
#define FIG_TURNS      4.0
#define WP_OUT_BEATS   1.0
#define WP_BACK_BEATS  1.0
#define WP_OVER        1.00
#define WP_COOL_BEATS  16.0
#define EYE_BEATS      8.0
#define EYE_COOL_BEATS 48.0
#define EYE_EASE       0.8
#define EYE_R          0.50
#define EYE_RING_BEATS 4.0
#define EYE_TURN_BEATS 32.0
#define BORED_LAPS     3.0
#define SPECIAL_E      0.80
#define SPECIAL_W      0.50
#define SPECIAL_I      0.30
#define WP_PHRASE      12.0
#define SWIPE_PX     40.0
#define TAP_MAX_S    0.45
#define ODIR_TAU     1.5
#define GRAV_K       0.38
#define GRAV_TAU     2.5
#define CEN_PERIOD   3.0
#define CEN_DAMP     0.85
#define CEN_VMAX     0.45
#define CEN_MARGIN   0.45
#define SENSOR_UV    vec2(0.0, -0.465)
#define BT_THR       4.0
#define BT_XY        2.5
#define BT_GYRO      1.2
#define BT_TAU       0.5
#define BT_REFR      0.45
#define HOLD_S       0.30
#define HOLD_PERIOD  1.2
#define HOLD_VMAX    1.2
#define FLING_SMOOTH 0.35
#define FLING_K      1.0
#define FLING_MAX    3.0
#define FLING_MIN    0.25
#define FLING_TAU    0.40
#define FLING_WP     1.5
#define KNOCK_R      0.55
#define KNOCK_K      14.0
#define KNOCK_TAU    0.30
#define TAP3_MAX_S   0.60
#define PINCH_ORBIT_BEATS 2.0
#define PINCH_R_MIN  0.08
#define PINCH_TAU_IN 0.15
#define PINCH_TAU_OUT 0.50
#define Y_USE        0.85
#define Y_MIN        0.80
#define SCATTER_BEATS  1.0
#define SCATTER_K      0.8
#define SCAT_R0        0.35
#define SCAT_R1        0.55
#define BT_TOUCH_S     0.30
#define SCATTER_IMPACT 0.5
#define ACCENT_K       1.2
#define ACCENT_SKIT    4.0
#define ACCENT_VBOOST  0.5
#define ACCENT_BEATS   0.25
#define ACCENT_W0      0.20
#define ACCENT_W1      0.60
#define FRENZY_BPM0    150.0
#define FRENZY_BPM1    170.0
#define FRENZY_E0      0.60
#define FRENZY_E1      0.80
#define FRENZY_D0      0.40
#define FRENZY_D1      0.65
#define FRENZY_BEATS   2.0
#define FRENZY_ON      0.5
#define FRENZY_OFF     0.35
#define FRENZY_VMAX    6.0
#define FRENZY_SEP     0.45
#define FRENZY_SPREAD  0.35

#define T_COLD  1000.0
#define T_SPAN  3000.0
vec3 bbColour(float T) {
	float t = T / 100.0;
	vec3 c;
	c.r = t <= 66.0 ? 1.0 : clamp(1.2929 * pow(t - 60.0, -0.1332), 0.0, 1.0);
	c.g = t <= 66.0 ? clamp((99.4708 * log(t) - 161.1196) / 255.0, 0.0, 1.0) : clamp(1.1299 * pow(t - 60.0, -0.0755), 0.0, 1.0);
	c.b = t >= 66.0 ? 1.0 : (t <= 19.0 ? 0.0 : clamp((138.5177 * log(max(t - 10.0, 1e-5)) - 305.0448) / 255.0, 0.0, 1.0));
	return c;
}
vec3 emberRadiance(uint Ir, uint Hr, float gain) {
	float I = float(Ir) / FIX;
	if (I <= 0.0) return vec3(0.0);
	float heat = clamp(float(Hr) / FIX / I, 0.0, 1.0);
	float T = T_COLD + T_SPAN * pow(heat, 1.15);
	float L = clamp(pow(T / 2800.0, 2.2), 0.05, 4.0);
	return bbColour(T) * L * I * gain;
}

uniform float time;
uniform int frame;
uniform float beatCount;
uniform float beatPhase;
uniform float beatConf;
uniform float beatBpm;
uniform int beatSource;

#ifdef COMPUTE
uniform vec2 resolution;
uniform vec2 touch;
uniform int pointerCount;
uniform vec3 pointers[10];
uniform bool screenLocked;
uniform float micAmplitude;
uniform vec3 gyroscope;
uniform vec3 gravity;
uniform vec3 linear;
uniform float u_ms_source, u_ms_change, u_ms_next_in, u_ms_drop_in, u_ms_repeat;
uniform vec4 u_ms_cur_a, u_ms_cur_b, u_ms_sec_a, u_ms_sec_b, u_ms_next_a, u_ms_next_b;
uniform highp sampler2D bands;
uniform highp sampler2D boid;

uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
float rnd(inout uint s) { s = pcg(s); return float(s) * (1.0 / 4294967296.0); }
vec3 rndDir(inout uint s) {
	float z = 2.0 * rnd(s) - 1.0, a = 6.2831853 * rnd(s);
	float r = sqrt(max(0.0, 1.0 - z * z));
	return vec3(r * cos(a), r * sin(a), z);
}
float loudN(float loud, float lmax) { return clamp(loud / max(lmax, 0.05), 0.0, 1.0); }
float hash11(float x) { return fract(sin(x * 127.1 + 311.7) * 43758.5453); }
float hashW(float a, float b) { return fract(sin(a * 91.7 + b * 47.3 + 13.1) * 43758.5453); }
vec3 rotAx(vec3 v, vec3 a, float ang) { float c = cos(ang), s = sin(ang); return v * c + cross(a, v) * s + a * dot(a, v) * (1.0 - c); }
vec2 vrot(vec2 v, float a) { float c = cos(a), s = sin(a); return vec2(c * v.x - s * v.y, s * v.x + c * v.y); }

vec3 glassPt(vec2 px) { vec2 uvc = (2.0 * px - resolution) / max(resolution.y, 1.0); return vec3(uvc * boxYW(), BOX_ZN); }
#define C_LOUD    0
#define C_LMAX    1
#define C_GROOVE  2
#define C_MIDS    3
#define C_BASS    4
#define C_HI      5
#define C_KPREV   6
#define C_PATH    7
#define C_KICK    8
#define C_TPREV   9
#define C_DT      10
#define C_ODFAVG  11
#define C_LOW0    12
#define C_MICAVG  36
#define C_MICFLR  37
#define C_MICK    38
#define C_MICPRV  39
#define C_INIT    40
#define C_MSTATE  41
#define C_MSEEN   42
#define C_MTGT    43
#define C_POCKET  44
#define C_PRESS   45
#define C_PUSH    46
#define C_SPLITW  47
#define C_HAT     48
#define C_HW0     49
#define C_HPOCK   54
#define C_HSEP    55
#define C_YROOM   58
#define C_CX      59
#define C_CY      60
#define C_CVX     61
#define C_CVY     62
#define C_SCAT    63
#define C_SCSEEN  64
#define C_SCDROP  65
#define C_ACCENT  66
#define C_SHA     67
#define C_SHB     68
#define C_SHT     69
#define C_SHCYC   70
#define C_SHSEEN  71
#define C_MORPHB  72
#define C_SPLV    73
#define C_BAL     74
#define C_BALCOOL 75
#define C_FRENZY  76
#define C_POSSON  77
#define C_POSSCYC 78
#define C_POSSW   79
#define C_AGG     80
#define C_TDOWN   81
#define C_TX0     82
#define C_TY0     83
#define C_TXP     84
#define C_TYP     85
#define C_TSWIPE  86
#define C_TT0     87
#define C_TAPQ    88
#define C_ODIR    89
#define C_ODIRT   90
#define C_OPH     91
#define C_FIGPH   92
#define C_FIGID   93
#define C_FIGAMP  94
#define C_FIGCOOL 95
#define C_FIGLAST 96
#define C_FIGSEEN 97
#define C_FIGBAR  98
#define C_FIGN    99
#define C_EYE_T   100
#define C_EYE_END 101
#define C_EYE_W   102
#define C_EYE_LAST 103
#define C_WP_START 104
#define C_WP_LAST 105
#define C_WP_AMT  106
#define C_WP_SIDE 107
#define C_EVT     108
#define C_BORED_N 109
#define C_BORED_LAP 110
#define C_TLT_A   111
#define C_TLT_T   112
#define C_TLT_TH  113
#define C_TLT_END 114
#define C_TLT_LP  115
#define C_BT_BASE 116
#define C_BT_REFR 117
#define C_BT_N    118
#define C_SPL2W   119
#define C_SPL2T   120
#define C_HDIR0   121
#define C_SPLITAX 125
#define C_BCPREV  126
#define C_BC      127
#define C_HOLD    128
#define C_HVX     129
#define C_HVY     130
#define C_FLX     131
#define C_FLY     132
#define C_KX      133
#define C_KY      134
#define C_KZ      135
#define C_KW      136
#define C_WP_SAME 137
#define C_PCPREV  138
#define C_PCMAX   139
#define C_K2X     140
#define C_K2Y     141
#define C_K2Z     142
#define C_K2W     143
#define C_PINMX   144
#define C_PINMY   145
#define C_PINR    147
#define C_PINW    148
#define C_PINPH   149
#define C_PINRW   150
#define C_PINUX   151
#define C_PINUY   152
#define C_PINV    153
#define C_TLAST   154
#define C_FIGH1   155
#define C_FIGH2   156
#define C_FIGMX   157
#define C_FIGMY   158
#define C_FIGREV  159
#define C_FIGSC   160
#define C_FIGOX   161
#define C_FIGLEN  164
#define C_PHRQ    165
#define C_POSSK   166
#define C_FARNEXT 167
#define C_FAREND  168
#define C_FARW    169
#define C_SPLITAXE 170
#define C_CLBS    171
#define C_CLBF    172
#define C_CLBWALL 173
#define C_CLBH    174
#define C_CLBBLK  175
#define C_SALT    177
#define C_SHH1    178
#define C_SHH2    179
#define C_RD      180
#define C_BRAIDT  181
#define C_BRAIDW  182
#define C_DASH    183
#define C_FARREQ  184
#define C_FARFORCE 185
#define C_PENTW   186
#define C_PENTR   187
#define C_PENTRW  188
#define C_PENTUSED 189
#define C_SWDT    190
#define C_OPHH    191
#define C_HUNTW   192
#define C_FARSH   193
#define C_FFW     194
#define C_HEN     195
#define C_HSPD    196
#define C_TUM     198
#define C_TUMR    199
#define C_BRPH    200
#define C_HELD    201
#define C_SLWP    202
#define C_BCH     197
#define C_EYE_LT  146
#define KICK_LO   4
#define KICK_HI   27

#define FLK_LOAD(img, k) vec4(imageLoad(img, ivec2(4 * (k), 0)).r, imageLoad(img, ivec2(4 * (k) + 1, 0)).r, \
		imageLoad(img, ivec2(4 * (k) + 2, 0)).r, imageLoad(img, ivec2(4 * (k) + 3, 0)).r)
#define FLK_STORE(img, k, v) { vec4 v_ = (v); imageStore(img, ivec2(4 * (k), 0), vec4(v_.x)); \
		imageStore(img, ivec2(4 * (k) + 1, 0), vec4(v_.y)); imageStore(img, ivec2(4 * (k) + 2, 0), vec4(v_.z)); \
		imageStore(img, ivec2(4 * (k) + 3, 0), vec4(v_.w)); }
#define LDR_LOAD(img, row, m) vec4(imageLoad(img, ivec2(4 * (m), row)).r, imageLoad(img, ivec2(4 * (m) + 1, row)).r, \
		imageLoad(img, ivec2(4 * (m) + 2, row)).r, imageLoad(img, ivec2(4 * (m) + 3, row)).r)
#define LDR_STORE(img, row, m, v) { vec4 v_ = (v); imageStore(img, ivec2(4 * (m), row), vec4(v_.x)); \
		imageStore(img, ivec2(4 * (m) + 1, row), vec4(v_.y)); imageStore(img, ivec2(4 * (m) + 2, row), vec4(v_.z)); \
		imageStore(img, ivec2(4 * (m) + 3, row), vec4(v_.w)); }

#define HUNTER_W(w) (w[2] + w[3] + w[4])
vec3 pocketOrbit(float ph, float phH, float bc, float bcH, float w[5], float pocket, vec3 sepX, float yRoom, float loud, float splitAx, float farSh, float tumA) {
	float t = bc * L_BIN_T;
	float wsum = max(w[0] + w[1] + w[2] + w[3] + w[4], 1e-4);
	float hw = clamp(HUNTER_W(w) / wsum, 0.0, 1.0);
	float sepS = L_SEP0 + L_SEP1 * pow(sin(t * 0.5), 4.0) + L_KICK_SEP * sepX.x;
	float wave = 0.5 - 0.5 * cos(6.2831853 * bcH / STRETCH_BEATS);
	float sepH = mix(SEP_NEAR, mix(SEP_FAR_Q, SEP_FAR_L, loud), wave) + L_KICK_SEP * sepX.x;
	float sep = mix(sepS, sepH, hw);
	sep += SPONGE_PUSH * sepX.y;
	sep *= 1.0 - WINDUP_K * sepX.z;
	float rose = 1.0 - pocket * (0.5 - 0.5 * cos(mix(ph, phH, hw) * 3.0));
	vec3 p0 = mix(vec3(cos(ph) * sep, sin(ph) * L_BOB, sin(2.0 * ph) * 0.5 * sep),
	              vec3(sin(2.0 * ph) * sep, sin(3.0 * ph + 0.7854) * 0.5 * sep, 0.14 * cos(5.0 * ph) * sep), 0.5 + 0.5 * sin(t * 0.37));
	float rr = sep * (1.0 - 0.6 * (0.5 - 0.5 * cos(3.0 * ph)));
	vec3 p1a = vec3(cos(ph) * rr, sin(3.0 * ph) * 2.0 * L_BOB, sin(ph) * rr);
	float rs = sep * (0.3 + 0.7 * sqrt(abs(sin(2.0 * ph))));
	vec3 p1b = vec3(cos(ph) * rs, sin(4.0 * ph) * 0.3 * sep, sin(ph) * rs);
	vec3 p1 = mix(p1a, p1b, 0.5 + 0.5 * sin(t * 0.29 + 1.0));
	vec3 p2 = vec3(sin(2.0 * phH), sin(3.0 * phH + 0.7854), 0.14 * cos(5.0 * phH)) * sep;
	vec2 st = 0.58 * vec2(cos(2.0 * phH), sin(2.0 * phH)) + 0.38 * vec2(cos(3.0 * phH), -sin(3.0 * phH));
	vec3 p3 = vec3(st, 0.25 * sin(5.0 * phH)) * sep * 1.15;
	float sMid = 0.5 * (SPLIT_EDGE + SPLIT_INNER), sRad = 0.5 * (SPLIT_EDGE - SPLIT_INNER);
	float ph4 = mix(phH, ph, clamp(farSh, 0.0, 1.0));
	float sPh = ph4 + SPLIT_WARP * sin(ph4);
	vec3 p4lr = vec3(sMid + sRad * cos(sPh), SPLIT_SWEEP * sin(2.0 * sPh), L_BOB * sin(3.0 * ph4));
	vec3 p4tb = vec3(SPLIT_SWEEP * sin(2.0 * sPh), sMid + sRad * cos(sPh), L_BOB * sin(3.0 * ph4));
	vec3 p4 = mix(p4lr, p4tb, splitAx);
	vec3 off = (p0 * w[0] + p1 * w[1] + p2 * w[2] + p3 * w[3]) / wsum;
	off.xz *= rose;
	off += p4 * w[4] / wsum;
	float tum = tumA;
	off.xy = mat2(cos(tum), -sin(tum), sin(tum), cos(tum)) * off.xy;
	float tum2 = tum * 0.8;
	off.yz = mat2(cos(tum2), -sin(tum2), sin(tum2), cos(tum2)) * off.yz;
	float ys = clamp(yRoom / Y_STRETCH_K, 1.0, Y_STRETCH_MAX);
	off.y *= mix(ys, 1.6, hw);
	off.x *= mix(SQUISH_X, 0.62, hw);
	float zf = -LASH_K * sqrt(off.z * off.z + 0.0025);
	off.z = mix(off.z, zf, hw);
	float rc = length(off.xy);
	if (rc > 1e-5) {
		float xR = clamp(rc / HEAD_MAX_R, 0.0, 1.0);
		float rN = HEAD_MAX_R * (1.0 - pow(1.0 - xR, RADIAL_SKEW));
		off.xy *= mix(1.0, rN / rc, hw * (1.0 - w[4] / wsum));
	}
	return off;
}
float circDist(float a, float b) { float d = abs(a - b); return min(d, N_MODES - d); }
void pocketWeights(float state, out float w[5]) {
	for (int i = 0; i < 5; i++) w[i] = max(0.0, 1.0 - circDist(state, float(i)));
}

vec3 figSlot(int sh, float u, float f) {
	float ud = (u - FIG_SPLIT) / (1.0 - FIG_SPLIT);
	if (sh == FG_HELIX) {
		if (u < FIG_SPLIT) return vec3(HELIX_W * (f < 0.5 ? -1.0 : 1.0), mix(HELIX_H, -HELIX_H, u / FIG_SPLIT), 0.0);
		if (u < HELIX_CLIMB) {
			float uc = (u - FIG_SPLIT) / (HELIX_CLIMB - FIG_SPLIT);
			float rr = HELIX_W * (1.0 - smoothstep(0.85, 1.0, uc));
			float p = 6.2831853 * FIG_TURNS * uc + (f < 0.5 ? 0.0 : 3.14159);
			return vec3(rr * cos(p), mix(-HELIX_H, HELIX_H, uc), rr * sin(p));
		}
		return vec3(0.0, mix(HELIX_H, -HELIX_H, (u - HELIX_CLIMB) / (1.0 - HELIX_CLIMB)), 0.0);
	}
	if (sh == FG_RINGV) {
		if (u < FIG_SPLIT) return vec3(0.0);
		float a = 6.2831853 * (0.5 * ud + (f < 0.5 ? 0.0 : 0.5));
		return vec3(0.0, cos(a), sin(a)) * RING_R;
	}
	if (sh == FG_HELIXF) {
		if (u < FIG_SPLIT) return vec3(0.0);
		float p = 6.2831853 * FIG_TURNS * ud + (f < 0.5 ? 0.0 : 3.14159);
		return vec3(HELIX_W * cos(p), HELIX_W * sin(p), mix(-RING_R, RING_R, ud));
	}
	if (sh == FG_LASH || sh == FG_PUNCHIN) {
		float ab = (u - FIG_SPLIT) / (1.0 - FIG_SPLIT);
		float tri = ab < 2.0 * PUNCH_SHOT ? (ab < PUNCH_SHOT ? ab / PUNCH_SHOT : (2.0 * PUNCH_SHOT - ab) / PUNCH_SHOT) : (sh == FG_LASH ? 0.0 : 1.0);
		return vec3(0.0, 0.0, -PUNCH_D * 0.6 * tri * (sh == FG_LASH ? 1.0 : -1.0));
	}
	if (sh == FG_SHEAR) {
		float x = mix(-RING_R, RING_R, u);
		return vec3(f < 0.5 ? x : -x, 0.16 * (f < 0.5 ? 1.0 : -1.0), 0.0);
	}
	if (sh == FG_LISSA) {
		float a = 6.2831853 * u + (f < 0.5 ? 0.0 : 1.5707963);
		return vec3(sin(2.0 * a), sin(3.0 * a + 0.7854), 0.14 * cos(5.0 * a)) * (RING_R * 0.68 * (f < 0.5 ? 1.0 : 0.86));
	}
	if (sh == FG_STAR) {
		float a = 6.2831853 * u + (f < 0.5 ? 0.0 : 0.6283185);
		vec2 p = 0.58 * vec2(cos(2.0 * a), sin(2.0 * a)) + 0.38 * vec2(cos(3.0 * a), -sin(3.0 * a));
		return vec3(p, 0.10 * sin(5.0 * a)) * (RING_R * 0.78 * (f < 0.5 ? 1.0 : 0.85));
	}
	if (sh == FG_TREFOIL) {
		float a = 6.2831853 * u + (f < 0.5 ? 0.0 : 3.14159);
		float r = 0.62 + 0.32 * cos(3.0 * a);
		return vec3(r * cos(2.0 * a), r * sin(2.0 * a), 0.16 * sin(3.0 * a)) * (RING_R * 0.75);
	}
	if (sh == FG_BUTTERFLY) {
		float a = 6.2831853 * u + (f < 0.5 ? 0.0 : 3.14159);
		float r = 0.48 - 0.22 * cos(4.0 * a) + 0.26 * pow(sin(2.0 * a), 2.0);
		return vec3(sin(a) * r, cos(a) * (r + 0.18 * cos(2.0 * a)), 0.10 * sin(4.0 * a)) * (RING_R * 1.05 * (f < 0.5 ? 1.0 : 0.84));
	}
	if (sh == FG_CLOVER) {
		float a = 6.2831853 * u + (f < 0.5 ? 0.0 : 0.7853982);
		vec2 p = 0.68 * vec2(cos(a), sin(a)) - 0.28 * vec2(cos(5.0 * a), sin(5.0 * a));
		return vec3(p, 0.12 * cos(4.0 * a)) * (RING_R * 0.74 * (f < 0.5 ? 1.0 : 0.78));
	}
	return vec3(0.0);
}
float figBeats(int id) { return id == FG_HELIX ? FIG_HELIX_BEATS : FIG_BEATS; }
#define LAB_N 64
int labFig(int mv) {
	if (mv >= 1 && mv <= 11) { int f[11] = int[11](FG_HELIX, FG_RINGV, FG_HELIXF, FG_LASH, FG_PUNCHIN, FG_SHEAR, FG_LISSA, FG_STAR, FG_TREFOIL, FG_BUTTERFLY, FG_CLOVER); return f[mv - 1]; }
	if (mv == 12) return FG_WALLPUNCH;
	if (mv == 13) return FG_EYERINGS;
	if (mv == 14) return FG_EYEARM;
	return 0;
}
int labBody(int mv) {
	int b[36] = int[36](SH_RING, SH_ARROW, SH_V, SH_BALL, SH_CORK, SH_RIBBON, SH_STRIPE, SH_FUNNEL, SH_DISK, SH_TORUS, SH_GALAXY, SH_JELLY,
	                    SH_COMET, SH_CROWN, SH_ATOM, SH_MANTA, SH_TREFOIL, SH_MOBIUS, SH_SLINKY, SH_BRAID, SH_BRAID2,
	                    SH_FB_WING, SH_FB_TEAR, SH_FB_VORTEX, SH_FB_PLAIT, SH_FB_MANTA, SH_FB_BELL, SH_FB_CAGE, SH_FB_KNOT, SH_FB_CADU, SH_FB_KLEIN,
	                    SH_FB_VIC, SH_FB_ECH, SH_FB_ASTERN, SH_FB_DOME, SH_FB_ABR);
	return (mv >= 29 && mv <= 64) ? b[mv - 29] : -1;
}
vec4 msNeutral(vec4 v) { return mix(v, vec4(0.5), vec4(lessThan(v, vec4(0.0)))); }
bool figIsTraced(int id) { return id >= 1 && id <= 15; }
bool figKicked(int id) { return id == FG_RINGV || id == FG_LASH || id == FG_SHEAR || id == FG_CLOVER || id == FG_EYERINGS || id == FG_EYEARM; }
int pickFigureAll(vec4 A, vec4 B, float last, float h1, float h2, float salt, bool wpOk, bool eyeOk, float eyeLast) {
	int ids[13] = int[13](FG_HELIX, FG_RINGV, FG_HELIXF, FG_LASH, FG_PUNCHIN, FG_SHEAR, FG_LISSA, FG_STAR, FG_TREFOIL, FG_BUTTERFLY, FG_CLOVER, FG_WALLPUNCH, FG_EYERINGS);
	float sc[13] = float[13](0.8 * A.x + 0.4 * A.z,
	                         0.8 * A.y + 0.4 * B.x,
	                         0.8 * A.w + 0.3 * A.z,
	                         1.1 * B.w + 0.4 * A.x,
	                         0.9 * B.w + 0.4 * A.x,
	                         0.7 * B.x + 0.6 * A.z,
	                         0.8 * A.z + 0.3 * B.y,
	                         0.8 * B.x + 0.3 * A.w,
	                         0.8 * B.z + 0.3 * A.y,
	                         0.8 * B.y + 0.3 * A.w,
	                         0.6 * A.x + 0.5 * A.z,
	                         0.8 * A.y + 0.5 * B.w,
	                         0.6 * A.y + 0.5 * A.x);
	int best = 0; float bs = -1e9;
	for (int n = 0; n < 13; n++) {
		if (figKicked(ids[n])) continue;
		if (ids[n] == FG_WALLPUNCH && !wpOk) continue;
		if (ids[n] == FG_EYERINGS && !eyeOk) continue;
		float id = float(ids[n]);
		float v = sc[n] + PICK_HASH * hash11(salt * 13.7 + float(n))
		        - (abs(id - last) < 0.5 ? PICK_PEN1 : 0.0) - (abs(id - h1) < 0.5 ? PICK_PEN2 : 0.0) - (abs(id - h2) < 0.5 ? PICK_PEN3 : 0.0);
		if (v > bs) { bs = v; best = ids[n]; }
	}
	if (best == FG_EYERINGS && eyeLast < 1.5) best = FG_EYEARM;
	return best;
}
vec3 eyeRings(int gi, float b) {
	int ring = gi / 24, j = gi % 24;
	float R = EYE_R * (1.0 - 0.2 * float(ring));
	float dir = (ring % 2 == 0) ? 1.0 : -1.0;
	float th = 6.2831853 * (float(j) / 24.0 + dir * b / EYE_RING_BEATS);
	vec3 p = vec3(cos(th), sin(th), 0.0) * R;
	for (int r = 3; r >= 0; r--) {
		if (r > ring) continue;
		vec3 ax = (r % 2 == 0) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
		p = rotAx(p, ax, 6.2831853 * b / (8.0 + 4.0 * float(r)) * ((r % 2 == 0) ? 1.0 : -1.0) + 0.6 * float(r));
	}
	return p;
}
vec3 eyeArmillary(int gi, float b) {
	float R = EYE_R, eps = 0.4091;
	int ring, j, n;
	if (gi < 16)      { ring = 0; j = gi;      n = 16; }
	else if (gi < 30) { ring = 1; j = gi - 16; n = 14; }
	else if (gi < 44) { ring = 2; j = gi - 30; n = 14; }
	else if (gi < 60) { ring = 3; j = gi - 44; n = 16; }
	else if (gi < 72) { ring = 4; j = gi - 60; n = 12; }
	else if (gi < 84) { ring = 5; j = gi - 72; n = 12; }
	else              { ring = 6; j = gi - 84; n = 12; }
	float sg = (ring % 2 == 0) ? 1.0 : -1.0;
	float th = 6.2831853 * (float(j) / float(n) + sg * b / EYE_RING_BEATS);
	vec3 p;
	if (ring == 0)      p = vec3(cos(th), 0.0, sin(th)) * R;
	else if (ring == 1) p = vec3(cos(th), sin(th), 0.0) * R;
	else if (ring == 2) p = vec3(0.0, sin(th), cos(th)) * R;
	else if (ring == 3) p = rotAx(rotAx(vec3(cos(th), 0.0, sin(th)) * R, vec3(0.0, 0.0, 1.0), eps), vec3(0.0, 1.0, 0.0), 6.2831853 * b / 64.0);
	else if (ring == 4) p = vec3(cos(th) * R * cos(eps),  R * sin(eps), sin(th) * R * cos(eps));
	else if (ring == 5) p = vec3(cos(th) * R * cos(eps), -R * sin(eps), sin(th) * R * cos(eps));
	else {
		float zz = 1.0 - 2.0 * (float(j) + 0.5) / float(n), rr = sqrt(max(0.0, 1.0 - zz * zz));
		float p2 = 2.3999632 * float(j) + 6.2831853 * b / 16.0;
		p = vec3(rr * cos(p2), zz, rr * sin(p2)) * (0.16 * R);
	}
	return rotAx(rotAx(p, normalize(vec3(1.0, 0.0, 0.3)), 0.45), vec3(0.0, 1.0, 0.0), 6.2831853 * b / EYE_TURN_BEATS);
}

float formLen(int sh) {
	if (sh == SH_ARROW) return 3.08;
	if (sh == SH_V) return 1.80;
	if (sh == SH_BALL) return 1.5;
	if (sh == SH_CORK) return 2.2;
	if (sh == SH_RIBBON) return 0.8;
	if (sh == SH_STRIPE) return 1.5;
	if (sh == SH_FUNNEL) return FUNNEL_LEN;
	if (sh == SH_COMET) return COMET_LEN;
	if (sh == SH_BRAID || sh == SH_BRAID2) return BRAID_SPAN / FORM_R;
	if (sh >= FB_BASE) return FS_LEN * FB_SCALE / FORM_R;
	return 0.6;
}
vec3 formSlot(int sh, int k, float df) {
	float fk = float(k) + df;
	if (sh == SH_RING) { float a = 6.2831853 * fk / float(N_LEAD); return vec3(cos(a), sin(a), -0.3); }
	if (sh == SH_ARROW) { float u = fk / float(N_LEAD - 1); return vec3(0.12 * sin(fk * 2.1), 0.12 * cos(fk * 2.1), -u * 3.08); }
	if (sh == SH_V) {
		float arm = k < N_LEAD / 2 ? -1.0 : 1.0;
		float j = (float(k % (N_LEAD / 2)) + df + 1.0) * 6.0 / float(N_LEAD / 2);
		return vec3(arm * j * 0.30, 0.05 * j, -j * 0.30);
	}
	if (sh == SH_CORK) { float u = fk / float(N_LEAD - 1), ph = 6.2831853 * 2.5 * u; return vec3(0.35 * cos(ph), 0.35 * sin(ph), -u * 2.2); }
	if (sh == SH_RIBBON) { float u = fk / float(N_LEAD - 1) * 2.0 - 1.0; return vec3(u * 1.3, 0.22 * sin(u * 4.7), -0.45 * u * u - 0.2); }
	if (sh == SH_COMET) {
		float u = fk / float(N_LEAD - 1);
		float th = 6.2831853 * fract(fk * 0.61803398875);
		float r = COMET_R * (1.0 - u);
		return vec3(r * cos(th), r * sin(th), -COMET_LEN * u);
	}
	float zf = 1.0 - 2.0 * (fk + 0.5) / float(N_LEAD);
	float rf = sqrt(max(0.0, 1.0 - zf * zf)), af = 2.39996323 * fk;
	return vec3(rf * cos(af), rf * sin(af), zf) * 0.75 - vec3(0.0, 0.0, 0.6);
}
vec3 stripeSlot(int k, float df, float bc) {
	int per = N_LEAD / 3;
	int ring = k / per;
	float lat = (float(ring) - 1.0) * 0.698;
	float a = 6.2831853 * ((float(k % per) + df) / float(per) + STRIPE_SPIN * bc + 0.5 * float(ring) / float(per));
	float r = 0.75 * cos(lat);
	return vec3(r * cos(a), r * sin(a), 0.75 * sin(lat) - 0.6);
}
vec3 funnelSlot(int k, float df, float bc) {
	float u = (float(k) + df) / float(N_LEAD - 1);
	float r = FUNNEL_R0 * pow(FUNNEL_R1 / FUNNEL_R0, u);
	float ph = 6.2831853 * (FUNNEL_TURNS * u + FUNNEL_SPIN * bc);
	return vec3(r * cos(ph), r * sin(ph), -u * FUNNEL_LEN);
}
vec3 friendlySlot(int sh, int k, float df, int f, float bc) {
	float fk = float(k) + df + 0.5 * float(f);
	if (sh == SH_TORUS) {
		float u = 6.2831853 * (fk / float(N_LEAD) + TOR_SPIN * bc), v = 6.2831853 * fract(fk * 0.61803398875);
		float rm = TOR_RM + TOR_RT * cos(v);
		return vec3(rm * cos(u), TOR_RT * sin(v), rm * sin(u)) * 2.0;
	}
	if (sh == SH_GALAXY) {
		float a = mod(float(k), 2.0) + 2.0 * float(f), tt = fk / float(N_LEAD);
		float ang = 6.2831853 * (a * 0.25 + tt * GAL_TWIST) + 6.2831853 * GAL_SPIN * bc;
		float r = mix(GAL_R0, GAL_R1, tt);
		float px = r * cos(ang), pz = r * sin(ang);
		return vec3(px, pz * sin(GAL_TILT), pz * cos(GAL_TILT)) * 2.0;
	}
	if (sh == SH_JELLY) {
		float pl = fract(bc / JEL_BEATS);
		float pulse = 1.0 - 0.22 * pow(1.0 - pl, 2.0);
		float th = 6.2831853 * fract(fk * 0.61803398875);
		if (mod(float(k), 5.0) < 3.0) {
			float y = clamp(fk / float(N_LEAD), 0.0, 1.0);
			float rr = JEL_BELL * pulse * cos(1.5707963 * y);
			return vec3(rr * cos(th), rr * sin(th), JEL_BELL * pulse * sin(1.5707963 * y)) * 2.0;
		}
		float tt = clamp(fk / float(N_LEAD), 0.0, 1.0);
		float rr = JEL_BELL * pulse * (0.85 + 0.15 * (1.0 - tt));
		return vec3(rr * cos(th), rr * sin(th), -JEL_TENT * tt) * 2.0;
	}
	if (sh == SH_CROWN) {
		float s = fk / float(N_LEAD);
		float a = 6.2831853 * (s + CROWN_SPIN * bc);
		float spike = pow(1.0 - abs(2.0 * fract(s * CROWN_N) - 1.0), 2.0);
		float r = CROWN_R + CROWN_SPIKE * spike;
		return vec3(r * cos(a), 0.5 * CROWN_SPIKE * spike, r * sin(a)) * 2.0;
	}
	if (sh == SH_ATOM) {
		float spin = 6.2831853 * ATOM_SPIN * bc;
		if (mod(float(k), 8.0) < 0.5) return 0.14 * vec3(cos(float(k)), sin(float(k) * 1.3), cos(float(k) * 0.7));
		int o = k % 3;
		float th = 6.2831853 * (float(k) / float(N_LEAD)) + spin + float(o) * 2.0943951;
		vec3 e = vec3(ATOM_A * cos(th), ATOM_B * sin(th), 0.0);
		float a = float(o) * ATOM_TILT;
		mat3 rx = mat3(1.0, 0.0, 0.0, 0.0, cos(a), -sin(a), 0.0, sin(a), cos(a));
		return rx * e * 2.0;
	}
	if (sh == SH_MANTA) {
		int cols = 8, rows = N_LEAD / cols;
		float fk2 = float(k) + df;
		float cx = (mod(fk2, float(cols)) / float(cols - 1)) * 2.0 - 1.0;
		float rw = (floor(fk2 / float(cols)) / float(rows - 1)) * 2.0 - 1.0;
		float ph = 6.2831853 * bc / MAN_BEATS;
		float y = MAN_W * sin(cx * 2.6 + rw * 1.1 + ph);
		return vec3(cx * MAN_X, y, rw * MAN_Z) * 1.6;
	}
	if (sh == SH_TREFOIL) {
		float tt = 6.2831853 * fk / float(N_LEAD) + 6.2831853 * TREF_SPIN * bc;
		return TREF_S * vec3(sin(tt) + 2.0 * sin(2.0 * tt), cos(tt) - 2.0 * cos(2.0 * tt), -sin(3.0 * tt)) * 1.6;
	}
	if (sh == SH_MOBIUS) {
		float uu = 6.2831853 * fk / float(N_LEAD) + 6.2831853 * MOB_SPIN * bc;
		float vv = MOB_W * sin(fk * 2.39996323);
		float ch = cos(uu * 0.5);
		vec3 C = MOB_R * vec3(cos(uu), sin(uu), 0.0);
		vec3 W = vec3(ch * cos(uu), ch * sin(uu), sin(uu * 0.5));
		return (C + vv * W) * 2.0;
	}
	if (sh == SH_SLINKY) {
		int rings = 6;
		float fk2 = float(k) + df;
		int ri = int(mod(float(k), float(rings)));
		float per = float(N_LEAD) / float(rings);
		float a = 6.2831853 * (floor(fk2 / float(rings)) / per) + 0.3 * float(f);
		float comp = 1.0 - SLK_AMP * (0.5 + 0.5 * cos(6.2831853 * bc / SLK_BEATS));
		float z = (float(ri) - 0.5 * float(rings - 1)) * SLK_GAP * comp;
		return vec3(SLK_R * cos(a), SLK_R * sin(a), z) * 2.0;
	}
	if (sh == SH_DISK) {
		float fine = 4.0 * float(k) + 1.5 + 4.0 * df;
		float r = DISK_R * sqrt((fine + 0.5) / float(N_LMAX));
		float th = 2.39996323 * fine + 6.2831853 * DISK_SPIN * bc;
		return vec3(r * cos(th), r * sin(th), 0.0) * 2.0;
	}
	return vec3(0.0);
}
bool isFriendly(int sh) { return sh == SH_TORUS || sh == SH_GALAXY || sh == SH_JELLY || sh == SH_CROWN || sh == SH_ATOM || sh == SH_MANTA || sh == SH_TREFOIL || sh == SH_MOBIUS || sh == SH_SLINKY || sh == SH_DISK; }
vec3 bodySlot(int sh, int k, float df, int f, float bc) {
	if (sh == SH_STRIPE) return stripeSlot(k, df, bc);
	if (sh == SH_FUNNEL) return funnelSlot(k, df, bc);
	if (isFriendly(sh)) return friendlySlot(sh, k, df, f, bc);
	return formSlot(sh, k, df);
}
vec3 braidSlot(int sh, int k, float bc) {
	int ns = sh == SH_BRAID2 ? 6 : 3;
	int s = k % ns, q = k / ns;
	float per = max(float(N_LEAD / ns), 1.0);
	float u = (float(q) + 0.5) / per;
	float s3 = float(s % 3), set = float(s / 3);
	float ang = 6.2831853 * (BRAID_TURNS * u + s3 / 3.0 + BRAID_SPIN * bc);
	float rad = BRAID_RAD * (1.0 + 0.9 * set);
	rad *= 1.0 + BRAID_RIPPLE * sin(6.2831853 * (BRAID_TURNS * 3.0 * u + s3));
	return vec3(rad * cos(ang), rad * sin(ang), -u * BRAID_SPAN);
}
vec3 waspOrbit(int g, float bc) {
	float fg = float(g);
	float R = mix(WASP_ORB0, WASP_ORB1, hashW(fg, 1.0));
	float ph = 6.2831853 * (hashW(fg, 2.0) + bc / (WASP_TURN * mix(0.7, 1.3, hashW(fg, 3.0)))) * (hashW(fg, 4.0) < 0.5 ? 1.0 : -1.0)
	         + WASP_SURGE * sin(3.14159 * bc / 2.0 + 6.2831853 * hashW(fg, 5.0));
	float r = R * (1.0 + 0.3 * sin(bc * 0.9 + fg * 1.7));
	vec3 n = normalize(vec3(hashW(fg, 6.0) - 0.5, 1.0, hashW(fg, 7.0) - 0.5) + vec3(0.6 * sin(bc * 0.13 + fg), 0.0, 0.6 * cos(bc * 0.11 + fg)));
	vec3 e1 = normalize(cross(n, vec3(0.0, 0.0, 1.0))), e2 = cross(n, e1);
	vec3 o = r * (cos(ph) * e1 + sin(ph) * e2) + n * 0.25 * sin(bc * 1.3 + fg * 2.3);
	o.x *= 0.6;
	return o;
}
vec3 waspCentre(int g, float bc) {
	float w = floor((bc + 1.0) / 2.0), p = fract((bc + 1.0) / 2.0);
	int t = int(hashW(float(g), w + 11.0) * 7.0);
	t += t >= g ? 1 : 0;
	float strike = WASP_STRIKE * pow(sin(3.14159 * p), 4.0) * step(0.25, hashW(float(g), w + 23.0));
	return mix(waspOrbit(g, bc), waspOrbit(t, bc), strike);
}
float possess(int g, float bc) {
	float cyc = floor(bc / POSS_BEATS), q = bc - cyc * POSS_BEATS;
	float grow = clamp((q - 1.0) / 5.0, 0.0, 1.0);
	float twitch = (q >= 1.0 && q < 7.0) ? (0.06 + 0.14 * grow) * exp(-6.0 * fract(q)) : 0.0;
	float tb = 2.0 + 4.0 * hashW(float(g), cyc + 41.0);
	float tear = smoothstep(tb, tb + 0.25, q) * (1.0 - smoothstep(7.0, 7.75, q)) * step(1.0 - POSS_TEAR, hashW(float(g), cyc + 43.0));
	return max(twitch, tear);
}
vec3 waspSlot(int k, float df, int f, float bc) {
	int c = k / WASP_PER;
	float fk = float(k % WASP_PER) + df;
	float zf = 1.0 - 2.0 * (fk + 0.5) / float(WASP_PER);
	float rf = sqrt(max(0.0, 1.0 - zf * zf)), af = 2.39996323 * fk + bc * 3.0;
	return waspCentre(f * 4 + c, bc) + vec3(rf * cos(af), zf, rf * sin(af)) * WASP_R;
}
vec3 columnSlot(int k, float df, int f) {
	float t = (float(k) + df + 0.5 * float(f)) / float(N_LEAD - 1);
	float th = COL_TWIST * t + 6.2831853 * 0.61803398875 * (float(k) + 0.5 * float(f));
	float r = COL_R0 + (COL_R1 - COL_R0) * t * t;
	return vec3(r * cos(th), COL_Y0 + COL_H * t, r * sin(th));
}
int splitLvl(int sh) {
	if (sh >= FB_BASE) return 0;
	if (sh == SH_STRIPE || sh == SH_DISK) return 2;
	if (sh == SH_TORUS || sh == SH_JELLY || sh == SH_COMET || sh == SH_ATOM || sh == SH_MANTA || sh == SH_TREFOIL || sh == SH_MOBIUS || sh == SH_SLINKY || sh == SH_BRAID2) return 1;
	return 0;
}
int nLeaders(float lv) { return lv > 1.0 + 1e-3 ? N_LMAX : (lv > 1e-3 ? 2 * N_LEAD : N_LEAD); }
int leaderParent(int m) { int g = m / N_LEAD, k = m % N_LEAD; return g == 3 ? k + N_LEAD : (g == 0 ? m : k); }
float leaderOff(int m, float lv) {
	int g = m / N_LEAD;
	float e1 = smoothstep(0.0, 1.0, clamp(lv, 0.0, 1.0)), e2 = smoothstep(0.0, 1.0, clamp(lv - 1.0, 0.0, 1.0));
	return ((g == 1 || g == 3) ? 0.25 : -0.25) * e1 + ((g == 2 || g == 3) ? 0.125 : -0.125) * e2;
}
int pickBody(int pocket, float R, float spd, int cur, int h1, int h2, float salt) {
	float fast = smoothstep(0.06, 0.18, spd), tight = 1.0 - smoothstep(0.12, 0.50, R);
	float straight = 1.0 - tight, mid = 1.0 - abs(2.0 * tight - 1.0), calm = 1.0 - fast;
	int ids[10]; float sc[10];
	if (pocket == 0) {
		ids = int[10](SH_MANTA, SH_JELLY, SH_MOBIUS, SH_SLINKY, SH_TORUS, SH_RING, SH_FB_MANTA, SH_FB_DOME, SH_FB_ABR, SH_FB_KLEIN);
		sc = float[10](1.0 - fast, (1.0 - fast) * 0.9, 0.7, (1.0 - tight) * 0.8, 0.75, (1.0 - fast) * (1.0 - tight) * 0.7,
		               calm * straight * 0.85 + 0.1, calm * 0.9, calm * straight, calm * 0.7);
	} else if (pocket == 1) {
		ids = int[10](SH_GALAXY, SH_CROWN, SH_ATOM, SH_TREFOIL, SH_DISK, SH_BALL, SH_FB_TEAR, SH_FB_BELL, SH_FB_KNOT, SH_FB_CAGE);
		sc = float[10](0.8, 0.75, 0.8, 0.6, (1.0 - fast) * 0.8, tight * 0.9,
		               0.4 + fast * straight * 0.6, 0.4 + tight * (1.0 - 0.5 * fast) * 0.5, 0.6, 0.4 + calm * 0.4);
	} else if (pocket == 2) {
		ids = int[10](SH_BRAID, SH_BRAID2, SH_CORK, SH_FB_VORTEX, SH_FB_PLAIT, SH_FB_CADU, SH_FB_CAGE, SH_FB_KNOT, SH_FB_ECH, SH_FB_KLEIN);
		sc = float[10](0.9, 0.85, 0.5, 0.4 + fast * mid * 0.6, 0.75, 0.4 + fast * mid * 0.5, 0.6, 0.6, 0.3 + fast * mid * 0.6, 0.55);
	} else {
		ids = int[10](SH_ARROW, SH_V, SH_CORK, SH_COMET, SH_FUNNEL, SH_STRIPE, SH_FB_WING, SH_FB_VIC, SH_FB_ECH, SH_FB_ASTERN);
		sc = float[10](fast * (1.0 - tight), tight * (1.0 - 0.5 * fast), fast * (1.0 - abs(2.0 * tight - 1.0)), fast * (1.0 - tight) * 0.9, fast * tight, 0.5,
		               fast * straight, fast * straight * 1.1, fast * mid, tight);
	}
	int best = ids[0]; float bs = -1e9;
	for (int n = 0; n < 10; n++) {
		float v = sc[n] + BODY_HASH * hash11(salt * 7.0 + float(n))
		        - (ids[n] == cur ? 0.8 : 0.0) - (ids[n] == h1 ? 0.6 : 0.0) - (ids[n] == h2 ? 0.4 : 0.0);
		if (v > bs) { bs = v; best = ids[n]; }
	}
	return best;
}
vec3 shapeLocal(int sel, float u, float v, float len, float rad, float beat, float reord) {
    float th = 6.2831853 * u;
    float env = sin(3.14159265 * pow(clamp(v, 0.0, 1.0), 0.72));
    if (sel == 0) {
        float s = 2.0 * u - 1.0;
        float sweep = pow(abs(s), 1.35);
        float thick = rad * 0.38 * env * sqrt(max(0.0, 1.0 - s * s));
        float arch  = FS_FOLD * rad * 1.30 * (1.0 - cos(s * 1.5707963));
        float w = 6.2831853 * v * 13.0;
        return vec3(s * rad * 1.85,
                    arch + thick * sin(w),
                    -len * (0.60 * sweep + 0.40 * v) + thick * cos(w));
    }
    if (sel == 1) {
        float r = rad * 1.35 * pow(v, 0.35) * pow(max(0.001, 1.0 - v), 0.65) * 2.25;
        return vec3(r * cos(th), r * sin(th), -len * v);
    }
    if (sel == 2) {
        float hth = 6.2831853 * (FS_HELIX_TURNS * v - 0.125 * beat);
        float Rh = rad * 0.78 * env, rt = rad * 0.34 * env;
        return vec3(Rh * cos(hth) + rt * cos(th), Rh * sin(hth) + rt * sin(th), -len * v);
    }
    if (sel == 3) {
        float st = u * 3.0, s3 = floor(st), fu = fract(st) * 6.2831853;
        float ph = 6.2831853 * (FS_BRAID_TURNS * v - 0.10 * beat) + 2.0943951 * (s3 + reord);
        vec2 c = vec2(rad * 1.20 * sin(ph), rad * 0.70 * sin(2.0 * ph)) * env;
        float rt = rad * 0.25 * env;
        return vec3(c.x + rt * cos(fu), c.y + rt * sin(fu), -len * v);
    }
    if (sel == 4) {
        float s = 2.0 * u - 1.0;
        float span = rad * 1.95 * env;
        float flap = 0.22 * rad * sin(6.2831853 * beat / 2.0 - abs(s) * 1.6) * abs(s);
        float core = rad * 0.34 * env * exp(-3.5 * s * s) * sin(6.2831853 * v * 8.5);
        return vec3(s * span, flap + core, -len * (v * 0.75 + 0.25 * s * s));
    }
    if (sel == 5) {
        float a2 = pow(clamp(v, 0.0, 1.0), 0.65) * 1.75;
        float R = rad * 1.55;
        return vec3(R * sin(a2) * cos(th), R * sin(a2) * sin(th), -len * 0.85 * (1.0 - cos(a2)));
    }
    if (sel == 6) {
        float st = u * 2.0, s2 = floor(st), fu = fract(st) * 6.2831853;
        float ph = 6.2831853 * (1.35 * v - 0.08 * beat) + 3.14159265 * (s2 + 0.5 * reord);
        float rt = rad * 0.24 * env;
        return vec3(rad * 1.30 * env * sin(3.0 * ph) + rt * cos(fu),
                    rad * 1.05 * env * sin(4.0 * ph + 0.7854) + rt * sin(fu),
                    -len * v + len * 0.18 * env * cos(2.0 * ph));
    }
    if (sel == 7) {
        float t = 6.2831853 * (v - 0.06 * beat) + 2.0943951 * reord;
        float r2 = 0.65 + 0.32 * cos(3.0 * t);
        float rt = rad * 0.26 * (0.45 + 0.55 * env);
        vec3 kPos = vec3(r2 * cos(2.0 * t), r2 * sin(2.0 * t), 0.38 * sin(3.0 * t)) * (rad * 1.45);
        return vec3(kPos.x + rt * cos(th), kPos.y + rt * sin(th), -len * (0.20 + 0.60 * v) + kPos.z);
    }
    if (sel == 8) {
        float st = u * 2.0, s2 = floor(st), fu = fract(st) * 6.2831853;
        float ph = 6.2831853 * (2.0 * v - 0.10 * beat) + 1.5707963 * reord;
        float crossR = rad * 1.35 * env * cos(ph) * (s2 < 0.5 ? 1.0 : -1.0);
        float spinA = 6.2831853 * (v + 0.05 * beat);
        float rt = rad * 0.26 * env;
        return vec3(crossR * cos(spinA) + rt * cos(fu),
                    crossR * sin(spinA) + rt * sin(fu),
                    -len * v);
    }
    if (sel == 10) {
        float r = rad * 0.30 * env;
        return vec3(r * cos(th), r * sin(th), -len * FIG_TRAIL_LEN * v);
    }
    float twist = 3.14159265 * (1.5 * v - 0.12 * beat + 0.5 * reord);
    vec2 lem = vec2(sin(th), 0.62 * sin(2.0 * th)) * (rad * 1.40 * env);
    float ct = cos(twist), stw = sin(twist);
    return vec3(ct * lem.x - stw * lem.y, stw * lem.x + ct * lem.y, -len * v);
}
vec3 flightFormLocal(int sel, int kp, int nPt, float scl) {
    int e = kp % FF_N, j = kp / FF_N, nSub = max((nPt + FF_N - 1) / FF_N, 1);
    float sp = FF_SPREAD * scl;
    vec3 c = vec3(0.0);
    if (sel == 11) {
        float p = floor(float(e + 1) * 0.5), sd = (e % 2 == 1) ? 1.0 : -1.0;
        c = vec3(sd * p * sp, p * sp * 0.15, -p * sp);
    } else if (sel == 12) {
        c = vec3(float(e) * sp, 0.0, -float(e) * sp);
    } else if (sel == 13) {
        c = vec3(0.0, ((e % 2 == 1) ? 0.15 : -0.15) * sp * min(float(e), 1.0), -float(e) * sp * 0.90);
    } else if (sel == 14) {
        float a = 1.5707963 * float(e) / float(FF_N - 1), th = 6.2831853 * fract(float(e) * 0.6180340);
        float R = sp * float(FF_N - 1) * 0.55;
        c = vec3(sin(a) * cos(th) * R, sin(a) * sin(th) * R, -(1.0 - cos(a)) * R * 1.2);
    } else {
        float q = float(e) - 0.5 * float(FF_N - 1);
        c = vec3(q * sp * 0.85, 0.0, -abs(q) * sp * 0.15);
    }
    float r = FF_R * scl * sqrt((float(j) + 0.5) / float(nSub)), ang = 2.3999632 * float(j);
    return c + vec3(r * cos(ang), r * sin(ang), -0.5 * r);
}
#endif

==compute init 1024==
layout(local_size_x = 64) in;
layout(rgba32f, binding = boid_UNIT) uniform writeonly highp image2D boid_w;
layout(r32f, binding = ctl_UNIT) uniform readonly highp image2D ctl;
void main() {
	int i = int(gl_GlobalInvocationID.x);
	if (i >= N_ALL || imageLoad(ctl, ivec2(C_INIT, 0)).r > 0.5) return;
	uint s = pcg(uint(i) * 7919u + 17u);
	int f = i / N_PER;
	vec3 p = vec3(f == 0 ? -0.3 : 0.3, 0.0, 0.0) + rndDir(s) * 0.15 * pow(rnd(s), 1.0 / 3.0);
	vec3 v = (vec3(rnd(s), rnd(s), rnd(s)) - 0.5) * 0.17;
	imageStore(boid_w, ivec2(i, 0), vec4(p, float(i % 2)));
	imageStore(boid_w, ivec2(i, 1), vec4(v, rnd(s)));
	imageStore(boid_w, ivec2(i, 2), vec4(1.0, 1.0, 0.0, 0.0));
}

==compute reduce 512==
layout(local_size_x = 256) in;
layout(r32f, binding = flk_UNIT) uniform highp image2D flk;
shared vec3 sP[256], sV[256];
shared float sR[256];
void main() {
	int f = int(gl_WorkGroupID.x), t = int(gl_LocalInvocationID.x);
	vec3 ps = vec3(0.0), vs = vec3(0.0);
	for (int e = t; e < N_PER; e += 256) {
		vec3 p0 = texelFetch(boid, ivec2(f * N_PER + e, 0), 0).xyz, v0 = texelFetch(boid, ivec2(f * N_PER + e, 1), 0).xyz;
		ps += dot(p0, p0) < 1e30 ? p0 : vec3(0.0);
		vs += dot(v0, v0) < 1e30 ? v0 : vec3(0.0);
	}
	sP[t] = ps; sV[t] = vs;
	barrier();
	for (int s = 128; s > 0; s >>= 1) { if (t < s) { sP[t] += sP[t + s]; sV[t] += sV[t + s]; } barrier(); }
	vec3 c = sP[0] / float(N_PER);
	barrier();
	float rs = 0.0;
	for (int e = t; e < N_PER; e += 256) {
		vec3 p0 = texelFetch(boid, ivec2(f * N_PER + e, 0), 0).xyz;
		if (!(dot(p0, p0) < 1e30)) p0 = vec3(0.0);
		rs += dot(p0 - c, p0 - c);
	}
	sR[t] = rs;
	barrier();
	for (int s = 128; s > 0; s >>= 1) { if (t < s) sR[t] += sR[t + s]; barrier(); }
	if (t == 0) {
		FLK_STORE(flk, f, vec4(c, sqrt(sR[0] / float(N_PER))));
		FLK_STORE(flk, 2 + f, vec4(sV[0] / float(N_PER), 0.0));
	}
}

==compute ctrl 1==
layout(local_size_x = 1) in;
layout(r32f, binding = ctl_UNIT) uniform highp image2D ctl;
layout(r32f, binding = flk_UNIT) uniform highp image2D flk;
#define FEAT_TAU    2.0
#define HI_TAU      0.3
#define PATH_BEATS  16.0
#define HAWK_TAU    0.35
#define HAWK_EDGE   1.3
#define WELL_G       2.5
#define PUSH_RATE    0.5
#define PUSH_EARLY   4.0
#define KICK_WOBBLE  0.12
#define POCKET_MAX   0.6
#define POCKET_TAU   1.0
#define PIR_R        0.035
#define PIR_BEATS    2.0
#define WINDUP_BEATS 8.0
#define SPONGE_NEAR  0.10
#define SPONGE_FAR   0.40
#define SPONGE_BEATS 4.0
#define SPLIT_E      0.85
#define SPLIT_I      0.20
#define SPLIT1_E     0.70
#define KICK_GATE   0.15
#define KICK_K      1.6
#define KICK_MIN    0.10
#define KICK_TAU    0.12
#define MIC_AVG_TAU    1.0
#define MIC_ONSET      0.07
#define MIC_FLOOR_RISE 0.004
#define MIC_HEARD_LO   0.15
#define MIC_HEARD_HI   0.30
#define MIC_SPAN       0.50
#define MIC_DYN_K      1.5
float C(int k) { return imageLoad(ctl, ivec2(k, 0)).r; }
void W(int k, float v) { imageStore(ctl, ivec2(k, 0), vec4(v)); }
float nd(float lin) { return clamp((20.0 * log(max(lin, 1e-9)) / log(10.0) + 60.0) / 60.0, 0.0, 1.0); }
float group(int lo, int hi) {
	float e = 0.0;
	for (int b = lo; b <= hi; b++) { float v = texelFetch(bands, ivec2(b, 0), 0).r; e += v * v; }
	return nd(sqrt(e));
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
	if (m == 3.0 && A.x >= SPLIT1_E && B.w >= SPLIT_I) m = 4.0;
	return m;
}
uniform float u_sub, u_bass, u_lowmid, u_mid, u_highmid, u_presence, u_brilliance, u_rms;
float dLev(int k) { return k == 0 ? u_sub : k == 1 ? u_bass : k == 2 ? u_lowmid : k == 3 ? u_mid : k == 4 ? u_highmid : k == 5 ? u_presence : k == 6 ? u_brilliance : u_rms; }
#ifdef DESKTOP
uniform float u_kick;
uniform float u_held;
uniform float u_move;
float dKick() { return u_kick; }
#endif
float isDouble(vec4 A, vec4 B) { return (A.x >= SPLIT_E && B.w >= SPLIT_I) ? 1.0 : 0.0; }
void main() {
	bool first = C(C_INIT) < 0.5;
	if (first) { W(C_INIT, 1.0); W(C_ODIR, 1.0); W(C_ODIRT, 1.0); W(C_WP_START, -1e4); W(C_WP_LAST, -1e4); W(C_EYE_LAST, -1e4); W(C_BT_REFR, 0.0); W(C_SALT, fract(time * 0.1373 + 0.29)); W(C_FIGH1, -1.0); W(C_FIGH2, -1.0); }
	float dt = clamp(time - C(C_TPREV), 0.002, 0.05);
	W(C_TPREV, time); W(C_DT, dt);

#ifdef DESKTOP
	float u_sub = dLev(0), u_bass = dLev(1), u_lowmid = dLev(2), u_mid = dLev(3);
	float u_highmid = dLev(4), u_presence = dLev(5), u_brilliance = dLev(6);
	float u_rms = dLev(7);
	bool clocked = beatConf > 0.25 && beatBpm > 30.0;
	float offB = min(beatPhase, 1.0 - beatPhase);
	bool onGrid = !clocked || offB < KICK_GATE;
	float kick = max(C(C_KICK) * exp(-dt / KICK_TAU), dKick());
	W(C_KICK, kick);
#else
	float u_sub = dLev(0), u_bass = dLev(1), u_lowmid = dLev(2), u_mid = dLev(3);
	float u_highmid = dLev(4), u_presence = dLev(5), u_brilliance = dLev(6);
	float u_rms = dLev(7);
	float flux = 0.0;
	for (int b = KICK_LO; b <= KICK_HI; b++) {
		float x = nd(texelFetch(bands, ivec2(b, 0), 0).r);
		flux += max(0.0, x - C(C_LOW0 + b - KICK_LO));
		W(C_LOW0 + b - KICK_LO, x);
	}
	float avg = C(C_ODFAVG);
	W(C_ODFAVG, avg + (flux - avg) * (1.0 - exp(-dt / 1.5)));
	bool clocked = beatConf > 0.25 && beatBpm > 30.0;
	float offB = min(beatPhase, 1.0 - beatPhase);
	bool onGrid = !clocked || offB < KICK_GATE;
	float kick = C(C_KICK) * exp(-dt / KICK_TAU);
	if (flux > max(KICK_MIN, KICK_K * avg) && onGrid) kick = 1.0;
	W(C_KICK, kick);
#endif
	float u_kick = kick;
	float jdspRms = u_rms;
	float micDb = nd(micAmplitude);
	float mAvg = C(C_MICAVG);
	W(C_MICAVG, mAvg + (micDb - mAvg) * (1.0 - exp(-dt / MIC_AVG_TAU)));
	float mFloor = C(C_MICFLR);
	mFloor = (mFloor <= 0.0 || micDb < mFloor) ? micDb : mFloor + dt * MIC_FLOOR_RISE;
	W(C_MICFLR, mFloor);
	float micHeard = smoothstep(MIC_HEARD_LO, MIC_HEARD_HI, micDb - mFloor) * (1.0 - smoothstep(0.08, 0.20, jdspRms));
	float mk = C(C_MICK) * exp(-dt / KICK_TAU);
	if (micDb - mAvg > MIC_ONSET && micDb > C(C_MICPRV) && onGrid) mk = 1.0;
	W(C_MICK, mk); W(C_MICPRV, micDb);
	if (micHeard > 0.0) u_kick = mix(u_kick, mk, micHeard);
	float heard = smoothstep(0.08, 0.20, jdspRms);
	float cached = u_ms_source > 0.5 ? 1.0 : 0.0;
	float cacheDrive = cached * (1.0 - heard);
	float micDyn = MIC_DYN_K * (micDb - mAvg) * micHeard;
	if (cacheDrive > 0.0) {
		float kStr = smoothstep(0.25, 0.7, u_ms_cur_a.x) * mix(0.6, 1.0, u_ms_cur_a.y);
		float kSyn = kStr * exp(-fract(beatCount) * 60.0 / max(beatBpm, 30.0) / KICK_TAU);
		u_kick = mix(u_kick, kSyn, cacheDrive * (1.0 - micHeard));
		u_rms = mix(u_rms, clamp(0.1 + 0.7 * u_ms_cur_a.x + micDyn, 0.0, 1.0), cacheDrive);
	} else if (micHeard > 0.0) {
		u_rms = mix(u_rms, 0.1 + 0.7 * clamp((micDb - mFloor) / MIC_SPAN, 0.0, 1.0), micHeard);
	}
	if (cacheDrive > 0.0) {
		float hiC = 0.6 * u_ms_cur_a.w + 0.4 * u_ms_cur_a.z;
		u_presence = mix(u_presence, hiC, cacheDrive);
		u_brilliance = mix(u_brilliance, hiC, cacheDrive);
	}
	float bcPrev = C(C_BCPREV);
	if (first || !(bcPrev == bcPrev) || abs(beatCount - bcPrev) > 8.0) bcPrev = beatCount;
	float dBReal = 0.0;
	if (clocked) { dBReal = clamp(beatCount - bcPrev, 0.0, dt * max(beatBpm, 60.0) / 60.0 * BC_CATCH); bcPrev += dBReal; }
	else bcPrev = beatCount;
	W(C_BCPREV, bcPrev);
	float mult = 1.0;
	float dBeats = max(dBReal * mult, dt * BASE_BPM / 60.0);
	float bc = mod(C(C_BC) + dBeats, 65536.0);
	W(C_BC, bc);
	if (beatSource == 3 && clocked) u_kick = max(u_kick, exp(-beatPhase * 60.0 / max(beatBpm, 30.0) / KICK_TAU));
	float kf = 1.0 - exp(-dt / FEAT_TAU);
	float loud = C(C_LOUD) + (u_rms - C(C_LOUD)) * kf;
	float lmax = max(max(loud, C(C_LMAX) - dt * 0.01), 0.02);
	W(C_LOUD, loud); W(C_LMAX, lmax);
	float L = loudN(loud, lmax);
	float sum = u_sub + u_bass + u_lowmid + u_mid + u_highmid + u_presence + u_brilliance + 1e-4;
	W(C_MIDS, C(C_MIDS) + ((u_lowmid + u_mid + u_highmid) / sum - C(C_MIDS)) * kf);
	W(C_BASS, C(C_BASS) + ((u_sub + u_bass) / sum - C(C_BASS)) * kf);
	W(C_HI, C(C_HI) + (max(u_presence, u_brilliance) - C(C_HI)) * (1.0 - exp(-dt / HI_TAU)));
	bool kickEdge = u_kick > 0.6 && C(C_KPREV) <= 0.6;
	W(C_KPREV, u_kick);
	float groove = C(C_GROOVE) + ((kickEdge ? 1.0 / max(dBeats, 1e-3) : 0.0) - C(C_GROOVE)) * kf;
	W(C_GROOVE, clamp(groove, 0.0, 4.0));
	W(C_PATH, mod(C(C_PATH) + dBeats * 6.2831853 / PATH_BEATS * (0.5 + L), 6.2831853));
	bool secChanged = u_ms_change != C(C_FIGSEEN);
	if (secChanged) { W(C_FIGSEEN, u_ms_change); W(C_EVT, bc); W(C_SALT, fract(C(C_SALT) * 7.31 + 0.618 + fract(u_ms_change * 0.0137))); }
	float salt = C(C_SALT);
#ifdef DESKTOP
	int mv = int(u_move + 0.5);
#else
	int mv = 0;
#endif
	bool lab = mv > 0;

	float edge = resolution.x / max(resolution.y, 1.0);
	float tapQ = C(C_TAPQ);
	bool tapNow = false;
	float hold = C(C_HOLD);
	vec2 hv = vec2(C(C_HVX), C(C_HVY));
	vec2 fling = vec2(C(C_FLX), C(C_FLY)) * exp(-dt / FLING_TAU);
	float knockW = C(C_KW) * exp(-dt / KNOCK_TAU);
	float throwSide = 0.0;
	bool swipeDash = false;
	int pc = screenLocked ? 0 : pointerCount, pcPrev = int(C(C_PCPREV) + 0.5);
	float pcMax = C(C_PCMAX);
	float knock2W = C(C_K2W) * exp(-dt / KNOCK_TAU);
	if (pc >= 2 && pc > pcPrev) {
		vec3 g2 = glassPt(pointers[min(pc, 10) - 1].xy);
		W(C_K2X, g2.x); W(C_K2Y, g2.y); W(C_K2Z, g2.z); knock2W = 1.0;
	}
	W(C_PCPREV, float(pc));
	if (pc > 0) W(C_TLAST, time);
	if (pc > 0) {
		if (C(C_TDOWN) < 0.5) { W(C_TDOWN, 1.0); W(C_TX0, touch.x); W(C_TY0, touch.y); W(C_TXP, touch.x); W(C_TYP, touch.y); W(C_TSWIPE, 0.0); W(C_TT0, time); hold = 0.0; hv = vec2(0.0); pcMax = 0.0; W(C_PENTUSED, 0.0);
			vec3 g = glassPt(touch);
			W(C_KX, g.x); W(C_KY, g.y); W(C_KZ, g.z); knockW = 1.0;
		} else {
			vec2 gNow = glassPt(touch).xy, gPrev = glassPt(vec2(C(C_TXP), C(C_TYP))).xy;
			hv = mix(hv, (gNow - gPrev) / dt, FLING_SMOOTH);
			W(C_TXP, touch.x); W(C_TYP, touch.y);
			if (C(C_TSWIPE) > 0.5 && hold < 0.5 && pcMax < 1.5 && length(hv) > SWIPE_DASH_V && time - C(C_SWDT) >= SWIPE_DASH_GAP) {
				swipeDash = true; W(C_SWDT, time);
				if (abs(hv.x) > abs(hv.y) && C(C_SLWP) != C(C_TT0)) {
					throwSide = sign(hv.x); tapNow = true; W(C_SLWP, C(C_TT0));
				}
			}
		}
		pcMax = max(pcMax, float(pc));
		if (pc >= 3 && (time - C(C_TT0) >= HOLD_S)) W(C_PENTUSED, 1.0);
		if (pcMax >= 1.5) { hold = 0.0; W(C_TSWIPE, 0.0); }
		else {
			if (hold < 0.5 && length(touch - vec2(C(C_TX0), C(C_TY0))) > SWIPE_PX) W(C_TSWIPE, 1.0);
			if (C(C_TSWIPE) < 0.5 && time - C(C_TT0) >= HOLD_S) hold = 1.0;
		}
	} else if (C(C_TDOWN) > 0.5) {
		W(C_TDOWN, 0.0);
		if (pcMax >= 2.5) {
			if (time - C(C_TT0) < TAP3_MAX_S && C(C_PENTUSED) < 0.5) { tapQ = 3.0; tapNow = true; }
		} else if (pcMax >= 1.5) {
		} else if (hold > 0.5) {
			hold = 0.0;
			float hs = length(hv) * FLING_K;
			if (hs > FLING_MIN) {
				fling = hv * FLING_K * (hs > FLING_MAX ? FLING_MAX / hs : 1.0);
				if (abs(fling.x) > FLING_WP && abs(fling.x) > abs(fling.y)) throwSide = sign(fling.x);
				W(C_EVT, bc);
			}
		} else if (C(C_TSWIPE) > 0.5) {
			vec2 v = vec2(C(C_TXP) - C(C_TX0), C(C_TYP) - C(C_TY0));
			vec2 r = 0.5 * (vec2(C(C_TX0), C(C_TY0)) + vec2(C(C_TXP), C(C_TYP))) - 0.5 * resolution;
			float cr = r.x * v.y - r.y * v.x;
			if (abs(cr) < 1e-3 * resolution.y * resolution.y) cr = -v.x;
			W(C_ODIRT, cr > 0.0 ? 1.0 : -1.0);
			W(C_EVT, bc);
		} else if (time - C(C_TT0) < TAP_MAX_S) {
			if (length(vec2(C(C_TX0), C(C_TY0)) - 0.5 * resolution) < MID_R_FRAC * resolution.x) W(C_FARREQ, 1.0);
			else { tapQ = max(tapQ, 2.0); tapNow = true; }
		}
	}
	W(C_HOLD, hold); W(C_HVX, hv.x); W(C_HVY, hv.y); W(C_FLX, fling.x); W(C_FLY, fling.y); W(C_KW, knockW);
	W(C_PCMAX, pcMax); W(C_K2W, knock2W);
	float pinW = C(C_PINW);
	{
		if (pc >= 2) {
			vec3 g0 = glassPt(pointers[0].xy), g1 = glassPt(pointers[1].xy);
			W(C_PINMX, 0.5 * (g0.x + g1.x)); W(C_PINMY, 0.5 * (g0.y + g1.y)); W(C_PINR, 0.5 * length(g0.xy - g1.xy));
			vec2 ax = g1.xy - g0.xy;
			if (dot(ax, ax) > 1e-6) { ax = normalize(ax); W(C_PINUX, ax.x); W(C_PINUY, ax.y); }
			else if (abs(C(C_PINUX)) + abs(C(C_PINUY)) < 0.5) { W(C_PINUX, 1.0); W(C_PINUY, 0.0); }
			pinW += (1.0 - pinW) * (1.0 - exp(-dt / PINCH_TAU_IN));
			if (pc >= 3) { vec3 g2 = glassPt(pointers[2].xy); W(C_PENTR, length(g2.xy - 0.5 * (g0.xy + g1.xy))); }
		} else pinW *= exp(-dt / PINCH_TAU_OUT);
		W(C_PINW, pinW);
		float pnW = C(C_PENTW);
		pnW += (((pc >= 3 && C(C_PENTUSED) > 0.5) ? 1.0 : 0.0) - pnW) * (1.0 - exp(-dt / (pc >= 3 ? PINCH_TAU_IN : PINCH_TAU_OUT)));
		W(C_PENTW, pnW);
	}
	float odir = C(C_ODIR);
	odir += (C(C_ODIRT) - odir) * (1.0 - exp(-dt / ODIR_TAU));
	W(C_ODIR, odir);
	float dash = max(C(C_DASH) - dBeats, 0.0);
	if (kickEdge || swipeDash) dash = DASH_BEATS;
	W(C_DASH, dash);
	float dashAmt = smoothstep(0.0, DASH_BEATS, dash);
	float oph = mod(C(C_OPH) + odir * dBeats * 6.2831853 / ORBIT_BEATS * (1.0 + DASH_SPD * dashAmt), 6.2831853);
	W(C_OPH, oph);
	float eNow = u_ms_cur_a.x > 0.0 ? u_ms_cur_a.x : L;
	float hEn = C(C_HEN);
	hEn += (smoothstep(HUNT_E0, HUNT_E1, eNow) - hEn) * (1.0 - exp(-dBeats / HUNT_E_BEATS));
	W(C_HEN, hEn);
	float hSpd = mix(1.0, ORBIT_SPEED_H, hEn);
	W(C_HSPD, hSpd);
	float bcH = mod(C(C_BCH) + dBeats * hSpd, 65536.0);
	W(C_BCH, bcH);
	float ophH = mod(C(C_OPHH) + odir * dBeats * 6.2831853 / ORBIT_BEATS * hSpd * (1.0 + DASH_SPD * dashAmt), 6.2831853);
	W(C_OPHH, ophH);
	W(C_PINPH, mod(C(C_PINPH) + odir * dBeats * 6.2831853 / PINCH_ORBIT_BEATS, 6.2831853));

	vec2 w2s = vec2(CAM_D / FOCAL);
	vec2 tgt = vec2(0.0);
	vec2 cenLim = vec2(boxXW(edge), boxYW()) - CEN_MARGIN;
	if (screenLocked) tgt = SENSOR_UV * w2s;
	else if (hold > 0.5) tgt = clamp(glassPt(touch).xy, -cenLim, cenLim);
	else if (pc >= 2) tgt = clamp(vec2(C(C_PINMX), C(C_PINMY)), -cenLim, cenLim);
	else if (length(gravity) > 0.5) {
		vec2 g = -gravity.xy / 9.81;
		tgt = clamp(g * GRAV_K, -vec2(boxXW(edge), boxYW()) + CEN_MARGIN, vec2(boxXW(edge), boxYW()) - CEN_MARGIN);
	}
	vec2 cen = vec2(C(C_CX), C(C_CY)), cv = vec2(C(C_CVX), C(C_CVY));
	{
		bool rush = hold > 0.5 || pc >= 2;
		float om = 6.2831853 / (rush ? HOLD_PERIOD : CEN_PERIOD);
		cv += (om * om * (tgt - cen) - 2.0 * CEN_DAMP * om * cv) * dt;
		float cs = length(cv), cmax = rush ? HOLD_VMAX : CEN_VMAX;
		if (cs > cmax) cv *= cmax / cs;
		cen += (cv + fling) * dt;
		cen = clamp(cen, -cenLim, cenLim);
		W(C_CX, cen.x); W(C_CY, cen.y); W(C_CVX, cv.x); W(C_CVY, cv.y);
	}
	float yRoom = max(Y_MIN, Y_USE * w2s.y - abs(cen.y));
	W(C_YROOM, yRoom);
	float scat = C(C_SCAT) * exp(-dBeats / SCATTER_BEATS);
	{
		float base = C(C_BT_BASE);
		base += (linear.z - base) * (1.0 - exp(-dt / BT_TAU));
		W(C_BT_BASE, base);
		float spike = abs(linear.z - base);
		if (spike > BT_THR && length(linear.xy) < BT_XY && length(gyroscope) < BT_GYRO && pointerCount == 0 && time > C(C_BT_REFR)
		    && time - C(C_TLAST) > BT_TOUCH_S) {
			scat = 1.0;
			W(C_BT_REFR, time + BT_REFR);
			W(C_BT_N, C(C_BT_N) + 1.0);
		}
	}

	float state = C(C_MSTATE), seen = C(C_MSEEN), target = C(C_MTGT), dblT = C(C_SPL2T);
	if (u_ms_change != seen) {
		seen = u_ms_change; target = pickMode(u_ms_sec_a, u_ms_sec_b); dblT = isDouble(u_ms_sec_a, u_ms_sec_b);
		W(C_SPLITAX, hashW(u_ms_change, 5.0) < 0.5 ? 0.0 : 1.0);
	}
	if (u_ms_source > 0.5 && u_ms_next_in > 0.0 && u_ms_next_in < PUSH_EARLY && u_ms_next_a.x >= 0.0) {
		target = pickMode(u_ms_next_a, u_ms_next_b); dblT = isDouble(u_ms_next_a, u_ms_next_b);
	}
	if (lab) {
		target = (mv >= 25 && mv <= 28) ? float(mv - 25) : ((mv == 17 || mv == 18) ? 4.0 : ((mv == 23 || mv == 24) ? 3.0 : 0.0));
		dblT = mv == 18 ? 1.0 : 0.0;
	}
	{
		float d = target - state;
		d -= N_MODES * floor(d / N_MODES + 0.5);
		if (abs(d) > 0.05) state += sign(d) * min(PUSH_RATE * dBeats, abs(d));
		else state += (-sin(state * 6.2831853) * WELL_G / 6.2831853) * dBeats;
		state += KICK_WOBBLE * u_kick * u_kick * u_kick * sin(oph * 2.0) * dBeats;
		state = mod(state, N_MODES);
		W(C_MSTATE, state); W(C_MSEEN, seen); W(C_MTGT, target);
	}
	float w[5];
	pocketWeights(state, w);
	float wsum = max(w[0] + w[1] + w[2] + w[3] + w[4], 1e-4);
	int pocketDom = 0;
	{ float bw = w[0]; for (int i = 1; i < 5; i++) if (w[i] > bw) { bw = w[i]; pocketDom = i; } }
	float hunterW = clamp(HUNTER_W(w) / wsum, 0.0, 1.0);
	float pocket = C(C_POCKET) + (POCKET_MAX * u_ms_cur_a.z - C(C_POCKET)) * (1.0 - exp(-dt / POCKET_TAU));
	W(C_POCKET, pocket);
	float hat = C(C_HI);
	float windup = (u_ms_source > 0.5 && u_ms_drop_in > 0.0) ? 1.0 - smoothstep(0.0, WINDUP_BEATS * 60.0 / max(beatBpm, 30.0), u_ms_drop_in) : 0.0;
	W(C_SPLITW, w[4]);
	{
		float dW = C(C_SPL2W);
		dW += (dblT * w[4] - dW) * (1.0 - exp(-dBeats / SPLIT2_EASE));
		W(C_SPL2W, clamp(dW, 0.0, 1.0)); W(C_SPL2T, dblT);
	}
	float push = C(C_PUSH);
	push += (u_kick - push) * (1.0 - exp(-dt / (u_kick > push ? 0.08 : 0.35)));
	W(C_PUSH, push);
	float t = bc * L_BIN_T;
	float sep0 = L_SEP0 + L_SEP1 * pow(sin(t * 0.5), 4.0) + L_KICK_SEP * push;
	float press = C(C_PRESS);
	press += (1.0 - smoothstep(SPONGE_NEAR, SPONGE_FAR, sep0) - press) * (1.0 - exp(-dBeats / SPONGE_BEATS));
	W(C_PRESS, press);
	vec3 sepX = vec3(push, press, windup);
	float farW = C(C_FARW), farNext = C(C_FARNEXT), farEnd = C(C_FAREND);
	if (!(farNext == farNext) || farNext > bc + 64.0 || farNext <= 0.0) farNext = bc + FAR_GAP0 + FAR_GAPJ * hashW(bc, 61.0 + salt);
	if (!(farEnd == farEnd) || farEnd > bc + 64.0) farEnd = 0.0;
	bool swarmDom = pocketDom <= 1;
	if (C(C_FARREQ) > 0.5) {
		W(C_FARREQ, 0.0); farEnd = bc + FAR_TAP_BEATS; W(C_FARFORCE, 1.0); W(C_EVT, bc);
		W(C_FIGID, 0.0); W(C_FIGPH, 0.0); W(C_FIGAMP, 0.0);
		W(C_EYE_END, min(C(C_EYE_END), bc));
		W(C_WP_START, -1e4); W(C_TLT_END, 0.0); W(C_CLBS, -1e4);
		scat = 1.0;
		W(C_SALT, fract(C(C_SALT) * 7.31 + 0.37)); W(C_SHSEEN, -12345.0);
	}
	if (lab) {
		if (mv == 16) { farEnd = bc + 2.0; W(C_FARFORCE, 1.0); } else { farEnd = 0.0; W(C_FARFORCE, 0.0); }
		farNext = bc + FAR_GAP0;
	}
	if (!lab && C(C_FARFORCE) > 0.5 && bc >= farEnd) {
		W(C_FARFORCE, 0.0);
		tapQ = max(tapQ, 2.0);
		W(C_SHSEEN, -12345.0);
	}
	if (bc >= farNext && bc >= farEnd) {
		bool freeNow = C(C_FIGID) < 0.5 && bc >= C(C_EYE_END) && C(C_PINW) < 0.01;
		if (swarmDom && freeNow) { farEnd = bc + FAR_BEATS * (0.75 + 0.5 * hashW(bc, 67.0 + salt)); W(C_EVT, bc); }
		farNext = bc + FAR_GAP0 + FAR_GAPJ * hashW(bc, 71.0 + salt);
	}
	float farForce = C(C_FARFORCE);
	farW += (((bc < farEnd && (swarmDom || farForce > 0.5)) ? 1.0 : 0.0) - farW) * (1.0 - exp(-dBeats / (farForce > 0.5 ? FAR_TAP_EASE : FAR_EASE)));
	W(C_FFW, farForce > 0.5 ? farW : C(C_FFW) * exp(-dBeats / FAR_EASE));
	W(C_FARW, farW); W(C_FARNEXT, farNext); W(C_FAREND, farEnd);
	float wE[5];
	for (int i = 0; i < 5; i++) wE[i] = w[i] * (1.0 - farW);
	wE[4] += farW;
	float splitAxE = (C(C_SPLITAX) * w[4] * (1.0 - farW) + farW) / max(w[4] * (1.0 - farW) + farW, 1e-4);
	W(C_SPLITAXE, splitAxE);
	float farSh = farW / max(w[4] * (1.0 - farW) + farW, 1e-4);
	W(C_FARSH, C(C_FARFORCE) > 0.5 ? 0.0 : farSh);
	if (C(C_FARFORCE) > 0.5) farSh = 0.0;
	W(C_HUNTW, clamp(HUNTER_W(w) / max(w[0] + w[1] + w[2] + w[3] + w[4], 1e-4), 0.0, 1.0));
	for (int i = 0; i < 5; i++) W(C_HW0 + i, wE[i]);
	float tumRate, tum;
	{
		float hwE = clamp(HUNTER_W(wE) / max(wE[0] + wE[1] + wE[2] + wE[3] + wE[4], 1e-4), 0.0, 1.0);
		tumRate = L_BIN_T * 1.2 * (1.0 - hwE);
		tum = mod(C(C_TUM) + dBeats * tumRate, 31.4159265);
		W(C_TUM, tum); W(C_TUMR, tumRate);
	}
	W(C_HPOCK, pocket); W(C_HSEP, sepX.x); W(C_HSEP + 1, sepX.y); W(C_HSEP + 2, sepX.z);
	float splitAx = C(C_SPLITAX);

	float figId = C(C_FIGID), figPh = C(C_FIGPH), figCool = C(C_FIGCOOL), figN = C(C_FIGN);
	float eyeT = C(C_EYE_T), eyeEnd = C(C_EYE_END), eyeW = C(C_EYE_W), eyeLast = C(C_EYE_LAST);
	float wpS = C(C_WP_START), wpL = C(C_WP_LAST);
	if (!(figCool == figCool) || figCool > bc + 64.0) figCool = 0.0;
	if (!(eyeEnd == eyeEnd) || eyeEnd > bc + 64.0) { eyeEnd = 0.0; eyeT = 0.0; }
	if (!(eyeLast == eyeLast) || eyeLast > bc + 1.0) eyeLast = -1e4;
	if (!(wpS == wpS) || wpS > bc + 4.0) wpS = -1e4;
	if (!(wpL == wpL) || wpL > bc + 4.0) wpL = -1e4;
	bool wpBusy = bc <= wpS + WP_OUT_BEATS + WP_BACK_BEATS;
	if (tapNow) { figId = 0.0; eyeEnd = min(eyeEnd, bc); wpS = -1e4; wpBusy = false; }
	bool busy = figId > 0.5 || bc < eyeEnd || wpBusy;
	{
		float lap = floor(bc / ORBIT_BEATS), lastLap = C(C_BORED_LAP);
		if (!(lastLap == lastLap) || lastLap > lap) lastLap = lap;
		if (lap != lastLap) {
			W(C_BORED_LAP, lap);
			float evt = C(C_EVT);
			float n = (evt >= (lap - 1.0) * ORBIT_BEATS) ? 0.0 : C(C_BORED_N) + 1.0;
			if (n >= BORED_LAPS) { n = 0.0; tapQ = max(tapQ, 2.0); }
			W(C_BORED_N, n);
		}
	}
	{
		bool secPunch = secChanged && (u_ms_sec_b.w >= 0.28 || (u_ms_sec_a.x >= 0.68 && u_ms_sec_a.y >= 0.48));
		bool heavySec = u_ms_sec_a.x >= SPECIAL_E && (u_ms_sec_a.y >= SPECIAL_W || u_ms_sec_b.w >= SPECIAL_I);
		float bar = floor(bc / WP_PHRASE);
		float phrQ = heavySec ? C(C_PHRQ) : 0.0;
		if (heavySec && bar != C(C_FIGBAR)) phrQ = 1.0;
		bool phrasePeak = phrQ > 0.5;
		W(C_FIGBAR, bar);
		bool livePeak = (u_ms_cur_a.x * max(u_ms_cur_b.w, 0.0) + 0.35 * push) > 0.68;
		bool musicTrig = secPunch || phrasePeak || livePeak;
		bool gateOpen = !busy && (bc >= figCool || tapQ > 1.5) && C(C_FARFORCE) < 0.5;
		int labPick = labFig(mv);
		bool labFire = lab && labPick > 0 && !busy;
		if (lab) { gateOpen = labFire; musicTrig = labFire; }
		if (gateOpen && (tapQ > 0.5 || musicTrig)) {
			figN += 1.0;
			bool wpOk = bc - wpL >= WP_COOL_BEATS;
			bool eyeOk = bc - eyeLast >= EYE_COOL_BEATS;
			int pick = labFire ? labPick
			         : (tapQ > 2.5) ? (C(C_EYE_LT) < 1.5 ? FG_EYEARM : FG_EYERINGS)
			                        : pickFigureAll(msNeutral(u_ms_sec_a), msNeutral(u_ms_sec_b), C(C_FIGLAST), C(C_FIGH1), C(C_FIGH2),
			                                        figN + 97.0 * salt, wpOk, eyeOk, C(C_EYE_LT));
			{
				float pid = float(pick == FG_EYEARM ? FG_EYERINGS : pick);
				W(C_FIGH2, C(C_FIGH1)); W(C_FIGH1, C(C_FIGLAST)); W(C_FIGLAST, pid);
			}
			phrQ = 0.0;
			if (pick == FG_WALLPUNCH) {
				wpS = floor(bc) + 1.0; wpL = wpS;
				W(C_WP_SIDE, FLK_LOAD(flk, 6).x - cen.x >= 0.0 ? 1.0 : -1.0);
				W(C_WP_SAME, 0.0);
			} else if (pick == FG_EYERINGS || pick == FG_EYEARM) {
				eyeT = pick == FG_EYERINGS ? 1.0 : 2.0; eyeEnd = bc + EYE_BEATS; eyeLast = bc;
				W(C_EYE_LT, eyeT);
			} else if (pick > 0) {
				figId = float(pick); figPh = 0.0;
				float vs = figN * 1.618 + salt * 31.0;
				W(C_FIGMX, hashW(vs, 3.0) < 0.5 ? -1.0 : 1.0);
				W(C_FIGMY, hashW(vs, 5.0) < 0.5 ? -1.0 : 1.0);
				bool oneWay = pick == FG_HELIX || pick == FG_LASH || pick == FG_PUNCHIN;
				W(C_FIGREV, (!oneWay && hashW(vs, 7.0) < VAR_REV_P) ? 1.0 : 0.0);
				W(C_FIGSC, mix(VAR_SC0, VAR_SC1, hashW(vs, 11.0)));
				W(C_FIGOX, (hashW(vs, 13.0) * 2.0 - 1.0) * VAR_OFF_XY);
				W(C_FIGOX + 1, (hashW(vs, 17.0) * 2.0 - 1.0) * VAR_OFF_XY);
				W(C_FIGOX + 2, mix(VAR_OFF_Z0, VAR_OFF_Z1, hashW(vs, 19.0)));
				float tv = hashW(vs, 23.0);
				W(C_FIGLEN, figBeats(pick) * (tv < 0.25 ? 0.75 : (tv < 0.75 ? 1.0 : 1.5)));
			}
			figCool = bc + FIG_COOL_BEATS;
			tapQ = 0.0;
			W(C_EVT, bc);
		}
		W(C_PHRQ, phrQ);
	}
	if (throwSide != 0.0 && !wpBusy) {
		wpS = floor(bc) + 1.0; wpL = wpS;
		W(C_WP_SIDE, throwSide); W(C_WP_SAME, 1.0);
		figCool = max(figCool, bc + FIG_COOL_BEATS);
		W(C_EVT, bc);
	}
	float figAmp = 0.0;
	if (figId > 0.5) {
		float fLen = C(C_FIGLEN);
		if (!(fLen > 0.5) || fLen > 16.0) fLen = figBeats(int(figId + 0.5));
		figPh += dBeats / fLen;
		if (figPh >= 1.0) { figId = 0.0; figPh = 0.0; }
		else {
			float e = FIG_EASE / fLen;
			figAmp = min(smoothstep(0.0, e, figPh), smoothstep(1.0, 1.0 - e, figPh));
		}
	}
	W(C_FIGID, figId); W(C_FIGPH, figPh); W(C_FIGAMP, figAmp); W(C_FIGCOOL, figCool); W(C_FIGN, figN); W(C_TAPQ, tapQ);
	eyeW += ((bc < eyeEnd ? 1.0 : 0.0) - eyeW) * (1.0 - exp(-dBeats / EYE_EASE));
	if (bc >= eyeEnd && eyeW < 0.01) eyeT = 0.0;
	W(C_EYE_T, eyeT); W(C_EYE_END, eyeEnd); W(C_EYE_W, clamp(eyeW, 0.0, 1.0)); W(C_EYE_LAST, eyeLast);
	float wpAmt = 0.0;
	{
		float tw = bc - wpS;
		if (tw >= 0.0 && tw < WP_OUT_BEATS) wpAmt = tw / WP_OUT_BEATS;
		else if (tw >= WP_OUT_BEATS && tw < WP_OUT_BEATS + WP_BACK_BEATS) wpAmt = 1.0 - (tw - WP_OUT_BEATS) / WP_BACK_BEATS;
		W(C_WP_START, wpS); W(C_WP_LAST, wpL); W(C_WP_AMT, wpAmt);
	}

	vec3 o = pocketOrbit(oph, ophH, bc, bcH, wE, pocket, sepX, yRoom, L, splitAxE, farSh, tum);
	if (pinW > 0.001) {
		float R = C(C_PINR) * CAM_D / (CAM_D + BOX_ZN);
		float room = min(min(boxXW(edge) - abs(cen.x), boxYW() - abs(cen.y)), min(BOX_ZB, -BOX_ZN)) - GOAL_MARGIN;
		R = clamp(R, PINCH_R_MIN, max(room, PINCH_R_MIN));
		W(C_PINRW, R);
		float pRoom = min(boxXW(edge) - abs(cen.x), boxYW() - abs(cen.y));
		if (PENT_SPHERE > 0) pRoom = min(pRoom, min(BOX_ZB, -BOX_ZN));
		float Rp = clamp(C(C_PENTR) * CAM_D / (CAM_D + BOX_ZN), PINCH_R_MIN, max(pRoom - GOAL_MARGIN, PINCH_R_MIN));
		W(C_PENTRW, Rp);
		float rimL = mix(6.2831853 * R, 9.5105652 * Rp, clamp(C(C_PENTW), 0.0, 1.0));
		W(C_PINV, rimL * (dBeats / dt) / PINCH_ORBIT_BEATS);
		float pph = C(C_PINPH);
		vec3 ua = vec3(C(C_PINUX), C(C_PINUY), 0.0);
		o = mix(o, (cos(pph) * ua + vec3(0.0, 0.0, sin(pph))) * R, pinW);
	}
	{
		float pir = 6.2831853 * bc / PIR_BEATS;
		vec3 po = vec3(cos(pir), sin(pir), 0.0) * PIR_R * clamp(0.6 * hat + 0.4 * u_ms_cur_a.w, 0.0, 1.0);
		float tum = t * 1.2;
		po.xy = mat2(cos(tum), -sin(tum), sin(tum), cos(tum)) * po.xy;
		o += po;
	}
	{
		float pp = TILT_POCKET_BEATS;
		float pIdx = floor(bc / pp);
		float tA = C(C_TLT_A), tT = C(C_TLT_T), tEnd = C(C_TLT_END), tTh = C(C_TLT_TH);
		if (!(tEnd == tEnd) || tEnd > bc + 64.0) { tEnd = 0.0; tT = 0.0; }
		float w3 = clamp(w[3] / wsum, 0.0, 1.0);
		if (pIdx != C(C_TLT_LP)) {
			W(C_TLT_LP, pIdx);
			if (bc >= tEnd && abs(tA) < 0.05 && w3 > 0.5) {
				vec3 oN = pocketOrbit(oph + odir * 0.125 * 6.2831853 / ORBIT_BEATS * 4.0, ophH + odir * 0.5 * 6.2831853 / ORBIT_BEATS * hSpd, bc + 0.5, bcH + 0.5 * hSpd, wE, pocket, sepX, yRoom, L, splitAxE, farSh, tum + 0.5 * tumRate);
				float thT = atan(oN.y - o.y, oN.x - o.x + 1e-6);
				float r1 = hash11(pIdx * 5.31 + 0.7 + 17.0 * salt), r2 = hash11(pIdx * 9.17 + 0.3 + 23.0 * salt);
				float gE = smoothstep(TILT_E0, TILT_E1, u_ms_cur_a.x);
				float sg = r2 < 0.5 ? -1.0 : 1.0;
				if (lab) r1 = mv == 23 ? -1.0 : (mv == 24 ? TILT_P_DIVE * gE + 0.5 * TILT_P_BANK : 9.0);
				if (r1 < TILT_P_DIVE * gE) { tTh = thT + 1.5707963; tT = sg * TILT_DIVE_A; tEnd = bc + pp; W(C_EVT, bc); }
				else if (r1 < TILT_P_DIVE * gE + TILT_P_BANK) { tTh = thT; tT = sg * TILT_BANK_A; tEnd = bc + pp; W(C_EVT, bc); }
			}
		}
		float want = bc < tEnd ? tT : 0.0;
		tA += (want - tA) * (1.0 - exp(-dBeats / (abs(want) > 1e-4 ? TILT_IN_BEATS : TILT_OUT_BEATS)));
		W(C_TLT_A, tA); W(C_TLT_T, tT); W(C_TLT_END, tEnd); W(C_TLT_TH, tTh);
	}
	float clbS = C(C_CLBS);
	if (!(clbS == clbS) || clbS > bc + 1.0) clbS = -1e4;
	{
		float blk = floor(bc / CLIMB_EVERY);
		if (blk != C(C_CLBBLK)) {
			W(C_CLBBLK, blk);
			bool freeC = bc > clbS + CLIMB_BEATS + CLIMB_COOL && !busy && pinW < 0.01 && hold < 0.5 && !screenLocked && farW < 0.05;
			if (lab) freeC = mv == 15 && bc > clbS + CLIMB_BEATS;
			if (freeC && (lab || hashW(blk, 41.0 + salt) < CLIMB_P)) {
				clbS = bc;
				W(C_CLBF, 2.0);
				W(C_CLBWALL, min(floor(hashW(blk, 47.0 + salt) * 4.0), 3.0));
				W(C_CLBH, CLIMB_H0 + CLIMB_HJ * hashW(blk, 53.0 + salt));
				W(C_EVT, bc);
			}
		}
		W(C_CLBS, clbS);
	}
	float clbW = 0.0, clbZ = BOX_ZB - GOAL_MARGIN;
	{
		float tc = (bc - clbS) / CLIMB_BEATS;
		if (tc >= 0.0 && tc < 1.0) {
			float zLo = BOX_ZB - GOAL_MARGIN, zTop = mix(zLo, BOX_ZN + GOAL_MARGIN, C(C_CLBH));
			clbW = smoothstep(0.0, 0.18, tc) * (1.0 - smoothstep(0.88, 1.0, tc));
			if (tc < 0.20) clbZ = zLo;
			else if (tc < 0.62) { float u = (tc - 0.20) / 0.42; clbZ = mix(zLo, zTop, 1.0 - (1.0 - u) * (1.0 - u)); }
			else if (tc < 0.74) clbZ = zTop + 0.04 * sin(6.2831853 * 3.0 * (tc - 0.62) / 0.12);
			else if (tc < 0.88) { float u = (tc - 0.74) / 0.14; clbZ = mix(zTop, zLo, u * u); }
			clbW *= 1.0 - pinW;
		}
	}
	vec3 c3 = vec3(cen, 0.0);
	uint seed = pcg(uint(bc * 977.0) + 12345u);
	for (int f = 0; f < 2; f++) {
		vec4 now = FLK_LOAD(flk, 6 + f);
		vec3 q = f == 0 ? o : -o;
		{
			float ta = C(C_TLT_A) * clamp(w[3] / wsum, 0.0, 1.0);
			if (abs(ta) > 1e-4) {
				float th = C(C_TLT_TH);
				vec3 ua = vec3(cos(th), sin(th), 0.0);
				q = rotAx(vec3(q.xy, 0.0), ua, ta) + vec3(0.0, 0.0, q.z);
			}
		}
		vec3 gLo = vec3(-boxXW(edge), -boxYW(), BOX_ZN) + GOAL_MARGIN, gHi = vec3(boxXW(edge), boxYW(), BOX_ZB) - GOAL_MARGIN;
		if (figAmp > 0.0) {
			int fid = int(figId + 0.5);
			vec3 reach = vec3(0.0);
			for (int n = 0; n < FIG_FIT_N; n++) reach = max(reach, abs(figSlot(fid, (float(n) + 0.5) / float(FIG_FIT_N), float(f))));
			vec3 vOff = vec3(C(C_FIGOX), C(C_FIGOX + 1), C(C_FIGOX + 2)) * figAmp;
			q += vOff * (1.0 - pinW);
			vec3 at = c3 + q;
			vec3 room = max(min(gHi - at, at - gLo), vec3(0.0));
			vec3 fitv = room / max(reach * FIG_R, vec3(1e-4));
			float fit = min(1.0, min(min(fitv.x, fitv.y), fitv.z));
			float vPh = C(C_FIGREV) > 0.5 ? 1.0 - figPh : figPh;
			vec3 vMir = vec3(C(C_FIGMX), C(C_FIGMY), 1.0);
			float vSc = clamp(C(C_FIGSC), VAR_SC0, VAR_SC1);
			fit = min(vSc, min(min(fitv.x, fitv.y), fitv.z));
			q += figSlot(fid, vPh, float(f)) * vMir * FIG_R * fit * figAmp * (1.0 - pinW);
		}
		if (wpAmt > 0.0) {
			float side = C(C_WP_SIDE) * (C(C_WP_SAME) > 0.5 || f == 0 ? 1.0 : -1.0);
			q.x = mix(q.x, side * WP_OVER * boxXW(edge) - cen.x, wpAmt * (1.0 - pinW));
		}
		if (clbW > 0.0) {
			float cf = C(C_CLBF);
			if (cf > 1.5 || abs(cf - float(f)) < 0.5) {
				int wl = int(C(C_CLBWALL) + 0.5);
				if (cf > 1.5 && f == 1) wl = wl ^ 1;
				vec3 tg = c3 + q;
				if (wl < 2) tg.x = (wl == 0 ? 1.0 : -1.0) * (boxXW(edge) - GOAL_MARGIN);
				else        tg.y = (wl == 2 ? 1.0 : -1.0) * (boxYW() - GOAL_MARGIN);
				tg.z = clbZ;
				q = mix(q, tg - c3, clbW);
			}
		}
		vec3 p = clamp(c3 + q, gLo, gHi);
		FLK_STORE(flk, 8 + f, now.w > 0.5 ? now : vec4(p, 1.0));
		FLK_STORE(flk, 6 + f, vec4(p, 1.0));
		vec4 hk = FLK_LOAD(flk, 4 + f);
		hk.w *= exp(-dt / HAWK_TAU);
		if (kickEdge) {
			vec4 fc = FLK_LOAD(flk, f);
			hk = vec4(fc.xyz + rndDir(seed) * (fc.w * HAWK_EDGE + 0.04), clamp(u_kick, 0.6, 1.0));
		}
		FLK_STORE(flk, 4 + f, hk);
	}
	FLK_STORE(flk, 12, vec4(c3, 1.0));

	{
		float dropIn = u_ms_drop_in, lastDrop = C(C_SCDROP);
		if (lastDrop > 0.0 && lastDrop < 1.0 && (dropIn < 0.0 || dropIn > lastDrop + 1.0)) scat = 1.0;
		W(C_SCDROP, dropIn);
		if (u_ms_change != C(C_SCSEEN)) {
			if (u_ms_cur_b.w > SCATTER_IMPACT || u_ms_sec_b.w > SCATTER_IMPACT) scat = 1.0;
			W(C_SCSEEN, u_ms_change);
		}
		W(C_SCAT, scat);
		float acc = C(C_ACCENT) * exp(-dBeats / ACCENT_BEATS);
		if (kickEdge) {
			float heavy = u_ms_cur_a.y >= 0.0 ? u_ms_cur_a.y : clamp(2.0 * C(C_BASS), 0.0, 1.0);
			acc = max(acc, smoothstep(0.6, 1.0, u_kick) * smoothstep(ACCENT_W0, ACCENT_W1, heavy));
		}
		W(C_ACCENT, acc);
		float fzT = clocked ? smoothstep(FRENZY_BPM0, FRENZY_BPM1, beatBpm) * smoothstep(FRENZY_E0, FRENZY_E1, u_ms_cur_a.x)
		                      * smoothstep(FRENZY_D0, FRENZY_D1, u_ms_cur_a.z) : 0.0;
		fzT *= clamp((w[3] + w[4]) / wsum, 0.0, 1.0);
		float fz = C(C_FRENZY) + (fzT - C(C_FRENZY)) * (1.0 - exp(-dBeats / FRENZY_BEATS));
		W(C_FRENZY, fz);
		vec4 m0 = FLK_LOAD(flk, 10), m1 = FLK_LOAD(flk, 11);
		float Rt = min(m0.x, m1.x), spd = 0.5 * (m0.y + m1.y);
		float shA = C(C_SHA), shB = C(C_SHB), shT = C(C_SHT);
		int h1 = int(C(C_SHH1) + 0.5), h2 = int(C(C_SHH2) + 0.5);
		int want = -1;
		bool onTurn = false;
		vec4 tp0 = FLK_LOAD(flk, 13), tp1 = FLK_LOAD(flk, 14);
		float trnB = min(tp0.w > 0.5 ? tp0.x : 99.0, tp1.w > 0.5 ? tp1.x : 99.0);
		float held = mod(bc - shT + 1024.0, 1024.0);
		if (u_ms_change != C(C_SHSEEN)) { W(C_SHSEEN, u_ms_change); want = pickBody(pocketDom, Rt, spd, int(shB), int(shA + 0.5), h1, bc + 31.0 * salt); }
		else if ((held > BODY_HOLD_MIN && trnB <= TURN_LEAD_BEATS) || held > BODY_HOLD_MAX) {
			onTurn = trnB <= TURN_LEAD_BEATS;
			want = pickBody(pocketDom, Rt, spd, int(shB), int(shA + 0.5), h1, floor(bc) + 31.0 * salt);
		}
		{
			float rd = C(C_RD);
			rd += (trnB <= REORDER_LEAD ? 1.0 : 0.0) * REORDER_STEPS * dBeats / REORDER_BEATS;
			W(C_RD, mod(rd, 3.0));
		}
		{
			float bEnd = C(C_BRAIDT);
			if (!(bEnd == bEnd) || bEnd > bc + 64.0) bEnd = 0.0;
#ifdef DESKTOP
			float held = clamp(u_held, 0.0, 1.0);
#else
			float held = 0.0;
#endif
			W(C_HELD, held);
			W(C_BRPH, fract(C(C_BRPH) + dBeats * FS_BR_SPIN * (1.0 + HELD_RUSH * held)));
			bool braidOn = bc < bEnd || held > HELD_BRAID_ON;
			if (lab) braidOn = mv == 22;
			if (!braidOn && floor(bc) != floor(bc - dBeats)) {
				float g = clocked ? smoothstep(BRAID_BPM0, BRAID_BPM1, beatBpm) * smoothstep(BRAID_E0, BRAID_E1, u_ms_cur_a.x)
				                    * smoothstep(BRAID_D0, BRAID_D1, u_ms_cur_a.z) : 0.0;
				if (hash11(floor(bc) * 3.17 + 0.5 + 13.0 * salt) < BRAID_P_MAX * g) { bEnd = floor(bc) + BRAID_BEATS; braidOn = true; }
			}
			W(C_BRAIDT, bEnd);
			float bw = C(C_BRAIDW);
			bw += ((braidOn ? 1.0 : 0.0) - bw) * (1.0 - exp(-dBeats / BRAID_EASE));
			W(C_BRAIDW, clamp(bw, 0.0, 1.0));
		}
		float pon = C(C_POSSON);
		float aggV = C(C_AGG);
		if (pon < 0.5 && fz > FRENZY_ON) pon = 1.0;
		else if (pon > 0.5 && fz < FRENZY_OFF) pon = 0.0;
		if (lab) pon = (mv >= 19 && mv <= 21) ? 1.0 : 0.0;
		W(C_POSSON, pon);
		W(C_POSSW, C(C_POSSW) + (pon - C(C_POSSW)) * (1.0 - exp(-dBeats / WASP_MORPH_BEATS)));
		if (pon > 0.5) {
			want = -1;
			float pc = floor(bc / POSS_BEATS);
			float kd = C(C_POSSK);
			bool newRound = pc != C(C_POSSCYC);
			if (newRound) {
				W(C_POSSCYC, pc);
				float k2 = min(floor(hashW(pc, 31.0 + salt) * 3.0), 2.0);
				if (abs(k2 - kd) < 0.5) k2 = mod(k2 + 1.0, 3.0);
				kd = k2; W(C_POSSK, kd);
			}
			if (lab) { float kl = mv == 19 ? 0.0 : (mv == 20 ? 2.0 : 1.0); if (abs(kl - kd) > 0.5) { kd = kl; W(C_POSSK, kd); newRound = true; } }
			bool drill = kd > 1.5;
			if (drill) {
				int figs[6] = int[6](SH_ARROW, SH_V, SH_FUNNEL, SH_CORK, SH_COMET, SH_STRIPE);
				int iA = min(int(hashW(pc, 17.0) * 6.0), 5);
				int iB = min(int(hashW(pc, 23.0) * 5.0), 4);
				iB += iB >= iA ? 1 : 0;
				float fs = floor((bc - pc * POSS_BEATS) / (POSS_BEATS * 0.5));
				want = figs[fs < 0.5 ? iA : iB];
				aggV = float(AGG_FIGS); W(C_AGG, aggV);
			} else if (newRound) {
				aggV = float(kd < 0.5 ? AGG_WASP : AGG_ERUPT); W(C_AGG, aggV);
				want = pickBody(3, Rt, spd, int(shB), int(shA + 0.5), h1, pc + 31.0 * salt);
			}
		}
		if (lab && pon < 0.5) want = labBody(mv) >= 0 ? labBody(mv) : -1;
		if (want >= 0 && want != int(shB)) {
			if (pon < 0.5) { W(C_SHH2, float(h1)); W(C_SHH1, shA); }
			shA = shB; shB = float(want); shT = bc;
			W(C_MORPHB, pon > 0.5 ? (int(aggV + 0.5) == AGG_FIGS ? FIG_MORPH_BEATS : WASP_MORPH_BEATS) : (onTurn ? TURN_MORPH_BEATS : FORM_MORPH_BEATS));
		} else if (want >= 0) shT = max(shT, bc - BODY_HOLD_MIN);
		W(C_SHA, shA); W(C_SHB, shB); W(C_SHT, shT);
		float lvT = float(splitLvl(int(shB)));
		if (pon > 0.5 && int(aggV + 0.5) == AGG_FIGS) lvT = max(lvT, FIG_SPLV);
		float lv = C(C_SPLV), dl = lvT - lv;
		W(C_SPLV, lv + sign(dl) * min(abs(dl), dBeats / SPLIT_STEP_BEATS));
		float bal = max(C(C_BAL) - dBeats, 0.0), bcool = max(C(C_BALCOOL) - dBeats, 0.0);
		if (kickEdge && bcool <= 0.0) { bal = BAL_HOLD_BEATS; bcool = BAL_COOL_BEATS; }
		W(C_BAL, bal); W(C_BALCOOL, bcool);
	}
}

==compute form 512==
layout(local_size_x = 256) in;
layout(r32f, binding = ctl_UNIT) uniform readonly highp image2D ctl;
layout(r32f, binding = flk_UNIT) uniform highp image2D flk;
layout(r32f, binding = ldr_UNIT) uniform highp image2D ldr;
float Cf(int k) { return imageLoad(ctl, ivec2(k, 0)).r; }
shared vec3 sPts[PRED_N];
shared float sArc[PRED_N];
shared float sR;
shared vec3 sHP[HH_N];
shared float sHA[HH_N];
shared int sHN;
vec3 pathSlot(vec3 o, float a0, float cz, float cw, vec3 corr) {
	float sTarget = a0 + o.z * cz;
	int n = 0;
	while (n < PRED_N - 2 && sArc[n + 1] < sTarget) n++;
	float segL = max(sArc[n + 1] - sArc[n], 1e-6);
	vec3 T = (sPts[n + 1] - sPts[n]) / segL;
	vec3 pos = sPts[n] + T * (sTarget - sArc[n]) + corr;
	vec3 side = normalize(cross(T, abs(T.y) < 0.95 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
	vec3 up = cross(side, T);
	vec3 cs = side * o.x * cw + up * o.y * cw;
	cs.x *= BODY_SQX;
	return pos + cs;
}
float gBrPh = 0.0;
bool gBraid = false;
vec3 flightSpineSlot(int sel, int k, float a0, float cz, float famp, vec3 corr, float beat, float reord) {
    int kp = k, nPt = N_LEAD;
    int strand = k % 3;
    if (gBraid) { kp = k / 3; nPt = N_LEAD / 3; }
    float v = (float(kp) + 0.5) / float(nPt);
    float u = fract(float(kp) * 0.61803398875);
    float len = FS_LEN * famp * cz;
    float rad = FS_RAD * famp / sqrt(max(cz, 0.4));
    vec3 o = shapeLocal(sel, u, v, len, rad, beat, reord);
    int role = kp % FS_FILL_DIV;
    float fill = (role == 0) ? FS_CORE_K : ((role == 1) ? FS_INNER_K : 1.0);
    if (sel >= 11) { o = flightFormLocal(sel, kp, nPt, famp); fill = 1.0; }
#if FS_BRAIDSETS == 2
    {
        int nS = FS_SETS * 3;
        int setI = k % FS_SETS, strand = (k / FS_SETS) % 3, along = k / nS;
        float nAl = ceil(float(N_LEAD) / float(nS));
        float vv = (float(along) + 0.5) / nAl;
        float phs = 6.2831853 * (float(strand) / 3.0 + FS_BR_TURNS * vv + FS_BR_SPIN * beat);
        float rr = FS_BR_RAD * famp;
        o = vec3((float(setI) - 0.5 * float(FS_SETS - 1)) * FS_SET_GAP + rr * 1.2 * sin(phs),
                 rr * 0.7 * sin(2.0 * phs), -len * vv);
        fill = 1.0;
    }
#endif
    vec2 plait = vec2(0.0);
    if (gBraid) {
        float phs = 6.2831853 * (float(strand) / 3.0 + FS_BR_TURNS * v + gBrPh);
        plait = BRL_RAD * famp * vec2(1.2 * sin(phs), 0.7 * sin(2.0 * phs));
    }
    vec2 oc = (o.xy * fill + plait) * vec2(1.0, FS_FLAT);
    float sTarget = a0 + o.z;
    int n = 0;
    while (n < PRED_N - 2 && sArc[n + 1] < sTarget) n++;
    float segL = max(sArc[n + 1] - sArc[n], 1e-6);
    vec3 T = (sPts[n + 1] - sPts[n]) / segL;
    vec3 pos = sPts[n] + T * (sTarget - sArc[n]) + corr;
    int nB = max(n - 3, 0), nF = min(n + 3, PRED_N - 2);
    vec3 Tb = normalize(sPts[nB + 1] - sPts[nB] + 1e-6);
    vec3 Tf = normalize(sPts[nF + 1] - sPts[nF] + 1e-6);
    vec3 toCam = vec3(0.0, 0.0, -1.0);
    vec3 ref = length(cross(T, toCam)) > 0.3 ? toCam : vec3(0.0, 1.0, 0.0);
    float bank = clamp(dot(cross(Tb, Tf), ref) * BANK_K, -1.15, 1.15);
    vec3 side0 = normalize(cross(T, ref));
    vec3 up0 = cross(side0, T);
    float cb = cos(bank), sb = sin(bank);
    vec3 sideF = cb * side0 + sb * up0;
    vec3 upF = -sb * side0 + cb * up0;
    float flare = 1.0 + 0.25 * abs(bank) * v;
    vec3 cs = (sideF * oc.x + upF * oc.y) * flare;
    cs.x *= BODY_SQX;
    return pos + cs;
}
vec3 figTrailSlot(int k, float famp) {
	if (sHN < 2) return sHP[0];
	float v = (float(k) + 0.5) / float(N_LEAD);
	float th = 6.2831853 * fract(float(k) * 0.61803398875);
	float env = sin(3.14159265 * pow(v, 0.72));
	float r = FS_RAD * famp * 0.30 * env;
	float sT = v * FS_LEN * FIG_TRAIL_LEN * famp;
	int n = 0;
	while (n < sHN - 2 && sHA[n + 1] < sT) n++;
	float segL = max(sHA[n + 1] - sHA[n], 1e-6);
	vec3 pos = mix(sHP[n], sHP[n + 1], clamp((sT - sHA[n]) / segL, 0.0, 1.0));
	vec3 T = normalize(sHP[n] - sHP[n + 1] + 1e-6);
	vec3 side = normalize(cross(T, vec3(0.0, 0.0, -1.0)) + 1e-6);
	vec3 up = cross(side, T);
	return pos + side * (r * cos(th)) + up * (r * sin(th) * FS_FLAT);
}
vec3 bodyAt(int sh, int k, float df, int f, float bc, float a0, float cz, float cw, vec3 corr, float scale, float famp, float reord) {
	if (isFB(sh)) return flightSpineSlot(sh - FB_BASE, k, a0, clamp(cz, FS_SQ_MIN, 1.0), famp * FB_SCALE * BODY_SIZE, corr, bc, reord);
	int kk = gBraid ? (k / 3) * 3 : k;
	vec3 o = (sh == SH_BRAID || sh == SH_BRAID2) ? braidSlot(sh, kk, bc) / vec3(cw, cw, cz) : bodySlot(sh, kk, df, f, bc) * scale;
	if (gBraid) {
		float v = (float(k / 3) + 0.5) / float(N_LEAD / 3);
		float phs = 6.2831853 * (float(k % 3) / 3.0 + FS_BR_TURNS * v + gBrPh);
		o.xy += BRL_RAD * famp * vec2(1.2 * sin(phs), 0.7 * sin(2.0 * phs)) / cw;
	}
	return pathSlot(o, a0, cz, cw, corr);
}
void main() {
	int f = int(gl_WorkGroupID.x), t = int(gl_LocalInvocationID.x);
	float sg = f == 0 ? 1.0 : -1.0;
	float dt = max(Cf(C_DT), 1e-4);
	float bc = Cf(C_BC);
	float w[5];
	for (int i = 0; i < 5; i++) w[i] = Cf(C_HW0 + i);
	float hPocket = Cf(C_HPOCK), yRoom = max(Cf(C_YROOM), 0.8);
	vec3 hSepX = vec3(Cf(C_HSEP), Cf(C_HSEP + 1), Cf(C_HSEP + 2));
	float L = clamp(Cf(C_LOUD) / max(Cf(C_LMAX), 0.05), 0.0, 1.0);
	float oph = Cf(C_OPH), odir = Cf(C_ODIR), splitAx = Cf(C_SPLITAXE);
	vec3 c3 = vec3(Cf(C_CX), Cf(C_CY), 0.0);
	float stepB = (PRED_BACK + PRED_FWD) / float(PRED_N - 1);
	if (t < PRED_N) {
		float db = -PRED_BACK + float(t) * stepB;
		sPts[t] = c3 + sg * pocketOrbit(oph + odir * db * 6.2831853 / ORBIT_BEATS, Cf(C_OPHH) + odir * db * 6.2831853 / ORBIT_BEATS * Cf(C_HSPD),
		                                bc + db, Cf(C_BCH) + db * Cf(C_HSPD), w, hPocket, hSepX, yRoom, L, splitAx, Cf(C_FARSH),
		                                Cf(C_TUM) + db * Cf(C_TUMR));
	}
	barrier();
	int i0 = int(round(PRED_BACK / stepB));
	if (t == 0) {
		sArc[0] = 0.0;
		for (int n = 1; n < PRED_N; n++) sArc[n] = sArc[n - 1] + length(sPts[n] - sPts[n - 1]);
		float kMax = 0.0;
		for (int n = i0 + 1; n < PRED_N - 1; n++) {
			vec3 u = sPts[n] - sPts[n - 1], v = sPts[n + 1] - sPts[n], ww = sPts[n + 1] - sPts[n - 1];
			float den = length(u) * length(v) * length(ww);
			if (den > 1e-9) kMax = max(kMax, 2.0 * length(cross(u, v)) / den);
		}
		sR = 1.0 / max(kMax, 1e-3);
		int i1 = min(i0 + 4, PRED_N - 1);
		FLK_STORE(flk, 10 + f, vec4(sR, (sArc[i1] - sArc[i0]) / (float(i1 - i0) * stepB), 0.0, 1.0));
		float bTo = 99.0, tA = 0.0;
		for (int n = i0 + 2; n < PRED_N - 3; n++) {
			vec3 Tb = normalize(sPts[n] - sPts[n - 2] + 1e-6), Tf = normalize(sPts[n + 2] - sPts[n] + 1e-6);
			float a = acos(clamp(dot(Tb, Tf), -1.0, 1.0));
			if (a > TURN_CORNER_A) { bTo = float(n - i0) * stepB; tA = a; break; }
		}
		FLK_STORE(flk, 13 + f, vec4(bTo, tA, 0.0, 1.0));
		vec3 hNow = FLK_LOAD(flk, 6 + f).xyz;
		LDR_STORE(ldr, 70 + f, frame % HH_N, vec4(hNow, time));
		sHP[0] = hNow; sHA[0] = 0.0;
		int cnt = 1;
		float tPrev = time;
		for (int n = 1; n < HH_N; n++) {
			vec4 hh = LDR_LOAD(ldr, 70 + f, ((frame - n) % HH_N + HH_N) % HH_N);
			if (hh.w <= 0.0 || hh.w >= tPrev || distance(hh.xyz, sHP[cnt - 1]) > 0.7) break;
			sHA[cnt] = sHA[cnt - 1] + distance(hh.xyz, sHP[cnt - 1]);
			sHP[cnt] = hh.xyz; tPrev = hh.w; cnt++;
		}
		sHN = cnt;
	}
	barrier();
	int m = min(t, N_LMAX - 1), k = m % N_LEAD;
	float splitL = Cf(C_SPLV);
	int nAct = nLeaders(splitL);
	vec4 lp = LDR_LOAD(ldr, 2 * f, m), lv4 = LDR_LOAD(ldr, 2 * f + 1, m);
	vec4 pp = LDR_LOAD(ldr, 2 * f, leaderParent(m)), pv = LDR_LOAD(ldr, 2 * f + 1, leaderParent(m));
	barrier();
	if (t >= N_LMAX) return;
	float df = leaderOff(m, splitL);
	int shA = int(Cf(C_SHA) + 0.5), shB = int(Cf(C_SHB) + 0.5);
	float morphB = Cf(C_MORPHB);
	if (morphB < 0.01) morphB = FORM_MORPH_BEATS;
	float mB = smoothstep(0.0, morphB, mod(bc - Cf(C_SHT) + 1024.0, 1024.0));
	float vmaxL = mix(LEAD_VMAX, FRENZY_VMAX, clamp(Cf(C_FRENZY), 0.0, 1.0));
	vmaxL = max(vmaxL, 1.25 * Cf(C_PINV) * smoothstep(0.0, 1.0, clamp(Cf(C_PINW), 0.0, 1.0)));
	float scale = FORM_R * mix(0.8, 1.25, smoothstep(0.1, 0.8, L)) * BODY_SIZE;
	vec3 hp = FLK_LOAD(flk, 6 + f).xyz;
	vec3 corr = hp - sPts[i0];
	float a0 = sArc[i0];
	float bodyLen = mix(formLen(shA), formLen(shB), mB) * scale;
	float cz = clamp(TURN_FIT * 3.14159 * sR / max(bodyLen, 1e-3), CZ_MIN, 1.0);
	float cw = 1.0 / sqrt(cz);
	float famp = mix(0.85, 1.15, smoothstep(0.1, 0.8, L));
	float reord = Cf(C_RD);
	gBrPh = Cf(C_BRPH);
	vec3 slot = mix(bodyAt(shA, k, df, f, bc, a0, cz, cw, corr, scale, famp, reord),
	                bodyAt(shB, k, df, f, bc, a0, cz, cw, corr, scale, famp, reord), mB);
	{
		float bwF = smoothstep(0.0, 1.0, clamp(Cf(C_BRAIDW), 0.0, 1.0));
		if (bwF > 0.001) {
			gBraid = true;
			slot = mix(slot, mix(bodyAt(shA, k, df, f, bc, a0, cz, cw, corr, scale, famp, reord),
			                     bodyAt(shB, k, df, f, bc, a0, cz, cw, corr, scale, famp, reord), mB), bwF);
			gBraid = false;
		}
	}
	{
		float figW = smoothstep(0.0, 1.0, clamp(Cf(C_FIGAMP), 0.0, 1.0));
		if (FIG_TRAIL_ON > 0 && figW > 0.001 && sHN >= 2) slot = mix(slot, figTrailSlot(k, famp), figW);
	}
	{
		int gk = k / 3;
		float ffw = clamp(Cf(C_FFW), 0.0, 1.0);
		float sw1 = smoothstep(0.0, 1.0, clamp(Cf(C_SPLITW), 0.0, 1.0)) * (1.0 - ffw), sw2 = smoothstep(0.0, 1.0, clamp(Cf(C_SPL2W), 0.0, 1.0)) * (1.0 - ffw);
		float ang = ((gk & 1) == 1 ? 1.5707963 * sw1 : 0.0);
		if (ang > 1e-4) { vec3 cp = slot - c3; cp.xy = vrot(cp.xy, ang); slot = c3 + cp; }
		if ((gk & 2) == 2 && sw2 > 1e-4) {
			float sgz = (((gk & 1) ^ f) == 0) ? -1.0 : 1.0;
			slot.z += sgz * SPLIT2_DZ * sw2;
		}
	}
	{
		float eW = smoothstep(0.0, 1.0, clamp(Cf(C_EYE_W), 0.0, 1.0));
		int eT = int(Cf(C_EYE_T) + 0.5);
		if (eW > 0.001 && eT > 0) {
			int gi = f * N_LEAD + k;
			vec3 e = (eT == 1) ? eyeRings(gi, bc) : eyeArmillary(gi, bc);
			float edge = resolution.x / max(resolution.y, 1.0);
			float room = min(min(boxXW(edge) - abs(c3.x), boxYW() - abs(c3.y)), min(BOX_ZB - c3.z, c3.z - BOX_ZN)) - GOAL_MARGIN;
			e *= min(1.0, max(room, 0.05) / EYE_R);
			slot = mix(slot, c3 + e, eW);
		}
	}
	{
		float pW = Cf(C_PINW);
		if (pW > 0.001) {
			float R = Cf(C_PINRW);
			int g = m / N_LEAD;
			float a = Cf(C_PINPH) + 6.2831853 * (float(f * N_LEAD + k) + 0.25 * float(g)) / float(2 * N_LEAD);
			vec3 ua = vec3(Cf(C_PINUX), Cf(C_PINUY), 0.0);
			vec3 rp = c3 + (cos(a) * ua + vec3(0.0, 0.0, sin(a))) * R;
			float pnW = smoothstep(0.0, 1.0, clamp(Cf(C_PENTW), 0.0, 1.0));
			if (pnW > 0.001) {
				float Rp = Cf(C_PENTRW);
				vec3 Q;
				if (PENT_SPHERE > 0) {
					int g2 = m / N_LEAD;
					int i2 = f * N_LEAD + k, per = (2 * N_LEAD) / SPHERE_MERID;
					float ring = float(i2 / per), j = float(i2 % per) + 0.25 * float(g2);
					float th = 3.14159265 * ring / float(SPHERE_MERID) + 6.2831853 * bc / SPHERE_TURN_BEATS;
					float aa = 6.2831853 * (j / float(per) + bc / SPHERE_FLOW_BEATS);
					Q = vec3(sin(aa) * cos(th), cos(aa), sin(aa) * sin(th)) * Rp;
				} else {
					float s5 = fract(a / 6.2831853) * 5.0, e = floor(s5), tt = s5 - e;
					float p0 = 6.2831853 * bc / PENT_TURN_BEATS + e * 2.5132741, p1 = p0 + 2.5132741;
					Q = vec3(mix(vec2(sin(p0), cos(p0)), vec2(sin(p1), cos(p1)), tt) * Rp, 0.0);
				}
				rp = mix(rp, c3 + Q, pnW);
			}
			slot = mix(slot, rp, pW);
		}
	}
	float possW = Cf(C_POSSW) * (1.0 - clamp(Cf(C_FFW), 0.0, 1.0));
	if (possW > 0.001) {
		int agg = int(Cf(C_AGG) + 0.5);
		if (agg == AGG_ERUPT) {
			float q = mod(bc, POSS_BEATS);
			float env = smoothstep(0.0, 0.8, q) * (1.0 - smoothstep(5.5, 7.2, q));
			vec3 col = columnSlot(k, df, f);
			float st = 0.6 + 0.4 * smoothstep(0.15, 0.7, Cf(C_PUSH));
			col.y = COL_Y0 + (col.y - COL_Y0) * st;
			slot = mix(slot, c3 + col, possW * env);
		} else if (agg == AGG_WASP) {
			slot = mix(slot, c3 + waspSlot(k, df, f, bc), possW * possess(f * 4 + k / WASP_PER, bc));
		}
	}
	{
		float edge = resolution.x / max(resolution.y, 1.0);
		vec3 lo = vec3(-boxXW(edge), -boxYW(), BOX_ZN) + GOAL_MARGIN, hi = vec3(boxXW(edge), boxYW(), BOX_ZB) - GOAL_MARGIN;
		vec3 off = slot - c3;
		float fit = 1.0;
		for (int ax = 0; ax < 3; ax++) {
			float o = off[ax];
			if (o > 1e-5) fit = min(fit, (hi[ax] - c3[ax]) / o);
			else if (o < -1e-5) fit = min(fit, (lo[ax] - c3[ax]) / o);
		}
		slot = c3 + off * clamp(fit, 0.0, 1.0);
	}
	vec4 prev = LDR_LOAD(ldr, 4 + f, m);
	vec3 vSlot = prev.w > 0.5 ? (slot - prev.xyz) / dt : vec3(0.0);
	float vs = length(vSlot);
	{
		float lag = lp.w > 0.5 ? length(slot - lp.xyz) : 0.0;
		vmaxL = clamp(max(LEAD_CATCH_V * vs, LEAD_CATCH_K * lag), vmaxL, max(vmaxL, LEAD_VMAX_HI));
	}
	if (vs > vmaxL) vSlot *= vmaxL / vs;
	LDR_STORE(ldr, 4 + f, m, vec4(slot, 1.0));
	if (m >= nAct) { LDR_STORE(ldr, 2 * f, m, vec4(0.0)); return; }
	if (lp.w < 0.5) {
		if (pp.w > 0.5) { lp = pp; lv4 = pv; }
		else { lp = vec4(slot, 1.0); lv4 = vec4(0.0); }
	}
	float om = 6.2831853 / LEAD_PERIOD;
	vec3 vCmd = vSlot * LEAD_FF;
	float lungeCap = 1.0;
	{
		float hW = clamp(Cf(C_HUNTW), 0.0, 1.0) * clamp(Cf(C_HEN), 0.0, 1.0);
		if (hW > 0.001) {
			uint ht = pcg(uint(f * N_LMAX + m) * 1597334677u + 11u);
			float ph = rnd(ht), f1 = 0.6 + 0.8 * rnd(ht), f2 = 0.5 + 0.9 * rnd(ht), amp = 0.5 + rnd(ht);
			float wa = TIP_WANDER * sin(time * 6.2831853 * TIP_WANDER_HZ * f1 + 6.2831853 * ph)
			                      * sin(time * 6.2831853 * TIP_WANDER_HZ * f2 * 0.61 + 17.0 * ph);
			float cyc = fract(bc / TIP_LUNGE_BEATS + ph), cycN = floor(bc / TIP_LUNGE_BEATS + ph);
			uint hc = pcg(uint(f * N_LMAX + m) * 2246822519u + uint(cycN) * 3266489917u + 29u);
			float fire = rnd(hc) < TIP_LUNGE_SKIP ? 0.0 : 1.0;
			float lunge = exp(-cyc * TIP_LUNGE_DECAY) * fire * amp * hW;
			float lvS = length(vSlot);
			vec3 fwdL = lvS > 1e-3 ? vSlot / lvS : vec3(1.0, 0.0, 0.0);
			vec3 sideC = normalize(cross(vec3(0.0, 0.0, 1.0), fwdL) + 1e-6);
			vCmd = vSlot * (LEAD_FF + TIP_LUNGE * lunge) + sideC * (wa * max(lvS, 0.3) * TIP_SWAY_K * hW);
			float cmdMax = max(lvS * LEAD_FF, TIP_VMAX);
			float cl = length(vCmd);
			if (cl > cmdMax) vCmd *= cmdMax / cl;
			lungeCap = max(1.0, cmdMax / max(vmaxL, 1e-3));
		}
	}
	vec3 v = lv4.xyz + (om * om * (slot - lp.xyz) + 2.0 * LEAD_DAMP * om * (vCmd - lv4.xyz)) * dt;
	float sp = length(v);
	if (sp > vmaxL * lungeCap) v *= vmaxL * lungeCap / sp;
	LDR_STORE(ldr, 2 * f, m, vec4(lp.xyz + v * dt, 1.0));
	if (m < N_LEAD) LDR_STORE(ldr, 6 + f * HIST_N + frame % HIST_N, m, vec4(lp.xyz + v * dt, time));
	LDR_STORE(ldr, 2 * f + 1, m, vec4(v, 0.0));
}

==compute boids 1024==
layout(local_size_x = 64) in;
layout(rgba32f, binding = boid_out_UNIT) uniform writeonly highp image2D boid_out;
layout(r32f, binding = ctl_UNIT) uniform readonly highp image2D ctl;
layout(r32f, binding = flk_UNIT) uniform readonly highp image2D flk;
layout(r32f, binding = ldr_UNIT) uniform readonly highp image2D ldr;
#ifdef DESKTOP
#define SEP_R0 (u_tune[9] > 0.0 ? u_tune[9] : 0.090)
#else
#define SEP_R0    0.090
#endif
#define SEP_R1    0.060
#define NB_R      0.25
#ifdef DESKTOP
#define W_SEP (u_tune[10] > 0.0 ? u_tune[10] : 5.0)
#else
#define W_SEP     5.0
#endif
#define W_ALIGN0  0.4
#ifdef DESKTOP
#define W_ALIGN1 (u_tune[11] > 0.0 ? u_tune[11] : 3.0)
#else
#define W_ALIGN1  3.0
#endif
#define W_COH0    0.20
#ifdef DESKTOP
#define W_COH1 (u_tune[12] > 0.0 ? u_tune[12] : 0.6)
#else
#define W_COH1    0.6
#endif
#ifdef DESKTOP
#define W_TGT (u_tune[6] > 0.0 ? u_tune[6] : 2.5)
#else
#define W_TGT     2.5
#endif
#define V_QUIET   0.16
#define V_LOUD    0.45
#define HAWK_R    0.22
#define HAWK_K    7.0
#ifdef DESKTOP
#define JIT_K (u_tune[13] > 0.0 ? u_tune[13] : 1.2)
#else
#define JIT_K     1.2
#endif
#define AIR_S     3.5
#ifdef DESKTOP
#define AIR_Q (u_tune[14] > 0.0 ? u_tune[14] : 2.5)
#else
#define AIR_Q     2.5
#endif
#define AIR_L     0.6
#define SPLIT_D   0.28
#define STRAG_SHARE 0.20
#define STRAG_COH   0.30
#define STRAG_VMAX  0.85
#ifdef DESKTOP
#define CATCH_K (u_tune[8] > 0.0 ? u_tune[8] : 2.0)
#else
#define CATCH_K     2.0
#endif
#define CATCH_D0    0.08
#define CATCH_D1    0.40
#define GOAL_NEAR   0.45
#define LEAD_VFF    1.0
#ifdef DESKTOP
#define TRACK_K (u_tune[7] > 0.0 ? u_tune[7] : 4.0)
#else
#define TRACK_K     4.0
#endif
#define GOAL_D0     0.03
#define GOAL_D1     0.20
float Cv(int k) { return imageLoad(ctl, ivec2(k, 0)).r; }
vec3 ldrPos(int row, int m) { return vec3(imageLoad(ldr, ivec2(4 * m, row)).r, imageLoad(ldr, ivec2(4 * m + 1, row)).r, imageLoad(ldr, ivec2(4 * m + 2, row)).r); }
void main() {
	int i = int(gl_GlobalInvocationID.x);
	if (i >= N_ALL) return;
	float u_dt = Cv(C_DT);
	vec4 P = texelFetch(boid, ivec2(i, 0), 0);
	vec4 V = texelFetch(boid, ivec2(i, 1), 0);
	int f = i / N_PER;
	vec3 headP = FLK_LOAD(flk, 6 + f).xyz;
	vec3 headV = (headP - FLK_LOAD(flk, 8 + f).xyz) / max(u_dt, 1e-4);
	if (i % N_PER == 0) {
		imageStore(boid_out, ivec2(i, 0), vec4(headP, P.w));
		imageStore(boid_out, ivec2(i, 1), vec4(headV, V.w));
		imageStore(boid_out, ivec2(i, 2), vec4(1.0, 1.0, 0.0, 1.0));
		return;
	}
	int nAct = nLeaders(Cv(C_SPLV));
	vec3 sp = vec3(0.0), sv = vec3(0.0);
	float sw2 = 0.0;
	int kNear = 0;
	if (Cv(C_BAL) > 0.0 || Cv(C_PINW) > 0.5) {
		int k = (i % N_PER) % nAct;
		kNear = k;
		sp = ldrPos(2 * f, k); sv = ldrPos(2 * f + 1, k); sw2 = 1.0;
	} else {
		float best = 1e9;
		for (int k = 0; k < nAct; k++) {
			vec3 dk = ldrPos(2 * f, k) - P.xyz;
			float d2k = dot(dk, dk);
			if (d2k < best) { best = d2k; kNear = k; }
		}
		for (int k = 0; k < nAct; k++) {
			vec3 lp = ldrPos(2 * f, k);
			vec3 d = lp - P.xyz;
			float wgt = exp(-(dot(d, d) - best) / (LEAD_SOFT * LEAD_SOFT));
			if (wgt < 0.01) continue;
			sp += lp * wgt; sv += ldrPos(2 * f + 1, k) * wgt; sw2 += wgt;
		}
	}
	float fzw = clamp(Cv(C_FRENZY), 0.0, 1.0);
	uint ho = pcg(uint(i) * 2246822519u + 3u);
	vec3 leadP = sp / sw2 + rndDir(ho) * LEAD_SPREAD * mix(1.0, FRENZY_SPREAD, fzw) * pow(rnd(ho), 0.333);
	vec3 leadV = sv / sw2;
	vec4 Lf = texelFetch(boid, ivec2(i, 2), 0);
	float lifeAge = Lf.z + u_dt, lifeSpan = Lf.w;
	if (!(lifeSpan > 0.0)) {
		uint rl = pcg(uint(i) * 2654435761u + 91u);
		lifeAge = rnd(rl) * LIFE0;
		lifeSpan = mix(LIFE0, LIFE1, rnd(rl));
	}
	if (lifeAge >= lifeSpan) {
		uint rb = pcg(uint(i) * 747796405u + uint(frame) * 2891336453u + 5u);
		lifeAge = 0.0;
		lifeSpan = mix(LIFE0, LIFE1, rnd(rb));
		P.xyz = leadP + rndDir(rb) * LIFE_BIRTH_R * rnd(rb);
		V.xyz = leadV + rndDir(rb) * LIFE_BIRTH_V;
	}
	float lifeTired = clamp(lifeAge / lifeSpan, 0.0, 1.0);
	float lifeStall = LIFE_STALL * pow(lifeTired, LIFE_STALL_K);
	float lifeSag = 0.5 + 0.5 * sin(time * 6.2831853 * LIFE_STALL_HZ * (0.7 + 0.9 * fract(float(i) * 0.3819660)) + 6.2831853 * fract(float(i) * 0.6180339))
	                           * sin(time * 6.2831853 * LIFE_STALL_HZ * 0.57 + 6.2831853 * fract(float(i) * 0.7548777));
	lifeSag = pow(clamp(lifeSag, 0.0, 1.0), LIFE_SAG_P);
	float lifeDrop = 1.0 - lifeStall * lifeSag;
	float lifeBirth = 1.0 - smoothstep(0.0, LIFE_BIRTH_T / max(lifeSpan, 1e-3), lifeTired);
	float lifeDeadH = pow(lifeTired, LIFE_HEAT_CURVE), lifeDeadI = pow(lifeTired, LIFE_BRIGHT_CURVE);
	float lifeI = (1.0 - LIFE_DEAD_I * lifeDeadI) * (1.0 + LIFE_BIRTH_I * lifeBirth);
	float lifeH = (1.0 - LIFE_DEAD_H * lifeDeadH) * (1.0 + LIFE_BIRTH_H * lifeBirth);

	float L = loudN(Cv(C_LOUD), Cv(C_LMAX));
	float groove = clamp(Cv(C_GROOVE), 0.0, 1.5);
	float sw = Cv(C_SPLITW);
	float SEP_R = mix(SEP_R0, SEP_R1, smoothstep(0.15, 0.75, L)) * mix(1.0, FRENZY_SEP, fzw);
	uint hs = pcg(uint(i) * 40503u + 7u);
	float strag = float(rnd(hs) < STRAG_SHARE);
	vec3 sep = vec3(0.0), ali = vec3(0.0), coh = vec3(0.0);
	float cnt = 0.0;
	for (int j = 0; j < N_ALL; j++) {
		if (j == i) continue;
		vec4 Q = texelFetch(boid, ivec2(j, 0), 0);
		vec3 d = Q.xyz - P.xyz;
		float r2 = dot(d, d);
		if (r2 > NB_R * NB_R) continue;
		float r = sqrt(r2);
		if (r < SEP_R) sep -= d / max(r, 1e-4) * (1.0 - r / SEP_R);
		if (j / N_PER != f) continue;
		if (j % N_PER == 0) continue;
		float g = mix(1.0, abs(Q.w - P.w) < 0.5 ? 1.0 : 0.0, sw);
		if (g <= 0.0) continue;
		ali += texelFetch(boid, ivec2(j, 1), 0).xyz * g;
		coh += Q.xyz * g;
		cnt += g;
	}
	float wA = mix(W_ALIGN0, W_ALIGN1, smoothstep(0.15, 0.75, L)) * (0.75 + 0.35 * groove);
	float wC = mix(W_COH0, W_COH1, L) * mix(1.0, STRAG_COH, strag);
	vec3 acc = W_SEP * sep;
	if (cnt > 0.0) acc += wA * (ali / cnt - V.xyz) + wC * (coh / cnt - P.xyz);
	vec3 T = leadP;
	if (sw > 0.0) {
		vec3 tn = dot(leadV, leadV) > 1e-8 ? normalize(leadV) : vec3(1.0, 0.0, 0.0);
		vec3 side = normalize(cross(tn, abs(tn.y) < 0.95 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
		T += side * SPLIT_D * sw * (P.w < 0.5 ? -1.0 : 1.0);
	}
	float vT = max(mix(V_QUIET, V_LOUD, smoothstep(0.1, 0.8, L)), 1.1 * length(headV));
	vT = max(vT, 1.1 * length(leadV) * max(fzw, LEAD_FOLLOW));
	vT *= lifeDrop;
	vec3 toT = T - P.xyz;
	float dT = length(toT);
	vT *= 1.0 + CATCH_K * smoothstep(CATCH_D0, CATCH_D1, dT);
	float goal = mix(GOAL_NEAR, 1.0, smoothstep(GOAL_D0, GOAL_D1, dT));
	float scatW = Cv(C_SCAT) * (1.0 - smoothstep(SCAT_R0, SCAT_R1, dT));
	goal = mix(goal, -SCATTER_K, scatW);
	goal -= ACCENT_K * Cv(C_ACCENT);
	{
		vec3 des = leadV * LEAD_VFF + normalize(toT + 1e-6) * vT * clamp(dT / 0.15, 0.3, 1.0);
		float dl = length(des);
		if (dl > 1.8 * vT) des *= 1.8 * vT / dl;
		float wT = max(W_TGT, TRACK_K * length(leadV));
		acc += wT * goal * (des - V.xyz);
	}
	vec4 hk = FLK_LOAD(flk, 4 + f);
	vec3 dh = P.xyz - hk.xyz;
	float rh = length(dh);
	if (hk.w > 0.01 && rh < HAWK_R) acc += dh / max(rh, 1e-4) * (1.0 - rh / HAWK_R) * HAWK_K * hk.w;
	float kw = Cv(C_KW);
	if (kw > 0.01) {
		vec3 dk = P.xyz - vec3(Cv(C_KX), Cv(C_KY), Cv(C_KZ));
		float rk = length(dk);
		if (rk < KNOCK_R) acc += dk / max(rk, 1e-4) * (1.0 - rk / KNOCK_R) * KNOCK_K * kw;
	}
	float kw2 = Cv(C_K2W);
	if (kw2 > 0.01) {
		vec3 dk = P.xyz - vec3(Cv(C_K2X), Cv(C_K2Y), Cv(C_K2Z));
		float rk = length(dk);
		if (rk < KNOCK_R) acc += dk / max(rk, 1e-4) * (1.0 - rk / KNOCK_R) * KNOCK_K * kw2;
	}
	float ta = time * 0.15;
	vec3 q = P.xyz * AIR_S;
	vec3 air = vec3(sin(q.z + ta) + cos(q.y + 1.3 * ta), sin(q.x + 0.8 * ta) + cos(q.z + 2.0), sin(q.y + 1.1 * ta) + cos(q.x + 0.7 * ta));
	acc += mix(AIR_Q, AIR_L, smoothstep(0.1, 0.7, L)) * (air * 0.25 * vT);
	uint s = pcg(uint(i) * 9781u + uint(frame) * 6271u);
	acc += JIT_K * Cv(C_HI) * (vec3(rnd(s), rnd(s), rnd(s)) - 0.5) * 2.0;
	acc += ACCENT_SKIT * Cv(C_ACCENT) * (vec3(rnd(s), rnd(s), rnd(s)) - 0.5) * 2.0;
	vec3 v = V.xyz + acc * u_dt;
	float spd = length(v);
	float vMin = 0.35 * vT, vMax = (1.8 * vT + 0.05) * mix(1.0, STRAG_VMAX, strag);
	vMax *= 1.0 + ACCENT_VBOOST * Cv(C_ACCENT);
	if (spd > vMax) v *= vMax / spd;
	else if (spd < vMin) v = (spd > 1e-5 ? v / spd : vec3(1.0, 0.0, 0.0)) * vMin;
	vec3 nP = P.xyz + v * u_dt;
	float mH = smoothstep(0.0, 1.0, clamp(Cv(C_HUNTW), 0.0, 1.0));
	mH *= 1.0 - ROPE_TOUCH * clamp(max(max(kw, kw2), Cv(C_SCAT)), 0.0, 1.0);
	if (mH > 0.001) {
		int jj = i % N_PER;
		int kS = jj % N_LEAD, wR = jj / N_LEAD;
		float perL = float((N_PER + N_LEAD - 1) / N_LEAD);
		float wvGl = smoothstep(WV_T0, WV_T1, lifeTired);
		float wvG = mix(1.0, WV_GRIP_MIN, wvGl);
		{
			uint hh3 = pcg(uint(i) * 2654435761u + uint(lifeSpan * 997.0) + 211u);
			float lt = clamp((lifeTired - WV_HOP_T) / (1.0 - WV_HOP_T), 0.0, 1.0);
			int hopN = lifeTired > WV_HOP_T ? int(floor(lt * WV_HOPS + rnd(hh3))) : 0;
			int k0 = kS;
			for (int h = 0; h < 4; h++) {
				if (h >= hopN) break;
				kS = (kS + 1 + int(rnd(hh3) * float(N_LEAD - 1))) % N_LEAD;
			}
			int g0 = (k0 / 3) & 3, gS = (kS / 3) & 3;
			kS = (kS + 3 * ((g0 - gS + 4) & 3)) % N_LEAD;
		}
		float Lr = 0.0, tPrev = 0.0;
		int nOk = -1;
		vec3 pPrev = vec3(0.0);
		for (int kk = 0; kk < HIST_N - 1; kk++) {
			int rr = 6 + f * HIST_N + ((frame - kk) % HIST_N + HIST_N) % HIST_N;
			vec4 hh = vec4(imageLoad(ldr, ivec2(4 * kS, rr)).r, imageLoad(ldr, ivec2(4 * kS + 1, rr)).r,
			               imageLoad(ldr, ivec2(4 * kS + 2, rr)).r, imageLoad(ldr, ivec2(4 * kS + 3, rr)).r);
			if (hh.w <= 0.0 || (kk > 0 && (hh.w >= tPrev || distance(hh.xyz, pPrev) > 0.7))) break;
			if (kk > 0) Lr += distance(hh.xyz, pPrev);
			pPrev = hh.xyz; tPrev = hh.w; nOk = kk;
		}
		if (nOk >= 1 && Lr > 1e-4) {
			uint hj = pcg(uint(i) * 3266489917u + 17u);
			float sFr = (float(wR) + 1.0) / perL;
			sFr = clamp(mix(sFr, lifeTired, WV_SLIDE) + (rnd(hj) - 0.5) * 2.0 * WV_S_JIT, 0.01, 1.0);
			float sTgt = sFr * min(FOL_ROPE_LEN, Lr);
			float acc2 = 0.0;
			vec3 pA = vec3(0.0), pB = vec3(0.0), vR = vec3(0.0);
			for (int kk = 0; kk <= nOk; kk++) {
				int rr = 6 + f * HIST_N + ((frame - kk) % HIST_N + HIST_N) % HIST_N;
				vec3 hp = vec3(imageLoad(ldr, ivec2(4 * kS, rr)).r, imageLoad(ldr, ivec2(4 * kS + 1, rr)).r, imageLoad(ldr, ivec2(4 * kS + 2, rr)).r);
				if (kk == 0) { pA = hp; pB = hp; continue; }
				float seg = distance(hp, pA);
				if (acc2 + seg >= sTgt || kk == nOk) {
					float tt = seg > 1e-6 ? clamp((sTgt - acc2) / seg, 0.0, 1.0) : 0.0;
					pB = mix(pA, hp, tt);
					vR = (pA - hp) / max(u_dt, 1e-4);
					break;
				}
				acc2 += seg; pA = hp;
			}
			vec3 bk = dot(vR, vR) > 1e-8 ? -normalize(vR) : vec3(0.0, 0.0, 1.0);
			vec3 sd = normalize(cross(bk, abs(bk.y) < 0.95 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
			vec3 up2 = cross(bk, sd);
			uint hw = pcg(uint(i) * 2246822519u + 3u);
			float sideW = (rnd(hw) - 0.5) * 2.0, liftW = (rnd(hw) - 0.5) * 2.0;
			uint hb2 = pcg(uint(i) * 1103515245u + 97u);
			float fa = 0.3 + 0.9 * rnd(hb2), fb = 0.3 + 0.9 * rnd(hb2), fc = 0.3 + 0.9 * rnd(hb2);
			float pa1 = 6.2831853 * rnd(hb2), pb1 = 6.2831853 * rnd(hb2), pc1 = 6.2831853 * rnd(hb2);
			float stragB = rnd(hb2) < FOL_STRAG_SHARE ? FOL_STRAG_MUL : 1.0;
			vec3 wob = vec3(sin(time * 6.2831853 * fa + pa1) * sin(time * 6.2831853 * fa * 0.37 + pb1),
			                sin(time * 6.2831853 * fb + pb1) * sin(time * 6.2831853 * fb * 0.41 + pc1),
			                sin(time * 6.2831853 * fc + pc1) * 0.35);
			float ageK = 1.0 + FOL_AGE_K * lifeTired * lifeTired;
			vec3 stW = pB + (sd * sideW + up2 * liftW) * FOL_GIRTH + (sd * wob.x + up2 * wob.z + cross(sd, up2) * wob.y) * FOL_WANDER * stragB * ageK;
			float per = mix(WV_PER_HOT, WV_PER_COLD, wvGl);
			float w0 = 6.2831853 / per;
			vec3 aW = (stW - P.xyz) * (w0 * w0) + (vR - V.xyz) * (2.0 * WV_DAMP * w0);
			float aL = length(aW);
			if (aL > WV_ACC) aW *= WV_ACC / aL;
			vec3 vD = V.xyz + aW * u_dt;
			vec3 nR = mix(P.xyz + vD * u_dt, stW, WV_SNAP * pow(wvG, 4.0));
			vec3 vF = (nR - P.xyz) / max(u_dt, 1e-4);
			float vl = length(vF);
			if (vl > ROPE_VMAX) vF *= ROPE_VMAX / vl;
			nP = mix(nP, nR, mH);
			v = mix(v, vF, mH);
		}
	}
	{
		float edge = resolution.x / max(resolution.y, 1.0);
		vec3 lo = vec3(-boxXW(edge), -boxYW(), BOX_ZN), hi = vec3(boxXW(edge), boxYW(), BOX_ZB);
		vec3 cl = clamp(nP, lo, hi);
		if (cl != nP) {
			if (nP.x != cl.x && v.x * (nP.x - cl.x) > 0.0) v.x = 0.0;
			if (nP.y != cl.y && v.y * (nP.y - cl.y) > 0.0) v.y = 0.0;
			if (nP.z != cl.z && v.z * (nP.z - cl.z) > 0.0) v.z = 0.0;
			nP = cl;
		}
	}
	bool bad = !(dot(nP, nP) < 1e30) || !(dot(v, v) < 1e30);
	if (bad) { nP = leadP; v = vec3(0.0); }
	imageStore(boid_out, ivec2(i, 0), vec4(nP, P.w));
	imageStore(boid_out, ivec2(i, 1), vec4(v, V.w));
	imageStore(boid_out, ivec2(i, 2), vec4(lifeI, lifeH, lifeAge, lifeSpan));
}

==compute spark 8192==
layout(local_size_x = 256) in;
layout(r32f, binding = ctl_UNIT) uniform readonly highp image2D ctl;
layout(r32f, binding = spk_UNIT) uniform highp image2D spk;
layout(r32f, binding = ldr_UNIT) uniform readonly highp image2D ldr;
#define SPK_FROM_LEADERS 0
#define SPK_RATE    3000.0
#define SPK_INHERIT 0.50
#define SPK_KICK    0.25
#define SPK_BIRTH_R 0.05
#define SPK_DRAG    0.50
#define SPK_GRAV    1.20
#define SPK_DOWN    vec3(0.0, 0.0, 1.0)
#define SPK_LIFE0   1.0
#define SPK_LIFE1   2.0
#define SPK_QUIET   0.25
#define SPK_FLUTTER 0.20
float Cs(int k) { return imageLoad(ctl, ivec2(k, 0)).r; }
ivec2 spkA(int n, int row) { return ivec2(4 * (n % SPK_TEXW), 2 * (n / SPK_TEXW) + row); }
vec4 SL(int row, int n) { ivec2 b = spkA(n, row); return vec4(imageLoad(spk, b).r, imageLoad(spk, b + ivec2(1, 0)).r, imageLoad(spk, b + ivec2(2, 0)).r, imageLoad(spk, b + ivec2(3, 0)).r); }
void SS(int row, int n, vec4 v) { ivec2 b = spkA(n, row); imageStore(spk, b, vec4(v.x)); imageStore(spk, b + ivec2(1, 0), vec4(v.y)); imageStore(spk, b + ivec2(2, 0), vec4(v.z)); imageStore(spk, b + ivec2(3, 0), vec4(v.w)); }
void main() {
	int n = int(gl_GlobalInvocationID.x);
	if (n >= SPK_N) return;
	float dt = Cs(C_DT);
	vec4 P = SL(0, n), V = SL(1, n);
	if (V.w <= 0.0) { P.w = 1e9; V.w = 1.0; }
	float headF = mod(time * SPK_RATE, float(SPK_N));
	float rel = mod(float(n) - headF + float(SPK_N), float(SPK_N));
	if (rel < SPK_RATE * dt) {
		uint s = pcg(uint(n) * 2654435761u + uint(time * 1000.0) * 97u);
		float L = clamp(Cs(C_LOUD) / max(Cs(C_LMAX), 0.05), 0.0, 1.0);
		if (rnd(s) > mix(SPK_QUIET, 1.0, L)) { SS(0, n, vec4(P.xyz, 1e9)); SS(1, n, V); return; }
#if SPK_FROM_LEADERS
		int f = rnd(s) < 0.5 ? 0 : 1;
		int m = int(rnd(s) * float(nLeaders(Cs(C_SPLV)))) ;
		vec3 Lp = vec3(imageLoad(ldr, ivec2(4 * m, 2 * f)).r, imageLoad(ldr, ivec2(4 * m + 1, 2 * f)).r, imageLoad(ldr, ivec2(4 * m + 2, 2 * f)).r);
		vec3 Lv = vec3(imageLoad(ldr, ivec2(4 * m, 2 * f + 1)).r, imageLoad(ldr, ivec2(4 * m + 1, 2 * f + 1)).r, imageLoad(ldr, ivec2(4 * m + 2, 2 * f + 1)).r);
#else
		int e = int(rnd(s) * float(N_ALL)) % N_ALL;
		if (e % N_PER == 0) e += 1;
		vec3 Lp = texelFetch(boid, ivec2(e, 0), 0).xyz, Lv = texelFetch(boid, ivec2(e, 1), 0).xyz;
#endif
		vec3 v = Lv * SPK_INHERIT + rndDir(s) * SPK_KICK * rnd(s);
		float life = mix(SPK_LIFE0, SPK_LIFE1, rnd(s));
		vec3 sp = Lp + rndDir(s) * SPK_BIRTH_R * rnd(s);
		SS(0, n, vec4(sp, 0.0));
		SS(1, n, vec4(v, life));
		return;
	}
	if (P.w >= V.w) { SS(0, n, P); SS(1, n, V); return; }
	vec3 v = V.xyz * exp(-SPK_DRAG * dt) + SPK_DOWN * (SPK_GRAV * dt);
	float fph = float(n) * 0.7548 + P.w * 3.0;
	v += SPK_FLUTTER * dt * vec3(sin(fph * 2.3 + time), 0.5 * sin(fph * 1.7 + 1.9 * time), cos(fph * 1.9 + 1.3 * time));
	vec3 nP = P.xyz + v * dt;
	float edge = resolution.x / max(resolution.y, 1.0);
	nP = clamp(nP, vec3(-boxXW(edge), -boxYW(), BOX_ZN), vec3(boxXW(edge), boxYW(), BOX_ZB));
	SS(0, n, vec4(nP, P.w + dt));
	SS(1, n, vec4(v, V.w));
}

==compute decay glowI==
layout(local_size_x = 16, local_size_y = 16) in;
layout(r32ui, binding = glowI_UNIT) uniform highp uimage2D glowI;
layout(r32ui, binding = glowH_UNIT) uniform highp uimage2D glowH;
layout(r32f, binding = ctl_UNIT) uniform readonly highp image2D ctl;
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(p, imageSize(glowI)))) return;
	float dt = imageLoad(ctl, ivec2(C_DT, 0)).r;
	float kI = exp(-dt / PERSIST_TAU_I), kH = exp(-dt / PERSIST_TAU_H);
	uint I = imageLoad(glowI, p).r, H = imageLoad(glowH, p).r;
	if ((I | H) == 0u) return;
	imageStore(glowI, p, uvec4(uint(float(I) * kI)));
	imageStore(glowH, p, uvec4(uint(float(H) * kH)));
}

==compute dart 900==
layout(local_size_x = 64) in;
layout(r32f, binding = ctl_UNIT) uniform readonly highp image2D ctl;
layout(r32ui, binding = glowI_UNIT) uniform highp uimage2D glowI;
layout(r32ui, binding = glowH_UNIT) uniform highp uimage2D glowH;
#define DT_L         2.3
#define DT_HW        0.32
#define DT_GLOW      0.55
#define DT_GLOW_W    1.10
#define DT_EMIT      0.70
#define DT_I         3.4
#define TR_EMB_W     0.0035
#define TR_COOL_SPREAD 0.30
#define TR_FLARE_V0  0.35
#define TR_FLARE_V1  1.80
#define TR_BASE      0.47
#define TR_BAND0     0.50
#define TR_FLARE_H   0.42
#define TR_BREATH    0.12
#define TR_FLICK     0.04
#define TR_FLARE_I   0.80
float Cd(int k) { return imageLoad(ctl, ivec2(k, 0)).r; }
float emberHeat0(int i, float vw, float fastE) {
	float seed = float(i) + vw * 1000.0;
	float r1 = fract(seed * 0.6180339), r2 = fract(seed * 0.7548777), r3 = fract(seed * 0.5698403);
	float base = TR_BASE - TR_COOL_SPREAD * (fract(seed * 0.3819660) - 0.35) * 2.0;
	float breath = TR_BREATH * sin(time * 6.2831853 * (0.35 + 0.5 * r1) + 6.2831853 * r2);
	float fSlow = sin(time * 6.2831853 * (3.0 + 4.0 * r3) + 40.0 * r1) * sin(time * 6.2831853 * (1.7 + 2.3 * r1) + 17.0 * r2);
	float fMid  = sin(time * 6.2831853 * (8.0 + 7.0 * r1) + 23.0 * r2) * sin(time * 6.2831853 * (4.5 + 5.0 * r3) + 9.0 * r1);
	float fFast = sin(time * 6.2831853 * (18.0 + 16.0 * r2) + 31.0 * r3) * sin(time * 6.2831853 * (11.0 + 9.0 * r1) + 13.0 * r2);
	float flick = TR_FLICK * mix(1.0, 0.4, fastE) * fSlow + TR_FLICK * 1.8 * smoothstep(0.0, 0.6, fastE) * fMid
	            + TR_FLICK * 3.2 * smoothstep(0.25, 1.0, fastE) * fFast;
	float h = clamp(base + breath + flick, 0.0, 1.0);
	h = mix(h * 1.2, TR_BAND0 + 0.22 * h, smoothstep(0.15, 0.35, h));
	return clamp(h + TR_FLARE_H * fastE, 0.0, 1.0);
}
void main() {
	int i = int(gl_GlobalInvocationID.x);
	if (i >= N_ALL || i % N_PER == 0) return;
	vec4 P = texelFetch(boid, ivec2(i, 0), 0), V = texelFetch(boid, ivec2(i, 1), 0);
	vec4 Lf = texelFetch(boid, ivec2(i, 2), 0);
	ivec2 canvas = imageSize(glowI);
	float emit = min(Cd(C_DT) * EMIT_REF_FPS, 2.0);
	float H = float(canvas.y);
	vec3 a = project(P.xyz), a2 = project(P.xyz + V.xyz * 0.03);
	vec2 pa = (a.xy * H + vec2(canvas)) * 0.5;
	vec2 sv = (a2.xy - a.xy) * H * 0.5;
	float spd = length(sv);
	vec2 dir = spd > 1e-3 ? sv / spd : vec2(0.0, 1.0);
	float fastE = smoothstep(TR_FLARE_V0, TR_FLARE_V1, length(V.xyz));
	float w = TR_EMB_W * a.z * H * 0.5;
	float h1 = fract(float(i) * 0.6180339), h2 = fract(float(i) * 0.7548777), h3 = fract(float(i) * 0.5698403);
	float Lc = max(DT_L * w * 0.75, 1.7) * mix(0.8, 1.25, h1);
	float Wm = max(DT_HW * w * 1.3, 0.55) * mix(0.80, 1.25, h2);
	float wob = 0.12 * sin(time * (5.0 + 9.0 * h3) + 17.0 * h1);
	dir = vec2(dir.x * cos(wob) - dir.y * sin(wob), dir.x * sin(wob) + dir.y * cos(wob));
	vec2 nrm = vec2(-dir.y, dir.x);
	float curv = (h3 - 0.5) * 0.42;
	float heat = clamp(emberHeat0(i, V.w, fastE) * Lf.y, 0.0, 1.0);
	float glowHeat = clamp(heat * 0.92, 0.0, 1.0);
	float wgMax = max(DT_GLOW_W * Wm * 1.35, 1.2);
	int Rr = int(ceil(0.5 * Lc + 3.0 * wgMax)) + 1;
	ivec2 c = ivec2(floor(pa));
	for (int y = -Rr; y <= Rr; y++) for (int x = -Rr; x <= Rr; x++) {
		ivec2 q = c + ivec2(x, y);
		if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, canvas))) continue;
		vec2 dp = vec2(q) + 0.5 - pa;
		float sx = dot(dp, dir);
		float t = (sx + 0.50 * Lc) / Lc;
		float tc = clamp(t, 0.0, 1.0);
		float vv = dot(dp, nrm) - curv * (sx * sx) / max(Lc, 1.0);
		float wd = Wm * 1.35 * sqrt(max(0.0, 1.0 - (2.0 * tc - 1.0) * (2.0 * tc - 1.0)));
		float sg = max(0.55 * wd, 0.40);
		float core = exp(-(vv / sg) * (vv / sg));
		float sheath = 0.22 * exp(-(vv / (2.1 * sg)) * (vv / (2.1 * sg)));
		float along = smoothstep(0.0, 0.40, tc) * smoothstep(1.0, 0.60, tc);
		float body = (t < 0.0 || t > 1.0) ? 0.0 : 1.0;
		float hb = clamp(heat - 0.04 * (sheath / max(core + sheath, 1e-4)), 0.0, 1.0);
		float Ib = DT_EMIT * DT_I * 0.95 * (core + sheath) * along * (1.0 + TR_FLARE_I * fastE) * body;
		float wg = max(DT_GLOW_W * max(wd, 0.7 * Wm), 1.2);
		float dEnd = t < 0.0 ? -t * Lc : (t > 1.0 ? (t - 1.0) * Lc : 0.0);
		float dOut = length(vec2(max(abs(vv) - wd, 0.0), dEnd));
		float Ig = DT_EMIT * DT_I * 0.95 * DT_GLOW * exp(-(dOut / wg) * (dOut / wg)) * max(along, 0.45) * (1.0 + 0.5 * TR_FLARE_I * fastE);
		float Igs = Ig * (1.0 - 0.6 * core * body);
		float I = (Ib + Igs) * emit * Lf.x;
		uint add = uint(I * FIX + 0.5);
		if (add == 0u) continue;
		float hp = (Ib * hb + Igs * glowHeat) / max(Ib + Igs, 1e-5);
		imageAtomicAdd(glowI, q, add);
		imageAtomicAdd(glowH, q, uint(I * hp * FIX + 0.5));
	}
}

==compute splatspk 8192==
layout(local_size_x = 64) in;
layout(r32f, binding = ctl_UNIT) uniform readonly highp image2D ctl;
layout(r32f, binding = spk_UNIT) uniform readonly highp image2D spk;
layout(r32ui, binding = glowI_UNIT) uniform highp uimage2D glowI;
layout(r32ui, binding = glowH_UNIT) uniform highp uimage2D glowH;
#define SPK_GAIN   1.90
#define SPK_SHARE  1.145
#define SPK_HMAX   0.80
#define SPK_BORN0  0.50
#define SPK_BORN1  0.70
#define SPK_COOL_EX 0.9
#define SPK_STAR   0.0
#define SPK_BREATH 0.09
#define SPK_FLICK_H 0.05
#define SPK_EMIT   0.75
#define SPK_GLOW   0.55
#define SPK_GLOW_R 2.0
#define SPK_GLOW_COOL 0.92
#define SPK_STAR_L 3.45
#define SPK_STAR_MAX 3.75
#define SPK_STAR_GAIN 1.7
#define SPK_POINT_R 0.55
#define SPK_POINT_MAX 0.9
#define SPK_W      0.0035
#define SPK_HOLD   0.55
#define SPK_FADE   1.0
#define SPK_FLICK  0.25
#define SPK_ASH    0.35
float Cp(int k) { return imageLoad(ctl, ivec2(k, 0)).r; }
ivec2 spkA(int n, int row) { return ivec2(4 * (n % SPK_TEXW), 2 * (n / SPK_TEXW) + row); }
vec4 SL(int row, int n) { ivec2 b = spkA(n, row); return vec4(imageLoad(spk, b).r, imageLoad(spk, b + ivec2(1, 0)).r, imageLoad(spk, b + ivec2(2, 0)).r, imageLoad(spk, b + ivec2(3, 0)).r); }
void main() {
	int i = int(gl_GlobalInvocationID.x);
	if (i >= SPK_N) return;
	vec4 P = SL(0, i), V = SL(1, i);
	if (V.w <= 0.0 || P.w >= V.w) return;
	float x = P.w / V.w;
	float fade = x < SPK_HOLD ? 1.0 - 0.15 * x / SPK_HOLD : 0.85 * pow(1.0 - (x - SPK_HOLD) / (1.0 - SPK_HOLD), SPK_FADE);
	float h1 = fract(float(i) * 0.6180339), h2 = fract(float(i) * 0.7548777), h3 = fract(float(i) * 0.5698403);
	float fl = 1.0 - SPK_FLICK * (0.5 + 0.5 * sin(time * (5.0 + 4.0 * h1) + float(i)));
	float ash = smoothstep(1.0 - SPK_ASH, 1.0, x);
	float size = mix(0.7, 1.5, h2 * h2 * h2);
	float emit = min(Cp(C_DT) * EMIT_REF_FPS, 2.0);
	float b = SPK_GAIN * SPK_SHARE * fade * fl * mix(0.8, 1.0, size / 1.5);
	vec3 a = project(P.xyz);
	ivec2 canvas = imageSize(glowI);
	float H = float(canvas.y);
	vec2 pa = (a.xy * H + vec2(canvas)) * 0.5;
	float w = max(SPK_W * a.z * H * 0.5 * size * (1.0 - 0.35 * ash), 0.6);
	float fastS = smoothstep(0.12, 1.0, length(V.xyz));
	float born = mix(SPK_BORN0, SPK_BORN1, h1);
	float coolEx = SPK_COOL_EX;
	float hBase = mix(born, 0.04, pow(x, coolEx));
	float sBreath = SPK_BREATH * sin(time * 6.2831853 * (0.6 + 0.8 * h2) + 6.2831853 * h3);
	float sSlow = sin(time * 6.2831853 * (3.0 + 4.0 * h3) + 40.0 * h1) * sin(time * 6.2831853 * (1.7 + 2.3 * h1) + 17.0 * h2);
	float sMid  = sin(time * 6.2831853 * (8.0 + 7.0 * h1) + 23.0 * h2) * sin(time * 6.2831853 * (4.5 + 5.0 * h3) + 9.0 * h1);
	float sFast = sin(time * 6.2831853 * (18.0 + 16.0 * h2) + 31.0 * h3) * sin(time * 6.2831853 * (11.0 + 9.0 * h1) + 13.0 * h2);
	float sFlick = SPK_FLICK_H * mix(1.0, 0.4, fastS) * sSlow + SPK_FLICK_H * 1.8 * smoothstep(0.0, 0.6, fastS) * sMid
	             + SPK_FLICK_H * 3.2 * smoothstep(0.25, 1.0, fastS) * sFast;
	float hp = clamp(hBase + sBreath + sFlick + 0.22 * fastS * (1.0 - x), 0.0, SPK_HMAX);
	float pr0 = clamp(w * SPK_POINT_R, 0.5, SPK_POINT_MAX);
	float pr = max(pr0, 0.8);
	float Lb = pr * SPK_STAR_L * (0.55 + 0.45 * min(b, 2.2));
	float ang0 = 6.2831853 * fract(float(i) * 0.1234567) + 0.25 * time * (h1 - 0.5);
	float sw = max(0.5 * pr, 0.40);
	vec2 dirs[3];
	float Lp[3], Lm[3], Wt[3];
	int nl = fract(float(i) * 0.7071068 + 0.31) < 0.5 ? 2 : 3;
	for (int j = 0; j < 3; j++) {
		float an = ang0 + (j == 0 ? 0.0 : (0.55 + 1.9 * fract(float(i) * (0.29 + 0.17 * float(j)) + 0.5 * float(j))) * float(j == 1 ? 1 : -1));
		dirs[j] = vec2(cos(an), sin(an));
		float r1 = fract(float(i) * (0.37 + 0.113 * float(j)) + 0.131 * float(j)), r2 = fract(float(i) * (0.61 + 0.071 * float(j)) + 0.217 * float(j));
		float r3 = fract(float(i) * (0.83 + 0.057 * float(j)) + 0.401 * float(j));
		float st = j == 0 ? 1.7 : 1.0;
		Lp[j] = min(Lb * st * (0.35 + 1.1 * r1), SPK_STAR_MAX);
		Lm[j] = min(Lb * st * (0.35 + 1.1 * r2), SPK_STAR_MAX);
		Wt[j] = j < nl ? SPK_STAR * (0.55 + 0.45 * r3) * (1.0 - 0.25 * (0.3 + 0.7 * fastS) + 0.25 * (0.3 + 0.7 * fastS) * sin(time * (9.0 + 14.0 * r1) + 20.0 * r2)) : 0.0;
	}
	float rg = max(SPK_GLOW_R * pr, 1.2);
	int Rp = int(ceil(max(max(SPK_STAR_MAX, pr * 3.5), 3.0 * rg)) + 1.0);
	ivec2 cp = ivec2(floor(pa));
	for (int y = -Rp; y <= Rp; y++) for (int x2 = -Rp; x2 <= Rp; x2++) {
		ivec2 q = cp + ivec2(x2, y);
		if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, canvas))) continue;
		vec2 dp = vec2(q) + 0.5 - pa;
		float d = length(dp);
		float I = 0.0;
		for (int j = 0; j < 3; j++) {
			float par = dot(dp, dirs[j]), perp = dp.x * -dirs[j].y + dp.y * dirs[j].x;
			float Lsp = par >= 0.0 ? Lp[j] : Lm[j];
			float ap = abs(par);
			if (ap >= Lsp) continue;
			float ff = 1.0 - ap / Lsp;
			I += SPK_EMIT * b * SPK_STAR_GAIN * Wt[j] * ff * ff * exp(-(perp / sw) * (perp / sw)) * smoothstep(0.0, pr, ap);
		}
		float Ih = SPK_EMIT * b * SPK_GLOW * exp(-(d / rg) * (d / rg));
		float Itot = (I + Ih) * emit;
		uint add = uint(Itot * FIX + 0.5);
		if (add == 0u) continue;
		imageAtomicAdd(glowI, q, add);
		imageAtomicAdd(glowH, q, uint((I * hp + Ih * hp * SPK_GLOW_COOL) * emit * FIX + 0.5));
	}
}

==compute rad rad==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = rad_UNIT) uniform writeonly highp image2D radImg;
layout(r32ui, binding = glowI_UNIT) uniform readonly highp uimage2D glowI;
layout(r32ui, binding = glowH_UNIT) uniform readonly highp uimage2D glowH;
#define EMB_GAIN 1.38
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(p, imageSize(radImg)))) return;
	imageStore(radImg, p, vec4(emberRadiance(imageLoad(glowI, p).r, imageLoad(glowH, p).r, EMB_GAIN), 1.0));
}

==compute blmdown blmA==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = blmA_UNIT) uniform writeonly highp image2D outImg;
uniform highp sampler2D rad;
void main() {
	ivec2 o = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(o, imageSize(outImg)))) return;
	ivec2 sm = textureSize(rad, 0) - 1;
	vec3 sum = vec3(0.0);
	for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) sum += min(texelFetch(rad, min(o * 2 + ivec2(i, j), sm), 0).rgb, vec3(BL_CLAMP));
	imageStore(outImg, o, vec4(sum * 0.25, 1.0));
}
==compute blmh blmT==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = blmT_UNIT) uniform writeonly highp image2D outImg;
uniform highp sampler2D blmA;
#define SIG 1.75
void main() {
	ivec2 o = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(outImg);
	if (any(greaterThanEqual(o, sz))) return;
	int Rk = int(ceil(3.0 * SIG));
	vec3 acc = vec3(0.0); float ws = 0.0;
	for (int k = -Rk; k <= Rk; k++) { float wt = exp(-float(k * k) / (2.0 * SIG * SIG)); acc += texelFetch(blmA, clamp(o + ivec2(k, 0), ivec2(0), sz - 1), 0).rgb * wt; ws += wt; }
	imageStore(outImg, o, vec4(acc / ws, 1.0));
}
==compute blmv blmA==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = blmA_UNIT) uniform writeonly highp image2D outImg;
uniform highp sampler2D blmT;
#define SIG 1.75
void main() {
	ivec2 o = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(outImg);
	if (any(greaterThanEqual(o, sz))) return;
	int Rk = int(ceil(3.0 * SIG));
	vec3 acc = vec3(0.0); float ws = 0.0;
	for (int k = -Rk; k <= Rk; k++) { float wt = exp(-float(k * k) / (2.0 * SIG * SIG)); acc += texelFetch(blmT, clamp(o + ivec2(0, k), ivec2(0), sz - 1), 0).rgb * wt; ws += wt; }
	imageStore(outImg, o, vec4(acc / ws, 1.0));
}
==compute blmdown2 blmB==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = blmB_UNIT) uniform writeonly highp image2D outImg;
uniform highp sampler2D blmA;
void main() {
	ivec2 o = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(o, imageSize(outImg)))) return;
	ivec2 sm = textureSize(blmA, 0) - 1;
	vec3 sum = vec3(0.0);
	for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) sum += texelFetch(blmA, min(o * 2 + ivec2(i, j), sm), 0).rgb;
	imageStore(outImg, o, vec4(sum * 0.25, 1.0));
}
==compute blmh2 blmU==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = blmU_UNIT) uniform writeonly highp image2D outImg;
uniform highp sampler2D blmB;
#define SIG 2.5
void main() {
	ivec2 o = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(outImg);
	if (any(greaterThanEqual(o, sz))) return;
	int Rk = int(ceil(3.0 * SIG));
	vec3 acc = vec3(0.0); float ws = 0.0;
	for (int k = -Rk; k <= Rk; k++) { float wt = exp(-float(k * k) / (2.0 * SIG * SIG)); acc += texelFetch(blmB, clamp(o + ivec2(k, 0), ivec2(0), sz - 1), 0).rgb * wt; ws += wt; }
	imageStore(outImg, o, vec4(acc / ws, 1.0));
}
==compute blmv2 blmB==
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = blmB_UNIT) uniform writeonly highp image2D outImg;
uniform highp sampler2D blmU;
#define SIG 2.5
void main() {
	ivec2 o = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(outImg);
	if (any(greaterThanEqual(o, sz))) return;
	int Rk = int(ceil(3.0 * SIG));
	vec3 acc = vec3(0.0); float ws = 0.0;
	for (int k = -Rk; k <= Rk; k++) { float wt = exp(-float(k * k) / (2.0 * SIG * SIG)); acc += texelFetch(blmU, clamp(o + ivec2(0, k), ivec2(0), sz - 1), 0).rgb * wt; ws += wt; }
	imageStore(outImg, o, vec4(acc / ws, 1.0));
}

==image==
uniform vec2 resolution;
uniform highp sampler2D rad;
uniform highp sampler2D blmA;
uniform highp sampler2D blmB;
uniform sampler2D dio_backC;
uniform sampler2D dio_wlC;
uniform sampler2D dio_wrC;
uniform sampler2D dio_topC;
uniform sampler2D dio_botC;
uniform highp sampler2D ctl;
out vec4 fragColor;
#define ROOM_SKIN    4
#define TANK_STEPS   4
#define TANK_R0      0.15
#define TANK_REFL    3.0
#define TANK_SOFT    0.35
#define TANK_GLASS   vec3(0.0)
#define TANK_BACK    0.45
#define TANK_SWARM   1.5
#define TANK_EDGE    vec3(0.0)
#define SPK_FILL     1.0
#define SPK_DIM      0.55
#define SPK_DOME_R   0.80
#define SPK_DOME_SOFT 0.45
#define CHIRP_COL    vec3(0.80, 0.0, 0.48)
#define CHIRP_HOT    vec3(1.00, 0.06, 0.12)
#define CHIRP_SCALE  0.6
#define CHIRP_ROOM_R 2.4
#define CHIRP_CEN    vec3(0.0, 0.0, 0.0)
#define CHIRP_DIM    0.55
#define CHIRP_MOUTH  0.30
#define CHIRP_SWARM  1.2
#define GB_STEP    0.25
#define GB_W_MINOR 0.30
#define GB_W_MAJOR 1.20
#define GB_GLASS   vec3(0.0, 0.020, 0.045)
#define GB_LINE    vec3(0.06, 0.78, 0.66)
#define GB_MOUTH   0.45
#define GB_BASE    0.55
#define GB_SWARM   5.0
#define BLOOM_A   0.55
#define BLOOM_B   0.85
#define TM_HUE    0.75
#define GLS_BASE  0.010
#define GLS_BACK  0.10
#define GLS_BACK_COL vec3(0.55, 0.45, 1.0)
#define GLS_SWARM 2.4
#define GLS_SIDE  0.85
#define GLS_TOP   0.70
vec3 paneLit(vec3 c, vec3 light) { return pow(c, vec3(2.2)) * (GLS_BASE + light); }
vec3 chirp3(vec3 P, vec3 cen, float loud, float pulse) {
	float lt = mod(time, 200.0);
	float t = 5000.0 - lt * 3.0 + sin(time / 3.0) * 5.0;
	float scale = CHIRP_SCALE * (1.0 - 0.045 * pulse - 0.12 * loud);
	float d = length(P - cen) / CHIRP_ROOM_R * 0.83 * scale, e = d * d * d;
	float strength = 0.5 + 0.5 * sin(e * 100.0);
	float height = 0.5 + 0.5 * sin(t * strength);
	float reach = 0.125 * (1.0 + 0.25 * loud + 0.35 * pulse);
	float alpha = clamp(1.0 - e / reach - (1.0 - height) * 0.014, 0.0, 1.0);
	vec3 col = mix(CHIRP_COL, CHIRP_HOT, clamp(0.45 * loud + 0.65 * pulse, 0.0, 1.0));
	float glow = 1.0 + 0.05 * loud + 0.5 * pulse;
	return col * height * alpha * glow;
}
vec3 glass(vec2 uv, vec3 swarmLight) {
	vec3 ro = camPos();
	vec3 rd = normalize(vec3(uv * (CAM_D + BOX_ZN) / FOCAL, BOX_ZN) - ro);
	float edge = resolution.x / resolution.y;
	vec3 bmin = vec3(-boxXW(edge), -boxYW(), BOX_ZN), bmax = vec3(boxXW(edge), boxYW(), BOX_ZB);
	vec3 inv = 1.0 / (rd + vec3(1e-7));
	vec3 t0 = (bmin - ro) * inv, t1 = (bmax - ro) * inv;
	vec3 tmx = max(t0, t1);
	float tF = min(min(tmx.x, tmx.y), tmx.z);
	vec3 P = ro + rd * tF;
	vec3 q = (P - bmin) / (bmax - bmin);
#if ROOM_SKIN == 0
	float loudRaw = texelFetch(ctl, ivec2(0, 0), 0).r, lmax = max(texelFetch(ctl, ivec2(1, 0), 0).r, 0.05);
	float loud = clamp(loudRaw / lmax, 0.0, 1.0), pulse = texelFetch(ctl, ivec2(46, 0), 0).r;
	vec3 cen = CHIRP_CEN;
	float depthDim = mix(CHIRP_MOUTH, 1.0, clamp((P.z - BOX_ZN) / (BOX_ZB - BOX_ZN), 0.0, 1.0));
	if (tF == tmx.x || tF == tmx.y) depthDim *= 0.85;
	return chirp3(P, cen, loud, pulse) * CHIRP_DIM * depthDim * (1.0 + CHIRP_SWARM * dot(swarmLight, vec3(0.333)));
#elif ROOM_SKIN == 4
	vec3 n = (tF == tmx.z) ? vec3(0.0, 0.0, -1.0)
	       : (tF == tmx.x) ? vec3(-sign(P.x), 0.0, 0.0) : vec3(0.0, -sign(P.y), 0.0);
	vec3 rr = reflect(rd, n);
	vec3 invr = 1.0 / (rr + vec3(1e-7));
	vec3 tmr = max((bmin - P) * invr, (bmax - P) * invr);
	float sMax = min(min(tmr.x, tmr.y), tmr.z);
	vec3 refl = vec3(0.0);
	for (int i = 0; i < TANK_STEPS; i++) {
		vec3 Q = P + rr * (sMax * (float(i) + 0.5) / float(TANK_STEPS));
		vec2 s = project(Q).xy;
		vec2 tuv = vec2(s.x / edge, s.y) * 0.5 + 0.5;
		float inb = step(0.0, tuv.x) * step(tuv.x, 1.0) * step(0.0, tuv.y) * step(tuv.y, 1.0);
		tuv = clamp(tuv, 0.0, 1.0);
		refl += mix(texture(rad, tuv).rgb, texture(blmA, tuv).rgb, TANK_SOFT) * inb;
	}
	refl /= float(TANK_STEPS);
	float cosv = abs(dot(rd, n));
	float fres = TANK_R0 + (1.0 - TANK_R0) * pow(1.0 - cosv, 5.0);
	float depth = clamp((P.z - BOX_ZN) / (BOX_ZB - BOX_ZN), 0.0, 1.0);
	float px = tF * 2.0 / resolution.y / max(length(vec3(uv, FOCAL)), 1e-3);
	float edgeD = (tF == tmx.z) ? min(boxXW(edge) - abs(P.x), boxYW() - abs(P.y))
	            : (tF == tmx.x) ? min(boxYW() - abs(P.y), BOX_ZB - P.z)
	                            : min(boxXW(edge) - abs(P.x), BOX_ZB - P.z);
	float edgeL = 1.0 - smoothstep(0.0, 2.0 * px, edgeD - 1.0 * px);
	vec3 glassC = TANK_GLASS * mix(1.0, TANK_BACK, depth) * (1.0 + TANK_SWARM * dot(swarmLight, vec3(0.333)));
	return glassC + refl * fres * TANK_REFL + TANK_EDGE * edgeL;
#elif ROOM_SKIN == 2 || ROOM_SKIN == 3
#if ROOM_SKIN == 3
	if (tF == tmx.z) {
		float loudRaw = texelFetch(ctl, ivec2(0, 0), 0).r, lmax = max(texelFetch(ctl, ivec2(1, 0), 0).r, 0.05);
		float loud = clamp(loudRaw / lmax, 0.0, 1.0), pulse = texelFetch(ctl, ivec2(46, 0), 0).r;
		vec3 Pb = vec3(P.xy - CHIRP_CEN.xy, 0.0) * (CHIRP_ROOM_R / (SPK_FILL * boxYW()));
		float swarm = dot(swarmLight, vec3(0.333));
		float dTop = length(vec2(P.x, P.y - boxYW())), dBot = length(vec2(P.x, P.y + boxYW()));
		float vig = smoothstep(SPK_DOME_R - SPK_DOME_SOFT, SPK_DOME_R + SPK_DOME_SOFT, dTop)
		          * smoothstep(SPK_DOME_R - SPK_DOME_SOFT, SPK_DOME_R + SPK_DOME_SOFT, dBot);
		float px = tF * 2.0 / resolution.y / max(length(vec3(uv, FOCAL)), 1e-3);
		float rim = 1.0 - smoothstep(0.0, 2.0 * px, min(boxXW(edge) - abs(P.x), boxYW() - abs(P.y)) - 1.0 * px);
		return (GB_GLASS * (GB_BASE + GB_SWARM * swarm)
		     + chirp3(Pb, vec3(0.0), loud, pulse) * SPK_DIM * (1.0 + CHIRP_SWARM * swarm)) * vig
		     + GB_LINE * GB_W_MAJOR * rim * (GB_BASE + GB_SWARM * swarm);
	}
#endif
	vec2 pw = (tF == tmx.z) ? P.xy
	        : (tF == tmx.x) ? vec2(P.z * sign(P.x) + 11.0, P.y)
	                        : vec2(P.x, P.z * sign(P.y) + 23.0);
	float px = tF * 2.0 / resolution.y / max(length(vec3(uv, FOCAL)), 1e-3);
	vec2 g = abs(fract(pw / GB_STEP) - 0.5) * GB_STEP;
	vec2 G = abs(fract(pw / (4.0 * GB_STEP)) - 0.5) * 4.0 * GB_STEP;
	float minor = 1.0 - smoothstep(0.0, 1.5 * px, min(g.x, g.y) - 0.5 * px);
	float major = 1.0 - smoothstep(0.0, 2.0 * px, min(G.x, G.y) - 1.0 * px);
	vec3 grid = GB_GLASS + GB_LINE * (GB_W_MINOR * minor + GB_W_MAJOR * major);
	float dim = mix(GB_MOUTH, 1.0, clamp((P.z - BOX_ZN) / (BOX_ZB - BOX_ZN), 0.0, 1.0));
	if (tF == tmx.x || tF == tmx.y) dim *= 1.95;
	return grid * dim * (GB_BASE + GB_SWARM * dot(swarmLight, vec3(0.333)));
#else
	vec3 light = GLS_BACK * GLS_BACK_COL + GLS_SWARM * swarmLight;
	vec3 c;
	if (tF == tmx.z) {        c = texture(dio_backC, vec2(q.y, 1.0 - q.x)).rgb; }
	else if (tF == tmx.x) {
		light *= GLS_SIDE;
		if (rd.x < 0.0) c = texture(dio_wlC, vec2(q.y, 1.0 - q.z)).rgb;
		else            c = texture(dio_wrC, vec2(q.y, q.z)).rgb;
	} else {
		light *= GLS_TOP;
		if (rd.y > 0.0) c = texture(dio_topC, vec2(q.x, 1.0 - q.z)).rgb;
		else            c = texture(dio_botC, vec2(q.x, q.z)).rgb;
	}
	return paneLit(c, light);
#endif
}
void main() {
	vec2 uv = gl_FragCoord.xy / resolution;
	vec2 uvc = (2.0 * gl_FragCoord.xy - resolution) / resolution.y;
	vec3 bA = texture(blmA, uv).rgb, bB = texture(blmB, uv).rgb;
	vec3 r = texture(rad, uv).rgb + BLOOM_A * bA + BLOOM_B * bB;
	vec3 g = glass(uvc, bB);
	vec3 c = r + g;
	float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
	vec3 perCh = 1.0 - exp(-c);
	vec3 keepHue = clamp(c * (1.0 - exp(-lum)) / max(lum, 1e-4), 0.0, 1.0);
	fragColor = vec4(mix(perCh, keepHue, TM_HUE), 1.0);
}
