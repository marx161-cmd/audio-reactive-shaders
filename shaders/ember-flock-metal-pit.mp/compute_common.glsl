#define N_PER    2304
#define N_ALL    (2 * N_PER)

#define TRAINS       0
#define LEAD_EVERY   8
#define TRAIN_K      0.85
#define TRAIN_W      0.035
#define HIST_N       32
#define AIR_H        2.8
#define STIR_N_F     64
#define STIR_FIX_F   1000.0
#define AIR_HOT0     0.12
#define AIR_HOT1     0.80
float airHot(sampler3D air, vec3 x) {
    vec3 uvw = x / (2.0 * AIR_H) + 0.5;
    if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) return 0.0;
    return smoothstep(AIR_HOT0, AIR_HOT1, texture(air, uvw).y);
}
#define N_LEAD   48
#define N_LMAX   512
#define LEAD_EMBERS 1
#define LEAD_STRIDE (N_PER / N_LEAD)
#define LEAD_KINEMATIC 1
#define FORMATIONS 1
#define BOUND_R  1.30

#define CAM_D    3.0
#define FOCAL    1.9
#define CAM_PITCH0 0.0
#define EYE_Y      0.00
#define EYE_ZN    -0.25
#define CAM_TY     0.0
#define RS_E          0.07033
#define HOLE_R_H      (2.0 * RS_E)
#define HOLE_REPEL_H  0.16
#define HOLE_REPEL_K  0.0
#define HOLE_SLOT_R   0.0
#define HOLE_LEAD_R   0.0
#define DISK_DIR     (-1.0)
#define DISK_V       0.90
#define DISK_R0      0.40
#define DISK_K       0.0
#define DISK_YK      2.0
#define DISK_RIN     0.16
#define DISK_ROUT0   0.70
#define DISK_ROUT1   1.10
#define DISK_TOR_R   1.10
vec3 diskVel(vec3 x) {
    float rho = length(x.xz);
    vec2 tang = rho > 1e-4 ? vec2(-x.z, x.x) / rho : vec2(0.0);
    float vt = DISK_V * sqrt(DISK_R0 / max(rho, 0.12));
    return vec3(tang * (vt * DISK_DIR), -DISK_YK * x.y);
}
float diskEnv(vec3 x) {
    float r = length(x);
    return smoothstep(DISK_RIN, DISK_RIN + 0.12, r) * (1.0 - smoothstep(DISK_ROUT0, DISK_ROUT1, r));
}
#define HSW_V        0.30
#define HSW_R0       0.80
#define HSW_K        0.0
#define SWARM_HSW_K  0.0
vec3 holeSwirlVel(vec3 x) {
    float rho = length(x.xz);
    vec2 tang = rho > 1e-4 ? vec2(-x.z, x.x) / rho : vec2(0.0);
    return vec3(tang * (HSW_V * sqrt(HSW_R0 / max(rho, 0.3)) * DISK_DIR), 0.0);
}
float holeSwirlEnv(vec3 x) {
    float r = length(x);
    return smoothstep(0.30, 0.60, r) * (1.0 - smoothstep(1.8, 2.3, r));
}
#define HOLE_SPRING_R 0.85
#define HOLE_SPRING_K 0.0
#define FLOOR_Y   (-1.00)
#define TOR_DIR      DISK_DIR
#define TOR_INVERT   0
#define TOR_V        0.26
#define TOR_R0       0.30
#define TOR_R1       0.95
#define TOR_OUT      0.10
#define TOR_UP       0.60
#define TOR_K        1.1
#define TOR_Y1       2.0
#define ORBIT_DIR    TOR_DIR
#define SWARM_TOR_K  2.2
float torEnv(vec3 x) { return smoothstep(FLOOR_Y, FLOOR_Y + 0.35, x.y) * (1.0 - smoothstep(TOR_Y1 - 0.7, TOR_Y1, x.y)); }
float torCore(vec3 x) {
    float yN = clamp((x.y - FLOOR_Y) / (TOR_Y1 - FLOOR_Y), 0.0, 1.0);
    return mix(TOR_R0, TOR_R1, TOR_INVERT > 0 ? 1.0 - yN : yN);
}
vec3 torSwirl(vec3 x) {
    vec2 rel = x.xz;
    float r = length(rel);
    float q = r / max(torCore(x), 1e-3);
    float vt = TOR_V * (1.0 - exp(-q * q)) / max(q, 0.35);
    vec2 tang = r > 1e-4 ? vec2(-rel.y, rel.x) / r : vec2(0.0);
    return vec3(tang * (vt * TOR_DIR), 0.0) * torEnv(x);
}
vec3 torVel(vec3 x) {
    vec2 rel = x.xz;
    float r = length(rel);
    float q = r / max(torCore(x), 1e-3);
    float vt = TOR_V * (1.0 - exp(-q * q)) / max(q, 0.35);
    vec2 tang = r > 1e-4 ? vec2(-rel.y, rel.x) / r : vec2(0.0);
    vec2 outw = r > 1e-4 ? rel / r : vec2(0.0);
    return vec3(tang * (vt * TOR_DIR) + outw * TOR_OUT * (1.0 - exp(-q * q)) * exp(-0.25 * q * q), TOR_UP * exp(-0.6 * q * q));
}
uniform vec3 u_cam_look;
uniform float u_time;
#define CAM_SPIN_S 45.0
mat3 camRot() {
    float y = 0.0, p = CAM_PITCH0;
    mat3 ry = mat3(cos(y), 0.0, -sin(y), 0.0, 1.0, 0.0, sin(y), 0.0, cos(y));
    mat3 rx = mat3(1.0, 0.0, 0.0, 0.0, cos(p), sin(p), 0.0, -sin(p), cos(p));
    return rx * ry;
}
vec3 project(vec3 p, mat3 R) {
    vec3 c = R * (p - vec3(0.0, CAM_TY, 0.0));
    float s = FOCAL / max(c.z + CAM_D, 0.3);
    return vec3((c.xy - vec2(0.0, EYE_Y)) * s + vec2(0.0, EYE_Y) * FOCAL / (CAM_D + EYE_ZN), s);
}

uniform sampler3D u_lens0;
uniform sampler3D u_lens1;
#define LNS_ND    vec3(0.31949187, 0.93968197, 0.12215866)
#define LNS_E1    vec3(0.94677274, -0.32190273, 0.00000000)
#define LNS_E2    vec3(0.03932321, 0.11565649, -0.99251059)
#define LNS_F     vec3(0.91613207, -0.39796348, -0.04823990)
#define LNS_CAM   vec3(-39.07986549, 16.97611056, 2.05779160)
#define LNS_LR0   (-0.22314355)
#define LNS_LR1   4.09434456
#define LNS_FOCAL 1.8
#define LNS_RES   2410.0
#define LNS_GAIN2 0.6
vec3 lensToBake(vec3 p) {
    float al = 6.2831853 * u_time / CAM_SPIN_S, ca = cos(al), sa = sin(al);
    vec3 q = vec3(p.x * ca - p.z * sa, p.y, p.x * sa + p.z * ca) / RS_E;
    return q.x * LNS_E1 + q.y * LNS_ND - q.z * LNS_E2;
}
vec3 lensCamW() {
    return transpose(camRot()) * vec3(0.0, 0.0, -CAM_D) + vec3(0.0, CAM_TY, 0.0);
}
vec3 lensP(vec3 p, int side) {
    if (side != 0) return vec3(0.0, 0.0, -1.0);
    vec3 c = camRot() * (p - vec3(0.0, CAM_TY, 0.0));
    if (c.z + CAM_D < 0.3) return vec3(0.0, 0.0, -1.0);
    float sc = FOCAL / (c.z + CAM_D);
    return vec3((c.xy - vec2(0.0, EYE_Y)) * sc + vec2(0.0, EYE_Y) * FOCAL / (CAM_D + EYE_ZN), sc);
}

#define PATH_R   0.42
vec3 knotPath(float s) {
    float r = 2.0 + cos(3.0 * s);
    return PATH_R * vec3(r * cos(2.0 * s), -1.2 * sin(3.0 * s), r * sin(2.0 * s)) / 3.0;
}

uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
float rnd(inout uint s) { s = pcg(s); return float(s) * (1.0 / 4294967296.0); }
vec3 rndDir(inout uint s) {
    float z = 2.0 * rnd(s) - 1.0, a = 6.2831853 * rnd(s);
    float r = sqrt(max(0.0, 1.0 - z * z));
    return vec3(r * cos(a), r * sin(a), z);
}

#define FORM_R          0.32
#define FORM_DEMO_BEATS 8.0
#define FORM_MORPH_BEATS 2.0
#define LEAD_PERIOD     0.30
#define LEAD_DAMP       0.75
#define LEAD_PERIOD_TAIL_MUL 2.0
#define LEAD_DAMP_TAIL_MUL   1.15
#define LEAD_VMAX       2.5
#define LEAD_ACC       14.0
#define PRED_BACK       8.0
#define PRED_FWD        6.0
#define PRED_N          57
#define TURN_FIT        0.5
#define CZ_MIN          0.30
#define LEAD_FF         1.0
#define BAL_HOLD_BEATS  1.0
#define LEAD_SOFT       0.12
#define LEAD_SPREAD     0.035
#define MINI_ON     0
#define MINI_A      1.0
#define MINI_B      2.0
#define MINI_C      3.0
#define MINI_R      0.045
#define MINI_DEP    0.40
#define MINI_BEATS  4.0
#define MINI_SPIN   0.5
#define MINI_LAG    0.12
#define MINI_JIT    0.7
#define MINI_RND    0.40
vec3 miniFigure(int grp, int w, float beat, vec3 headV) {
    float lap = 6.2831853 * beat / max(MINI_BEATS, 0.25);
    float ph = lap + float(grp) * MINI_LAG
             + float(w) * (6.2831853 / float(max(LEAD_STRIDE - 1, 1)) + MINI_JIT);
    vec2 q = vec2(sin(MINI_A * ph), sin(MINI_B * ph));
    float ro = lap * MINI_SPIN;
    float c = cos(ro), s = sin(ro);
    q = mat2(c, -s, s, c) * q;
    vec3 right = transpose(camRot()) * vec3(1.0, 0.0, 0.0);
    vec3 up    = transpose(camRot()) * vec3(0.0, 1.0, 0.0);
    vec3 fw    = length(headV) > 1e-4 ? normalize(headV) : vec3(0.0, 0.0, 1.0);
    return (right * q.x + up * q.y + fw * (MINI_DEP * sin(MINI_C * ph))) * MINI_R;
}
#define BRAID_ON     1
#define BRAID_TWIST  1.0
#define BRAID_LEN    1.7
#define BRAID_X      0.45
vec3 braidFigure(int grp, int w, float beat, vec3 headV) {
    float nb = float(max(1, (LEAD_STRIDE - 1 + 2) / 3));
    int s = w % 3, b = w / 3;
    float step = beat / max(MINI_BEATS, 0.25);
    float ang  = 6.2831853 * (BRAID_TWIST * float(b) / 3.0 + float(s) / 3.0 + step) + float(grp) * MINI_LAG;
    float along = (nb > 1.0) ? 2.0 * float(b) / (nb - 1.0) - 1.0 : 0.0;
    float r = MINI_R * (1.0 + BRAID_X * sin(3.1415927 * (float(b) + 2.0 * step)));
    vec3 right = transpose(camRot()) * vec3(1.0, 0.0, 0.0);
    vec3 up    = transpose(camRot()) * vec3(0.0, 1.0, 0.0);
    vec3 fw    = length(headV) > 1e-4 ? normalize(headV) : vec3(0.0, 0.0, 1.0);
    return right * (r * cos(ang)) + up * (r * sin(ang)) + fw * (0.5 * BRAID_LEN * MINI_R * along);
}
#define ORB_ON      1
#define ORB_BEATS   8.0
#define ORB_R       0.055
#define ORB_SPIN    1.0
#define ORB_DEP     0.25
#define ORB_LAG     0.0
vec3 swarmOrbit(int grp, int w, float beat) {
    float cyc = 6.2831853 * beat / max(ORB_BEATS, 0.5) + float(grp) * ORB_LAG;
    float env = 0.5 - 0.5 * cos(cyc);
    float oa  = cyc * ORB_SPIN;
    float sd  = (w % 2 == 0) ? 1.0 : -1.0;
    vec3 right = transpose(camRot()) * vec3(1.0, 0.0, 0.0);
    vec3 up    = transpose(camRot()) * vec3(0.0, 1.0, 0.0);
    vec3 toCam = transpose(camRot()) * vec3(0.0, 0.0, -1.0);
    vec3 axis  = right * cos(oa) + up * sin(oa) + toCam * (ORB_DEP * cos(oa * 0.7));
    return axis * (sd * ORB_R * env);
}
#define BAL_COOL_BEATS  8.0
#define N_FORMS    25
#define FORM_STRIPE 9
#define STRIPE_SPIN 0.125
#define FORM_FORCE  -1
#define FORM_FUNNEL 10
#define FUNNEL_R0    0.06
#define FUNNEL_R1    0.60
#define FUNNEL_LEN   2.0
#define FUNNEL_TURNS 3.0
#define FUNNEL_SPIN  0.25
#define FORM_CHAIN 4
#define FORM_HELIX 7
#define FORM_RING  8
#define HELIX_R     0.16
#define HELIX_TURNS 1.5
#define HELIX_SPIN  0.0625
#define RING_RMIN   0.45
#define FORM_TORUS   14
#define FORM_GALAXY  15
#define FORM_JELLY   16
#define TOR_RM       0.52
#define TOR_RT       0.17
#define TOR_SPIN     0.125
#define GAL_R0       0.10
#define GAL_R1       0.86
#define GAL_TWIST    2.2
#define GAL_TILT     0.62
#define GAL_Y        0.34
#define GAL_SPIN     0.0625
#define JEL_BELL     0.42
#define JEL_TENT     0.60
#define JEL_Y        0.34
#define JEL_BEATS    2.0
#define FORM_COMET   17
#define FORM_CROWN   18
#define FORM_ATOM    19
#define COMET_R      0.30
#define COMET_LEN    2.4
#define CROWN_R      0.42
#define CROWN_SPIKE  0.26
#define CROWN_N      8.0
#define CROWN_Y      0.28
#define CROWN_SPIN   0.125
#define ATOM_A       0.52
#define ATOM_B       0.24
#define ATOM_TILT    1.0471976
#define ATOM_Y       0.32
#define ATOM_SPIN    0.0625
#define FORM_MANTA    20
#define FORM_TREFOIL  21
#define FORM_MOBIUS   22
#define FORM_SLINKY   24
#define MAN_X   0.70
#define MAN_Z   0.90
#define MAN_Y   0.34
#define MAN_W   0.16
#define MAN_BEATS 2.0
#define TREF_S  0.30
#define TREF_Y  0.30
#define TREF_SPIN 0.0625
#define MOB_R   0.50
#define MOB_W   0.16
#define MOB_Y   0.30
#define MOB_SPIN 0.0625
#define SLK_R   0.40
#define SLK_GAP 0.16
#define SLK_Y   0.30
#define SLK_AMP 0.55
#define SLK_BEATS 1.0
#define SLK_SPIN 0.0
#define FORM_COLUMN  40
#define COL_BASE     0.20
#define COL_H        1.35
#define COL_R0       0.045
#define COL_R1       0.34
#define COL_TWIST    6.2831853
#define AGG_WASP     0
#define AGG_ERUPT    1
#define AGG_FIGS     2
#define FIG_MORPH_BEATS 0.25
#define FIG_SPLV     1.0

#define CHAIN_MID  (N_LEAD * 5 / 12)
#define CHAIN_SAG  0.10
vec3 chainSlot(vec3 head, float fk) {
    float u = fk / float(CHAIN_MID);
    vec3 p = head * (1.0 - u);
    vec3 perp = normalize(cross(head + vec3(1e-4), vec3(0.0, 1.0, 0.0)) + vec3(0.0, 1e-4, 0.0));
    return p + perp * CHAIN_SAG * sin(3.14159 * clamp(u, 0.0, 2.0) * 0.5);
}
#define FORM_WASP  11
#define WASP_PER     64
#define WASP_SWARMS  (N_LEAD / WASP_PER)
#define WASP_R       0.05
#define WASP_ORB0    0.40
#define WASP_ORB1    0.75
#define WASP_TURN    4.0
#define WASP_SURGE   0.3
#define WASP_STRIKE  1.0
#define WASP_CLASH_BEATS 0.4
#define WASP_MORPH_BEATS 0.5
#define FORM_DISK    12
#define FORM_VDOME   13
#define MIRROR_GAP   0.07
#define DISK_R       0.45
#define DISK_SPIN    0.125
#define VDOME_R      0.62
#define VDOME_SPIN   0.03125
uniform sampler2D u_vor;
bool isMirror(int sh) { return false; }
bool isWorld(int sh) { return sh == FORM_CHAIN || sh == FORM_HELIX || sh == FORM_RING || sh == FORM_WASP || sh == FORM_COLUMN || sh == FORM_TORUS || sh == FORM_GALAXY || sh == FORM_JELLY || sh == FORM_CROWN || sh == FORM_ATOM || sh == FORM_MANTA || sh == FORM_TREFOIL || sh == FORM_MOBIUS || sh == FORM_SLINKY || isMirror(sh); }
float hashW(float a, float b) { return fract(sin(a * 91.7 + b * 47.3 + 13.1) * 43758.5453); }
vec3 waspOrbit(int g, float bc) {
    float fg = float(g);
    float R = mix(WASP_ORB0, WASP_ORB1, hashW(fg, 1.0));
    float ph = 6.2831853 * (hashW(fg, 2.0) + bc / (WASP_TURN * mix(0.7, 1.3, hashW(fg, 3.0))))
             * (hashW(fg, 4.0) < 0.5 ? 1.0 : -1.0)
             + WASP_SURGE * sin(3.14159 * bc / 2.0 + 6.2831853 * hashW(fg, 5.0));
    float r = R * (1.0 + 0.3 * sin(bc * 0.9 + fg * 1.7));
    vec3 n = normalize(vec3(hashW(fg, 6.0) - 0.5, 1.0, hashW(fg, 7.0) - 0.5) + vec3(0.6 * sin(bc * 0.13 + fg), 0.0, 0.6 * cos(bc * 0.11 + fg)));
    vec3 e1 = normalize(cross(n, vec3(0.0, 0.0, 1.0))), e2 = cross(n, e1);
    return r * (cos(ph) * e1 + sin(ph) * e2) + n * 0.25 * sin(bc * 1.3 + fg * 2.3);
}
void waspDuel(float w, out int a, out int b) {
    a = int(hashW(w, 31.0) * 8.0);
    b = int(hashW(w, 37.0) * 7.0);
    b += b >= a ? 1 : 0;
}
#define POSS_BEATS   8.0
#define POSS_TEAR    0.2
float possess(int g, float bc) {
    float cyc = floor(bc / POSS_BEATS), q = bc - cyc * POSS_BEATS;
    float grow = clamp((q - 1.0) / 5.0, 0.0, 1.0);
    float twitch = (q >= 1.0 && q < 7.0) ? (0.06 + 0.14 * grow) * exp(-6.0 * fract(q)) : 0.0;
    float tb = 2.0 + 4.0 * hashW(float(g), cyc + 41.0);
    float tear = smoothstep(tb, tb + 0.25, q) * (1.0 - smoothstep(7.0, 7.75, q))
               * step(1.0 - POSS_TEAR, hashW(float(g), cyc + 43.0));
    return max(twitch, tear);
}
vec3 waspCentre(int g, float bc) {
    float w = floor((bc + 1.0) / 2.0), p = fract((bc + 1.0) / 2.0);
    int a, b;
    waspDuel(w, a, b);
    vec3 o = waspOrbit(g, bc);
    if (g != a && g != b) return o;
    vec3 mid = 0.5 * (waspOrbit(a, bc) + waspOrbit(b, bc));
    float charge = p < 0.5 ? smoothstep(0.0, 0.5, p) : 1.0 - smoothstep(0.5, 1.0, p);
    return mix(o, mid, WASP_STRIKE * charge * charge);
}
vec3 worldSlot(int sh, vec3 head, int k, float df, int f, float bc) {
    if (sh == FORM_CHAIN) return chainSlot(head, float(k) + df);
    if (isMirror(sh)) {
        float fine = 4.0 * float(k) + 1.5 + 4.0 * df;
        if (sh == FORM_DISK) {
            float r = DISK_R * sqrt((fine + 0.5) / float(N_LMAX));
            float th = 2.39996323 * fine + 6.2831853 * DISK_SPIN * bc;
            vec3 e = normalize(vec3(-head.z, 0.0, head.x) + vec3(1e-4, 0.0, 0.0));
            return vec3(head.x, FLOOR_Y + MIRROR_GAP, head.z) + e * r * cos(th) + vec3(0.0, r * abs(sin(th)), 0.0);
        }
        float x = clamp(fine, 0.0, float(N_LMAX - 1));
        int i0 = int(floor(x)), i1 = min(i0 + 1, N_LMAX - 1);
        vec3 q = mix(texelFetch(u_vor, ivec2(f * N_LMAX + i0, 0), 0).xyz, texelFetch(u_vor, ivec2(f * N_LMAX + i1, 0), 0).xyz, fract(x));
        float a = 6.2831853 * VDOME_SPIN * bc;
        q.xz = mat2(cos(a), -sin(a), sin(a), cos(a)) * q.xz;
        return vec3(0.0, FLOOR_Y + MIRROR_GAP, 0.0) + q * VDOME_R;
    }
    if (sh == FORM_WASP) {
        int c = k / WASP_PER;
        float fk = float(k % WASP_PER) + df;
        float zf = 1.0 - 2.0 * (fk + 0.5) / float(WASP_PER);
        float rf = sqrt(max(0.0, 1.0 - zf * zf)), af = 2.39996323 * fk + bc * 3.0;
        return waspCentre(f * WASP_SWARMS + c, bc) + vec3(rf * cos(af), zf, rf * sin(af)) * WASP_R;
    }
    if (sh == FORM_COLUMN) {
        float t = (float(k) + df + 0.5 * float(f)) / float(N_LEAD - 1);
        float th = COL_TWIST * t + 6.2831853 * 0.61803398875 * (float(k) + 0.5 * float(f));
        float r = COL_R0 + (COL_R1 - COL_R0) * t * t;
        return vec3(r * cos(th), FLOOR_Y + COL_BASE + COL_H * t, r * sin(th));
    }
    if (sh == FORM_TORUS) {
        float fk = float(k) + df + 0.5 * float(f);
        float u = 6.2831853 * (fk / float(N_LEAD) + TOR_SPIN * bc);
        float v = 6.2831853 * fract(fk * 0.61803398875);
        float rm = TOR_RM + TOR_RT * cos(v);
        return vec3(rm * cos(u), TOR_RT * sin(v), rm * sin(u));
    }
    if (sh == FORM_GALAXY) {
        float fk = float(k) + df + 0.5 * float(f);
        float a = mod(float(k), 2.0) + 2.0 * float(f);
        float t = fk / float(N_LEAD);
        float ang = 6.2831853 * (a * 0.25 + t * GAL_TWIST) + 6.2831853 * GAL_SPIN * bc;
        float r = mix(GAL_R0, GAL_R1, t);
        float px = r * cos(ang), pz = r * sin(ang);
        return vec3(px, GAL_Y + pz * sin(GAL_TILT), pz * cos(GAL_TILT));
    }
    if (sh == FORM_JELLY) {
        float pl = fract(bc / JEL_BEATS);
        float pulse = 1.0 - 0.22 * pow(1.0 - pl, 2.0);
        float fk = float(k) + df + 0.5 * float(f);
        float th = 6.2831853 * fract(fk * 0.61803398875);
        if (mod(float(k), 5.0) < 3.0) {
            float y = clamp(fk / float(N_LEAD), 0.0, 1.0);
            float rr = JEL_BELL * pulse * cos(1.5707963 * y);
            return vec3(rr * cos(th), JEL_Y + JEL_BELL * pulse * sin(1.5707963 * y), rr * sin(th));
        }
        float tt = clamp(fk / float(N_LEAD), 0.0, 1.0);
        float rr = JEL_BELL * pulse * (0.85 + 0.15 * (1.0 - tt));
        return vec3(rr * cos(th), JEL_Y - JEL_TENT * tt, rr * sin(th));
    }
    if (sh == FORM_CROWN) {
        float fk = float(k) + df + 0.5 * float(f);
        float s = fk / float(N_LEAD);
        float a = 6.2831853 * (s + CROWN_SPIN * bc);
        float spike = pow(1.0 - abs(2.0 * fract(s * CROWN_N) - 1.0), 2.0);
        float r = CROWN_R + CROWN_SPIKE * spike;
        return vec3(r * cos(a), CROWN_Y + 0.5 * CROWN_SPIKE * spike, r * sin(a));
    }
    if (sh == FORM_ATOM) {
        float spin = 6.2831853 * ATOM_SPIN * bc;
        if (mod(float(k), 8.0) < 0.5) {
            return vec3(0.0, ATOM_Y, 0.0)
                 + 0.07 * vec3(cos(float(k)), sin(float(k) * 1.3), cos(float(k) * 0.7));
        }
        int o = k % 3;
        float th = 6.2831853 * (float(k) / float(N_LEAD)) + spin + float(o) * 2.0943951;
        vec3 e = vec3(ATOM_A * cos(th), ATOM_B * sin(th), 0.0);
        float a = float(o) * ATOM_TILT;
        mat3 rx = mat3(1.0, 0.0, 0.0, 0.0, cos(a), -sin(a), 0.0, sin(a), cos(a));
        return rx * e + vec3(0.0, ATOM_Y, 0.0);
    }
    if (sh == FORM_MANTA) {
        int cols = 8;
        int rows = N_LEAD / cols;
        float fk = float(k) + df;
        float cx = (mod(fk, float(cols)) / float(cols - 1)) * 2.0 - 1.0;
        float rw = (floor(fk / float(cols)) / float(rows - 1)) * 2.0 - 1.0;
        float ph = 6.2831853 * bc / MAN_BEATS;
        float y = MAN_W * sin(cx * 2.6 + rw * 1.1 + ph);
        return vec3(cx * MAN_X, MAN_Y + y, rw * MAN_Z);
    }
    if (sh == FORM_TREFOIL) {
        float fk = float(k) + df + 0.5 * float(f);
        float t = 6.2831853 * fk / float(N_LEAD) + 6.2831853 * TREF_SPIN * bc;
        vec3 p = vec3(sin(t) + 2.0 * sin(2.0 * t), cos(t) - 2.0 * cos(2.0 * t), -sin(3.0 * t));
        return TREF_S * p + vec3(0.0, TREF_Y, 0.0);
    }
    if (sh == FORM_MOBIUS) {
        float fk = float(k) + df + 0.5 * float(f);
        float uu = 6.2831853 * fk / float(N_LEAD) + 6.2831853 * MOB_SPIN * bc;
        float vv = MOB_W * sin(fk * 2.39996323);
        float ch = cos(uu * 0.5);
        vec3 C = MOB_R * vec3(cos(uu), sin(uu), 0.0);
        vec3 W = vec3(ch * cos(uu), ch * sin(uu), sin(uu * 0.5));
        return C + vv * W + vec3(0.0, MOB_Y, 0.0);
    }
    if (sh == FORM_SLINKY) {
        int rings = 6;
        float fk = float(k) + df;
        int ri = int(mod(float(k), float(rings)));
        float per = float(N_LEAD) / float(rings);
        float a = 6.2831853 * (floor(fk / float(rings)) / per) + 6.2831853 * SLK_SPIN * bc + 0.3 * float(f);
        float comp = 1.0 - SLK_AMP * (0.5 + 0.5 * cos(6.2831853 * bc / SLK_BEATS));
        float y = (float(ri) - 0.5 * float(rings - 1)) * SLK_GAP * comp;
        return vec3(SLK_R * cos(a), SLK_Y + y, SLK_R * sin(a));
    }
    vec3 h0 = f == 0 ? head : -head;
    float hl = max(length(h0), 1e-3);
    vec3 ax = h0 / hl;
    vec3 e1 = normalize(cross(ax, abs(ax.y) < 0.95 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    vec3 e2 = cross(ax, e1);
    float u = (float(k) + df) / float(N_LEAD - 1);
    if (sh == FORM_HELIX) {
        float a = f == 0 ? u : 1.0 - u;
        float phi = 6.2831853 * (HELIX_TURNS * a + HELIX_SPIN * bc) + (f == 1 ? 3.14159 : 0.0);
        vec3 onAxis = mix(h0, -h0, a);
        return onAxis + (cos(phi) * e1 + sin(phi) * e2) * HELIX_R * sqrt(sin(3.14159 * a) + 0.05);
    }
    float R = max(hl, RING_RMIN);
    float th = (f == 0 ? 0.0 : 3.14159) + (u - 0.5) * 3.14159;
    return R * (cos(th) * ax + sin(th) * e2);
}
float formLen(int sh) {
    if (sh == 1) return 3.08;
    if (sh == 2) return 1.80;
    if (sh == 3) return 1.5;
    if (sh == 5) return 2.2;
    if (sh == 6) return 0.8;
    if (sh == FORM_STRIPE) return 1.5;
    if (sh == FORM_FUNNEL) return FUNNEL_LEN;
    if (sh == FORM_COMET) return COMET_LEN;
    return 0.6;
}
vec3 formSlot(int sh, int k, float df) {
    float fk = float(k) + df;
    if (sh == 0) {
        float a = 6.2831853 * fk / float(N_LEAD);
        return vec3(cos(a), sin(a), -0.3);
    }
    if (sh == 1) {
        float u = fk / float(N_LEAD - 1);
        return vec3(0.12 * sin(fk * 2.1), 0.12 * cos(fk * 2.1), -u * 3.08);
    }
    if (sh == 2) {
        float arm = k < N_LEAD / 2 ? -1.0 : 1.0;
        float j = (float(k % (N_LEAD / 2)) + df + 1.0) * 6.0 / float(N_LEAD / 2);
        return vec3(arm * j * 0.30, 0.05 * j, -j * 0.30);
    }
    if (sh == 5) {
        float u = fk / float(N_LEAD - 1);
        float ph = 6.2831853 * 2.5 * u;
        return vec3(0.35 * cos(ph), 0.35 * sin(ph), -u * 2.2);
    }
    if (sh == 6) {
        float u = fk / float(N_LEAD - 1) * 2.0 - 1.0;
        return vec3(u * 1.3, 0.22 * sin(u * 4.7), -0.45 * u * u - 0.2);
    }
    if (sh == FORM_COMET) {
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
vec3 bodySlot(int sh, int k, float df, float bc) {
    if (sh == FORM_STRIPE) return stripeSlot(k, df, bc);
    if (sh == FORM_FUNNEL) return funnelSlot(k, df, bc);
    return formSlot(sh, k, df);
}

#define SPLIT_LEADERS    0
#define SPLIT_STEP_BEATS 0.25
#define SPLIT_DEMO       0
#define SPLIT_DEMO_BEATS 4.0
int splitLvl(int sh) {
    if (sh == FORM_STRIPE || sh == FORM_GALAXY) return 2;
    if (sh == FORM_RING || sh == FORM_CHAIN || sh == FORM_TORUS || sh == FORM_JELLY || sh == FORM_COMET || sh == FORM_ATOM || sh == FORM_MANTA || sh == FORM_TREFOIL || sh == FORM_MOBIUS || sh == FORM_SLINKY) return 1;
    if (isMirror(sh)) return 2;
    return 0;
}
#if SPLIT_LEADERS
int nLeaders(float lv) { return lv > 1.0 + 1e-3 ? N_LMAX : (lv > 1e-3 ? 2 * N_LEAD : N_LEAD); }
#else
int nLeaders(float lv) { return N_LEAD; }
#endif
int leaderParent(int m) { int g = m / N_LEAD, k = m % N_LEAD; return g == 3 ? k + N_LEAD : (g == 0 ? m : k); }
float leaderOff(int m, float lv) {
#if SPLIT_LEADERS
    int g = m / N_LEAD;
    float e1 = smoothstep(0.0, 1.0, clamp(lv, 0.0, 1.0)), e2 = smoothstep(0.0, 1.0, clamp(lv - 1.0, 0.0, 1.0));
    float d1 = (g == 1 || g == 3) ? 0.25 : -0.25;
    float d2 = (g == 2 || g == 3) ? 0.125 : -0.125;
    return d1 * e1 + d2 * e2;
#else
    return 0.0;
#endif
}

#define SCATTER_BEATS      1.0
#define SCATTER_K          1.2
#define SCATTER_IMPACT     0.5
#define SCATTER_DEMO       0
#define SCATTER_DEMO_BEATS 16.0
#define ACCENT_K           1.2
#define ACCENT_SKIT        1.2
#define ACCENT_VBOOST      0.5
#define ACCENT_BEATS       0.25
#define ACCENT_W0          0.20
#define ACCENT_W1          0.60
#define FRENZY_BPM0   150.0
#define FRENZY_BPM1   170.0
#define FRENZY_E0     0.60
#define FRENZY_E1     0.80
#define FRENZY_D0     0.40
#define FRENZY_D1     0.65
#define FRENZY_BEATS  2.0
#define FRENZY_ON     0.5
#define FRENZY_OFF    0.35
#define FRENZY_VMAX   6.0
#define FRENZY_SEP    0.45
#define FRENZY_SPREAD 0.35
#define FOL_FIX  1000.0
#define FOL_OFF  4.0
#define FOL_MIN  4
#if TRAINS
  #define FOL_N      (N_PER / LEAD_EVERY)
  #define FOL_MIN_E  (max(1, FOL_MIN / LEAD_EVERY))
#else
  #define FOL_N      N_PER
  #define FOL_MIN_E  FOL_MIN
#endif
#define SOCIAL_SPARSE_K 0.8
#define SOCIAL_MAX      1.0
#define SPECIAL_E  0.80
#define SPECIAL_W  0.50
#define SPECIAL_I  0.30
bool isSpecial(int sh) { return sh == 2 || sh == FORM_CHAIN || sh == FORM_HELIX || sh == FORM_RING || sh == FORM_STRIPE || isMirror(sh); }
float hash11(float x) { return fract(sin(x * 127.1 + 311.7) * 43758.5453); }
int pickNormal(float R, float spd, int cur, float salt) {
    float fast = smoothstep(0.06, 0.18, spd), tight = 1.0 - smoothstep(0.12, 0.50, R);
    int ids[16] = int[16](0, 1, 3, 5, 6, FORM_FUNNEL, FORM_TORUS, FORM_GALAXY, FORM_JELLY, FORM_COMET, FORM_CROWN, FORM_ATOM, FORM_MANTA, FORM_TREFOIL, FORM_MOBIUS, FORM_SLINKY);
    float sc[16] = float[16]((1.0 - fast) * (1.0 - tight) * 0.7,
                           fast * (1.0 - tight),
                           tight * (1.0 - 0.5 * fast),
                           fast * (1.0 - abs(2.0 * tight - 1.0)),
                           (1.0 - fast) * (1.0 - tight),
                           fast * tight,
                           (1.0 - abs(2.0 * tight - 1.0)) * 0.45,
                           (1.0 - fast) * (1.0 - tight) * 0.9,
                           (1.0 - fast) * 0.8,
                           fast * (1.0 - tight) * 0.8,
                           (1.0 - fast) * 0.7,
                           (1.0 - fast) * (1.0 - tight) * 0.85,
                           (1.0 - fast) * (1.0 - tight) * 0.85,
                           0.5,
                           (1.0 - fast) * 0.7,
                           0.5);
    int best = ids[0];
    float bs = -1e9;
    for (int n = 0; n < 16; n++) {
        float v = sc[n] + 0.35 * hash11(salt * 7.0 + float(n)) - (ids[n] == cur ? 0.6 : 0.0);
        if (v > bs) { bs = v; best = ids[n]; }
    }
    return best;
}
#define WATER_REPEL_H 0.06
#define WATER_REPEL_K 0.08
#define LEAD_FLOOR    0.16
#define REFL_MIN  0.50
#define REFL_MAX  0.80
vec3 camPosW(mat3 R) { return transpose(R) * vec3(0.0, 0.0, -CAM_D) + vec3(0.0, CAM_TY, 0.0); }
vec3 mirrorY(vec3 p) { return vec3(p.x, 2.0 * FLOOR_Y - p.y, p.z); }
#define REFL_SQUASH 0.42
#define N_RIP      96
#define RIP_SPEED  0.34
#define RIP_LIFE   2.8
#define RIP_W      0.10
#define RIP_K      130.0
#define RIP_FRONT  0.014
#define RIP_TRAIL  0.14
#define RIP_AMP    1.5
#define RIP_GLINT  0.022
#define RIP_DISP   0.055
#define RIP_TRIG   0.06
#define RIP_VMIN   0.08
uniform sampler2D u_rip;
float rippleAt(vec2 xz) {
    float h = 0.0;
    for (int r = 0; r < N_RIP; r++) {
        vec4 q = texelFetch(u_rip, ivec2(r, 0), 0);
        float age = u_time - q.z;
        if (q.w <= 0.0 || age < 0.0 || age > RIP_LIFE) continue;
        float rr = length(xz - q.xy);
        float d = rr - RIP_SPEED * age;
        float fade = 1.0 - age / RIP_LIFE;
        fade = pow(fade, 1.4);
        float spread = inversesqrt(1.0 + 4.0 * rr);
        if (d > RIP_FRONT * 3.0 || d < -RIP_TRAIL * 3.0) {
        } else {
            float env = d > 0.0 ? exp(-(d * d) / (RIP_FRONT * RIP_FRONT)) : exp(d / RIP_TRAIL);
            h += RIP_AMP * q.w * fade * spread * env * cos(RIP_K * d);
        }
        if (age < 0.22) h += RIP_AMP * q.w * (1.0 - age / 0.22) * exp(-(rr * rr) / (RIP_GLINT * RIP_GLINT));
    }
    return h;
}
vec3 projectRefl(vec3 pm, mat3 R) {
    vec3 a = project(pm, R), c = project(vec3(pm.x, FLOOR_Y, pm.z), R);
    a.xy = c.xy + (a.xy - c.xy) * REFL_SQUASH;
    float rp = rippleAt(pm.xz);
    a.xy += vec2(0.4, 1.0) * rp * RIP_DISP;
    return a;
}
float reflGain(vec3 pm, mat3 R) {
    vec3 d = normalize(pm - camPosW(R));
    return mix(REFL_MIN, REFL_MAX, pow(1.0 - abs(d.y), 5.0));
}

#define BOX_ZN  -0.25
#define BOX_ZB   2.50
float boxXW(float edge) { return edge * (BOX_ZN + CAM_D) / FOCAL; }
float boxYW()           { return (BOX_ZN + CAM_D) / FOCAL; }
#define SWARM_Z  2.05
vec3 swarmCentre() { return transpose(camRot()) * vec3(0.0, 0.0, SWARM_Z) + vec3(0.0, CAM_TY, 0.0); }
