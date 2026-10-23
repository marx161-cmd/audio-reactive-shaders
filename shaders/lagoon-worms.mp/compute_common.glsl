#define N_PER    1024
#define N_ALL    2048
#define SPK_N    16384
#define N_LEAD   48
#define N_LMAX   192
#define FORMATIONS 1
#define BOUND_R  1.10

#define CAM_D    3.0
#define FOCAL    1.9
#define CAM_TY   0.0
#define CAM_SPIN_S 60.0
#define CAM_TILT_A 0.65
#define CAM_TILT_S 97.0
#define FLOOR_Y  (-100.0)
#define FOG_K    0.55
#define FOG_D0   2.2
uniform vec3 u_cam_look;
uniform float u_time;
#define CAM_YAW   0.6
#define CAM_PITCH 0.35
mat3 camRot() {
    float y = CAM_YAW, p = CAM_PITCH;
    mat3 ry = mat3(cos(y), 0.0, -sin(y), 0.0, 1.0, 0.0, sin(y), 0.0, cos(y));
    mat3 rx = mat3(1.0, 0.0, 0.0, 0.0, cos(p), sin(p), 0.0, -sin(p), cos(p));
    return rx * ry;
}
float fogAt(vec3 p, mat3 R) { return exp(-max((R * (p - vec3(0.0, CAM_TY, 0.0))).z + CAM_D - FOG_D0, 0.0) * FOG_K); }
vec3 project(vec3 p, mat3 R) {
    vec3 c = R * (p - vec3(0.0, CAM_TY, 0.0));
    float s = FOCAL / max(c.z + CAM_D, 0.3);
    return vec3(c.xy * s, s);
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
#define LEAD_PERIOD     0.35
#define LEAD_DAMP       0.75
#define LEAD_VMAX       2.5
#define PRED_BACK       6.0
#define PRED_FWD        6.0
#define PRED_N          49
#define TURN_FIT        0.5
#define CZ_MIN          0.30
#define LEAD_FF         1.0
#define BAL_HOLD_BEATS  1.0
#define LEAD_SOFT       0.18
#define LEAD_SPREAD     0.10
#define BAL_COOL_BEATS  8.0
#define N_FORMS    16
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

#define CHAIN_MID  (N_LEAD * 5 / 12)
#define CHAIN_SAG  0.10
vec3 chainSlot(vec3 head, float fk) {
    float u = fk / float(CHAIN_MID);
    vec3 p = head * (1.0 - u);
    vec3 perp = normalize(cross(head + vec3(1e-4), vec3(0.0, 1.0, 0.0)) + vec3(0.0, 1e-4, 0.0));
    return p + perp * CHAIN_SAG * sin(3.14159 * clamp(u, 0.0, 2.0) * 0.5);
}
#define FORM_WASP  11
#define WASP_PER     12
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
bool isMirror(int sh) { return sh == FORM_DISK || sh == FORM_VDOME; }
#define FORM_SHELL   14
#define SHELL_R      0.72
#define SHELL_RINGS  6
#define SHELL_SPIN_S 0.15
vec3 shellAxis(float bc) { return normalize(vec3(0.35 * sin(bc * 0.049), 1.0, 0.35 * cos(bc * 0.067))); }
#define FORM_TORUS   15
#define TORUS_R      0.95
#define TORUS_TUBE   0.27
#define TORUS_SPIN_S 0.10
vec3 gTorusN = vec3(0.0, 1.0, 0.0);
#define INTERACT     0
#define COV_N        8
#define COV_H        1.4
int covCell(vec3 p) {
    ivec3 c = clamp(ivec3(floor((p / (2.0 * COV_H) + 0.5) * float(COV_N))), ivec3(0), ivec3(COV_N - 1));
    return c.x + COV_N * (c.y + COV_N * c.z);
}
#define CORE_R       0.32
#define CORE_K       1.6
bool isShell(int sh) { return sh == FORM_SHELL || sh == FORM_TORUS; }
float gShut = 0.0, gShutLat = 0.0;
#define SHUT_SLIDE   0.20
bool isWorld(int sh) { return sh == FORM_CHAIN || sh == FORM_HELIX || sh == FORM_RING || sh == FORM_WASP || isMirror(sh) || isShell(sh); }
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
    if (sh == FORM_TORUS) {
        float gi = float(f * N_LMAX) + clamp(4.0 * float(k) + 1.5 + 4.0 * df, 0.0, float(N_LMAX) - 0.001);
        float iu = floor(gi / 8.0), iv = gi - iu * 8.0;
        float u = 6.2831853 * (iu / float(2 * N_LMAX / 8) + TORUS_SPIN_S * u_time);
        float v = 6.2831853 * (iv / 8.0 + 0.5 * iu / float(2 * N_LMAX / 8));
        vec3 n = gTorusN;
        vec3 t1 = normalize(cross(n, abs(n.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0))), t2 = cross(n, t1);
        vec3 radial = cos(u) * t1 + sin(u) * t2;
        return (TORUS_R + TORUS_TUBE * cos(v)) * radial + TORUS_TUBE * sin(v) * n;
    }
    if (isShell(sh)) {
        float gi = float(f * N_LMAX) + clamp(4.0 * float(k) + 1.5 + 4.0 * df, 0.0, float(N_LMAX) - 0.001);
        float per = float(2 * N_LMAX / SHELL_RINGS);
        float ring = floor(gi / per), pos = gi - ring * per;
        float lat = (ring - 0.5 * float(SHELL_RINGS - 1)) * (2.4 / float(SHELL_RINGS));
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            float tl = float(sgn) * gShutLat;
            lat += gShut * SHUT_SLIDE * sign(tl - lat) * exp(-pow((lat - tl) / 0.32, 2.0));
        }
        float a = 6.2831853 * (pos / per + SHELL_SPIN_S * u_time + 0.37 * ring);
        vec3 ax = shellAxis(bc);
        vec3 t1 = normalize(cross(ax, abs(ax.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0))), t2 = cross(ax, t1);
        return SHELL_R * (cos(lat) * (cos(a) * t1 + sin(a) * t2) + sin(lat) * ax);
    }
    if (isMirror(sh)) {
        float fine = 4.0 * float(k) + 1.5 + 4.0 * df;
        if (sh == FORM_DISK) {
            float r = DISK_R * sqrt((fine + 0.5) / float(N_LMAX));
            float th = 2.39996323 * fine + 6.2831853 * DISK_SPIN * bc;
            vec3 e = normalize(vec3(-head.z, 0.0, head.x) + vec3(1e-4, 0.0, 0.0));
            return head + e * r * cos(th) + vec3(0.0, r * sin(th), 0.0);
        }
        float x = clamp(fine, 0.0, float(N_LMAX - 1));
        int i0 = int(floor(x)), i1 = min(i0 + 1, N_LMAX - 1);
        vec3 q = mix(texelFetch(u_vor, ivec2(f * N_LMAX + i0, 0), 0).xyz, texelFetch(u_vor, ivec2(f * N_LMAX + i1, 0), 0).xyz, fract(x));
        float a = 6.2831853 * VDOME_SPIN * bc;
        q.xz = mat2(cos(a), -sin(a), sin(a), cos(a)) * q.xz;
        return q * VDOME_R;
    }
    if (sh == FORM_WASP) {
        int c = k / WASP_PER;
        float fk = float(k % WASP_PER) + df;
        float zf = 1.0 - 2.0 * (fk + 0.5) / float(WASP_PER);
        float rf = sqrt(max(0.0, 1.0 - zf * zf)), af = 2.39996323 * fk + bc * 3.0;
        return waspCentre(f * 4 + c, bc) + vec3(rf * cos(af), zf, rf * sin(af)) * WASP_R;
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

#define SPLIT_STEP_BEATS 0.25
#define SPLIT_DEMO       0
#define SPLIT_DEMO_BEATS 4.0
int splitLvl(int sh) {
    if (sh == FORM_STRIPE) return 2;
    if (sh == FORM_RING || sh == FORM_CHAIN) return 1;
    if (isMirror(sh)) return 2;
    if (isShell(sh)) return 2;
    return 0;
}
int nLeaders(float lv) { return lv > 1.0 + 1e-3 ? N_LMAX : (lv > 1e-3 ? 2 * N_LEAD : N_LEAD); }
int leaderParent(int m) { int g = m / N_LEAD, k = m % N_LEAD; return g == 3 ? k + N_LEAD : (g == 0 ? m : k); }
float leaderOff(int m, float lv) {
    int g = m / N_LEAD;
    float e1 = smoothstep(0.0, 1.0, clamp(lv, 0.0, 1.0)), e2 = smoothstep(0.0, 1.0, clamp(lv - 1.0, 0.0, 1.0));
    float d1 = (g == 1 || g == 3) ? 0.25 : -0.25;
    float d2 = (g == 2 || g == 3) ? 0.125 : -0.125;
    return d1 * e1 + d2 * e2;
}

#define SCATTER_BEATS      1.0
#define SCATTER_K          1.2
#define SCATTER_IMPACT     0.5
#define SCATTER_DEMO       0
#define SCATTER_DEMO_BEATS 16.0
#define ACCENT_K           0.6
#define ACCENT_BEATS       0.25
#define ACCENT_W0          0.35
#define ACCENT_W1          0.80
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
#define SPECIAL_E  0.80
#define SPECIAL_W  0.50
#define SPECIAL_I  0.30
bool isSpecial(int sh) { return sh == 2 || sh == FORM_CHAIN || sh == FORM_HELIX || sh == FORM_RING || sh == FORM_DISK || isShell(sh); }
float hash11(float x) { return fract(sin(x * 127.1 + 311.7) * 43758.5453); }
int pickNormal(float R, float spd, int cur, float salt) {
    if (hash11(salt * 3.1 + 0.7) < 0.75) return FORM_SHELL;
    float fast = smoothstep(0.06, 0.18, spd), tight = 1.0 - smoothstep(0.12, 0.50, R);
    int ids[9] = int[9](FORM_RING, FORM_CHAIN, FORM_FUNNEL, 5, FORM_DISK, 1, 6, FORM_SHELL, FORM_TORUS);
    float sc[9] = float[9](0.5 * (1.0 - fast) + 0.1,
                           0.45 * (1.0 - tight) + 0.1,
                           fast * tight,
                           fast * (1.0 - abs(2.0 * tight - 1.0)),
                           0.7 * (1.0 - fast) * tight,
                           0.6 * fast * (1.0 - tight),
                           0.5 * (1.0 - fast) * (1.0 - tight),
                           -1.0,
                           0.6);
    int best = ids[0];
    float bs = -1e9;
    for (int n = 0; n < 9; n++) {
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
#define RIP_SPEED  0.45
#define RIP_LIFE   1.6
#define RIP_W      0.10
#define RIP_K      28.0
#define RIP_DISP   0.035
#define RIP_TRIG   0.06
#define RIP_VMIN   0.08
uniform sampler2D u_rip;
float rippleAt(vec2 xz) {
    float h = 0.0;
    for (int r = 0; r < N_RIP; r++) {
        vec4 q = texelFetch(u_rip, ivec2(r, 0), 0);
        float age = u_time - q.z;
        if (q.w <= 0.0 || age < 0.0 || age > RIP_LIFE) continue;
        float d = length(xz - q.xy) - RIP_SPEED * age;
        if (abs(d) > 3.0 * RIP_W) continue;
        h += q.w * (1.0 - age / RIP_LIFE) * exp(-(d * d) / (RIP_W * RIP_W)) * sin(RIP_K * d);
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
