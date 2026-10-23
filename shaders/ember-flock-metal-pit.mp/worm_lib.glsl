#define BIN_T         L_BIN_T
#define BIN_SEP0      L_SEP0
#define BIN_SEP1      L_SEP1
#define BIN_BOB       L_BOB
#define BIN_KICK_SEP  L_KICK_SEP
#define WORM_SCALE    1.15
#define WORM_MAX_R    1.25
#define SPLIT_EDGE    1.00
#define SPLIT_INNER   0.20
#define SPLIT_SWEEP   0.45
#define SPLIT_WARP    0.40
#define STRETCH_BEATS 8.0
#define SEP_NEAR      0.10
#define SEP_FAR_QUIET 0.70
#define SEP_FAR_LOUD  0.28
#define FIG_MIRROR_K  0.05
#define PLANE_ROT     1.5707963
#define SQUISH_X      0.45
#define FIG_ROT       0.0
#define FIG_EDGE1     1.5707963
#define TILT_DEG      0.7853982
#define FIG_LANE_N     4.0
#define FIG_LANE_DZ    0.0
#define FIG_LANE_BEATS 8.0
#define RUSH_BEATS    0.9
#define RUSH_SHORT    0.05
#define HAWK_KICK  0.25
#define HAWK_SPD   1.5
#define SCATTER_SPD 3.0
#define REPEL_A    0.06
#define REPEL_R0   0.18
#define REPEL_SWIRL 2.5
vec3 wHeadOffAt(float bc, vec4 w, float wE, float pocket, vec3 sepX, float edge, float blade, float camLock, float laneZ, float loud) {
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
    float ed = edge;
    float ce = cos(ed), se = sin(ed);
    camP = camP * ce + cross(n, camP) * se + n * dot(n, camP) * (1.0 - ce);
    vec3 locked = transpose(camRot()) * camP;
    return (camLock > 0.5 ? locked : nat) * WORM_SCALE;
}
vec3 softCore(vec3 p) {
    float r = length(p);
    vec3 dir = r > 1e-4 ? p / r : vec3(1.0, 0.0, 0.0);
    vec3 tan = normalize(cross(vec3(0.0, 1.0, 0.0), dir));
    return p + (dir + REPEL_SWIRL * tan) * (REPEL_A * exp(-r / REPEL_R0));
}
#define SWIM_SPLIT   0.03125
#define HELIX_H      0.62
#define HELIX_W      0.16
#define HELIX_CLIMB  0.80
#define FORM_HELIX_BEATS 5.0
#define RING_R       0.65
#define SWIM_SPIRAL_TURNS 3.0
#define SWIM_SPIRAL_COILS 5.0
#define PUNCH_D      1.5
#define PUNCH_SHOT   0.28
#define SUCK_MIN     0.12
#define SWIM_WAVE_TURNS   2.5
#define SWIM_WAVE_COILS   4.0
#define SWIM_HELIX_TURNS 4.0
#define SWIM_FACE_TURNS  4.0
#define SWIM_HELIX_SPIN  0.0625
#define SWIM_R      0.85
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
            float ud = (u - SWIM_SPLIT) / (uEnd - SWIM_SPLIT);
            float y  = mix(-HELIX_H, HELIX_H, ud);
            float rr = HELIX_W * (1.0 - smoothstep(0.85, 1.0, ud));
            float ph = 6.2831853 * SWIM_HELIX_TURNS * ud + (f < 0.5 ? 0.0 : 3.14159);
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
int pickFigure(vec4 A, vec4 B, float R, float spd, float cur, float salt, bool fireSpecial, float repeat,
               float lastSp, float win) {
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
