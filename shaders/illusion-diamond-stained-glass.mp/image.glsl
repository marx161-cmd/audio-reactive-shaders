// Illusion-Diamond-Stained-Glass: image pass.
//
// Traces the gem per pixel and composites it over the stained-glass backdrop.

#define CAM_Z      6.0
#define FOV        0.21
#define BG_Z      -3.0
#define BG_GAIN    0.85
#define FRONT_DIM  0.35
#define DISP_EXAG  1.5
#define GEM_A      2.3818
#define GEM_B      0.0121
#define MAX_NODES  16
#define W_MIN      0.004
#define GEPS       1e-4
#define STAR_EMIT  8.0
#define CAP_REFL   0.92
const vec3 LAMBDA_UM = vec3(0.630, 0.550, 0.460);

vec4 gQ;
vec3 toWorld(vec3 v) { return quatRot(gQ, v); }
vec3 toObj(vec3 v) { return quatRot(vec4(-gQ.xyz, gQ.w), v); }
vec3 keepHue(vec3 c) { float m = max(max(c.r, c.g), c.b); return m > 1.0 ? c / m : c; }

float fresnelD(float cosi, float n1, float n2) {
    float s = n1 / n2 * sqrt(max(1.0 - cosi * cosi, 0.0));
    if (s >= 1.0) return 1.0;
    float cost = sqrt(1.0 - s * s);
    float rs = (n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost);
    float rp = (n1 * cost - n2 * cosi) / (n1 * cost + n2 * cosi);
    return 0.5 * (rs * rs + rp * rp);
}

vec2 screenSt(vec3 P) {
    vec2 uv = P.xy / ((CAM_Z - P.z) * FOV);
    return (uv * iResolution.y + iResolution.xy) * 0.5 / iResolution.xy;
}
#define GLASS_BASE   0.06
#define GLASS_LIGHT  1.4
#define LEAD_COL     0.012
#define MOVE_LIGHT   1
#define LIGHT_SIGMA  0.20
#define LIGHT_COL    vec3(1.0, 0.12, 0.08)
#define LIGHT_PEAK   1.6
#define LIGHT_SPEED  vec2(0.031, 0.022)
float h11(float x) { return fract(sin(x * 12.9898) * 43758.5453); }
vec3 bgAt(vec2 st) {
    float lod = max(log2(float(textureSize(u_backdrop, 0).x) / iResolution.x), 0.0);
    vec3 c = textureLod(u_backdrop, st, lod).rgb;
    vec4 g = textureLod(u_glass, st, lod);
    float lead = g.r, id = g.g * 255.0, edge = g.b, var = g.a * 2.0 - 1.0;
    float h1 = h11(id), h2 = h11(id + 17.0), kind = h11(id + 41.0);
    float t = iTime;
    float lvl;
    if (kind < 0.94) {
        lvl = 0.82 + 0.18 * sin(t * (0.3 + 0.6 * h2) + 6.2832 * h1);
    } else if (kind < 0.97) {
        lvl = 0.08 + 0.9 * pow(max(sin(t * (0.2 + 0.4 * h2) + 6.2832 * h1), 0.0), 12.0);
    } else {
        lvl = 0.3 + u_bass * 0.8 + u_onset * 0.4;
    }
    vec2 q = st - 0.5;
    lvl *= 0.92 + 0.08 * sin(length(q) * 22.0 - t * 0.7);
    vec3 lightMul = vec3(1.0);
#if MOVE_LIGHT
    vec2 lp = abs(fract(t * LIGHT_SPEED + vec2(0.13, 0.57)) * 2.0 - 1.0) * 0.8 + 0.1;
    vec2 dl = st - lp;
    lightMul += LIGHT_COL * (LIGHT_PEAK * exp(-dot(dl, dl) / (2.0 * LIGHT_SIGMA * LIGHT_SIGMA)));
#endif
    float tone = mix(0.7, 1.15, smoothstep(0.0, 0.6, edge)) * (1.0 + 0.35 * var);
    vec3 lit = c * (GLASS_BASE + GLASS_LIGHT * lvl * tone * lightMul);
    return mix(lit, vec3(LEAD_COL), lead);
}
vec3 envW(vec3 P, vec3 D) {
    if (D.z < -0.02) return bgAt(screenSt(P + D * ((BG_Z - P.z) / D.z)));
    vec3 Dm = vec3(D.xy, -max(abs(D.z), 0.02));
    return bgAt(screenSt(P + Dm * ((BG_Z - P.z) / Dm.z))) * FRONT_DIM;
}

bool stoneHit(vec3 o, vec3 d, out float tN, out int jN) {
    tN = -1e9; float tF = 1e9; jN = -1;
    for (int j = 0; j < STONE_N; j++) {
        vec4 P = STONE[j];
        float dn = dot(d, P.xyz), dist = P.w - dot(o, P.xyz);
        if (abs(dn) < 1e-9) { if (dist < 0.0) return false; continue; }
        float t = dist / dn;
        if (dn < 0.0) { if (t > tN) { tN = t; jN = j; } }
        else if (t < tF) tF = t;
    }
    return tN <= tF && jN >= 0;
}
float stoneExit(vec3 o, vec3 d, out int jF) {
    float tF = 1e9; jF = 0;
    for (int j = 0; j < STONE_N; j++) {
        vec4 P = STONE[j];
        float dn = dot(d, P.xyz);
        if (dn <= 1e-9) continue;
        float t = (P.w - dot(o, P.xyz)) / dn;
        if (t < tF) { tF = t; jF = j; }
    }
    return tF;
}

#define STAR_PULSE   0.25
#define STAR_GRAN    14.0
#define STAR_CHURN   0.08
#define FIELD_DENS   0.985
float h31(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
float vnoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(h31(i), h31(i + vec3(1, 0, 0)), f.x), mix(h31(i + vec3(0, 1, 0)), h31(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(h31(i + vec3(0, 0, 1)), h31(i + vec3(1, 0, 1)), f.x), mix(h31(i + vec3(0, 1, 1)), h31(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float fbm(vec3 p) { float a = 0.5, s = 0.0; for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = p * 2.03 + 17.1; a *= 0.5; } return s; }
#define STAR_CELL    0.035
vec3 h33(vec3 p) { return vec3(h31(p), h31(p + 19.19), h31(p + 47.7)); }
vec2 worley(vec3 p, float t) {
    vec3 i = floor(p), f = fract(p);
    float d1 = 9.0, d2 = 9.0;
    for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
        vec3 c = vec3(x, y, z);
        vec3 o = h33(i + c);
        o = 0.5 + 0.42 * sin(t * (0.6 + o) + 6.2831 * o);
        float d = length(c + o - f);
        if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) d2 = d;
    }
    return vec2(d1, d2);
}
vec2 photosphere(vec3 sp) {
    vec3 p = sp / STAR_CELL;
    p += 0.9 * vec3(fbm(p * 0.35 + iTime * 0.05), fbm(p * 0.35 + 7.3 - iTime * 0.04), fbm(p * 0.35 + 13.1));
    vec2 w = worley(p, iTime * 0.35);
    float dome = 1.0 - smoothstep(0.0, 0.85, w.x);
    float lane = smoothstep(0.0, 0.45, w.y - w.x);
    float turb = 0.85 + 0.3 * fbm(p * 2.7 - iTime * 0.2);
    float temp = clamp(0.25 + 0.75 * dome * lane, 0.0, 1.0) * turb;
    float sup = 0.85 + 0.3 * fbm(sp / (STAR_CELL * 9.0) + iTime * 0.01);
    float spot = fbm(sp / (STAR_CELL * 14.0) + 3.7);
    float umbra = smoothstep(0.70, 0.76, spot), pen = smoothstep(0.62, 0.70, spot);
    float fib = 0.75 + 0.25 * sin(atan(sp.y, sp.x) * 90.0 + spot * 40.0);
    float b = (0.45 + 0.75 * temp) * sup * mix(1.0, mix(0.45 * fib, 0.08, umbra), pen);
    return vec2(b, temp * mix(1.0, 0.3, pen));
}
#if PORTAL_N > 0
#define STAR_DIST    1.0
vec3 starRoom(vec3 pl, vec3 dir) {
    float pulse = 1.0 + STAR_PULSE * u_bass;
    vec3 q = dir * 180.0, cell = floor(q);
    float hs = h31(cell);
    vec3 col = vec3(0.012, 0.009, 0.02) * (0.5 + fbm(dir * 3.0));
    if (hs > FIELD_DENS) {
        vec3 cp = cell + 0.5 + 0.3 * (vec3(h31(cell + 1.7), h31(cell + 3.1), h31(cell + 5.3)) - 0.5);
        col += vec3(0.8, 0.85, 1.0) * smoothstep(0.4, 0.0, length(q - cp)) * (hs - FIELD_DENS) / (1.0 - FIELD_DENS) * 0.7;
    }
    vec3 C = STAR_DIR * STAR_DIST;
    float Rs = STAR_DIST * sin(STAR_RAD);
    vec3 oc = pl - C;
    float b = dot(oc, dir), h = b * b - (dot(oc, oc) - Rs * Rs);
    if (h > 0.0 && -b - sqrt(h) > 0.0) {
        vec3 sp = pl + dir * (-b - sqrt(h)) - C;
        vec3 nS = sp / Rs;
        float mu = max(dot(nS, -dir), 0.0);
        vec2 ph = photosphere(sp);
        float limb = 1.0 - 0.6 * (1.0 - mu) - 0.2 * (1.0 - mu) * (1.0 - mu);
        vec3 tcol = mix(vec3(1.0, 0.42, 0.12), vec3(1.0, 0.93, 0.78), ph.y);
        col = STAR_COL * tcol * STAR_I * pulse * limb * ph.x * mix(vec3(1.0, 0.7, 0.45), vec3(1.0), mu);
    } else {
        float x = max(sqrt(max(dot(oc, oc) - b * b, 0.0)) / Rs - 1.0, 0.0);
        float ang = atan(oc.y + dir.y, oc.x + dir.x);
        float streak = 0.7 + 0.6 * fbm(vec3(ang * 3.0, x * 2.0 - iTime * 0.03, 0.0));
        col += STAR_COL * STAR_I * pulse * (vec3(1.0, 0.55, 0.4) * 0.45 * exp(-x * 40.0) + 0.10 * exp(-x * 4.0) * streak);
    }
    return 1.0 - exp(-col);
}
uniform sampler2D u_portal;
vec3 portalView(vec3 pl, vec3 dir) {
#ifdef PORTAL_IMAGE
    vec3 hp = pl + dir * ((PORTAL_IMG_DIST - pl.z) / max(dir.z, 0.05));
    vec2 uv = hp.xy / PORTAL_IMG_SIZE + 0.5;
    uv = 1.0 - abs(1.0 - mod(uv, 2.0));
    return textureLod(u_portal, uv, 0.5).rgb;
#else
    return starRoom(pl, dir);
#endif
}
int portalOf(int j) { for (int i = 0; i < PORTAL_N; i++) if (PORTAL_F[i] == j) return i; return -1; }
vec3 portalLocal(int j, vec3 dOut) {
    vec3 z = STONE[j].xyz;
    vec3 x = normalize(cross(abs(z.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0), z));
    return vec3(dot(dOut, x), dot(dOut, cross(z, x)), dot(dOut, z));
}
#endif

vec3 traceCh(vec3 o, vec3 d, int ch) {
    float nI = GEM_A + GEM_B * DISP_EXAG / (LAMBDA_UM[ch] * LAMBDA_UM[ch]);
    vec3 acc = vec3(0.0), w = vec3(1.0);
    float tN; int jN;
    stoneHit(o, d, tN, jN);
    vec3 p = o + d * tN, nrm = STONE[jN].xyz;
    float R = fresnelD(clamp(-dot(d, nrm), 0.0, 1.0), 1.0, nI);
    acc += R * envW(toWorld(p), toWorld(reflect(d, nrm)));
    d = refract(d, nrm, 1.0 / nI); o = p - nrm * GEPS; w *= 1.0 - R;
    int nb = 0;
    for (int n = 0; n < MAX_NODES; n++) {
        if (max(max(w.r, w.g), w.b) < W_MIN) break;
        int jF;
        float tb = stoneExit(o, d, jF);
        int kind = 0, id = -1;
#if WALL_N > 0
        for (int i = 0; i < WALL_N; i++) {
            float dn = dot(d, WALL_M[i]);
            if (abs(dn) < 1e-9) continue;
            float t = -dot(o, WALL_M[i]) / dn;
            if (t > GEPS && t < tb && dot(o + d * t, WALL_U[i]) >= 0.0) { tb = t; kind = 1; id = i; }
        }
#endif
#if CAP_N > 0
        for (int i = 0; i < CAP_N; i++) {
            vec3 oc = o - CAP_CR[i].xyz;
            float b = dot(oc, d), h = b * b - (dot(oc, oc) - CAP_CR[i].w * CAP_CR[i].w);
            if (h <= 0.0) continue;
            h = sqrt(h);
            for (int r = 0; r < 2; r++) {
                float t = r == 0 ? -b - h : -b + h;
                if (t > GEPS && t < tb && dot(o + d * t - CAP_CR[i].xyz, CAP_AC[i].xyz) >= CAP_AC[i].w * CAP_CR[i].w) {
                    tb = t; kind = 2; id = i; break;
                }
            }
        }
#endif
#if OBJ_N > 0
        for (int i = 0; i < OBJ_N; i++) {
            vec3 oc = o - OBJ_POS_R[i].xyz;
            float b = dot(oc, d), h = b * b - (dot(oc, oc) - OBJ_POS_R[i].w * OBJ_POS_R[i].w);
            if (h <= 0.0) continue;
            float t = -b - sqrt(h);
            if (t > GEPS && t < tb) { tb = t; kind = 3; id = i; }
        }
#endif
        p = o + d * tb;
#if OBJ_N > 0
        if (kind == 3) {
            vec3 sn = normalize(p - OBJ_POS_R[id].xyz);
            acc += w * OBJ_COL[id] * STAR_EMIT * (0.55 + 0.45 * abs(dot(sn, d)));
            break;
        }
#endif
#if CAP_N > 0
        if (kind == 2) {
            vec3 cn = (p - CAP_CR[id].xyz) / CAP_CR[id].w;
            d = reflect(d, cn); w *= CAP_REFL; o = p + d * GEPS; nb++;
            continue;
        }
#endif
#if WALL_N > 0
        if (kind == 1) {
            vec3 m = WALL_M[id];
            float Rw = fresnelD(abs(dot(d, m)), nI, 1.0);
            if (Rw > 0.5) { d = reflect(d, m); w *= Rw; nb++; } else w *= 1.0 - Rw;
            o = p + d * GEPS;
            continue;
        }
#endif
        nrm = STONE[jF].xyz;
        R = fresnelD(clamp(dot(d, nrm), 0.0, 1.0), nI, 1.0);
        if (R >= 1.0) { d = reflect(d, nrm); o = p - nrm * GEPS; nb++; continue; }
        vec3 dOut = refract(d, -nrm, nI);
        vec3 env;
#if PORTAL_N > 0
        if (nb >= PORTAL_MIN && nb <= PORTAL_MAX && portalOf(jF) >= 0)
            env = portalView(portalLocal(jF, p - STONE[jF].xyz * STONE[jF].w), portalLocal(jF, dOut));
        else
#endif
        env = envW(toWorld(p), toWorld(dOut));
        acc += w * (1.0 - R) * env;
        if (R > 0.5) { d = reflect(d, nrm); w *= R; o = p - nrm * GEPS; nb++; continue; }
        w = vec3(0.0);
        break;
    }
    acc += w * envW(toWorld(o), toWorld(d));
    return acc;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    gQ = normalize(texelFetch(iChannel0, ivec2(0, 0), 0));
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0.0, 0.0, CAM_Z);
    vec3 rd = normalize(vec3(uv * FOV, -1.0));
    vec3 bg = bgAt(fragCoord / iResolution.xy);
    float b = dot(ro, rd), h = b * b - (dot(ro, ro) - STONE_RB * STONE_RB);
    if (h <= 0.0) { fragColor = vec4(bg, 1.0); return; }
    vec3 o = toObj(ro), d = toObj(rd);
    float tN; int jN;
    if (!stoneHit(o, d, tN, jN) || tN <= 0.0) { fragColor = vec4(bg, 1.0); return; }
    vec3 col;
    for (int ch = 0; ch < 3; ch++) col[ch] = traceCh(o, d, ch)[ch];
    fragColor = vec4(keepHue(max(col, 0.0)), 1.0);
}
