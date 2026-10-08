// Nebulabrot-Infinite-Dive: image pass.
//
// Samples the density layers at the current view, cross-fades from the global
// bake into the window bakes as the zoom passes them, and tone-maps the escape
// bands into colour. The music drives colour and brightness through the bands;
// bass makes the dive run faster and flashes the loop point.

uniform sampler2D u_nebula;
uniform float u_bass, u_mid, u_presence, u_brilliance, u_onset, u_rms;
uniform float u_bass_accum, u_color_phase;

#define ZOOM0    1.15
#define ZMAX     6.0
#define CYCLE    48.0
#define BASS_DIVE 4.0

const vec2 TARGETS[4] = vec2[4](vec2(-1.0695, 0.1273), vec2(-0.1398, 0.6586),
                                vec2(0.2586, 0.1273), vec2(-0.8039, 0.7914));
vec2 target(int k) {
    vec2 t = TARGETS[(k / 2) % 4];
    return (k % 2 == 1) ? vec2(t.x, -t.y) : t;
}

#define FLOOR_SHORT 0.36
#define FLOOR_MID   0.012
#define FLOOR_LONG  0.004
vec3 tone(float d, float fl, float g) {
    float v = max(d - fl, 0.0) / (1.0 - fl);
    return vec3(1.0 - exp(-g * pow(v, 1.15)));
}

#define EFF_WARP     0.0
#define EFF_PARALLAX 0.0
#define EFF_DODGE    0.5
#define EFF_RELIEF   1.0
#define EDGE_MODE    0
#define EDGE_S_MIN   0.35
#define EDGE_S_MAX   1.6
#define EDGE_SHAPE_LOD 7.0
#define EFF_SLICE    1.0
#define AUTO_DIVE    0
#define GRADE_CONTRAST 1.6
#define GRADE_FLAT   0.75
#define GRADE_EDGE   1.25
#define SAT_FLAT     1.2
#define FINAL_CONTRAST 0.35
#define FINAL_SAT    1.25
#define SAT_EDGE     2.0
#define EDGE_K       2.5
#define RELIEF_MODE  1
#define TERR_N       12
#define TERR_LO      -5.5
#define TERR_HI      0.7
#define TERR_GLOW    0.6
#define TERR_GLOW_LO 0.45
#define TERR_GLOW_HI 0.7
const vec3 TERR_GLOW_COL = vec3(1.00, 0.42, 0.04);
#define TERR_SHADOW_PX 4.0
#define BLEND_MODE   1

uniform float u_onset_smooth;

uniform sampler2D u_win0;
uniform sampler2D u_win1;
uniform sampler2D u_glitter;
uniform sampler2D u_anti;

#define ANTI_MODE  1
#define ANTI_READY 1
#define ANTI_GAIN  0.3
#define ANTI_K     12.0
const vec3 A_TINT0 = vec3(1.00, 0.55, 0.20);
const vec3 A_TINT1 = vec3(0.40, 1.00, 0.60);
const vec3 A_TINT2 = vec3(0.45, 0.70, 1.00);

#define GLITTER_READY 1
#define GLITTER_GAIN  0.55
#define GLITTER_HP_LOD 3.0
#define GLITTER_HP_AMT 1.0
#define GLITTER_BAND_W vec3(0.0, 0.0, 1.0)
#define GLITTER_BLEND    1
#define GLITTER_OPACITY  0.5
#define GLITTER_HL_GAP   0.45
#define GLITTER_BLEND_GAIN 1.8
#define GLITTER_OVL      0.0
#define GLITTER_EQ       1
#define GLITTER_EQ_GAMMA 1.5
#define GLITTER_EQ_LC    3.0
#define GLITTER_K     12.0
const vec3 G_TINT0 = vec3(0.30, 0.80, 1.00);
const vec3 G_TINT1 = vec3(1.00, 0.60, 0.25);
const vec3 G_TINT2 = vec3(0.85, 0.90, 1.00);
const vec3 G_BASE  = vec3(0.55, 0.85, 1.00);

#define NEB_EDGE_FADE 0.06
vec2 toTex(vec2 c) { return vec2((c.x - (CX - HALF)) / (2.0 * HALF), (c.y + HALF) / (2.0 * HALF)); }

float gZoom = 1.0;
float gLodFine = 0.0;

vec4 nebTex(vec2 t, float lod) {
    if (any(lessThan(t, vec2(0.0))) || any(greaterThan(t, vec2(1.0)))) return vec4(0.0);
    vec4 g = textureLod(u_nebula, t, max(lod, 0.0));
    vec2 me = min(t, 1.0 - t);
    g *= smoothstep(0.0, NEB_EDGE_FADE, min(me.x, me.y));
    vec2 c = vec2(CX - HALF, -HALF) + t * (2.0 * HALF);
    vec4 v = g;
    float fz0 = smoothstep(WIN_FADE_LO, WIN_FADE_HI, gZoom) * float(WIN_READY);
    if (fz0 > 0.0) {
        vec2 tw = (c - vec2(WIN[0].x - WIN[0].y, -WIN[0].y)) / (2.0 * WIN[0].y);
        if (all(greaterThanEqual(tw, vec2(0.0))) && all(lessThanEqual(tw, vec2(1.0)))) {
            vec2 m = min(tw, 1.0 - tw);
            float edge = smoothstep(0.0, 0.04, min(m.x, m.y));
            float lodW = max(lod - log2(HALF / WIN[0].y), 0.0);
            v = mix(v, textureLod(u_win0, tw, lodW), fz0 * edge);
        }
    }
#if WIN1_READY
    float fz1 = smoothstep(WIN1_FADE_LO, WIN1_FADE_HI, gZoom);
    if (fz1 > 0.0) {
        vec2 tw = (c - vec2(WIN[1].x - WIN[1].y, -WIN[1].y)) / (2.0 * WIN[1].y);
        if (all(greaterThanEqual(tw, vec2(0.0))) && all(lessThanEqual(tw, vec2(1.0)))) {
            vec2 m = min(tw, 1.0 - tw);
            float edge = smoothstep(0.0, 0.04, min(m.x, m.y));
            float lodW = max(lod - log2(HALF / WIN[1].y), 0.0);
            v = mix(v, textureLod(u_win1, tw, lodW), fz1 * edge);
        }
    }
#endif
    return v;
    return g;
}

float canyonReach(vec2 p, float lodCopy) {
    vec2 q = DROSTE_F + (p - DROSTE_F) / DROSTE_CANYON_G;
    vec4 vb = nebTex(toTex(q), DROSTE_SIL_LOD);
    float body = smoothstep(log(DROSTE_SIL_LO), log(DROSTE_SIL_HI), log(1e-3 + vb.b + 0.5 * vb.g));
    float tT = clamp((q.x + 1.35) / (-1.84 + 1.35), 0.0, 1.0);
    float tB = clamp((q.x - 0.50) / (1.14 - 0.50), 0.0, 1.0);
    float inT = step(-1.84, q.x) * step(q.x, -1.35);
    float inB = step(0.50, q.x) * step(q.x, 1.14);
    float w = DROSTE_TAIL_W * mix(1.0, 0.33, inT > 0.5 ? tT : tB);
    float onAxis = 1.0 - smoothstep(0.0, w, abs(q.y - DROSTE_F.y));
    float core = max(body, onAxis * max(inT, inB));
    vec4 vo = nebTex(toTex(q), DROSTE_FALL_LOD);
    float outer = DROSTE_FALL_MAX * smoothstep(log(DROSTE_FALL_LO), log(DROSTE_SIL_LO), log(1e-3 + vo.b + 0.5 * vo.g));
    return max(core, outer);
}

vec4 cutCanyon(vec4 raw, vec4 inner, float reach) {
    float h = log(1e-3 + raw.b + 0.5 * raw.g);
    float thr = mix(-9.0, 5.5, reach);
    float d = thr - h;
    float aB = smoothstep(DROSTE_RIM.x, DROSTE_RIM.y, d);
    float aM = smoothstep(DROSTE_RIM.x + 0.1, DROSTE_RIM.y + 0.25, d);
    float aS = smoothstep(DROSTE_RIM.y, DROSTE_RIM.z, d);
    float flare = DROSTE_FLARE * exp(-pow((d - DROSTE_RIM.y) / 0.35, 2.0));
    vec4 o = vec4(mix(raw.r, inner.r, aS), mix(raw.g, inner.g, aM), mix(raw.b, inner.b, aB), 0.0);
    o.r += flare * (1.0 - aS) * max(raw.r, 0.3);
    return o;
}

float gLoopA = 0.0;
float gLoopFlash = 0.0;
float gLoopGlow = 0.0;
#define LOOP_CUT_LO 0.45
#define LOOP_CUT_HI 0.51
#define LOOP_SOLID_LO (0.515 * LOOP_Z6)
#define LOOP_FLASH_LO 30.0
#define LOOP_FLASH_HI 60.0
#define LOOP_FLASH_BASE 0.04
#define LOOP_FLASH_GAIN 0.9
const vec3 LOOP_TINT = vec3(0.71, 0.86, 1.00);
#define LOOP_TINT_SAT 1.8
#define LOOP_TINT_WHITE 0.85
#define LOOP_TINT_HOLD (0.184 * LOOP_Z6)
#define LOOP_TINT_FREE (0.66 * LOOP_Z6)

vec4 loopTex(vec2 t, float lod) {
    vec2 c = vec2(CX - HALF, -HALF) + t * (2.0 * HALF);
    float gz = gZoom; gZoom = gz / LOOP_K;
    vec4 v = nebTex(toTex(loopMap(c)), lod + log2(LOOP_K));
    gZoom = gz;
    return v;
}

float loopSil(vec2 t) {
    vec2 c = vec2(CX - HALF, -HALF) + t * (2.0 * HALF);
    float gz = gZoom; gZoom = gz / LOOP_K;
    vec4 vb = nebTex(toTex(loopMap(c)), DROSTE_SIL_LOD);
    gZoom = gz;
    return smoothstep(log(DROSTE_SIL_LO), log(DROSTE_SIL_HI), log(1e-3 + vb.b + 0.5 * vb.g));
}
#define BODY_SMOKE_CUT 0.5
#define LOOP_KEY_LO 0.004
#define LOOP_KEY_HI 0.03
float loopKey(vec4 v) {
    return smoothstep(log(LOOP_KEY_LO), log(LOOP_KEY_HI), log(1e-4 + v.b + 0.5 * v.g));
}
vec2 loopW(vec2 t, vec4 vc) {
    float k = loopKey(vc);
    return vec2(1.0 - gLoopA * k, gLoopA * k + (1.0 - gLoopA) * gLoopFlash * loopSil(t));
}

vec4 sceneTex(vec2 t, float lod) {
    vec4 v = nebTex(t, lod);
#if LOOP_ON
    if (gLoopA + gLoopFlash > 0.0) {
        vec4 vc = loopTex(t, lod);
        vec2 w = loopW(t, vc);
        v = v * w.x + vc * w.y;
    }
    if (gLoopGlow > 0.0) {
        vec2 c = vec2(CX - HALF, -HALF) + t * (2.0 * HALF);
        float gz = gZoom; gZoom = gz * LOOP_K;
        vec4 vh = nebTex(toTex(LOOP_S6 + (c - LOOP_S0) / LOOP_K), lod - log2(LOOP_K));
        gZoom = gz;
        v = mix(v, vh, gLoopGlow * (1.0 - loopKey(v)));
    }
#endif
#if DROSTE_ON
    vec2 c = vec2(CX - HALF, -HALF) + t * (2.0 * HALF);
    vec2 p[4]; float reach[4]; vec4 raw[4];
    p[0] = c; raw[0] = v;
    int K = 0;
    float sk = 1.0;
    for (int k = 1; k <= 3; k++) {
        sk *= DROSTE_S;
        p[k] = DROSTE_F + (c - DROSTE_F) * sk;
        reach[k] = canyonReach(p[k], lod + log2(sk));
        if (reach[k] <= 0.0) break;
        raw[k] = nebTex(toTex(p[k]), lod + log2(sk));
        K = k;
    }
    if (K == 0) return v;
    vec4 pic = raw[K];
    for (int k = K - 1; k >= 0; k--) pic = cutCanyon(raw[k], pic, reach[k + 1]);
    return pic;
#else
    return v;
#endif
}
float layerAt(vec2 t, int i) {
    if (any(lessThan(t, vec2(0.0))) || any(greaterThan(t, vec2(1.0)))) return 0.0;
    return sceneTex(t, gLodFine)[i];
}
float hBody(vec2 t, float lod) {
    vec4 v = sceneTex(t, lod);
    return log(1e-3 + v.b + 0.5 * v.g);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;

    float zoom, rot = 0.0;
    vec2 centre;
#if AUTO_DIVE
    float tc_ = (iTime + BASS_DIVE * u_bass_accum) / CYCLE;
    int k = int(floor(tc_));
    float ph = fract(tc_);
    float depth = smoothstep(0.05, 0.40, ph) * (1.0 - smoothstep(0.62, 0.95, ph));
    zoom = ZOOM0 * pow(ZMAX / ZOOM0, depth);
    vec2 overview = vec2(CX - 0.15, 0.0);
    vec2 tgt = target(k);
    tgt += (0.06 / ZMAX) * vec2(sin(6.2831 * ph + float(k)), cos(4.7 * ph + 1.3 * float(k)));
    centre = mix(overview, tgt, smoothstep(0.0, 1.0, depth));
#else
    vec4 view = texelFetch(iChannel0, ivec2(0, 0), 0);
    centre = view.xy; zoom = view.z; rot = view.w;
    if (zoom < 0.5) { centre = vec2(CX - 0.15, 0.0); zoom = 1.15; }
#endif

    vec2 sv = vec2(-uv.y, uv.x);
    sv = vec2(cos(rot) * sv.x - sin(rot) * sv.y, sin(rot) * sv.x + cos(rot) * sv.y);
    vec2 c = centre + (HALF / zoom) * sv;
    vec2 tc = toTex(c);
    vec2 tcC = toTex(centre);
    float px = 1.0 / (zoom * iResolution.y);
    gZoom = zoom;
    gLodFine = log2(16384.0 * px);
#if LOOP_ON
    {
        float dView = length(centre - LOOP_S6) * zoom / HALF;
        gLoopA = smoothstep(log(LOOP_SOLID_LO), log(LOOP_Z6 * 0.999), log(zoom))
               * (1.0 - smoothstep(0.5, 1.0, dView));
        gLoopFlash = smoothstep(log(LOOP_FLASH_LO), log(LOOP_FLASH_HI), log(zoom))
                   * (1.0 - smoothstep(1.2, 2.0, dView))
                   * (LOOP_FLASH_BASE + LOOP_FLASH_GAIN * (0.6 * u_bass + 0.8 * u_onset));
    }
    gLoopGlow = gLoopA > 0.0 ? 0.0 : 1.0 - smoothstep(LOOP_CUT_LO, LOOP_CUT_HI, centre.x);
    vec2 tcL = toTex(loopMap(c));
    float lodL = gLodFine + log2(LOOP_K);
#endif
    float lodG = gLodFine + 2.0;

    float e = 6.0 * px;
    vec2 g = vec2(hBody(tc + vec2(e, 0.0), lodG) - hBody(tc - vec2(e, 0.0), lodG),
                  hBody(tc + vec2(0.0, e), lodG) - hBody(tc - vec2(0.0, e), lodG)) * 0.5;

    float kick = u_bass * EFF_PARALLAX;
    vec2 tcLong  = tcC + (tc - tcC) * (1.0 + 0.07 * kick);
    vec2 tcMid   = tcC + (tc - tcC) * (1.0 + 0.03 * kick);
    float highs = 0.5 * (u_presence + u_brilliance);
    vec2 warp = g * (14.0 * px) * (0.3 + 1.5 * highs) * EFF_WARP;
    tcMid += warp;
    vec2 tcShort = tc + 1.6 * warp;

    float dL = layerAt(tcLong, 2), dM = layerAt(tcMid, 1), dS = layerAt(tcShort, 0);

    float gLong  = 0.8 + 1.6 * u_bass + 0.8 * u_onset;
    float gMid   = 0.7 + 1.4 * u_mid;
    float gShort = 0.6 + 1.0 * u_presence + 1.0 * u_brilliance;

    float cph = 0.25 * u_color_phase;
    vec3 cLong  = mix(vec3(1.00, 0.45, 0.12), vec3(1.00, 0.25, 0.35), 0.5 + 0.5 * sin(cph));
    vec3 cMid   = mix(vec3(0.25, 0.95, 0.55), vec3(0.20, 0.75, 0.95), 0.5 + 0.5 * sin(cph + 2.1));
    vec3 cShort = mix(vec3(0.35, 0.40, 1.00), vec3(0.65, 0.35, 1.00), 0.5 + 0.5 * sin(cph + 4.2));

    vec3 aL = cLong  * tone(dL, FLOOR_LONG, gLong);
    vec3 aM = cMid   * tone(dM, FLOOR_MID, gMid) * 0.8;
    vec3 aS = cShort * tone(dS, FLOOR_SHORT, gShort) * 0.55;
#if LOOP_ON
    {
        float kB = 0.0;
        if (gLoopGlow > 0.0) kB = gLoopGlow * loopKey(nebTex(tc, gLodFine));
        else if (gLoopA > 0.0) kB = gLoopA * loopKey(loopTex(tc, gLodFine));
        aS *= 1.0 - BODY_SMOKE_CUT * kB;
    }
#endif

    float lodFine = gLodFine;
    float hFull = hBody(tc, lodFine);

    vec3 smoke = aS + aM;
    vec3 hot = clamp(aL * (0.4 + 1.2 * u_mid) * EFF_DODGE, 0.0, 0.95);
    vec3 light = (smoke + aL) / max(1.0 - hot, 0.05);
#if RELIEF_MODE == 0
    vec3 nrm = normalize(vec3(-g * 3.0, 1.0));
    float lit = 0.35 + 0.9 * max(dot(nrm, normalize(vec3(-0.45, 0.5, 0.74))), 0.0);
    light = mix(light, max(light + lit - 1.0, 0.0), 0.6 * EFF_RELIEF);
#else
    float terrStep = (TERR_HI - TERR_LO) / float(TERR_N);
    float lift = 0.8 * terrStep * u_onset_smooth;
    float hq = (hFull + lift - TERR_LO) / terrStep;
    float lvl = clamp(floor(hq), 0.0, float(TERR_N));
    float fq = fract(hq);
    float dLine = min(fq, 1.0 - fq) / max(fwidth(hq), 1e-5);
    float line = (1.0 - smoothstep(0.5, 1.5, dLine)) * step(0.5, lvl);
    vec2 toL = normalize(toTex(centre + (HALF / zoom) * vec2(-0.45, -0.5)) - tcC);
    float shadow = 0.0, rim = 0.0;
    for (int i = 1; i <= 2; i++) {
        float dist = float(i) * TERR_SHADOW_PX * px;
        float lUp = floor((hBody(tc + toL * dist, lodFine) + lift - TERR_LO) / terrStep);
        float lDn = floor((hBody(tc - toL * dist, lodFine) + lift - TERR_LO) / terrStep);
        shadow = max(shadow, clamp(lUp - lvl, 0.0, 3.0) / 3.0);
        rim = max(rim, clamp(lvl - lDn, 0.0, 1.0));
    }
    float lit = (0.45 + 0.55 * lvl / float(TERR_N)) * (1.0 - 0.7 * shadow) + 0.25 * rim;
    lit *= 1.0 - 0.35 * line;
    light *= mix(1.0, lit, EFF_RELIEF);
#endif
    float hl = hBody(tc, lodG);
    float hs = hFull;
#if EDGE_MODE == 0
    float thr = log(0.6) - 3.2 * u_onset_smooth;
    float front = smoothstep(thr, thr + 0.12, hs) * (1.0 - smoothstep(thr + 0.12, thr + 0.24, hs));
#else
    float esc = mix(EDGE_S_MIN, EDGE_S_MAX, u_onset_smooth);
    vec2 eq = c / esc;
    vec4 ev = textureLod(u_nebula, clamp(toTex(eq), 0.0, 1.0), EDGE_SHAPE_LOD);
    float inside = smoothstep(log(0.004), log(0.03), log(1e-3 + ev.b + 0.5 * ev.g));
    float surf = mix(-6.0, 1.5, inside);
    float front = 1.0 - smoothstep(0.08, 0.18, abs(hs - surf));
    front *= step(-6.2, hs);
#endif
    light += front * vec3(0.55, 0.80, 1.00) * (0.3 + 2.0 * u_onset_smooth) * EFF_SLICE;
#if RELIEF_MODE == 1
    float bodyW = smoothstep(TERR_GLOW_LO, TERR_GLOW_HI, lvl / float(TERR_N));
    float glowLine = exp(-dLine * dLine / 2.0);
    light += glowLine * bodyW * TERR_GLOW_COL * (0.3 + 2.0 * u_onset_smooth) * TERR_GLOW;
#endif

    vec3 w = vec3(tone(dS, FLOOR_SHORT, 1.0).x, tone(dM, FLOOR_MID, 1.0).x, tone(dL, FLOOR_LONG, 1.0).x) + 1e-4;
    vec3 hue = (cShort * w.x + cMid * w.y + cLong * w.z) / (w.x + w.y + w.z);
    float hFine = hFull;
    float detail = 0.5 + 0.5 * tanh(1.3 * (hFine - hl));
    float presence = 1.0 - exp(-3.0 * (w.x + w.y + w.z));
    vec3 albedo = mix(vec3(0.5), mix(vec3(0.75), hue, 0.6) * (0.35 + 0.9 * detail), presence);

    vec3 L = 1.0 - exp(-light);
    vec3 col;
    if (BLEND_MODE == 0) {
        col = 1.0 - exp(-2.0 * albedo * light);
    } else if (BLEND_MODE == 1) {
        col = mix(2.0 * albedo * L, 1.0 - 2.0 * (1.0 - albedo) * (1.0 - L), step(0.5, albedo));
    } else {
        col = (1.0 - 2.0 * L) * albedo * albedo + 2.0 * L * albedo;
    }
    col *= smoothstep(0.0, 0.08, L.r + L.g + L.b);
    col = clamp(col, 0.0, 1.0);

    float edge = clamp(EDGE_K * length(g) + 2.0 * abs(detail - 0.5), 0.0, 1.0);
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    float v = pow(luma, GRADE_CONTRAST) * mix(GRADE_FLAT, GRADE_EDGE, edge);
    vec3 chroma = (col - luma) * (v / max(luma, 1e-4));
    col = v + chroma * mix(SAT_FLAT, SAT_EDGE, edge);
#if ANTI_READY && ANTI_MODE == 0
    vec3 an = texture(u_anti, clamp(tc, 0.0, 1.0)).rgb;
    vec3 ac = log(1.0 + ANTI_K * max(an, 0.0)) / log(1.0 + ANTI_K);
    vec3 A = (A_TINT0 * ac.r + A_TINT1 * ac.g + A_TINT2 * ac.b) * ANTI_GAIN * (0.85 + 0.3 * u_bass);
    col = 1.0 - (1.0 - clamp(A, 0.0, 1.0)) * (1.0 - clamp(col, 0.0, 1.0));
#endif
#if GLITTER_READY
    vec3 gl = textureLod(u_glitter, clamp(tc, 0.0, 1.0), max(gLodFine, 0.0)).rgb;
    vec3 glb = textureLod(u_glitter, clamp(tc, 0.0, 1.0), max(gLodFine, 0.0) + GLITTER_HP_LOD).rgb;
#if LOOP_ON
    vec3 grL = vec3(0.0), glbL = vec3(0.0);
    float kc = gLoopA > 0.0 ? gLoopA * loopSil(tc) : 0.0;
    if (kc > 0.0) {
        float inL = step(0.0, min(min(tcL.x, tcL.y), 1.0 - max(tcL.x, tcL.y)));
        grL = inL * textureLod(u_glitter, clamp(tcL, 0.0, 1.0), max(lodL, 0.0)).rgb;
        glbL = inL * textureLod(u_glitter, clamp(tcL, 0.0, 1.0), max(lodL, 0.0) + GLITTER_HP_LOD).rgb;
        gl = mix(gl, grL, kc); glb = mix(glb, glbL, kc);
    }
    vec2 tcH = toTex(LOOP_S6 + (c - LOOP_S0) / LOOP_K);
    float lodH = gLodFine - log2(LOOP_K);
    float kh = 0.0;
    vec3 grH = vec3(0.0);
    if (gLoopGlow > 0.0) {
        vec4 vb = nebTex(tc, DROSTE_SIL_LOD);
        float silV = smoothstep(log(DROSTE_SIL_LO), log(DROSTE_SIL_HI), log(1e-3 + vb.b + 0.5 * vb.g));
        kh = gLoopGlow * (1.0 - silV);
        grH = textureLod(u_glitter, clamp(tcH, 0.0, 1.0), max(lodH, 0.0)).rgb;
        vec3 glbH = textureLod(u_glitter, clamp(tcH, 0.0, 1.0), max(lodH, 0.0) + GLITTER_HP_LOD).rgb;
        gl = mix(gl, grH, kh); glb = mix(glb, glbH, kh);
    }
#endif
    gl = max(gl - GLITTER_HP_AMT * glb, 0.0);
#if GLITTER_EQ
    vec3 gr = textureLod(u_glitter, clamp(tc, 0.0, 1.0), max(gLodFine, 0.0)).rgb;
#if LOOP_ON
    gr = mix(mix(gr, grL, kc), grH, kh);
#endif
    vec3 gd = clamp(GLITTER_EQ_LC * (gr - glb), 0.0, 1.0);
    vec3 gc = pow(gr, vec3(GLITTER_EQ_GAMMA)) * gd;
#else
    vec3 gc = log(1.0 + GLITTER_K * max(gl, 0.0)) / log(1.0 + GLITTER_K);
#endif
    vec3 bandHue = (G_TINT0 * gc.r + G_TINT1 * gc.g + G_TINT2 * gc.b) / max(gc.r + gc.g + gc.b, 1e-4);
#if ANTI_READY && ANTI_MODE == 1
    vec3 an2 = textureLod(u_anti, clamp(tc, 0.0, 1.0), max(gLodFine - 1.0, 0.0) + 2.0).rgb;
#if LOOP_ON
    if (kc > 0.0)
        an2 = mix(an2, textureLod(u_anti, clamp(tcL, 0.0, 1.0), max(lodL - 1.0, 0.0) + 2.0).rgb, kc);
    if (kh > 0.0)
        an2 = mix(an2, textureLod(u_anti, clamp(tcH, 0.0, 1.0), max(lodH - 1.0, 0.0) + 2.0).rgb, kh);
#endif
    vec3 ac2 = log(1.0 + ANTI_K * max(an2, 0.0)) / log(1.0 + ANTI_K);
    float asum = ac2.r + ac2.g + ac2.b;
    vec3 petalHue = (A_TINT0 * ac2.r + A_TINT1 * ac2.g + A_TINT2 * ac2.b) / max(asum, 1e-4);
    vec3 glHue = mix(G_BASE, petalHue, smoothstep(0.02, 0.25, asum));
    glHue = mix(glHue, bandHue, 0.3);
#else
    vec3 glHue = bandHue;
#endif
    vec3 G = glHue * dot(gc, GLITTER_BAND_W)
           * GLITTER_GAIN * (0.8 + 0.6 * (u_presence + u_brilliance) * 0.5);
    vec3 cb0 = clamp(col, 0.0, 1.0), Gb = clamp(G * GLITTER_BLEND_GAIN, 0.0, 1.0);
    vec3 blended;
    if (GLITTER_BLEND == 1) {
        vec3 hb = GLITTER_HL_GAP + (1.0 - GLITTER_HL_GAP) * Gb;
        blended = mix(2.0 * cb0 * hb, 1.0 - 2.0 * (1.0 - cb0) * (1.0 - hb), step(0.5, hb));
    }
    else if (GLITTER_BLEND == 2)
        blended = max(cb0 + Gb - 1.0, 0.0);
    else
        blended = 1.0 - (1.0 - cb0) * (1.0 - Gb);
    col = mix(cb0, blended, GLITTER_OPACITY);
    float ob = mix(0.5, clamp(gc.g, 0.0, 1.0), GLITTER_OVL);
    vec3 cb = clamp(col, 0.0, 1.0);
    col = mix(2.0 * cb * ob, 1.0 - 2.0 * (1.0 - cb) * (1.0 - ob), step(0.5, cb));
#endif
#if LOOP_ON
    if (gLoopA + gLoopFlash > 0.0) {
        vec4 hv = nebTex(tc, lodG), cv = loopTex(tc, lodG);
        vec2 w = loopW(tc, cv);
        float hb = (hv.r + hv.g + hv.b) * w.x;
        float cb = (cv.r + cv.g + cv.b) * w.y;
        float share = cb / (cb + hb + 1e-3);
        float tintAmt = 1.0 - smoothstep(log(LOOP_TINT_HOLD), log(LOOP_TINT_FREE), log(zoom));
        const vec3 LW = vec3(0.2126, 0.7152, 0.0722);
        float tl = dot(LOOP_TINT, LW);
        vec3 tint = max(tl + (LOOP_TINT - tl) * LOOP_TINT_SAT, 0.0);
        tint /= dot(tint, LW);
        float peak = clamp(0.6 * u_bass + 0.8 * u_onset, 0.0, 1.0);
        tint = mix(tint, vec3(1.0), LOOP_TINT_WHITE * peak);
        float lum = dot(col, LW);
        col = mix(col, lum * tint, share * tintAmt);
    }
#endif
    col = clamp(col, 0.0, 1.0);
    {
        float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
        float ls = mix(l, l * l * (3.0 - 2.0 * l), FINAL_CONTRAST);
        col = clamp(col * (ls / max(l, 1e-4)), 0.0, 1.0);
        l = dot(col, vec3(0.2126, 0.7152, 0.0722));
        col = clamp(l + (col - l) * FINAL_SAT, 0.0, 1.0);
    }
    fragColor = vec4(col, 1.0);
}
