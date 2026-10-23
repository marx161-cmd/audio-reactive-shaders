// Ember-Flock-Metal-Pit: image pass.
//
// Two flocks of burning embers led by hunter heads, seen from above in a pit
// whose walls are black matte liquid metal that wrinkles where embers hit and
// reflects the flocks. Dense self-shadowed smoke rises toward the camera.
// Movement = the phone version's choreography (pockets, figures, eye-catchers)
// run on the desktop; the 18 steering knobs ship at the tuned setting.

uniform usampler2D u_glowI;
uniform usampler2D u_glowH;
uniform usampler2D u_glowR;
uniform sampler2D u_bloomA;
uniform sampler2D u_bloomB;
#define FIX       1024.0

#define BB_T_COLD   1000.0
#define BB_T_SPAN   2600.0
#define BB_HEAT_P   1.15
#define BB_T_REF    2800.0
#define BB_SB_EXP   4.0
#define BB_L_MIN    0.18
#define BB_L_MAX    6.0
#define DBG_LEAD_GREEN 0
#define DBG_LEAD_HEAT  2.5
vec2 bbLocus(float T) {
    if (T < 1667.0) {
        T = max(T, 1000.0);
        float u = (0.860117757 + 1.54118254e-4 * T + 1.28641212e-7 * T * T) / (1.0 + 8.42420235e-4 * T + 7.08145163e-7 * T * T);
        float v = (0.317398726 + 4.22806245e-5 * T + 4.20481691e-8 * T * T) / (1.0 - 2.89741816e-5 * T + 1.61456053e-7 * T * T);
        float d = 2.0 * u - 8.0 * v + 4.0;
        return vec2(3.0 * u / d, 2.0 * v / d);
    }
    T = clamp(T, 1667.0, 25000.0);
    float t2 = T * T, t3 = t2 * T;
    float x = (T < 4000.0)
        ? (-0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / T + 0.179910)
        : (-3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / T + 0.240390);
    float y = (T < 2222.0)
        ? (-1.1063814 * x * x * x - 1.34811020 * x * x + 2.18555832 * x - 0.20219683)
        : ((T < 4000.0)
            ? (-0.9549476 * x * x * x - 1.37418593 * x * x + 2.09137015 * x - 0.16748867)
            : (3.0817580 * x * x * x - 5.87338670 * x * x + 3.75112997 * x - 0.37001483));
    return vec2(x, y);
}
vec3 bbLinear(float T) {
    vec2 xy = bbLocus(T);
    vec3 XYZ = vec3(xy.x / xy.y, 1.0, (1.0 - xy.x - xy.y) / xy.y);
    vec3 rgb = vec3( 3.2404542 * XYZ.x - 1.5371385 * XYZ.y - 0.4985314 * XYZ.z,
                    -0.9692660 * XYZ.x + 1.8760108 * XYZ.y + 0.0415560 * XYZ.z,
                     0.0556434 * XYZ.x - 0.2040259 * XYZ.y + 1.0572252 * XYZ.z);
    rgb = max(rgb, vec3(0.0));
    return rgb / max(max(rgb.r, max(rgb.g, rgb.b)), 1e-5);
}
vec3 bbRadiance(uint Ir, uint Hr, float gain) {
    float I = float(Ir) / FIX;
    if (I <= 0.0) return vec3(0.0);
    float heat = float(Hr) / FIX / I;
    float T = BB_T_COLD + BB_T_SPAN * pow(clamp(heat, 0.0, 1.0), BB_HEAT_P);
    float L = clamp(pow(T / BB_T_REF, BB_SB_EXP), BB_L_MIN, BB_L_MAX);
#if DBG_LEAD_GREEN
    if (heat > DBG_LEAD_HEAT * 0.5) return vec3(0.0, 1.0, 0.0) * L * I * gain;
#endif
    return bbLinear(T) * L * I * gain;
}
uniform sampler2D u_backdrop;
uniform sampler2D u_glass;
uniform sampler2D u_smoke;
#define SMOKE_ON      1
#define SMOKE_ALBEDO  0.35
#define SMOKE_EMBER   1.00
#define SMOKE_VOL      1
#define SMOKE_VOL_K    0.80
#define SMOKE_LV_K     1.0
#define SMOKE_VOL_COL  vec3(1.0, 0.22, 0.04)
#define SMOKE_AMB_BLOOM 0.00
#define SMOKE_BACK_COL vec3(1.0, 0.35, 0.15)
#define SMOKE_BACK    0.00
#define GLASS_ON     0
#define GLASS_BASE   0.006
#define GLASS_BACK   0.60
#define GLASS_BACK_COL vec3(1.0, 0.02, 0.22)
#define GLASS_LIGHT  1.20
#define GLASS_WIDE   1.00
#define GLASS_TIGHT  0.35
#define LEAD_COL     0.0
vec3 glassAt(vec2 fragCoord, vec2 res, vec2 uvB) {
    vec2 tsz = vec2(textureSize(u_backdrop, 0));
    vec2 st = vec2(0.5 + (fragCoord.x - 0.5 * res.x) / res.y * (tsz.y / tsz.x), fragCoord.y / res.y);
    float lod = max(log2(float(textureSize(u_backdrop, 0).y) / res.y), 0.0);
    vec3 c = textureLod(u_backdrop, st, lod).rgb;
    c = pow(c, vec3(2.2));
    vec4 g = textureLod(u_glass, st, lod);
    float lead = g.r, edge = g.b, var = g.a * 2.0 - 1.0;
    float tone = mix(0.7, 1.15, smoothstep(0.0, 0.6, edge)) * (1.0 + 0.35 * var);
    vec3 light = GLASS_WIDE * texture(u_bloomB, uvB).rgb + GLASS_TIGHT * texture(u_bloomA, uvB).rgb;
    vec3 lit = c * (GLASS_BASE + (GLASS_BACK * GLASS_BACK_COL + GLASS_LIGHT * light) * tone);
    return mix(lit, vec3(LEAD_COL), lead);
}
#define BOX_ON     1
#define BOX_ZN    -0.25
#define BOX_ZB     2.50
#define BOX_CAM_D  3.0
#define BOX_FOCAL  1.9
#define BOX_TILE   2.0
#define BOX_SHADE_BACK 2.50
#define BOX_SHADE_SIDE 0.80
#define BOX_SHADE_FLR  0.65
uniform sampler2D u_wallC;  uniform sampler2D u_wallD;
uniform sampler2D u_flrC;   uniform sampler2D u_flrD;
uniform sampler2D u_backC;  uniform sampler2D u_backD;
#define RED_ONLY   1
#define RED_M0     0.30
#define RED_M1     0.65
#define RED_OTHER  0.15
#define RED_GAIN   1.6
#define CANDLE_ON     1
#define CANDLE_FLICK  0.55
#define CANDLE_DIP    0.45
#define CANDLE_SHIM   0.50
#define CANDLE_SHIM_S 9.0
float candle(float pr, vec2 st) {
    float t = iTime, ph = 6.2831853 * pr;
    float f = 0.55 * sin(t * (5.1 + 2.3 * pr) + ph) + 0.30 * sin(t * (11.7 + 4.1 * pr) + 2.7 * ph) + 0.15 * sin(t * (23.0 + 7.0 * pr) + 5.3 * ph);
    float dip = smoothstep(0.92, 1.0, sin(t * (0.9 + 0.6 * pr) + 3.1 * ph));
    float sh = sin(dot(st, vec2(CANDLE_SHIM_S, CANDLE_SHIM_S * 0.7)) + t * (1.7 + pr) + ph)
             * sin(dot(st, vec2(-CANDLE_SHIM_S * 0.6, CANDLE_SHIM_S)) - t * (1.1 + 0.5 * pr) + 1.9 * ph);
    return max(0.0, 1.0 + CANDLE_FLICK * f - CANDLE_DIP * dip + CANDLE_SHIM * sh);
}
vec3 paneLit(vec3 c, vec4 g, vec3 light, vec2 st) {
    c = pow(c, vec3(2.2));
    float lead = g.r, edge = g.b, var = g.a * 2.0 - 1.0;
    float tone = mix(0.7, 1.15, smoothstep(0.0, 0.6, edge)) * (1.0 + 0.35 * var);
#if RED_ONLY
    float redness = clamp((c.r - max(c.g, c.b)) / max(c.r, 1e-3), 0.0, 1.0);
    float rm = smoothstep(RED_M0, RED_M1, redness);
    float cdl = 1.0;
#if CANDLE_ON
    cdl = candle(g.g, st);
#endif
    vec3 lit = c * (GLASS_BASE + (RED_GAIN * rm * cdl * GLASS_BACK * GLASS_BACK_COL
                                  + GLASS_LIGHT * light * mix(RED_OTHER, RED_GAIN, rm)) * tone);
#else
    vec3 lit = c * (GLASS_BASE + (GLASS_BACK * GLASS_BACK_COL + GLASS_LIGHT * light) * tone);
#endif
    return mix(lit, vec3(LEAD_COL), lead);
}
vec2 mirrorRep(vec2 t) { return 1.0 - abs(mod(t, 2.0) - 1.0); }
#define EYE_Y        0.00
#define TANK_STEPS   4
#define TANK_R0      0.15
#define TANK_REFL    3.0
#define TANK_SOFT    0.35
#define ROOM_METAL   1
#define LM_SCALE     1.6
#define LM_BUMP      0.6
#define LM_WRAP      0.5
#define LM_F0        0.14
#define LM_TINT      vec3(0.40, 0.38, 0.36)
#define LM_DIFF      0.06
#define LM_SPEC      0.35
#define LM_SPEC_P    10.0
#define LM_RIDGE     0.12
#define LM_LIGHT_R   0.35
#define LM_BASE      0.004
#define LM_EMBER_COL vec3(1.0, 0.38, 0.10)
uniform sampler2D u_flk;
uniform sampler2D u_stir;
#define STIR_TILES   5
#define STIR_ROT     1.0
#define STIR_DENT    0.0
#define STIR_DENT_N  1.0
#define LM_CALM      0.06
#define STIR_FULL    0.8
uniform sampler2D u_ripple;
#define WAVE_H       1.0
#define WAVE_N       0.5
#define WAVE_ADV     0.05
#define WAVE_WAKE    0.04
float waveAt(int tile, vec2 uv) {
    uv = clamp(uv, 0.5 / 256.0, 1.0 - 0.5 / 256.0);
    return texture(u_ripple, vec2(uv.x, (uv.y + float(tile)) / float(STIR_TILES))).x;
}
vec2 stirAt(int tile, vec2 uv) {
    uv = clamp(uv, 0.5 / 256.0, 1.0 - 0.5 / 256.0);
    return texture(u_stir, vec2(uv.x, (uv.y + float(tile)) / float(STIR_TILES))).xy;
}
void lm_rot(inout vec2 p, float a) { float c = cos(a), s = sin(a); p = vec2(c * p.x + s * p.y, -s * p.x + c * p.y); }
float lm_hash(vec2 co) { return fract(sin(dot(co.xy, vec2(12.9898, 58.233))) * 13758.5453); }
float lm_psin(float a) { return 0.5 + 0.5 * sin(a); }
float lm_tanh(float x) { float x2 = x * x; return clamp(x * (27.0 + x2) / (27.0 + 9.0 * x2), -1.0, 1.0); }
float lm_onoise(vec2 x) { x *= 0.5; float a = sin(x.x), b = sin(x.y); return mix(a, b, lm_psin(6.2831853 * lm_tanh(a * b + a + b))); }
float lm_vnoise(vec2 x) {
    vec2 i = floor(x), w = fract(x);
    vec2 u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
    float a = lm_hash(i), b = lm_hash(i + vec2(1.0, 0.0)), c = lm_hash(i + vec2(0.0, 1.0)), d = lm_hash(i + vec2(1.0, 1.0));
    return a + (b - a) * u.x + (c - a) * u.y + (d - c + a - b) * u.x * u.y;
}
float lm_fbm(vec2 p, int oct) {
    vec2 op = p;
    float h = 0.0, d = 0.0, a = 1.0;
    for (int i = 0; i < 7; ++i) {
        if (i >= oct) break;
        h += a * lm_onoise(p); d += a; a *= 0.45;
        p += -vec2(1.23, 1.5); p *= 2.03; lm_rot(p, 1.2);
    }
    return mix(h / d, -0.5 * (h / d), pow(lm_vnoise(0.9 * op), 0.25));
}
float lm_warp(vec2 p, float ph) {
    vec2 v = vec2(lm_fbm(p, 5), lm_fbm(p + 0.7, 5));
    lm_rot(v, 1.0 + iTime * 0.1 + ph);
    vec2 vv = vec2(lm_fbm(p + 3.7 * v, 7), lm_fbm(p - 2.7 * v.yx + 0.7, 7));
    lm_rot(vv, -1.0 + iTime * 0.21315 - 0.7 * ph);
    return lm_fbm(p + 1.4 * vv, 3);
}
float lm_height(vec2 p, float ph) {
    float a = 0.005 * iTime;
    p += 5.0 * vec2(cos(a), sin(a)); p *= 2.0; p += 13.0;
    return 0.35 * lm_tanh(3.0 * lm_warp(p, ph)) / 3.0;
}
#define FLOW_V       0.15
#define FLOW_STRETCH 0.60
#define FLOW_PERIOD  20.0
float lmFlow(vec2 q, vec2 fd, float ph) {
    if (fd == vec2(0.0)) return lm_height(q * LM_SCALE, ph);
    q -= fd * dot(q, fd) * (1.0 - FLOW_STRETCH);
    float a = fract(iTime / FLOW_PERIOD), b = fract(a + 0.5);
    float wa = 1.0 - abs(2.0 * a - 1.0), wb = 1.0 - wa;
    float ha = lm_height((q - fd * (a * FLOW_PERIOD * FLOW_V)) * LM_SCALE, ph);
    float hb = lm_height((q - fd * (b * FLOW_PERIOD * FLOW_V) + vec2(17.3, 5.1)) * LM_SCALE, ph);
    return (wa * ha + wb * hb) / sqrt(wa * wa + wb * wb);
}
vec3 boxAtMetal(vec2 fragCoord, vec2 res) {
    vec2 xy = (2.0 * fragCoord - res) / res.y;
    float edge = res.x / res.y;
    float XW = edge * (BOX_ZN + BOX_CAM_D) / BOX_FOCAL, YW = (BOX_ZN + BOX_CAM_D) / BOX_FOCAL;
    vec3 ro = vec3(0.0, EYE_Y, -BOX_CAM_D);
    vec3 rd = normalize(vec3(xy * (BOX_CAM_D + BOX_ZN) / BOX_FOCAL, BOX_ZN) - ro);
    vec3 bmin = vec3(-XW, -YW, BOX_ZN), bmax = vec3(XW, YW, BOX_ZB);
    vec3 inv = 1.0 / (rd + vec3(1e-7));
    vec3 tmx = max((bmin - ro) * inv, (bmax - ro) * inv);
    float tF = min(min(tmx.x, tmx.y), tmx.z);
    vec3 P = ro + rd * tF;
    vec3 n0; vec2 pw; vec3 tu, tv;
    int tile; vec2 suv, sext;
    float ZL = BOX_ZB - BOX_ZN;
    if (tF == tmx.z)      { n0 = vec3(0.0, 0.0, -1.0);        pw = P.xy;                         tu = vec3(1.0, 0.0, 0.0); tv = vec3(0.0, 1.0, 0.0);
                            tile = 0; suv = vec2((P.x / XW + 1.0) * 0.5, (P.y / YW + 1.0) * 0.5); sext = vec2(2.0 * XW, 2.0 * YW); }
    else if (tF == tmx.x) { n0 = vec3(-sign(P.x), 0.0, 0.0); pw = vec2(P.z * sign(P.x), P.y);   tu = vec3(0.0, 0.0, sign(P.x)); tv = vec3(0.0, 1.0, 0.0);
                            tile = P.x < 0.0 ? 1 : 2; suv = vec2((P.z - BOX_ZN) / ZL, (P.y / YW + 1.0) * 0.5); sext = vec2(ZL, 2.0 * YW); }
    else                  { n0 = vec3(0.0, -sign(P.y), 0.0); pw = vec2(P.x, P.z * sign(P.y));   tu = vec3(1.0, 0.0, 0.0); tv = vec3(0.0, 0.0, sign(P.y));
                            tile = P.y < 0.0 ? 3 : 4; suv = vec2((P.x / XW + 1.0) * 0.5, (P.z - BOX_ZN) / ZL); sext = vec2(2.0 * XW, ZL); }
    vec2 st = stirAt(tile, suv);
    float ph = STIR_ROT * st.x;
    float amp = mix(LM_CALM, 1.0, smoothstep(0.0, STIR_FULL, st.x));
    float su = (tile == 1 || tile == 2) ? sign(P.x) : 1.0;
    float wv = WAVE_H * waveAt(tile, suv);
    vec2 gW;
    {
        float tx = 1.0 / 256.0, ty = 1.0 / 256.0;
        float sv = (tile >= 3 ? sign(P.y) : 1.0);
        gW = WAVE_H * vec2((waveAt(tile, suv + vec2(su * tx, 0.0)) - waveAt(tile, suv - vec2(su * tx, 0.0))) / (2.0 * tx * sext.x),
                           (waveAt(tile, suv + vec2(0.0, sv * ty)) - waveAt(tile, suv - vec2(0.0, sv * ty))) / (2.0 * ty * sext.y));
    }
    vec2 pwR = pw - WAVE_ADV * gW;
    vec2 fd = (tile == 1 || tile == 2) ? vec2(sign(P.x), 0.0) : (tile >= 3 ? vec2(0.0, sign(P.y)) : vec2(0.0));
    amp = max(amp, mix(LM_CALM, 1.0, smoothstep(0.0, WAVE_WAKE, abs(wv))));
    float h0 = amp * lmFlow(pwR, fd, ph);
    float h = h0 - STIR_DENT * st.y + WAVE_N * wv;
    const float eps = 0.012;
    float dxE = (stirAt(tile, suv + vec2(su * eps / sext.x, 0.0)).y - st.y) / eps;
    float dyE = (stirAt(tile, suv + vec2(0.0, (tile >= 3 ? sign(P.y) : 1.0) * eps / sext.y)).y - st.y) / eps;
    float gx = (amp * lmFlow(pwR + vec2(eps, 0.0), fd, ph) - h0) / eps - STIR_DENT * STIR_DENT_N * dxE + WAVE_N * gW.x;
    float gy = (amp * lmFlow(pwR + vec2(0.0, eps), fd, ph) - h0) / eps - STIR_DENT * STIR_DENT_N * dyE + WAVE_N * gW.y;
    vec3 n = normalize(n0 - LM_BUMP * (gx * tu + gy * tv));
    vec3 rr = reflect(rd, n);
    vec3 invr = 1.0 / (rr + vec3(1e-7));
    vec3 tmr = max((bmin - P) * invr, (bmax - P) * invr);
    float sMax = max(min(min(tmr.x, tmr.y), tmr.z), 0.0);
    vec3 refl = vec3(0.0);
    for (int i = 0; i < TANK_STEPS; i++) {
        vec3 Q = P + rr * (sMax * (float(i) + 0.5) / float(TANK_STEPS));
        float s = BOX_FOCAL / max(Q.z + BOX_CAM_D, 0.3);
        vec2 sc = (Q.xy - vec2(0.0, EYE_Y)) * s + vec2(0.0, EYE_Y) * BOX_FOCAL / (BOX_CAM_D + BOX_ZN);
        vec2 tuv = vec2(sc.x / edge, sc.y) * 0.5 + 0.5;
        float inb = step(0.0, tuv.x) * step(tuv.x, 1.0) * step(0.0, tuv.y) * step(tuv.y, 1.0);
        tuv = clamp(tuv, 0.0, 1.0);
        refl += mix(texture(u_bloomA, tuv).rgb, texture(u_bloomB, tuv).rgb, TANK_SOFT) * inb;
    }
    refl /= float(TANK_STEPS);
    float cosv = max(dot(-rd, n), 0.0);
    float F = LM_F0 + (1.0 - LM_F0) * pow(1.0 - cosv, 5.0);
    vec4 Fk0 = texelFetch(u_flk, ivec2(0, 0), 0), Fk1 = texelFetch(u_flk, ivec2(1, 0), 0);
    float lit = 0.0;
    for (int f = 0; f < 2; f++) {
        vec4 Fk = f == 0 ? Fk0 : Fk1;
        vec3 L = Fk.xyz - P; float dl = length(L); L /= max(dl, 1e-4);
        float reach = Fk.w + LM_LIGHT_R;
        float atten = 1.0 / (1.0 + (dl * dl) / (reach * reach));
        float diff = max((dot(n, L) + LM_WRAP) / (1.0 + LM_WRAP), 0.0);
        float spec = pow(max(dot(reflect(-L, n), -rd), 0.0), LM_SPEC_P);
        lit += (LM_DIFF * diff + LM_SPEC * spec) * atten;
    }
    vec3 L0 = Fk0.xyz - P;
    vec2 lp = vec2(dot(L0, tu), dot(L0, tv));
    float oh = amp * lm_height((pw + 0.05 * normalize(lp + vec2(1e-4))) * LM_SCALE, ph);
    float ridge = max(smoothstep(0.0, 0.125, h) - smoothstep(0.0, 0.125, oh), 0.0);
    return LM_TINT * (refl * F * TANK_REFL + LM_BASE) + LM_EMBER_COL * LM_TINT * (lit + LM_RIDGE * ridge);
}
vec3 boxAtTank(vec2 fragCoord, vec2 res, vec2 uvB);
vec3 boxAt(vec2 fragCoord, vec2 res, vec2 uvB) {
#if ROOM_METAL
    return boxAtMetal(fragCoord, res);
#else
    return boxAtTank(fragCoord, res, uvB);
#endif
}
vec3 boxAtTank(vec2 fragCoord, vec2 res, vec2 uvB) {
    vec2 xy = (2.0 * fragCoord - res) / res.y;
    float edge = res.x / res.y;
    float XW = edge * (BOX_ZN + BOX_CAM_D) / BOX_FOCAL, YW = (BOX_ZN + BOX_CAM_D) / BOX_FOCAL;
    vec3 ro = vec3(0.0, EYE_Y, -BOX_CAM_D);
    vec3 rd = normalize(vec3(xy * (BOX_CAM_D + BOX_ZN) / BOX_FOCAL, BOX_ZN) - ro);
    vec3 bmin = vec3(-XW, -YW, BOX_ZN), bmax = vec3(XW, YW, BOX_ZB);
    vec3 inv = 1.0 / (rd + vec3(1e-7));
    vec3 tmx = max((bmin - ro) * inv, (bmax - ro) * inv);
    float tF = min(min(tmx.x, tmx.y), tmx.z);
    vec3 P = ro + rd * tF;
    vec3 n = (tF == tmx.z) ? vec3(0.0, 0.0, -1.0)
           : (tF == tmx.x) ? vec3(-sign(P.x), 0.0, 0.0) : vec3(0.0, -sign(P.y), 0.0);
    vec3 rr = reflect(rd, n);
    vec3 invr = 1.0 / (rr + vec3(1e-7));
    vec3 tmr = max((bmin - P) * invr, (bmax - P) * invr);
    float sMax = min(min(tmr.x, tmr.y), tmr.z);
    vec3 refl = vec3(0.0);
    for (int i = 0; i < TANK_STEPS; i++) {
        vec3 Q = P + rr * (sMax * (float(i) + 0.5) / float(TANK_STEPS));
        float s = BOX_FOCAL / max(Q.z + BOX_CAM_D, 0.3);
        vec2 sc = (Q.xy - vec2(0.0, EYE_Y)) * s + vec2(0.0, EYE_Y) * BOX_FOCAL / (BOX_CAM_D + BOX_ZN);
        vec2 tuv = vec2(sc.x / edge, sc.y) * 0.5 + 0.5;
        float inb = step(0.0, tuv.x) * step(tuv.x, 1.0) * step(0.0, tuv.y) * step(tuv.y, 1.0);
        tuv = clamp(tuv, 0.0, 1.0);
        refl += mix(texture(u_bloomA, tuv).rgb, texture(u_bloomB, tuv).rgb, TANK_SOFT) * inb;
    }
    refl /= float(TANK_STEPS);
    float cosv = abs(dot(rd, n));
    float fres = TANK_R0 + (1.0 - TANK_R0) * pow(1.0 - cosv, 5.0);
    return refl * fres * TANK_REFL;
}
#define EMB_GAIN   1.38
#define BLOOM_A    0.55
#define BLOOM_B    0.85
#define EXPOSURE   1.0
#define FILMIC     0
#define HUE_KEEP   1
#define SRGB_OUT   1

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 p = ivec2(fragCoord);
    vec3 rad = bbRadiance(texelFetch(u_glowI, p, 0).r, texelFetch(u_glowH, p, 0).r, EMB_GAIN);
    vec2 uv = (fragCoord) / vec2(textureSize(u_glowI, 0));
    rad += BLOOM_A * texture(u_bloomA, uv).rgb + BLOOM_B * texture(u_bloomB, uv).rgb;
    vec3 bgRad = vec3(0.0);
#if GLASS_ON
    bgRad = glassAt(fragCoord, vec2(textureSize(u_glowI, 0)), uv);
#endif
#if BOX_ON
    bgRad = boxAt(fragCoord, vec2(textureSize(u_glowI, 0)), uv);
#endif
#if SMOKE_ON
    {
        float tauS = texture(u_smoke, uv).r;
        float T = exp(-tauS);
#if SMOKE_VOL
        vec3 Ls = texture(u_smoke, uv).gba;
        vec3 amb = SMOKE_AMB_BLOOM * (texture(u_bloomB, uv).rgb + 0.35 * texture(u_bloomA, uv).rgb)
                 + float(GLASS_ON + BOX_ON) * SMOKE_BACK * GLASS_BACK * SMOKE_BACK_COL;
        bgRad = bgRad * T + SMOKE_ALBEDO * (SMOKE_VOL_K * SMOKE_LV_K * Ls + amb * (1.0 - T));
#else
        vec3 lightS = SMOKE_EMBER * (texture(u_bloomB, uv).rgb + 0.35 * texture(u_bloomA, uv).rgb)
                    + float(GLASS_ON + BOX_ON) * SMOKE_BACK * GLASS_BACK * SMOKE_BACK_COL;
        bgRad = bgRad * T + SMOKE_ALBEDO * lightS * (1.0 - T);
#endif
    }
#endif
    rad += bgRad;
    vec3 col;
#if HUE_KEEP
    float m = max(rad.r, max(rad.g, rad.b));
    float tm = 1.0 - exp(-m * EXPOSURE);
#if FILMIC
    tm = clamp((tm * (2.51 * tm + 0.03)) / (tm * (2.43 * tm + 0.59) + 0.14), 0.0, 1.0);
#endif
    col = rad * (tm / max(m, 1e-6));
#else
    col = 1.0 - exp(-rad * EXPOSURE);
#if FILMIC
    col = clamp((col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14), 0.0, 1.0);
#endif
#endif
#if SRGB_OUT
    col = mix(col * 12.92, 1.055 * pow(max(col, vec3(1e-6)), vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), col));
#endif
    fragColor = vec4(col, 1.0);
}
