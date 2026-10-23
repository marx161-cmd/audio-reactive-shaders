// Lagoon-Worms-Pad: image pass. Lagoon-Worms steered from a DroidPad pad
// (contrib/pads/mycontrolpad-2.json): tilt, depth, hues, pocket, frenzy.

uniform usampler2D u_glowA;
uniform usampler2D u_glowB;
uniform usampler2D u_glowR;
uniform vec2 u_pad_hue;

#define W_SPLAT_K    256.0
#define W_GLOW_GAIN  0.026
#define W_GOO_COL_A  vec3(0.12, 1.00, 0.70)
#define W_GOO_COL_B  vec3(0.10, 0.0, 0.55)
#define W_GOO_GAIN_A 1.0
#define W_GOO_GAIN_B 3.0
#define W_BODY_LIFT 0.35
#define W_SLIME_RED   vec3(1.00, 0.10, 0.10)
#define W_SLIME_MAG   0.10
#define W_SLIME_COL   mix(W_SLIME_RED, vec3(1.0, 0.0, 1.0), W_SLIME_MAG)
#define W_SLIME_GAIN  2.5
vec3 wHue(vec3 c, float t) {
    float a = 6.2831853 * t, cs = cos(a), sn = sin(a);
    float r = (0.299 + 0.701 * cs + 0.168 * sn) * c.r
            + (0.587 - 0.587 * cs + 0.330 * sn) * c.g
            + (0.114 - 0.114 * cs - 0.497 * sn) * c.b;
    float g = (0.299 - 0.299 * cs - 0.328 * sn) * c.r
            + (0.587 + 0.413 * cs + 0.035 * sn) * c.g
            + (0.114 - 0.114 * cs + 0.292 * sn) * c.b;
    float b = (0.299 - 0.300 * cs + 1.250 * sn) * c.r
            + (0.587 - 0.588 * cs - 1.050 * sn) * c.g
            + (0.114 + 0.886 * cs - 0.203 * sn) * c.b;
    return clamp(vec3(r, g, b), 0.0, 1.0);
}
#define W_BLOOM_GAIN 0.15
#define W_DIM_LIFT   0.65
#define W_TONE_REF   100.0
float wGlowAt(usampler2D tex, ivec2 p) {
    ivec2 sz = textureSize(tex, 0);
    p = clamp(p, ivec2(0), sz - 1);
    return float(texelFetch(tex, p, 0).r) / W_SPLAT_K;
}
vec3 wormGlow(ivec2 ip) {
    float bA = 0.0, bB = 0.0, bR = 0.0, wsum = 0.0;
    for (int ring = 1; ring <= 3; ring++) {
        float rad = float(1 << ring);
        float w = 1.0 / float(ring);
        for (int k = 0; k < 8; k++) {
            float a = 0.7853982 * (float(k) + 0.5 * float(ring & 1));
            ivec2 q = ip + ivec2(round(rad * vec2(cos(a), sin(a))));
            bA += w * wGlowAt(u_glowA, q);
            bB += w * wGlowAt(u_glowB, q);
            bR += w * wGlowAt(u_glowR, q);
        }
        wsum += 8.0 * w;
    }
    float xA = wGlowAt(u_glowA, ip) + W_BLOOM_GAIN * bA / wsum * 6.0;
    float xB = wGlowAt(u_glowB, ip) + W_BLOOM_GAIN * bB / wsum * 6.0;
    float xR = wGlowAt(u_glowR, ip) + W_BLOOM_GAIN * bR / wsum * 6.0;
    float eA = W_GOO_GAIN_A * W_GLOW_GAIN * W_TONE_REF * pow(max(xA, 0.0) / W_TONE_REF, W_DIM_LIFT);
    float eB = W_GOO_GAIN_B * W_GLOW_GAIN * W_TONE_REF * pow(max(xB, 0.0) / W_TONE_REF, W_DIM_LIFT);
    float eR = W_SLIME_GAIN * W_GLOW_GAIN * W_TONE_REF * pow(max(xR, 0.0) / W_TONE_REF, W_DIM_LIFT);
    vec3 colA = wHue(W_GOO_COL_A, u_pad_hue.x);
    vec3 colB = wHue(W_GOO_COL_B, u_pad_hue.y);
    return (1.0 - exp(-eA * colA))
         + (1.0 - exp(-eB * colB))
         + (1.0 - exp(-eR * W_SLIME_COL));
}

uniform float u_time;
#define CAM_D      3.0
#define FOCAL      1.9
#define CAM_TY     0.0
#define CAM_SPIN_S 60.0
#define CAM_TILT_A 0.65
#define CAM_TILT_S 97.0
#define DEEP_COL   vec3(0.000, 0.004, 0.010)
#define UP_COL     vec3(0.006, 0.026, 0.045)
mat3 camRotI() {
    float y = 0.6, p = 0.35;
    mat3 ry = mat3(cos(y), 0.0, -sin(y), 0.0, 1.0, 0.0, sin(y), 0.0, cos(y));
    mat3 rx = mat3(1.0, 0.0, 0.0, 0.0, cos(p), sin(p), 0.0, -sin(p), cos(p));
    return rx * ry;
}
vec3 waterSheen(vec2 fragCoord) {
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec3 d = normalize(transpose(camRotI()) * vec3(uv, FOCAL));
    return mix(DEEP_COL, UP_COL, pow(smoothstep(-0.4, 1.0, d.y), 1.5));
}

uniform sampler2D u_worm;
#define BODY_R    0.028
#define BODY_COL  vec3(0.55, 1.00, 0.85)
#define BODY_GAIN 2.6
#define FOG_D0    2.2
#define FOG_K     0.55
vec3 bodyGlow(vec2 uv, vec2 frenzy) {
    vec3 col = vec3(0.0);
    mat3 R = camRotI();
    for (int w = 0; w < 2; w++) {
        vec3 cw = R * texelFetch(u_worm, ivec2(w, 0), 0).xyz;
        float z = cw.z + CAM_D;
        if (z < 0.3) continue;
        float s = FOCAL / z;
        vec2 d = (uv - cw.xy * s) / max(BODY_R * s, 1e-4);
        float fog = exp(-max(z - FOG_D0, 0.0) * FOG_K);
        vec3  hue = wHue((w == 0) ? W_GOO_COL_A : W_GOO_COL_B,
                         (w == 0) ? u_pad_hue.x : u_pad_hue.y);
        float hgn = (w == 0) ? W_GOO_GAIN_A : W_GOO_GAIN_B;
        float bf  = exp(-dot(d, d) * 1.5);
        float bi  = BODY_GAIN * hgn * bf * fog;
        vec3  bh  = mix(hue, W_SLIME_COL, (w == 0) ? frenzy.x : frenzy.y);
        col += mix(bh, vec3(1.0), W_BODY_LIFT) * bi;
    }
    return col;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 p = ivec2(fragCoord);
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec4 sl = texelFetch(u_worm, ivec2(5, 0), 0);
    fragColor = vec4(waterSheen(fragCoord) + wormGlow(p) + bodyGlow(uv, vec2(sl.x, sl.z)), 1.0);
}
