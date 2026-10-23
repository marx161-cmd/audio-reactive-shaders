// Lagoon-Worms: image pass.
//
// Two glowing worms in a compute fluid, point-mirrored through the centre,
// shedding goo that takes their colour at birth. Every section change one worm
// picks a new colour from a mood palette; hard transitions switch both. Stirred
// goo re-ignites, dying goo becomes afterlife embers.

uniform usampler2D u_accR;
uniform usampler2D u_accG;
uniform usampler2D u_accB;

#define W_SPLAT_K    256.0
#define W_GLOW_GAIN  0.026
#define W_BODY_LIFT 0.35
#define W_BLOOM_GAIN 0.15
#define W_DIM_LIFT   0.65
#define W_TONE_REF   100.0
vec3 wAccAt(ivec2 p) {
    ivec2 sz = textureSize(u_accR, 0);
    p = clamp(p, ivec2(0), sz - 1);
    return vec3(float(texelFetch(u_accR, p, 0).r),
                float(texelFetch(u_accG, p, 0).r),
                float(texelFetch(u_accB, p, 0).r)) / W_SPLAT_K;
}
vec3 wormGlow(ivec2 ip) {
    vec3 bl = vec3(0.0);
    float wsum = 0.0;
    for (int ring = 1; ring <= 3; ring++) {
        float rad = float(1 << ring);
        float w = 1.0 / float(ring);
        for (int k = 0; k < 8; k++) {
            float a = 0.7853982 * (float(k) + 0.5 * float(ring & 1));
            bl += w * wAccAt(ip + ivec2(round(rad * vec2(cos(a), sin(a)))));
        }
        wsum += 8.0 * w;
    }
    vec3 x = wAccAt(ip) + W_BLOOM_GAIN * bl / wsum * 6.0;
    float L = max(x.r, max(x.g, x.b));
    if (L <= 0.0) return vec3(0.0);
    float e = W_GLOW_GAIN * W_TONE_REF * pow(L / W_TONE_REF, W_DIM_LIFT);
    return 1.0 - exp(-e * (x / L));
}

uniform float u_time;
#define CAM_D      3.0
#define FOCAL      1.9
#define CAM_TY     0.0
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
#define BODY_GAIN 2.6
#define FOG_D0    2.2
#define FOG_K     0.55
vec3 bodyGlow(vec2 uv, float white) {
    vec3 col = vec3(0.0);
    mat3 R = camRotI();
    for (int w = 0; w < 2; w++) {
        vec3 cw = R * texelFetch(u_worm, ivec2(w, 0), 0).xyz;
        float z = cw.z + CAM_D;
        if (z < 0.3) continue;
        float s = FOCAL / z;
        vec2 d = (uv - cw.xy * s) / max(BODY_R * s, 1e-4);
        float fog = exp(-max(z - FOG_D0, 0.0) * FOG_K);
        vec3  hue = mix(texelFetch(u_worm, ivec2(6 + w, 0), 0).rgb, vec3(1.0), white);
        float bf  = exp(-dot(d, d) * 1.5);
        float bi  = BODY_GAIN * bf * fog;
        col += mix(hue, vec3(1.0), W_BODY_LIFT) * bi;
    }
    return col;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 p = ivec2(fragCoord);
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec4 sl = texelFetch(u_worm, ivec2(5, 0), 0);
    fragColor = vec4(waterSheen(fragCoord) + wormGlow(p) + bodyGlow(uv, sl.y), 1.0);
}
