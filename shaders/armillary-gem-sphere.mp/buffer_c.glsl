#define MIST_STEPS   40
#define MIST_R       2.4
#define MIST_D       0.55
#define MIST_EXT     0.35
#define MIST_G       0.55
#define MIST_I       0.026
#define MIST_CON     0.0021
#define MIST_DRIFT   vec3(0.0, -0.05, 0.03)

float hg(float c, float g) { float g2 = g * g; return (1.0 - g2) / (12.566371 * pow(1.0 + g2 - 2.0 * g * c, 1.5)); }

vec3 shadowT(vec3 x) {
    vec3 T = vec3(1.0);
    float dist = length(x);
    vec3 d = -x / dist;
    float tmax = dist - SAPH_R * 0.7;
    if (tmax <= 0.0) return T;
    for (int k = 0; k < 3; k++) {
        mat3 iG = transpose(gM[k]);
        vec3 ol = iG * x, dl = iG * d;
        float Rb = RING_R[k] + RING_TUBE[k] * BAR_EXT_R / C30 + 0.005, zb = RING_TUBE[k] * BAR_EXT_Z + 0.002;
        float b = dot(ol, dl), c = dot(ol, ol) - Rb * Rb, h = b * b - c;
        if (h < 0.0) continue;
        h = sqrt(h);
        float t0 = -b - h, t1 = -b + h;
        if (abs(dl.z) > 1e-7) {
            float za = (-zb - ol.z) / dl.z, zc = (zb - ol.z) / dl.z;
            t0 = max(t0, min(za, zc)); t1 = min(t1, max(za, zc));
        } else if (abs(ol.z) > zb) continue;
        t0 = max(t0, 0.0); t1 = min(t1, tmax);
        if (t1 <= t0) continue;
        float n = GEM_CA[k] + GEM_CB[k] * DISP_EXAG / 0.3025;
        for (int i = 0; i < 6; i++) {
            float tN, tF; int jN, jF;
            int base = (k * 6 + i) * 8;
            if (!barHit(ol, dl, base, tN, jN, tF, jF)) continue;
            float a0 = max(tN, 0.0), a1 = min(tF, tmax);
            if (a1 <= a0) continue;
            T *= exp(-GEM_SIGMA[k] * (a1 - a0));
            if (tN >= 0.0 && jN >= 2) T *= 1.0 - fresnelD(clamp(-dot(dl, BP[base + jN].xyz), 0.0, 1.0), 1.0, n);
            if (tF <= tmax && jF >= 2) T *= 1.0 - fresnelD(clamp(dot(dl, BP[base + jF].xyz), 0.0, 1.0), n, 1.0);
        }
    }
    return T;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    if (LIGHTNING == 1) { fragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0.0, 0.0, CAM_Z);
    vec3 rd = normalize(vec3(uv * FOV, -1.0));
    setupGimbal();
    buildConnectors();
    float b = dot(ro, rd), c = dot(ro, ro) - MIST_R * MIST_R, h = b * b - c;
    if (h <= 0.0) { fragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
    h = sqrt(h);
    float t0 = max(-b - h, 0.0), t1 = -b + h;
    float dt = (t1 - t0) / float(MIST_STEPS);
    float jit = fract(sin(dot(fragCoord, vec2(12.9898, 78.233))) * 43758.5453);
    vec3 L = vec3(0.0); float Tv = 1.0;
    vec3 sc = SAPH_COL * saphPulse();
    float cp = conPulse();
    vec3 drift = MIST_DRIFT * iTime;
    for (int i = 0; i < MIST_STEPS; i++) {
        float t = t0 + (float(i) + jit) * dt;
        vec3 x = ro + rd * t;
        float r = length(x);
        float rho = MIST_D * (0.45 + 0.55 * noise3(x * 1.8 + drift) * (0.6 + 0.4 * noise3(x * 4.3 - drift * 1.7)))
                  * smoothstep(MIST_R, MIST_R * 0.55, r);
        if (rho <= 1e-4) continue;
        vec3 Ls = sc * MIST_I / (r * r + 0.01) * shadowT(x) * hg(dot(x / max(r, 1e-4), -rd), MIST_G) * 12.566371;
        vec3 Lc = vec3(0.0);
        for (int k = 0; k < N_CON; k++) { float dd = segDistC(x, gCA[k], gCB[k]); Lc += conCol(k) * cp * MIST_CON / (dd * dd + 0.002); }
        L += Tv * rho * dt * (Ls + Lc);
        Tv *= exp(-rho * MIST_EXT * dt);
    }
    fragColor = vec4(L, Tv);
}
