#define SCALE        2.0
#define BEAT_GAIN    0.5
#define BEAT_SWELL   0.6
#define RING_BOLT_LIT 0.012
#define CRUST_ALBEDO 0.35
#define CRAWLERS     2
#define CRAWL_ARC    0.9
#define CRAWL_SEGS   6
#define CRAWL_SPEED  1.2
uniform sampler2D u_hops;
#define HOP_SLOTS    3
#define HOP_PERIOD   0.45
#define HOP_GROW     0.30
#define HOP_I        1.4
#define HOP_W        0.0013
#define HOP_ARCH     1.0
const float HOP_CHORD[3] = float[](0.06, 0.13, 0.26);
#define BALL_ARC_POW 14.0
#define BALL_ARC_I   5.0
#define BALL_RIM_I   1.2
#define BALL_AURA    0.75
#define BALL_AURA_R  1.6
#define LIMB         0.25
#define RELIEF       0.004
#define WALL_Z       -2.4
#define WALL_ALBEDO  0.55
#define WALL_GAIN    2.2
uniform sampler2D u_bolts;
#define BOLT_HALF    0.32
#define BOLT_MARGIN  0.03
#define BOLT_BRIGHT  4.5
#define BOLT_HALO    0.15

vec3 raySeg(vec3 ro, vec3 rd, vec3 a, vec3 b) {
    vec3 v = b - a, w = ro - a;
    float bb = dot(rd, v), c = dot(v, v), d = dot(rd, w), e = dot(v, w);
    float D = max(c - bb * bb, 1e-6);
    float tc = clamp((e - bb * d) / D, 0.0, 1.0);
    vec3 pc = a + v * tc;
    float sc = max(dot(pc - ro, rd), 0.0);
    return vec3(sc, length(ro + rd * sc - pc), tc);
}

#define CR_OCT    10
#define CR_GAIN   2.2
vec3 crustAt(vec3 po, int k, out float hr) {
    float beatDrift = BEAT_GAIN * (u_beat_count + BEAT_SWELL * sin(6.28318531 * u_beat_count) / 6.28318531);
    float tt = 0.05 * iTime + 0.4 * u_rms_accum + 1.2 * u_bass_accum + beatDrift;
    FFOut f = flowFluid(po * SCALE, tt, CR_OCT, vec3(0.0), RING_RGB[k], vec3(0.0), 0.0, 0.0, 0.0);
    hr = f.deck;
    return f.col * CR_GAIN;
}

#define BOLT_RING_COLOUR 1
vec3 boltColour(int k, float bv, float halo) {
    if (BOLT_RING_COLOUR == 0)
        return mix(ELEC_BLUE, HOT_WHITE, bv) * bv * BOLT_BRIGHT + ELEC_BLUE * BOLT_HALO * halo;
    vec3 hue = RING_RGB[k];
    return mix(hue, HOT_WHITE, 0.35 * bv) * bv * BOLT_BRIGHT + hue * BOLT_HALO * halo;
}

vec2 toScreen(vec3 X) { return X.xy / ((CAM_Z - X.z) * FOV); }

float boltPiece(vec2 uv, vec3 ro, vec3 a, vec3 b, int tile, float u0, float u1, float Lperp, float tLimit) {
    vec2 sa = toScreen(a), sb = toScreen(b);
    vec2 ab = sb - sa;
    float L = length(ab);
    if (L < 1e-5) return 0.0;
    vec2 d = ab / L, nn = vec2(-d.y, d.x);
    float px = 2.0 / iResolution.y;
    float best = 0.0, bestAlong = 0.0;
    for (int j = 0; j < 4; j++) {
        vec2 o = (vec2(j & 1, j >> 1) - 0.5) * px;
        float along = dot(uv + o - sa, d) / L;
        if (along < 0.0 || along > 1.0) continue;
        float sAlong = mix(u0, u1, along);
        float u = (sAlong + BOLT_MARGIN) / (1.0 + 2.0 * BOLT_MARGIN);
        float v = dot(uv + o - sa, nn) / Lperp / BOLT_HALF * 0.5 + 0.5;
        if (v < 0.0 || v > 1.0) continue;
        float t = textureLod(u_bolts, vec2(u, (float(tile) + v) / float(BOLT_N)), 0.0).r;
        if (t > best) { best = t; bestAlong = along; }
    }
    if (best <= 0.0) return 0.0;
    if (length(mix(a, b, bestAlong) - ro) > tLimit) return 0.0;
    return best;
}

float boltAt(vec2 uv, vec3 ro, vec3 a, vec3 b, int tile, float tHit) {
    vec2 sa = toScreen(a), sb = toScreen(b);
    vec2 ab = sb - sa;
    float L = length(ab);
    if (L < 1e-4) return 0.0;
    vec2 d = ab / L, nn = vec2(-d.y, d.x);
    float px = 2.0 / iResolution.y;
    float best = 0.0;
    for (int j = 0; j < 4; j++) {
        vec2 o = (vec2(j & 1, j >> 1) - 0.5) * px;
        float sAlong = dot(uv + o - sa, d) / L;
        float sAcross = dot(uv + o - sa, nn) / L;
        float u = (sAlong + BOLT_MARGIN) / (1.0 + 2.0 * BOLT_MARGIN);
        float v = sAcross / BOLT_HALF * 0.5 + 0.5;
        if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) continue;
        vec2 tuv = vec2(u, (float(tile) + v) / float(BOLT_N));
        best = max(best, textureLod(u_bolts, tuv, 0.0).r);
    }
    if (best <= 0.0) return 0.0;
    float sAlong = clamp(dot(uv - sa, d) / L, 0.0, 1.0);
    if (length(mix(a, b, sAlong) - ro) > tHit) return 0.0;
    return best;
}

float segDist(vec3 x, vec3 a, vec3 b) {
    vec3 ab = b - a;
    float t = clamp(dot(x - a, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
    return length(x - a - ab * t);
}

vec2 gOut[18];
#define PIPE_I       3.6
#define PIPE_DECAY   0.10
#define CON_LIGHT    0.006
vec3 mistAt(vec3 o, vec3 d) {
    vec3 P = d.z < -0.02 ? o + d * ((-1.0 - o.z) / d.z) : o + d * 3.0;
    vec2 uv = toScreen(P);
    vec2 st = (uv * iResolution.y + iResolution.xy) * 0.5 / iResolution.xy;
    return textureLod(iChannel2, st, 0.0).rgb + backdropAt(bgHitSt(o, d));
}
vec3 envAt(vec3 o, vec3 d) {
    vec3 base = vec3(0.018, 0.022, 0.035);
    if (LIGHTNING == 0) return mistAt(o, d);
    if (d.z > -0.02) return base * 0.5;
    vec3 P = o + d * ((MIRROR_Z - o.z) / d.z);
    if (LIGHTNING == 0) {
        float r2 = dot(P, P);
        float sw = 0.5 + 0.5 * sin(P.x * 3.1 + 0.7 * sin(P.y * 2.3 + iTime * 0.2))
                              * sin(P.y * 2.7 - 0.6 * sin(P.x * 1.9 - iTime * 0.15));
        return base * (0.6 + 0.8 * sw) + saphLight(P) * (-MIRROR_Z) * inversesqrt(r2) * (1.0 + 2.0 * sw) * ENV_LIT;
    }
    vec2 p = toScreen(P);
    vec3 halo = vec3(0.0);
    for (int k = 0; k < 3; k++) {
        float dmin = 1e9;
        for (int i = 0; i < 6; i++) {
            vec2 a = gOut[k * 6 + i], b = gOut[k * 6 + (i + 1) % 6], ab = b - a;
            vec2 at = a + ab * clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
            dmin = min(dmin, length(p - at));
        }
        halo += RING_RGB[k] * (0.5 + 0.5 * u_mid)
              * (0.30 * exp(-pow(dmin / 0.035, 2.0)) + 0.06 / (1.0 + pow(dmin / 0.25, 2.0)));
    }
    return base + halo;
}
vec3 gBA[10], gBB[10]; float gBF[10]; int gBK[10];
vec3 boltLightAt(vec3 p, out vec3 toBright) {
    vec3 L = vec3(0.0); float bestI = 0.0; toBright = vec3(0.0, 0.0, 1.0);
    for (int j = 0; j < 10; j++) {
        if (gBF[j] <= 0.0) continue;
        vec3 ab = gBB[j] - gBA[j];
        float hh = clamp(dot(p - gBA[j], ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
        vec3 toL = gBA[j] + ab * hh - p;
        float d2 = dot(toL, toL);
        float I = gBF[j] / (d2 + 0.004);
        L += RING_RGB[gBK[j]] * I;
        if (I > bestI) { bestI = I; toBright = toL * inversesqrt(max(d2, 1e-8)); }
    }
    return L;
}
vec3 boltGlow3(vec3 o, vec3 d, float tEnd) {
    vec3 g = vec3(0.0);
    for (int j = 0; j < 10; j++) {
        if (gBF[j] <= 0.0) continue;
        vec3 sr = raySeg(o, d, gBA[j], gBB[j]);
        if (sr.x > tEnd) continue;
        g += RING_RGB[gBK[j]] * gBF[j] * BOLT_HALO * exp(-sr.y * sr.y / 0.0004);
    }
    return g;
}

vec3 traceGem(vec3 ro, vec3 rd, int ch, int pol, out bool usedQuartz) {
    vec3 acc = vec3(0.0);
    usedQuartz = false;
    vec3 o = ro, d = rd; vec3 w = vec3(1.0); int med = -1;
    float nSaph = GEM_CA[1] + GEM_CB[1] * DISP_EXAG / (LAMBDA_UM[ch] * LAMBDA_UM[ch]);
    for (int n = 0; n < MAX_NODES; n++) {
        if (max(max(w.r, w.g), w.b) < W_MIN) break;
        if (med < 0) {
            int bar, face;
            float t = gemNearest(o, d, 1e9, bar, face);
            float bS = dot(o, d), hS = bS * bS - (dot(o, o) - 0.0144);
            if (hS > 0.0 && -bS + sqrt(hS) > GEPS) {
                float tN, tF; int jN, jF;
                if (saphHit(o, d, tN, jN, tF, jF) && tN > GEPS && tN < t) { t = tN; bar = 18; face = jN; }
            }
            {
                int ci; vec3 nC;
                float tc = conNearest(o, d, t, ci, nC);
                if (ci >= 0) {
                    float mu = abs(dot(nC, d));
                    acc += w * conCol(ci) * CON_EMIT * conPulse() * (0.55 + 0.45 * mu);
                    break;
                }
            }
            if (bar < 0) {
                acc += w * (envAt(o, d) + (LIGHTNING == 1 ? boltGlow3(o, d, 1e9) : vec3(0.0)));
                break;
            }
            if (LIGHTNING == 1) acc += w * boltGlow3(o, d, t);
            vec3 p = o + d * t;
            vec3 nW; float n2;
            if (bar == 18) { nW = gS * normalize(vec3(SAPH[face].xy / SAPH_SW, SAPH[face].z)); n2 = nSaph; }
            else {
                int k = bar / 6;
                if (k == 0) usedQuartz = true;
                nW = gM[k] * BP[bar * 8 + face].xyz;
                n2 = gemIOR(k, ch, pol, transpose(gM[k]) * d, bar - k * 6);
            }
            float cosi = clamp(-dot(d, nW), 0.0, 1.0);
            float R = fresnelD(cosi, 1.0, n2);
            vec3 rf = reflect(d, nW);
            acc += w * R * envAt(p, rf);
            if (bar != 18) {
                if (LIGHTNING == 0) {
                    vec3 Ls = saphLight(p);
                    acc += w * R * Ls * SAPH_GLINT * pow(max(dot(rf, normalize(-p)), 0.0), 60.0);
                    for (int c = 0; c < N_CON; c++) {
                        vec3 cm = 0.5 * (gCA[c] + gCB[c]) - p;
                        float r2 = dot(cm, cm);
                        acc += w * R * conCol(c) * conPulse() * CON_LIGHT / (r2 + 0.002) * SAPH_GLINT
                             * pow(max(dot(rf, cm * inversesqrt(r2)), 0.0), 60.0);
                    }
                    if (bar / 6 == RUBY_RING) acc.r += w.r * (1.0 - R) * RUBY_FLUOR * (Ls.b + Ls.g) * 0.5;
                } else {
                    vec3 toB; vec3 bl = boltLightAt(p, toB);
                    acc += w * R * bl * GEM_GLINT * pow(max(dot(rf, toB), 0.0), 40.0) * 40.0;
                    if (bar / 6 == RUBY_RING) acc.r += w.r * (1.0 - R) * RUBY_FLUOR * 0.5 * (bl.g + bl.b) * 0.02;
                }
            }
            d = refract(d, nW, 1.0 / n2); o = p - nW * GEPS; w *= 1.0 - R; med = bar;
        } else if (med == 18) {
            float tN, tF; int jN, jF;
            saphHit(o, d, tN, jN, tF, jF);
            if (jF < 0) break;
            float L = max(tF, 0.0);
            acc += w * SAPH_COL * saphPulse() * SAPH_EMIT * L;
            w *= exp(-SAPH_SIGMA * L);
            vec3 p = o + d * L;
            vec3 nW = gS * normalize(vec3(SAPH[jF].xy / SAPH_SW, SAPH[jF].z));
            float cosi = clamp(dot(d, nW), 0.0, 1.0);
            float R = fresnelD(cosi, nSaph, 1.0);
            if (R >= 1.0 || R > 0.5) { d = reflect(d, nW); o = p - nW * GEPS; if (R < 1.0) w *= R; continue; }
            d = refract(d, -nW, nSaph); o = p + nW * GEPS; w *= 1.0 - R; med = -1;
        } else {
            int k = med / 6, i = med - k * 6;
            mat3 iG = transpose(gM[k]);
            vec3 ol = iG * o, dl = iG * d;
            float tN, tF; int jN, jF;
            barHit(ol, dl, med * 8, tN, jN, tF, jF);
            if (jF < 0) break;
            float L = max(tF, 0.0);
            {
                int ci; vec3 nC;
                float tc = conNearest(o, d, L, ci, nC);
                if (ci >= 0) {
                    acc += w * conCol(ci) * CON_EMIT * conPulse() * (0.55 + 0.45 * abs(dot(nC, d)));
                    break;
                }
            }
            if (LIGHTNING == 0) {
                vec3 mid = o + d * (0.5 * L);
                vec3 g = vec3(0.0);
                for (int c = 0; c < N_CON; c++) g += conCol(c) * exp(-segDistC(mid, gCA[c], gCB[c]) / PIPE_DECAY);
                acc += w * g * conPulse() * PIPE_I * L;
                if (k == RUBY_RING) {
                    vec3 Ls = saphLight(mid);
                    float blue = 0.5 * (Ls.b + Ls.g);
                    for (int c = 0; c < N_CON; c++) {
                        float dd = segDistC(mid, gCA[c], gCB[c]);
                        vec3 cc = conCol(c);
                        blue += 0.5 * (cc.b + cc.g) * conPulse() * CON_LIGHT / (dd * dd + 0.002);
                    }
                    acc.r += w.r * RUBY_FLUOR_VOL * blue * L;
                }
            }
            w *= exp(-GEM_SIGMA[k] * L);
            vec3 p = o + d * L;
            if (jF < 2) {
                med = k * 6 + (jF == 0 ? (i + 5) % 6 : (i + 1) % 6);
                o = p + d * GEPS;
                continue;
            }
            vec3 nW = gM[k] * BP[med * 8 + jF].xyz;
            float n1 = gemIOR(k, ch, pol, dl, i);
            float cosi = clamp(dot(d, nW), 0.0, 1.0);
            float R = fresnelD(cosi, n1, 1.0);
            if (R >= 1.0) { d = reflect(d, nW); o = p - nW * GEPS; continue; }
            vec3 dOut = refract(d, -nW, n1);
            if (R > 0.5) { d = reflect(d, nW); o = p - nW * GEPS; w *= R; continue; }
            d = dOut; o = p + nW * GEPS; w *= 1.0 - R; med = -1;
        }
    }
    return acc;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0.0, 0.0, CAM_Z);
    vec3 rd = normalize(vec3(uv * FOV, -1.0));

    setupGimbal();

    float fk = flicker();
    vec3 bA[10], bB[10]; float bFl[10]; int bK[10], bTile[10];
    buildBolts(bA, bB, bFl, bK, bTile);
    if (LIGHTNING == 0) for (int j = 0; j < 10; j++) bFl[j] = 0.0;
    for (int j = 0; j < 10; j++) { gBA[j] = bA[j]; gBB[j] = bB[j]; gBF[j] = bFl[j]; gBK[j] = bK[j]; }

    setupGems();
    for (int k = 0; k < 3; k++) for (int i = 0; i < 6; i++)
        gOut[k * 6 + i] = toScreen(gM[k] * hexPt(k, RING_VOFF[k] + A60 * float(i), 0.0));
    gS = saphRot();
    buildConnectors();
    float tHit = 1e9; int hid = -1;
    {
        int bar, face;
        float t = gemNearest(ro, rd, 1e9, bar, face);
        if (bar >= 0) { tHit = t; hid = bar / 6; }
        float tN, tF; int jN, jF;
        if (saphHit(ro, rd, tN, jN, tF, jF) && tN > 0.0 && tN < tHit) { tHit = tN; hid = 3; }
        int ci; vec3 nC;
        float tc = conNearest(ro, rd, tHit, ci, nC);
        if (ci >= 0) { tHit = tc; hid = 4; }
    }
    vec3 col = vec3(0.0);
    if (hid >= 0) {
        bool uq = false, u1;
        for (int ch = 0; ch < 3; ch++) { col[ch] = traceGem(ro, rd, ch, 0, u1)[ch]; uq = uq || u1; }
        if (uq && GEM_BIREF == 1) { bool dummy; col = 0.5 * (col + traceGem(ro, rd, 1, 1, dummy)); }
        if (LIGHTNING == 1 && hid < 3) {
            vec3 p = ro + rd * tHit;
            vec3 pl = transpose(gM[hid]) * p;
            float hr;
            col += max(crustAt(pl, hid, hr), 0.0) * GEM_GLOW * (0.5 + 0.5 * u_mid);
        }
    }
    float b = dot(ro, rd);
    float tc = -dot(ro, rd);
    float bdist = max(length(ro + rd * tc), 1e-3);

    vec3 glow = vec3(0.0);
    bool ballHit = false;
    float aura = 0.0;
    float oc = dot(ro, ro) - ORB_R * ORB_R, h = b * b - oc;
    if (LIGHTNING == 1 && h > 0.0) {
        float tNear = -b - sqrt(h);
        if (tNear < tHit) {
            vec3 pN = ro + rd * tNear;
            vec3 ns = pN / ORB_R;
            float arcs = pow(lightningField(ns * 2.8, iTime), BALL_ARC_POW);
            float plasmaNoise = lfbm(ns * 5.0 * 0.65 - vec3(0.0, iTime * 4.0, 0.0));
            float rim = pow(1.0 - max(dot(-rd, ns), 0.0), 6.0);
            vec3 pc = mix(DEEP_VIOLET, ELEC_BLUE, plasmaNoise);
            vec3 em = mix(pc, HOT_WHITE, clamp(arcs * 1.5, 0.0, 1.0)) * arcs * BALL_ARC_I * (1.0 + 2.0 * u_onset)
                    + pc * rim * BALL_RIM_I
                    + DEEP_VIOLET * 0.02;
            glow += em * fk;
            col = vec3(0.0);
            ballHit = true;
        }
    }
    if (LIGHTNING == 1 && !ballHit && (tc < tHit || hid < 0))
        aura = BALL_AURA * exp(-pow(max(bdist - ORB_R, 0.0) / (ORB_R * BALL_AURA_R), 2.0));

    for (int i = 0; i < 10; i++) {
        if (bFl[i] <= 0.0) continue;
        float bv = boltAt(uv, ro, bA[i], bB[i], bTile[i], tHit);
        vec3 sr = raySeg(ro, rd, bA[i], bB[i]);
        float halo = sr.x < tHit ? exp(-sr.y * sr.y / 0.0004) : 0.0;
        glow += boltColour(bK[i], bv, halo) * bFl[i];
    }

    {
        float bps = max(S(3).y, 1e-3);
        for (int k = 0; k < N_RINGS; k++) {
            float rate = RING_NOTES[k];
            float note = floor(u_beat_count * rate);
            float since = fract(u_beat_count * rate) / (rate * bps);
            float fl = flashEnvelope(since, BOLT_TAU[k]) * (0.6 + 0.4 * S(4)[k]);
            if (fl < 0.01 || LIGHTNING == 0) continue;
            float Re = RING_R[k] + RING_TUBE[k];
            for (int c = 0; c < CRAWLERS; c++) {
                float id = 90.0 + float(k * CRAWLERS + c);
                float a0 = 6.28318531 * lhash(vec3(note, id, 4.4)) + RING_DIR[k] * CRAWL_SPEED * since;
                int tile = int(floor(lhash(vec3(note, id, 9.2)) * float(BOLT_N))) % BOLT_N;
                vec3 pts[CRAWL_SEGS + 1];
                float Ls = 0.0;
                for (int i = 0; i <= CRAWL_SEGS; i++) {
                    float a = a0 + CRAWL_ARC * float(i) / float(CRAWL_SEGS);
                    pts[i] = gM[k] * hexPt(k, a, RING_TUBE[k]);
                    if (i > 0) Ls += length(toScreen(pts[i]) - toScreen(pts[i - 1]));
                }
                float bv = 0.0;
                for (int i = 0; i < CRAWL_SEGS; i++)
                    bv = max(bv, boltPiece(uv, ro, pts[i], pts[i + 1], tile,
                                           float(i) / float(CRAWL_SEGS), float(i + 1) / float(CRAWL_SEGS),
                                           max(Ls, 1e-4), tHit + 0.03));
                glow += boltColour(k, bv, 0.0) * fl;
            }
        }
    }

    float feed = 0.0;
    for (int i = 4; i < 10; i++) feed = max(feed, bFl[i]);
    feed = clamp(feed, 0.0, 1.0);

    for (int k = 0; k < N_RINGS; k++) {
        vec3 hue = RING_RGB[k];
        float R = RING_R[k], rt = RING_TUBE[k];
        for (int sl = 0; sl < HOP_SLOTS; sl++) {
            float id = float(k * HOP_SLOTS + sl);
            float ph = iTime / HOP_PERIOD + id * 0.37;
            float cyc = floor(ph), f = fract(ph);
            int hop = int(floor(lhash(vec3(cyc, id, 1.1)) * 9.0)) % 9;
            float chord = HOP_CHORD[hop / 3] * (0.8 + 0.4 * lhash(vec3(cyc, id, 2.2)));
            float phi = 6.28318531 * lhash(vec3(cyc, id, 3.3));
            float th = 6.28318531 * lhash(vec3(cyc, id, 4.4));
            float th2 = th + (lhash(vec3(cyc, id, 5.5)) - 0.5) * 0.6;
            float phi2 = phi + RING_DIR[k] * chord / R;
            vec3 A = gM[k] * (hexPt(k, phi, 0.0) + rt * (cos(th) * hexEdgeN(k, phi) + vec3(0.0, 0.0, sin(th))));
            vec3 B = gM[k] * (hexPt(k, phi2, 0.0) + rt * (cos(th2) * hexEdgeN(k, phi2) + vec3(0.0, 0.0, sin(th2))));
            vec2 sA = toScreen(A), sB = toScreen(B);
            float pad = 0.8 * chord / ((CAM_Z - 0.5 * (A.z + B.z)) * FOV);
            if (uv.x < min(sA.x, sB.x) - pad || uv.x > max(sA.x, sB.x) + pad ||
                uv.y < min(sA.y, sB.y) - pad || uv.y > max(sA.y, sB.y) + pad) continue;
            float phm = 0.5 * (phi + phi2);
            vec3 mid = 0.5 * (A + B);
            vec3 core = gM[k] * hexPt(k, phm, 0.0);
            vec3 ex = normalize(B - A);
            vec3 up = normalize(mid - core); up = normalize(up - ex * dot(up, ex));
            vec3 ey = cross(up, ex);
            float Lw = length(B - A);
            float grow = clamp(f / HOP_GROW, 0.0, 1.0);
            float since = (f - HOP_GROW) * HOP_PERIOD;
            float hl = (f < HOP_GROW ? 0.35 : flashEnvelope(since, 0.08)) * HOP_I * (0.4 + 0.6 * feed);
            if (hl < 0.01 || LIGHTNING == 0) continue;
            int n = int(texelFetch(u_hops, ivec2(0, hop), 0).x);
            float best = 1e9, bestDepth = 0.0;
            for (int sg = 0; sg < 128; sg++) {
                if (sg >= n) break;
                vec4 q0 = texelFetch(u_hops, ivec2(1 + 2 * sg, hop), 0);
                if (q0.w > grow) continue;
                vec4 q1 = texelFetch(u_hops, ivec2(2 + 2 * sg, hop), 0);
                vec3 P0 = A + Lw * (q0.x * ex + q0.y * ey + q0.z * HOP_ARCH * up);
                vec3 P1 = A + Lw * (q1.x * ex + q1.y * ey + q1.z * HOP_ARCH * up);
                vec3 r = raySeg(ro, rd, P0, P1);
                if (r.x > tHit + 0.01) continue;
                float w = r.y / (1.0 - 0.2 * q1.w);
                if (w < best) { best = w; bestDepth = q1.w; }
            }
            float core2 = exp(-best * best / (HOP_W * HOP_W));
            float g = core2 + 0.25 * exp(-best * best / (9.0 * HOP_W * HOP_W));
            glow += mix(hue, HOT_WHITE, 0.35 * core2) * g * hl;
        }
    }

    col += glow / (1.0 + glow);
    fragColor = vec4(keepHue(max(col, 0.0)), hid >= 0 || ballHit ? 1.0 : aura);
}
