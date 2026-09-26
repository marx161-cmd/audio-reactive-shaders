// Tonnetz-Hex-Lattice: image pass, a heightfield renderer.
//
// The scene is a single-valued heightmap (floor plus hex pins whose heights
// come from Buffer A), so each ray marches its ground projection near to far
// and refines the hit analytically, instead of sphere-tracing an SDF.
// Lighting, heightfield shadows and AO are per surface.

#define S 0.5
#define MAX_STEPS 96
#define COARSE_STEP_FRAC 0.6
#define COARSE_STEP_GROWTH 0.05
#define MAX_DIST 150.0
#define HEIGHT_SCALE 1.5
#define BASE_HEIGHT 0.05
#define GLOW_KNEE 1.0

#define STUD_HEIGHT 0.045
#define BEVEL_FLAT 0.85

float hexEdgeDist(vec2 p, float inradius) {
    const float c30 = 0.8660254, s30 = 0.5;
    p = vec2(c30 * p.x - s30 * p.y, s30 * p.x + c30 * p.y);
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735027);
    vec2 q = abs(p);
    q -= 2.0 * min(dot(k.xy, q), 0.0) * k.xy;
    vec2 d = vec2(clamp(q.x, -k.z * inradius, k.z * inradius), inradius);
    return length(q - d) * sign(q.y - inradius);
}

#define RING_GROWTH 0.06
float ringPitchAt(float row) { return RING_PITCH * (1.0 + RING_GROWTH * row); }

float sampleField(vec2 tileCenter, out float h, out float raw) {
    float r = length(tileCenter);
    float theta = atan(tileCenter.y, tileCenter.x);
    if (theta < 0.0) theta += 2.0 * PI;

    if (r < HUB_RADIUS) {
        float bass = 0.0;
        for (int c = 0; c < COLS; c++) {
            bass = max(bass, texelFetch(iChannel0, ivec2(c, ROWS - 1), 0).r);
        }
        h = bass;
        raw = bass;
        return 0.0;
    }

    #define HEX_RADIUS_BASE 7.0
    #define FALLOFF_POW 3.5
    float colStep = 2.0 * PI / float(COLS);
    float colF = theta / colStep - 0.5;
    float colA = floor(colF);

    float row = -1.0;
    float acc = HUB_RADIUS;
    for (int i = 0; i < ROWS; i++) {
        float pitch = ringPitchAt(float(i));
        if (r < acc + pitch) { row = float(i); break; }
        acc += pitch;
    }
    if (row < 0.0) { h = 0.0; raw = 0.0; return 0.0; }

    vec2 myId, myLocal;
    hexInfo(tileCenter, S, myId, myLocal);

    float best = 0.0;
    for (int dr = -1; dr <= 1; dr++) {
        int candRow = int(row) + dr;
        if (candRow < 0 || candRow >= ROWS) continue;

        float rowAcc = HUB_RADIUS;
        for (int i = 0; i < candRow; i++) rowAcc += ringPitchAt(float(i));
        float rC = rowAcc + ringPitchAt(float(candRow)) * 0.5;
        float rowFrac = float(candRow) / float(ROWS - 1);
        #define HEX_RADIUS_RATIO_POW 0.6
        float rC0 = HUB_RADIUS + ringPitchAt(0.0) * 0.5;
        float hexRadius = HEX_RADIUS_BASE * pow(rC / rC0, HEX_RADIUS_RATIO_POW);

        float frac = colF - colA;
        float beamWidth = mix(0.6, 1.4, rowFrac);
        float contrib[2];
        float sidePeak[2];
        float sideDragPeak[2];
        for (int side = 0; side < 2; side++) {
            float colIdxF = colA + float(side) + 0.5;
            int colIdx = int(mod(colA + float(side), float(COLS)));
            vec4 cell = texelFetch(iChannel0, ivec2(colIdx, ROWS - 1 - candRow), 0);
            float peak = cell.r;
            float dragPeak = cell.b;
            sidePeak[side] = peak;
            sideDragPeak[side] = dragPeak;
            if (peak <= 0.0 && dragPeak <= 0.0) { contrib[side] = 0.0; continue; }

            float thetaC = colIdxF * colStep;
            vec2 center = rC * vec2(cos(thetaC), sin(thetaC));
            vec2 pinId, pinLocal;
            hexInfo(center, S, pinId, pinLocal);

            vec2 d = myId - pinId;
            float hexDist = (abs(d.x) + abs(d.x + d.y) + abs(d.y)) * 0.5;
            float dragT = clamp(hexDist / hexRadius, 0.0, 1.0);
            float effectivePeak = mix(peak, dragPeak, dragT);
            float falloff = pow(clamp(1.0 - hexDist / hexRadius, 0.0, 1.0), FALLOFF_POW);

            #define CENTER_BOOST 1.1
            #define SURROUND_DAMPER_MIN 0.92
            #define SURROUND_DAMPER_MAX 0.98
            if (hexDist < 0.5) {
                falloff *= CENTER_BOOST;
            } else {
                float rnd = hash21(myId + 41.0);
                falloff *= mix(SURROUND_DAMPER_MIN, SURROUND_DAMPER_MAX, rnd);
            }

            float ownFrac = (side == 0) ? frac : (1.0 - frac);
            float angularWeight = 1.0 - smoothstep(0.0, beamWidth, ownFrac);
            float centerContrib = effectivePeak * falloff * angularWeight;

            float gapContrib = 0.0;
            {
                for (int g = -1; g <= 1; g += 2) {
                    float gThetaC = thetaC + float(g) * colStep * 0.5;
                    vec2 gCenter = rC * vec2(cos(gThetaC), sin(gThetaC));
                    vec2 gPinId, gPinLocal;
                    hexInfo(gCenter, S, gPinId, gPinLocal);
                    vec2 gd = myId - gPinId;
                    float gHexDist = (abs(gd.x) + abs(gd.x + gd.y) + abs(gd.y)) * 0.5;
                    float gDragT = clamp(gHexDist / hexRadius, 0.0, 1.0);
                    float gEffectivePeak = mix(peak, dragPeak, gDragT);
                    float gFalloff = pow(clamp(1.0 - gHexDist / hexRadius, 0.0, 1.0), FALLOFF_POW);
                    gapContrib = max(gapContrib, gEffectivePeak * gFalloff);
                }
            }

            contrib[side] = max(centerContrib, gapContrib);
        }

        float bridgePeak = min(sidePeak[0], sidePeak[1]);
        float bridgeDragPeak = min(sideDragPeak[0], sideDragPeak[1]);
        float bridgeContrib = 0.0;
        if (bridgePeak > 0.0 || bridgeDragPeak > 0.0) {
            float boundaryThetaC = (colA + 1.0) * colStep;
            vec2 bCenter = rC * vec2(cos(boundaryThetaC), sin(boundaryThetaC));
            vec2 bPinId, bPinLocal;
            hexInfo(bCenter, S, bPinId, bPinLocal);
            vec2 bd = myId - bPinId;
            float bHexDist = (abs(bd.x) + abs(bd.x + bd.y) + abs(bd.y)) * 0.5;
            float bDragT = clamp(bHexDist / hexRadius, 0.0, 1.0);
            float bEffectivePeak = mix(bridgePeak, bridgeDragPeak, bDragT);
            float bFalloff = pow(clamp(1.0 - bHexDist / hexRadius, 0.0, 1.0), FALLOFF_POW);
            bridgeContrib = bEffectivePeak * bFalloff;
        }

        float k = 0.2;
        float hBlend = clamp(0.5 + 0.5 * (contrib[0] - contrib[1]) / k, 0.0, 1.0);
        float rowBest = mix(contrib[1], contrib[0], hBlend) + k * hBlend * (1.0 - hBlend);
        best = max(best, max(rowBest, bridgeContrib));
    }

    float maxR = HUB_RADIUS;
    for (int i = 0; i < ROWS; i++) maxR += ringPitchAt(float(i));
    float outerRingWidth = ringPitchAt(float(ROWS - 1));
    float fadeT = smoothstep(maxR - outerRingWidth, maxR, r);
    #define RIM_FADE_POW 2.5
    float rimFade = 1.0 - pow(fadeT, RIM_FADE_POW);
    best *= rimFade;

    if (row > float(ROWS - 1) - 0.5) {
        float ring8Width = ringPitchAt(float(ROWS - 1));
        float ring8InnerEdge = maxR - ring8Width;
        float innerTaperEnd = ring8InnerEdge + ring8Width * 0.15;
        best *= smoothstep(ring8InnerEdge, innerTaperEnd, r);
    }

    h = best;
    raw = best;
    return 0.0;
}

#define DIP_OUTER_RINGS 3
float outerRingInnerRadius() {
    float maxR = HUB_RADIUS;
    for (int i = 0; i < ROWS; i++) maxR += ringPitchAt(float(i));
    float innerR = maxR;
    for (int i = 0; i < DIP_OUTER_RINGS; i++) innerR -= ringPitchAt(float(ROWS - 1 - i));
    return innerR;
}

#define TREBLE_INNER_RINGS 3
float trebleOuterRadius() {
    float r = HUB_RADIUS;
    for (int i = 0; i < TREBLE_INNER_RINGS; i++) r += ringPitchAt(float(i));
    return r;
}

float columnAt(vec2 xz, out vec2 cellId) {
    vec2 local;
    hexInfo(xz, S, cellId, local);
    vec2 tileCenter = xz - local;
    float h, raw;
    sampleField(tileCenter, h, raw);
    float dir = (length(tileCenter) < HUB_RADIUS) ? -1.0 : 1.0;
    return BASE_HEIGHT + STUD_HEIGHT + dir * max(h, 0.0) * HEIGHT_SCALE;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    vec3 ro = vec3(0.0, 16.0, -46.0);
    vec3 forward = normalize(vec3(0.0, -0.33, 0.94));
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = cross(forward, right);
    vec3 rd = normalize(forward + uv.x * right + uv.y * up);

    float t = MAX_DIST;
    vec3 col = vec3(0.05, 0.10, 0.13);
    bool hit = false;
    bool isSky = rd.y >= 0.0;

    if (isSky) {
        float sky = clamp(rd.y, 0.0, 1.0);
        col = mix(vec3(0.05, 0.10, 0.13), vec3(0.01, 0.02, 0.035), sky);
    } else {

    float minH = BASE_HEIGHT + STUD_HEIGHT - HEIGHT_CEIL * HEIGHT_SCALE;
    float maxH = BASE_HEIGHT + STUD_HEIGHT + HEIGHT_CEIL * HEIGHT_SCALE;
    float tStart = max((ro.y - maxH) / (-rd.y), 0.0);
    float tEnd = min((ro.y - minH) / (-rd.y), MAX_DIST);

    bool hitSide = false;
    vec2 hitCellId = vec2(0.0);
    vec2 hitWallDir = vec2(0.0);
    float hitWallLo = 0.0, hitWallHi = 0.0;

    float tc = tStart;
    vec2 curCellId;
    float curColH = columnAt((ro + rd * tc).xz, curCellId);

    for (int i = 0; i < MAX_STEPS; i++) {
        if (tc >= tEnd) break;
        float step = S * COARSE_STEP_FRAC * (1.0 + tc * COARSE_STEP_GROWTH);
        float tNext = min(tc + step, tEnd);
        vec3 pNext = ro + rd * tNext;
        vec2 nextCellId;
        float nextColH = columnAt(pNext.xz, nextCellId);

        if (nextCellId == curCellId) {
            if (pNext.y <= curColH) {
                hit = true; hitSide = false; hitCellId = curCellId;
                t = clamp((ro.y - curColH) / (-rd.y), tc, tNext);
                break;
            }
            tc = tNext; curColH = nextColH;
            continue;
        }

        float lo = tc, hi = tNext;
        for (int j = 0; j < 12; j++) {
            float mid = (lo + hi) * 0.5;
            vec2 midId, midLocal;
            hexInfo((ro + rd * mid).xz, S, midId, midLocal);
            if (midId == curCellId) lo = mid; else hi = mid;
        }
        float tBound = hi;
        float yBound = (ro + rd * tBound).y;

        if (yBound <= curColH) {
            hit = true; hitSide = false; hitCellId = curCellId;
            t = clamp((ro.y - curColH) / (-rd.y), tc, tBound);
            break;
        }
        if (yBound <= max(curColH, nextColH)) {
            bool nextTaller = nextColH > curColH;
            vec2 tallId = nextTaller ? nextCellId : curCellId;
            vec2 shortId = nextTaller ? curCellId : nextCellId;
            hit = true; hitSide = true; hitCellId = tallId; t = tBound;
            hitWallDir = normalize(hexCenter(shortId, S) - hexCenter(tallId, S));
            hitWallLo = min(curColH, nextColH);
            hitWallHi = max(curColH, nextColH);
            break;
        }
        tc = tNext; curCellId = nextCellId; curColH = nextColH;
    }

    if (!hit) {
        t = tEnd;
        hit = t < MAX_DIST;
        hitSide = false;
        hitCellId = curCellId;
    }

    if (hit) {
        vec3 p = ro + rd * t;
        vec3 n;
        vec3 albedo;
        float rHit = length(hexCenter(hitCellId, S));
        bool inTreble = rHit < trebleOuterRadius();
        float ringBlend = ringPitchAt(float(ROWS - 1)) * 0.5;
        float bassZoneT = smoothstep(outerRingInnerRadius() - ringBlend, outerRingInnerRadius() + ringBlend, rHit);
        float glow = 0.0;
        vec3 boostJ = vec3(0.0);
        float underGlow = 0.0;
        float hotGlow = 0.0;

        if (!hitSide) {
            vec2 local = p.xz - hexCenter(hitCellId, S);
            float h, raw;
            sampleField(hexCenter(hitCellId, S), h, raw);
            float topElev = h * HEIGHT_SCALE;
            float inradius = S * 0.8660254;
            float dTop = (hexEdgeDist(local, inradius) + inradius) / inradius;
            float isTop = 1.0 - smoothstep(BEVEL_FLAT - 0.08, BEVEL_FLAT, dTop);
            #define PLANE_HEIGHT_TREBLE 2.52
            #define PLANE_HEIGHT_MID 2.2
            #define PLANE_HEIGHT_BASS 2.0
            float planeHeight = mix(PLANE_HEIGHT_MID, PLANE_HEIGHT_BASS, bassZoneT);
            planeHeight = inTreble ? PLANE_HEIGHT_TREBLE : planeHeight;
            float above = step(planeHeight, topElev);

            n = vec3(0.0, 1.0, 0.0);

            vec3 baseAlbedo = vec3(0.10, 0.11, 0.14);
            vec3 boostBaseCol = mix(vec3(0.05, 0.85, 1.0), vec3(1.1, 0.06, 0.15), bassZoneT);
            boostBaseCol = inTreble ? vec3(0.2, 1.15, 0.4) : boostBaseCol;
            vec3 boost = boostBaseCol * (above * isTop);
            float tj = hash21(hitCellId + 17.0);
            boostJ = boost * (0.9 + 0.2 * fract(tj * 7.0));
            albedo = baseAlbedo + boostJ;
            glow = above * isTop;

            #define PLANE_HEIGHT_HOT (HEIGHT_CEIL * HEIGHT_SCALE * 0.6)
            #define PLANE_SOFT_HOT 0.5
            #define PLANE_HEIGHT_HOT_TREBLE (HEIGHT_CEIL * HEIGHT_SCALE * 0.75)
            float planeHeightHot = inTreble ? PLANE_HEIGHT_HOT_TREBLE : PLANE_HEIGHT_HOT;
            float hot = step(planeHeightHot, topElev);
            hotGlow = hot * isTop;

            underGlow = smoothstep(0.82, 1.0, dTop) * (1.0 - above);
        } else {
            n = vec3(hitWallDir.x, 0.0, hitWallDir.y);

            float tj = hash21(hitCellId + 31.0);
            albedo = vec3(0.10, 0.11, 0.14) * (0.85 + 0.3 * fract(tj * 7.0));

            float v = clamp((p.y - hitWallLo) / max(hitWallHi - hitWallLo, 0.0001), 0.0, 1.0);
            float ridge = smoothstep(0.9, 1.0, v);
            vec3 ridgeCol = mix(vec3(0.05, 0.75, 0.9), vec3(0.85, 0.05, 0.15), bassZoneT);
            ridgeCol = inTreble ? vec3(0.15, 0.95, 0.35) : ridgeCol;
            albedo = mix(albedo, ridgeCol, ridge * 0.8);
        }

        vec3 underGlowCol = vec3(0.95, 0.05, 0.13);
        vec3 hotCol = mix(vec3(0.75, 1.0, 1.0), vec3(0.1, 0.95, 0.9), bassZoneT);
        hotCol = inTreble ? vec3(0.85, 1.0, 0.8) : hotCol;

        vec3 peakCol = mix(boostJ, hotCol, hotGlow);

        vec3 lit = albedo;
        lit += peakCol * glow;
        lit += underGlowCol * underGlow;

        col = lit;
    }

    float fog = 1.0 - exp(-t * 0.015);
    vec3 fogCol = vec3(0.05, 0.10, 0.13);
    col = mix(col, fogCol, clamp(fog, 0.0, 1.0));
    }

    col = col / (1.0 + col);

    fragColor = vec4(col, 1.0);
}
