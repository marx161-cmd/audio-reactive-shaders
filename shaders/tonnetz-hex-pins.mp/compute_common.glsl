#define H_LIM      0.36
#define H_SINK     0.34

uniform sampler2D u_field;

#define PI 3.14159265359
#define ROWS 9
#define COLS 12
#define HUB_RADIUS 1.4
#define RING_PITCH 2.0
#define RING_GROWTH 0.06
#define S 0.5
#define HEIGHT_SCALE 1.5
float ringPitchAt(float row) { return RING_PITCH * (1.0 + RING_GROWTH * row); }
#define WHEEL_SCALE 23.72

vec2 hexRound(vec2 axial) {
    float x = axial.x, z = axial.y, y = -x - z;
    float rx = round(x), ry = round(y), rz = round(z);
    float xDiff = abs(rx - x), yDiff = abs(ry - y), zDiff = abs(rz - z);
    if (xDiff > yDiff && xDiff > zDiff) rx = -ry - rz;
    else if (yDiff > zDiff)             ry = -rx - rz;
    else                                rz = -rx - ry;
    return vec2(rx, rz);
}
vec2 hexCenter(vec2 cellId, float size) {
    return size * vec2(1.7320508 * cellId.x + 0.8660254 * cellId.y, 1.5 * cellId.y);
}
void hexInfo(vec2 p, float size, out vec2 cellId, out vec2 local) {
    float q = (0.57735027 * p.x - 0.33333333 * p.y) / size;
    float r = (0.66666667 * p.y) / size;
    cellId = hexRound(vec2(q, r));
    local = p - hexCenter(cellId, size);
}
vec2 hexFrac(vec2 p, float size) {
    return vec2((0.57735027 * p.x - 0.33333333 * p.y) / size, (0.66666667 * p.y) / size);
}
float hexDistTo(vec2 myIdF, vec2 pinId) {
    vec2 d = myIdF - pinId;
    return (abs(d.x) + abs(d.x + d.y) + abs(d.y)) * 0.5;
}

float v2Field(vec2 tileCenter, float rnd, out float oct, out float pc) {
    oct = -1.0;
    pc = -1.0;
    float r = length(tileCenter);
    float theta = atan(tileCenter.y, tileCenter.x);
    if (theta < 0.0) theta += 2.0 * PI;

    if (r < HUB_RADIUS) {
        float bass = 0.0;
        for (int c = 0; c < COLS; c++) {
            float hv = texelFetch(u_field, ivec2(c, ROWS - 1), 0).r;
            if (hv > bass) { bass = hv; pc = float(c); }
        }
        #define HUB_HEX_RADIUS (HUB_RADIUS / S * 1.25)
        float d0 = hexDistTo(hexFrac(tileCenter, S), vec2(0.0));
        float d = d0 < 0.5 ? d0 : max(d0 + 0.6 * (2.0 * fract(rnd * 7.31 + 0.17) - 1.0), 0.5);
        float f = pow(clamp(1.0 - d / HUB_HEX_RADIUS, 0.0, 1.0), 3.5);
        f *= d0 < 0.5 ? 1.1 : mix(0.65, 1.0, rnd);
        oct = float(ROWS - 1);
        return bass * f;
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
    if (row < 0.0) return 0.0;

    float wSum = 0.0, oSum = 0.0;
    vec2 myIdF = hexFrac(tileCenter, S);

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
            vec4 cell = texelFetch(u_field, ivec2(colIdx, ROWS - 1 - candRow), 0);
            float peak = cell.r;
            float dragPeak = cell.b;
            sidePeak[side] = peak;
            sideDragPeak[side] = dragPeak;
            if (peak <= 0.0 && dragPeak <= 0.0) { contrib[side] = 0.0; continue; }

            float thetaC = colIdxF * colStep;
            vec2 center = rC * vec2(cos(thetaC), sin(thetaC));
            vec2 pinId, pinLocal;
            hexInfo(center, S, pinId, pinLocal);

            float hexDist0 = hexDistTo(myIdF, pinId);
            #define DIST_JITTER 0.6
            float hexDist = hexDist0 < 0.5 ? hexDist0
                          : max(hexDist0 + DIST_JITTER * (2.0 * fract(rnd * 7.31 + 0.17) - 1.0), 0.5);
            float dragT = clamp(hexDist / hexRadius, 0.0, 1.0);
            float effectivePeak = mix(peak, dragPeak, dragT);
            float falloff = pow(clamp(1.0 - hexDist / hexRadius, 0.0, 1.0), FALLOFF_POW);

            #define CENTER_BOOST 1.1
            #define SURROUND_DAMPER_MIN 0.65
            #define SURROUND_DAMPER_MAX 1.0
            if (hexDist0 < 0.5) falloff *= CENTER_BOOST;
            else falloff *= mix(SURROUND_DAMPER_MIN, SURROUND_DAMPER_MAX, rnd);

            float ownFrac = (side == 0) ? frac : (1.0 - frac);
            float angularWeight = 1.0 - smoothstep(0.0, beamWidth, ownFrac);
            float centerContrib = effectivePeak * falloff * angularWeight;

            float gapContrib = 0.0;
            for (int g = -1; g <= 1; g += 2) {
                float gThetaC = thetaC + float(g) * colStep * 0.5;
                vec2 gCenter = rC * vec2(cos(gThetaC), sin(gThetaC));
                vec2 gPinId, gPinLocal;
                hexInfo(gCenter, S, gPinId, gPinLocal);
                float gHexDist = max(hexDistTo(myIdF, gPinId) + DIST_JITTER * (2.0 * fract(rnd * 7.31 + 0.17) - 1.0), 0.0);
                float gDragT = clamp(gHexDist / hexRadius, 0.0, 1.0);
                float gEffectivePeak = mix(peak, dragPeak, gDragT);
                float gFalloff = pow(clamp(1.0 - gHexDist / hexRadius, 0.0, 1.0), FALLOFF_POW);
                gapContrib = max(gapContrib, gEffectivePeak * gFalloff * mix(SURROUND_DAMPER_MIN, SURROUND_DAMPER_MAX, rnd));
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
            float bHexDist = max(hexDistTo(myIdF, bPinId) + DIST_JITTER * (2.0 * fract(rnd * 7.31 + 0.17) - 1.0), 0.0);
            float bDragT = clamp(bHexDist / hexRadius, 0.0, 1.0);
            float bEffectivePeak = mix(bridgePeak, bridgeDragPeak, bDragT);
            float bFalloff = pow(clamp(1.0 - bHexDist / hexRadius, 0.0, 1.0), FALLOFF_POW);
            bridgeContrib = bEffectivePeak * bFalloff * mix(SURROUND_DAMPER_MIN, SURROUND_DAMPER_MAX, rnd);
        }

        float k = 0.2;
        float hBlend = clamp(0.5 + 0.5 * (contrib[0] - contrib[1]) / k, 0.0, 1.0);
        float rowBest = mix(contrib[1], contrib[0], hBlend) + k * hBlend * (1.0 - hBlend);
        float rc = max(rowBest, bridgeContrib);
        if (rc > best) pc = mod(colA + (contrib[1] > contrib[0] ? 1.0 : 0.0), float(COLS));
        best = max(best, rc);
        float w = rc * rc * rc * rc;
        wSum += w;
        oSum += w * float(ROWS - 1 - candRow);
    }
    if (wSum > 1e-8) oct = oSum / wSum;

    float maxR = HUB_RADIUS;
    for (int i = 0; i < ROWS; i++) maxR += ringPitchAt(float(i));
    float outerRingWidth = ringPitchAt(float(ROWS - 1));
    float fadeT = smoothstep(maxR - outerRingWidth, maxR, r);
    #define RIM_FADE_POW 2.5
    best *= 1.0 - pow(fadeT, RIM_FADE_POW);
    if (row > float(ROWS - 1) - 0.5) {
        float ring8Width = ringPitchAt(float(ROWS - 1));
        float ring8InnerEdge = maxR - ring8Width;
        best *= smoothstep(ring8InnerEdge, ring8InnerEdge + ring8Width * 0.15, r);
    }
    return best;
}

#define BR_IN       1.015
#define BR_OUT      1.135
#define BR_STEPS    16
#define BR_H        0.17
#define BR_SPREAD   0.75
uniform sampler2D u_beatring;

float beatRingAt(vec2 p, float rnd) {
    float r = length(p);
    if (r < BR_IN - 0.02 || r > BR_OUT + 0.02) return 0.0;
    float radial = smoothstep(BR_IN - 0.02, BR_IN + 0.015, r) * (1.0 - smoothstep(BR_OUT - 0.015, BR_OUT + 0.02, r));
    float phase = fract(0.25 - atan(p.y, p.x) / 6.2831853);
    float stepF = phase * float(BR_STEPS);
    float best = 0.0;
    for (int k = -1; k <= 1; k++) {
        int st = int(floor(stepF + 0.5)) + k;
        int sw = ((st % BR_STEPS) + BR_STEPS) % BR_STEPS;
        float lvl = texelFetch(u_beatring, ivec2(sw, 0), 0).r;
        if (lvl <= 0.0) continue;
        float d = abs(stepF - float(st)) / BR_SPREAD;
        d = d + 0.25 * (2.0 * fract(rnd * 7.31 + 0.17) - 1.0);
        float f = pow(clamp(1.0 - d, 0.0, 1.0), 2.0) * mix(0.65, 1.0, rnd);
        best = max(best, lvl * f);
    }
    return BR_H * best * radial;
}

float pressureAt(vec2 p, float rnd, out float oct, out float pc) {
    vec2 pv = p * WHEEL_SCALE;
    float h = max(v2Field(pv, rnd, oct, pc), 0.0);
    float hr = beatRingAt(p, rnd) * WHEEL_SCALE / HEIGHT_SCALE;
    if (hr > h) { h = hr; pc = 12.0; }
    return h * HEIGHT_SCALE / WHEEL_SCALE;
}

vec2 axialToWorld(vec2 ax, float s) {
    return vec2(s * 1.7320508 * (ax.x + 0.5 * ax.y), s * 1.5 * ax.y);
}

uint hashU(uvec2 q) {
    uint h = q.x * 1597334677u ^ q.y * 3812015801u;
    h ^= h >> 16; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    return h;
}
float hash1(ivec2 ip) { return float(hashU(uvec2(ip)) & 0xffffffu) / 16777215.0; }
