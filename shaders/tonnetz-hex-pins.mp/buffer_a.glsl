#define USE_CLEAN 1
uniform sampler2D u_clean;

float rawPeakAt(int semitone, int octave) {
    float peak = 0.0;
    for (int b = 0; b < N_BANDS; b++) {
        if (int(bandCol(b)) == semitone && int(bandRow(b)) == octave) {
#if USE_CLEAN
            peak = max(peak, texelFetch(u_clean, ivec2(b, 0), 0).r);
#else
            peak = max(peak, texelFetch(bands, ivec2(b, 0), 0).r);
#endif
        }
    }
    return peak;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 ip = ivec2(fragCoord);

    if (ip.x >= COLS || ip.y >= ROWS) {
        fragColor = vec4(0.0);
        return;
    }

    float peak = rawPeakAt(ip.x, ip.y);

#define OLD_HARMONIC 0
    if (OLD_HARMONIC == 1 && ip.y > 0) {
        float fundamental = rawPeakAt(ip.x, ip.y - 1);
        peak = max(0.0, peak - 0.3 * fundamental);

        int fifthSource = int(mod(float(ip.x - 7), 12.0));
        float fifthFund = rawPeakAt(fifthSource, ip.y - 1);
        peak = max(0.0, peak - 0.15 * fifthFund);
    }

    float octaveMean = 0.0;
    float octaveMax = 0.0;
    float octaveMin = 1e9;
    for (int s = 0; s < 12; s++) {
        float p = rawPeakAt(s, ip.y);
        octaveMean += p;
        octaveMax = max(octaveMax, p);
        octaveMin = min(octaveMin, p);
    }
    octaveMean /= 12.0;

    float flatness = (octaveMax - octaveMin) > 1e-4
        ? clamp((octaveMean - octaveMin) / (octaveMax - octaveMin), 0.0, 1.0)
        : 0.0;
    #define GATE_TONAL_SCALE 0.45
    #define GATE_FLAT_LO 0.30
    #define GATE_FLAT_HI 0.60
    float gateScale = mix(GATE_TONAL_SCALE, 1.0,
                          smoothstep(GATE_FLAT_LO, GATE_FLAT_HI, flatness));
    float floorFactor = (0.35 + 0.025 * float(ip.y)) * gateScale;
    peak = max(0.0, peak - floorFactor * octaveMean);

    #define TREBLE_BOOST 2.0
    float regBoost = mix(1.0, TREBLE_BOOST, float(ip.y) / float(ROWS - 1));
    peak *= regBoost;

    float gamma = 1.2 - 0.05 * float(ip.y);
    float e = pow(peak, gamma);
    #define CLEAN_GAIN 0.75
    float target = clamp(e * HEIGHT_GAIN * CLEAN_GAIN, 0.0, HEIGHT_CEIL);

    #define SETTLE_TAU_BASE 0.0025
    #define SETTLE_TAU_PER_OCTAVE 0.0012
    #define RELEASE_WEIGHT 30.0
    float settleTauUp = SETTLE_TAU_BASE + SETTLE_TAU_PER_OCTAVE * float(ip.y);
    float prevH = texelFetch(iChannel0, ip, 0).g;
    float settleTau = target > prevH ? settleTauUp : settleTauUp * RELEASE_WEIGHT;
    float alpha = clamp(1.0 - exp(-iTimeDelta / settleTau), 0.0, 1.0);
    float cleanH = mix(prevH, target, alpha);

    float colMain = 0.0;
    for (int r = 0; r < ROWS; r++) colMain = max(colMain, texelFetch(iChannel0, ivec2(ip.x, r), 0).g);
    int leftCol = int(mod(float(ip.x - 1), float(COLS)));
    int rightCol = int(mod(float(ip.x + 1), float(COLS)));
    float leftMain = 0.0, rightMain = 0.0;
    for (int r = 0; r < ROWS; r++) {
        leftMain = max(leftMain, texelFetch(iChannel0, ivec2(leftCol, r), 0).g);
        rightMain = max(rightMain, texelFetch(iChannel0, ivec2(rightCol, r), 0).g);
    }
    #define PULL_STRENGTH 0.03
    #define BRIDGE_STRENGTH 0.45
    float dispH = max(cleanH, PULL_STRENGTH * colMain);
    dispH = max(dispH, BRIDGE_STRENGTH * min(leftMain, rightMain));

    #define DRAG_TAU 1.4
    float dragPrev = texelFetch(iChannel0, ip, 0).b;
    float dragAlpha = clamp(1.0 - exp(-iTimeDelta / DRAG_TAU), 0.0, 1.0);
    float dragH = mix(dragPrev, cleanH, dragAlpha);

    #define ONSET_NORM 1.5
    float cellOnset = clamp((cleanH - dragH) / ONSET_NORM, 0.0, 1.0);

    fragColor = vec4(dispH, cleanH, dragH, cellOnset);
}
