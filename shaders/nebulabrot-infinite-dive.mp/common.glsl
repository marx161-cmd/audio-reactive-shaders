// Nebulabrot-Infinite-Dive: common code, prepended to every pass.
//
// The Nebulabrot (Buddhabrot coloured by escape time) from precomputed density
// textures: a 16384^2 global bake, two deep-zoom window bakes in the same
// brightness units, an equalised Metropolis "glitter" bake and an anti-
// Buddhabrot layer (nebula_bake_modal.py + the *_prep.py scripts). The view
// dives along a path of stops; the last stop is a copy of the start, so the dive
// loops forever. Audio arrives in the `bands` texture: 120 semitone bands, 0..1.

#define CX      -0.4
#define HALF     1.6
#define Z_MIN    1.0
#define Z_SHARP  6.8

#define DROSTE_ON 0
#define DROSTE_F  vec2(0.0, 0.0)
#define DROSTE_S  3.99
#define DROSTE_Z1      2.0
#define DROSTE_CANYON_G 1.4
#define DROSTE_SIL_LOD 6.3
#define DROSTE_SIL_LO  0.006
#define DROSTE_SIL_HI  0.02
#define DROSTE_FALL_LOD 8.5
#define DROSTE_FALL_LO  0.0015
#define DROSTE_FALL_MAX 0.7
#define DROSTE_RIM vec3(-0.2, 0.3, 3.5)
#define DROSTE_FLARE 0.9
#define DROSTE_TAIL_W   0.04

#define LOOP_ON 1
#define LOOP_S6 vec2(0.70436, 0.0)
#define LOOP_S0 vec2(CX, 0.0)
#define LOOP_Z0 Z_MIN
#define LOOP_Z6 666.0
#define LOOP_K  (LOOP_Z6 / LOOP_Z0)
vec2 loopMap(vec2 c) { return LOOP_S0 + (c - LOOP_S6) * LOOP_K; }

#define WIN_READY 1
#define WIN1_READY 1
#define N_WIN (1 + WIN1_READY)
const vec2 WIN[2] = vec2[2](vec2(0.5244, 0.2), vec2(0.70567, 0.008));
#define WIN1_FADE_LO 40.0
#define WIN1_FADE_HI 54.0
#define WIN_FADE_LO 4.0
#define WIN_FADE_HI 6.0

int windowAt(vec2 c, float margin) {
    for (int i = N_WIN - 1; i >= 0; i--)
        if (abs(c.x - WIN[i].x) < WIN[i].y * (1.0 - margin) && abs(c.y) < WIN[i].y * (1.0 - margin)) return i;
    return -1;
}
float zoomCap(vec2 c) {
#if DROSTE_ON
    if (length(c - DROSTE_F) < 0.3) return max(Z_SHARP, DROSTE_S * DROSTE_Z1 * 1.01);
#endif
#if LOOP_ON
    if (length(c - LOOP_S6) < 0.0015) return max(LOOP_Z6, Z_SHARP * HALF / WIN[1].y);
#endif
    int i = windowAt(c, 0.0);
    return (i < 0 || WIN_READY == 0) ? Z_SHARP : Z_SHARP * HALF / WIN[i].y;
}
