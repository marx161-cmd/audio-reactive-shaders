// Nebulabrot-Infinite-Dive: Buffer A, the camera.
//
// Texel 0 holds the parked view (centre, zoom, rotation), saved by the renderer
// to shader_state/ and restored on start. Flights to dive stops (cam_target.txt,
// see tools/dive.py) take 16 beats of the beat clock.

uniform vec3 u_cam_look;
uniform vec3 u_cam_move;
uniform vec4 u_persist0;
uniform float u_persist_valid;
uniform float u_autozoom;
uniform float u_bass;
uniform vec4 u_cam_target;
uniform float u_cam_target_id;
uniform float u_beat_count;

#define BEAT_WRAP 1024.0
#define FLY_BEATS 16.0
#define FLY_SURGE 8.0
#define FLY_SWELL 0.8

#define AUTOZOOM_RATE 0.07
#define AUTOZOOM_BASS 0.8
#define AUTOZOOM_GLIDE 3.0

#define LOOK_PX   (1.0 / 0.0022)
#define ZOOM_RATE 0.075

vec4 S(int px) { return texelFetch(iChannel0, ivec2(px, 0), 0); }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 ip = ivec2(fragCoord);
    if (ip.y != 0 || ip.x > 4) { fragColor = vec4(0.0); return; }
    bool seeded = S(2).x > 0.5;
    vec4 view = seeded ? S(0)
              : (u_persist_valid > 0.5 && u_persist0.z >= Z_MIN) ? u_persist0
              : vec4(CX - 0.15, 0.0, 1.15, 0.0);
    vec4 last = seeded ? S(1) : vec4(u_cam_look, u_cam_move.z);

    vec2 dLook = (u_cam_look.xy - last.xy) * LOOK_PX;
    float dRoll = u_cam_look.z - last.z;
    float dF = u_cam_move.z - last.w;

    if (u_autozoom > 0.5) {
        float dt = clamp(iTimeDelta, 0.0, 0.05);
        dF += AUTOZOOM_RATE * (1.0 + AUTOZOOM_BASS * u_bass) * dt / ZOOM_RATE;
        view.xy += (DROSTE_F - view.xy) * (1.0 - exp(-dt / AUTOZOOM_GLIDE));
    }

    view.z = clamp(view.z * exp(dF * ZOOM_RATE), Z_MIN, zoomCap(view.xy));
    view.w += dRoll;

    vec2 duv = 2.0 * dLook / iResolution.y;
    float cr = cos(view.w), sr = sin(view.w);
    vec2 s = vec2(-duv.y, duv.x);
    s = vec2(cr * s.x - sr * s.y, sr * s.x + cr * s.y);
    view.xy -= (HALF / view.z) * s;
    view.x = clamp(view.x, CX - HALF, CX + HALF);
    view.y = clamp(view.y, -HALF, HALF);
    view.z = min(view.z, zoomCap(view.xy));

#if DROSTE_ON
    if (view.z >= DROSTE_S * DROSTE_Z1 && length(view.xy - DROSTE_F) < 0.02) {
        view.xy = DROSTE_F + (view.xy - DROSTE_F) * DROSTE_S;
        view.z /= DROSTE_S;
    }
#endif

    vec4 fStart = S(3), fState = seeded ? S(4) : vec4(0.0, u_cam_target_id, 0.0, 0.0);
    bool steered = dot(dLook, dLook) > 0.0 || abs(dF) > 0.0 || abs(dRoll) > 0.0;
    if (u_cam_target_id != fState.y && u_cam_target.z > 0.0) {
#if LOOP_ON
        if (windowAt(u_cam_target.xy, 0.0) == 1 && view.z < LOOP_Z0 * 1.05
            && length(view.xy - LOOP_S0) * view.z / HALF < 0.5) {
            view.xy = LOOP_S6 + (view.xy - LOOP_S0) / LOOP_K;
            view.z = min(view.z * LOOP_K, LOOP_Z6);
        }
#endif
        fStart = view;
        fState = vec4(u_beat_count, u_cam_target_id, 1.0, 0.0);
    }
    if (fState.z > 0.5 && steered && u_autozoom < 0.5) fState.z = 0.0;
    if (fState.z > 0.5) {
        fState.w += mod(u_beat_count - fState.x + BEAT_WRAP, BEAT_WRAP);
        fState.x = u_beat_count;
        float b = min(fState.w, FLY_BEATS);
        const float TAU = 6.28318531;
        float sF = (b - FLY_SWELL * FLY_SURGE / TAU * sin(TAU * b / FLY_SURGE)) / FLY_BEATS;
        sF = clamp(sF, 0.0, 1.0);
        float e = sF * sF * sF * (sF * (sF * 6.0 - 15.0) + 10.0);
        vec4 tgt = u_cam_target;
        tgt.z = clamp(tgt.z, Z_MIN, zoomCap(tgt.xy));
        float dist = length(tgt.xy - fStart.xy);
        float zFit = clamp(0.8 * HALF / max(dist, 1e-9), Z_MIN, 1e9);
        float lz0 = log(fStart.z), lz1 = log(tgt.z);
        float dip = max(0.0, min(lz0, lz1) - log(zFit));
        view.z = exp(mix(lz0, lz1, e) - dip * sin(3.14159265 * e));
        view.xy = mix(fStart.xy, tgt.xy, e);
        view.w = mix(fStart.w, tgt.w, e);
        if (b >= FLY_BEATS) fState.z = 0.0;
    }
#if LOOP_ON
    if (fState.z < 0.5 && view.z >= LOOP_Z6 * 0.999
        && length(view.xy - LOOP_S6) * view.z / HALF < 0.5) {
        view.xy = loopMap(view.xy);
        view.z = max(view.z / LOOP_K, Z_MIN);
    }
#endif
    if (ip.x == 3) { fragColor = fStart; return; }
    if (ip.x == 4) { fragColor = fState; return; }

    if (ip.x == 0) fragColor = view;
    else if (ip.x == 1) fragColor = vec4(u_cam_look, u_cam_move.z);
    else fragColor = vec4(1.0);
}
