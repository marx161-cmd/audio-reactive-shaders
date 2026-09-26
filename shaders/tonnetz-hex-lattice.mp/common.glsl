// Tonnetz-Hex-Lattice: common code, prepended to every pass.
//
// Maps the 120 semitone bands (27.5 Hz .. 28 kHz, 10 octaves) onto a polar hex
// field: angle = pitch class, ring = octave, pin height (image pass) = loudness.
// Also the hex-grid helpers and the colour palette.

const float PI = 3.14159265359;

uniform sampler2D bands;

#define N_BANDS 120
#define OCTAVES 10.0
#define LOG2RATIO (OCTAVES / float(N_BANDS))
#define PC0 9.0

#define ROWS 9
#define COLS 12
#define ROW_OCTAVES 1.0
#define HUB_RADIUS 1.4
#define RING_PITCH 2.0

#define SEMIS_PER_BAND (12.0 * LOG2RATIO)

float bandPc(int b)  { return mod(PC0 + (float(b) + 0.5) * SEMIS_PER_BAND, 12.0); }
float bandReg(int b) { return (float(b) + 0.5) * LOG2RATIO; }
float bandRow(int b) { return floor(bandReg(b) / ROW_OCTAVES); }
float bandCol(int b) { return mod(floor(bandPc(b) + 0.5), float(COLS)); }

#define HEIGHT_GAIN 8.0
#define HEIGHT_CEIL 5.0

#define BAND_GAMMA 2.0

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

#define PLANE_HEIGHT 1.8
#define PLANE_SOFT 0.8

vec3 fieldColor(float elevation) {
    float above = smoothstep(PLANE_HEIGHT, PLANE_HEIGHT + PLANE_SOFT, elevation);
    return mix(vec3(0.0), vec3(0.92, 0.05, 0.32), above);
}

vec2 hexRound(vec2 axial) {
    float x = axial.x, z = axial.y, y = -x - z;
    float rx = round(x), ry = round(y), rz = round(z);
    float xDiff = abs(rx - x), yDiff = abs(ry - y), zDiff = abs(rz - z);
    if (xDiff > yDiff && xDiff > zDiff) {
        rx = -ry - rz;
    } else if (yDiff > zDiff) {
        ry = -rx - rz;
    } else {
        rz = -rx - ry;
    }
    return vec2(rx, rz);
}

vec2 hexCenter(vec2 cellId, float size) {
    return size * vec2(
        1.7320508 * cellId.x + 0.8660254 * cellId.y,
        1.5 * cellId.y
    );
}

void hexInfo(vec2 p, float size, out vec2 cellId, out vec2 local) {
    float q = (0.57735027 * p.x - 0.33333333 * p.y) / size;
    float r = (0.66666667 * p.y) / size;
    cellId = hexRound(vec2(q, r));
    local = p - hexCenter(cellId, size);
}

float sdHexPrism(vec3 p, vec2 h) {
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735027);
    vec3 q = abs(p);
    q.xy -= 2.0 * min(dot(k.xy, q.xy), 0.0) * k.xy;
    vec2 d = vec2(
        length(q.xy - vec2(clamp(q.x, -k.z * h.x, k.z * h.x), h.x)) * sign(q.y - h.x),
        q.z - h.y
    );
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
