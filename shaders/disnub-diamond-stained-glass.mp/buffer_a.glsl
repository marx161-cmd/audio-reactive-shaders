// Disnub-Diamond-Stained-Glass: Buffer A, the gem's orientation (mouse trackball with fling),
// row 0 px 0..2, self-wired on iChannel0.

vec4 S(int px) { return texelFetch(iChannel0, ivec2(px, 0), 0); }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 ip = ivec2(fragCoord);
    if (ip.y != 0 || ip.x > 2) { fragColor = vec4(0.0); return; }
    float dt = clamp(iTimeDelta, 1e-4, 0.05);
    bool seeded = S(1).w > 0.5;
    vec4 q = seeded ? normalize(S(0)) : normalize(vec4(0.3, 0.2, 0.0, 1.0));
    vec3 omega = seeded ? S(1).xyz : vec3(0.0);
    vec2 prevM = S(2).xy * iResolution.xy;
    vec2 curM = iMouse.xy;
    bool steering = (u_steer > 0.5) && seeded && (S(2).w > 0.5) && (dot(prevM, prevM) > 0.0);
    if (steering) {
        vec2 u0 = (-iResolution.xy + 2.0 * prevM) / iResolution.y;
        vec2 u1 = (-iResolution.xy + 2.0 * curM) / iResolution.y;
        vec2 d = u1 - u0;
        float dl = length(d);
        omega = dl > 1e-6 ? (vec3(-d.y, d.x, 0.0) / dl) * (asin(min(dl, 1.0)) * TRACKBALL_GAIN) : vec3(0.0);
    } else {
        omega *= exp(-dt / SPIN_TAU);
        if (dot(omega, omega) < 1e-8) omega = vec3(0.0);
    }
    float om = length(omega);
    if (om > SPIN_MAX) omega *= SPIN_MAX / om;
    if (om > 1e-7) q = normalize(quatMul(quatAxisAngle(omega, om), q));
    q = normalize(quatMul(quatAxisAngle(vec3(0.0, 1.0, 0.0), BASE_SPIN * dt), q));
    if (ip.x == 0) fragColor = q;
    else if (ip.x == 1) fragColor = vec4(omega, 1.0);
    else fragColor = vec4(curM / iResolution.xy, 0.0, u_steer);
}
