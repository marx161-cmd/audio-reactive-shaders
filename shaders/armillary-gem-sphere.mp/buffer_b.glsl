vec3 bandRaw() {
    return vec3(0.6 * u_bass + 0.4 * u_sub,
                0.5 * u_lowmid + 0.5 * u_mid,
                (u_highmid + u_presence + u_brilliance) / 3.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 ip = ivec2(fragCoord);
    if (ip.y != 0 || ip.x > 7) { fragColor = vec4(0.0); return; }
    int px = ip.x;
    float dt = clamp(iTimeDelta, 1e-4, 0.05);
    bool seeded = S(3).w > 0.5;

    float db = u_beat_count - S(3).x;
    float bpsNow = (db > 0.0 && db < 0.5) ? db / dt : 0.0;
    float bps = seeded ? mix(S(3).y, bpsNow, 1.0 - exp(-dt / TEMPO_TAU)) : 0.0;
    if (px == 3) { fragColor = vec4(u_beat_count, bps, 0.0, 1.0); return; }

    vec3 raw = bandRaw();
    float kr = 1.0 - exp(-dt / RANGE_TAU);
    vec3 fl = seeded ? min(raw, mix(S(5).xyz, raw, kr)) : raw;
    vec3 ce = seeded ? max(raw, mix(S(6).xyz, raw, kr)) : raw + 0.1;
    if (px == 5) { fragColor = vec4(fl, 0.0); return; }
    if (px == 6) { fragColor = vec4(ce, 0.0); return; }
    if (px == 4) {
        vec3 tgt = clamp((raw - fl) / max(ce - fl, vec3(0.08)), 0.0, 1.0);
        vec3 cur = seeded ? S(4).xyz : tgt;
        vec3 tau = mix(vec3(LEVEL_RELEASE), vec3(LEVEL_ATTACK), step(cur, tgt));
        fragColor = vec4(cur + (tgt - cur) * (1.0 - exp(-dt / tau)), 0.0);
        return;
    }

    if (px == 7) {
        float yaw = seeded ? S(7).x : 0.0;
        fragColor = vec4(mod(yaw + YAW_SHARE * S(0).y * dt, 6.28318531), 0.0, 0.0, 0.0);
        return;
    }

    int k = px;
    vec4 st = seeded ? S(k) : vec4(0.0);
    float ang = st.x, vel = st.y;
    float q = 3.14159265;
    float target = RING_DIR[k] * q * RING_NOTES[k] * u_beat_count;
    float err = mod(target - ang + 0.5 * q, q) - 0.5 * q;
    float vRef = RING_DIR[k] * q * RING_NOTES[k] * bps + LOCK_GAIN / RING_TAU[k] * err;
    vel += (vRef - vel) * (1.0 - exp(-dt / RING_TAU[k]));
    ang = mod(ang + vel * dt, 6.28318531);
    fragColor = vec4(ang, vel, 0.0, 0.0);
}
