// Kerr-Black-Hole-Visualizer: Buffer A, state and ephemeris.
//
// Only the top-left STATE_W x STATE_H texels are written; everything else
// returns 0. Self-wired on iChannel0 for frame-to-frame feedback.
//   row 0  camera, audio levels (AGC, bass/mid/treble), gear phases, beat clock, meteors
//   row 1  cleaned band levels
//   row 2  strike envelopes, then the major bodies per copy (xyz = position, w = radius)
//   row 3  moons per copy (xyz = position, w = radius)
// The image pass reads every body position from here and does no orbital math.

float prevPhase(int g) {
    return statePhase(g);
}

uniform sampler2D stems;

uniform float u_tbeat;
uniform float u_beat_phase;
uniform float u_beat_count;
uniform float u_beat_conf;

#define KICK_BAND_LO  4
#define KICK_BAND_HI  27
#define KICK_THRESH   0.08
#define KICK_REFRACTORY 0.28
#define KICK_BASE_TAU 1.0

vec3 copyToWorld(vec3 pLocal, int k) { return copyRot(pLocal, k); }

vec3 pulsarPos(int k) { return starOrbitPos(0); }
vec3 dwarfPos(int k)  { return starOrbitPos(1); }

vec3 pgiantPos(int k) { return planetTouchPos(PT_RP_PG,  PT_ECC_PG,  PT_PSI_PG,  PGIANT_R, PT_ALPHA_PG,  PT_BETA_PG,  ORBIT_BEATS_PG,  PT_SUB_PG,  PT_ARR_PG);  }
vec3 gg3Pos(int k)    { return planetTouchPos(PT_RP_GG3, PT_ECC_GG3, PT_PSI_GG3, GG3_R, PT_ALPHA_GG3, PT_BETA_GG3, ORBIT_BEATS_GG3, PT_SUB_GG3, PT_ARR_GG3); }
vec3 dgaPos(int k)    { return planetTouchPos(PT_RP_DGA, PT_ECC_DGA, PT_PSI_DGA, DGA_R,    PT_ALPHA_DGA, PT_BETA_DGA, ORBIT_BEATS_DGA, PT_SUB_DGA, PT_ARR_DGA); }

void meteorSpawn(float tNow, out vec4 el0, out vec4 el1, out vec4 el2) {
    vec4  clock     = texelFetch(iChannel0, ivec2(STATE_BEAT_PX, 0), 0);
    float tBeat     = clock.r <= 0.05 ? 0.5 : clock.r;
    float timeSince = max(clock.b, 0.0);
    float sd        = floor(tNow * 0.37);

    float coinKind = h21(vec2(sd, 21.3));
    float coinLoop = h21(vec2(sd, 33.9));
    bool  bound    = coinKind > 0.55;
    float loops    = (coinLoop > 0.5) ? 2.0 : 1.0;

    float e, nMot, lead;
    if (bound) {
        float mf = max(2.0, 2.0 * floor(MET_LOOP_BEATS / 2.0 + 0.5));
        float T  = mf * tBeat;
        e    = MET_ECC_BOUND;
        nMot = 6.28318531 / T;
        lead = 0.5 * T;
    } else {
        e    = MET_ECC;
        nMot = meteorEdgeM(e) / max(MET_SPAN_BEATS * tBeat, 1e-3);
        lead = MET_SPAN_BEATS * tBeat;
    }

    float rest     = 2.0 * floor(2.0 * h21(vec2(sd, 12.7)) + 1.0);
    float lastBeat = tNow - timeSince;
    float tPeri    = lastBeat + tBeat * (ceil(timeSince / tBeat + 0.001)
                                         + ceil(lead / tBeat) + rest);

    vec3  n  = normalize(vec3(h21(vec2(sd, 1.7)) - 0.5,
                              h21(vec2(sd, 5.3)) - 0.5,
                              h21(vec2(sd, 9.1)) - 0.5) + vec3(0.0, 0.0, 0.22));
    vec3  ref = (abs(n.y) > 0.9) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3  b1  = normalize(cross(ref, n));
    vec3  b2  = cross(n, b1);
    float ph  = 6.28318531 * h21(vec2(sd, 17.5));
    vec3  u   = normalize(b1 * cos(ph) + b2 * sin(ph));

    el0 = vec4(tPeri, nMot, MET_PERI, e);
    el1 = vec4(u, loops);
    el2 = vec4(n, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 ip = ivec2(fragCoord);
    if (ip.x >= STATE_W || ip.y >= STATE_H) { fragColor = vec4(0.0); return; }

    if (ip.y == 0) {
        int px = ip.x;
        if (px > STATE_BEATCNT_PX && px >= PULSE_PX + PULSE_N) { fragColor = vec4(0.0); return; }

        if (px == 12) {
            float mx = 0.0;
            for (int i = 0; i < int(N_BANDS); i++)
                mx = max(mx, texelFetch(iChannel0, ivec2(i, 1), 0).b);
            float agc = texelFetch(iChannel0, ivec2(12, 0), 0).r;
            agc = mx > agc ? mix(agc, mx, AGC_ATTACK)
                           : max(mx, agc - (AGC_RELEASE_DBPS / DB_SPAN) * iTimeDelta);
            agc = max(agc, (AGC_FLOOR_DB - DB_MIN) / DB_SPAN);
            fragColor = vec4(agc, 0.0, 0.0, 1.0);
            return;
        }

        if (px >= 13 && px <= 15) {
            int start = (px - 13) * 36;
            float fMax = 0.0;
            for (int k = 0; k < 36; k++)
                fMax = max(fMax, texelFetch(iChannel0, ivec2(start + k, 1), 0).r);
            fragColor = vec4(fMax, 0.0, 0.0, 1.0);
            return;
        }

        float elev = texelFetch(iChannel0, ivec2(4, 0), 0).r;
        float yaw  = texelFetch(iChannel0, ivec2(5, 0), 0).r;
        float init = texelFetch(iChannel0, ivec2(6, 0), 0).r;
        float fE   = texelFetch(iChannel0, ivec2(7, 0), 0).r;
        float fY   = texelFetch(iChannel0, ivec2(8, 0), 0).r;
        vec3  omega = texelFetch(iChannel0, ivec2(9, 0), 0).xyz;

        vec4 q = normalize(vec4(texelFetch(iChannel0, ivec2(0, 0), 0).r,
                                texelFetch(iChannel0, ivec2(1, 0), 0).r,
                                texelFetch(iChannel0, ivec2(2, 0), 0).r,
                                texelFetch(iChannel0, ivec2(3, 0), 0).r));

        if (init < 0.5) {
            q = viewQuat(u_cam_elev, u_cam_yaw);
            omega = vec3(0.0);
            elev = u_cam_elev; yaw = u_cam_yaw;
            fE = u_cam_elev;   fY = u_cam_yaw;
        }

        if (abs(u_cam_elev - fE) > 1e-6 || abs(u_cam_yaw - fY) > 1e-6) {
            float dE = u_cam_elev - fE;
            float dY = u_cam_yaw  - fY;
            q = normalize(quatMul(quatMul(quatAxisAngle(vec3(1.0, 0.0, 0.0), dE),
                                          quatAxisAngle(vec3(0.0, 1.0, 0.0), dY)), q));
            elev += dE; yaw += dY;
            fE = u_cam_elev; fY = u_cam_yaw;
        }

        vec2 curM  = iMouse.xy;
        vec2 prevM = vec2(texelFetch(iChannel0, ivec2(10, 0), 0).r,
                          texelFetch(iChannel0, ivec2(11, 0), 0).r) * iResolution.xy;
        bool steering = (u_steer > 0.5) && (dot(prevM, prevM) > 0.0);
        if (steering) {
            vec2 u0 = (-iResolution.xy + 2.0 * prevM) / iResolution.y;
            vec2 u1 = (-iResolution.xy + 2.0 * curM)  / iResolution.y;
            vec2  d  = u1 - u0;
            float dl = length(d);
            omega = (dl > 1e-6)
                  ? (vec3(-d.y, d.x, 0.0) / dl) * (asin(min(dl, 1.0)) * TRACKBALL_GAIN)
                  : vec3(0.0);
        } else {
            omega *= exp(-iTimeDelta / SPIN_TAU);
            if (dot(omega, omega) < 1e-6) omega = vec3(0.0);
        }
        float om = length(omega);
        if (om > SPIN_MAX) omega *= SPIN_MAX / om;

        float ang = length(omega);
        if (ang > 1e-6) {
            vec4 dq = vec4(omega / ang * sin(ang * 0.5), cos(ang * 0.5));
            q = normalize(quatMul(dq, q));
        }

        if (px == 4)  { fragColor = vec4(elev, 0.0, 0.0, 1.0); return; }
        if (px == 5)  { fragColor = vec4(yaw,  0.0, 0.0, 1.0); return; }
        if (px == 6)  { fragColor = vec4(1.0,  0.0, 0.0, 1.0); return; }
        if (px == 7)  { fragColor = vec4(fE,   0.0, 0.0, 1.0); return; }
        if (px == 8)  { fragColor = vec4(fY,   0.0, 0.0, 1.0); return; }
        if (px == 9)  { fragColor = vec4(omega, 1.0); return; }
        if (px == 10) { fragColor = vec4(curM.x / iResolution.x, 0.0, 0.0, 1.0); return; }
        if (px == 11) { fragColor = vec4(curM.y / iResolution.y, 0.0, 0.0, 1.0); return; }

        if (px >= 17 && px <= 24) {
            int   g   = px - 17;
            float fg  = float(g);
            float prev = prevPhase(g);

            float sma = 1.0;
            if      (g == GEAR_BINARY) sma = BARY_REF;
            else if (g == GEAR_PGIANT) sma = ORB_PGIANT;
            else if (g == GEAR_PROCK)  sma = ORB_PROCK;
            else if (g == GEAR_DGA)    sma = ORB_DGA;
            else if (g == GEAR_DGB)    sma = ORB_DGB;
            else if (g == GEAR_STAR0)  sma = BARY_R_0;
            else if (g == GEAR_STAR1)  sma = BARY_R_1;

            float idleRate = (0.0115 / pow(sma / BARY_REF, 1.5)) * iTimeDelta;

            fragColor = vec4(fract(prev + idleRate), 0.0, 0.0, 1.0);
            return;
        }

        if (px == STATE_BEAT_PX) {
            if (u_tbeat > 0.05) {
                fragColor = vec4(u_tbeat, u_beat_phase,
                                 u_beat_phase * u_tbeat, u_beat_conf);
                return;
            }

            vec4  prev      = texelFetch(iChannel0, ivec2(STATE_BEAT_PX, 0), 0);
            float tBeat     = prev.r <= 0.05 ? 0.5 : prev.r;
            float phase     = prev.g;
            float timeSince = prev.b + iTimeDelta;

            float kickNow = 0.0;
            for (int b = KICK_BAND_LO; b <= KICK_BAND_HI; b++) {
                kickNow = max(kickNow, texelFetch(stems, ivec2(b, 0), 0).r);
            }

            float baseCoef = exp(-iTimeDelta / KICK_BASE_TAU);
            float baseline = baseCoef * prev.a + (1.0 - baseCoef) * kickNow;
            float kick     = kickNow - baseline;
            if (kick > KICK_THRESH && timeSince > KICK_REFRACTORY) {
                if (timeSince < 1.2) tBeat = mix(tBeat, timeSince, 0.30);
                phase = 0.0;
                timeSince = 0.0;
            } else {
                phase = fract(phase + iTimeDelta / max(tBeat, 0.05));
            }
            fragColor = vec4(tBeat, phase, timeSince, baseline);
            return;
        }

        if (px == STATE_BEATCNT_PX) {
            if (u_beat_count > 0.0) {
                fragColor = vec4(u_beat_count, 0.0, 0.0, 0.0);
                return;
            }

            vec4  prevC  = texelFetch(iChannel0, ivec2(STATE_BEATCNT_PX, 0), 0);
            vec4  clk    = texelFetch(iChannel0, ivec2(STATE_BEAT_PX, 0), 0);
            float tBeatC = clk.r <= 0.05 ? 0.5 : clk.r;
            float count  = mod(prevC.r + iTimeDelta / max(tBeatC, 0.05), BEAT_WRAP);
            fragColor = vec4(count, 0.0, 0.0, 0.0);
            return;
        }

        if (px >= PULSE_DET_PX && px < PULSE_PX + PULSE_N) {
            vec4  det  = texelFetch(iChannel0, ivec2(PULSE_DET_PX, 0), 0);
            float kickNow = 0.0;
            for (int b = KICK_BAND_LO; b <= KICK_BAND_HI; b++)
                kickNow = max(kickNow, texelFetch(stems, ivec2(b, 0), 0).r);
            float baseCoef = exp(-iTimeDelta / KICK_BASE_TAU);
            float baseline = baseCoef * det.r + (1.0 - baseCoef) * kickNow;
            float kick     = kickNow - baseline;
            float since    = det.g + iTimeDelta;
            bool  tracked  = u_tbeat > 0.05 && u_beat_count > 0.0;
            bool  hit;
            float str;
            if (tracked) {
                hit = det.a > 0.0 && floor(u_beat_count) != floor(det.a)
                      && u_beat_conf > 0.3;
                str = clamp(texelFetch(iChannel0, ivec2(13, 0), 0).r * 1.2, 0.2, 1.0);
            } else {
                hit = kick > KICK_THRESH && since > KICK_REFRACTORY;
                str = clamp(kick / (KICK_THRESH * 4.0), 0.25, 1.0);
            }
            int   slot     = int(det.b + 0.5) % PULSE_N;
            if (px == PULSE_DET_PX) {
                fragColor = vec4(baseline, hit ? 0.0 : since,
                                 float(hit ? (slot + 1) % PULSE_N : slot),
                                 tracked ? u_beat_count : 0.0);
                return;
            }
            vec4 p = texelFetch(iChannel0, ivec2(px, 0), 0);
            if (hit && px - PULSE_PX == slot)
                fragColor = vec4(0.0, str, 0.0, 1.0);
            else
                fragColor = vec4(p.r + iTimeDelta, p.g, 0.0, 1.0);
            return;
        }

        if (px >= STATE_MET_PX && px <= STATE_MET_PX + 2) {
            vec4 e0 = texelFetch(iChannel0, ivec2(STATE_MET_PX,     0), 0);
            vec4 e1 = texelFetch(iChannel0, ivec2(STATE_MET_PX + 1, 0), 0);
            vec4 e2 = texelFetch(iChannel0, ivec2(STATE_MET_PX + 2, 0), 0);

            float tEnd = meteorTEnd(e0, e1);
            if (e2.w > 0.5 && e0.y > 1e-3 && iTime < tEnd) {
                fragColor = (px == STATE_MET_PX)     ? e0
                          : (px == STATE_MET_PX + 1) ? e1 : e2;
                return;
            }

            vec4 n0, n1, n2;
            meteorSpawn(iTime, n0, n1, n2);
            fragColor = (px == STATE_MET_PX)     ? n0
                      : (px == STATE_MET_PX + 1) ? n1 : n2;
            return;
        }

        float o = 0.0;
        if      (px == 0) o = q.x;
        else if (px == 1) o = q.y;
        else if (px == 2) o = q.z;
        else if (px == 3) o = q.w;
        fragColor = vec4(o, 0.0, 0.0, 1.0);
        return;
    }

    if (ip.y == 1) {
        int b = ip.x;
        if (b >= int(N_BANDS)) { fragColor = vec4(0.0); return; }
        float fb = float(b);

        float a = amp(fb);

        float x = max(0.0, a - LEAK_K * max(amp(fb - 1.0), amp(fb + 1.0)));
        x = max(0.0, x - H2_K * amp(fb - 12.0) - H3_K * amp(fb - 19.0) - H4_K * amp(fb - 24.0));
        float sum = 0.0, cnt = 0.0;
        for (float k = -6.0; k <= 6.0; k += 1.0) {
            float i = fb + k;
            if (i >= 0.0 && i < N_BANDS) { sum += amp(i); cnt += 1.0; }
        }
        float floorFactor = mix(FLOOR_K, FLOOR_K * 0.35, fb / N_BANDS);
        x = max(0.0, x - floorFactor * (sum / cnt));

        float db    = 6.0206 * log2(max(x, 1e-9)) + TILT_DB * (fb / 12.0 - 4.5);
        float level = clamp((db - DB_MIN) / DB_SPAN, 0.0, 1.0);
        float agc   = max(texelFetch(iChannel0, ivec2(12, 0), 0).r,
                          (AGC_FLOOR_DB - DB_MIN) / DB_SPAN);
        float target = clamp((level - agc) * (DB_SPAN / DISPLAY_DB) + 1.0, 0.0, 1.0);

        float prevB = texelFetch(iChannel0, ivec2(b, 1), 0).g;
        float tau   = (target > prevB) ? BAND_ATTACK_TAU : BAND_RELEASE_TAU;
        float raw   = mix(prevB, target, clamp(1.0 - exp(-iTimeDelta / tau), 0.0, 1.0));
        float clean = pow(raw, 1.5);

        fragColor = vec4(clean, clean, level, 1.0);
        return;
    }

    if (ip.y == 2) {
        int px = ip.x;

        if (px < N_TARGETS) {
            float lvl   = texelFetch(iChannel0, ivec2(13 + px, 0), 0).r;
            vec4  prev  = texelFetch(iChannel0, ivec2(px, 2), 0);
            float above = lvl > ARC_GATE ? 1.0 : 0.0;
            float rising = (above > 0.5 && prev.a <= 0.5) ? 1.0 : 0.0;
            float env = rising > 0.5 ? min(1.0, 0.45 + lvl * 0.75)
                                     : prev.g * exp(-iTimeDelta / ARC_TAU);

            float loud = max(max(texelFetch(iChannel0, ivec2(13, 0), 0).r,
                                 texelFetch(iChannel0, ivec2(14, 0), 0).r),
                                 texelFetch(iChannel0, ivec2(15, 0), 0).r);
            if (loud < IDLE_GATE) {
                float epoch = floor(iTime / IDLE_PERIOD);
                float h     = fract(sin(epoch * 91.37) * 43758.5453);
                float pick  = floor(h * float(N_TARGETS));
                float when  = epoch * IDLE_PERIOD + fract(h * 7.13) * IDLE_PERIOD * 0.8;
                if (float(px) == pick && abs(iTime - when) < 0.02) env = IDLE_LEVEL;
            }
            fragColor = vec4(lvl, env, 0.0, above);
            return;
        }

        int idx = px - ROW2_BODY_BASE;
        int k   = idx / BODIES_PER_COPY;
        int b   = idx - k * BODIES_PER_COPY;
        if (k >= N_COPIES) { fragColor = vec4(0.0); return; }

        if (b == 0) { fragColor = vec4(pulsarPos(k), PULSAR_R); return; }
        if (b == 1) { fragColor = vec4(dwarfPos(k),  DWARF_R);  return; }
        if (b == 2) { fragColor = vec4(pgiantPos(k), PGIANT_R); return; }
        if (b == 3) { fragColor = vec4(dgaPos(k),    DGA_R);    return; }
        if (b == 4) { fragColor = vec4(gg3Pos(k),    GG3_R);    return; }
        fragColor = vec4(0.0, 0.0, 0.0, SPHERE_RADIUS);
        return;
    }

    int mIdxAll = ip.x;
    int kc = mIdxAll / MOONS_PER_COPY;
    int m  = mIdxAll - kc * MOONS_PER_COPY;
    if (kc >= N_COPIES) { fragColor = vec4(0.0); return; }
    float cpM = copyPhase(kc);

    float fm = float(m);
    vec3  j  = moonJitter(fm);
    float jA = j.x;
    float jB = j.y;
    float jC = (j.z - 0.5);

    float beatCountRaw = texelFetch(iChannel0, ivec2(STATE_BEATCNT_PX, 0), 0).r;
    float bcI = floor(beatCountRaw), bcF = fract(beatCountRaw);
    float beatCount = bcI + mix(bcF, 1.0 - pow(1.0 - bcF, MOON_SURGE), MOON_SURGE_MIX);

    vec3  parent;
    float parentGear, ph, dist, rad, inc;

    if (m < M0_DGA) {
        float k = fm;
        parent = pgiantPos(kc);  parentGear = float(GEAR_PGIANT);
        float rate = (m == 0) ? 3.97 : (m == 1) ? 5.83 : (m == 2) ? 7.19 : 9.41;
        rate += 0.15 * (jB - 0.5);
        float beatsPer = (m == 0) ? MOON_BEATS_PG0 : (m == 1) ? MOON_BEATS_PG1
                       : (m == 2) ? MOON_BEATS_PG2 : MOON_BEATS_PG3;
        ph     = fract(beatCount / beatsPer + jA + cpM + cloneClock(kc));
        dist   = PGIANT_M_R0 + PGIANT_M_DR * k;
        rad    = 0.0110 + 0.0025 * k;
        inc    = ((m == 0) ? -0.31 : (m == 1) ? 0.12 : (m == 2) ? -0.08 : 0.34)
               + jC * 0.10;
    } else {
        float k = fm - float(M0_DGA);
        parent = dgaPos(kc);     parentGear = float(GEAR_DGA);
        float beatsPerD = (k < 0.5) ? MOON_BEATS_DGA0 : MOON_BEATS_DGA1;
        ph     = fract(beatCount / beatsPerD + jA + cpM + cloneClock(kc));
        dist   = DGA_M_R0 + DGA_M_DR * k;
        rad    = 0.0160 + 0.0020 * k;
        inc    = jC * 0.035;
    }

    vec3 u, v, n;
    gearBasis(parentGear, u, v, n);
    vec3 uA = u;
    vec3 vA = v * cos(inc) + n * sin(inc);

    float th = ph * TAU;
    fragColor = vec4(parent + copyToWorld((uA * cos(th) + vA * sin(th)) * dist, kc), rad);
}
