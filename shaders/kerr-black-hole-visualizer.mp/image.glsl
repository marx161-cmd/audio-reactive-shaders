// Kerr-Black-Hole-Visualizer: image pass.
//
// Per pixel: walk the precomputed Kerr-Newman light path, test it against the
// accretion disk and every body (culled system -> star family -> planet ->
// moons), then shade the sky where the path escapes. Body positions come from
// Buffer A.

float sSlot(int px)       { return texelFetch(iChannel0, ivec2(px, 0), 0).r; }
float macroBand(int px)   { return sSlot(px); }
float arcEnv(int i)       { return texelFetch(iChannel0, ivec2(i, 2), 0).g; }
vec4  bodyAt(int k, int b) {
    return texelFetch(iChannel0, ivec2(ROW2_BODY_BASE + k * BODIES_PER_COPY + b, 2), 0);
}
vec4  baryAt(int k)       { return bodyAt(k, 6); }
vec4  moonAt(int k, int m) { return texelFetch(iChannel0, ivec2(k * MOONS_PER_COPY + m, 3), 0); }

vec4 gBody[5];

float shadowAll(vec3 wp, vec3 L, float dL, int selfIdx, int lightIdx) {
    float s = 1.0;
    for (int i = 0; i < 5; i++) {
        if (i == selfIdx || i == lightIdx) continue;
        s *= occludeBy(wp, L, dL, gBody[i].xyz, gBody[i].w);
    }
    return s;
}

bool testRingPlane(float row, vec3 er, vec3 et, vec3 c, vec3 nrm,
                   out float ringArc, out vec3 ringHit, out float cosInc) {
    vec3  prev = geoPoint(row, 0, er, et);
    float dPrev = dot(prev - c, nrm);
    float arc  = 0.0;
    for (int i = 1; i < int(GEO_N_S); i++) {
        vec3  cur = geoPoint(row, i, er, et);
        float dCur = dot(cur - c, nrm);
        float seg = length(cur - prev);
        if (dPrev * dCur < 0.0) {
            float f = dPrev / (dPrev - dCur);
            ringHit = mix(prev, cur, f);
            ringArc = arc + seg * f;
            cosInc  = abs(dot(normalize(cur - prev), nrm));
            return true;
        }
        arc  += seg;
        prev  = cur;
        dPrev = dCur;
    }
    return false;
}

void sceneCopy(int kCopy, vec3 ro, vec3 rd, float row, vec3 er, vec3 et,
               float occLimit,
               float bass, float mid, float treble,
               float envStar, float envGiant, float envMoons,
               out vec4 solid, out float depth, out vec3 glowOut,
               out vec3 diskEmitOut, out float metHeadD2Out,
               out bool metLiveOut, out float metHeatOut, out float metDoomOut) {
    solid = vec4(0.0); depth = 1e5; glowOut = vec3(0.0);

    bool  diskOwner = (kCopy == 0);
    vec3  dn      = diskNormal();
    vec3  mn      = vec3(0.0);
    vec3  emitD   = vec3(0.0);
    float diskTr  = 1.0;
    float headD2  = 1e30;
    bool  metLive = false;
    vec3  mpos    = vec3(0.0);
    float heat    = 0.0, doom = 0.0;
    vec4  me0 = vec4(0.0), me1 = vec4(0.0), me2 = vec4(0.0);
    float tIn = 0.0;
    float dPrevD = 0.0, dPrevM = 0.0;
    if (diskOwner) {
        me0 = texelFetch(iChannel0, ivec2(STATE_MET_PX,     0), 0);
        me1 = texelFetch(iChannel0, ivec2(STATE_MET_PX + 1, 0), 0);
        me2 = texelFetch(iChannel0, ivec2(STATE_MET_PX + 2, 0), 0);
        tIn = meteorTStart(me0, me1);
        float tOut = meteorTEnd(me0, me1);
        float mr = 1e9;
        metLive = (me2.w > 0.5) && (me0.y > 1e-4) && (iTime >= tIn) && (iTime <= tOut);
        if (metLive) {
            vec3 mvel, pF, vF; float rF;
            meteorAt(me0, me1, me2, iTime,        mpos, mvel, mr);
            meteorAt(me0, me1, me2, iTime + 0.05, pF,   vF,   rF);
            heat = clamp((length(pF - mpos) / 0.05) * 0.22, 0.10, 1.8);
            doom = smoothstep(0.30, 0.0, abs(iTime - me0.x));
            metLive = metLive && (mr < MET_VIS_R);
        }
        mn = me2.xyz;
    }

    vec4 bary = baryAt(kCopy);

    vec3 hitP   = vec3(0.0);
    vec3 hitV   = rd;

    vec4 bPulsar = bodyAt(kCopy, 0);
    vec4 bDwarf  = bodyAt(kCopy, 1);
    vec4 bPG     = bodyAt(kCopy, 2);
    vec4 bDA     = bodyAt(kCopy, 3);
    vec4 bG3     = bodyAt(kCopy, 4);

    gBody[0] = bPulsar; gBody[1] = bDwarf;
    gBody[2] = bPG;     gBody[3] = bDA;     gBody[4] = bG3;

    float minT   = 1e5;
    float hitId  = 0.0;
    vec3  hitC   = vec3(0.0);
    float hitR   = 1.0;
    int   hitSelf = -1;
    float hitSec  = 0.0;

    vec3  prev = geoPoint(row, 0, er, et);
    float arc  = 0.0;

    if (diskOwner) {
        dPrevD = dot(prev, dn);
        dPrevM = dot(prev, mn);
    }

    float glareD2P = 1e30;
    float glareD2D = 1e30;
    vec3 pNearPos = ro;
    vec3 pNearDir = rd;
    float pNearArc = 0.0;
    vec3  pHaloPos = ro;
    float pHaloArc = 0.0;
    int   pNearJ   = 1;
    float pNearTs  = 0.0;

    for (int i = 1; i < int(GEO_N_S); i++) {
        vec3  cur = geoPoint(row, i, er, et);
        float seg = length(cur - prev);

        if (diskOwner && arc < minT) {
            float dCurD = dot(cur, dn);
            float dCurM = dot(cur, mn);
            vec3  sdir = (cur - prev) / max(seg, 1e-6);
            if (DISK_SHOW == 1 && dPrevD * dCurD < 0.0) {
                float fx = dPrevD / (dPrevD - dCurD);
                vec3  X  = mix(prev, cur, fx);
                {
                    vec3 scrR = normalize(cross(-normalize(ro), dn));
                    X -= normalize(scrR - dot(scrR, dn) * dn) * (DISK_SHIFT * BH_RS);
                }
                float rW = length(X) / BH_RS, aW = KERR_A * 0.5;
                float rB = sqrt(max(rW * rW - aW * aW, 0.0));
                if (rB >= DSIM_RIN && rB <= DSIM_ROUT) {
                    vec3  e1 = normalize(cross(dn, vec3(0.0, 0.0, 1.0)));
                    vec3  e2 = cross(dn, e1);
                    float phB  = atan(dot(X, e2), dot(X, e1));
                    float cosI = abs(dot(sdir, dn));
                    float colS = 2.50663 * DISK_HR * rB / max(cosI, 0.05);
                    float phUse = (DISK_UNDER_FLIP == 1 && dPrevD < 0.0) ? -phB : phB;
                    vec4  sh = diskShareShade(rB, phUse, colS, kerrW(0), bass);
                    emitD  += sh.rgb * DISK_GLOW * diskTr;
                    diskTr *= mix(1.0, sh.a, DISK_COVER);
                }
            }
            if (metLive) {
                if (dPrevM * dCurM < 0.0) {
                    float f = dPrevM / (dPrevM - dCurM);
                    vec3  X = mix(prev, cur, f);
                    float dens = meteorTrailDensity(me0, me1, me2, X, iTime, tIn);
                    if (dens > 0.0) {
                        float cosI = abs(dot(sdir, mn));
                        emitD += mix(COMET_COL_DEEP, COMET_COL_BODY, dens)
                               * dens * MET_TRAIL_GAIN / max(cosI, 0.12)
                               * (0.55 + 1.2 * heat);
                    }
                }
                headD2 = min(headD2, geoSegPointD2(prev, cur, mpos));
            }
            dPrevD = dCurD;
            dPrevM = dCurM;
        }

        float d2p = geoSegPointD2(prev, cur, bPulsar.xyz);
        if (d2p < glareD2P) {
            glareD2P = d2p;
            pNearPos = prev;
            pNearDir = normalize(cur - prev);
            pNearArc = arc;
            {
                vec3  dseg = cur - prev;
                float ts = clamp(dot(bPulsar.xyz - prev, dseg) / max(dot(dseg, dseg), 1e-12), 0.0, 1.0);
                pHaloPos = prev + dseg * ts;
                pHaloArc = arc + seg * ts;
                pNearJ = i; pNearTs = ts;
            }
        }
        glareD2D = min(glareD2D, geoSegPointD2(prev, cur, bDwarf.xyz));

        float bestF  = 2.0;
        float bestId = 0.0;
        vec4  bestB  = vec4(0.0);
        vec3  bestP  = vec3(0.0);
        float th; vec3 hp;

        if (geoSegSphere(prev, cur, bPulsar.xyz, bPulsar.w, th, hp)) {
            if (th < bestF) { bestF = th; bestId = 1.0; bestB = bPulsar; bestP = hp; } }
        if (geoSegSphere(prev, cur, bDwarf.xyz, bDwarf.w, th, hp)) {
            if (th < bestF) { bestF = th; bestId = 2.0; bestB = bDwarf; bestP = hp; } }
        if (geoSegSphere(prev, cur, bPG.xyz, bPG.w, th, hp)) {
            if (th < bestF) { bestF = th; bestId = 3.0; bestB = bPG; bestP = hp; } }
        if (geoSegSphere(prev, cur, bDA.xyz, bDA.w, th, hp)) {
            if (th < bestF) { bestF = th; bestId = 4.0; bestB = bDA; bestP = hp; } }
        if (geoSegSphere(prev, cur, bG3.xyz, bG3.w, th, hp)) {
            if (th < bestF) { bestF = th; bestId = 5.0; bestB = bG3; bestP = hp; } }

        if (geoSegPointD2(prev, cur, bPG.xyz) < PGIANT_BOUND * PGIANT_BOUND) {
            for (int m = M0_PGIANT; m < M0_DGA; m++) {
                vec4 mo = moonAt(kCopy, m);
                if (geoSegSphere(prev, cur, mo.xyz, mo.w, th, hp)) {
                    if (th < bestF) { bestF = th; bestId = 10.0 + float(m); bestB = mo; bestP = hp; }
                }
            }
        }
        if (geoSegPointD2(prev, cur, bDA.xyz) < DGA_BOUND * DGA_BOUND) {
            for (int m = M0_DGA; m < TOTAL_MOONS; m++) {
                vec4 mo = moonAt(kCopy, m);
                if (geoSegSphere(prev, cur, mo.xyz, mo.w, th, hp)) {
                    if (th < bestF) { bestF = th; bestId = 10.0 + float(m); bestB = mo; bestP = hp; }
                }
            }
        }

        if (bestId > 0.5) {
            minT    = arc + seg * bestF;
            hitId   = bestId;
            hitC    = bestB.xyz;
            hitR    = bestB.w;
            hitSelf = (bestId < 6.0) ? int(bestId) - 1 : -1;
            hitSec  = 0.0;
            hitP    = bestP;
            hitV    = normalize(cur - prev);
            break;
        }

        arc  += seg;
        prev  = cur;
    }

    float ringT = 1e5, ringOpac = 0.0;
    vec3  ringShade = vec3(0.0);
    vec3  rn = copyRot(ringNormalDGA(kCopy), kCopy);
    float trRing, dnr;
    vec3  rpHit;
    if (testRingPlane(row, er, et, bDA.xyz, rn, trRing, rpHit, dnr)) {
        if (trRing > 0.0 && trRing < minT) {
            vec3  rp   = rpHit - bDA.xyz;
            float dens = ringDensity(length(rp), bDA.w, kCopy);
            if (dens > 0.0) {
                vec3 wp = rpHit;
                vec3 d1 = bPulsar.xyz - wp;  float l1 = length(d1);
                vec3 d2 = bDwarf.xyz  - wp;  float l2 = length(d2);
                float sh1 = occludeBy(wp, d1 / l1, l1, bDA.xyz, bDA.w);
                float sh2 = occludeBy(wp, d2 / l2, l2, bDA.xyz, bDA.w);
                ringShade = RING_ALBEDO
                          * (STAR1_COL * (sh1 * 1.05 + 0.10)
                           + STAR2_COL * (sh2 * 0.95)
                           + diskBodyLight(wp, rn) * RING_DISKLIGHT)
                          * (0.60 + 0.45 * dens);
                ringOpac = 1.0 - exp(-dens * 2.0 / max(dnr, 0.1));
                ringT    = trRing;
            }
        }
    }

    if (minT > occLimit) { hitId = 0.0; minT = 1e5; }

    vec3 col = vec3(0.0);
    float cover = 0.0;
    if (hitId > 0.5) {
        cover = 1.0;
        vec3 n = (hitP - hitC) / hitR;

        if (hitId == 1.0) {
            col = neutronStarSurface(hitP - hitC, hitV, hitR, bass, iTime);
        } else if (hitId == 2.0) {
            col = redDwarfSurface(hitP - hitC, hitV, hitR, mid, iTime, copyDwarfTint(kCopy));
        } else {
            vec3 albT; float isGiant = 0.0;
            vec3 gu, gv, gn;

            if (hitId == 3.0) {
                gearBasis(float(GEAR_PGIANT), gu, gv, gn); gn = copyRot(gn, kCopy);
                albT = rockAlbedo(n, gn, iTime, 0.37,
                  mix(PG_ROCK_BASE,
                      hueRotate(PG_HUE_BASE,
                                lapHue(ORBIT_BEATS_PG, PT_HUEPH(PT_ARR_PG, PT_SUB_PG, ORBIT_BEATS_PG), PG_HUE_OFF)),
                      PG_HUE_SAT));
            } else if (hitId == 4.0) {
                albT = giantAlbedo(n, copyRot(ringNormalDGA(kCopy), kCopy), iTime, DGA_HUE + copyGiantShift(kCopy)
                                    + lapHue(ORBIT_BEATS_DGA, PT_HUEPH(PT_ARR_DGA, PT_SUB_DGA, ORBIT_BEATS_DGA), 0.0), DGA_BANDS,
                                   0.042, 0.026) * DGA_TINT;
                isGiant = 1.0;
            } else if (hitId == 5.0) {
                albT = giantAlbedo(n, diskNormal(), iTime, GG3_HUE + copyGiantShift(kCopy)
                                    + lapHue(ORBIT_BEATS_GG3, PT_HUEPH(PT_ARR_GG3, PT_SUB_GG3, ORBIT_BEATS_GG3), 0.0), GG3_BANDS,
                                   0.042, 0.026) * GG3_TINT;
                isGiant = 1.0;
            } else {
                float mi = hitId - 10.0;
                vec3 j = moonJitter(mi);
                vec3 ax = copyRot(normalize(vec3(j.x - 0.5, 1.0, j.z - 0.5)), kCopy);
                albT = rockAlbedo(n, ax, iTime, fract(mi * 0.2718 + 0.13),
                                  mix(vec3(0.86, 0.84, 0.82), vec3(1.02, 0.94, 0.80),
                                      fract(mi * 0.35)));
                if (kCopy == 0 && mi == float(M0_PGIANT)) albT *= 1.65;
            }

            vec3  d1 = bPulsar.xyz - hitP;  float l1 = length(d1);
            vec3  d2 = bDwarf.xyz  - hitP;  float l2 = length(d2);
            float sh1 = shadowAll(hitP, d1 / l1, l1, hitSelf, 0);
            float sh2 = shadowAll(hitP, d2 / l2, l2, hitSelf, 1);

            col = shadeBinary(n, hitV, hitP, bPulsar.xyz, bDwarf.xyz, albT,
                              envMoons, iTime, isGiant, bass, sh1, sh2);
        }

        if (hitSec > 0.5) col *= vec3(0.72, 0.88, 1.22) * 0.80;
    }

    if (ringOpac > 0.001 && ringT < minT && ringT < occLimit) {
        col   = mix(col, ringShade, ringOpac);
        cover = max(cover, ringOpac);
        minT  = min(minT, ringT);
    }

    float occT = min(minT, occLimit);

    vec3 glow = vec3(0.0);

    if (glareD2P < 1e29) {
        int   j  = pNearJ;
        vec3  pA = geoPoint(row, max(j - 2, 0), er, et);
        vec3  pB = geoPoint(row, j - 1, er, et);
        vec3  pC = geoPoint(row, j, er, et);
        vec3  pD = geoPoint(row, min(j + 1, int(GEO_N_S) - 1), er, et);
        vec3  dCur  = normalize(pC - pB);
        vec3  dPrev = (j >= 2) ? normalize(pB - pA) : dCur;
        vec3  dNext = (j + 1 < int(GEO_N_S)) ? normalize(pD - pC) : dCur;
        pNearDir = normalize((1.0 - pNearTs) * (dPrev + dCur) + pNearTs * (dCur + dNext));
        pNearPos = pHaloPos;
        pNearArc = pHaloArc;
    }
    float tLeft = (occLimit < 1e4) ? max(arc - pNearArc, 0.0) : 1e5;
    tLeft = min(tLeft, max(minT - pNearArc, 0.0));

    glow += hueRotate(calcPulsarJets(pNearPos, pNearDir, bPulsar.xyz,
                                     copyRot(getPulsarAxis(), kCopy),
                                     bass, treble, tLeft, iTime),
                      copyCoronaShift(kCopy));

    if (glareD2P < 1e29) {
        float dRay = sqrt(glareD2P);
        float k = CORONA_R / max(dRay, CORONA_R * 0.22);
        float f = pow(k, CORONA_POW);
        vec3  cc = hueRotate(mix(CORONA_EDGE, CORONA_CORE, clamp(f * 0.30, 0.0, 1.0)),
                             copyCoronaShift(kCopy));
        glow += cc * f * (0.30 + bass * 1.10);
        {
            vec3  off = pHaloPos - bPulsar.xyz;
            vec2  o2  = vec2(dot(off, gCamR), dot(off, gCamU));
            float ca = cos(SPIKE_ANGLE), sa = sin(SPIKE_ANGLE);
            o2 = vec2(ca * o2.x + sa * o2.y, -sa * o2.x + ca * o2.y);
            float sp = 0.0;
            for (int a = 0; a < 2; a++) {
                float along = abs(a == 0 ? o2.x : o2.y);
                float across = abs(a == 0 ? o2.y : o2.x);
                sp += exp(-(across * across) / (SPIKE_W * SPIKE_W))
                    * pow(CORONA_R / (along + CORONA_R), SPIKE_POW);
            }
            glow += cc * SPIKE_GAIN * sp * (0.30 + bass * 1.10);
        }
#if SURF_HQ
        {
            float hk = GLOWQ_P_R * CORONA_R / (dRay + GLOWQ_P_R * CORONA_R);
            glow += hueRotate(CORONA_EDGE, copyCoronaShift(kCopy))
                  * GLOWQ_P_GAIN * pow(hk, GLOWQ_P_POW) * (0.30 + bass * 1.10);
        }
#endif
    }

    {
        vec3 mAxis = copyRot(getPulsarAxis(), kCopy);
        float tLeftH = (occLimit < 1e4) ? max(arc - pHaloArc, 0.0) : 1e5;
        tLeftH = min(tLeftH, max(minT - pHaloArc, 0.0));
        vec2 hb = intersectSphere(pHaloPos - bPulsar.xyz, pNearDir, HALO_BOUND);
        if (hb.y > 0.0 && glareD2P < HALO_BOUND * HALO_BOUND) {
            float t0 = hb.x;
            float t1 = min(hb.y, tLeftH);
            if (t1 > t0) {
                float dt = (t1 - t0) / float(HALO_STEPS);
                float acc = 0.0;
                for (int i = 0; i < HALO_STEPS; i++) {
                    vec3 p = pHaloPos + pNearDir * (t0 + (float(i) + 0.5) * dt) - bPulsar.xyz;
                    acc += synchrotronDensity(p, mAxis, bass, treble, iTime);
                }
                acc *= dt * HALO_GAIN;
                vec3 sc = mix(vec3(0.20, 0.60, 1.00), vec3(0.70, 0.90, 1.40),
                              clamp(bass * 1.3, 0.0, 1.0));
                sc = mix(sc, vec3(0.62, 0.40, 1.30), clamp(treble * 0.8, 0.0, 1.0));
                sc = hueRotate(sc, copyCoronaShift(kCopy));
                glow += sc * acc;
            }
        }
    }
    if (glareD2D < 1e29) {
        float dRay = sqrt(glareD2D);
        float k = bDwarf.w / max(dRay, bDwarf.w * 0.88);
        vec3 dg = copyDwarfTint(kCopy);
        dg = dg / max(max(dg.r, dg.g), dg.b);
        glow += dg * 1.10 * pow(k, 3.5) * (0.35 + mid * 1.1);
#if SURF_HQ
        {
            float hk = GLOWQ_D_R * bDwarf.w / (dRay + GLOWQ_D_R * bDwarf.w);
            glow += dg * GLOWQ_D_GAIN * pow(hk, GLOWQ_D_POW) * (0.35 + mid * 1.1);
        }
#endif
    }

    if (diskOwner) {
        diskEmitOut = emitD;
        gDiskTr = diskTr;
        metHeadD2Out = headD2; metLiveOut = metLive;
        metHeatOut = heat; metDoomOut = doom;
    } else {
        diskEmitOut = vec3(0.0); metHeadD2Out = 1e30; metLiveOut = false;
        metHeatOut = 0.0; metDoomOut = 0.0;
    }

    solid   = vec4(col, cover);
    depth   = minT;
    glowOut = glow;
}

uniform sampler2DArray u_knsky;
vec3 knSky(vec3 rdL, float treble, float skyB) {
    vec3 c = vec3(0.0);
    for (int k = 0; k < 4; k++) {
        vec4 d = texelFetch(u_knsky, ivec3(gPix, k), 0);
        if (d.w > 0.0) c += d.w * skyColorDispersed(rdL, normalize(d.xyz) - rdL, treble, skyB);
    }
    return c;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    gPix = ivec2(fragCoord);
    vec2 uv = (-iResolution.xy + 2.0 * fragCoord.xy) / iResolution.y;

    float bass   = macroBand(13);
    float mid    = macroBand(14);
    float treble = macroBand(15);

    float envStar  = arcEnv(0);
    float envGiant = arcEnv(1);
    float envMoons = arcEnv(2);

    vec3 camND = diskNormal();
    vec3 camE1 = normalize(cross(camND, vec3(0.0, 0.0, 1.0)));
    vec3 camE2 = cross(camND, camE1);

    #define CAM_AZ0    (-3.103562)
    #define CAM_EL0    (0.087266)
    #define CAM_ROLL0  (0.0)

    #define CAM_LOCKED 1

    float camAz   = CAM_AZ0;
    float camEl   = CAM_EL0;
    float camRoll = CAM_ROLL0;
    float camR    = GEO_CAM_R * BH_RS;

    vec3 camDir = cos(camEl) * (cos(camAz) * camE1 + sin(camAz) * camE2)
                + sin(camEl) * camND;
    vec3 roL = camR * camDir;

    vec3 camF  = -camDir;
    vec3 camR3 = normalize(cross(camF, camND));
    vec3 camU  = cross(camR3, camF);
    float cr = cos(camRoll), sr = sin(camRoll);
    vec3 rr = normalize(camR3 * cr + camU * sr);
    camU  = cross(rr, camF);
    camR3 = rr;

    vec3 rdL = normalize(camR3 * (uv.x - SENSOR_POS.x)
                       + camU  * (uv.y - SENSOR_POS.y)
                       + camF  * 1.8);

    gRo = roL; gRd = rdL;

    gHoleDir = normalize(-roL);
    gCamR = camR3; gCamU = camU;
    gBass    = bass;
    gPondR   = length(uv);
    vec3  gEr, gEt;
    geoFrame(roL, rdL, gEr, gEt);
    float geoB = kerrWarpB(geoImpactParam(roL, rdL), gEr, gEt);
    bool  capt = geoCaptured(geoB);
    float gRow  = geoRow(max(geoB, 0.0));
    vec3  gExit = geoExitDir(gRow, gEr, gEt);
    {
        vec3 pp = rdL  - dot(rdL,  gHoleDir) * gHoleDir;
        vec3 ep = gExit - dot(gExit, gHoleDir) * gHoleDir;
        float side = dot(ep, pp * inversesqrt(max(dot(pp, pp), 1e-12)));
        gOutE   = capt ? 0.0 : smoothstep(-EIN_EDGE, EIN_EDGE, side);
        gSplitM = capt ? 0.0 : smoothstep(-EIN_OVER - EIN_EDGE, -EIN_OVER, side);
        gPondM  = capt ? 1.0 : 1.0 - smoothstep(EIN_OVER, EIN_OVER + EIN_EDGE, side);
    }

    vec3  skyBend = (geoB >= 0.0) ? (gExit - rdL) : vec3(0.0);
    float skyB    = (geoB >= 0.0) ? geoB * BH_RS  : 1e5;

    vec3 nebAdd = capt ? vec3(0.0) : nebulaAt(gExit, bass);

    if (intersectSphere(roL, rdL, SPHERE_RADIUS).y <= 0.0) {
        vec3 bg = knSky(rdL, treble, skyB);
        bg += nebAdd;
        fragColor = vec4(globalGrade(bg / (1.0 + bg * 0.32)), 1.0);
        return;
    }

    float occLimit = capt ? max(dot(-roL, rdL), 0.0) : 1e5;

    vec4  bestSolid = vec4(0.0);
    float bestDepth = 1e5;
    vec3  glowSum   = vec3(0.0);

    vec3  diskEmit  = vec3(0.0);
    float metHeadD2 = 1e30;
    bool  metLive   = false;
    float metHeat   = 0.0, metDoom = 0.0;

    gDiskTr = 1.0;
    for (int k = 0; k < N_COPIES; k++) {
        vec4  cs; float cd; vec3 cg;
        vec3  dkEmit; float dkHead; bool dkLive; float dkHeat, dkDoom;
        sceneCopy(k, roL, rdL, gRow, gEr, gEt, occLimit, bass, mid, treble,
                  envStar, envGiant, envMoons, cs, cd, cg,
                  dkEmit, dkHead, dkLive, dkHeat, dkDoom);
        glowSum += cg;
        if (cd < bestDepth) { bestDepth = cd; bestSolid = cs; }
        if (k == 0) {
            diskEmit = dkEmit; metHeadD2 = dkHead; metLive = dkLive;
            metHeat = dkHeat; metDoom = dkDoom;
        }
    }

    vec3 col = knSky(rdL, treble, skyB);
    col += nebAdd;
    col = mix(col, bestSolid.rgb, clamp(bestSolid.a, 0.0, 1.0));
    col *= gDiskTr;
    col += glowSum;

    col += diskToneSum(diskEmit);

    if (geoB >= 0.0 && GEO_KERR == 0) {
        float caustic = exp(-pow((geoB - GEO_B_CRIT) / PRING_THICK, 2.0));
        if (caustic > 1e-3) {
            vec4  sP = bodyAt(0, 0);
            vec4  sD = bodyAt(0, 1);
            float aP = max(0.0, dot(-camDir, normalize(sP.xyz)));
            float aD = max(0.0, dot(-camDir, normalize(sD.xyz)));
            if (aP > PRING_ALIGN) col += STAR1_COL * caustic * pow(aP, 64.0) * PRING_GAIN;
            if (aD > PRING_ALIGN) col += copyDwarfTint(0) * caustic * pow(aD, 64.0) * PRING_GAIN;
        }
    }

    if (metLive) {
        float dPerp = sqrt(metHeadD2);
        float core  = exp(-(dPerp * dPerp) / (MET_CORE_R * MET_CORE_R));
        float glowH = exp(-(dPerp * dPerp) / (MET_GLOW_R * MET_GLOW_R));
        col += COMET_COL_HEAD * core  * (1.6 + 2.4 * metHeat + 7.0 * metDoom);
        col += COMET_COL_HEAD * glowH * (0.20 + 0.7 * metHeat + 3.0 * metDoom);
    }

    col = col / (1.0 + col * 0.32);
    col = globalGrade(col);

    fragColor = vec4(col, 1.0);
}
