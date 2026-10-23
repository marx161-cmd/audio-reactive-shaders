// Kerr-Orrery: image pass.
//
// The Kerr-Newman black hole with its two stars, planets and moons on plain Kepler
// orbits timed to the tempo, seen from a camera that circles the hole on the
// equator. The camera rests inside a song section and swings a quarter turn on
// every section change. Light paths are the precomputed Kerr-Newman bake
// (u_kerrpaths, u_knsky) rotated with the camera; every 16 beats a star is torn
// apart and its gas stream and disk play from a pre-lensed flipbook
// (u_slurp_dwarf, rendered offline by the original Kerr-Newman renderer).
// Optional: director/ plans the bodies per song (song replay mode only).

float sSlot(int px)       { return texelFetch(iChannel0, ivec2(px, 0), 0).r; }
float macroBand(int px)   { return sSlot(px); }
float arcEnv(int i)       { return texelFetch(iChannel0, ivec2(i, 2), 0).g; }
vec4  bodyAt(int k, int b) {
    return texelFetch(iChannel0, ivec2(ROW2_BODY_BASE + k * BODIES_PER_COPY + b, 2), 0);
}
vec4  baryAt(int k)       { return bodyAt(k, 6); }
vec4  moonAt(int k, int m) { return texelFetch(iChannel0, ivec2(k * MOONS_PER_COPY + m, 3), 0); }

vec4 gBody[8];

vec3 planLight(vec3 n, vec3 rd, vec3 wp, vec3 albT, float isGas, float bass, int selfK);
float shadowAll(vec3 wp, vec3 L, float dL, int selfIdx, int lightIdx) {
    float s = 1.0;
    for (int i = 0; i < 8; i++) {
        if (i == selfIdx || i == lightIdx) continue;
        if (gBody[i].w <= 0.0) continue;
        s *= occludeBy(wp, L, dL, gBody[i].xyz, gBody[i].w);
    }
    return s;
}

bool testRingPlane(float row, vec3 er, vec3 et, vec3 c, vec3 nrm,
                   out float ringArc, out vec3 ringHit, out float cosInc) {
    vec3  prev = geoPoint(row, 0, er, et);
    float dPrev = dot(prev - c, nrm);
    float arc  = 0.0;
    bool noTail = geoCaptured(0.0);
    for (int i = 1; i <= int(GEO_N_S); i++) {
        if (i == int(GEO_N_S) && noTail) break;
        vec3  cur = (i == int(GEO_N_S)) ? prev + geoExitDir(row, er, et) * GEO_TAIL_LEN : geoPoint(row, i, er, et);
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
    vec4 bRA = (ROSTER_ON == 1) ? texelFetch(iChannel0, ivec2(ROSTER_PX0,     2), 0) : vec4(0.0);
    vec4 bRB = (ROSTER_ON == 1) ? texelFetch(iChannel0, ivec2(ROSTER_PX0 + 1, 2), 0) : vec4(0.0);
    gBody[5] = bRA;     gBody[6] = bRB;
    vec4 bRC = (VIC_ON == 1) ? texelFetch(iChannel0, ivec2(VIC_PX, 2), 0) : vec4(0.0);
    bool legacy = legacyOn();
    if (!legacy) { bPulsar.w = 0.0; bDwarf.w = 0.0; bPG.w = 0.0; bDA.w = 0.0; bG3.w = 0.0; bRA.w = 0.0; bRB.w = 0.0; }
    gBody[7] = bRC;

    float minT   = 1e5;
    float hitId  = 0.0;
    vec3  hitC   = vec3(0.0);
    float hitR   = 1.0;
    int   hitSelf = -1;
    float hitSec  = 0.0;

    vec3  prev = geoPoint(row, 0, er, et);
    float arc  = 0.0;
    bool  dHadGas = false;
    float dPassTr = 1.0;

    if (diskOwner) {
        dPrevD = dot(prev, dn);
        dPrevM = dot(prev, mn);
    }
    vec4  knAcc   = vec4(0.0);
    float knPhase = fract(sin(dot(vec2(gPix), vec2(12.9898, 78.233))) * 43758.5453);
    float knShell = 0.0;
    float knLam   = (KN_DISK == 1 && diskOwner) ? kerrW(0) : 0.0;
    vec3  knE1    = normalize(cross(dn, vec3(0.0, 0.0, 1.0)));
    vec3  knE2    = cross(dn, knE1);
    vec3  knPrevQ = vec3(dot(prev, knE1), dot(prev, dn), dot(prev, knE2)) / BH_RS;
    float knPrevR = knKSRadius(dot(knPrevQ.xz, knPrevQ.xz), knPrevQ.y * knPrevQ.y);
    vec3  knSrc   = (VIC_ON == 1) ? tdeHeader(1).xyz : ((gKNStar == 0) ? bPulsar.xyz : bDwarf.xyz);
    float knPhD   = atan(dot(knSrc, knE2), dot(knSrc, knE1));

    vec4  tdeH  = (TDE_ON == 1 && diskOwner) ? tdeHeader(0) : vec4(0.0);
    vec3  tdeN  = tdeHeader(2).xyz;
    vec3  boomC = tdeHeader(3).xyz; float boomD2 = 1e30; vec3 boomP = vec3(0.0);
    float tdePrevD = dot(prev, tdeN);
    int   tdeCross = 0;
    float glareD2P = 1e30;
    float glareD2D = 1e30;
    float glareD2V = 1e30;
    float pGlareN = 1e30; int pGlareK = -1;
    float pAtmN = 1e30; int pAtmK = -1;
    float pRingT = 1e5; int pRingK = -1; vec3 pRingX = vec3(0.0); float pRingInc = 1.0;
    vec3  pGlareX = vec3(0.0);
    float pJetI = 0.0; int pJetK = -1;
    float pIonI = 0.0; int pIonK = -1;
    float pDustI = 0.0; int pDustK = -1;
    vec3 pNearPos = ro;
    vec3 pNearDir = rd;
    float pNearArc = 0.0;
    vec3  pHaloPos = ro;
    float pHaloArc = 0.0;
    int   pNearJ   = 1;
    float pNearTs  = 0.0;

    bool noTail = geoCaptured(0.0);
    for (int i = 1; i <= int(GEO_N_S); i++) {
        bool  tail = (i == int(GEO_N_S));
        if (tail && noTail) break;
        vec3  cur = tail ? prev + geoExitDir(row, er, et) * GEO_TAIL_LEN : geoPoint(row, i, er, et);
        float seg = length(cur - prev);

        if (diskOwner && arc < minT && !tail) {
            float dCurD = dot(cur, dn);
            float dCurM = dot(cur, mn);
            vec3  sdir = (cur - prev) / max(seg, 1e-6);
            if (KN_DISK == 1) {
                vec3  qC = vec3(dot(cur, knE1), dot(cur, dn), dot(cur, knE2)) / BH_RS;
                float rC = knKSRadius(dot(qC.xz, qC.xz), qC.y * qC.y);
                knShell += knShellStep(knPrevQ, qC, knPrevR, rC);
                if (gKNOn) knDiskChord(knPrevQ, qC, arc / BH_RS, knLam, knShell, knPhD, knAcc, knPhase);
                knPrevQ = qC; knPrevR = rC;
            }
            bool dSegGas = false;
            if (DISK_SHOW == 1 && gDiskEdge > DISK_R_IN_VIS) {
                float rA_ = length(prev) / BH_RS, rC_ = length(cur) / BH_RS;
                float hB = min(DISK_H_CAP * max(diskHR0(rA_), diskHR0(rC_)), DISK_HR_MAX) * DISK_H_SCALE;
                float zB = DISK_ZMAX * hB * max(rA_, rC_) * BH_RS;
                float dz = dCurD - dPrevD;
                float t0 = 0.0, t1 = 1.0;
                if (abs(dz) > 1e-9) {
                    float ta = (-zB - dPrevD) / dz, tb = (zB - dPrevD) / dz;
                    t0 = max(0.0, min(ta, tb)); t1 = min(1.0, max(ta, tb));
                } else if (abs(dPrevD) >= zB) {
                    t1 = -1.0;
                }
                {
                    vec3  dv = cur - prev;
                    float Ro = 1.02 * sqrt(gDiskEdge * gDiskEdge + 0.25 * KERR_A * KERR_A) * BH_RS;
                    float A_ = dot(dv, dv), B_ = dot(prev, dv), C_ = dot(prev, prev) - Ro * Ro;
                    float disc = B_ * B_ - A_ * C_;
                    if (disc <= 0.0 || A_ < 1e-12) { if (C_ > 0.0) t1 = -1.0; }
                    else {
                        float sq = sqrt(disc);
                        t0 = max(t0, (-B_ - sq) / A_); t1 = min(t1, (-B_ + sq) / A_);
                    }
                }
                if (t1 > t0) {
                    float aW  = KERR_A * 0.5;
                    vec3  e1  = normalize(cross(dn, vec3(0.0, 0.0, 1.0)));
                    vec3  e2  = cross(dn, e1);
                    float Ltot = seg * (t1 - t0) / BH_RS;
                    float rMid = length(mix(prev, cur, 0.5 * (t0 + t1))) / BH_RS;
                    int   nP   = int(clamp(ceil(Ltot / max(0.35 * rMid, 1e-3)), 1.0, float(DISK_VOL_N)));
                    bool  under = (DISK_UNDER_FLIP == 1 && dCurD > dPrevD);
                    for (int k = 0; k < DISK_VOL_N; k++) {
                        if (k >= nP) break;
                        float ta = t0 + (t1 - t0) * float(k)     / float(nP);
                        float tb = t0 + (t1 - t0) * float(k + 1) / float(nP);
                        vec3  X  = mix(prev, cur, 0.5 * (ta + tb));
                        float zW = dot(X, dn);
                        float rW = length(X - zW * dn) / BH_RS;
                        float rB = sqrt(max(rW * rW - aW * aW, 0.0));
                        if (rB < max(DSIM_RIN, DISK_R_IN_VIS) || rB > min(DSIM_ROUT, gDiskEdge)) continue;
                        float phB = atan(dot(X, e2), dot(X, e1));
                        float phUse = under ? -phB : phB;
                        float wrapM = flareWrap(phB + FLARE_FRONT_RAG * knPerlin(vec3(1.3 * rB, 0.5 * zW / BH_RS, 2.0 * gFlareDt)),
                                                atan(dot(bDwarf.xyz, e2), dot(bDwarf.xyz, e1)));
                        if (wrapM <= 0.0) continue;
                        float H   = diskLiveCH(rB, phUse).y;
                        float za  = (dPrevD + dz * ta) / BH_RS, zb = (dPrevD + dz * tb) / BH_RS;
                        float col = gaussColumn(za, zb, Ltot / float(nP), H) * wrapM;
                        if (col < 1e-5 * H) continue;
                        vec4  sh  = diskVolShade(rB, phUse, 0.5 * (za + zb), col, kerrW(0), bass);
                        emitD  += sh.rgb * DISK_GLOW * diskTr * dPassTr;
                        dPassTr *= sh.a;
                        diskTr *= mix(1.0, sh.a, DISK_COVER);
                        dSegGas = true;
                    }
                }
            }
            if (occLimit < 1e4 && dHadGas && !dSegGas) diskTr = 0.0;
            dHadGas = dHadGas || dSegGas;
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
        if (d2p < glareD2P && !tail && legacy) {
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
        if (legacy) glareD2D = min(glareD2D, geoSegPointD2(prev, cur, bDwarf.xyz));
        if (bRC.w > 0.0) glareD2V = min(glareD2V, geoSegPointD2(prev, cur, bRC.xyz));
        if (tdeH.w > 0.5 && arc < minT && !tail) {
            vec3 bab = cur - prev; float bL2 = max(dot(bab, bab), 1e-12);
            float bt = clamp(dot(boomC - prev, bab) / bL2, 0.0, 1.0);
            vec3 bq = prev + bab * bt; float bd2 = dot(bq - boomC, bq - boomC);
            if (bd2 < boomD2) { boomD2 = bd2; boomP = bq; }
        }
        if (tdeH.w > 0.5 && tdeCross < 2 && arc < minT && !tail) {
            float dC = dot(cur, tdeN);
            if (tdePrevD * dC < 0.0) {
                vec3 Pc = mix(prev, cur, tdePrevD / (tdePrevD - dC));
                gTDEEmit += tdeEmitAt(Pc);
                tdeCross++;
            }
            tdePrevD = dC;
        }

        float bestF  = 2.0;
        float bestId = 0.0;
        vec4  bestB  = vec4(0.0);
        vec3  bestP  = vec3(0.0);
        float th; vec3 hp;

        if (geoSegSphere(prev, cur, bPulsar.xyz, bPulsar.w, th, hp)) {
            if (th < bestF) { bestF = th; bestId = 1.0; bestB = bPulsar; bestP = hp; } }
        if (geoSegDrop(prev, cur, bDwarf.xyz, bDwarf.w, dropK(bDwarf.xyz), th, hp)) {
            if (th < bestF) { bestF = th; bestId = 2.0; bestB = bDwarf; bestP = hp; } }
        if (geoSegSphere(prev, cur, bPG.xyz, bPG.w, th, hp)) {
            if (th < bestF) { bestF = th; bestId = 3.0; bestB = bPG; bestP = hp; } }
        if (geoSegSphere(prev, cur, bDA.xyz, bDA.w, th, hp)) {
            if (th < bestF) { bestF = th; bestId = 4.0; bestB = bDA; bestP = hp; } }
        if (geoSegSphere(prev, cur, bG3.xyz, bG3.w, th, hp)) {
            if (th < bestF) { bestF = th; bestId = 5.0; bestB = bG3; bestP = hp; } }
        if (ROSTER_ON == 1) {
            if (bRA.w > 0.0 && geoSegSphere(prev, cur, bRA.xyz, bRA.w, th, hp)) {
                if (th < bestF) { bestF = th; bestId = 6.0; bestB = bRA; bestP = hp; } }
            if (bRB.w > 0.0 && geoSegDrop(prev, cur, bRB.xyz, bRB.w, dropKB(bRB.xyz), th, hp)) {
                if (th < bestF) { bestF = th; bestId = 7.0; bestB = bRB; bestP = hp; } }
        }
        if (planOn()) for (int k = 0; k < PLAN_SLOTS; k++) {
            vec4 sb = texelFetch(iChannel0, ivec2(k, SLOT_ROW), 0);
            if (sb.w <= 0.0) break;
            vec4 si = texelFetch(iChannel0, ivec2(k, SLOT_ROW2), 0);
            if (si.y < 0.5) {
                float dn = geoSegPointD2(prev, cur, sb.xyz) / (sb.w * sb.w);
                if (dn < pGlareN) {
                    pGlareN = dn; pGlareK = k;
                    vec3 dsg = cur - prev;
                    pGlareX = prev + dsg * clamp(dot(sb.xyz - prev, dsg) / max(dot(dsg, dsg), 1e-12), 0.0, 1.0);
                }
                if (si.z > 0.0) {
                    vec4 ja = texelFetch(iChannel0, ivec2(k, SLOT_ROW3), 0);
                    if (ja.w > 0.0 && geoSegPointD2(prev, cur, sb.xyz) < (ja.w + 0.2) * (ja.w + 0.2)) {
                        for (int sd = -1; sd <= 1; sd += 2) {
                            float sj, dj = segSegD(prev, cur, sb.xyz, sb.xyz + float(sd) * ja.xyz * ja.w, sj);
                            float wj = 0.004 + 0.035 * sj * ja.w;
                            float I  = exp(-dj / wj) * pow(1.0 - sj, 1.2);
                            if (I > pJetI) { pJetI = I; pJetK = k; }
                        }
                    }
                }
            } else if (si.y > 1.5) {
                vec4 td = texelFetch(iChannel0, ivec2(k, SLOT_ROW3), 0);
                float L = td.w;
                if (L > 0.0 && geoSegPointD2(prev, cur, sb.xyz) < (L + 0.2) * (L + 0.2)) {
                    float st, dt = segSegD(prev, cur, sb.xyz, sb.xyz + normalize(sb.xyz) * L, st);
                    float I = exp(-dt / (sb.w * 1.5 + 0.05 * st * L)) * pow(1.0 - st, 1.5);
                    if (I > pIonI) { pIonI = I; pIonK = k; }
                    float sdu, ddu = segSegD(prev, cur, sb.xyz, sb.xyz + td.xyz * L * 0.6, sdu);
                    float Id = exp(-ddu / (sb.w * 2.0 + 0.06 * sdu * L)) * pow(1.0 - sdu, 1.6);
                    if (Id > pDustI) { pDustI = Id; pDustK = k; }
                }
            } else if (si.y < 1.5) {
                if (si.z > 0.0) {
                    float dn = geoSegPointD2(prev, cur, sb.xyz) / (sb.w * sb.w);
                    if (dn < pAtmN) { pAtmN = dn; pAtmK = k; }
                }
                {
                    float bnd = sb.w * 5.6;
                    if (texelFetch(iChannel0, ivec2(k, MOON_ROW0), 0).w > 0.0 && geoSegPointD2(prev, cur, sb.xyz) < bnd * bnd) {
                        for (int m = 0; m < MOON_MAX; m++) {
                            vec4 mo = texelFetch(iChannel0, ivec2(k, MOON_ROW0 + m), 0);
                            if (mo.w <= 0.0) break;
                            if (geoSegSphere(prev, cur, mo.xyz, mo.w, th, hp)) {
                                if (th < bestF) { bestF = th; bestId = 1000.0 + float(k * MOON_MAX + m); bestB = mo; bestP = hp; } }
                        }
                    }
                }
                vec4 rg = texelFetch(iChannel0, ivec2(k, SLOT_ROW3), 0);
                if (rg.w > 0.0 && geoSegPointD2(prev, cur, sb.xyz) < rg.w * rg.w) {
                    float d0 = dot(prev - sb.xyz, rg.xyz), d1 = dot(cur - sb.xyz, rg.xyz);
                    if (d0 * d1 < 0.0) {
                        float f = d0 / (d0 - d1);
                        float tA = arc + seg * f;
                        if (tA < pRingT) {
                            vec3 X = mix(prev, cur, f);
                            if (planRingDensity(int(si.x + 0.5), length(X - sb.xyz) / sb.w) > 0.0) {
                                pRingT = tA; pRingK = k; pRingX = X;
                                pRingInc = abs(dot(normalize(cur - prev), rg.xyz));
                            }
                        }
                    }
                }
            }
            if (geoSegSphere(prev, cur, sb.xyz, sb.w, th, hp)) {
                if (th < bestF) { bestF = th; bestId = 100.0 + float(k); bestB = sb; bestP = hp; } }
        }
        if (bRC.w > 0.0 && geoSegDrop(prev, cur, bRC.xyz, bRC.w, dropK(bRC.xyz), th, hp)) {
            if (th < bestF) { bestF = th; bestId = 8.0; bestB = bRC; bestP = hp; } }

        if (legacy && geoSegPointD2(prev, cur, bPG.xyz) < PGIANT_BOUND * PGIANT_BOUND) {
            for (int m = M0_PGIANT; m < M0_DGA; m++) {
                vec4 mo = moonAt(kCopy, m);
                if (geoSegSphere(prev, cur, mo.xyz, mo.w, th, hp)) {
                    if (th < bestF) { bestF = th; bestId = 10.0 + float(m); bestB = mo; bestP = hp; }
                }
            }
        }
        if (legacy && geoSegPointD2(prev, cur, bDA.xyz) < DGA_BOUND * DGA_BOUND) {
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
            hitSelf = (bestId < 9.0) ? int(bestId) - 1 : -1;
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
    if (legacy && testRingPlane(row, er, et, bDA.xyz, rn, trRing, rpHit, dnr)) {
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

    if (pRingK >= 0 && pRingT < ringT && pRingT < minT) {
        vec4  sb = texelFetch(iChannel0, ivec2(pRingK, SLOT_ROW), 0);
        vec4  rg = texelFetch(iChannel0, ivec2(pRingK, SLOT_ROW3), 0);
        int   j  = int(texelFetch(iChannel0, ivec2(pRingK, SLOT_ROW2), 0).x + 0.5);
        float dens = planRingDensity(j, length(pRingX - sb.xyz) / sb.w);
        vec3  wp = pRingX;
        vec3  d1 = bPulsar.xyz - wp;  float l1 = length(d1);
        vec3  d2 = bDwarf.xyz  - wp;  float l2 = length(d2);
        float sh1 = occludeBy(wp, d1 / l1, l1, sb.xyz, sb.w);
        float sh2 = occludeBy(wp, d2 / l2, l2, sb.xyz, sb.w);
        vec3  alb = hueRotate(RING_ALBEDO, planRow(j, 6).x * TAU);
        ringShade = alb * (STAR1_COL * (sh1 * 1.05 + 0.10) + STAR2_COL * (sh2 * 0.95)
                         + diskBodyLight(wp, rg.xyz) * RING_DISKLIGHT) * (0.60 + 0.45 * dens);
        ringOpac  = 1.0 - exp(-dens * 2.0 / max(pRingInc, 0.1));
        ringT     = pRingT;
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
            vec3 dn = dropNormal(hitP, hitC, hitR, dropK(hitC));
            col = redDwarfSurface(dn * hitR, hitV, hitR, mid, iTime, copyDwarfTint(kCopy));
        } else if (hitId >= 100.0 && hitId < 1000.0 && planInfo(hitId).y < 0.5) {
            int  j  = int(planInfo(hitId).x + 0.5);
            vec4 r3 = planRow(j, 3);
            vec3 cs = redDwarfSurface(n * hitR, hitV, hitR, mid, iTime, vec3(1.0));
            float l = dot(cs, vec3(0.299, 0.587, 0.114));
            vec3 kc = kelvinToRgb(r3.x);
            kc = max(mix(vec3(dot(kc, vec3(0.299, 0.587, 0.114))), kc, r3.y), 0.0);
            col = l * kc * 3.2 * (0.7 + 0.3 * r3.z);
        } else if (hitId >= 100.0 && hitId < 1000.0 && planInfo(hitId).y > 1.5) {
            vec4 r3 = planRow(int(planInfo(hitId).x + 0.5), 3);
            col = vec3(1.0, 0.98, 0.95) * 1.6 * r3.z;
        } else if (hitId == 8.0) {
            vec3 dn = dropNormal(hitP, hitC, hitR, dropK(hitC));
            vec3 cs = redDwarfSurface(dn * hitR, hitV, hitR, mid, iTime, VIC_COL);
            col = dot(cs, vec3(0.299, 0.587, 0.114)) * VIC_COL * 2.4;
        } else {
            vec3 albT; float isGiant = 0.0;
            vec3 gu, gv, gn;
            vec3 emis = vec3(0.0);

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
            } else if (hitId == 6.0) {
                albT = lavaSurface(n, iTime, lavaHeat(hitC), emis);
            } else if (hitId == 7.0) {
                n = dropNormal(hitP, hitC, hitR, dropKB(hitC));
                albT = giantAlbedo(n, diskNormal(), iTime, 0.58, 0.8, 0.042, 0.026);
                isGiant = 1.0;
            } else if (hitId >= 1000.0) {
                float mi = hitId - 1000.0;
                albT = rockAlbedo(n, vec3(0.0, 1.0, 0.0), iTime, fract(mi * 0.2718 + 0.13),
                                  mix(vec3(0.86, 0.84, 0.82), vec3(1.02, 0.94, 0.80), fract(mi * 0.35)));
            } else if (hitId >= 100.0) {
                int  j  = int(planInfo(hitId).x + 0.5);
                vec4 r1 = planRow(j, 1), r2 = planRow(j, 2), r3 = planRow(j, 3), r4 = planRow(j, 4);
                vec3 ax = normalize(cross(r1.xyz, r2.xyz));
                if (r3.x < 0.5) {
                    albT = giantAlbedo(n, ax, iTime, r3.y, r3.z, 0.042, 0.026) * r4.rgb;
                    isGiant = 1.0;
                } else if (r3.x < 1.5) {
                    albT = rockAlbedo(n, ax, iTime, fract(float(j) * 0.37 + 0.13), r4.rgb);
                } else {
                    albT = lavaSurface(n, iTime, lavaHeat(hitC), emis) * r4.rgb;
                }
                vec4 r7 = planRow(j, 7);
                if (r7.w > 0.0) {
                    float lat = abs(dot(n, ax));
                    float band = smoothstep(0.70, 0.80, lat) * smoothstep(0.95, 0.86, lat);
                    float lon = atan(dot(n, normalize(r1.xyz)), dot(n, normalize(r2.xyz)));
                    emis += r7.rgb * r7.w * band * (0.55 + 0.45 * sin(lon * 7.0 + iTime * 1.7));
                }
                if (r3.w > 0.0)
                    emis += r7.rgb * r3.w * 1.2 * pow(1.0 - clamp(dot(n, -hitV), 0.0, 1.0), 3.0);
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

            if (hitId >= 100.0 && planOn())
                col = planLight(n, hitV, hitP, albT, isGiant, bass, (hitId < 1000.0) ? int(hitId - 100.0 + 0.5) : -1);
            else
                col = shadeBinary(n, hitV, hitP, bPulsar.xyz, bDwarf.xyz, albT,
                                  envMoons, iTime, isGiant, bass, sh1, sh2);
            col += emis;
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

    if (legacy) glow += hueRotate(calcPulsarJets(pNearPos, pNearDir, bPulsar.xyz,
                                     (PULSAR_SPIN_HOLE == 1 ? getPulsarAxisAt(bPulsar.xyz) : copyRot(getPulsarAxis(), kCopy)),
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
        vec3 mAxis = (PULSAR_SPIN_HOLE == 1 ? getPulsarAxisAt(bPulsar.xyz) : copyRot(getPulsarAxis(), kCopy));
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

    if (boomD2 < 1e29) gTDEEmit += tdeBoomAtP(boomP);
    if (glareD2V < 1e29 && bRC.w > 0.0) {
        float dRay = sqrt(glareD2V);
        float k = bRC.w / max(dRay, bRC.w * 0.88);
        vec3 vg = VIC_COL / max(max(VIC_COL.r, VIC_COL.g), VIC_COL.b);
        glow += vg * 1.10 * pow(k, 3.5) * (0.35 + mid * 1.1);
#if SURF_HQ
        {
            float hk = GLOWQ_D_R * bRC.w / (dRay + GLOWQ_D_R * bRC.w);
            glow += vg * GLOWQ_D_GAIN * pow(hk, GLOWQ_D_POW) * (0.35 + mid * 1.1);
        }
#endif
    }

    if (pAtmK >= 0) {
        float d = sqrt(pAtmN);
        if (d > 1.0) {
            int  j  = int(texelFetch(iChannel0, ivec2(pAtmK, SLOT_ROW2), 0).x + 0.5);
            glow += planRow(j, 7).rgb * planRow(j, 3).w * 0.8 * exp(-(d - 1.0) / 0.07);
        }
    }
    if (pGlareK >= 0) {
        vec4  sb = texelFetch(iChannel0, ivec2(pGlareK, SLOT_ROW), 0);
        vec4  r3 = planRow(int(texelFetch(iChannel0, ivec2(pGlareK, SLOT_ROW2), 0).x + 0.5), 3);
        float dRay = sqrt(pGlareN) * sb.w;
        vec3  kc = kelvinToRgb(r3.x);
        kc = max(mix(vec3(dot(kc, vec3(0.299, 0.587, 0.114))), kc, r3.y), 0.0);
        kc /= max(max(kc.r, kc.g), max(kc.b, 1e-3));
        float k = sb.w / max(dRay, sb.w * 0.88);
        glow += kc * 1.10 * pow(k, 3.5) * (0.35 + mid * 1.1) * r3.z;
        int  jg = int(texelFetch(iChannel0, ivec2(pGlareK, SLOT_ROW2), 0).x + 0.5);
        vec4 r4 = planRow(jg, 4);
#if SURF_HQ
        float cr = GLOWQ_D_R * max(r4.x, 0.2);
        float hk = cr * sb.w / (dRay + cr * sb.w);
        glow += kc * GLOWQ_D_GAIN * pow(hk, GLOWQ_D_POW) * (0.35 + mid * 1.1) * r3.z;
#endif
        float dn = sqrt(pGlareN);
        if (r4.y > 0.0 && dn > 1.0) {
            vec3 e1 = normalize(planRow(jg, 1).xyz), e2 = normalize(planRow(jg, 2).xyz);
            vec3 q  = pGlareX - sb.xyz;
            float an = atan(dot(q, e2), dot(q, e1));
            float fl = pow(0.5 + 0.5 * sin(an * 5.0 + iTime * 0.6 + float(jg)), 10.0)
                     * (0.6 + 0.4 * sin(an * 11.0 - iTime * 1.3));
            glow += kc * 1.6 * r4.y * fl * exp(-(dn - 1.0) / 0.25);
        }
    }
    if (pJetK >= 0) {
        int  jj = int(texelFetch(iChannel0, ivec2(pJetK, SLOT_ROW2), 0).x + 0.5);
        vec3 kc = kelvinToRgb(planRow(jj, 3).x);
        glow += mix(kc, vec3(0.55, 0.72, 1.0), 0.6) * 1.4 * planRow(jj, 3).w * pJetI * (0.5 + 0.8 * treble);
    }
    if (pIonK >= 0) {
        int  jc = int(texelFetch(iChannel0, ivec2(pIonK, SLOT_ROW2), 0).x + 0.5);
        vec4 c3 = planRow(jc, 3);
        glow += kelvinToRgb(c3.y) * 1.2 * c3.z * pIonI;
    }
    if (pDustK >= 0) {
        int  jc = int(texelFetch(iChannel0, ivec2(pDustK, SLOT_ROW2), 0).x + 0.5);
        vec4 c3 = planRow(jc, 3);
        if (c3.w > 0.4 || planRow(jc, 4).x > 1.5) glow += vec3(1.0, 0.82, 0.55) * 0.5 * c3.z * pDustI;
    }

    if (diskOwner) {
        if (KN_DISK == 1) {
            gKNEmit = knToneMap(knAcc) * KN_GAIN;
            gKNTr   = pow(vec3(1.0 - knAcc.a), vec3(1.0, 1.6, 2.5));
        }
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
#define SKY_DEBUG 0
vec3 knSky(vec3 rdL, float treble, float skyB) {
    vec3 c = vec3(0.0);
    float wsum = 0.0;
    if (SKY_DEBUG == 1) {
        vec4 best = vec4(0.0);
        for (int k = 0; k < 4; k++) {
            vec4 d = texelFetch(u_knsky, ivec3(gPix, k), 0);
            if (d.w > best.w) best = d;
        }
        gFootScale = 1.0;
        float wt = texelFetch(u_knsky, ivec3(gPix, 0), 0).w + texelFetch(u_knsky, ivec3(gPix, 1), 0).w
                 + texelFetch(u_knsky, ivec3(gPix, 2), 0).w + texelFetch(u_knsky, ivec3(gPix, 3), 0).w;
        return (best.w > 0.0) ? min(wt, 1.0) * skyColorDispersed(rdL, camSpin(normalize(best.xyz)) - rdL, treble, skyB) : c;
    }
    for (int k = 0; k < 4; k++) {
        vec4 d = texelFetch(u_knsky, ivec3(gPix, k), 0);
        gFootScale = (SKY_CLUSTER_FOOT == 1) ? sqrt(clamp(d.w, 0.0, 1.0)) : 1.0;
        if (d.w > 0.0) {
            vec3 sc = skyColorDispersed(rdL, camSpin(normalize(d.xyz)) - rdL, treble, skyB);
            if (SKY_PROCEDURAL == 1) { c = max(c, sc); wsum += d.w; }
            else c += d.w * sc;
        }
    }
    if (SKY_PROCEDURAL == 1) c *= min(wsum, 1.0);
    gFootScale = 1.0;
    return c;
}

uniform sampler2DArray u_slurp_dwarf;
#define SLURP_ORIGIN   vec2(494.0, 436.0)
#define SLURP_SIZE     1536.0
#define SLURP_FRAMES   85
#define SLURP_BEAT0    (-2.5)
#define SLURP_BSTEP    (0.125)
#define SLURP_DWARF_GAIN   1.0
#define SLURP_DWARF_COVER  0.4
const float SLURP_DWARF_PERI[6] = float[6](25.30, 41.36, 57.41, 73.47, 89.52, 105.58);

vec3 slurpComposite(vec3 col, sampler2DArray flip, const float peri[6], float gain, float cover) {
    vec2 uv = (vec2(gPix) - SLURP_ORIGIN + 1.0) / SLURP_SIZE;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return col;
    float cm = tempoBeatCount() - vicPeri(tempoBeatCount());
    #define SLURP_FORCE_FRAME -1
    if (SLURP_FORCE_FRAME >= 0) cm = peri[0] + SLURP_BEAT0 + float(SLURP_FORCE_FRAME) * SLURP_BSTEP;
    #define SLURP_PREVIEW_FPS 0.0
    if (SLURP_PREVIEW_FPS > 0.0) cm = peri[0] + SLURP_BEAT0 + mod(iTime * SLURP_PREVIEW_FPS, float(SLURP_FRAMES - 1)) * SLURP_BSTEP;
    for (int k = 0; k < 1; k++) {
        float f = (cm - SLURP_BEAT0) / SLURP_BSTEP;
        if (f < 0.0 || f > float(SLURP_FRAMES - 1)) continue;
        float f0 = floor(f);
        vec4 a = texture(flip, vec3(uv, f0));
        vec4 b = texture(flip, vec3(uv, min(f0 + 1.0, float(SLURP_FRAMES - 1))));
        vec4 s = mix(a, b, f - f0);
        return col * (1.0 - cover * clamp(s.a, 0.0, 1.0)) + s.rgb * gain;
    }
    return col;
}

#define STAR_LUM      0.35
#define STAR_CORE     0.03
#define HOLE_LIGHT    0.0
#define HOLE_FALL     6.0
#define DISK_LIGHT_K  1.0
#define DISK_LIGHT_COL vec3(0.75, 0.86, 1.0)
float diskEventEnv() {
    if (!planOn()) return 0.0;
    float cm = tempoBeatCount() - vicPeri(tempoBeatCount());
    return smoothstep(2.5, 3.75, cm) * (1.0 - smoothstep(6.5, 8.0, cm));
}
#define HOLE_LIGHT_COL vec3(1.00, 0.86, 0.68)
#define PLAN_AMBIENT  vec3(0.0)
vec3 planLight(vec3 n, vec3 rd, vec3 wp, vec3 albT, float isGas, float bass, int selfK) {
    vec3  Lc  = normalize(-wp);
    float ndc = clamp((dot(n, Lc) + 0.15) / 1.15, 0.0, 1.0);
    vec3  lightSum = HOLE_LIGHT_COL * HOLE_LIGHT / (1.0 + HOLE_FALL * dot(wp, wp));
    float dEnv = diskEventEnv();
    vec3  col = albT * lightSum * ndc + albT * PLAN_AMBIENT;
    if (dEnv > 0.0)
        col += albT * DISK_LIGHT_COL * dot(diskBodyLight(wp, n), vec3(0.299, 0.587, 0.114)) * DISK_LIGHT_K * dEnv;
    vec3  rimC = lightSum * 0.4;
    for (int k = 0; k < PLAN_SLOTS; k++) {
        vec4 sb = texelFetch(iChannel0, ivec2(k, SLOT_ROW), 0);
        if (sb.w <= 0.0) break;
        vec4 si = texelFetch(iChannel0, ivec2(k, SLOT_ROW2), 0);
        if (si.y > 0.5) continue;
        vec4 r3 = planRow(int(si.x + 0.5), 3);
        if (planRow(int(si.x + 0.5), 4).z > 0.5) continue;
        vec3  kc = kelvinToRgb(r3.x);
        kc = max(mix(vec3(dot(kc, vec3(0.299, 0.587, 0.114))), kc, r3.y), 0.0);
        vec3  dL = sb.xyz - wp;
        float d2 = dot(dL, dL), dl = sqrt(d2);
        vec3  L  = dL / dl;
        float att = STAR_LUM * r3.z * (0.4 + sb.w / 0.05) / (STAR_CORE + d2);
        float sh = 1.0;
#ifdef OFFLINE
        for (int m = 0; m < PLAN_SLOTS; m++) {
            vec4 ob = texelFetch(iChannel0, ivec2(m, SLOT_ROW), 0);
            if (ob.w <= 0.0) break;
            if (m == k || m == selfK) continue;
            sh *= occludeBy(wp, L, dl, ob.xyz, ob.w);
        }
#endif
        float ndl = clamp(dot(n, L), 0.0, 1.0);
        col += albT * kc * ndl * att * sh;
        float sp = pow(max(0.0, dot(n, normalize(L - rd))), isGas > 0.5 ? 11.0 : 20.0);
        col += kc * 1.1 * sp * att * sh * (isGas > 0.5 ? 0.22 : 0.8);
        rimC += kc * att * sh;
    }
    float rim = pow(1.0 - clamp(dot(-rd, n), 0.0, 1.0), 3.2);
    col += rimC * rim * 0.25;
    if (isGas > 0.5) {
        float gasRim = pow(1.0 - clamp(dot(-rd, n), 0.0, 1.0), 4.5);
        col += normalize(albT + 1e-4) * rimC * gasRim * 0.6;
    }
    return col;
}

#define BAKE_RES 2410.0
#define STAR_FLARE_ON 0
#define FLARE_GAIN   0.30
#define FLARE_SIDE2  0.35
#define FLARE_N      4
#define FLARE_NEAR0  180.0
#define FLARE_NEAR1  300.0
vec3 starFlares(vec2 frag) {
    if (!planOn() || STAR_FLARE_ON == 0) return vec3(0.0);
    const float kk[4] = float[4](0.35, 0.70, 1.00, 1.50);
    const float gw[4] = float[4](0.55, 0.35, 0.80, 0.30);
    vec2 C = iResolution.xy * 0.5;
    vec3 acc = vec3(0.0);
    for (int k = 0; k < PLAN_SLOTS; k++) {
        vec4 sb = texelFetch(iChannel0, ivec2(k, SLOT_ROW), 0);
        if (sb.w <= 0.0) break;
        vec4 si = texelFetch(iChannel0, ivec2(k, SLOT_ROW2), 0);
        if (si.y > 0.5) continue;
        vec4 im = texelFetch(iChannel0, ivec2(k, FLARE_ROW), 0);
        int  j  = int(si.x + 0.5);
        vec4 r3 = planRow(j, 3);
        vec3 kc = kelvinToRgb(r3.x);
        kc = max(mix(vec3(dot(kc, vec3(0.299, 0.587, 0.114))), kc, r3.y), 0.0);
        float bright = r3.z * (0.4 + sb.w / 0.05);
        for (int side = 0; side < 2; side++) {
            vec2 S = (side == 0) ? im.xy : im.zw;
            if (S.x < 0.0) continue;
            vec2  D = C - S;
            float dpx = length(D) / iResolution.y * 2410.0;
            float prox = 1.0 - smoothstep(FLARE_NEAR0, FLARE_NEAR1, dpx);
            if (prox <= 0.0) continue;
            float s2 = (side == 0) ? 1.0 : FLARE_SIDE2;
            for (int g = 0; g < FLARE_N; g++) {
                vec2  G  = C + D * kk[g];
                float rad = iResolution.y * (0.0025 + 0.002 * float(g));
                float dd = length(frag - G) / rad;
                if (dd > 1.5) continue;
                float disc = smoothstep(1.0, 0.92, dd) * 0.6 + exp(-dd * dd * 8.0) * 0.8;
                vec3  tint = hueRotate(kc, 0.35 * float(g) - 0.5);
                acc += tint * disc * gw[g] * bright * prox * s2 * FLARE_GAIN;
            }
        }
    }
    return acc;
}

#define CAM_VIB_AMP  0.0006
#define CAM_VIB_HZ   11.0
float camVib() {
    float b = macroBand(13);
    return CAM_VIB_AMP * b * b * (vnoise(vec2(iTime * CAM_VIB_HZ, 3.7)) - 0.5) * 2.0;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 bakeOff = floor((iResolution.xy - vec2(BAKE_RES)) * 0.5);
    vec2 fc = fragCoord - bakeOff;
    if (fc.x < 0.0 || fc.y < 0.0 || fc.x >= BAKE_RES || fc.y >= BAKE_RES) { fragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
    gPix = ivec2(fc);
    {
        float a = camAzOff() + camVib();
        gSpinN = diskNormal(); gSpinC = cos(a); gSpinS = sin(a);
    }
    gDiskEdge = flareEdge();
    knLifeFrame();
    vec2 uv = (-vec2(BAKE_RES) + 2.0 * fc) / BAKE_RES;

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

    float camAz   = CAM_AZ0 + camAzOff() + camVib();
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
    gSheetDir = normalize(skyRotate(gHoleDir));
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
    float nebExt = capt ? 0.0 : nebulaExt(gExit);

    if (intersectSphere(roL, rdL, SPHERE_RADIUS).y <= 0.0) {
        vec3 bg = knSky(rdL, treble, skyB) * (1.0 - nebExt);
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
    gKNEmit = vec3(0.0); gKNTr = vec3(1.0); gTDEEmit = vec3(0.0);
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

    vec3 col = knSky(rdL, treble, skyB) * (1.0 - nebExt);
    col += nebAdd;
    col = mix(col, bestSolid.rgb, clamp(bestSolid.a, 0.0, 1.0));
    col *= gDiskTr;
    col *= gKNTr;
    col += glowSum;
    col += gKNEmit;
    col += gTDEEmit;

    col += diskToneSum(diskEmit);

    #define SLURP_ON 1
    if (SLURP_ON == 1) col = slurpComposite(col, u_slurp_dwarf, SLURP_DWARF_PERI, SLURP_DWARF_GAIN, SLURP_DWARF_COVER);

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

    col += starFlares(fc);

    col = col / (1.0 + col * 0.32);
    col = globalGrade(col);

    fragColor = vec4(col, 1.0);
}
