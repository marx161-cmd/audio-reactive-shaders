// Armillary-Gem-Sphere: image pass.
//
// A beat-locked gimbal of hexagonal gem rings (ruby, diamond, diamond) with
// pentagonal rods, exact polytope tracing per RGB channel (Fresnel, total
// internal reflection, dispersion), a round-brilliant sapphire core as the only
// light, glowing crystal connectors, volumetric mist, and a lit image backdrop
// (depth + normals) that the escaping rays land on.
// Buffers: A state, B/C mist, D the sapphire's light with ring shadows.

#define METAL_FLOW   0.05
#define PI  3.141592654
#define TAU (2.0*PI)

void rot(inout vec2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  p = vec2(c*p.x + s*p.y, -s*p.x + c*p.y);
}

float hash(in vec2 co) {
  return fract(sin(dot(co.xy ,vec2(12.9898,58.233))) * 13758.5453);
}

float psin(float a) {
  return 0.5 + 0.5*sin(a);
}

float tanh_approx(float x) {
  float x2 = x*x;
  return clamp(x*(27.0 + x2)/(27.0+9.0*x2), -1.0, 1.0);
}

float onoise(vec2 x) {
  x *= 0.5;
  float a = sin(x.x);
  float b = sin(x.y);
  float c = mix(a, b, psin(TAU*tanh_approx(a*b+a+b)));

  return c;
}

float vnoise(vec2 x) {
  vec2 i = floor(x);
  vec2 w = fract(x);

  vec2 u = w*w*w*(w*(w*6.0-15.0)+10.0);

  float a = hash(i+vec2(0.0,0.0));
  float b = hash(i+vec2(1.0,0.0));
  float c = hash(i+vec2(0.0,1.0));
  float d = hash(i+vec2(1.0,1.0));

  float k0 =   a;
  float k1 =   b - a;
  float k2 =   c - a;
  float k3 =   d - c + a - b;

  return k0 + k1*u.x + k2*u.y + k3*u.x*u.y;
}

float fbm1(vec2 p) {
  vec2 op = p;
  const float aa = 0.45;
  const float pp = 2.03;
  const vec2 oo = -vec2(1.23, 1.5);
  const float rr = 1.2;

  float h = 0.0;
  float d = 0.0;
  float a = 1.0;

  for (int i = 0; i < 5; ++i) {
    h += a*onoise(p);
    d += (a);
    a *= aa;
    p += oo;
    p *= pp;
    rot(p, rr);
  }

  return mix((h/d), -0.5*(h/d), pow(vnoise(0.9*op), 0.25));
}

float fbm2(vec2 p) {
  vec2 op = p;
  const float aa = 0.45;
  const float pp = 2.03;
  const vec2 oo = -vec2(1.23, 1.5);
  const float rr = 1.2;

  float h = 0.0;
  float d = 0.0;
  float a = 1.0;

  for (int i = 0; i < 7; ++i) {
    h += a*onoise(p);
    d += (a);
    a *= aa;
    p += oo;
    p *= pp;
    rot(p, rr);
  }

  return mix((h/d), -0.5*(h/d), pow(vnoise(0.9*op), 0.25));
}

float fbm3(vec2 p) {
  vec2 op = p;
  const float aa = 0.45;
  const float pp = 2.03;
  const vec2 oo = -vec2(1.23, 1.5);
  const float rr = 1.2;

  float h = 0.0;
  float d = 0.0;
  float a = 1.0;

  for (int i = 0; i < 3; ++i) {
    h += a*onoise(p);
    d += (a);
    a *= aa;
    p += oo;
    p *= pp;
    rot(p, rr);
  }

  return mix((h/d), -0.5*(h/d), pow(vnoise(0.9*op), 0.25));
}

float warp(vec2 p) {
  vec2 v = vec2(fbm1(p), fbm1(p+0.7*vec2(1.0, 1.0)));

  rot(v, 1.0 + iTime*0.1);

  float amp = 1.0;

  vec2 vv = vec2(fbm2(p + 3.7*amp*v), fbm2(p + -2.7*amp*v.yx+0.7*vec2(1.0, 1.0)));

  rot(vv, -1.0 + iTime*0.21315);

  return fbm3(p + 1.4*amp*vv);
}

float height(vec2 p) {
  p.y += METAL_FLOW*iTime;
  p *= 2.0;
  p += 13.0;
  float h = warp(p);
  float rs = 3.0;
  return 0.35*tanh_approx(rs*h)/rs;
}

vec3 postProcess(vec3 col, vec2 q)  {
  col=clamp(col,0.0,1.0);
  col=col*0.6+0.4*col*col*(3.0-2.0*col);
  col=mix(col, vec3(dot(col, vec3(0.33))), -0.4);
  col*=0.5+0.5*pow(19.0*q.x*q.y*(1.0-q.x)*(1.0-q.y),0.7);
  return col;
}

#define METAL_LIFT   0.35
#define METAL_FALL   0.35
#define METAL_RING   0.10
#define METAL_BOLT   0.6
#define METAL_BOLT_CAP 0.8
#define RING_PTS     12
#define HALO_I       0.30
#define HALO_W       0.035
#define HALO_TAIL    0.06
#define HALO_TAIL_R  0.25
const vec3 METAL_BASE = vec3(0.018, 0.022, 0.035);
#define METAL_SPEC     0.6
#define METAL_SPEC_POW 20.0
#define METAL_DIFF     0.0
#define EDGE_LO        0.20
#define EDGE_HI        0.40
vec2 toScreenM(vec3 X) { return X.xy / ((CAM_Z - X.z) * FOV); }
vec3 metalLight(vec3 pp, vec3 n, vec2 at, vec3 c) {
  vec3 ld = normalize(vec3(at.x, -METAL_LIFT, at.y) - pp);
  float diff = max(dot(ld, n), 0.0);
  float d = length(at - pp.xz);
  float edge = smoothstep(EDGE_LO, EDGE_HI, length(n.xz));
  return c * (METAL_SPEC * pow(diff, METAL_SPEC_POW) * edge + METAL_DIFF * pow(diff, 1.5)) / (1.0 + pow(d / METAL_FALL, 2.0));
}
vec3 liquidMetal(vec2 fragCoord) {
  vec2 q = fragCoord/iResolution.xy;
  vec2 p = -1. + 2. * q;
  p.x*=iResolution.x/iResolution.y;

  float h = height(p);
  vec3 pp = vec3(p.x, h, p.y);
  vec3 n = normalize(vec3(-2.0*dFdx(h), -4.0/iResolution.y, -2.0*dFdy(h)));

  setupGimbal();
  vec3 col = vec3(0.0);
  if (LIGHTNING == 0) {
    vec3 sc = SAPH_COL * saphPulse();
    col += metalLight(pp, n, toScreenM(vec3(0.0)), sc * 3.5);
    col = postProcess(col, q);
    float dc = length(p - toScreenM(vec3(0.0)));
    return col + vec3(0.05, 0.06, 0.09) * (0.6 + 0.8 * smoothstep(-0.1, 0.1, h))
         + sc * (0.22 * exp(-pow(dc / 0.15, 2.0)) + 0.12 / (1.0 + pow(dc / 0.5, 2.0)));
  }
  for (int k = 0; k < N_RINGS; k++) {
    vec3 c = RING_RGB[k] * METAL_RING * (0.5 + 0.5 * u_mid);
    for (int i = 0; i < RING_PTS; i++) {
      float a = RING_VOFF[k] + 6.28318531 * float(i) / float(RING_PTS);
      col += metalLight(pp, n, toScreenM(gM[k] * hexPt(k, a, 0.0)), c);
    }
  }
  vec3 halo = vec3(0.0);
  for (int k = 0; k < N_RINGS; k++) {
    float dmin = 1e9;
    vec2 prev = toScreenM(gM[k] * hexPt(k, RING_VOFF[k], 0.0));
    for (int i = 1; i <= RING_PTS; i++) {
      float a = RING_VOFF[k] + 6.28318531 * float(i) / float(RING_PTS);
      vec2 cur = toScreenM(gM[k] * hexPt(k, a, 0.0));
      vec2 ab = cur - prev;
      vec2 at = prev + ab * clamp(dot(p - prev, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
      dmin = min(dmin, length(p - at));
      prev = cur;
    }
    halo += RING_RGB[k] * (0.5 + 0.5 * u_mid)
          * (HALO_I * exp(-pow(dmin / HALO_W, 2.0)) + HALO_TAIL / (1.0 + pow(dmin / HALO_TAIL_R, 2.0)));
  }

  vec3 bA[10], bB[10]; float bFl[10]; int bK[10], bTile[10];
  buildBolts(bA, bB, bFl, bK, bTile);
  vec3 boltL = vec3(0.0);
  for (int j = 0; j < 10; j++) {
    if (bFl[j] <= 0.0) continue;
    vec2 sa = toScreenM(bA[j]), sb = toScreenM(bB[j]), ab = sb - sa;
    vec2 at = sa + ab * clamp(dot(p - sa, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
    boltL += metalLight(pp, n, at, RING_RGB[bK[j]] * bFl[j] * METAL_BOLT);
  }
  float bm = max(max(boltL.r, boltL.g), boltL.b);
  col += boltL * (METAL_BOLT_CAP / (METAL_BOLT_CAP + bm));

  col = postProcess(col, q);
  return col + METAL_BASE * (0.8 + 0.4 * smoothstep(-0.1, 0.1, h)) + halo;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec4 direct = texture(iChannel0, fragCoord / iResolution.xy);
    if (LIGHTNING == 0) {
        vec2 st = fragCoord / iResolution.xy;
        setupGimbal(); buildConnectors();
        vec3 mist = texture(iChannel2, st).rgb + backdropAt(st);
        fragColor = vec4(keepHue(max(direct.rgb + (1.0 - direct.a) * mist, 0.0)), 1.0);
        return;
    }
    vec3 metal = liquidMetal(fragCoord);
    vec3 col = direct.rgb + (1.0 - direct.a) * metal;
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
