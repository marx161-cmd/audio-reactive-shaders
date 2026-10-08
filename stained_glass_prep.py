#!/usr/bin/env python3
"""Derive shader masks from a stained-glass PNG.
Usage: stained_glass_prep.py [dir]   (dir has source_highcontrast.png + source_original.png)
Outputs into dir:
  lead_mask.png    255 = lead came
  pane_id.png      RGB-packed pane index (id = R + G*256 + B*65536), 0 = lead/none
  pane_rand.png    grey, per-pane random value 1..255 (phase/flicker seed), 0 = lead
  edge_dist.png    grey, distance from lead inside each pane (normalised per-pane 0..255)
  glass_var.png    grey, high-pass luma of ORIGINAL (glass streak/thickness variation)
  lead_normal.png  tangent normal map of lead relief
  preview.png      lead (red) over original, downscaled for eyeballing
"""
import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi

d = sys.argv[1] if len(sys.argv) > 1 else '.'
hc = np.asarray(Image.open(f'{d}/source_highcontrast.png').convert('RGB')).astype(np.float32)
og = np.asarray(Image.open(f'{d}/source_original.png').convert('RGB')).astype(np.float32)

# --- lead: dark AND (nearly) unsaturated, on the contrast-boosted copy
mx, mn = hc.max(2), hc.min(2)
sat = (mx - mn) / (mx + 1e-6)
LEAD_V, LEAD_S = 24, 0.5
lead = (mx < LEAD_V) & (sat < LEAD_S)
lead = ndi.binary_closing(lead, structure=np.ones((3, 3)), iterations=2)   # bridge breaks in came
# drop lead specks (dark-glass noise) that aren't part of a connected line network
ll, ln_ = ndi.label(lead)
lsz = ndi.sum(lead, ll, range(1, ln_ + 1))
keepl = np.zeros(ln_ + 1, bool); keepl[1:] = lsz >= 400
lead = keepl[ll]
lead = ndi.binary_dilation(lead, structure=np.ones((3, 3)), iterations=1)   # cover dark fringe at came edges

# grow into adjacent dark pixels the saturation test rejected (bluish edge of thick came)
darkish = mx < 55
lead = ndi.binary_dilation(lead, structure=np.ones((3, 3)), iterations=6, mask=darkish | lead)

# --- panes = connected components of non-lead
glass = ~lead
lab, n = ndi.label(glass)
sizes = ndi.sum(glass, lab, index=np.arange(1, n + 1))
keep = np.zeros(n + 1, bool); keep[1:] = sizes >= 150
# renumber kept panes 1..K; tiny leftovers merge back to lead-less "0"
lead = lead | ((lab > 0) & ~keep[lab])   # tiny non-lead specks inside the came = holes: fill them
remap = np.zeros(n + 1, np.int64); remap[keep] = np.arange(1, keep.sum() + 1)
pid = remap[lab]
K = int(keep.sum())
print(f'lead coverage {lead.mean():.3f}  panes kept {K} (dropped {n-K} tiny)')

# --- outputs
Image.fromarray((lead * 255).astype(np.uint8)).save(f'{d}/lead_mask.png')
rgb = np.stack([pid & 255, (pid >> 8) & 255, (pid >> 16) & 255], -1).astype(np.uint8)
Image.fromarray(rgb).save(f'{d}/pane_id.png')
rng = np.random.default_rng(7)
rv = np.concatenate([[0], rng.integers(1, 256, K)]).astype(np.uint8)
Image.fromarray(rv[pid]).save(f'{d}/pane_rand.png')

# per-pane normalised edge distance (0 at lead, 1 at pane centre)
dist = ndi.distance_transform_edt(pid > 0)
pmax = ndi.maximum(dist, pid, index=np.arange(K + 1)); pmax[0] = 1
ed = np.clip(dist / np.maximum(pmax[pid], 1e-6), 0, 1)
Image.fromarray((ed * 255).astype(np.uint8)).save(f'{d}/edge_dist.png')

# glass variation: high-pass luma of the original
L = og.mean(2)
hp = L - ndi.gaussian_filter(L, 12)
hp = hp / (np.percentile(np.abs(hp[pid > 0]), 99) + 1e-6)
Image.fromarray(np.clip(hp * 0.5 + 0.5, 0, 1).__mul__(255).astype(np.uint8)).save(f'{d}/glass_var.png')

# lead relief normal: soft height from lead mask
h = ndi.gaussian_filter(lead.astype(np.float32), 2.5)
gy, gx = np.gradient(h)
s = 6.0
nx, ny, nz = -gx * s, -gy * s, np.ones_like(h)
ln = np.sqrt(nx**2 + ny**2 + nz**2)
nrm = np.stack([nx / ln, ny / ln, nz / ln], -1) * 0.5 + 0.5
Image.fromarray((nrm * 255).astype(np.uint8)).save(f'{d}/lead_normal.png')

# preview
pv = og.copy(); pv[lead] = [255, 40, 40]
Image.fromarray(pv.astype(np.uint8)).resize((1400, 1400), Image.LANCZOS).save(f'{d}/preview.png')

# packed RGBA for the shader: R lead, G pane_rand, B edge_dist, A glass_var (one fetch for all masks)
packed = np.stack([lead * 255, rv[pid], (ed * 255), np.clip(hp * 0.5 + 0.5, 0, 1) * 255], -1).astype(np.uint8)
Image.fromarray(packed, 'RGBA').save(f'{d}/glass_data.png')
