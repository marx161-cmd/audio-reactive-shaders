"""dwarf_box REBIRTH phase 2: the shot search (sim.py _rb2_search) in its OWN process -- in a thread it fought the
renderer for the GIL (2.3 s alone -> 11.7 s live). Usage: rb2_worker.py <in.npz> <out.pkl> | <in.pkl> <out.pkl> rescue (sim.py rb2_plan starts it)."""
import importlib.util
import os
import pickle
import sys

import numpy as np

try:
    os.nice(10)
except OSError:
    pass

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dwarf_box_sim", os.path.join(here, "sim.py"))
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)
if len(sys.argv) > 3 and sys.argv[3] == "rails":
    res = sim._rails_bake()
    tmp = sys.argv[2] + ".part.npy"
    np.save(tmp, res)
    os.replace(tmp, sys.argv[2])
    sys.exit(0)
elif len(sys.argv) > 3 and sys.argv[3] == "snipe":
    with open(sys.argv[1], "rb") as fh:
        d = pickle.load(fh)
    sim._LAUNCH = d.get("field")
    res, rows = sim.sniper_search(d["x0"], d["v0"], d["rad"], d["t0"], d["cands"], d["rails"], d["P0V0"], d["beat"],
                                  d["t_ready"], sim.sniper_model_load(), np.random.default_rng())
    sim.sniper_log(rows, "live")
elif len(sys.argv) > 3 and sys.argv[3] == "farm":
    with open(sys.argv[1], "rb") as fh:
        d = pickle.load(fh)
    sim._LAUNCH = d.get("field")
    res = sim._fm_search(d["rails"], d["t_ready"])
elif len(sys.argv) > 3 and sys.argv[3] == "rescue":
    with open(sys.argv[1], "rb") as fh:
        d = pickle.load(fh)
    sim._LAUNCH = d.get("field")
    res = sim._rs_search(d["x0"], d["v0"], d["rad"], d["t0"], d["T1s"], d["TAUs"], d["rails"], d["t_ready"])
else:
    d = np.load(sys.argv[1])
    res = sim._rb2_plan_all(d["P"], d["V"], d["rad"], float(d["t_hit"]), float(d["t_ready"]))
tmp = sys.argv[2] + ".part"
with open(tmp, "wb") as fh:
    pickle.dump(res, fh)
os.replace(tmp, sys.argv[2])
