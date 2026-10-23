"""dwarf_box THE THINKING SQUID (sim.py ps_worker_loop): the pulsar chase's squid planner in its OWN process -- it searches
each requested plan for as long as the beat lasts and learns which patterns win. Usage: ps_worker.py <req.pkl> <out.pkl> <token>
(sim.py ps_worker_ask starts it; it exits when the token changes or nothing is asked for 30 s)."""
import importlib.util
import os
import sys

try:
    os.nice(10)
except OSError:
    pass

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dwarf_box_sim", os.path.join(here, "sim.py"))
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)
sim.ps_worker_loop(sys.argv[1], sys.argv[2], sys.argv[3])
