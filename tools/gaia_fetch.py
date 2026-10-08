#!/usr/bin/env python3
"""Fetch Gaia DR3 stars brighter than G_MAX from the ESA TAP server, in sky chunks.

2026-10-03 (scope.md "v12 sky: real catalogue POINT stars"): the catalogue for
the point-star sky. Only 4 columns (ra, dec, phot_g_mean_mag, bp_rp), only
G < 16.4 (103,936,997 stars, counted on the archive 2026-10-03) -- NOT the
~600 GB bulk dump.

Chunks are source_id ranges (source_id = HEALPix level-12 index * 2^35), so
each chunk is a sky patch. One .npy per chunk in OUT/chunks/; re-running skips
chunks already on disk, so an interrupted run just resumes.

    python3 tools/gaia_fetch.py            # fetch (resumable)
    python3 tools/gaia_fetch.py --check    # total rows on disk vs EXPECTED
"""
import csv, io, os, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

G_MAX = 16.4
EXPECTED = 103_936_997            # COUNT(*) WHERE phot_g_mean_mag < 16.4, 2026-10-03
NCHUNK = 4096
WORKERS = 6
HPX12 = 12 * 4 ** 12
SID = 2 ** 35
TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "textures", "starfield", "gaia")
DT = np.dtype([("ra", "<f8"), ("dec", "<f8"), ("g", "<f4"), ("bp_rp", "<f4")])


def chunk_path(i):
    return os.path.join(OUT, "chunks", f"c{i:04d}.npy")


def fetch(i):
    lo = HPX12 * i // NCHUNK * SID
    hi = HPX12 * (i + 1) // NCHUNK * SID
    q = (f"SELECT ra, dec, phot_g_mean_mag, bp_rp FROM gaiadr3.gaia_source "
         f"WHERE source_id >= {lo} AND source_id < {hi} AND phot_g_mean_mag < {G_MAX}")
    body = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                   "FORMAT": "csv", "QUERY": q}).encode()
    for attempt in range(6):
        try:
            with urllib.request.urlopen(TAP, body, timeout=600) as r:
                text = r.read().decode()
            rows = csv.reader(io.StringIO(text))
            head = next(rows)
            if head != ["ra", "dec", "phot_g_mean_mag", "bp_rp"]:
                raise RuntimeError(f"unexpected reply: {text[:200]!r}")
            data = [(float(a), float(b), float(c), float(d) if d else np.nan)
                    for a, b, c, d in rows]
            arr = np.array(data, dtype=DT)
            tmp = chunk_path(i) + ".tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, chunk_path(i))
            return i, len(arr)
        except Exception as e:
            wait = 5 * 2 ** attempt
            print(f"chunk {i}: attempt {attempt + 1} failed ({e}); retry in {wait}s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError(f"chunk {i} failed after retries")


def check():
    n = sum(len(np.load(chunk_path(i), mmap_mode="r"))
            for i in range(NCHUNK) if os.path.exists(chunk_path(i)))
    have = sum(os.path.exists(chunk_path(i)) for i in range(NCHUNK))
    print(f"chunks {have}/{NCHUNK}, rows {n:,} / expected {EXPECTED:,}"
          f" -> {'OK' if have == NCHUNK and n == EXPECTED else 'INCOMPLETE/MISMATCH'}")


def main():
    os.makedirs(os.path.join(OUT, "chunks"), exist_ok=True)
    if "--check" in sys.argv:
        return check()
    todo = [i for i in range(NCHUNK) if not os.path.exists(chunk_path(i))]
    print(f"{len(todo)} chunks to fetch ({NCHUNK - len(todo)} already done)", flush=True)
    t0, done, rows = time.time(), 0, 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for f in as_completed([ex.submit(fetch, i) for i in todo]):
            i, n = f.result()
            done += 1; rows += n
            if done % 64 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"{done}/{len(todo)} chunks, {rows:,} rows, {el:.0f}s, "
                      f"eta {el / done * (len(todo) - done):.0f}s", flush=True)
    check()


if __name__ == "__main__":
    main()
