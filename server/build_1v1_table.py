"""Build the monolithic 1v1 table (Option A, gamma=1.0 truncated horizon).

Model
-----
No match cap in the real game -> infinite game, but we solve it as a
TRUNCATED horizon with gamma=1.0: iterating the depth-2 operator T_2 gives the
exact values of the (2k)-ply game; as k grows the values converge to the
infinite-game value (games resolve: bank-burst beats the turtle). Wins are
worth exactly +1 (maximally fight-promoting), no gamma tuning needed.

Phase 4 (turns >= 7, stationary budget): backward induction over the
belief-state grid (hits, mover, own_bank, own_sh, R) until max-abs-delta < tol
or max-iters. Ramp turns 1..6: one exact backward-induction pass per turn
(leaves = the already-computed turn+2 table, or the phase-4 table for turns 6,5).

Parallelism: multiprocessing (NOT threads — shared-heap allocator contention
kills thread scaling). Each worker solves a contiguous slice of the grid via
the single-threaded Rust call `solve_1v1_step`.

Checkpoints: binary grids + meta.json in --ckpt-dir, saved after every phase-4
iteration and every ramp turn. Resuming continues from the last checkpoint.

Output: belief-key CSV (hA,hB,to_move,own_bank,own_sh,R,turn,value); turn>=7 is
clamped to 7 by the engine (phase-4 values are turn-invariant).

Usage (server, inside tmux):
  python server/build_1v1_table.py --workers 56 --out cote_cfr/1v1_table.csv
Resume after a crash: same command (reads --ckpt-dir/meta.json).
"""
import argparse
import json
import os
import struct
import sys
import time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import _cote_cfr as c  # noqa: E402


def grid_size(L):
    nh = L["hits_max"] - L["hits_min"] + 1
    return nh * nh * 2 * (L["bank_max"] + 1) * (L["sh_max"] + 1) * (L["r_max"] + 1)


def gidx(hA, hB, mv, bank, sh, r, L):
    nh = L["hits_max"] - L["hits_min"] + 1
    return (((((hA - L["hits_min"]) * nh + (hB - L["hits_min"])) * 2 + mv)
             * (L["bank_max"] + 1) + bank) * (L["sh_max"] + 1) + sh) * (L["r_max"] + 1) + r


def pack(arr):
    return struct.pack("<%dd" % len(arr), *arr)


def unpack(data):
    return list(struct.unpack("<%dd" % (len(data) // 8), data))


def ckpt_file(d, name):
    return os.path.join(d, name + ".bin")


def save_ckpt(d, name, arr):
    os.makedirs(d, exist_ok=True)
    tmp = ckpt_file(d, name) + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(pack(arr))
    os.replace(tmp, ckpt_file(d, name))


def load_ckpt(d, name):
    with open(ckpt_file(d, name), "rb") as fh:
        return unpack(fh.read())


def write_meta(d, meta):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "meta.json")
    cur = {}
    if os.path.exists(p):
        try:
            with open(p) as fh:
                cur = json.load(fh)
        except Exception:
            cur = {}
    cur.update(meta)
    with open(p, "w") as fh:
        json.dump(cur, fh, indent=1)


def read_meta(d):
    p = os.path.join(d, "meta.json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


def _solve_slice(args):
    """Worker: solve one contiguous slice [start, end) of the grid."""
    leaf, start, end, root_turn, L, gamma, solve_iters, solve_depth = args
    out, delta = c.solve_1v1_step(
        leaf, start=start, end=end, root_turn=root_turn,
        hits_min=L["hits_min"], hits_max=L["hits_max"],
        bank_max=L["bank_max"], sh_max=L["sh_max"], r_max=L["r_max"],
        gamma=gamma, solve_iters=solve_iters, solve_depth=solve_depth)
    return start, out, delta


def solve_full(pool, nworkers, leaf, root_turn, L, gamma, solve_iters, solve_depth=2):
    """One backward-induction step over the whole grid via the worker pool."""
    n = grid_size(L)
    nworkers = min(nworkers, n)
    edges = [i * n // nworkers for i in range(nworkers + 1)]
    tasks = [(leaf, edges[i], edges[i + 1], root_turn, L, gamma, solve_iters, solve_depth)
             for i in range(nworkers)]
    out = [0.0] * n
    max_delta = 0.0
    for start, vals, delta in pool.map(_solve_slice, tasks):
        out[start:start + len(vals)] = vals
        max_delta = max(max_delta, delta)
    return out, max_delta


def repr_values(V, L):
    hm = L["hits_max"]

    def v(hA, hB, mv, bk, sh, r):
        hA, hB = min(hA, hm), min(hB, hm)
        return V[gidx(hA, hB, mv, bk, sh, r, L)]

    return "fresh(%d,%d)=%+.3f fresh(%d,%d)=%+.3f burst(%d,%d,b4,R4)=%+.3f" % (
        hm, hm, v(hm, hm, 0, 0, 0, 0),
        max(1, hm - 2), max(1, hm - 2), v(max(1, hm - 2), max(1, hm - 2), 0, 0, 0, 0),
        hm, hm, v(hm, hm, 0, 4, 0, 4))


def run_phase4(pool, L, args, meta):
    if meta and meta.get("phase4_done") and meta.get("iters", 0) >= args.max_iters:
        return load_ckpt(args.ckpt_dir, "phase4")
    if (meta and os.path.exists(ckpt_file(args.ckpt_dir, "phase4"))
            and (not meta.get("phase4_done") or meta.get("iters", 0) < args.max_iters)):
        V = load_ckpt(args.ckpt_dir, "phase4")
        start = meta.get("iters", 0)
        print("[resume] phase-4 from iteration %d (max_iters=%d)"
              % (start, args.max_iters), flush=True)
    else:
        V = [0.0] * grid_size(L)
        start = 0
    t_all = time.time()
    per_iter = None
    it = start
    for it in range(start + 1, args.max_iters + 1):
        t0 = time.time()
        V, delta = solve_full(pool, args.workers, V, root_turn=7, L=L,
                              gamma=args.gamma, solve_iters=args.solve_iters)
        dt = time.time() - t0
        save_ckpt(args.ckpt_dir, "phase4", V)
        write_meta(args.ckpt_dir, {"phase": "phase4", "iter": it, "layout": L})
        if per_iter is None:
            per_iter = dt
        eta = per_iter * (args.max_iters - it) / 60.0
        print("[phase4 iter %2d/%d] delta=%.5f horizon=%2d plies  %s  "
              "iter=%5.1fs eta~%5.1fmin"
              % (it, args.max_iters, delta, 2 * it, repr_values(V, L), dt, eta), flush=True)
        if delta < args.tol:
            print("[phase4 converged at iter %d (delta<%.4f)]" % (it, args.tol), flush=True)
            break
    print("[phase4 done: %d iterations, %5.1fs total]" % (it, time.time() - t_all), flush=True)
    write_meta(args.ckpt_dir, {"phase": "phase4", "phase4_done": True, "iters": it,
                               "layout": L})
    return V


def run_ramp(pool, L, args, phase4):
    ramp = {}
    for k in (6, 5, 4, 3, 2, 1):
        if os.path.exists(ckpt_file(args.ckpt_dir, "ramp%d" % k)):
            ramp[k] = load_ckpt(args.ckpt_dir, "ramp%d" % k)
            print("[resume] ramp turn %d loaded" % k, flush=True)
            continue
        leaf = ramp.get(k + 2, phase4)  # noqa: F841 - kept for resume compatibility
        t0 = time.time()
        # FIXED (was: one pass, depth=2, leaf=turn+2 cascade). The cascade is
        # wrong: solve_1v1_step depth=2 with a turn+2 leaf gives values far from
        # the deep resolve for turns 1..6 (verified: turn1/2/4 were 0.40/0.33/
        # 0.18 vs gold 0.23/0.09/-0.12). Correct: each ramp turn k is solved to
        # depth (7 - k) so its leaves land in the converged phase-4 table, i.e.
        # it IS the deep resolve the bot would do at runtime.
        depth = max(1, 7 - k)
        V, _ = solve_full(pool, args.workers, phase4, root_turn=k, L=L,
                          gamma=args.gamma, solve_iters=args.solve_iters,
                          solve_depth=depth)
        ramp[k] = V
        save_ckpt(args.ckpt_dir, "ramp%d" % k, V)
        write_meta(args.ckpt_dir, {"phase": "ramp", "turn": k, "layout": L})
        print("[ramp turn %d] depth=%d done in %.1fs" % (k, depth, time.time() - t0), flush=True)
    return ramp


def dump_csv(args, L, phase4, ramp):
    t0 = time.time()
    with open(args.out, "w") as fh:
        fh.write("hA,hB,to_move,own_bank,own_sh,R,turn,value\n")
        for turn in (7, 6, 5, 4, 3, 2, 1):
            grid = phase4 if turn == 7 else ramp[turn]
            buf = []
            for hA in range(L["hits_min"], L["hits_max"] + 1):
                for hB in range(L["hits_min"], L["hits_max"] + 1):
                    for mv in (0, 1):
                        for bk in range(L["bank_max"] + 1):
                            for sh in range(L["sh_max"] + 1):
                                for r in range(L["r_max"] + 1):
                                    val = grid[gidx(hA, hB, mv, bk, sh, r, L)]
                                    buf.append("%d,%d,%d,%d,%d,%d,%d,%.6f\n"
                                               % (hA, hB, mv, bk, sh, r, turn, val))
                                    if len(buf) >= 200000:
                                        fh.write("".join(buf))
                                        buf.clear()
            fh.write("".join(buf))
            print("[csv] turn %d written" % turn, flush=True)
    print("[csv] %d rows -> %s in %.1fs"
          % (grid_size(L) * 7, args.out, time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=os.path.join(REPO, "server", "ckpt_1v1"))
    ap.add_argument("--out", default=os.path.join(REPO, "cote_cfr", "1v1_table.csv"))
    ap.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()")
    ap.add_argument("--hits-min", type=int, default=1)
    ap.add_argument("--hits-max", type=int, default=16)
    ap.add_argument("--bank-max", type=int, default=4)
    ap.add_argument("--sh-max", type=int, default=8)
    ap.add_argument("--r-max", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--solve-iters", type=int, default=150)
    ap.add_argument("--max-iters", type=int, default=35)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--no-ramp", action="store_true")
    ap.add_argument("--no-dump", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.workers <= 0:
        args.workers = os.cpu_count() or 1

    L = {"hits_min": args.hits_min, "hits_max": args.hits_max,
         "bank_max": args.bank_max, "sh_max": args.sh_max, "r_max": args.r_max}
    print("[layout] hits %d..%d bank<=%d sh<=%d R<=%d grid=%d workers=%d"
          % (L["hits_min"], L["hits_max"], L["bank_max"], L["sh_max"],
             L["r_max"], grid_size(L), args.workers), flush=True)

    meta = None if args.force else read_meta(args.ckpt_dir)
    if meta is not None and meta.get("layout") != L:
        print("[warn] checkpoint layout %s != requested %s; ignoring checkpoints"
              % (meta.get("layout"), L), flush=True)
        meta = None
    pool = Pool(args.workers)
    try:
        phase4 = run_phase4(pool, L, args, meta)
        if not args.no_ramp:
            ramp = run_ramp(pool, L, args, phase4)
        else:
            ramp = {}
        if not args.no_dump:
            dump_csv(args, L, phase4, ramp)
    finally:
        pool.close()
        pool.join()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
