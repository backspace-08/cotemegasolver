"""1v1 equilibrium solver on the FULL game tree with perfect-recall info sets.

Why this exists (imperfect-recall fix)
-------------------------------------
`solver1v1.py` dedups states into a DAG and collapses each player's hidden
(shields, bank) split into their sum inside `info_key`. That collapsing forgets
the player's past observations (revealed shields, revealed banks), which inform
the belief over the *current* hidden split. Merging histories with different
observation sequences into one info set makes the abstract game imperfect-recall,
and CFR does not converge on imperfect-recall games: exploitability oscillates
and never reaches ~0 no matter the iteration count or CFR variant.

This module builds the FULL game tree (no state dedup; every history is its own
node) and, with ``full_history=True``, defines info sets by each player's OWN
complete observation sequence (the sequence of (turn, hA, hB, own bank, R)
values at its decisions). That restores perfect recall, and CFR converges:
measured cap=6 hp=10000: 0.67 -> 0.05 in ~300 iters, -> ~0.005-0.01 in ~500-800.

Abstraction
-----------
Real HP is abstracted to hits-to-kill: `k = ceil(hp / (opp_atk * opp_mult))`.
An exchange with `landed` landed hits decrements the defender's `k`. The only
abstraction error is damage rounding (< 100 HP per exchange, far below one hit).

CFR: forward/backward regret matching (CFR+, alternating), vectorized with
numpy, batched over tree depth levels. `full_history=False` keeps the original
collapsed info sets (for comparison).

Cost: the tree is much larger than the DAG (cap=6 hp=10000: ~5.1M nodes vs 7.8k
DAG states) and the hot passes are memory-bound (thread count does not help).

Utility from player A's perspective: +1 A wins, -1 B wins, 0 draw.
"""
import sys
import time
from collections import deque
from math import ceil

import numpy as np

from .rules import multiplier

TURN_ACTIONS = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3}


def base(t):
    return min(TURN_ACTIONS.get(t, 4), 4)


def actions_for(budget):
    out = []
    for a in range(budget + 1):
        for d in range(budget - a + 1):
            b = budget - a - d
            if b <= 4:
                out.append((a, d, b))
    return tuple(out)


class Solver:
    def __init__(self, typeA, typeB, atkA, atkB, hpA, hpB,
                 cap=20, stall_cap=3, start_turn=1, opp_remainder=0,
                 first_move=0, start_states=None, root_weights=None,
                 cfr_plus=True, dtype=np.float32,
                 gamma=1.0, full_history=False):
        self.full_history = full_history
        self.typeA, self.typeB = typeA, typeB
        self.atkA, self.atkB = atkA, atkB
        self.hpA, self.hpB = hpA, hpB          # real HP, kept for reporting
        self.multA = multiplier(typeA, typeB)  # A attacking B
        self.multB = multiplier(typeB, typeA)  # B attacking A
        self.dA = atkA * self.multA            # unrounded per-hit dmg A->B
        self.dB = atkB * self.multB            # unrounded per-hit dmg B->A
        self.hA0 = max(1, ceil(hpA / self.dB)) if hpA else 0
        self.hB0 = max(1, ceil(hpB / self.dA)) if hpB else 0
        self.cap = cap
        self.stall_cap = stall_cap
        self.cfr_plus = cfr_plus
        self.dtype = dtype
        self.gamma = gamma
        self._t = 0
        # start states: either explicit (mid-game position) or built from
        # first_move / opp_remainder (clean start at start_turn).
        if start_states is not None:
            self.start_states = [tuple(st) for st in start_states]
            base_turn = max(st[6] for st in self.start_states)
        else:
            first_move = int(first_move)
            if first_move == 0:
                # A acts first; B's hidden remainder is opp_remainder.
                self.start_states = [
                    (self.hA0, self.hB0, 0, bankB, 0, shB, start_turn, 0)
                    for shB in range(opp_remainder + 1)
                    for bankB in (opp_remainder - shB,)
                ]
            else:
                # B acts first; A's hidden remainder is opp_remainder.
                self.start_states = [
                    (self.hA0, self.hB0, bankA, 0, shA, 0, start_turn, 1)
                    for shA in range(opp_remainder + 1)
                    for bankA in (opp_remainder - shA,)
                ]
            base_turn = start_turn
        self.turn_cap = base_turn + cap
        self.start_root = self.start_states[0] if self.start_states else \
            (self.hA0, self.hB0, 0, 0, 0, 0, start_turn, 0)
        self.root_weights = root_weights or [1.0] * len(self.start_states)
        self.ACT = {b: actions_for(b) for b in range(0, 9)}
        self.value = {}
        self._value_dirty = True
        self._build_graph()

    def per_hit(self, attacker):
        """Unrounded per-hit damage of `attacker` ('A' or 'B')."""
        return self.dA if attacker == 'A' else self.dB

    def real_hp(self, hits, attacker_of):
        """Representative real HP for a hits count (bucket top, exact kill count)."""
        d = self.per_hit(attacker_of)
        return hits * d

    # ------------------------------------------------------------------ rules
    def actions(self, st):
        hA, hB, bankA, bankB, shA, shB, turn, to_move = st
        own = bankA if to_move == 0 else bankB
        return self.ACT[min(8, base(turn) + own)]

    def transition(self, st, act):
        hA, hB, bankA, bankB, shA, shB, turn, to_move = st
        a, d, b = act
        if to_move == 0:
            landed = max(0, a - shB)
            if landed:
                hB = max(0, hB - landed)
            return (hA, hB, min(4, b), bankB, d, 0, turn + 1, 1)
        landed = max(0, a - shA)
        if landed:
            hA = max(0, hA - landed)
        return (hA, hB, bankA, min(4, b), 0, d, turn + 1, 0)

    def terminal(self, st):
        hA, hB, bankA, bankB, shA, shB, turn, to_move = st
        if hA <= 0:
            return -1.0
        if hB <= 0:
            return 1.0
        if turn > self.turn_cap:
            return 0.0
        return None

    def info_key(self, st):
        hA, hB, bankA, bankB, shA, shB, turn, to_move = st
        if to_move == 0:
            return (turn, hA, hB, 0, bankA, shB + bankB)
        return (turn, hA, hB, 1, bankB, shA + bankA)

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        # TREE build: every history is a distinct node, no state dedup.
        self.states = []
        self.term = []
        self.roots = []
        queue = deque()
        for st in self.start_states:
            self.roots.append(len(self.states))
            self.states.append(st)
            self.term.append(self.terminal(st))
            queue.append(len(self.states) - 1)
        self.children = [[] for _ in self.start_states]
        self.parent = [-1] * len(self.start_states)
        while queue:
            i = queue.popleft()
            st = self.states[i]
            if self.term[i] is not None:
                continue
            for act in self.actions(st):
                child = self.transition(st, act)
                cid = len(self.states)
                self.states.append(child)
                self.term.append(self.terminal(child))
                self.children[i].append(cid)
                self.parent.append(i)
                self.children.append([])
                queue.append(cid)
        n = len(self.states)
        if getattr(self, "full_history", False):
            # perfect-recall info sets: each player remembers its OWN full
            # observation sequence (the info_key at each of its decisions).
            a_seq = [None] * n
            b_seq = [None] * n
            for i, st in enumerate(self.states):
                p = self.parent[i]
                if p >= 0:
                    a_seq[i] = a_seq[p]
                    b_seq[i] = b_seq[p]
                ik = self.info_key(st)
                if st[7] == 0:
                    a_seq[i] = (a_seq[i] or ()) + (ik,)
                else:
                    b_seq[i] = (b_seq[i] or ()) + (ik,)
            self.info_of = [a_seq[i] if st[7] == 0 else b_seq[i]
                            for i, st in enumerate(self.states)]
        else:
            self.info_of = [self.info_key(st) for st in self.states]
        # ---- numpy structures ---------------------------------------------
        self.n_states = n
        self.is_term = np.array([t is not None for t in self.term], dtype=bool)
        self.term_arr = np.array([t if t is not None else 0.0
                                  for t in self.term], dtype=np.float64)
        self.to_move = np.array([st[7] for st in self.states], dtype=np.int32)
        # info sets: only for acting (non-terminal) states
        self.info_list = []
        self.info_to_id = {}
        for i in range(n):
            if self.is_term[i]:
                continue
            key = self.info_of[i]
            if key not in self.info_to_id:
                self.info_to_id[key] = len(self.info_list)
                self.info_list.append(key)
        self.n_infos = len(self.info_list)
        self.info_id = np.zeros(n, dtype=np.int32)
        for i in range(n):
            if not self.is_term[i]:
                self.info_id[i] = self.info_to_id[self.info_of[i]]
        # actions per info set / valid mask / max width
        self.n_acts_info = np.array(
            [len(self.ACT[min(8, base(kk[0]) + kk[4])])
             for kk in (key[-1] if isinstance(key[0], tuple) else key
                        for key in self.info_list)], dtype=np.int32)
        self.max_acts = int(self.n_acts_info.max()) if self.n_infos else 1
        self.a_move = np.array([kk[3] == 0 for kk in
                                (key[-1] if isinstance(key[0], tuple) else key
                                 for key in self.info_list)],
                               dtype=bool)
        self.valid = np.zeros((self.n_infos, self.max_acts), dtype=bool)
        for iid, na in enumerate(self.n_acts_info):
            self.valid[iid, :na] = True
        # padded children
        self.child_pad = np.full((n, self.max_acts), -1, dtype=np.int32)
        for i in range(n):
            if self.is_term[i]:
                continue
            self.child_pad[i, :len(self.children[i])] = self.children[i]
        self.roots_arr = np.array(self.roots, dtype=np.int64)
        self.root_w = np.array(self.root_weights, dtype=np.float64)
        # topological levels (children have strictly higher indices)
        level = np.zeros(n, dtype=np.int32)
        for i in range(n):
            if self.is_term[i]:
                continue
            li = level[i] + 1
            for ci in self.children[i]:
                if level[ci] < li:
                    level[ci] = li
        max_level = int(level.max()) + 1 if n else 0
        self.levels = [np.nonzero(level == L)[0] for L in range(max_level)]
        # regrets / average strategy / value, dense arrays
        self.regret = np.zeros((self.n_infos, self.max_acts), dtype=self.dtype)
        self.avg = np.zeros((self.n_infos, self.max_acts), dtype=self.dtype)
        self._sig = np.zeros((self.n_infos, self.max_acts), dtype=self.dtype)
        
        # OPTIMIZATION: pre-compute constant masks
        self.act_mask = ~self.is_term
        self.sign = np.where(self.a_move, 1.0, -1.0)
        
        # OPTIMIZATION: pre-allocate reusable temporary arrays
        self._rA = np.zeros(n, dtype=np.float64)
        self._rB = np.zeros(n, dtype=np.float64)
        self._v_state = np.zeros(n, dtype=np.float64)
        self._cfv_sum = np.zeros((self.n_infos, self.max_acts), dtype=self.dtype)
        self._other_sum = np.zeros(self.n_infos, dtype=np.float64)
        self._own_sum = np.zeros(self.n_infos, dtype=np.float64)
        self._v_info = np.zeros(self.n_infos, dtype=np.float64)
        
        # OPTIMIZATION: Pre-compute flattened indices for reach propagation
        self._reach_data_a = []
        self._reach_data_b = []
        
        for idx in self.levels:
            if idx.size == 0:
                self._reach_data_a.append(None)
                self._reach_data_b.append(None)
                continue
                
            a_idx = idx[self.to_move[idx] == 0]
            b_idx = idx[self.to_move[idx] == 1]
            
            if a_idx.size:
                c = self.child_pad[a_idx]
                flat = c.ravel()
                m = flat >= 0
                dst = flat[m]
                src_repeated = np.repeat(np.arange(a_idx.size), self.max_acts)[m]
                info_flat = self.info_id[a_idx][src_repeated]
                action_flat = np.tile(np.arange(self.max_acts), a_idx.size)[m]
                self._reach_data_a.append((a_idx, dst, src_repeated, info_flat, action_flat))
            else:
                self._reach_data_a.append(None)
            
            if b_idx.size:
                c = self.child_pad[b_idx]
                flat = c.ravel()
                m = flat >= 0
                dst = flat[m]
                src_repeated = np.repeat(np.arange(b_idx.size), self.max_acts)[m]
                info_flat = self.info_id[b_idx][src_repeated]
                action_flat = np.tile(np.arange(self.max_acts), b_idx.size)[m]
                self._reach_data_b.append((b_idx, dst, src_repeated, info_flat, action_flat))
            else:
                self._reach_data_b.append(None)
        
        # Pre-compute backward pass structures
        self._value_data = []
        for idx in self.levels:
            if idx.size == 0:
                self._value_data.append(None)
                continue
            nt = idx[~self.is_term[idx]]
            if nt.size == 0:
                self._value_data.append(None)
                continue
            cp = self.child_pad[nt]
            mask = cp >= 0
            self._value_data.append((nt, cp, mask))

    # ------------------------------------------------------------------- CFR
    def release_graph(self, keep_query=True):
        """Free the Python graph structures after training to save memory.

        ``iteration()`` only needs the numpy structures (child_pad, reach_data,
        value_data, arrays), never the Python lists built during graph
        construction. ``release_graph()`` drops the heavy Python lists
        (states/index/children/levels/roots/term/start_states) so a heavy
        cap=20 solver trains in ~0.3 GB instead of ~0.9 GB. Best-response and
        outcome-rate helpers that read those Python structures become
        unavailable after the release. ``keep_query`` keeps info_of/roots_arr
        so ``average_strategy``, ``start_strategy`` and SolverPolicy still work.
        """
        for attr in ("states", "index", "children", "levels", "roots",
                     "term", "start_states"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        if not keep_query:
            for attr in ("info_to_id", "info_list", "info_of"):
                if hasattr(self, attr):
                    setattr(self, attr, None)

    def _ensure(self, info):
        if info not in self.info_to_id:
            raise KeyError(info)
        return self.info_to_id[info]

    def _strategy(self, info):
        """Current (regret-matching) strategy as a tuple, from this iteration's
        cache when available."""
        iid = self._ensure(info)
        row = self._sig[iid, :self.n_acts_info[iid]]
        return tuple(float(x) for x in row)

    def _compute_sig(self):
        """Regret-matching strategy for ALL info sets, vectorized."""
        pos = np.maximum(self.regret, 0.0)
        total = pos.sum(axis=1)
        nv = np.maximum(self.n_acts_info, 1)
        uniform = 1.0 / nv
        self._sig = np.where(self.valid,
                             np.where(total[:, None] > 0,
                                      pos / np.maximum(total[:, None], 1e-12),
                                      uniform[:, None]),
                             0.0)

    def _sync_value_dict(self):
        """Lazy sync: only update value dict when needed."""
        if self._value_dirty:
            for iid, key in enumerate(self.info_list):
                self.value[key] = float(self._v_info[iid])
            self._value_dirty = False

    def iteration(self):
        n = self.n_states
        self._compute_sig()
        
        # ---- forward: reach weights (optimized) ---------------------------
        rA = self._rA
        rB = self._rB
        rA.fill(0.0)
        rB.fill(0.0)
        rA[self.roots_arr] = self.root_w
        rB[self.roots_arr] = self.root_w
        sig = self._sig
        
        for data_a, data_b in zip(self._reach_data_a, self._reach_data_b):
            if data_a is not None:
                a_idx, dst, src_rep, info_flat, act_flat = data_a
                wa = rA[a_idx][src_rep]
                wb = rB[a_idx][src_rep]
                sa = sig[info_flat, act_flat]
                rA += np.bincount(dst, weights=wa * sa, minlength=n)
                rB += np.bincount(dst, weights=wb, minlength=n)
            
            if data_b is not None:
                b_idx, dst, src_rep, info_flat, act_flat = data_b
                wb = rB[b_idx][src_rep]
                wa = rA[b_idx][src_rep]
                sb = sig[info_flat, act_flat]
                rB += np.bincount(dst, weights=wb * sb, minlength=n)
                rA += np.bincount(dst, weights=wa, minlength=n)
        
        # ---- concrete-state values (optimized) ----------------------------
        v_state = self._v_state
        np.copyto(v_state, self.term_arr, where=self.is_term)
        np.putmask(v_state, ~self.is_term, 0.0)
        
        for data in reversed(self._value_data):
            if data is None:
                continue
            nt, cp, mask = data
            vc = np.where(mask, v_state[np.where(mask, cp, 0)], 0.0)
            s = sig[self.info_id[nt]]
            v_state[nt] = self.gamma * (s * vc).sum(axis=1)
        
        # ---- counterfactual values ----------------------------------------
        other = np.where(self.to_move == 0, rB, rA)
        own = np.where(self.to_move == 0, rA, rB)
        safe = np.where(self.child_pad >= 0, self.child_pad, 0)
        vc = np.where(self.child_pad >= 0, v_state[safe], 0.0)
        contrib = other[:, None] * vc
        cfv_sum = self._cfv_sum
        other_sum = self._other_sum
        own_sum = self._own_sum
        cfv_sum.fill(0.0)
        other_sum.fill(0.0)
        own_sum.fill(0.0)
        np.add.at(cfv_sum, self.info_id[self.act_mask], contrib[self.act_mask])
        np.add.at(other_sum, self.info_id[self.act_mask], other[self.act_mask])
        np.add.at(own_sum, self.info_id[self.act_mask], own[self.act_mask])
        with np.errstate(divide='ignore', invalid='ignore'):
            cfv = np.where(other_sum[:, None] > 0,
                           cfv_sum / other_sum[:, None], 0.0)
        v_info = self._v_info
        v_info[:] = (sig * cfv).sum(axis=1)
        
        # ---- regret / average-strategy updates ----------------------------
        delta = other_sum[:, None] * self.sign[:, None] * (cfv - v_info[:, None])
        if self.cfr_plus:
            self._t += 1
            update_A = (self._t % 2 == 1)
            mask = (self.a_move == update_A)
            self.regret = np.maximum(
                self.regret + np.where(self.valid & mask[:, None], delta, 0.0),
                0.0)
            self.avg += np.where(self.valid & mask[:, None],
                                 self._t * own_sum[:, None] * sig, 0.0)
        else:
            self.regret += np.where(self.valid, delta, 0.0)
            self.avg += np.where(self.valid, own_sum[:, None] * sig, 0.0)
        
        self._value_dirty = True

    def average_strategy(self, info):
        iid = self._ensure(info)
        avg = self.avg[iid, :self.n_acts_info[iid]]
        total = float(avg.sum())
        if total <= 0:
            k = len(avg)
            return tuple(1.0 / k for _ in range(k))
        return tuple(float(x) / total for x in avg)

    def run(self, iterations, silent=False, report_every=10,
            tol=1e-4, opening_tol=0.005, min_iters=15):
        t0 = time.time()
        start_info = self.info_of[self.roots[0]]
        prev_v = None
        prev_open = None
        stable = 0
        for it in range(1, iterations + 1):
            self.iteration()
            self._sync_value_dict()
            v = self.value.get(start_info, 0.0)
            opening = self.average_strategy(start_info)
            v_stable = prev_v is not None and abs(v - prev_v) < tol
            o_stable = prev_open is not None and sum(
                abs(x - y) for x, y in zip(opening, prev_open)) < opening_tol
            stable = stable + 1 if (v_stable and o_stable) else 0
            prev_v, prev_open = v, opening
            if not silent and it % report_every == 0:
                print(f"  iter={it:5d} V(A)={v:+.4f} win_rate={ (v+1)/2:.4f} "
                      f"infosets={self.n_infos:6d} "
                      f"{time.time()-t0:6.2f}s", flush=True)
            if stable >= 2 and it >= min_iters:
                if not silent:
                    print(f"  converged at iter {it} "
                          f"({time.time()-t0:.2f}s)", flush=True)
                break
        return self.value.get(start_info, 0.0)

    def best_response_value(self, br_player):
        self._sync_value_dict()
        opp = 1 - br_player
        opp_sig = {}
        for i in range(len(self.states)):
            if self.term[i] is not None or not self.children[i]:
                continue
            info = self.info_of[i]
            if self.states[i][7] == opp and info not in opp_sig:
                opp_sig[info] = self.average_strategy(info)
        br = [0.0] * len(self.states)
        for i in range(len(self.states) - 1, -1, -1):
            st = self.states[i]
            t = self.term[i]
            if t is not None:
                br[i] = t if br_player == 0 else -t
                continue
            if not self.children[i]:
                br[i] = 0.0
                continue
            if st[7] == br_player:
                br[i] = max(br[j] for j in self.children[i])
            else:
                sig = opp_sig[self.info_of[i]]
                br[i] = sum(sig[k] * br[j]
                            for k, j in enumerate(self.children[i]))
        return br[0]

    def strategy_at(self, info):
        iid = self._ensure(info)
        sig = self.average_strategy(info)
        acts = self.ACT[min(8, base(info[0]) + info[4])]
        return [(acts[j], round(sig[j], 4)) for j in range(len(acts))
                if sig[j] > 0.001]

    def info_set_br_value(self, br_player, passes=128):
        """Best-response value with the CORRECT info-set constraint: the BR
        player must pick one action per info set (they cannot see which hidden
        split they face). This is the valid exploitability measure; the
        per-concrete-state max in `best_response_value` is illegal (it gives
        the BR the hidden state and always yields ~1.0)."""
        self._sync_value_dict()
        opp = 1 - br_player
        n = self.n_states
        opp_sig = {}
        for i, st in enumerate(self.states):
            if self.term[i] is not None:
                continue
            info = self.info_of[i]
            if st[7] == opp and info not in opp_sig:
                opp_sig[info] = self.average_strategy(info)
        opp_reach = np.zeros(n)
        opp_reach[self.roots_arr] = self.root_w
        for i in range(n):
            if self.term[i] is not None or not self.children[i]:
                continue
            if self.states[i][7] == opp:
                sig = opp_sig[self.info_of[i]]
                for j, ci in enumerate(self.children[i]):
                    opp_reach[ci] += opp_reach[i] * sig[j]
            else:
                for ci in self.children[i]:
                    opp_reach[ci] += opp_reach[i]
        br_groups = {}
        for i, st in enumerate(self.states):
            if self.term[i] is not None or not self.children[i]:
                continue
            if st[7] == br_player:
                br_groups.setdefault(self.info_of[i], []).append(i)
        v = np.zeros(n)
        for i in range(n):
            t = self.term[i]
            if t is not None:
                v[i] = t if br_player == 0 else -t
        for _pass in range(passes):
            changed = False
            for i in range(n - 1, -1, -1):
                if self.term[i] is not None or not self.children[i]:
                    continue
                if self.states[i][7] == opp:
                    sig = opp_sig[self.info_of[i]]
                    nv = self.gamma * sum(sig[j] * v[ci]
                                          for j, ci in enumerate(self.children[i]))
                    if abs(nv - v[i]) > 1e-9:
                        v[i] = nv
                        changed = True
            for info, idxs in br_groups.items():
                tot = sum(opp_reach[i] for i in idxs) or 1.0
                best_a, best_val = None, None
                for a in range(len(self.children[idxs[0]])):
                    av = self.gamma * sum(opp_reach[i] * v[self.children[i][a]]
                                          for i in idxs) / tot
                    if best_val is None or av > best_val:
                        best_val, best_a = av, a
                for i in idxs:
                    nv = self.gamma * v[self.children[i][best_a]]
                    if abs(nv - v[i]) > 1e-9:
                        v[i] = nv
                        changed = True
            if not changed:
                self.br_passes_used = _pass + 1
                break
        else:
            self.br_passes_used = passes
        weights = self.root_w / self.root_w.sum()
        return float(np.dot(weights, v[self.roots_arr]))

    def profile_value(self):
        """Discounted value of the AVERAGE profile: both players follow the
        average strategy, terminal payoffs discounted by gamma^depth. This is
        the correct reference value for gamma-consistent exploitability."""
        n = self.n_states
        v = [0.0] * n
        for i in range(n - 1, -1, -1):
            t = self.term[i]
            if t is not None:
                v[i] = t
            elif self.children[i]:
                sig = self.average_strategy(self.info_of[i])
                v[i] = self.gamma * sum(sig[j] * v[ci]
                                        for j, ci in enumerate(self.children[i]))
        weights = self.root_w / self.root_w.sum()
        return float(np.dot(weights, [v[int(r)] for r in self.roots_arr]))

    def true_exploitability(self):
        """Exploitability of the AVERAGE strategy profile (what the solver
        actually plays), measured with the info-set-constrained best response
        in the SAME (possibly discounted) game as the profile. Reference is the
        average profile's own (discounted) value. Returns (expl_A, expl_B)
        with A's value in A's terms; a profile is near-Nash when both are
        small. NOTE: `best_response_value` (per-state max) overstates and must
        not be used for this."""
        value_avg = self.profile_value()
        brA = self.info_set_br_value(0)
        brB = self.info_set_br_value(1)   # B's utility, in B's terms
        return brA - value_avg, value_avg + brB

    def outcome_rates(self):
        n = self.n_states
        w = np.zeros(n)
        w[self.roots_arr] = self.root_w
        winA = winB = draw = 0.0
        for i in range(n):
            t = self.term[i]
            if t is not None or w[i] <= 0:
                if t is not None:
                    if t > 0:
                        winA += w[i]
                    elif t < 0:
                        winB += w[i]
                    else:
                        draw += w[i]
                continue
            sig = self.average_strategy(self.info_of[i])
            for j, ci in enumerate(self.children[i]):
                w[ci] += w[i] * sig[j]
        total = winA + winB + draw
        return winA / total, winB / total, draw / total

    def start_strategy(self):
        info = self.info_of[self.roots_arr[0]]
        sig = self.average_strategy(info)
        acts = self.actions(self.start_root)
        return [(acts[j], round(sig[j], 4)) for j in range(len(acts))
                if sig[j] > 0.001]


if __name__ == "__main__":
    args = sys.argv[1:]
    from .rules import Type
    tA = args[0] if args else "C"
    tB = args[1] if len(args) > 1 else "C"
    aA = int(args[2]) if len(args) > 2 else 2000
    aB = int(args[3]) if len(args) > 3 else 2000
    hA = int(args[4]) if len(args) > 4 else 10000
    hB = int(args[5]) if len(args) > 5 else 10000
    iters = int(args[6]) if len(args) > 6 else 200
    s = Solver(getattr(Type, tA), getattr(Type, tB), aA, aB, hA, hB, cap=20)
    print(f"graph: states={len(s.states):,} infosets={s.n_infos:,} "
          f"hits A={s.hA0} B={s.hB0}")
    s.run(iters)
    v = s.value[s.info_of[s.roots_arr[0]]]
    eA, eB = s.true_exploitability()
    w, l, d = s.outcome_rates()
    print(f"\nV(current)={v:+.4f}  avg-profile W={w:.3f} L={l:.3f} D={d:.3f}")
    print(f"true exploitability of AVERAGE strategy: A={eA:+.4f} B={eB:+.4f} "
          f"total={eA+eB:.4f}  (near 0 = Nash)")
    print("opening:")
    for act, p in s.start_strategy():
        print(f"  a{act[0]}/d{act[1]}/b{act[2]}  p={p:.4f}")
