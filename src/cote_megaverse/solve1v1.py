"""Exact 1v1 equilibrium solver — CFR via forward/backward DP on the game DAG.

Game: one character per side, fixed 2000 ATK (20 units), type multiplier 1.0,
HP in units of 100 (initial 60). No switches.

DAG: turn and HP progress monotonically, so a pure tree-walk explodes through
transpositions. Each CFR iteration instead does:
  forward pass  : realization weights r_A, r_B over all reachable states
  backward pass : counterfactual values aggregated per info set + regret update

Draw rule (matches the benchmark's spirit, keeps the graph finite): if a side
goes `stall_cap` consecutive own turns without attacking, the game is drawn;
also drawn if `turn > cap`.

Info set of the acting side (every component is public):
  (turn, hpA, hpB, to_move, own_bank, r, own_stall, opp_stall)
  r = opp_shields + opp_bank (the opponent's last public remainder).

Everything is in player A's utility: +1 A wins, -1 B wins, 0 draw.
"""
import sys
import time
from collections import deque

TURN_ACTIONS = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3}
ATK = 20  # 2000 / 100


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
    def __init__(self, cap=30, stall_cap=3, atk=ATK, start_hp=60, start=None):
        self.cap = cap
        self.stall_cap = stall_cap
        self.atk = atk
        if start is None:
            start = (start_hp, start_hp, 0, 0, 0, 0, 0, 0, 1, 0)
        self.start = start
        self.ACT = {b: actions_for(b) for b in range(0, 9)}
        self.regret = {}
        self.avg = {}
        self.value = {}
        self._build_graph()

    # ------------------------------------------------------------------ rules
    def actions(self, st):
        hpA, hpB, bankA, bankB, shA, shB, stA, stB, turn, to_move = st
        own = bankA if to_move == 0 else bankB
        return self.ACT[min(8, base(turn) + own)]

    def transition(self, st, act):
        hpA, hpB, bankA, bankB, shA, shB, stA, stB, turn, to_move = st
        a, d, b = act
        if to_move == 0:
            dmg = max(0, a - shB) * self.atk
            if dmg:
                hpB = max(0, hpB - dmg)
            stA = 0 if a else stA + 1
            return (hpA, hpB, min(4, b), bankB, d, 0, stA, stB, turn + 1, 1)
        dmg = max(0, a - shA) * self.atk
        if dmg:
            hpA = max(0, hpA - dmg)
        stB = 0 if a else stB + 1
        return (hpA, hpB, bankA, min(4, b), 0, d, stA, stB, turn + 1, 0)

    def terminal(self, st):
        hpA, hpB, bankA, bankB, shA, shB, stA, stB, turn, to_move = st
        if hpA <= 0:
            return -1.0
        if hpB <= 0:
            return 1.0
        if turn > self.cap:
            return 0.0
        own_stall = stA if to_move == 0 else stB
        if own_stall >= self.stall_cap:
            return 0.0
        return None

    def info_key(self, st):
        hpA, hpB, bankA, bankB, shA, shB, stA, stB, turn, to_move = st
        if to_move == 0:
            return (turn, hpA, hpB, 0, bankA, shB + bankB, stA, stB)
        return (turn, hpA, hpB, 1, bankB, shA + bankA, stB, stA)

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        self.states = []
        self.index = {}
        self.term = []
        queue = deque([self.start])
        while queue:
            st = queue.popleft()
            if st in self.index:
                continue
            self.index[st] = len(self.states)
            self.states.append(st)
            t = self.terminal(st)
            self.term.append(t)
            if t is None:
                for act in self.actions(st):
                    child = self.transition(st, act)
                    if child not in self.index:
                        queue.append(child)
        n = len(self.states)
        self.info_of = [self.info_key(st) for st in self.states]
        self.children = [[] for _ in range(n)]
        for i, st in enumerate(self.states):
            if self.term[i] is not None:
                continue
            for act in self.actions(st):
                child = self.transition(st, act)
                self.children[i].append(self.index[child])

    # ------------------------------------------------------------------- CFR
    def _ensure(self, info):
        if info not in self.regret:
            k = len(self.ACT[min(8, base(info[0]) + info[4])])
            self.regret[info] = [0.0] * k
            self.avg[info] = [0.0] * k

    def _strategy(self, info):
        self._ensure(info)
        r = self.regret[info]
        total = sum(max(x, 0.0) for x in r)
        if total <= 0:
            k = len(r)
            return tuple(1.0 / k for _ in range(k))
        return tuple(max(x, 0.0) / total for x in r)

    def iteration(self):
        n = len(self.states)
        rA = [0.0] * n
        rB = [0.0] * n
        rA[0] = rB[0] = 1.0
        # forward
        for i in range(n):
            if self.term[i] is not None or not self.children[i]:
                continue
            info = self.info_of[i]
            sig = self._strategy(info)
            wa, wb = rA[i], rB[i]
            to_move = self.states[i][9]
            if to_move == 0:
                for j, ci in enumerate(self.children[i]):
                    p = sig[j]
                    rA[ci] += wa * p
                    rB[ci] += wb
            else:
                for j, ci in enumerate(self.children[i]):
                    p = sig[j]
                    rB[ci] += wb * p
                    rA[ci] += wa
        # backward
        cfv_num = {}
        Rother = {}
        Rown = {}
        for i in range(n - 1, -1, -1):
            if self.term[i] is not None or not self.children[i]:
                continue
            st = self.states[i]
            info = self.info_of[i]
            other = rB[i] if st[9] == 0 else rA[i]
            own = rA[i] if st[9] == 0 else rB[i]
            if other <= 0:
                continue
            bucket = cfv_num.get(info)
            if bucket is None:
                bucket = [0.0] * len(self.children[i])
                cfv_num[info] = bucket
                Rother[info] = 0.0
                Rown[info] = 0.0
            for j, ci in enumerate(self.children[i]):
                t = self.term[ci]
                u = t if t is not None else self.value.get(self.info_of[ci], 0.0)
                bucket[j] += other * u
            Rother[info] += other
            Rown[info] += own
        for info, bucket in cfv_num.items():
            R_other = Rother[info]
            R_own = Rown[info]
            sig = self._strategy(info)
            cfv = [b / R_other for b in bucket]
            v = sum(sig[j] * cfv[j] for j in range(len(cfv)))
            reg = self.regret[info]
            avg = self.avg[info]
            if info[3] == 0:  # A maximizes A's value
                for j in range(len(cfv)):
                    reg[j] += R_other * (cfv[j] - v)
                    avg[j] += R_own * sig[j]
            else:             # B minimizes A's value
                for j in range(len(cfv)):
                    reg[j] += R_other * (v - cfv[j])
                    avg[j] += R_own * sig[j]
            self.value[info] = v

    def average_strategy(self, info):
        self._ensure(info)
        avg = self.avg[info]
        total = sum(avg)
        if total <= 0:
            k = len(avg)
            return tuple(1.0 / k for _ in range(k))
        return tuple(x / total for x in avg)

    def run(self, iterations, silent=False, report_every=10):
        t0 = time.time()
        start_info = self.info_of[0]
        for it in range(1, iterations + 1):
            self.iteration()
            if not silent and it % report_every == 0:
                v = self.value.get(start_info, 0.0)
                print(f"  iter={it:5d} V(A)={v:+.4f} win_rate={ (v+1)/2:.4f} "
                      f"infosets={len(self.avg):6d} "
                      f"{time.time()-t0:6.1f}s", flush=True)
        return self.value.get(start_info, 0.0)

    # ------------------------------------------------------- best response
    def best_response_value(self, br_player):
        """Value for `br_player` (0=A,1=B) against the other's AVERAGE strategy.

        Returns BR player's expected utility (in their own terms). Combined with
        the equilibrium value this yields exploitability.
        """
        opp = 1 - br_player
        # precompute opponent's avg strategies
        opp_sig = {}
        for i in range(len(self.states)):
            if self.term[i] is not None or not self.children[i]:
                continue
            info = self.info_of[i]
            if self.states[i][9] == opp and info not in opp_sig:
                opp_sig[info] = self.average_strategy(info)
        br = [0.0] * len(self.states)   # BR player's utility from each state
        for i in range(len(self.states) - 1, -1, -1):
            st = self.states[i]
            t = self.term[i]
            if t is not None:
                # value in BR player's terms
                br[i] = t if br_player == 0 else -t
                continue
            if not self.children[i]:
                br[i] = 0.0
                continue
            info = self.info_of[i]
            to_move = st[9]
            if to_move == br_player:
                br[i] = max(br[j] for j in self.children[i])
            else:
                sig = opp_sig[info]
                br[i] = sum(sig[k] * br[j]
                            for k, j in enumerate(self.children[i]))
        return br[0]


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    stall = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    s = Solver(cap=cap, stall_cap=stall)
    print(f"graph: states={len(s.states):,} infosets={len(set(s.info_of)):,}")
    t0 = time.time()
    v = s.run(iters)
    print(f"\nafter {iters} iters: V(A)={v:+.4f} win_rate(A)={ (v+1)/2:.4f} "
          f"time={time.time()-t0:.1f}s")
