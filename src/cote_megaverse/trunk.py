"""Full-game (3v3) trunk rules for the CFR-D / subgame-solving solver.

The trunk is the upper level of the game: all characters, their stack order,
voluntary switches, forced promotion on death, banks, shields and the turn
schedule. It is solved by CFR; when the game reduces to a 1v1 (each side has
one living character), the trunk hands off to a belief-conditioned 1v1 "leaf"
(the 1v1 table) whose value is used as the terminal value.

Abstraction
-----------
Real HP is tracked in units of 10: ``hp_units = hp // 10``. Damage matches
rules.py exactly: each hit deals ``round(atk * mult / 100) * 100`` HP ==
``round(atk * mult / 100) * 10`` units, total = per-hit times ``landed``.
Each character's HP is tracked individually (a character switched out keeps its
damaged HP and returns damaged).

State
-----
A tuple ``(orderA, hpA, bankA, shA, orderB, hpB, bankB, shB, turn, to_move)``:
  orderX  tuple of living character ids in stack order, front = active
  hpX     tuple of per-character HP units, parallel to orderX
  bankX   stored bonus 0..4 (own bank is drained into the budget each turn)
  shX     held shields (hidden from the opponent)
  turn    side-turn count, to_move 0 = A acts, 1 = B acts

Character attributes (type, atk, hp) are fixed per game instance and stored on
the Trunk, not in the state. Hits-to-kill against a given active opponent:
``ceil(hp_units / per_hit_units)`` (exact, per-hit damage is linear).
"""
from __future__ import annotations

MAX_BANK = 4
TURN_ACTIONS = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3}


def base(t):
    return min(TURN_ACTIONS.get(t, 4), 4)


def multiplier(attacker: int, defender: int) -> float:
    if (attacker + 1) % 4 == defender:
        return 1.3
    if (defender + 1) % 4 == attacker:
        return 0.7
    return 1.0


def to_units(hp: int) -> int:
    return hp // 10


class Trunk:
    """Full 3v3 trunk for one concrete game instance (fixed teams)."""

    def __init__(self, teamA, teamB, cap=20, start_turn=1, first_move=0,
                 opp_remainder=0):
        # teamX: tuple of (type:int, atk:int, hp:int); ids 0..n-1 per side.
        self.teamA = tuple(teamA)
        self.teamB = tuple(teamB)
        self.unitsA = [to_units(c[2]) for c in self.teamA]
        self.unitsB = [to_units(c[2]) for c in self.teamB]
        self.cap = cap
        self.turn_cap = start_turn + cap
        self.start_turn = start_turn
        self.first_move = first_move
        self.opp_remainder = opp_remainder
        self.idsA = tuple(range(len(teamA)))
        self.idsB = tuple(range(len(teamB)))
        self._start_states()

    def _start_states(self):
        orderA, hpA = self.idsA, tuple(self.unitsA)
        orderB, hpB = self.idsB, tuple(self.unitsB)
        if self.first_move == 0:
            self.start_states = [
                (orderA, hpA, 0, 0, orderB, hpB, bankB, shB,
                 self.start_turn, 0)
                for shB in range(self.opp_remainder + 1)
                for bankB in (self.opp_remainder - shB,)
            ]
        else:
            self.start_states = [
                (orderA, hpA, bankA, shA, orderB, hpB, 0, 0,
                 self.start_turn, 1)
                for shA in range(self.opp_remainder + 1)
                for bankA in (self.opp_remainder - shA,)
            ]

    # ------------------------------------------------------------ damage
    def per_hit_units(self, actor_side, actor_id, defender_side, defender_id):
        a = self.teamA if actor_side == 0 else self.teamB
        d = self.teamB if actor_side == 0 else self.teamA
        atk = a[actor_id][1]
        m = multiplier(a[actor_id][0], d[defender_id][0])
        return round(atk * m / 100) * 10

    def dmg_total_units(self, actor_side, actor_id, defender_side,
                        defender_id, landed):
        if landed <= 0:
            return 0
        return self.per_hit_units(actor_side, actor_id, defender_side,
                                  defender_id) * landed

    def hits(self, hp_units, actor_side, actor_id, defender_side, defender_id):
        ph = self.per_hit_units(actor_side, actor_id, defender_side, defender_id)
        if ph <= 0:
            return 10 ** 6
        return max(1, (hp_units + ph - 1) // ph)

    # ------------------------------------------------------------ budget
    def budget(self, st):
        orderA, hpA, bankA, shA, orderB, hpB, bankB, shB, turn, tm = st
        own = bankA if tm == 0 else bankB
        return base(turn) + own

    # ------------------------------------------------------------ actions
    def actions(self, st):
        orderA, hpA, bankA, shA, orderB, hpB, bankB, shB, turn, tm = st
        order = orderA if tm == 0 else orderB
        bgt = self.budget(st)
        out = []
        for sw in (None,) + tuple(order[1:]):
            rem = bgt - (1 if sw is not None else 0)
            for a in range(rem + 1):
                for d in range(rem - a + 1):
                    b = rem - a - d
                    if b <= MAX_BANK:
                        out.append((a, d, b, sw))
        return out

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _promote(order, hp, dead):
        pairs = [(c, h) for c, h in zip(order, hp) if c != dead]
        if not pairs:
            return (), ()
        return ((pairs[0][0],) + tuple(c for c, _ in pairs[1:]),
                (pairs[0][1],) + tuple(h for _, h in pairs[1:]))

    # ------------------------------------------------------------ transition
    def transition(self, st, act):
        orderA, hpA, bankA, shA, orderB, hpB, bankB, shB, turn, tm = st
        a, d, b, sw = act
        if tm == 0:
            order = list(orderA)
            hp = list(hpA)
            if sw is not None:
                i = order.index(sw)
                order.insert(0, order.pop(i))
                hp.insert(0, hp.pop(i))
            dmg = self.dmg_total_units(0, order[0], 1, orderB[0], max(0, a - shB))
            landed = max(0, a - shB)
            if landed:
                hpb = list(hpB)
                hpb[0] = max(0, hpb[0] - dmg)
                if hpb[0] <= 0:
                    orderB, hpB = self._promote(orderB, hpb, orderB[0])
                else:
                    hpB = tuple(hpb)
            return (tuple(order), tuple(hp), min(4, b), d, orderB, hpB,
                    bankB, 0, turn + 1, 1)
        order = list(orderB)
        hp = list(hpB)
        if sw is not None:
            i = order.index(sw)
            order.insert(0, order.pop(i))
            hp.insert(0, hp.pop(i))
        dmg = self.dmg_total_units(1, order[0], 0, orderA[0], max(0, a - shA))
        landed = max(0, a - shA)
        if landed:
            hpa = list(hpA)
            hpa[0] = max(0, hpa[0] - dmg)
            if hpa[0] <= 0:
                orderA, hpA = self._promote(orderA, hpa, orderA[0])
            else:
                hpA = tuple(hpa)
        return (orderA, hpA, bankA, 0, tuple(order), tuple(hp), min(4, b), d,
                turn + 1, 0)

    # ------------------------------------------------------------ terminal
    def terminal(self, st):
        orderA, hpA, bankA, shA, orderB, hpB, bankB, shB, turn, tm = st
        if not orderA:
            return -1.0
        if not orderB:
            return 1.0
        if turn > self.turn_cap:
            return 0.0
        return None

    # ------------------------------------------------------------ info key
    def info_key(self, st):
        orderA, hpA, bankA, shA, orderB, hpB, bankB, shB, turn, tm = st
        if tm == 0:
            return (turn, orderA, hpA, orderB, hpB, bankA, shB + bankB)
        return (turn, orderB, hpB, orderA, hpA, bankB, shA + bankA)
