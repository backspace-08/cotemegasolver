"""Value-network leaf evaluation for the MicroTree override (DeepStack/ReBeL).

Decodes the Rust flat leaf states back into the actor-relative feature vector,
feeds the belief over the ACTOR's opponent (the caller's belief when the actor
is the caller's side, else a uniform prior over the other split), and returns
the value from A's perspective (flipping the actor value when B acts).

IMPORTANT: this module must stay import-torch-free at the top level. The
multiprocessing workers on Windows (spawn) deadlock on torch's DLL import, so
the workers only ever use the pure-numpy path (NumpyValueNet).
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(BASE, ".."))
sys.path.insert(0, REPO)

MAX_CHARS = 3
BELIEF_DIM = 9

_IN_DIM = MAX_CHARS * 7 * 2 + 5 + BELIEF_DIM


def char_slot(team, order, hp, k):
    """Stack-order character slot: [type onehot(4), atk/2100, hp/600, alive]."""
    if k >= len(order):
        return [0.0] * 4 + [0.0, 0.0, 0.0]
    idx = order[k]
    typ, atk, _ = team[idx]
    onehot = [0.0] * 4
    onehot[typ] = 1.0
    return onehot + [atk / 2100.0, hp[k] / 600.0, 1.0]


def flat_to_features(team_a, team_b, fs, belief):
    p = 0

    def read():
        nonlocal p
        v = fs[p]
        p += 1
        return v

    la = read()
    order_a = [read() for _ in range(la)]
    hp_a = [read() for _ in range(la)]
    bank_a = read()
    sh_a = read()
    lb = read()
    order_b = [read() for _ in range(lb)]
    hp_b = [read() for _ in range(lb)]
    bank_b = read()
    sh_b = read()
    turn = read()
    to_move = read()
    if to_move == 0:
        actor_team, opp_team = team_a, team_b
        actor_order, opp_order = order_a, order_b
        actor_hp, opp_hp = hp_a, hp_b
        actor_bank, actor_sh = bank_a, sh_a
        r_opp = sh_b + bank_b
    else:
        actor_team, opp_team = team_b, team_a
        actor_order, opp_order = order_b, order_a
        actor_hp, opp_hp = hp_b, hp_a
        actor_bank, actor_sh = bank_b, sh_b
        r_opp = sh_a + bank_a
    feats = []
    for k in range(MAX_CHARS):
        feats.extend(char_slot(actor_team, actor_order, actor_hp, k))
    for k in range(MAX_CHARS):
        feats.extend(char_slot(opp_team, opp_order, opp_hp, k))
    feats += [actor_bank / 4.0, actor_sh / 4.0,
              turn / 9.0, float(to_move), r_opp / 4.0]
    bel = belief + [0.0] * (BELIEF_DIM - len(belief))
    feats += bel[:BELIEF_DIM]
    return feats


def uniform_belief(r):
    return [1.0 / (r + 1)] * (r + 1)


class NumpyValueNet:
    """Pure-numpy forward pass of the trained MLP (no torch import needed, so
    multiprocessing workers on Windows/spawn cannot deadlock on torch's DLLs)."""

    def __init__(self, npz_path):
        import numpy as np
        z = np.load(npz_path)
        self.w1, self.b1 = z["w1"], z["b1"]
        self.w2, self.b2 = z["w2"], z["b2"]
        self.w3, self.b3 = z["w3"], z["b3"]

    def __call__(self, x):
        import numpy as np
        h = np.maximum(x @ self.w1.T + self.b1, 0.0)
        h = np.maximum(h @ self.w2.T + self.b2, 0.0)
        return np.tanh(h @ self.w3.T + self.b3).reshape(-1)


class ValueLeaf:
    """Evaluates 2v2/3v3 leaves with the trained network.

    ``net`` is either a torch ValueNet or a NumpyValueNet; the belief over the
    opponent's split is passed per call (it updates every turn). Leaves where
    the actor is the other side get a uniform prior over that side's split (we
    do not hold their belief).
    """

    def __init__(self, model_path):
        if model_path.endswith(".npz"):
            self.net = NumpyValueNet(model_path)
        else:
            import torch
            from server.train_value_net import ValueNet
            self.net = ValueNet(_IN_DIM)
            self.net.load_state_dict(torch.load(model_path,
                                                map_location="cpu",
                                                weights_only=True))
            self.net.eval()

    def override(self, team_a, team_b, leaf_states, bot_belief, bot_side,
                 batch=512):
        if not leaf_states:
            return []
        import numpy as np
        x = []
        keys = []
        for fs in leaf_states:
            actor = fs[-1]
            if actor == bot_side:
                belief = bot_belief
            else:
                r_other = _other_r(fs)
                belief = uniform_belief(r_other)
            x.append(flat_to_features(team_a, team_b, fs, belief))
            keys.append(tuple(fs))
        X = np.array(x, dtype=np.float32)
        if isinstance(self.net, NumpyValueNet):
            av = self.net(X)
        else:
            import torch
            with torch.no_grad():
                av = self.net(torch.from_numpy(X)).numpy()
        override = []
        for j in range(len(keys)):
            actor = leaf_states[j][-1]
            a_value = float(av[j]) if actor == 0 else -float(av[j])
            override.append((keys[j], a_value))
        return override


def _other_r(fs):
    """Sum of the other side's split from a flat state (only the tail matters)."""
    # lenA orderA* hpA* bankA shA lenB orderB* hpB* bankB shB turn to_move
    la = fs[0]
    i = 1 + la + la + 2
    lb = fs[i]
    bank_b = fs[i + 1 + lb + lb]
    sh_b = fs[i + 1 + lb + lb + 1]
    bank_a = fs[1 + la + la]
    sh_a = fs[1 + la + la + 1]
    actor = fs[-1]
    return (sh_b + bank_b) if actor == 0 else (sh_a + bank_a)
