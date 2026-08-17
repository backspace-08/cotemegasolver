//! COTE Megaverse CFR engine in Rust.
//!
//! First prototype: a port of the full-game trunk rules (`src/cote_megaverse/trunk.py`)
//! with PyO3 bindings, so a depth-limited FH-CFR micro-tree search can run without
//! Python per-node overhead. State/action are flat `Vec<i32>` encodings:
//!
//! state  = [lenA, orderA..., hpA..., bankA, shA,
//!           lenB, orderB..., hpB..., bankB, shB, turn, to_move]
//! action = [a, d, b, sw]  (sw = -1 for no switch, else the target char id)

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const MAX_BANK: i32 = 4;

#[inline]
fn base(t: i32) -> i32 {
    match t {
        1 => 1,
        2..=4 => 2,
        5..=6 => 3,
        _ => 4,
    }
}

#[inline]
fn multiplier(attacker: u8, defender: u8) -> f64 {
    let a = attacker as i32;
    let d = defender as i32;
    if (a + 1) % 4 == d {
        1.3
    } else if (d + 1) % 4 == a {
        0.7
    } else {
        1.0
    }
}

#[derive(Clone, Debug)]
struct State {
    order_a: Vec<u8>,
    hp_a: Vec<i32>,
    bank_a: i32,
    sh_a: i32,
    order_b: Vec<u8>,
    hp_b: Vec<i32>,
    bank_b: i32,
    sh_b: i32,
    turn: i32,
    to_move: i32,
}

#[derive(Clone, Copy, Debug)]
struct Action {
    a: i32,
    d: i32,
    b: i32,
    sw: Option<u8>,
}

fn read(p: &mut usize, raw: &[i32]) -> PyResult<i32> {
    let v = raw
        .get(*p)
        .copied()
        .ok_or_else(|| PyValueError::new_err("state vector too short"))?;
    *p += 1;
    Ok(v)
}

fn decode_state(raw: &[i32]) -> PyResult<State> {
    let mut p = 0usize;
    let len_a = read(&mut p, raw)? as usize;
    let mut order_a = Vec::with_capacity(len_a);
    let mut hp_a = Vec::with_capacity(len_a);
    for _ in 0..len_a {
        order_a.push(read(&mut p, raw)? as u8);
    }
    for _ in 0..len_a {
        hp_a.push(read(&mut p, raw)?);
    }
    let bank_a = read(&mut p, raw)?;
    let sh_a = read(&mut p, raw)?;
    let len_b = read(&mut p, raw)? as usize;
    let mut order_b = Vec::with_capacity(len_b);
    let mut hp_b = Vec::with_capacity(len_b);
    for _ in 0..len_b {
        order_b.push(read(&mut p, raw)? as u8);
    }
    for _ in 0..len_b {
        hp_b.push(read(&mut p, raw)?);
    }
    let bank_b = read(&mut p, raw)?;
    let sh_b = read(&mut p, raw)?;
    let turn = read(&mut p, raw)?;
    let to_move = read(&mut p, raw)?;
    Ok(State {
        order_a,
        hp_a,
        bank_a,
        sh_a,
        order_b,
        hp_b,
        bank_b,
        sh_b,
        turn,
        to_move,
    })
}

fn encode_state(s: &State) -> Vec<i32> {
    let mut out = Vec::with_capacity(2 + 2 * (s.order_a.len() + s.order_b.len()) + 8);
    out.push(s.order_a.len() as i32);
    out.extend(s.order_a.iter().map(|&x| x as i32));
    out.extend(s.hp_a.iter().copied());
    out.push(s.bank_a);
    out.push(s.sh_a);
    out.push(s.order_b.len() as i32);
    out.extend(s.order_b.iter().map(|&x| x as i32));
    out.extend(s.hp_b.iter().copied());
    out.push(s.bank_b);
    out.push(s.sh_b);
    out.push(s.turn);
    out.push(s.to_move);
    out
}

#[derive(Clone)]
struct Trunk {
    team_a: Vec<(u8, i32, i32)>, // (type, atk, max_hp_units)
    team_b: Vec<(u8, i32, i32)>,
    turn_cap: i32,
    compress: bool, // E>=6: cap candidate splits to ~10-12 macro-actions per target
}

impl Trunk {
    fn new(team_a: Vec<(u8, i32, i32)>, team_b: Vec<(u8, i32, i32)>, cap: i32, start_turn: i32) -> Self {
        Trunk {
            team_a,
            team_b,
            turn_cap: start_turn + cap,
            compress: false,
        }
    }

    fn new_compressed(team_a: Vec<(u8, i32, i32)>, team_b: Vec<(u8, i32, i32)>, cap: i32, start_turn: i32) -> Self {
        let mut t = Self::new(team_a, team_b, cap, start_turn);
        t.compress = true;
        t
    }

    /// Python round() semantics: round half to even (rules.py rounded_damage).
    fn py_round(x: f64) -> i64 {
        let f = x.floor();
        let diff = x - f;
        if diff < 0.5 {
            f as i64
        } else if diff > 0.5 {
            f as i64 + 1
        } else {
            let fi = f as i64;
            if fi % 2 == 0 {
                fi
            } else {
                fi + 1
            }
        }
    }

    /// Damage of one hit in units of 10 HP, matching rules.py per_hit_damage:
    /// ``round(atk * mult / 100) * 100`` HP == *10 in units.
    fn per_hit_units(&self, actor_side: u8, actor_id: u8, defender_side: u8, defender_id: u8) -> i32 {
        let a = if actor_side == 0 { &self.team_a } else { &self.team_b };
        let d = if actor_side == 0 { &self.team_b } else { &self.team_a };
        let atk = a[actor_id as usize].1;
        let m = multiplier(a[actor_id as usize].0, d[defender_id as usize].0);
        (Self::py_round(atk as f64 * m / 100.0) * 10) as i32
    }

    /// Total damage in units of 10 HP for ``landed`` hits (per-hit rounding).
    fn dmg_total_units(&self, actor_side: u8, actor_id: u8, defender_side: u8, defender_id: u8, landed: i32) -> i32 {
        if landed <= 0 {
            return 0;
        }
        self.per_hit_units(actor_side, actor_id, defender_side, defender_id) * landed
    }

    /// Hits-to-kill for a defender with ``hp_units`` (units of 10 HP): the
    /// smallest h with per_hit_units * h >= hp_units (exact, per-hit linear).
    fn hits_to_kill(&self, actor_side: u8, actor_id: u8, defender_side: u8, defender_id: u8, hp_units: i32) -> i32 {
        if hp_units <= 0 {
            return 1;
        }
        let ph = self.per_hit_units(actor_side, actor_id, defender_side, defender_id);
        if ph <= 0 {
            return i32::MAX;
        }
        (hp_units + ph - 1) / ph
    }

    fn budget(&self, s: &State) -> i32 {
        let own = if s.to_move == 0 { s.bank_a } else { s.bank_b };
        base(s.turn) + own
    }

    fn actions(&self, s: &State) -> Vec<Action> {
        let order = if s.to_move == 0 { &s.order_a } else { &s.order_b };
        let bgt = self.budget(s);
        let switches: Vec<Option<u8>> = std::iter::once(None).chain(order.iter().skip(1).map(|&c| Some(c))).collect();
        if self.compress {
            return self.actions_compressed(s, &switches, bgt);
        }
        let mut out = Vec::new();
        for sw in switches {
            let rem = bgt - if sw.is_some() { 1 } else { 0 };
            for a in 0..=rem {
                for d in 0..=(rem - a) {
                    let b = rem - a - d;
                    if b <= MAX_BANK {
                        out.push(Action { a, d, b, sw });
                    }
                }
            }
        }
        out
    }

    /// 3v3 action-grid compression: for each target (stay or switch k), emit a
    /// compact macro set (~10-12 actions): the three pure corners plus the
    /// attack/shield, attack/bank and shield/bank trade-off lines. Keeps the
    /// branch factor tractable for E >= 6 (full 3v3 rosters).
    fn actions_compressed(&self, s: &State, switches: &[Option<u8>], bgt: i32) -> Vec<Action> {
        let mut out = Vec::new();
        for &sw in switches {
            let e = bgt - if sw.is_some() { 1 } else { 0 };
            if e <= 0 {
                continue;
            }
            // three pure corners
            if e <= MAX_BANK {
                out.push(Action { a: 0, d: 0, b: e, sw });
            }
            out.push(Action { a: e, d: 0, b: 0, sw });
            out.push(Action { a: 0, d: e, b: 0, sw });
            // attack/defense trade-off line
            for a in [1, e / 2, e - 1] {
                if a <= 0 || a >= e {
                    continue;
                }
                out.push(Action { a, d: e - a, b: 0, sw });
            }
            // attack/bank trade-off line
            for a in [1, e / 2, e - 1] {
                if a <= 0 || a >= e || e - a > MAX_BANK {
                    continue;
                }
                out.push(Action { a, d: 0, b: e - a, sw });
            }
            // defense/bank trade-off line
            for d in [1, e / 2, e - 1] {
                if d <= 0 || d >= e || e - d > MAX_BANK {
                    continue;
                }
                out.push(Action { a: 0, d, b: e - d, sw });
            }
        }
        out
    }

    fn promote(order: &[u8], hp: &[i32], dead: u8) -> (Vec<u8>, Vec<i32>) {
        let mut no = Vec::new();
        let mut nh = Vec::new();
        for (i, &c) in order.iter().enumerate() {
            if c != dead {
                no.push(c);
                nh.push(hp[i]);
            }
        }
        (no, nh)
    }

    fn transition(&self, s: &State, act: &Action) -> Option<State> {
        let a = act.a;
        let d = act.d;
        let b = act.b;
        if s.to_move == 0 {
            let mut order_a = s.order_a.clone();
            let mut hp_a = s.hp_a.clone();
            if let Some(sw) = act.sw {
                let i = order_a.iter().position(|&c| c == sw)?;
                let o = order_a.remove(i);
                order_a.insert(0, o);
                let h = hp_a.remove(i);
                hp_a.insert(0, h);
            }
            let landed = (a - s.sh_b).max(0);
            let dmg = self.dmg_total_units(0, order_a[0], 1, s.order_b[0], landed);
            let mut order_b = s.order_b.clone();
            let mut hp_b = s.hp_b.clone();
            if landed > 0 {
                hp_b[0] = (hp_b[0] - dmg).max(0);
                if hp_b[0] <= 0 {
                    let dead = order_b[0];
                    let (no, nh) = Self::promote(&order_b, &hp_b, dead);
                    order_b = no;
                    hp_b = nh;
                }
            }
            Some(State {
                order_a,
                hp_a,
                bank_a: b.min(MAX_BANK),
                sh_a: d,
                order_b,
                hp_b,
                bank_b: s.bank_b,
                sh_b: 0,
                turn: s.turn + 1,
                to_move: 1,
            })
        } else {
            let mut order_b = s.order_b.clone();
            let mut hp_b = s.hp_b.clone();
            if let Some(sw) = act.sw {
                let i = order_b.iter().position(|&c| c == sw)?;
                let o = order_b.remove(i);
                order_b.insert(0, o);
                let h = hp_b.remove(i);
                hp_b.insert(0, h);
            }
            let landed = (a - s.sh_a).max(0);
            let dmg = self.dmg_total_units(1, order_b[0], 0, s.order_a[0], landed);
            let mut order_a = s.order_a.clone();
            let mut hp_a = s.hp_a.clone();
            if landed > 0 {
                hp_a[0] = (hp_a[0] - dmg).max(0);
                if hp_a[0] <= 0 {
                    let dead = order_a[0];
                    let (no, nh) = Self::promote(&order_a, &hp_a, dead);
                    order_a = no;
                    hp_a = nh;
                }
            }
            Some(State {
                order_a,
                hp_a,
                bank_a: s.bank_a,
                sh_a: 0,
                order_b,
                hp_b,
                bank_b: b.min(MAX_BANK),
                sh_b: d,
                turn: s.turn + 1,
                to_move: 0,
            })
        }
    }

    fn terminal(&self, s: &State) -> Option<f64> {
        if s.order_a.is_empty() {
            return Some(-1.0);
        }
        if s.order_b.is_empty() {
            return Some(1.0);
        }
        if s.turn > self.turn_cap {
            return Some(0.0);
        }
        None
    }
}

// ------------------------------------------------------------------ micro-tree
// Depth-limited FH-CFR (perfect-recall info sets) over a small tree from the
// current position. The whole loop runs in Rust; only the final root strategy
// crosses the PyO3 boundary.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

/// Canonical-belief 1v1 value table, belief-key format:
/// (hA, hB, to_move, own_bank, own_sh, R, turn) -> value, where (own_bank,
/// own_sh) is the ACTING player's own known split and R = opponent sh + bank
/// (public remainder). turn >= 7 is clamped to 7 (stationary phase 4).
/// Values are the equilibrium-belief 1v1 values (belief-dependence known).
type VKey = (i32, i32, i32, i32, i32, i32, i32);
static V1_TABLE: OnceLock<Mutex<HashMap<VKey, f64>>> = OnceLock::new();

fn v1_table() -> &'static Mutex<HashMap<VKey, f64>> {
    V1_TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Layout of the dense belief-state grid used by the 1v1 table builder.
/// A belief-state is (hA, hB, to_move, own_bank, own_sh, R); the value is the
/// belief-root equilibrium value over the opponent's (bank, sh) splits of R.
#[derive(Clone, Copy)]
struct V1Layout {
    hits_min: usize,
    hits_max: usize,
    bank_max: usize,
    sh_max: usize,
    r_max: usize,
}

impl V1Layout {
    fn nh(&self) -> usize {
        self.hits_max - self.hits_min + 1
    }
    fn nb(&self) -> usize {
        self.bank_max + 1
    }
    fn nsh(&self) -> usize {
        self.sh_max + 1
    }
    fn nr(&self) -> usize {
        self.r_max + 1
    }
    fn size(&self) -> usize {
        self.nh() * self.nh() * 2 * self.nb() * self.nsh() * self.nr()
    }
    fn index(&self, hA: i32, hB: i32, mv: i32, bank: i32, sh: i32, r: i32) -> Option<usize> {
        if !(self.hits_min as i32..=self.hits_max as i32).contains(&hA) {
            return None;
        }
        if !(self.hits_min as i32..=self.hits_max as i32).contains(&hB) {
            return None;
        }
        if !(0..=self.bank_max as i32).contains(&bank) {
            return None;
        }
        if !(0..=self.sh_max as i32).contains(&sh) {
            return None;
        }
        if !(0..=self.r_max as i32).contains(&r) {
            return None;
        }
        if !(0..=1).contains(&mv) {
            return None;
        }
        let nh = self.nh();
        let (hA, hB, mv, bank, sh, r) =
            (hA as usize, hB as usize, mv as usize, bank as usize, sh as usize, r as usize);
        let idx = ((((hA - self.hits_min) * nh + (hB - self.hits_min)) * 2 + mv) * self.nb() + bank)
            * self.nsh()
            + sh;
        Some(idx * self.nr() + r)
    }
    /// Decode a flat index into (hA, hB, to_move, own_bank, own_sh, R).
    fn decode(&self, i: usize) -> Option<(i32, i32, i32, i32, i32, i32)> {
        if i >= self.size() {
            return None;
        }
        let mut x = i;
        let r = x % self.nr();
        x /= self.nr();
        let sh = x % self.nsh();
        x /= self.nsh();
        let bk = x % self.nb();
        x /= self.nb();
        let mv = x % 2;
        x /= 2;
        let hb = x % self.nh() + self.hits_min;
        x /= self.nh();
        let ha = x + self.hits_min;
        Some((ha as i32, hb as i32, mv as i32, bk as i32, sh as i32, r as i32))
    }
}

fn info_key_flat(s: &State) -> Vec<i32> {
    // acting player's public observation: (turn, orderA, hpA, orderB, hpB, own_bank, R)
    let mut out = Vec::with_capacity(16);
    out.push(s.turn);
    out.push(s.order_a.len() as i32);
    out.extend(s.order_a.iter().map(|&x| x as i32));
    out.extend(s.hp_a.iter().copied());
    out.push(s.order_b.len() as i32);
    out.extend(s.order_b.iter().map(|&x| x as i32));
    out.extend(s.hp_b.iter().copied());
    if s.to_move == 0 {
        out.push(s.bank_a);
        out.push(s.sh_b + s.bank_b); // R
    } else {
        out.push(s.bank_b);
        out.push(s.sh_a + s.bank_a); // R
    }
    out
}

struct InfoSet {
    n_acts: usize,
    regret: Vec<f64>,
    avg: Vec<f64>,
}

struct MicroSolver {
    trunk: Trunk,
    depth: u8,
    gamma: f64,
    root_states: Vec<(State, f64)>,
    roots: Vec<(usize, f64)>,
    states: Vec<State>,
    terminals: Vec<Option<f64>>,
    depths: Vec<u8>,
    actions_of: Vec<Vec<Action>>,
    children: Vec<Vec<usize>>,
    info_id: Vec<usize>,
    parent: Vec<usize>,
    full_key: Vec<Vec<i32>>,
    infos: Vec<InfoSet>,
    tm_of: Vec<u8>,
    keep: HashMap<(u8, Vec<i32>), Vec<bool>>,
    leaf_override: HashMap<Vec<i32>, f64>,
    v1_flat: Option<Arc<Vec<f64>>>,
    v1_layout: Option<V1Layout>,
    r_a: Vec<f64>,
    r_b: Vec<f64>,
    v: Vec<f64>,
    reach_opp_sum: Vec<f64>,
    reach_self_sum: Vec<f64>,
    cfv_contrib: Vec<Vec<f64>>,
    // Regret pruning: after `prune_after` iterations, skip traversing actions
    // with non-positive regret (their regret-matching weight is 0 anyway, so
    // values are unchanged; this only skips the tree walk). Recomputed each
    // iteration so positive-regret actions always stay in.
    mask: Vec<Vec<bool>>,
    prune_after: usize,
    iter_count: usize,
}

impl MicroSolver {
    fn new(trunk: Trunk, root_states: Vec<(State, f64)>, depth: u8, gamma: f64) -> Self {
        let mut s = MicroSolver {
            trunk,
            depth,
            gamma,
            root_states,
            roots: Vec::new(),
            states: Vec::new(),
            terminals: Vec::new(),
            depths: Vec::new(),
            actions_of: Vec::new(),
            children: Vec::new(),
            info_id: Vec::new(),
            parent: Vec::new(),
            full_key: Vec::new(),
            infos: Vec::new(),
            tm_of: Vec::new(),
            keep: HashMap::new(),
            leaf_override: HashMap::new(),
            v1_flat: None,
            v1_layout: None,
            r_a: Vec::new(),
            r_b: Vec::new(),
            v: Vec::new(),
            reach_opp_sum: Vec::new(),
            reach_self_sum: Vec::new(),
            cfv_contrib: Vec::new(),
            mask: Vec::new(),
            prune_after: 0,
            iter_count: 0,
        };
        s.build();
        s
    }

    fn get_or_add_info(&mut self, map: &mut HashMap<(u8, Vec<i32>), usize>, tm: u8, full: &[i32]) -> usize {
        let key = (tm, full.to_vec());
        if let Some(&id) = map.get(&key) {
            return id;
        }
        let id = self.infos.len();
        self.infos.push(InfoSet { n_acts: 0, regret: Vec::new(), avg: Vec::new() });
        map.insert(key, id);
        id
    }

    fn build(&mut self) {
        let mut info_map: HashMap<(u8, Vec<i32>), usize> = HashMap::new();
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

        let root_states = self.root_states.iter().cloned().collect::<Vec<_>>();
        for (root, w) in root_states {
            let ik = info_key_flat(&root);
            let i = self.states.len();
            self.roots.push((i, w));
            self.states.push(root.clone());
            self.terminals.push(self.trunk.terminal(&root));
            self.depths.push(0);
            self.parent.push(usize::MAX);
            self.full_key.push(ik.clone());
            let iid = self.get_or_add_info(&mut info_map, root.to_move as u8, &ik);
            self.info_id.push(iid);
            self.actions_of.push(Vec::new());
            self.children.push(Vec::new());
            queue.push_back(i);
        }

        while let Some(i) = queue.pop_front() {
            if self.terminals[i].is_some() || self.depths[i] >= self.depth {
                continue;
            }
            let key = (self.states[i].to_move as u8, self.full_key[i].clone());
            let mut acts = self.trunk.actions(&self.states[i]);
            if let Some(mask) = self.keep.get(&key) {
                acts = acts
                    .into_iter()
                    .zip(mask.iter())
                    .filter(|(_, &k)| k)
                    .map(|(a, _)| a)
                    .collect();
            }
            self.actions_of[i] = acts.clone();
            for act in acts {
                if let Some(child) = self.trunk.transition(&self.states[i], &act) {
                    let cik = info_key_flat(&child);
                    let cid = self.states.len();
                    self.states.push(child.clone());
                    self.terminals.push(self.trunk.terminal(&child));
                    self.depths.push(self.depths[i] + 1);
                    // Perfect recall: the info set is the acting player's full
                    // observation sequence. Incrementally extend the previous
                    // own sequence (the depth-2 ancestor, since turns strictly
                    // alternate) with the current observation.
                    let mut fk = if self.depths[i] >= 1 {
                        self.full_key[self.parent[i]].clone()
                    } else {
                        Vec::new()
                    };
                    fk.extend_from_slice(&cik);
                    self.parent.push(i);
                    self.full_key.push(fk.clone());
                    let ciid = self.get_or_add_info(&mut info_map, child.to_move as u8, &fk);
                    self.info_id.push(ciid);
                    self.children[i].push(cid);
                    self.actions_of.push(Vec::new());
                    self.children.push(Vec::new());
                    queue.push_back(cid);
                }
            }
        }

        // finalize info sets: n_acts and to_move from the first node (single O(n) pass),
        // then allocate arrays
        let n = self.states.len();
        self.tm_of = vec![255; self.infos.len()];
        for i in 0..n {
            let iid = self.info_id[i];
            if self.tm_of[iid] == 255 {
                self.tm_of[iid] = self.states[i].to_move as u8;
                self.infos[iid].n_acts = self.actions_of[i].len();
            }
        }
        for iid in 0..self.infos.len() {
            let na = self.infos[iid].n_acts;
            self.infos[iid].regret = vec![0.0; na];
            self.infos[iid].avg = vec![0.0; na];
        }
        self.r_a = vec![0.0; n];
        self.r_b = vec![0.0; n];
        self.v = vec![0.0; n];
        let n_infos = self.infos.len();
        let max_acts = self.infos.iter().map(|x| x.n_acts).max().unwrap_or(1);
        self.reach_opp_sum = vec![0.0; n_infos];
        self.reach_self_sum = vec![0.0; n_infos];
        self.cfv_contrib = vec![vec![0.0; max_acts]; n_infos];
        self.mask = vec![vec![true; max_acts]; n_infos];
    }

    fn leaf_value(&self, i: usize) -> f64 {
        let s = &self.states[i];
        if s.order_a.is_empty() {
            return -1.0;
        }
        if s.order_b.is_empty() {
            return 1.0;
        }
        // Exact 1v1 value: dense grid (table builder) or the loaded belief table.
        if s.order_a.len() == 1 && s.order_b.len() == 1 {
            if let Some(v) = self.v1_leaf_value(s) {
                return v;
            }
            if let Some(key) = self.v1_belief_key(s) {
                if let Ok(guard) = v1_table().lock() {
                    if let Some(&v) = guard.get(&key) {
                        return v;
                    }
                }
            }
        }
        // Learned (belief-conditioned) value injected from Python: exact-HP
        // 2v2/3v3 leaves are evaluated by the value network, keyed by the
        // flat state encoding.
        if let Some(&v) = self.leaf_override.get(&encode_state(s)) {
            return v;
        }
        // Material fallback (2v2/3v3 leaves or a table miss). Bodies are the
        // decisive quantity in 3v3 (a one-body lead is nearly a win), so they
        // outweigh the HP term: an EV-neutral body-for-body trade must not look
        // like standing still, or CFR retreats into a shield turtle.
        let ha: i32 = s.hp_a.iter().sum();
        let hb: i32 = s.hp_b.iter().sum();
        let material = (ha - hb) as f64 / 6000.0;
        let bodies = (s.order_a.len() as f64 - s.order_b.len() as f64) * 0.35;
        (material + bodies).clamp(-1.0, 1.0)
    }

    /// Canonical-belief 1v1 key of a 1v1 state: (hA, hB, mover, own bank,
    /// own sh, R, turn-clamped). None unless the game is a 1v1 duel.
    fn v1_belief_key(&self, s: &State) -> Option<VKey> {
        if s.order_a.len() != 1 || s.order_b.len() != 1 {
            return None;
        }
        let da = self.trunk.dmg_total_units(0, s.order_a[0], 1, s.order_b[0], 1);
        let db = self.trunk.dmg_total_units(1, s.order_b[0], 0, s.order_a[0], 1);
        if da <= 0 || db <= 0 {
            return None;
        }
        let hb = self.trunk.hits_to_kill(0, s.order_a[0], 1, s.order_b[0], s.hp_b[0]);
        let ha = self.trunk.hits_to_kill(1, s.order_b[0], 0, s.order_a[0], s.hp_a[0]);
        let mv = s.to_move;
        let bank = if mv == 0 { s.bank_a } else { s.bank_b };
        let sh = if mv == 0 { s.sh_a } else { s.sh_b };
        let r = if mv == 0 { s.sh_b + s.bank_b } else { s.sh_a + s.bank_a };
        let t = if s.turn >= 7 { 7 } else { s.turn };
        Some((ha.max(1), hb.max(1), mv, bank, sh, r, t))
    }

    /// Dense-grid 1v1 value (table-builder leaf path), turn-invariant.
    fn v1_leaf_value(&self, s: &State) -> Option<f64> {
        let flat = self.v1_flat.as_ref()?;
        let layout = self.v1_layout.as_ref()?;
        if s.order_a.len() != 1 || s.order_b.len() != 1 {
            return None;
        }
        let da = self.trunk.dmg_total_units(0, s.order_a[0], 1, s.order_b[0], 1);
        let db = self.trunk.dmg_total_units(1, s.order_b[0], 0, s.order_a[0], 1);
        if da <= 0 || db <= 0 {
            return None;
        }
        let hb = self.trunk.hits_to_kill(0, s.order_a[0], 1, s.order_b[0], s.hp_b[0]);
        let ha = self.trunk.hits_to_kill(1, s.order_b[0], 0, s.order_a[0], s.hp_a[0]);
        let mv = s.to_move;
        let bank = if mv == 0 { s.bank_a } else { s.bank_b };
        let sh = if mv == 0 { s.sh_a } else { s.sh_b };
        let r = if mv == 0 { s.sh_b + s.bank_b } else { s.sh_a + s.bank_a };
        let idx = layout.index(ha.max(1), hb.max(1), mv, bank, sh, r)?;
        flat.get(idx).copied()
    }

    fn sigs(&self) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(self.infos.len());
        for info in &self.infos {
            let na = info.n_acts;
            let mut sig = vec![0.0; na];
            let pos: Vec<f64> = info.regret.iter().map(|r| r.max(0.0)).collect();
            let total: f64 = pos.iter().sum();
            if total > 0.0 {
                for (k, v) in sig.iter_mut().enumerate() {
                    *v = pos[k] / total;
                }
            } else {
                for v in sig.iter_mut() {
                    *v = 1.0 / na as f64;
                }
            }
            out.push(sig);
        }
        out
    }

    fn iterate(&mut self) {
        let n = self.states.len();
        let sigs = self.sigs();
        self.iter_count += 1;
        let do_prune = self.prune_after > 0 && self.iter_count > self.prune_after;
        if do_prune {
            for iid in 0..self.infos.len() {
                let na = self.infos[iid].n_acts;
                let mut any = false;
                for a in 0..na {
                    let keep = self.infos[iid].regret[a] > 0.0;
                    self.mask[iid][a] = keep;
                    any |= keep;
                }
                if !any {
                    for a in 0..na {
                        self.mask[iid][a] = true;
                    }
                }
            }
        }
        // reach
        self.r_a.fill(0.0);
        self.r_b.fill(0.0);
        for &(ri, w) in &self.roots {
            if self.states[ri].to_move == 0 {
                self.r_a[ri] = 1.0;
                self.r_b[ri] = w;
            } else {
                self.r_b[ri] = 1.0;
                self.r_a[ri] = w;
            }
        }
        for i in 0..n {
            if self.terminals[i].is_some() || self.depths[i] >= self.depth {
                continue;
            }
            let iid = self.info_id[i];
            let sig = &sigs[iid];
            let m = if do_prune { Some(&self.mask[iid]) } else { None };
            for (a, &ci) in self.children[i].iter().enumerate() {
                if let Some(mm) = m {
                    if !mm[a] {
                        continue;
                    }
                }
                let p = sig[a];
                if self.states[i].to_move == 0 {
                    self.r_a[ci] += self.r_a[i] * p;
                    self.r_b[ci] += self.r_b[i];
                } else {
                    self.r_b[ci] += self.r_b[i] * p;
                    self.r_a[ci] += self.r_a[i];
                }
            }
        }
        // value (children have higher ids)
        for i in (0..n).rev() {
            if let Some(t) = self.terminals[i] {
                self.v[i] = t;
            } else if self.depths[i] >= self.depth {
                self.v[i] = self.leaf_value(i);
            } else {
                let iid = self.info_id[i];
                let sig = &sigs[iid];
                let m = if do_prune { Some(&self.mask[iid]) } else { None };
                let mut acc = 0.0;
                for (a, &ci) in self.children[i].iter().enumerate() {
                    if let Some(mm) = m {
                        if !mm[a] {
                            continue;
                        }
                    }
                    acc += sig[a] * self.v[ci];
                }
                self.v[i] = self.gamma * acc;
            }
        }
        // cfv accumulation
        self.reach_opp_sum.fill(0.0);
        self.reach_self_sum.fill(0.0);
        for row in self.cfv_contrib.iter_mut() {
            row.fill(0.0);
        }
        for i in 0..n {
            if self.terminals[i].is_some() || self.depths[i] >= self.depth {
                continue;
            }
            let iid = self.info_id[i];
            let tm = self.states[i].to_move;
            let opp = if tm == 0 { self.r_b[i] } else { self.r_a[i] };
            let slf = if tm == 0 { self.r_a[i] } else { self.r_b[i] };
            self.reach_opp_sum[iid] += opp;
            self.reach_self_sum[iid] += slf;
            let m = if do_prune { Some(&self.mask[iid]) } else { None };
            for (a, &ci) in self.children[i].iter().enumerate() {
                if let Some(mm) = m {
                    if !mm[a] {
                        continue;
                    }
                }
                self.cfv_contrib[iid][a] += opp * self.v[ci];
            }
        }
        // regret / avg update (CFR+)
        for iid in 0..self.infos.len() {
            let na = self.infos[iid].n_acts;
            let rsum = self.reach_opp_sum[iid];
            let mut v_info = 0.0;
            for a in 0..na {
                let cfv = if rsum > 0.0 { self.cfv_contrib[iid][a] / rsum } else { 0.0 };
                v_info += sigs[iid][a] * cfv;
            }
            // sign: A maximizes (+), B minimizes (-)
            let a_move = self.a_move_of_info(iid);
            let sign = if a_move { 1.0 } else { -1.0 };
            for a in 0..na {
                let cfv = if rsum > 0.0 { self.cfv_contrib[iid][a] / rsum } else { 0.0 };
                let delta = rsum * sign * (cfv - v_info);
                let r = self.infos[iid].regret[a] + delta;
                self.infos[iid].regret[a] = r.max(0.0);
                self.infos[iid].avg[a] += self.reach_self_sum[iid] * sigs[iid][a];
            }
        }
    }

    fn a_move_of_info(&self, iid: usize) -> bool {
        self.tm_of[iid] == 0
    }

    /// Average-profile value of the tree: the expected payoff when both
    /// players follow their AVERAGE strategy (``avg``), not the current
    /// regret-matching strategy. CFR's current strategy oscillates while the
    /// average converges, so using ``self.v`` (computed from the current
    /// strategy) makes the reported value unstable.
    fn avg_profile_value(&self) -> f64 {
        let n = self.states.len();
        let mut sigs: Vec<Vec<f64>> = Vec::with_capacity(self.infos.len());
        for info in &self.infos {
            let na = info.n_acts;
            let total: f64 = info.avg.iter().sum();
            if total > 0.0 {
                sigs.push(info.avg.iter().map(|x| x / total).collect());
            } else {
                sigs.push(vec![1.0 / na as f64; na]);
            }
        }
        let mut val = vec![0.0f64; n];
        for i in (0..n).rev() {
            if let Some(t) = self.terminals[i] {
                val[i] = t;
            } else if self.depths[i] >= self.depth {
                val[i] = self.leaf_value(i);
            } else {
                let sig = &sigs[self.info_id[i]];
                let mut acc = 0.0;
                for (a, &ci) in self.children[i].iter().enumerate() {
                    acc += sig[a] * val[ci];
                }
                val[i] = self.gamma * acc;
            }
        }
        let mut value = 0.0;
        for &(ri, w) in &self.roots {
            value += w * val[ri];
        }
        value
    }

    fn root_strategy(&self) -> (Vec<f64>, f64) {
        let iid = self.info_id[self.roots[0].0];
        let na = self.infos[iid].n_acts;
        let avg = &self.infos[iid].avg;
        let total: f64 = avg.iter().sum();
        let probs: Vec<f64> = if total > 0.0 {
            avg.iter().map(|x| x / total).collect()
        } else {
            vec![1.0 / na as f64; na]
        };
        (probs, self.avg_profile_value())
    }

    /// Opponent's average-strategy reach over their NEXT hidden split
    /// (shields, bank), set by their first decision in this subgame. This is
    /// the Continuum/DeepStack reach that becomes the belief prior at the next
    /// resolve (Continual Subgame Resolving). Computed under the AVERAGE
    /// strategy of both players; the opponent's first action determines their
    /// new split (defends -> shields, bonuses -> bank).
    fn opponent_reach(&self) -> HashMap<(i32, i32), f64> {
        let n = self.states.len();
        let mut sigs: Vec<Vec<f64>> = Vec::with_capacity(self.infos.len());
        for info in &self.infos {
            let na = info.n_acts;
            let total: f64 = info.avg.iter().sum();
            if total > 0.0 {
                sigs.push(info.avg.iter().map(|x| x / total).collect());
            } else {
                sigs.push(vec![1.0 / na as f64; na]);
            }
        }
        let mut ra = vec![0.0f64; n];
        let mut rb = vec![0.0f64; n];
        for &(ri, w) in &self.roots {
            if self.states[ri].to_move == 0 {
                ra[ri] = 1.0;
                rb[ri] = w;
            } else {
                rb[ri] = 1.0;
                ra[ri] = w;
            }
        }
        for i in 0..n {
            if self.terminals[i].is_some() || self.depths[i] >= self.depth {
                continue;
            }
            let sig = &sigs[self.info_id[i]];
            for (a, &ci) in self.children[i].iter().enumerate() {
                let p = sig[a];
                if self.states[i].to_move == 0 {
                    ra[ci] += ra[i] * p;
                    rb[ci] += rb[i];
                } else {
                    rb[ci] += rb[i] * p;
                    ra[ci] += ra[i];
                }
            }
        }
        let mut out: HashMap<(i32, i32), f64> = HashMap::new();
        for i in 0..n {
            if self.depths[i] != 1 || self.terminals[i].is_some()
                || self.depths[i] >= self.depth {
                continue;
            }
            let tm = self.states[i].to_move;
            let (self_r, opp_r) = if tm == 0 { (ra[i], rb[i]) } else { (rb[i], ra[i]) };
            let w_line = self_r * opp_r;
            if w_line <= 0.0 {
                continue;
            }
            let sig = &sigs[self.info_id[i]];
            for (a, &ci) in self.children[i].iter().enumerate() {
                let st = &self.states[ci];
                let (sh_new, bank_new) =
                    if tm == 0 { (st.sh_a, st.bank_a) } else { (st.sh_b, st.bank_b) };
                *out.entry((sh_new, bank_new)).or_insert(0.0) += w_line * sig[a];
            }
        }
        out
    }

    fn rebuild(&mut self) {
        self.roots.clear();
        self.states.clear();
        self.terminals.clear();
        self.depths.clear();
        self.actions_of.clear();
        self.children.clear();
        self.info_id.clear();
        self.parent.clear();
        self.full_key.clear();
        self.infos.clear();
        self.tm_of.clear();
        self.r_a.clear();
        self.r_b.clear();
        self.v.clear();
        self.reach_opp_sum.clear();
        self.reach_self_sum.clear();
        self.cfv_contrib.clear();
        self.mask.clear();
        self.iter_count = 0;
        self.build();
    }

    /// Keep only the actions the average strategy actually plays, at most
    /// max_acts per info set: always keep the top `always_keep` actions, plus
    /// any action with weight >= eps. Returns true if any info set changed.
    fn prune_support(&mut self, eps: f64, max_acts: usize, always_keep: usize) -> bool {
        let mut changed = false;
        let mut seen = vec![false; self.infos.len()];
        for i in 0..self.states.len() {
            let iid = self.info_id[i];
            if seen[iid] {
                continue;
            }
            seen[iid] = true;
            let na = self.infos[iid].n_acts;
            if na == 0 {
                continue;
            }
            let avg = &self.infos[iid].avg;
            let total: f64 = avg.iter().sum();
            let mut w: Vec<(f64, usize)> = if total > 0.0 {
                avg.iter().enumerate().map(|(a, &x)| (x / total, a)).collect()
            } else {
                (0..na).map(|a| (1.0 / na as f64, a)).collect()
            };
            w.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let mut mask = vec![false; na];
            let mut kept = 0usize;
            for &(weight, a) in &w {
                if kept >= max_acts {
                    break;
                }
                if kept < always_keep || weight >= eps {
                    mask[a] = true;
                    kept += 1;
                }
            }
            let key = (self.tm_of[iid] as u8, self.full_key[i].clone());
            if let Some(old) = self.keep.get(&key) {
                if *old != mask {
                    changed = true;
                }
            } else {
                changed = true;
            }
            self.keep.insert(key, mask);
        }
        changed
    }

    /// CFR loop: burn-in on the full tree, then prune+rebuild rounds that
    /// concentrate the tree on the strategy support, refining each round.
    fn run(&mut self, iters: usize, burn_in: usize, eps: f64, max_acts: usize, always_keep: usize) {
        let burn = burn_in.min(iters);
        for _ in 0..burn {
            self.iterate();
        }
        let mut remaining = iters - burn;
        for round in 0..3 {
            if remaining == 0 {
                break;
            }
            if !self.prune_support(eps, max_acts, always_keep) {
                break;
            }
            self.rebuild();
            let per = (remaining / (3 - round)).max(1);
            for _ in 0..per {
                self.iterate();
            }
            remaining -= per;
        }
        for _ in 0..remaining {
            self.iterate();
        }
    }
}

// ------------------------------------------------------------------ PyO3

#[pyclass]
struct PyTrunk {
    inner: Trunk,
}

#[pymethods]
impl PyTrunk {
    #[new]
    #[pyo3(signature = (team_a, team_b, cap=20, start_turn=1))]
    fn new(team_a: Vec<(u8, i32, i32)>, team_b: Vec<(u8, i32, i32)>, cap: i32, start_turn: i32) -> Self {
        PyTrunk {
            inner: Trunk::new(team_a, team_b, cap, start_turn),
        }
    }

    fn transition(&self, state: Vec<i32>, action: Vec<i32>) -> PyResult<Option<Vec<i32>>> {
        if action.len() < 4 {
            return Err(PyValueError::new_err("action must be [a, d, b, sw]"));
        }
        let st = decode_state(&state)?;
        let act = Action {
            a: action[0],
            d: action[1],
            b: action[2],
            sw: if action[3] < 0 { None } else { Some(action[3] as u8) },
        };
        Ok(self.inner.transition(&st, &act).map(|s| encode_state(&s)))
    }

    fn actions(&self, state: Vec<i32>) -> PyResult<Vec<Vec<i32>>> {
        let st = decode_state(&state)?;
        Ok(self
            .inner
            .actions(&st)
            .iter()
            .map(|ac| vec![ac.a, ac.d, ac.b, ac.sw.map(|x| x as i32).unwrap_or(-1)])
            .collect())
    }

    fn terminal(&self, state: Vec<i32>) -> PyResult<Option<f64>> {
        let st = decode_state(&state)?;
        Ok(self.inner.terminal(&st))
    }

    /// Active matchup hits-to-kill: (hits A needs to kill B's active, hits B needs to kill A's active).
    fn hits(&self, state: Vec<i32>) -> PyResult<(i32, i32)> {
        let st = decode_state(&state)?;
        let hb = self.inner.hits_to_kill(0, st.order_a[0], 1, st.order_b[0], st.hp_b[0]);
        let ha = self.inner.hits_to_kill(1, st.order_b[0], 0, st.order_a[0], st.hp_a[0]);
        Ok((ha.max(1), hb.max(1)))
    }
}

fn solve_micro_impl(
    team_a: Vec<(u8, i32, i32)>,
    team_b: Vec<(u8, i32, i32)>,
    root_states: Vec<(Vec<i32>, f64)>,
    depth: u8,
    iters: usize,
    gamma: f64,
    cap: i32,
    start_turn: i32,
    prune: bool,
    prune_after: usize,
) -> PyResult<(Vec<Vec<i32>>, Vec<f64>, f64)> {
    let mut roots = Vec::with_capacity(root_states.len());
    let mut wsum = 0.0;
    for (raw, w) in root_states {
        roots.push((decode_state(&raw)?, w));
        wsum += w;
    }
    if roots.is_empty() {
        return Err(PyValueError::new_err("no root states"));
    }
    if wsum <= 0.0 {
        return Err(PyValueError::new_err("root weights must sum to > 0"));
    }
    for (_, w) in roots.iter_mut() {
        *w /= wsum;
    }
    let trunk = Trunk::new(team_a, team_b, cap, start_turn);
    let mut solver = MicroSolver::new(trunk, roots, depth, gamma);
    solver.prune_after = prune_after;
    if prune {
        // Burn-in on the full tree long enough for the strategy support to
        // mature before pruning. NOTE: on depth-3+ trees this needs thousands
        // of iterations to be safe; the default (prune=false) runs the full
        // tree for all iterations instead.
        let burn_in = ((iters / 2).max(40)).min(iters);
        solver.run(iters, burn_in, 0.001, 10, 4);
    } else {
        solver.run(iters, iters, 0.001, 10, 4);
    }
    let acts = &solver.actions_of[solver.roots[0].0];
    let (probs, value) = solver.root_strategy();
    let actions: Vec<Vec<i32>> = acts
        .iter()
        .map(|ac| vec![ac.a, ac.d, ac.b, ac.sw.map(|x| x as i32).unwrap_or(-1)])
        .collect();
    Ok((actions, probs, value))
}

#[pyfunction]
#[pyo3(signature = (team_a, team_b, state, depth=4, iters=500, gamma=0.995, cap=20, start_turn=1, prune=false, prune_after=0))]
#[allow(clippy::too_many_arguments)]
fn solve_micro(
    team_a: Vec<(u8, i32, i32)>,
    team_b: Vec<(u8, i32, i32)>,
    state: Vec<i32>,
    depth: u8,
    iters: usize,
    gamma: f64,
    cap: i32,
    start_turn: i32,
    prune: bool,
    prune_after: usize,
) -> PyResult<(Vec<Vec<i32>>, Vec<f64>, f64)> {
    solve_micro_impl(team_a, team_b, vec![(state, 1.0)], depth, iters, gamma, cap, start_turn, prune, prune_after)
}

/// Belief-weighted root: the acting player's info set is shared across all root
/// states (same public observation), the opponent's "reach" into each is the
/// belief weight. The returned value is the belief-weighted average.
#[pyfunction]
#[pyo3(signature = (team_a, team_b, states, depth=4, iters=500, gamma=0.995, cap=20, start_turn=1, prune=false, prune_after=0))]
#[allow(clippy::too_many_arguments)]
fn solve_micro_belief(
    team_a: Vec<(u8, i32, i32)>,
    team_b: Vec<(u8, i32, i32)>,
    states: Vec<(Vec<i32>, f64)>,
    depth: u8,
    iters: usize,
    gamma: f64,
    cap: i32,
    start_turn: i32,
    prune: bool,
    prune_after: usize,
) -> PyResult<(Vec<Vec<i32>>, Vec<f64>, f64)> {
    solve_micro_impl(team_a, team_b, states, depth, iters, gamma, cap, start_turn, prune, prune_after)
}

/// Depth-limited subgame solver with pluggable leaf values: the leaves are
/// exposed to Python so a value network can evaluate them, then `solve` runs
/// CFR with those learned values (belief-conditioned, DeepStack/ReBeL style).
#[pyclass]
struct MicroTree {
    inner: MicroSolver,
}

#[pymethods]
impl MicroTree {
    #[new]
    #[pyo3(signature = (team_a, team_b, states, depth=3, cap=6, start_turn=1, compress=false))]
    fn new(
        team_a: Vec<(u8, i32, i32)>,
        team_b: Vec<(u8, i32, i32)>,
        states: Vec<(Vec<i32>, f64)>,
        depth: u8,
        cap: i32,
        start_turn: i32,
        compress: bool,
    ) -> PyResult<Self> {
        let mut roots = Vec::with_capacity(states.len());
        let mut wsum = 0.0;
        for (raw, w) in states {
            roots.push((decode_state(&raw)?, w));
            wsum += w;
        }
        if roots.is_empty() {
            return Err(PyValueError::new_err("no root states"));
        }
        if wsum <= 0.0 {
            return Err(PyValueError::new_err("root weights must sum to > 0"));
        }
        for (_, w) in roots.iter_mut() {
            *w /= wsum;
        }
        let trunk = if compress {
            Trunk::new_compressed(team_a, team_b, cap, start_turn)
        } else {
            Trunk::new(team_a, team_b, cap, start_turn)
        };
        Ok(MicroTree { inner: MicroSolver::new(trunk, roots, depth, 0.995) })
    }

    fn node_count(&self) -> usize {
        self.inner.states.len()
    }

    /// Flat-encoded states of all depth-limited non-terminal leaves (the nodes
    /// whose value the search would need). 1v1 leaves are excluded here: the
    /// engine's exact table already covers them.
    fn leaf_states(&self) -> Vec<Vec<i32>> {
        let n = self.inner.states.len();
        let depth = self.inner.depth;
        let mut out = Vec::new();
        for i in 0..n {
            if self.inner.depths[i] == depth && self.inner.terminals[i].is_none() {
                let s = &self.inner.states[i];
                if s.order_a.len() > 1 || s.order_b.len() > 1 {
                    out.push(encode_state(s));
                }
            }
        }
        out
    }

    /// Run CFR with the given leaf values (flat state -> value).
    #[pyo3(signature = (iters, gamma=0.995, override_=vec![], prune_after=0))]
    fn solve(&mut self, iters: usize, gamma: f64, override_: Vec<(Vec<i32>, f64)>, prune_after: usize) {
        self.inner.gamma = gamma;
        self.inner.prune_after = prune_after;
        self.inner.leaf_override.clear();
        for (k, v) in override_ {
            self.inner.leaf_override.insert(k, v);
        }
        for _ in 0..iters {
            self.inner.iterate();
        }
    }

    fn strategy(&self) -> (Vec<Vec<i32>>, Vec<f64>, f64) {
        let acts = &self.inner.actions_of[self.inner.roots[0].0];
        let (probs, value) = self.inner.root_strategy();
        let actions: Vec<Vec<i32>> = acts
            .iter()
            .map(|ac| vec![ac.a, ac.d, ac.b, ac.sw.map(|x| x as i32).unwrap_or(-1)])
            .collect();
        (actions, probs, value)
    }

    /// Opponent reach over their next (shields, bank) split under the average
    /// strategy - the belief prior for Continual Subgame Resolving.
    fn opponent_reach(&self) -> HashMap<(i32, i32), f64> {
        self.inner.opponent_reach()
    }
}

/// Build the micro-tree without iterating; returns (n_nodes, n_infos).
#[pyfunction]
#[pyo3(signature = (team_a, team_b, state, depth=4, cap=20, start_turn=1))]
fn micro_stats(
    team_a: Vec<(u8, i32, i32)>,
    team_b: Vec<(u8, i32, i32)>,
    state: Vec<i32>,
    depth: u8,
    cap: i32,
    start_turn: i32,
) -> PyResult<(usize, usize)> {
    let root = decode_state(&state)?;
    let trunk = Trunk::new(team_a, team_b, cap, start_turn);
    let solver = MicroSolver::new(trunk, vec![(root, 1.0)], depth, 0.995);
    Ok((solver.states.len(), solver.infos.len()))
}

#[pyfunction]
fn load_1v1_table(path: String) -> PyResult<usize> {
    let text = std::fs::read_to_string(&path)
        .map_err(|e| PyValueError::new_err(format!("cannot read {}: {}", path, e)))?;
    let mut guard = v1_table()
        .lock()
        .map_err(|_| PyValueError::new_err("table lock poisoned"))?;
    guard.clear();
    let mut count = 0usize;
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 8 {
            continue;
        }
        let parse = |s: &str| -> i32 { s.trim().parse::<i32>().unwrap_or(0) };
        let key = (
            parse(f[0]),
            parse(f[1]),
            parse(f[2]),
            parse(f[3]),
            parse(f[4]),
            parse(f[5]),
            parse(f[6]),
        );
        let val = f[7].trim().parse::<f64>().unwrap_or(0.0);
        guard.insert(key, val);
        count += 1;
    }
    Ok(count)
}

/// One backward-induction step over a slice [start, end) of the dense
/// belief-state grid: for every (hA, hB, mover, own_bank, own_sh, R), build a
/// depth-2 belief-root micro-tree (roots = the opponent's (bank, sh) splits of
/// R, uniform), solve it with CFR, and store the belief-root value. Leaves are
/// evaluated from `leaf_flat` (same dense layout). Parallelism is provided by
/// the caller (processes); this function is single-threaded. Returns
/// (slice_values, max_abs_delta within the slice).
#[pyfunction]
#[pyo3(signature = (leaf_flat, start=0, end=0, root_turn=7, hits_min=1, hits_max=16,
                    bank_max=4, sh_max=8, r_max=8, gamma=1.0, solve_iters=200,
                    solve_depth=2))]
fn solve_1v1_step(
    py: Python<'_>,
    leaf_flat: Vec<f64>,
    start: usize,
    end: usize,
    root_turn: i32,
    hits_min: usize,
    hits_max: usize,
    bank_max: usize,
    sh_max: usize,
    r_max: usize,
    gamma: f64,
    solve_iters: usize,
    solve_depth: u8,
) -> PyResult<(Vec<f64>, f64)> {
    let layout = V1Layout { hits_min, hits_max, bank_max, sh_max, r_max };
    if leaf_flat.len() != layout.size() {
        return Err(PyValueError::new_err(format!(
            "leaf table size {} != expected {}",
            leaf_flat.len(),
            layout.size()
        )));
    }
    if !(1..=100).contains(&root_turn) {
        return Err(PyValueError::new_err("root_turn out of range"));
    }
    if !(1..=16).contains(&solve_depth) {
        return Err(PyValueError::new_err("solve_depth out of range (1..=16)"));
    }
    let n = layout.size();
    let end = if end == 0 { n } else { end.min(n) };
    let start = start.min(end);
    if start >= end {
        return Ok((Vec::new(), 0.0));
    }
    let team_a = vec![(0u8, 2000i32, (hits_max as i32) * 200)];
    let team_b = vec![(0u8, 2000i32, (hits_max as i32) * 200)];
    let trunk = Trunk::new(team_a, team_b, 30, root_turn);
    let leaf = Arc::new(leaf_flat);

    let (out, max_delta) = py
        .allow_threads(move || -> Result<(Vec<f64>, f64), String> {
            let mut out = vec![0.0f64; end - start];
            for (k, i) in (start..end).enumerate() {
                let (hA, hB, mv, bk, sh, r) = layout.decode(i).expect("in-range index");
                let roots = build_belief_roots(&layout, &trunk, &leaf, hA, hB, mv, bk, sh, r,
                                               root_turn, gamma, solve_iters);
                let mut ms = MicroSolver::new(trunk.clone(), roots, solve_depth, gamma);
                ms.v1_flat = Some(Arc::clone(&leaf));
                ms.v1_layout = Some(layout);
                for _ in 0..solve_iters {
                    ms.iterate();
                }
                let (_, v) = ms.root_strategy();
                out[k] = v;
            }
            let max_delta = out
                .iter()
                .enumerate()
                .map(|(k, a)| (a - leaf[start + k]).abs())
                .fold(0.0f64, f64::max);
            Ok((out, max_delta))
        })
        .map_err(PyValueError::new_err)?;
    Ok((out, max_delta))
}

/// Belief-root states for one belief-state: all (bank_opp, sh_opp) with
/// bank_opp + sh_opp == r, weighted by a MAX-ENTROPY UNIFORM prior: every
/// legal split gets exactly 1/(R+1). No type hypotheses, no desperation, no
/// magic weights - identical to the runtime prior in ``infoset.OpponentModel``
/// so table builder and resolver believe the same thing.
fn build_belief_roots(
    layout: &V1Layout,
    _trunk: &Trunk,
    _leaf: &Arc<Vec<f64>>,
    hA: i32,
    hB: i32,
    mv: i32,
    bk: i32,
    sh: i32,
    r: i32,
    turn: i32,
    _gamma: f64,
    _solve_iters: usize,
) -> Vec<(State, f64)> {
    let bmin = (r - layout.sh_max as i32).max(0);
    let bmax = r.min(layout.bank_max as i32);
    let n = (bmax - bmin + 1) as f64;
    let mut roots = Vec::with_capacity((bmax - bmin + 1) as usize);
    let w = if r == 0 { 1.0 } else { 1.0 / n };

    for bank_opp in bmin..=bmax {
        let sh_opp = r - bank_opp;
        let state = if mv == 0 {
            State {
                order_a: vec![0],
                hp_a: vec![hA * 200],
                bank_a: bk,
                sh_a: sh,
                order_b: vec![0],
                hp_b: vec![hB * 200],
                bank_b: bank_opp,
                sh_b: sh_opp,
                turn,
                to_move: 0,
            }
        } else {
            State {
                order_a: vec![0],
                hp_a: vec![hA * 200],
                bank_a: bank_opp,
                sh_a: sh_opp,
                order_b: vec![0],
                hp_b: vec![hB * 200],
                bank_b: bk,
                sh_b: sh,
                turn,
                to_move: 1,
            }
        };
        roots.push((state, w));
    }
    roots
}

#[pymodule]
fn _cote_cfr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrunk>()?;
    m.add_class::<MicroTree>()?;
    m.add_function(wrap_pyfunction!(solve_micro, m)?)?;
    m.add_function(wrap_pyfunction!(solve_micro_belief, m)?)?;
    m.add_function(wrap_pyfunction!(micro_stats, m)?)?;
    m.add_function(wrap_pyfunction!(load_1v1_table, m)?)?;
    m.add_function(wrap_pyfunction!(solve_1v1_step, m)?)?;
    Ok(())
}
