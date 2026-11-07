"""
Advanced RL Debugging Challenge
===============================
Problem  : "Pit-Walk" stochastic grid world (cliff + mid-grid pits, 10% action slip,
           time limit).  A tabular TD agent (Q-learning / SARSA) must learn to
           reach the goal while avoiding the pits.

Every bug is a switch in `BUGS`.  An empty set gives the CORRECT implementation,
so each bug can be injected alone (ablation) or all together (the "faulty" agent).

Bug catalogue (id -> category)
------------------------------
B1 state_repr : State/action representation error      idx = r*ROWS + c   (should be r*COLS + c)
B2 reward     : Reward design error                    living reward +0.5 instead of -1
B3 terminal   : Terminal-state handling error          bootstrap from auto-reset start state at episode end
B4 td_target  : TD target/update error                 target uses max Q(s) instead of max Q(s')
B5 gamma      : Discount-factor error                  gamma = 0.5 instead of 0.99
B6 alpha      : Learning-rate error                    alpha = 1.0 instead of 0.1 (stochastic env)
B7 epsilon    : Exploration-exploitation error         eps decays *per step* by 0.9 -> greedy after ~50 steps
B8 tiebreak   : Policy/action-selection error          np.argmax always picks lowest index on ties
"""
import random
import sys
import numpy as np

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
ROWS, COLS = 5, 8
START, GOAL = (4, 0), (4, 7)
PITS = {(4, c) for c in range(1, 7)} | {(1, 3), (2, 3)}
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
ACTION_NAMES = ["up", "right", "down", "left"]
N_ACTIONS = 4
GOAL_R, PIT_R, STEP_R = 20.0, -50.0, -1.0


class PitWalk:
    def __init__(self, rng, slip=0.1, max_steps=100):
        self.rng, self.slip, self.max_steps = rng, slip, max_steps
        self.pos, self.t = START, 0

    def reset(self):
        self.pos, self.t = START, 0
        return self.pos

    def step(self, a):
        if self.rng.random() < self.slip:
            a = self.rng.randrange(N_ACTIONS)
        dr, dc = MOVES[a]
        r = min(max(self.pos[0] + dr, 0), ROWS - 1)
        c = min(max(self.pos[1] + dc, 0), COLS - 1)
        self.pos = (r, c)
        self.t += 1
        if self.pos in PITS:
            return self.pos, "pit", True, False
        if self.pos == GOAL:
            return self.pos, "goal", True, False
        return self.pos, "step", False, self.t >= self.max_steps


TRUE_REWARD = {"goal": GOAL_R, "pit": PIT_R, "step": STEP_R}

ALL_BUGS = ["state_repr", "reward", "terminal", "td_target",
            "gamma", "alpha", "epsilon", "tiebreak"]
BUG_LABEL = {
    "state_repr": "B1 state repr", "reward": "B2 reward", "terminal": "B3 terminal",
    "td_target": "B4 TD target", "gamma": "B5 gamma", "alpha": "B6 alpha",
    "epsilon": "B7 epsilon", "tiebreak": "B8 tie-break",
}


def state_index(pos, bugs):
    r, c = pos
    return r * ROWS + c if "state_repr" in bugs else r * COLS + c


# ----------------------------------------------------------------------------
# Agent / training loop
# ----------------------------------------------------------------------------
def greedy(q, rng, bugs):
    if "tiebreak" in bugs:
        return int(np.argmax(q))
    best = np.flatnonzero(q == q.max())
    return int(best[rng.randrange(len(best))])


def select(q, eps, rng, bugs):
    if rng.random() < eps:
        return rng.randrange(N_ACTIONS)
    return greedy(q, rng, bugs)


def train(bugs=frozenset(), algo="qlearning", episodes=500, seed=0):
    bugs = frozenset(bugs)
    rng = random.Random(seed)
    env = PitWalk(rng)
    Q = np.zeros((ROWS * COLS, N_ACTIONS))

    alpha = 1.0 if "alpha" in bugs else 0.1
    gamma = 0.5 if "gamma" in bugs else 0.99
    eps, eps_min = 1.0, (0.0 if "epsilon" in bugs else 0.05)

    ep_true, ep_agent, ep_len, ep_out = [], [], [], []
    for _ in range(episodes):
        pos = env.reset()
        s = state_index(pos, bugs)
        a = select(Q[s], eps, rng, bugs)
        G_true = G_agent = 0.0
        while True:
            pos2, kind, term, trunc = env.step(a)
            s2 = state_index(pos2, bugs)
            r_true = TRUE_REWARD[kind]
            r = 0.5 if ("reward" in bugs and kind == "step") else r_true

            a2 = select(Q[s2], eps, rng, bugs)

            # ---- TD target -------------------------------------------------
            if "terminal" in bugs and term:
                s_boot = state_index(START, bugs)      # BUG: auto-reset obs used as s'
                boot_mask = 1.0
            else:
                s_boot = s2
                boot_mask = 0.0 if term else 1.0       # FIX: no bootstrap on termination
            if "td_target" in bugs:
                s_boot = s                             # BUG: bootstrap from s, not s'
            if algo == "qlearning":
                nxt = Q[s_boot].max()
            else:                                      # SARSA: on-policy next action
                nxt = Q[s, a2] if "td_target" in bugs else Q[s_boot, a2]
            target = r + gamma * boot_mask * nxt
            Q[s, a] += alpha * (target - Q[s, a])

            G_true += r_true
            G_agent += r
            if "epsilon" in bugs:
                eps = max(eps_min, eps * 0.9)          # BUG: decays every STEP
            s, a = s2, a2
            if term or trunc:
                ep_out.append(kind if term else "timeout")
                break
        ep_true.append(G_true); ep_agent.append(G_agent); ep_len.append(env.t)
        if "epsilon" not in bugs:
            eps = max(eps_min, eps * 0.99)             # FIX: decay once per episode

    # greedy evaluation (eps = 0) with a fresh env
    erng = random.Random(10_000 + seed)
    eenv = PitWalk(erng)
    ev_ret, ev_len, ev_succ, ev_pit = [], [], [], []
    for _ in range(200):
        pos = eenv.reset(); G = 0.0
        while True:
            a = greedy(Q[state_index(pos, bugs)], erng, bugs)
            pos, kind, term, trunc = eenv.step(a)
            G += TRUE_REWARD[kind]
            if term or trunc:
                break
        ev_ret.append(G); ev_len.append(eenv.t)
        ev_succ.append(kind == "goal"); ev_pit.append(kind == "pit")
    return dict(
        true_return=np.array(ep_true), agent_return=np.array(ep_agent),
        length=np.array(ep_len), outcome=np.array(ep_out), Q=Q,
        eval_return=float(np.mean(ev_ret)), eval_len=float(np.mean(ev_len)),
        eval_success=float(np.mean(ev_succ)), eval_pit=float(np.mean(ev_pit)),
    )


# ----------------------------------------------------------------------------
# Experiment definitions
# ----------------------------------------------------------------------------
def configs():
    cfg = {
        "Corrected Q-learning": (frozenset(), "qlearning"),
        "Corrected SARSA": (frozenset(), "sarsa"),
        "Buggy (all 8 bugs)": (frozenset(ALL_BUGS), "qlearning"),
    }
    for b in ALL_BUGS:
        cfg[f"only {BUG_LABEL[b]}"] = (frozenset([b]), "qlearning")
    return cfg


def _job(args):
    name, bugs, algo, seed, episodes = args
    out = train(bugs, algo, episodes, seed)
    return name, seed, out


if __name__ == "__main__":
    import multiprocessing as mp
    import pickle
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    cfg = configs()
    jobs = [(n, b, a, s, episodes) for n, (b, a) in cfg.items() for s in range(n_seeds)]
    with mp.Pool() as pool:
        res = pool.map(_job, jobs, chunksize=4)
    results = {n: {} for n in cfg}
    for n, s, o in res:
        results[n][s] = o
    with open("results.pkl", "wb") as f:
        pickle.dump(dict(results=results, n_seeds=n_seeds, episodes=episodes,
                         cfg={n: (sorted(b), a) for n, (b, a) in cfg.items()}), f)
    print("done", len(jobs), "runs")
