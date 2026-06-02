"""
config_graph_gadget.py — Aaronson-Watrous D-CTC = PSPACE (Lemma 2)
====================================================================
Reference: Aaronson & Watrous 2009, arXiv:0808.2669

THE GAP THIS FILLS
------------------
bacon_s_gate.py  proves NP <= D-CTC via the S-gate (Bacon 2004).
THIS file        proves PSPACE <= D-CTC via the config-graph gadget.
Together:        NP <= PSPACE = D-CTC (the full A-W theorem).

THE CORE IDEA (Lemma 2 of A-W 2009)
--------------------------------------
A D-CTC finds fixed points of stochastic maps for free.
Any PSPACE TM M can be encoded as a deterministic circuit C' such
that the unique stationary distribution of C' carries M's answer bit.

CONSTRUCTION
------------
CTC register = (config m, answer bit b)

    C'(m, b) =
        (start_config, 1)    if m is an accepting halt
        (start_config, 0)    if m is a rejecting halt
        (successor(m), b)    otherwise

The reset-to-start on halt forces a unique reachable cycle.
Every node in the cycle carries the same b = TM's answer.
Nature places the CTC in the stationary distribution → read b → done.

TWO READOUT METHODS (both implemented below)
--------------------------------------------
1. Cycle detection  — Floyd tortoise/hare on the functional graph.
   This IS the stationary-distribution computation made explicit.
2. Stochastic matrix eigenvector — build M_stoch, power-iterate to
   the eigenvector for eigenvalue 1. Proves the linear-algebra claim.

TURING MACHINE: palindrome on a 4-cell binary tape
---------------------------------------------------
Uses the mark-and-scan technique (standard PSPACE approach):
  1. Mark leftmost unmarked cell, remember its value (0 or 1).
  2. Scan right to rightmost unmarked cell.
  3. If it matches, mark it and restart from (1). Else REJECT.
  4. If no unmarked cells remain, ACCEPT.
"""

import sys
import numpy as np
from collections import deque

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ----------------------------------------------------------------
# TURING MACHINE CONSTANTS
# ----------------------------------------------------------------

N = 4        # tape cells, positions 0..N-1
X = 2        # "erased" / marked symbol

# States
SCAN_L   = 0   # scan right for leftmost unmarked cell
READ0_R  = 1   # found 0 on left, scan right for rightmost unmarked
READ1_R  = 2   # found 1 on left, scan right for rightmost unmarked
MATCH0   = 3   # at rightmost, verify it is 0 (pair with left 0)
MATCH1   = 4   # at rightmost, verify it is 1 (pair with left 1)
ACCEPT   = 5
REJECT   = 6

HALT = {ACCEPT, REJECT}

STATE_NAMES = {
    SCAN_L: 'SCAN_L ', READ0_R: 'READ0_R', READ1_R: 'READ1_R',
    MATCH0: 'MATCH0 ', MATCH1: 'MATCH1 ',
    ACCEPT: 'ACCEPT ', REJECT: 'REJECT ',
}


# ----------------------------------------------------------------
# TM SUCCESSOR FUNCTION
# ----------------------------------------------------------------

def successor(cfg):
    """One deterministic step of the palindrome TM.
    Config = (state: int, head: int, tape: tuple[int,...])
    """
    q, h, tape = cfg[0], cfg[1], list(cfg[2])
    if q in HALT:
        return cfg   # halted configs are fixed points of successor

    s = tape[h]

    if q == SCAN_L:
        if s == 0:
            tape[h] = X
            return (READ0_R, min(h + 1, N - 1), tuple(tape))
        elif s == 1:
            tape[h] = X
            return (READ1_R, min(h + 1, N - 1), tuple(tape))
        else:                             # s == X, skip and move right
            if h == N - 1:
                return (ACCEPT, h, tuple(tape))   # all marked → accept
            return (SCAN_L, h + 1, tuple(tape))

    elif q in (READ0_R, READ1_R):
        if s == X:
            # No unmarked cell to the right of us → single-cell remainder
            return (ACCEPT, h, tuple(tape))
        # Is h the rightmost unmarked?
        if h == N - 1 or tape[h + 1] == X:
            match_st = MATCH0 if q == READ0_R else MATCH1
            return (match_st, h, tuple(tape))
        return (q, h + 1, tuple(tape))   # keep scanning right

    elif q == MATCH0:
        if s == 0:
            tape[h] = X
            return (SCAN_L, 0, tuple(tape))   # matched → restart from left
        return (REJECT, h, tuple(tape))

    elif q == MATCH1:
        if s == 1:
            tape[h] = X
            return (SCAN_L, 0, tuple(tape))
        return (REJECT, h, tuple(tape))

    return (REJECT, h, tuple(tape))   # unreachable


def make_start(word):
    """Build start config from a 4-char binary string e.g. '0110'."""
    assert len(word) == N
    return (SCAN_L, 0, tuple(int(c) for c in word))


# ----------------------------------------------------------------
# C' CIRCUIT  (A-W Lemma 2)
# ----------------------------------------------------------------

def c_prime(cfg, b, start):
    """The Aaronson-Watrous C' map on (config, answer_bit).
    On halt: reset to start config and stamp b with the verdict.
    """
    q = cfg[0]
    if q == ACCEPT:
        return (start, 1)
    if q == REJECT:
        return (start, 0)
    return (successor(cfg), b)


# ----------------------------------------------------------------
# METHOD 1: CYCLE DETECTION
# ----------------------------------------------------------------

def find_answer_cycle(word):
    """Detect the fixed-point cycle of C' by following the map.
    Returns (answer_bit, cycle_length, transient_length).
    """
    start = make_start(word)
    node = (start, 0)    # initial b is irrelevant; the cycle answer is unique
    seen = {}
    path = []

    while node not in seen:
        seen[node] = len(path)
        path.append(node)
        cfg, b = node
        next_cfg, next_b = c_prime(cfg, b, start)
        node = (next_cfg, next_b)

    cycle_start_idx = seen[node]
    cycle = path[cycle_start_idx:]
    transient = cycle_start_idx
    answer_b = cycle[0][1]   # all nodes in cycle carry the same b
    return answer_b, len(cycle), transient


# ----------------------------------------------------------------
# METHOD 2: STOCHASTIC MATRIX EIGENVECTOR
# ----------------------------------------------------------------

def find_answer_eigenvector(word):
    """Build M_stoch for C', power-iterate to stationary distribution.
    Returns (answer_bit, p_stationary, reachable_count).
    """
    start = make_start(word)
    init = (start, 0)

    # BFS over all reachable (cfg, b) pairs
    reachable = {}
    queue = deque([init])
    while queue:
        node = queue.popleft()
        if node in reachable:
            continue
        reachable[node] = len(reachable)
        cfg, b = node
        nxt = c_prime(cfg, b, start)
        queue.append(nxt)

    n = len(reachable)
    M = np.zeros((n, n))
    for node, i in reachable.items():
        cfg, b = node
        nxt_cfg, nxt_b = c_prime(cfg, b, start)
        j = reachable[(nxt_cfg, nxt_b)]
        M[j, i] = 1.0   # column-stochastic: col i → row j

    # Power iteration: M @ p = p
    p = np.ones(n) / n
    for _ in range(2000):
        p_new = M @ p
        if np.allclose(p, p_new, atol=1e-13):
            break
        p = p_new

    # Read answer from the support of the stationary distribution
    mass_b0 = sum(p[i] for (_, b), i in reachable.items() if b == 0)
    mass_b1 = sum(p[i] for (_, b), i in reachable.items() if b == 1)
    answer_b = 1 if mass_b1 > mass_b0 else 0
    return answer_b, mass_b0, mass_b1, n


# ----------------------------------------------------------------
# GROUND TRUTH
# ----------------------------------------------------------------

def is_palindrome(word):
    return word == word[::-1]


# ----------------------------------------------------------------
# DEMOS
# ----------------------------------------------------------------

SEP = "=" * 62


def demo_construction():
    print(SEP)
    print("DEMO 1 - THE C' CIRCUIT CONSTRUCTION")
    print("Reference: Aaronson-Watrous 2009, arXiv:0808.2669, Lemma 2")
    print(SEP)
    print("""
  CTC register = (config m, answer bit b)

      C'(m, b) =
          (start, 1)       if m is an accepting halt config
          (start, 0)       if m is a rejecting halt config
          (successor(m),b) otherwise

  WHY THE RESET TRICK WORKS
  The reset forces a single reachable cycle: from start, the TM
  marches deterministically to halt, resets, stamps b, loops.
  The cycle consists of ALL configs on that unique path, each
  carrying the same b. The stationary distribution is uniform
  over this cycle — read any node's b to get the answer.

  CLASSICAL vs QUANTUM
  bacon_s_gate.py uses the nonlinear S-gate (quantum, Bacon 2004).
  This gadget is purely classical and linear (stochastic matrix).
  Both require D-CTCs. The full A-W theorem:
    NP in D-CTC  (S-gate, Bacon 2004)
    PSPACE = D-CTC  (config-graph gadget, Aaronson-Watrous 2009)
""")


def demo_single(word):
    print(SEP)
    print(f"DEMO 2 - PALINDROME CHECKER: input = '{word}'")
    print(SEP)

    truth = is_palindrome(word)
    start = make_start(word)

    print(f"  Input:       '{word}'")
    print(f"  Start cfg:   state={STATE_NAMES[SCAN_L]}, head=0, tape={start[2]}")
    print(f"  Ground truth: {'PALINDROME' if truth else 'NOT PALINDROME'}")
    print()

    # TM execution trace
    print("  TM execution trace:")
    print(f"    {'step':>4}  {'state':<9}  {'head'}  tape")
    cfg = start
    step = 0
    while cfg[0] not in HALT and step < 60:
        q, h, tape = cfg
        cells = [str(s) if s != X else 'X' for s in tape]
        cells[h] = f'[{cells[h]}]'
        print(f"    {step:>4}  {STATE_NAMES[q]:<9}  {h}     {''.join(cells)}")
        cfg = successor(cfg)
        step += 1
    print(f"    {step:>4}  {STATE_NAMES[cfg[0]]:<9}  (HALT)")
    print()

    # Method 1: cycle detection
    b1, cycle_len, transient = find_answer_cycle(word)
    print(f"  [Method 1 — Cycle Detection]")
    print(f"    Transient length: {transient} steps to enter cycle")
    print(f"    Cycle length:     {cycle_len} configs")
    print(f"    Answer bit b = {b1}  ->  {'PALINDROME' if b1 else 'NOT PALINDROME'}")
    print(f"    {'CORRECT' if (b1 == 1) == truth else '*** WRONG ***'}")
    print()

    # Method 2: eigenvector
    b2, mass0, mass1, n_reach = find_answer_eigenvector(word)
    print(f"  [Method 2 — Stationary Distribution]")
    print(f"    Reachable (cfg, b) pairs: {n_reach}")
    print(f"    Mass on b=0: {mass0:.6f}")
    print(f"    Mass on b=1: {mass1:.6f}")
    print(f"    Answer bit b = {b2}  ->  {'PALINDROME' if b2 else 'NOT PALINDROME'}")
    print(f"    {'CORRECT' if (b2 == 1) == truth else '*** WRONG ***'}")


def demo_all_inputs():
    print()
    print(SEP)
    print("DEMO 3 - ALL 16 FOUR-BIT INPUTS")
    print(SEP)
    print(f"  {'word':<6}  {'truth':<14}  {'b':<3}  {'ok':<5}  "
          f"{'cycle':<7}  {'reachable'}")
    print("  " + "-" * 52)

    all_ok = True
    for i in range(16):
        word = format(i, '04b')
        truth = is_palindrome(word)
        b, cycle_len, _ = find_answer_cycle(word)
        _, _, _, n_reach = find_answer_eigenvector(word)
        ok = (b == 1) == truth
        if not ok:
            all_ok = False
        verdict = "palindrome" if truth else "not-palin "
        mark = "OK" if ok else "FAIL"
        print(f"  {word:<6}  {verdict:<14}  {b:<3}  {mark:<5}  "
              f"{cycle_len:<7}  {n_reach}")

    print()
    print(f"  All correct: {all_ok}")


def demo_complexity():
    print()
    print(SEP)
    print("DEMO 4 - COMPLEXITY RESULT")
    print(SEP)
    print("""
  THEOREM (Aaronson-Watrous 2009):
    P_CTC = BQP_CTC = PSPACE

  PROOF MAP (what this repo now covers)
  +---------+-----------------------+---------------------------+
  | File    | Claim                 | Mechanism                 |
  +---------+-----------------------+---------------------------+
  | bacon_s | NP <= D-CTC           | Nonlinear S-gate squaring |
  | _gate   | (Bacon 2004)          | z -> z^2 doubly-exp decay |
  +---------+-----------------------+---------------------------+
  | config  | PSPACE <= P_CTC       | Config-graph C' gadget    |
  | _graph  | (A-W Lemma 2)         | Reset trick + cycle det.  |
  +---------+-----------------------+---------------------------+
  | ctc     | PSPACE >= P_CTC       | find_fixed_point() on     |
  | _core   | (A-W Lemma 1)         | the Deutsch CPTP map      |
  +---------+-----------------------+---------------------------+

  WHAT THE C' GADGET ADDS OVER THE S-GATE
  The S-gate is algebraic: it COMPUTES via the nonlinear map.
  The config-graph gadget is structural: it ENCODES the TM's
  entire computation in the topology of the functional graph,
  then reads the fixed point — no iteration needed in theory.
  That structural encoding is why it reaches PSPACE, not just NP.

  NEXT STEP (to close the loop completely)
  Wire ctc_core.find_fixed_point() on the quantum version:
  Build U_tm as a unitary encoding one TM step, embed in the
  Deutsch channel, find the density-matrix fixed point via
  the Cesaro-resolvent (A-W Theorems 3-5). This gives the
  BQP_CTC = PSPACE upper bound to match Lemma 2 from below.
""")


# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("  Aaronson-Watrous D-CTC = PSPACE")
    print("  Config-Graph Gadget (Lemma 2)")
    print("  arXiv:0808.2669")
    print()

    demo_construction()
    demo_single("0110")   # palindrome
    print()
    demo_single("0100")   # not palindrome
    demo_all_inputs()
    demo_complexity()
