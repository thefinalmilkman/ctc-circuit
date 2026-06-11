"""
Bacon S-Gate: D-CTC Nonlinear SAT Amplifier
============================================
Based on:  Bacon 2004, quant-ph/0309189
Complexity: Aaronson-Watrous 2009, arXiv:0808.2669

THE IDEA
--------
D-CTCs grant a nonlinear gate forbidden by standard QM.
The Bacon S-gate squares the Bloch-z component:

    S( rho ) = 1/2 * (I + nz^2 * Z)   where nz = Tr(Z @ rho)

This breaks linearity — no quantum channel can do it.

AMPLIFICATION
-------------
To decide 3-SAT on n variables with s solutions:

    1. Prepare state with Bloch-z = gamma = 1 - s / 2^(n-1)
    2. Iterate S:   gamma_p = gamma^(2^p)   [doubly exponential]
    3. Read:   gamma_p -> 1  =>  UNSAT (s=0)
               gamma_p -> 0  =>  SAT   (s>=1)

Poly(n) iterations suffice. This puts NP in poly-gate CTC circuits.
The full PSPACE result (A-W 2009) uses the config-graph stationary
distribution gadget instead — encodes any PSPACE TM step, reads the
fixed point.

NOTE: this file implements S(rho) as the ALGEBRAIC formula directly.
For the same squaring realised as a genuine Deutsch fixed point of
Bacon's exact 2-qubit unitary U = SWAP.CNOT (his Eq. 11, solved by
ctc_core.find_fixed_point), see s_gate_unitary.py — the squaring appears
in the chronology-respecting OUTPUT, not the CTC register.
"""

import itertools
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -- Pauli matrices ------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)


# -- State helpers -------------------------------------------------------------

def bloch_z(rho):
    """Bloch-z component of a qubit density matrix."""
    return float(np.real(np.trace(Z @ rho)))


def from_bloch_z(z):
    """Diagonal qubit density matrix with given Bloch-z."""
    z = float(np.clip(z, -1.0, 1.0))
    return np.array([[(1 + z) / 2, 0], [0, (1 - z) / 2]], dtype=complex)


# -- S-Gate --------------------------------------------------------------------

def s_gate(rho):
    """
    Bacon S-gate: D-CTC nonlinear map. Squares the Bloch-z component.
    Standard QM forbids this — D-CTCs provide it via the fixed-point
    consistency condition.

    Reference: Bacon 2004 quant-ph/0309189, Eq. for the S operator.
    """
    nz = bloch_z(rho)
    return 0.5 * (I2 + nz ** 2 * Z)


def iterate_s(rho_init, n_iter):
    """Apply S-gate n_iter times. Returns list of Bloch-z values."""
    trace = [bloch_z(rho_init)]
    rho = rho_init.copy()
    for _ in range(n_iter):
        rho = s_gate(rho)
        trace.append(bloch_z(rho))
    return trace


# -- 3-SAT ---------------------------------------------------------------------

def eval_clause(clause, assignment):
    """Check one clause. Literals are signed ints: +k means x_k, -k means NOT x_k."""
    return any(
        (assignment[abs(lit) - 1] == 1) == (lit > 0)
        for lit in clause
    )


def count_solutions(clauses, n_vars):
    """Brute-force count of satisfying assignments (small n only)."""
    s = 0
    for bits in itertools.product([0, 1], repeat=n_vars):
        if all(eval_clause(c, bits) for c in clauses):
            s += 1
    return s


# -- Amplifier -----------------------------------------------------------------

def run_amplifier(clauses, n_vars, n_iter=18):
    """
    Full Bacon amplifier.
    Returns (s, gamma, z_trace).
    """
    s     = count_solutions(clauses, n_vars)
    gamma = 1.0 - s / (2 ** (n_vars - 1))
    rho   = from_bloch_z(gamma)
    trace = iterate_s(rho, n_iter)
    return s, gamma, trace


# -- ASCII plot ----------------------------------------------------------------

def ascii_bar(value, width=40):
    """Horizontal bar for a value in [-1, 1]."""
    pct   = (value + 1) / 2          # map [-1,1] -> [0,1]
    filled = int(round(pct * width))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + " " * (width - filled) + "]"


# -- Demos ---------------------------------------------------------------------

def demo_squaring():
    sep = "=" * 62
    print(sep)
    print("DEMO 1 — S-GATE: Bloch-z SQUARING")
    print("Reference: Bacon 2004 (quant-ph/0309189)")
    print(sep)
    print("  Action:  S(rho) = 1/2*(I + nz^2 * Z)")
    print("  Standard QM: no quantum channel can square nz (nonlinear).")
    print("  D-CTCs:  the fixed-point consistency condition supplies it.")
    print()
    print(f"  {'z_in':>8}  {'S(z)':>8}  {'predicted z^2':>14}  {'match':>6}")
    print(f"  {'--------':>8}  {'--------':>8}  {'---------------':>14}  {'------':>6}")
    for z in [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0, -0.3, -0.7, -1.0]:
        rho   = from_bloch_z(z)
        rho_s = s_gate(rho)
        z_out = bloch_z(rho_s)
        pred  = z ** 2
        ok    = "OK" if abs(z_out - pred) < 1e-12 else "FAIL"
        print(f"  {z:>8.4f}  {z_out:>8.4f}  {pred:>14.4f}  {ok:>6}")
    print()


def demo_amplification():
    sep = "=" * 62
    print(sep)
    print("DEMO 2 — AMPLIFICATION: Doubly-Exponential Collapse")
    print(sep)
    print("  z_p = z_0^(2^p)   [doubly exponential in p]")
    print("  UNSAT: z=1 stays at 1.   SAT: |z|<1 collapses to 0.")
    print()

    n_iter = 8
    cases = [
        ("UNSAT  z=1.000",  1.000),
        ("SAT    z=0.900",  0.900),
        ("SAT    z=0.700",  0.700),
        ("SAT    z=0.500",  0.500),
        ("SAT    z=0.100",  0.100),
        ("SAT    z=-0.500", -0.500),
        ("SAT    z=-0.900", -0.900),
    ]

    header = f"  {'Case':<20} " + " ".join(f"p={p:<6}" for p in range(n_iter + 1))
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, z0 in cases:
        rho   = from_bloch_z(z0)
        trace = iterate_s(rho, n_iter)
        row   = f"  {label:<20} " + " ".join(f"{v:>7.4f}" for v in trace)
        print(row)

    print()
    print("  Note: (-0.5)^2 = 0.25 -> 0.0625 -> ... -> 0. Negative gamma works.")
    print()


def demo_3sat():
    sep = "=" * 62
    print(sep)
    print("DEMO 3 — 3-SAT INSTANCES")
    print("Reference: Bacon 2004 — NP in poly-gate D-CTC circuits")
    print(sep)

    instances = [
        {
            "name": "UNSAT — x1 AND (NOT x1)",
            "n_vars": 2,
            "clauses": [(1, 1, 1), (-1, -1, -1)],
            # clause 1: x1 must be True; clause 2: x1 must be False. Impossible.
        },
        {
            "name": "UNSAT — all 8 sign patterns (n=3)",
            "n_vars": 3,
            "clauses": [
                ( 1,  2,  3), (-1, -2, -3),
                ( 1, -2,  3), (-1,  2, -3),
                ( 1,  2, -3), (-1, -2,  3),
                ( 1, -2, -3), (-1,  2,  3),
            ],
            # Every assignment kills exactly one clause.
        },
        {
            "name": "SAT — unique solution (1,1,1)",
            "n_vars": 3,
            "clauses": [
                ( 1,  2,  3),   # kills (0,0,0)
                (-1,  2,  3),   # kills (1,0,0)
                ( 1, -2,  3),   # kills (0,1,0)
                ( 1,  2, -3),   # kills (0,0,1)
                (-1, -2,  3),   # kills (1,1,0)
                (-1,  2, -3),   # kills (1,0,1)
                ( 1, -2, -3),   # kills (0,1,1)
            ],
        },
        {
            "name": "SAT — 6 solutions (n=3)",
            "n_vars": 3,
            "clauses": [(1, 2, 3), (-1, -2, -3)],
            # Fails only (0,0,0) and (1,1,1). 6 solutions.
        },
    ]

    n_iter = 16

    for inst in instances:
        n_vars  = inst["n_vars"]
        clauses = inst["clauses"]
        name    = inst["name"]

        s, gamma, trace = run_amplifier(clauses, n_vars, n_iter)
        final = trace[-1]

        print(f"\n  +- {name}")
        print(f"  |  n={n_vars} vars, {len(clauses)} clauses")
        print(f"  |  Solutions: s = {s} / {2**n_vars}")
        print(f"  |  gamma = 1 - {s}/{2**(n_vars-1)} = {gamma:.6f}")
        print(f"  |")

        # Show iterations until convergence or 12 steps
        print(f"  |  {'p':>3}  {'z_p':>12}  bar (z mapped to [0,1])")
        show_iters = list(range(min(13, n_iter + 1)))
        for p in show_iters:
            z   = trace[p]
            bar = ascii_bar(z, width=30)
            print(f"  |  {p:>3}  {z:>12.6f}  {bar}")
        if n_iter + 1 > len(show_iters):
            print(f"  |  ... (converged)")

        verdict = "UNSAT" if final > 0.95 else ("SAT" if abs(final) < 0.001 else "converging")
        correct = (s == 0 and verdict == "UNSAT") or (s > 0 and verdict == "SAT")
        mark    = "CORRECT" if correct else "CHECK"
        print(f"  |")
        print(f"  +- Final z = {final:.2e}  ->  {verdict}  [{mark}]")

    print()


def demo_complexity():
    sep = "=" * 62
    print(sep)
    print("DEMO 4 — COMPLEXITY CLAIMS")
    print(sep)
    print("""
  D-CTC = PSPACE  (Aaronson-Watrous 2009, arXiv:0808.2669)
  P-CTC = PP      (Lloyd et al. + Brun-Wilde)
  PP    < PSPACE  (believed under standard assumptions)

  -- The S-gate / NP demo (this file) ----------------------
  S-gate squaring is the nonlinearity D-CTCs grant.
  Bacon 2004 shows this solves SAT in poly(n) gates:

    gamma = 1 - s/2^(n-1)  [encodes solution count]
    After p applications:  gamma_p = gamma^(2^p)
    |gamma| < 1 collapses to 0 doubly-exponentially.
    gamma = 1 stays at 1 (UNSAT signal).
    Poly(n) iterations needed for reliable discrimination.

  -- The full PSPACE result (Aaronson-Watrous) -------------
  Stronger than NP: encodes ANY PSPACE Turing machine.
  Circuit: TM step function as the D-CTC transition map.
  Stationary distribution of the consistency condition
  = the answer. The CTC solves f(x)=x — for free.

  -- Why D-CTC != P-CTC ------------------------------------
  P-CTC fixed-point = postselection -> PP, not PSPACE.
  Same unitary, different consistency rule, weaker class.
  Ringbauer et al. 2014 confirmed this in a 4-photon lab.
  Your ctc_core.py + ptc_circuit.py confirm it here.

  -- Caveats (Bennett 2009 linearity trap) -----------------
  D-CTC = PSPACE holds under the A-W definition:
    worst-case over all fixed points, inputs as basis states.
  Whether it's "physical" for mixed-state inputs is a live
  30-year-old foundational debate. The Bacon S-gate demo is
  valid under the standard complexity-theory definition.
""")


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("  Bacon S-Gate: D-CTC Nonlinear SAT Amplifier")
    print("  Bacon 2004 (quant-ph/0309189)")
    print("  Aaronson-Watrous 2009 (arXiv:0808.2669)")
    print()

    demo_squaring()
    demo_amplification()
    demo_3sat()
    demo_complexity()
