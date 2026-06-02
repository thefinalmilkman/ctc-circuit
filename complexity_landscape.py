"""
complexity_landscape.py
=======================
Capstone summary of the CTC circuit proof repository.
Consolidates results from all six proof modules into a single
runnable document: diagram, table, collapse theorems, open questions,
and physical interpretation.

Run with: python complexity_landscape.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP = "=" * 62

# ---------------------------------------------------------------------------
# 1. ASCII complexity diagram
# ---------------------------------------------------------------------------

DIAGRAM = r"""
COMPLEXITY LANDSCAPE WITH CLOSED TIMELIKE CURVES
{sep}

Standard inclusions (no CTCs):

  P  <=  BPP  <=  BQP  <=  PP  <=  PSPACE  <=  EXP
  |               |         |        |
  |               |        #P       NP  (NP <= PP, #P <= PSPACE)
  |               |
  `-- NP <= PSPACE (NP sits between BQP and PP, exact position unknown)

Positions of NP and #P:
  BQP <= PP,  NP <= PP  (both believed strict, not proved)
  #P  <= PSPACE          (Toda's theorem: PH <= P^#P <= PSPACE)

With CTCs (this repo):

  +----------+   collapses to   +----------+
  | P-CTC    |  =============>  |   PP     |   (Lloyd 2011 + Aaronson 2005)
  +----------+                  +----------+
        |                             |
        |    P-CTC = PostBQP = PP     |
        |                             |
  +----------+   collapses to   +----------+
  | D-CTC    |  =============>  | PSPACE   |   (Aaronson-Watrous 2009)
  +----------+                  +----------+

Full chain with CTC labels:

  P  <=  BPP  <=  BQP  <=  [PP = P-CTC]  <=  [PSPACE = D-CTC]  <=  EXP
                              ^                       ^
                         P-CTC collapses         D-CTC collapses
                         classical=postQM        classical=quantum

  NP  <=  PP  = P-CTC        (NP is "inside" the P-CTC layer)
  #P  <=  PSPACE = D-CTC     (counting problems inside D-CTC reach)
{sep}
""".format(sep=SEP)

# ---------------------------------------------------------------------------
# 2. Proof table
# ---------------------------------------------------------------------------

TABLE_HEADER = """
PROOF MODULE TABLE
{sep}
  File                      Result                          Method
  -----------------------------------------------------------------------""".format(
    sep=SEP
)

ROWS = [
    (
        "ctc_core.py",
        "D-CTC fixed-point iteration",
        "Deutsch 1991 consistency condition",
    ),
    (
        "ptc_circuit.py",
        "P-CTC Bell-pair simulation",
        "Ringbauer 2014 linear-algebra projection",
    ),
    (
        "bacon_s_gate.py",
        "NP <= D-CTC  (S-gate SAT amplifier)",
        "Bacon 2004: S-gate squares Bloch-z",
    ),
    (
        "config_graph_gadget.py",
        "PSPACE = D-CTC  (upper + lower bound)",
        "A-W Lemma 2: C' config-graph gadget",
    ),
    (
        "quantum_upper_bound.py",
        "BQP_CTC <= PSPACE",
        "A-W Thms 3-5: Cesaro resolvent",
    ),
    (
        "ptc_pp_result.py",
        "P-CTC = PP  (PostBQP = PP)",
        "Lloyd 2011 + Aaronson 2005",
    ),
]

# ---------------------------------------------------------------------------
# 3. Collapse results
# ---------------------------------------------------------------------------

COLLAPSE = """
KEY COLLAPSE THEOREMS
{sep}

Theorem A  [D-CTC = PSPACE]  (Aaronson & Watrous 2009)
  Lower bound:  PSPACE <= D-CTC
    * Any PSPACE computation can be encoded as a CTC consistency
      problem.  The D-CTC fixed-point iteration solves PSPACE-complete
      problems (TQBF, config-graph reachability) in one "round."
    * Proved here via config_graph_gadget.py (A-W Lemma 2).

  Upper bound:  D-CTC <= PSPACE
    * A classical CTC circuit of polynomial size can be simulated by
      exhaustive search over fixed-points, which lives in PSPACE.
    * Quantum CTC circuits add no extra power: BQP_CTC <= PSPACE
      (quantum_upper_bound.py, Cesaro resolvent argument).

  Corollary:  classical and quantum are IDENTICAL under D-CTCs.
    BPP_CTC = BQP_CTC = PSPACE.
    Quantum speedup vanishes entirely when time machines are available.

-----------------------------------------------------------------------

Theorem B  [P-CTC = PP]  (Lloyd et al. 2011 + Aaronson 2005)
  * P-CTCs implement postselected quantum computation.
  * PostBQP = PP  (Aaronson 2005).
  * Therefore  P-CTC = PP.
  * Proved here via ptc_pp_result.py.

  Corollary:  classical-with-postselection = quantum-with-postselection.
    PostP = PostBPP = PostBQP = PP under P-CTCs.
    Again, quantum advantage disappears once you postselect on
    self-consistent histories.
{sep}
""".format(
    sep=SEP
)

# ---------------------------------------------------------------------------
# 4. Open questions
# ---------------------------------------------------------------------------

OPEN = """
OPEN QUESTIONS
{sep}

1. Is PP strictly less than PSPACE?
   * Widely believed: PP < PSPACE (i.e., counting is easier than
     full alternation), but no proof exists.
   * Equivalently: is P-CTC strictly weaker than D-CTC?
   * Resolving PP vs PSPACE would separate the two CTC models.

2. Are D-CTCs physically realizable?
   * General relativity permits closed timelike curves (Godel 1949,
     Kerr metric, traversable wormholes).
   * Whether quantum mechanics is consistent on a CTC background
     is unsettled.  Deutsch's model assumes fixed-point consistency;
     P-CTC model assumes path-integral / postselection consistency.
     They agree on classical inputs, disagree on quantum mixtures.

3. Does NP collapse to P under any CTC model?
   * D-CTC gives NP <= PSPACE = D-CTC, so NP problems are solvable.
   * But NP <= P under D-CTCs requires PSPACE = P, which would be
     a far larger collapse.  No evidence for this.

4. Can the S-gate be physically implemented?
   * bacon_s_gate.py uses an algebraic squaring map on Bloch-z.
   * A unitary implementation requires ancilla + measurement.
   * Whether the full SAT amplifier is experimentally feasible
     remains open.

5. Do P-CTCs and D-CTCs agree on all problems, or just some?
   * Known: they agree on all unitary inputs (Lloyd 2011).
   * They disagree on mixed states (Pienaar 2013 examples exist).
   * Full characterization of the agreement set is open.
{sep}
""".format(
    sep=SEP
)

# ---------------------------------------------------------------------------
# 5. Physical interpretation
# ---------------------------------------------------------------------------

PHYSICAL = """
PHYSICAL INTERPRETATION
{sep}

A closed timelike curve is a worldline that loops back to its own
past.  In the computational picture:

  * The CTC region is a register whose final state must equal its
    initial state (self-consistency / fixed-point condition).

  * The "computation" is whatever happens to the chronology-respecting
    (CR) register as it interacts with the CTC register.

  * Deutsch's D-CTC model finds a fixed point of the combined
    CR + CTC evolution.  This is exactly what PSPACE algorithms do:
    they explore an exponentially large configuration graph and
    certify reachability -- the same structure as the config-graph
    gadget (config_graph_gadget.py).

Slogan (Aaronson-Watrous 2009):
  "A time machine is a fixed-point computer.
   Fixed-point computers are exactly as powerful as PSPACE."

Slogan (Lloyd 2011):
  "A postselected time machine is a postselected quantum computer.
   Postselected quantum computers are exactly as powerful as PP."

Consequence for physics:
  If CTCs exist and Deutsch's model is correct, then any agent with
  access to a time machine can solve:
    - satisfiability (NP)       -- trivially
    - counting problems (#P)    -- trivially
    - alternating reachability (PSPACE-complete) -- with D-CTC
    - but NOT (under current beliefs) EXP-complete problems.

  The universe, if it contains time machines, is "only" as powerful
  as polynomial space -- a remarkable compression of apparent
  computational omnipotence into a well-understood complexity class.

  P-CTCs are strictly weaker (modulo PP vs PSPACE): they solve
  counting and postselection problems but, if PP < PSPACE, cannot
  solve the full PSPACE-complete suite.

  In both models, quantum mechanics offers no advantage over
  classical computation once time travel is available.  The
  quantum-classical gap, if it exists, lives strictly between
  BQP and PP -- a region untouched by either CTC model.
{sep}
""".format(
    sep=SEP
)

# ---------------------------------------------------------------------------
# Main output
# ---------------------------------------------------------------------------


def main():
    print()
    print(SEP)
    print("  CTC CIRCUIT PROOF REPOSITORY -- CAPSTONE SUMMARY")
    print("  Aaronson-Watrous 2009  |  Lloyd 2011  |  Bacon 2004")
    print(SEP)

    # 1. Diagram
    print(DIAGRAM)

    # 2. Table
    print(TABLE_HEADER)
    for fname, result, method in ROWS:
        # wrap long result/method strings at 36 chars for the last column
        print(f"  {fname:<26}{result:<34}{method}")
    print()
    print(SEP)

    # 3. Collapse theorems
    print(COLLAPSE)

    # 4. Open questions
    print(OPEN)

    # 5. Physical interpretation
    print(PHYSICAL)

    print("  End of complexity_landscape.py")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
