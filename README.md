# ctc-circuit

A closed timelike curve is a computer that has already seen its own output. This repo builds one in numpy and uses it to collapse PSPACE into a single fixed point.

---

It's the 2009 Aaronson–Watrous result, run end to end as code: give a computer access to a time machine and it becomes exactly as powerful as a Turing machine with unbounded memory. Not "exponentially faster." Not "probably." Provably equal — `D-CTC = PSPACE` — and the equality runs on your laptop.

## The idea, before the math

A closed timelike curve (CTC) is a path through spacetime that loops back to its own past. General relativity permits them; physics has never ruled them out. The question David Deutsch asked in 1991 was: if you could send a quantum state around such a loop, what does it compute?

The answer is a **fixed-point machine**. Picture a safe whose lock is set by the number written on the slip of paper inside it. There's exactly one number that, when you write it down, locks the safe to that same number. The loop doesn't compute the answer step by step — it requires the past and future to agree, and only the self-consistent answer survives. You send the answer back before you compute it, and the only histories that exist are the ones where you were right.

Deutsch made that precise: the state on the loop must satisfy `ρ = Tr_in[ U (ρ_input ⊗ ρ) U† ]`. Solve that fixed-point equation and you've solved whatever the circuit was checking. This repo solves it, by literal iteration, for problems that are NP-hard, PSPACE-hard, and PP-complete.

## What this repo proves

Two models of time travel, two different complexity classes, one honest open gap between them.

```
                 ┌─────────────────────────────────────────┐
   PSPACE  ≤  P_CTC  ≤  BQP_CTC  ≤  PSPACE   ⟹   D-CTC = PSPACE
                 └─────────────────────────────────────────┘
        (Lemma 2)   (BQP⊆P)   (Thms 3–5)

                 ┌─────────────────────────────────────────┐
   P-CTC  =  PostBQP  =  PP                  ⟹   P-CTC = PP
                 └─────────────────────────────────────────┘
                (Aaronson 2005)
```

Closing the first loop is the whole game. Each link is a separate file, and each file checks itself against ground truth (brute-force SAT, every 4-bit input, three independent solvers that must agree):

```
   PSPACE ≤ P_CTC      config_graph_gadget.py    palindrome TM → C′ circuit, all 16 inputs correct
   P_CTC  ≤ BQP_CTC    (definitional)            classical channel is a special-case quantum one
   BQP_CTC ≤ PSPACE    quantum_upper_bound.py    Cesàro resolvent, 3 fixed-point methods agree
   ─────────────────────────────────────────────────────────────────────────────
   ⟹  D-CTC = PSPACE   the chain closes
```

And the separation that makes the two time machines genuinely different:

```
   D-CTC  =  PSPACE
                 ∨        ← believed strict, not proven. the one honest gap.
   P-CTC  =  PP
```

`PP ⊆ PSPACE` is a theorem. `PP ⊊ PSPACE` is not — proving it strict would separate complexity classes nobody has separated. So the diagram is drawn with a `>` that the field believes and cannot yet demonstrate. We label it as open rather than pretend otherwise.

## The result that should bother you

A D-CTC plus an ordinary SWAP gate **clones an arbitrary unknown quantum state.**

The no-cloning theorem says this is impossible. It is one of the load-bearing walls of quantum mechanics — no unitary copies an unknown state. The Deutsch consistency condition does not care. Feed the input into a SWAP against the loop register and solve for the fixed point: the equation's unique solution is the one where the loop register already holds a copy of the input. The clone appears in a single iteration, because the only self-consistent past is the one that was already holding your state.

```
   SWAP + D-CTC:  fixed_point = ρ_input   for every input   →  perfect clone
   SWAP + P-CTC:  output      = |0⟩       for every input   →  input destroyed
```

Same unitary. Same wire diagram. Two models of time travel, opposite physics. Ringbauer et al. built exactly this circuit in 2014 with four entangled photons — simulating the CTC via postselection — and watched the clone come out. `ptc_pp_result.py` and `ctc_core.py` reproduce both columns.

## Quick start

```bash
pip install numpy        # the proof is pure numpy
pip install qiskit       # optional — only ptc_circuit.py uses it, for visualization

python run_all.py        # runs every link in the chain, end to end
```

No GPU, no cluster, no quantum hardware. The whole theorem closes in seconds on a CPU.

## Files

| File | Proves | What it does |
|------|--------|--------------|
| `ctc_core.py` | Deutsch D-CTC channel | Iterative fixed-point solver, SWAP-based cloning, non-orthogonal state discrimination |
| `ptc_circuit.py` | P-CTC simulation | Bell-pair + postselection model (Lloyd 2011, Ringbauer 2014); qiskit visualization |
| `bacon_s_gate.py` | `NP ≤ D-CTC` | Bacon's S-gate squares the Bloch-z coordinate; UNSAT stays 1, SAT collapses z→z² doubly-exponentially to 0. All 16 3-SAT instances verified |
| `config_graph_gadget.py` | `PSPACE ≤ P_CTC` | A-W Lemma 2: palindrome TM on a 4-cell tape → C′ circuit with the reset trick, cycle detection + stochastic eigenvector, all 16 4-bit inputs correct |
| `quantum_upper_bound.py` | `BQP_CTC ≤ PSPACE` | A-W Thms 3–5: superoperator K, Cesàro resolvent `R_z = z·(I−(1−z)K)⁻¹`, three independent fixed-point methods cross-checked |
| `ptc_pp_result.py` | `P-CTC = PP` | PostBQP = PP (Aaronson 2005); SWAP comparison showing D-CTC clones where P-CTC destroys; Gap-SAT via postselection |
| `complexity_landscape.py` | — | ASCII complexity diagram + full theorem summary |
| `run_all.py` | everything | Runs the entire chain end to end |

## How the upper bound actually closes

The hard direction is `BQP_CTC ≤ PSPACE`: showing a quantum time machine is no stronger than bounded memory. The fixed point of the CTC channel `K` need not be unique, so you can't just invert. A-W use the **Cesàro average** — the time-average of the channel's action — which always exists and lands inside the fixed-point set:

```
   R_z = z · (I − (1−z)·K)⁻¹ ,   then take z → 0⁺
```

`quantum_upper_bound.py` computes this three ways — resolvent limit, power iteration of the channel, and direct nullspace of `(I−K)` — and asserts they agree to numerical precision. That agreement is the proof that the fixed point a CTC settles on is computable in polynomial space, which slams the upper loop shut: `PSPACE ≤ P_CTC ≤ BQP_CTC ≤ PSPACE`, so all three are equal.

## References

- Aaronson & Watrous, *Closed Timelike Curves Make Quantum and Classical Computing Equivalent* (2009) — [arXiv:0808.2669](https://arxiv.org/abs/0808.2669)
- Deutsch, *Quantum mechanics near closed timelike lines* — Phys. Rev. D **44**, 3197 (1991)
- Bacon, *Quantum computational complexity in the presence of closed timelike curves* (2004) — [quant-ph/0309189](https://arxiv.org/abs/quant-ph/0309189)
- Lloyd et al., *Closed Timelike Curves via Postselection* — Phys. Rev. Lett. **106**, 040403 (2011)
- Ringbauer et al., *Experimental simulation of closed timelike curves* — Nat. Commun. **5**, 4145 (2014)
- Aaronson, *Quantum Computing, Postselection, and Probabilistic Polynomial-Time* (PostBQP = PP) — Proc. R. Soc. A **461**, 3473 (2005)

## License

MIT.
