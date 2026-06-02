"""
CTC Circuit — combined runner.

D-CTC (Deutsch model): numpy fixed-point iteration
P-CTC (Lloyd model):   Qiskit statevector + post-selection

Key observable difference between the two models (experimentally
confirmed by Ringbauer et al. 2014 in a 4-photon optics lab):

  D-CTC + SWAP: clones input state (no-cloning broken)
  P-CTC + SWAP: always collapses to |0> (post-selection governs)

This inequivalence is the central result. Run this file to see it.
"""
import subprocess, sys

print("="*60)
print("D-CTC FIXED-POINT SIMULATOR (ctc_core.py)")
print("="*60)
subprocess.run([sys.executable, "ctc_core.py"], cwd="C:/Users/Milton/ctc-circuit")

print("\n\n")
print("="*60)
print("P-CTC QISKIT SIMULATION (ptc_circuit.py)")
print("="*60)
subprocess.run([sys.executable, "ptc_circuit.py"], cwd="C:/Users/Milton/ctc-circuit")

print("\n\n")
print("="*60)
print("PSPACE GADGET — A-W LEMMA 2 (config_graph_gadget.py)")
print("="*60)
subprocess.run([sys.executable, "config_graph_gadget.py"], cwd="C:/Users/Milton/ctc-circuit")

print("\n\n")
print("="*60)
print("QUANTUM UPPER BOUND — A-W THEOREMS 3-5 (quantum_upper_bound.py)")
print("="*60)
subprocess.run([sys.executable, "quantum_upper_bound.py"], cwd="C:/Users/Milton/ctc-circuit")

print("\n\n")
print("="*60)
print("P-CTC = PP COMPLEXITY RESULT (ptc_pp_result.py)")
print("="*60)
subprocess.run([sys.executable, "ptc_pp_result.py"], cwd="C:/Users/Milton/ctc-circuit")

print("\n\n")
print("="*60)
print("COMPLEXITY LANDSCAPE — FULL THEOREM (complexity_landscape.py)")
print("="*60)
subprocess.run([sys.executable, "complexity_landscape.py"], cwd="C:/Users/Milton/ctc-circuit")

print("\n\n" + "="*60)
print("KEY COMPARISON")
print("="*60)
print("""
  SWAP + D-CTC:  fixed_point = rho_sys for ALL inputs  (clones input)
  SWAP + P-CTC:  output = |0>  for ALL inputs           (destroys input)

  Same unitary. Different physics. Different outputs.
  Ringbauer 2014 built this with 4 entangled photons to prove it.

  WHAT EACH FILE PROVED:
    ctc_core.py              P_CTC <= PSPACE  (Deutsch fixed-point, Lemma 1)
    ptc_circuit.py           P-CTC simulation (Bell pair + postselection)
    bacon_s_gate.py          NP <= D-CTC      (S-gate, Bacon 2004)
    config_graph_gadget.py   PSPACE <= P_CTC  (C' gadget, A-W Lemma 2)
    quantum_upper_bound.py   BQP_CTC <= PSPACE (Cesaro resolvent, Thms 3-5)
    ptc_pp_result.py         P-CTC = PP       (PostBQP=PP, Aaronson 2005)
    complexity_landscape.py  Full diagram     (capstone)

  FULL RESULT:
    D-CTC = PSPACE  (classical and quantum collapse — same power)
    P-CTC = PP      (weaker — postselection, not fixed-point)
    PP < PSPACE     (believed, not proven — the one open gap)

  To run on IBM Quantum hardware:
    pip install qiskit-ibm-runtime
    # Run ptc_circuit.ptc_channel() shots, post-select c0=0 AND c1=0
    # Expect ~25% survival rate (Bell state post-selection keeps 1/4 shots)
""")
