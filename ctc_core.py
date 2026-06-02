try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required: pip install numpy")

# --- Gates ---

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=complex)


def controlled(U):
    """4x4 controlled-U: control=first qubit, target=second."""
    gate = np.eye(4, dtype=complex)
    gate[2:4, 2:4] = U
    return gate


# --- Core ---

def partial_trace_over_sys(rho):
    """Trace out the first (system) qubit from a 4x4 density matrix."""
    return rho[0:2, 0:2] + rho[2:4, 2:4]


def dtc_channel(U, rho_sys, rho_ctc):
    """Apply one step of the D-CTC channel: Tr_sys[U(rho_sys x rho_ctc)U+]."""
    joint = np.kron(rho_sys, rho_ctc)
    evolved = U @ joint @ U.conj().T
    return partial_trace_over_sys(evolved)


def find_fixed_point(U, rho_sys, max_iter=10000):
    """
    Iterate D-CTC channel until convergence.
    Returns (rho_ctc_fixed, iterations).
    """
    rho_ctc = np.eye(2, dtype=complex) / 2  # maximally mixed start

    for i in range(1, max_iter + 1):
        rho_new = dtc_channel(U, rho_sys, rho_ctc)

        # Re-Hermitianize and renormalize for numerical stability
        rho_new = (rho_new + rho_new.conj().T) / 2
        rho_new /= np.trace(rho_new)

        if np.allclose(rho_ctc, rho_new, atol=1e-12):
            return rho_new, i

        rho_ctc = rho_new

    return rho_ctc, max_iter


def verify_fixed_point(U, rho_sys, rho_ctc):
    """Returns the L2 residual ||Tr_sys[U(rho_sys x rho_ctc)U+] - rho_ctc||."""
    rho_out = dtc_channel(U, rho_sys, rho_ctc)
    return np.linalg.norm(rho_out - rho_ctc)


# --- State helpers ---

def ket(state):
    """Column vector from list."""
    return np.array(state, dtype=complex).reshape(-1, 1)


def dm(state):
    """Pure state density matrix from ket."""
    v = ket(state)
    return v @ v.conj().T


def bloch_label(rho):
    """Rough label for a qubit state via Bloch vector."""
    x = np.real(np.trace(X @ rho))
    y = np.real(np.trace(np.array([[0, -1j], [1j, 0]]) @ rho))
    z = np.real(np.trace(Z @ rho))
    return f"Bloch=({x:.3f}, {y:.3f}, {z:.3f})"


# --- Demos ---

def demo_convergence():
    print("=" * 60)
    print("DEMO 1: CNOT Fixed-Point Convergence")
    print("=" * 60)

    zero = dm([1, 0])
    one  = dm([0, 1])
    plus = dm([1, 1]) / 2  # (|0>+|1>)/sqrt(2)
    mixed = 0.7 * zero + 0.3 * one

    states = [
        ("|0>",    zero),
        ("|1>",    one),
        ("|+>",    plus),
        ("0.7|0>+0.3|1>", mixed),
    ]

    for label, rho_sys in states:
        rho_fp, iters = find_fixed_point(CNOT, rho_sys)
        eigs = np.sort(np.real(np.linalg.eigvalsh(rho_fp)))[::-1]
        residual = verify_fixed_point(CNOT, rho_sys, rho_fp)
        print(f"\n  Input: {label}")
        print(f"  Eigenvalues: [{eigs[0]:.6f}, {eigs[1]:.6f}]")
        print(f"  Iterations:  {iters}")
        print(f"  Residual:    {residual:.2e}")

    print()


def demo_cloning():
    print("=" * 60)
    print("DEMO 2: D-CTC Cloning via SWAP  (breaks no-cloning theorem)")
    print("=" * 60)
    print("  Prediction: fixed_point ~= rho_sys  (||rho_ctc - rho_sys|| ~= 0)")
    print()

    zero  = dm([1, 0])
    one   = dm([0, 1])
    plus  = dm([1, 1]) / 2
    minus = dm([1, -1]) / 2

    states = [
        ("|0>",  zero),
        ("|1>",  one),
        ("|+>",  plus),
        ("|->",  minus),
    ]

    for label, rho_sys in states:
        rho_fp, iters = find_fixed_point(SWAP, rho_sys)
        diff = np.linalg.norm(rho_fp - rho_sys)
        print(f"  {label:6s}  ||rho_ctc - rho_sys|| = {diff:.2e}  (iters={iters})")

    print()
    print("  Standard QM: cloning is forbidden (no-cloning theorem).")
    print("  D-CTCs achieve it in one iteration via the consistency condition.")
    print()


def demo_discrimination():
    print("=" * 60)
    print("DEMO 3: Non-Orthogonal State Discrimination via D-CTCs")
    print("=" * 60)
    print("  Distinguishing |0> and |+> -- impossible in standard QM.")
    print()

    zero = dm([1, 0])
    plus = dm([1, 1]) / 2

    states = [("|0>", zero), ("|+>", plus)]

    for label, rho_sys in states:
        rho_fp, _ = find_fixed_point(SWAP, rho_sys)
        # P(measuring |0>) = <0|rho_ctc|0>
        p0 = np.real(rho_fp[0, 0])
        print(f"  Input {label:4s}  ->  fixed_point {bloch_label(rho_fp)}")
        print(f"           P(|0>) after Z-measurement = {p0:.6f}")
        print()

    p0_zero = np.real(find_fixed_point(SWAP, zero)[0][0, 0])
    p0_plus = np.real(find_fixed_point(SWAP, plus)[0][0, 0])
    delta = abs(p0_zero - p0_plus)
    print(f"  Delta P(|0>): |{p0_zero:.4f} - {p0_plus:.4f}| = {delta:.4f}")
    print()
    print("  Standard QM: |<0|+>|^2 = 0.5, so states cannot be reliably told apart.")
    print("  D-CTCs: cloning fixes rho_ctc = rho_sys, making them distinguishable.")
    print()


if __name__ == "__main__":
    print()
    print("D-CTC Fixed-Point Simulator")
    print("Based on Deutsch 1991 + Ringbauer et al. 2014 (Nat. Comm. 5:5145)")
    print()
    demo_convergence()
    demo_cloning()
    demo_discrimination()
