"""
sqd_lanczos.py — Sampled Quantum Diagonalization subspace projector.

Provides the ``SampledSubspaceProjector`` class, which evaluates quantum
operators within a bitstring subspace obtained from a quantum sampler.
Used as the core engine of the SQD-Liouville-Lanczos pipeline.

Author       : Olivier Lebreux
Last modified: 12-04-2026
"""

from qiskit.quantum_info import SparsePauliOp
from LiouvilleLanczos.Quantum_computer.QC_lanczos import relative_simplify_spo
import numpy as np


class SampledSubspaceProjector:
    """
    Evaluates quantum operators within a sampled bitstring subspace.

    Given a set of bitstrings and their associated coefficients — typically
    obtained from a quantum sampler — this class projects operator expectation
    values and matrix elements onto that subspace. It serves as the inner-product
    engine for the SQD-Liouville-Lanczos algorithm.

    Attributes
    ----------
    states : np.ndarray, shape (N, q)
        Array of N bitstrings, each of length q (number of qubits).
    coeffs : np.ndarray, shape (N,)
        Complex coefficients of the state superposition over the bitstrings.
    eps : float or None
        Relative threshold used to simplify SparsePauliOp terms. If None,
        no simplification is applied.
    N : int
        Number of sampled bitstrings.
    q : int
        Number of qubits.
    """

    def __init__(
        self, states: np.ndarray, coeffs: np.ndarray = None, epsilon: float = None
    ):
        """
        Construct a SampledSubspaceProjector.

        Args:
            states: (N, q) array of bitstrings
            coeffs: (N,) coefficients of the state superposition
            epsilon: threshold for operator simplification
        """
        self.states = states
        self.coeffs = coeffs
        self.eps = epsilon
        self.N, self.q = self.states.shape

    # -------------------------
    # Average op function
    # -------------------------
    def operator_average_value(self, op: SparsePauliOp):
        """
        Compute the expectation value ⟨ψ|op|ψ⟩ in the bitstring subspace.

        Evaluates the operator by explicitly constructing the (N × N × k)
        Pauli matrix elements over all pairs of sampled bitstrings, then
        contracting with the state coefficients.

        The Pauli labels are reversed to match Qiskit's qubit ordering
        convention (little-endian, rightmost qubit = qubit 0).

        Parameters
        ----------
        op : SparsePauliOp
            Operator to evaluate, expressed as a sum of Pauli strings.

        Returns
        -------
        complex
            Expectation value ⟨ψ|op|ψ⟩.
        """
        P = np.array([list(label[::-1]) for label in op.paulis.to_labels()])
        p = op.coeffs

        N = self.N
        k = P.shape[0]

        b_iq = self.states[:, None, None, :]
        b_jq = self.states[None, :, None, :]
        P = P[None, None, :, :]

        container = np.zeros((N, N, k, self.q), dtype=complex)

        container += ((b_iq == 0) & (b_jq == 0) & (P == "I")) * 1
        container += ((b_iq == 1) & (b_jq == 1) & (P == "I")) * 1
        container += ((b_iq == 0) & (b_jq == 1) & (P == "X")) * 1
        container += ((b_iq == 1) & (b_jq == 0) & (P == "X")) * 1
        container += ((b_iq == 0) & (b_jq == 1) & (P == "Y")) * (-1j)
        container += ((b_iq == 1) & (b_jq == 0) & (P == "Y")) * (1j)
        container += ((b_iq == 0) & (b_jq == 0) & (P == "Z")) * 1
        container += ((b_iq == 1) & (b_jq == 1) & (P == "Z")) * (-1)

        prod_q = np.prod(container, axis=-1)
        result = np.sum(
            p[None, None, :]
            * np.conj(self.coeffs)[:, None, None]
            * self.coeffs[None, :, None]
            * prod_q
        )

        return result

    # -------------------------
    # Inner product used in SQD
    # -------------------------
    def inner_product_sqd(
        self, A: SparsePauliOp, B: SparsePauliOp, real_result=False, Name=None
    ):
        """
        Compute the SQD inner product ⟨ψ|{A, B†}|ψ⟩ / 2.

        Builds the anticommutator {A, B†} = A·B† + B†·A, optionally
        simplifies it, then evaluates its expectation value in the bitstring
        subspace via ``operator_average_value``.

        This symmetric inner product is used by the Liouville-Lanczos
        algorithm to construct the Gram matrix over the Krylov space.

        Parameters
        ----------
        A : SparsePauliOp
            Left operator.
        B : SparsePauliOp
            Right operator. Its adjoint B† is computed internally.
        real_result : bool, optional
            Unused — reserved for future enforcement of real-valued output.
        Name : str, optional
            Unused — reserved for debugging / labelling purposes.

        Returns
        -------
        complex
            Value of the inner product ⟨ψ|{A, B†}|ψ⟩.
        """
        Bc = B.adjoint()
        f = A @ Bc + Bc @ A
        f = relative_simplify_spo(f, self.eps)

        return self.operator_average_value(f)

    # -------------------------
    # Build H_tilde matrix
    # -------------------------
    def H_tilde_matrix(self, op: SparsePauliOp):
        """
        Build the projected Hamiltonian matrix H̃ in the bitstring subspace.

        Computes the (N × N) matrix of elements H̃ᵢⱼ = ⟨bᵢ|op|bⱼ⟩ where
        {|bᵢ⟩} are the sampled bitstring basis states. The result is used
        to diagonalize the Hamiltonian within the SQD subspace and extract
        approximate ground-state energies and eigenvectors.

        The implementation mirrors ``operator_average_value`` but omits the
        contraction with ``coeffs``, returning the full matrix instead of a
        scalar expectation value.

        Parameters
        ----------
        op : SparsePauliOp
            Hamiltonian (or any operator) to project onto the subspace,
            expressed as a sum of Pauli strings.

        Returns
        -------
        np.ndarray, shape (N, N), dtype complex
            Projected matrix H̃ᵢⱼ = ⟨bᵢ|op|bⱼ⟩.
        """

        P = np.array([list(label[::-1]) for label in op.paulis.to_labels()])
        p = op.coeffs

        N = self.N
        k = P.shape[0]

        b_iq = self.states[:, None, None, :]
        b_jq = self.states[None, :, None, :]
        P = P[None, None, :, :]

        container = np.zeros((N, N, k, self.q), dtype=complex)

        container += ((b_iq == 0) & (b_jq == 0) & (P == "I")) * 1
        container += ((b_iq == 1) & (b_jq == 1) & (P == "I")) * 1
        container += ((b_iq == 0) & (b_jq == 1) & (P == "X")) * 1
        container += ((b_iq == 1) & (b_jq == 0) & (P == "X")) * 1
        container += ((b_iq == 0) & (b_jq == 1) & (P == "Y")) * (-1j)
        container += ((b_iq == 1) & (b_jq == 0) & (P == "Y")) * (1j)
        container += ((b_iq == 0) & (b_jq == 0) & (P == "Z")) * 1
        container += ((b_iq == 1) & (b_jq == 1) & (P == "Z")) * (-1)

        container = np.prod(container, axis=-1)

        H_tilde = np.sum(container * p[None, None, :], axis=-1)

        return H_tilde
