from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from LiouvilleLanczos.Quantum_computer.QC_lanczos import relative_simplify_spo
import numpy as np
from LiouvilleLanczos.Lanczos_components import Inner_product as Base_inner_product
import time


#%%

class SampledSubspaceProjector:
    def __init__(self, states: np.ndarray, coeffs: np.ndarray, epsilon: float):
        """
        Utility class to evaluate operators in a bitstring basis.

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
        f = op

        f_z = f.paulis.z.astype(np.int8)
        f_x = f.paulis.x.astype(np.int8)

        ij_states = np.mod(self.states[:, None, :] + self.states[None, :, :], 2)
        ijk_states_obs_x = np.mod(ij_states[:, :, None, :] + f_x[None, None, :, :], 2)

        ijk_delta = np.all(ijk_states_obs_x == 0, axis=-1)

        k_phase_y = np.choose(
            np.sum(f_z * f_x, axis=-1),
            [1, -1j, -1, 1j],
            mode="wrap",
        )

        ik_eigenvalues = np.choose(
            np.sum(self.states[:, None, :] * f_z[None, :, :], axis=-1),
            [1, -1],
            mode="wrap",
        )

        k_p = np.asarray(f.coeffs, dtype=complex)

        ij_mat = np.einsum(
            "k,ijk,k,ik->ij",
            k_p,
            ijk_delta,
            k_phase_y,
            ik_eigenvalues,
        )

        expval = np.einsum(
            "i,j,ij",
            np.conj(self.coeffs),
            self.coeffs,
            ij_mat,
        )

        return complex(expval)
    
    # -------------------------
    # Inner product used in SQD
    # -------------------------
    def inner_product_sqd(self, A: SparsePauliOp, B: SparsePauliOp, real_result = False, Name=None):
        Bc = B.adjoint()
        f = A @ Bc + Bc @ A
        f = relative_simplify_spo(f, self.eps)

        return self.operator_average_value(f)
    
    # -------------------------
    # Build H_tilde matrix
    # -------------------------
    def H_tilde_matrix(self, op: SparsePauliOp):
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
   