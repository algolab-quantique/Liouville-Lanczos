from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from LiouvilleLanczos.Quantum_computer.QC_lanczos import relative_simplify_spo
import numpy as np
from LiouvilleLanczos.Lanczos_components import Inner_product as Base_inner_product
import time


#%%
def operator_average_value(f:SparsePauliOp, states, coeffs):
    """Compute the average value of a given operator over a set of quantum states.
    Args:
        f (SparsePauliOp): The operator represented as a SparsePauliOp.
        states (array): Array of quantum states represented as chars.
        coeffs (array): Vector of coefficients corresponding to each state.
    Returns:
        complex: The average value of the operator over the states.
    """
    p = np.array(f.coeffs)  # Vecteur de coefficients
    P = np.array([list(p[::-1]) for p in f.paulis.to_labels()]) 
    # Matrice des pauli characters 
    
    N, q = states.shape # nombre d'états et nombre de qubits
    k = len(P)  # nombre de termes de Pauli

    P = P[None, None, :, :]  # (1, 1, k, q)

    b_iq = states[:, None, None, :]   # (N, 1, 1, q)
    b_jq = states[None, :, None, :]   # (1, N, 1, q)  

    container = np.zeros((N, N, k, q), dtype=complex)
    
    start = time.perf_counter()
    container += ( (b_iq == '0') & (b_jq == '0') & (P == 'I') ) * 1 # Met 1 partout où les conditions sont remplies
    container += ( (b_iq == '1') & (b_jq == '1') & (P == 'I') ) * 1
    container += ( (b_iq == '0') & (b_jq == '1') & (P == 'X') ) * 1
    container += ( (b_iq == '1') & (b_jq == '0') & (P == 'X') ) * 1
    container += ( (b_iq == '0') & (b_jq == '1') & (P == 'Y') ) * (-1j)
    container += ( (b_iq == '1') & (b_jq == '0') & (P == 'Y') ) * (1j)
    container += ( (b_iq == '0') & (b_jq == '0') & (P == 'Z') ) * 1
    container += ( (b_iq == '1') & (b_jq == '1') & (P == 'Z') ) * (-1)
    end = time.perf_counter()
    print("container:" , f"{end - start:.6f} s")

    prod_q = np.prod(container, axis=-1)  
    result = np.sum(p[None, None, :] * np.conj(coeffs)[:, None, None] * coeffs[None, :, None] * prod_q)
    return result

class inner_product_spo_sqd(Base_inner_product):
    def __init__(self,
            states: np.ndarray,
            coeffs: np.ndarray, 
            epsilon: float,
    ):
        self.states = states
        self.coeffs = coeffs
        self.eps = epsilon
    def __call__(self, A:SparsePauliOp, B:SparsePauliOp, real_result=False, Name=None):

        Bc = B.adjoint()
        f = A@Bc+Bc@A
        f = relative_simplify_spo(f,self.eps)

        return operator_average_value(f, self.states, self.coeffs)
