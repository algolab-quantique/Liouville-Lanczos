from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from LiouvilleLanczos.Quantum_computer.QC_lanczos import relative_simplify_spo
import numpy as np
from LiouvilleLanczos.Lanczos_components import Inner_product as Base_inner_product
import time


#%%
def operator_average_value(f:SparsePauliOp, states:np.ndarray, coeffs:np.ndarray):
    """Compute the average value of a given operator over a set of quantum states.
    Args:
        f (SparsePauliOp): The operator represented as a SparsePauliOp.
        states (array): Array of quantum states represented as boolean.
        coeffs (array): Vector of coefficients corresponding to each state.
    Returns:
        complex: The average value of the operator over the states.
    """
    states = np.asarray(states).astype(np.int8)

    p = np.asarray(f.coeffs)

    X = f.paulis.x.astype(np.int8)   # (k,q)
    Z = f.paulis.z.astype(np.int8)

    b_iq = states[:,None,None,:]   # (N,1,1,q)
    b_jq = states[None,:,None,:]   # (1,N,1,q)

    # Delta : bi = bj xor Xk
    delta = (b_iq == (b_jq ^ X))
    delta = np.all(delta, axis=-1)    # (N,N,k)

    # Phases
    phase_z = (-1)**(Z & b_iq)
    phase_y = (-1j)**(Z & X)

    equation = phase_z * phase_y * delta[...,None]

    # Produit sur q
    container = np.prod(equation, axis=-1)   # (N,N,k)

    result = np.sum(p[None,None,:] * np.conj(coeffs)[:,None,None] * coeffs[None,:,None] * container)

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
