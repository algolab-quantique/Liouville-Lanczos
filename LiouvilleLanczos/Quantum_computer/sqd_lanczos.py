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
    states = np.asarray(states, dtype=np.int8)

    f_z = f.paulis.z.astype(np.int8)   # (k,q)
    f_x = f.paulis.x.astype(np.int8)

    ij_states = np.mod(states[:,None,:] + states[None,:,:],2)
    ijk_states_obs_x = np.mod(ij_states[:,:,None,:] + f_x[None,None,:,:],2)

    ijk_delta = np.all(ijk_states_obs_x == 0, axis=-1)   # (N,N,k)

    k_phase_y = np.choose(np.sum(f_z * f_x, axis=-1), [1, -1j, -1, 1j],mode='wrap')   
    ik_eigenvalues = np.choose(np.sum(states[:,None,:] * f_z[None,:,:], axis=-1), [1, -1],mode='wrap')   

    k_p = np.asarray(f.coeffs, dtype=complex)

    expval = np.einsum('i,j,k,ijk,k,ik->', np.conj(coeffs), coeffs, k_p, ijk_delta, k_phase_y, ik_eigenvalues)

    return complex(expval)

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
