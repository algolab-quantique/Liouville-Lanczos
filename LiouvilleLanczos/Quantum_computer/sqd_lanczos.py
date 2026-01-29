from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from LiouvilleLanczos.Quantum_computer.QC_lanczos import relative_simplify_spo
import numpy as np
from averageO import average_operator
from LiouvilleLanczos.Lanczos_components import Inner_product as Base_inner_product


#%%
class inner_product_spo_of_an_averageO(Base_inner_product):
    def __init__(self,
            estimator: StatevectorEstimator, 
            epsilon: int,
    ):
        self.estimator = estimator
        self.eps = epsilon
    def __call__(self, A:SparsePauliOp, B:SparsePauliOp, dict):

        Bc = B.adjoint()
        f = A@Bc+Bc@A
        f = relative_simplify_spo(f,self.eps)
        states = []
        coeffs = []

        for idx, coeff in enumerate(dict.data):
            if np.abs(coeff) > 1e-12:   
                bitstring = format(idx, f"0{dict.num_qubits}b")
                states.append([int(b) for b in bitstring])
                coeffs.append(coeff)

        return average_operator(f, states, coeffs)
#%%