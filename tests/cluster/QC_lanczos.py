"""
    Liouville-Lanczos: A library for Many-Body Green's function on quantum and classical computer.
    Copyright (C) 2024  Alexandre Foley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


"""
Components for the Liouvillian recursion method on a Quantum computer using Qiskit.
The lanczos algorithm implementation meant to use these components is located in Lanczos.py
"""

#%%
from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
import numpy as np
from typing import Optional
from qiskit_ibm_runtime import RuntimeJobFailureError
from qiskit.quantum_info import SparsePauliOp
from LiouvilleLanczos.Lanczos_components import Inner_product as Base_inner_product,Summation as Base_summation
from LiouvilleLanczos.Lanczos_components import Liouvillian as BaseLiouvillian
from collections import defaultdict


def relative_simplify_spo(ope:SparsePauliOp,eps:float, chunk_size:int = 50000):
    """
    relative simplify truncates terms with a relative participation smaller than eps.
    Qiskit quantum info sparse pauli operator based implementation.
    performs much better than slo.
    """
    n = ope.num_qubits
    acc = defaultdict(complex)

    for start in range(0, len(ope), chunk_size):
        stop = min(start + chunk_size, len(ope))
        sub = ope[start:stop]

        labels = sub.paulis.to_labels()
        coeffs = sub.coeffs

        for label, coeff in zip(labels, coeffs):
            if abs(coeff) > eps:
                acc[label] += complex(coeff)

    items = [(label, coeff) for label, coeff in acc.items() if abs(coeff) > eps]

    if not items:
        return SparsePauliOp.from_list([("I" * n, 0.0)], num_qubits=n)

    return SparsePauliOp.from_list(items, num_qubits=n)


def separate_imag(op: SparsePauliOp):
    """
    split a non hermiation operator in two hermitian operator.
    op = A+iB
    input: op
    returns: A,B
    """
    coeffs_real = []
    coeffs_img = []
    for coeff in op.coeffs:
        coeffs_real.append(np.real(coeff))
        coeffs_img.append(np.imag(coeff))
    pauli_op_real = SparsePauliOp(op.paulis, coeffs_real).chop()
    pauli_op_imag = SparsePauliOp(op.paulis, coeffs_img).chop()
    return pauli_op_real, pauli_op_imag



class inner_product_spo(Base_inner_product):
    """
    Operator Inner product implementation for sparse pauli operators.
    Compute $\\bra{\\psi} \\rho \\{A,B^\\dagger \\} \\ket{\\psi}$
    where $\\ket{\\psi}$ is a quantum state specified by a quantum circuit.
    $A$ and $B$ are quantum operators specified by sparse pauli operators.
    Uses ibm runtime estimatorV2 interface to submit tasks to quantum computers.
    """
    def __init__(self, state: QuantumCircuit, estimator: BaseEstimatorV2, epsilon: int):
        """
        Constructor for the inner product.
        requires
        - state: a quantum circuit that produce a desired state.
        - estimator: an EstimatorV2 estimator to submit jobs to a quantum computer.
        - epsilon: target accuracy, smaller relative values are truncated. 
        """
        self.state = state
        self.estimator = estimator
        self.eps = epsilon

    def _spo_has_terms(self, op: SparsePauliOp, eps: float) -> bool:
        return len(op) > 0 and np.any(np.abs(op.coeffs) >= eps)
    
    def _estimate_spo_chunked(self,
        observable: SparsePauliOp,
        eps: float,
        chunk_size: int = 1000,
        name: Optional[str] = None,
    ):
        """
        Estimate <state|observable|state> in chunks so Qiskit never converts
        one huge SparsePauliOp into a massive internal dict.
        """
        total = 0.0 + 0.0j

        n_terms = len(observable)
        for start in range(0, n_terms, chunk_size):
            stop = min(start + chunk_size, n_terms)

            obs_chunk = observable[start:stop]

            try:
                result = self.estimator.run([(self.state, obs_chunk)]).result()
                ev = result[0].data.evs
                total += complex(np.asarray(ev).item())
            except RuntimeJobFailureError as e:
                dump_qpu_error((self.state, obs_chunk), e, note=name or f"chunk {start}:{stop}")
                raise

        return total
        
    def __call__(self,A:SparsePauliOp,B:SparsePauliOp,real_result:bool=False,Name:Optional[str]=None):
        """
        compute the innerproduct between that A and B operator.
        set real_result to True to bypass computation of an imaginary part in the result.
        Set a Name to easily identify the associated jobs on IBM quantum.
        """
        Bc = B.adjoint()
        left = relative_simplify_spo(A@Bc, self.eps)
        right = relative_simplify_spo(Bc@A, self.eps)
        f = relative_simplify_spo(left + right ,self.eps)
        del left, right, Bc
        obs_real, obs_imag = separate_imag(f)        
        out = complex(0)
        if self._spo_has_terms(obs_real, self.eps):
            isa_obs_real = obs_real.apply_layout(self.state.layout)
            try:
                out += np.real(self._estimate_spo_chunked(isa_obs_real, self.eps, name=Name))
            except RuntimeJobFailureError as e:
                dump_qpu_error((self.state, isa_obs_real), e, note="real")
            del isa_obs_real

        if not real_result and any(np.abs(obs_imag.coeffs)>=self.eps):
            isa_obs_imag = obs_imag.apply_layout(self.state.layout)
            try:
                out_imag = np.real(self._estimate_spo_chunked(isa_obs_imag, self.eps, name=Name))
            except RuntimeJobFailureError as e:
                dump_qpu_error((self.state, isa_obs_imag), e, note="imag")
            del isa_obs_imag
            out += out_imag * 1j

        return out
    
    
        

def dump_qpu_error(pub, e, note=""):
    print(note)
    import matplotlib.pyplot as plt
    print(pub)
    pub[0].draw('mpl', fold=-1)
    plt.savefig("problematic_circuit.pdf")
    raise e

class Liouvillian_spo(BaseLiouvillian):
    """
    Sparse pauli operator based implementation of the Liouvillian.
    """
    def __init__(self,eps = 1e-10):
        """
        initialise with a relative precision. operator with smaller coefficients 
        are truncated.
        """
        self.eps = eps
    def __call__(self,H,A):
        """
        Compute the result of the Liouvillian for system with Hamiltonian H on 
        operator A.
        """
        left = relative_simplify_spo(H@A, self.eps)
        right = relative_simplify_spo(A@H, self.eps)
        return relative_simplify_spo(left-right,self.eps)

class sum_spo(Base_summation):
    """
    Sum up to 3 sparse pauli operator
    """
    def __init__(self,eps):
        """
        target relative precision of the results.
        """
        self.eps = eps
    def __call__(self,*X):
        """
        perform the sum
        """
        if len(X) > 2:
            A =  X[0]+X[1]+X[2]
        else:
            A = X[0]+X[1]
        return relative_simplify_spo(A,self.eps)
