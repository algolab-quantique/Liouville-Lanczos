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
from qiskit_nature.second_q import operators as op
from qiskit_nature.second_q.operators import commutators
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import BaseEstimatorV2, BaseSamplerV2
from qiskit_nature.second_q.mappers import QubitMapper

from qiskit.circuit.library import TwoLocal, RYGate, RZGate, CXGate
from qiskit.circuit import Instruction, Qubit
import numpy as np
from typing import Optional

from qiskit.transpiler import PassManager

from pauliarray import WeightedPauliArray, PauliArray
from pauliarray.partition.commutating_paulis.exclusive_fct import (  # partition_same_z,
    partition_general_commutating,
    partition_same_x,
    partition_same_x_plus_special,
)
from pauliarray.diagonalisation.commutating_paulis.with_circuits import (
    general_to_diagonal as general_to_diagonal_with_circuit,
)

from qiskit.quantum_info import SparsePauliOp

from ..Lanczos_components import Inner_product as Base_inner_product,Summation as Base_summation
from ..Lanczos_components import Liouvillian as BaseLiouvillian


def relative_simplify_spo(ope:SparsePauliOp,eps:float):
    """
    relative simplify truncates terms with a relative participation smaller than eps.
    Qiskit quantum info sparse pauli operator based implementation.
    performs much better than slo.
    """
    return ope.simplify(atol=1e-17,rtol=eps)

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

def evaluate_diag_expval(diag_obs: PauliArray, samples: dict[str: float]):
    #evaluate expval of diagonal pauli string on dictionary of samples
    expval = 0
    z_string = diag_obs.z_strings[0]
    x_string = diag_obs.x_strings[0]
    assert np.array_equal(x_string, [False]*diag_obs.num_qubits), 'Observable is not diagonal'
    for basis_state in samples:
        sign = 1
        for i in range(len(basis_state)):
            if z_string[-1-i] and basis_state[i] == '1':
                sign *= -1
        expval += sign * samples[basis_state]
    return expval
        

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
        
    def __call__(self,A:SparsePauliOp,B:SparsePauliOp,real_result:bool=False,Name:Optional[str]=None):
        """
        compute the innerproduct between that A and B operator.
        set real_result to True to bypass computation of an imaginary part in the result.
        Set a Name to easily identify the associated jobs on IBM quantum.
        """
        Bc = B.adjoint()
        f = A@Bc+Bc@A
        f = relative_simplify_spo(f,self.eps)
        obs_real, obs_imag = separate_imag(f)
        # print(f"A={A}\nB={B}\nf={f}")
        #imaginary contribution are necessarily error.
        try: #Add name to the list of tag for this job.
            if Name is not None:
                tags = self.estimator.options.environment.job_tags
                tags.append(Name)
                self.estimator.options.update(job_tags = tags)
        except:
            ...
        isa_obs_real = obs_real.apply_layout(self.state.layout)
        out = np.real(self.estimator.run([(self.state,isa_obs_real)]).result()[0].data.evs)
        if not real_result:
            isa_obs_imag = obs_imag.apply_layout(self.state.layout)
            out_imag = np.real(self.estimator.run([(self.state,isa_obs_imag)]).result()[0].data.evs)
            out += out_imag * 1j

        try: #remove the name from the list of tags of the upcoming jobs
            if Name is not None:
                tags = self.estimator.options.environment.job_tags
                tags = tags[:-1] # removes the appended name.
                self.estimator.options.update(job_tags = tags)
        except:
            ...
        return out[0]
    
class smart_inner_product_spo(Base_inner_product):
    def __init__(self, state: QuantumCircuit, sampler: BaseSamplerV2, epsilon: int, exp_dict = {}):
        self.state = state
        self.sampler = sampler
        self.eps = epsilon
        self.exp_dict = exp_dict
    
    def __call__(self,A:SparsePauliOp,B:SparsePauliOp,real_result:bool=False,Name:Optional[str]=None):
        Bc = B.adjoint()
        f = A@Bc+Bc@A
        f = relative_simplify_spo(f,self.eps)
        #obs_real, obs_imag = separate_imag(f)
        #check dictionary for previously calculated expectation values
        f_expval = 0
        labels_to_eval = []
        coeffs_to_eval = []
        for pauli in f:
            isa_pauli = pauli.apply_layout(self.state.layout)
            coeff = pauli.coeffs[0]
            pauli_string = str(isa_pauli.paulis[0])
            if pauli_string in self.exp_dict:
                str_expval = self.exp_dict[pauli_string]
                f_expval += str_expval * coeff
            else:
                labels_to_eval.append(pauli_string)
                coeffs_to_eval.append(coeff)

        try: #Add name to the list of tag for this job.
            if Name is not None:
                tags = self.sampler.options.environment.job_tags
                tags.append(Name)
                self.sampler.options.update(job_tags = tags)
        except:
            ...
        
        if len(labels_to_eval) != 0:
            to_eval = WeightedPauliArray.from_labels_and_weights(labels_to_eval, coeffs_to_eval)
            #group paulis into commuting cliques
            cliques = to_eval.partition_with_fct(partition_same_x_plus_special)
            #produce measurement circuits associated with each clique
            for clique in cliques:
                diag_part, factors_part, transformations_part_circuits = general_to_diagonal_with_circuit(clique.paulis)
                qc = self.state.compose(transformations_part_circuits)
                qc.measure_all()
                samples = self.sampler.run(qc).result().quasi_dists[0].binary_probabilities()
                #reconstruct expvals from samples, add to dictionary
                for i, obs in enumerate(clique.paulis):
                    expval = evaluate_diag_expval(diag_part[i], samples)
                    self.exp_dict[obs.to_labels()[0]] = expval
                    f_expval += factors_part[i] * expval
        return f_expval

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
        comm = H@A-A@H
        return relative_simplify_spo(comm,self.eps)

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

def relative_simplify_slo(ope:op.SparseLabelOp,eps:float):
    """
    relative simplify truncates terms with a relative participation smaller than eps.
    qiskit nature's sparse label op based implementation of relative simplify.
    Sparse label ops are very slow. Avoid.
    """
    v = max(ope.items(),key = lambda x: np.abs(x[1]))[1]
    if isinstance(ope, op.FermionicOp):
        ope = ope.normal_order()
    return ope.simplify(eps*v)

class inner_product_slo(Base_inner_product):
    """
    don't use this.
    """
    def __init__(self,state:QuantumCircuit,estimator:BaseEstimatorV2, mapper:QubitMapper,epsilon:int = 1e-10):
        self.state = state
        self.estimator = estimator
        self.eps = epsilon
        self.mapper = mapper
    
    def __call__(self,A:op.SparseLabelOp,B:op.SparseLabelOp, Name:Optional[str], real_result:bool = True):
        f = commutators.anti_commutator(A,B.adjoint())
        f = relative_simplify_slo(f,self.eps)
        obs = self.mapper.map(f)
        #separate between real and imaginary observables
        obs_real, obs_imag = separate_imag(obs)
        try: #Add name to the list of tag for this job.
            if Name is not None:
                tags = self.estimator.options.environment.job_tags
                tags.append(Name)
                self.estimator.options.update(job_tags = tags)
        except:
            ...
        #assume incoming circuit is transpiled
        isa_obs_real = obs_real.apply_layout(self.state.layout)
        out = np.real(self.estimator.run([(self.state,isa_obs_real)]).result()[0].data.evs)
        if not real_result:
            isa_obs_imag = obs_imag.apply_layout(self.state.layout)
            out_imag = np.real(self.estimator.run([(self.state,isa_obs_imag)]).result()[0].data.evs)
            out = out + out_imag * 1j
        
        try: #remove the name from the list of tags of the upcoming jobs
            if Name is not None:
                tags = self.estimator.options.environment.job_tags
                tags = tags[:-1] # removes the appended name.
                self.estimator.options.update(job_tags = tags)
        except:
            ...
        return out[0]

class Liouvillian_slo(BaseLiouvillian):
    """
    dont' use this
    """
    def __init__(self,eps = 1e-10):
        self.eps = eps
    def __call__(self,H,A):
        comm = commutators.commutator(H,A)
        return relative_simplify_slo(comm,self.eps)
    
class sum_slo(Base_summation):
    """
    don't use this.
    """
    def __init__(self,eps):
        self.eps = eps
    def __call__(self,*X):
        if len(X) > 2:
            A =  X[0]+X[1]+X[2]
        else:
            A = X[0]+X[1]
        return relative_simplify_slo(A,self.eps)