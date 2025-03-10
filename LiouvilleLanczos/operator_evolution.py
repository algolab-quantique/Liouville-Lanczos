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
#%%
import pauliarray as pa
import numpy as np
from pauliarray.conversion.qiskit import operator_from_sparse_pauli

def pa_partition_Hamiltonian(H: pa.Operator, qubits: list[int]):
    new_paulis = []
    new_weights = []
    for i, pauli in enumerate(H.paulis.to_labels()):
        for qubit in qubits:
            if pauli[qubit] != 'I':
                new_paulis.append(pauli)
                new_weights.append(H.weights[i])
                break
    if len(new_weights) == 0:
        return None
    return pa.Operator.from_labels_and_weights(new_paulis, new_weights)

class obs_time_evolver():
    def __init__(self, sim = 'PauliArray'):
        self.sim = sim
    #assume first order trotterization for now
    def __call__(self, obs: pa.Operator, H: pa.Operator, timesteps: int, dt: float, eps: float):
        if self.sim == 'PauliArray':
            assert type(obs) == type(H) == pa.Operator, 'Observable and Hamiltonian must be PauliArray Operators'
            #identify locality of observable
            id = pa.Operator.from_labels_and_weights('I'*H.num_qubits,1)
            for t in range(timesteps):
                indices = set()
                for label in obs.paulis.to_labels():
                    for i, char in enumerate(label):
                        if char != 'I':
                            indices.add(i)
                H_t = pa_partition_Hamiltonian(H, indices)
                if H_t is None:
                    H_t = pa.Operator.from_labels_and_weights(H.paulis[0].to_labels(), H.weights[0]) #if H commutes with observable, choose a random pauli string
                for i, pauli in enumerate(H_t.paulis):
                    pauli_op = pa.Operator.from_paulis(pauli)
                    pauli_trotter_op = id.mul_scalar(np.cos(dt)) + pauli_op.mul_scalar(-1j*np.sin(dt))
                    obs = pauli_trotter_op.adjoint().compose_operator(obs.compose_operator(pauli_trotter_op).simplify()).simplify()
            obs = obs.remove_small_weights(eps)
            return obs
        raise NotImplementedError
#%%