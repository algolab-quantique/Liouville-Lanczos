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
from qiskit.circuit.library import TwoLocal
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, ParameterVector, Qubit
from qiskit.circuit.parameterexpression import ParameterValueType, ParameterExpression

import numpy as np
from qiskit.circuit.library.standard_gates import (
    IGate,
    XGate,
    YGate,
    ZGate,
    RXGate,
    RYGate,
    RZGate,
    HGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    RXXGate,
    RYYGate,
    RZXGate,
    RZZGate,
    SwapGate,
    CXGate,
    CYGate,
    CZGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CHGate,
)

# from qiskit_research.mzm_generation.utils import transpile_circuit
# from qiskit_research.utils.pulse_scaling import 
# from qiskit_research.utils.pulse_scaling import cr_scaling_passes
ecr = QuantumCircuit(2)
ecr.ecr(0,1)
def flat(ham,reps=1, parameter_prefix: str = "θ"):
    nqbits = ham.num_spin_orbitals
    return TwoLocal(nqbits,['rx','ry'],ecr,'pairwise',reps,name="[rx,ry],ecr,sca,reps={}".format(reps),parameter_prefix=parameter_prefix)
def SCA(ham,reps=1, parameter_prefix: str = "θ"):
    nqbits = ham.num_spin_orbitals
    return TwoLocal(nqbits,['rx','ry'],ecr,'sca',reps,name="[rx,ry],ecr,sca,reps={}".format(reps),parameter_prefix=parameter_prefix)


def two_rots_layer(ham,layers,reps=1,parameter_prefix:str =  "θ"):
    nqbits = ham.num_spin_orbitals
    L = np.sum([len(l) for l in layers])*2
    par = ParameterVector(parameter_prefix,4*L*(reps+1))
    circ = QuantumCircuit(nqbits)
    par_count = 0
    for i in range(reps):
        for layer in layers:
            for (qbit0,qbit1) in layer:
                circ.rx(par[par_count],qbit0)
                circ.ry(par[par_count+1],qbit0)
                par_count += 2
                circ.rx(par[par_count],qbit1)
                circ.ry(par[par_count+1],qbit1)
                par_count += 2
                circ.compose(ecr,[qbit0,qbit1],inplace=True)
    for qbit in range(nqbits):
        circ.rx(par[par_count],qbit)
        circ.ry(par[par_count+1],qbit)
        par_count += 2
    return circ

def two_rots_line(ham,reps=1,parameter_prefix:str =  "θ"):
    nqbits = ham.num_spin_orbitals
    layers = [(i,i+1) for i in range(nqbits-1)]
    return two_rots_layer(ham,[layers],reps,parameter_prefix)

def two_rots_flat(ham,reps=1, parameter_prefix: str = "θ"):
    nqbits = ham.num_spin_orbitals
    even = [ (i,i+1) for i in range(0,nqbits-1,2)]
    odd = [ (i,i+1) for i in range(1,nqbits-1,2)]
    return two_rots_layer(ham,[even,odd],reps,parameter_prefix) 
# %%

class S():
    def __init__(self,n:int):
        self.num_spin_orbitals = n
T = S(4)
#%%
# two_rots_flat(T,1).draw()
# #%%
# two_rots_line(T,1).draw()
# # %%
class ControllableHEA(TwoLocal):
    
    def __init__(self, 
        num_qubits: int,
        ent_map,
        reps: int,
        skip_final_rotation_layer: bool = False,
        su2_gates: Instruction = None):

        if su2_gates is None:
            su2_gates = [RYGate]

        super().__init__(
            num_qubits=num_qubits,
            rotation_blocks=su2_gates,
            entanglement_blocks=CXGate,
            entanglement= [ent_map, ent_map[::-1]],
            reps= 2 * reps,
            skip_unentangled_qubits=False,
            skip_final_rotation_layer=skip_final_rotation_layer,
            parameter_prefix="θ",
            insert_barriers=True,
            initial_state=None,
            name= 'c-HEA',
            flatten=None,
        )
    
    def add_control(self, control_groups: list[tuple[int]]):
        #control_map is a list of groups of qubits controlled by the same qubit
        ctrl_map = [None]*self.num_qubits
        num_controls = len(control_groups)
        for i in range(num_controls):
            group = control_groups[i]
            for qubit in group:
                assert ctrl_map[qubit] == None, 'Each qubit can only have 1 control'
                ctrl_map[qubit] = i
        #new controlled circuit
        q_r = QuantumRegister(self.num_qubits + num_controls, 'q')
        controlled_HEA = QuantumCircuit(q_r)
        #qubit -1 is the control qubit for all single qubit operations
        control_qubits = [Qubit(q_r, q_r.size - 1 - i) for i in range(num_controls)]

        for instruction in self.decompose().data:
            print(instruction)
            gate = instruction.operation
            qubit_indices = [instruction.qubits[i]._index for i in range(gate.num_qubits)]
            qubits = [Qubit(q_r, index) for index in qubit_indices]
            if gate.num_qubits == 1:
                gate = gate.control()
                print(qubits[0])
                print(control_qubits[ctrl_map[qubit_indices[0]]])
                qubits = (control_qubits[ctrl_map[qubit_indices[0]]], qubits[0])
            controlled_HEA.append(gate, qubits)
        return controlled_HEA

def Real_NP_ansatz(n_qubits: int, reps: int, ent_map, param_name: str):
    qc = QuantumCircuit(n_qubits)
    n_2q_gates = len(ent_map)
    thetas = ParameterVector(param_name, n_2q_gates * reps)
    for i in range(len(thetas)):
        qubit_pair = ent_map[i%n_2q_gates]
        qc.cx(qubit_pair[0], qubit_pair[1])
        qc.cry(thetas[i], qubit_pair[1], qubit_pair[0])
        qc.cx(qubit_pair[0], qubit_pair[1])
    return qc



        


