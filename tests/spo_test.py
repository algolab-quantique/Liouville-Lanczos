#%%
from LiouvilleLanczos.Quantum_computer.QC_lanczos import Liouvillian_spo,smart_inner_product_spo,sum_spo, Liouvillian_slo, inner_product_slo, sum_slo
from LiouvilleLanczos.Quantum_computer.Hamiltonian import Line_Hubbard,BoundaryCondition
from LiouvilleLanczos.Lanczos import Lanczos
from LiouvilleLanczos.matrix_impl import MatrixState_inner_product,Matrix_Liouvillian,Matrix_sum
from LiouvilleLanczos.Quantum_computer.Mapping import find_best_layout
from LiouvilleLanczos.Green import CF_Green,Green_matrix,Lehmann_Green,PolyCF_Green,PolyLehmann_Green
from qiskit.primitives import StatevectorEstimator as pEstimator
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from LiouvilleLanczos.Quantum_computer.QC_lanczos import inner_product_spo
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit import QuantumCircuit
import numpy as np
from qiskit import transpile
from datetime import datetime
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
#%%
bt = 0.7854074074074073
GS_analytical = QuantumCircuit(4)
GS_analytical.h(0)
GS_analytical.x(1)
GS_analytical.cx(0,1)
GS_analytical.ry(bt,2)
GS_analytical.x(3)
GS_analytical.cx(2,3)
GS_analytical.cx(1,3)
GS_analytical.cx(1,2)
GS_analytical.cz(1,2)
GS_analytical.swap(1,2)
state = GS_analytical
A = SparsePauliOp(['XIII', 'IXII', 'IIXI', 'IIIX'], [1,2,3,4])
B = SparsePauliOp(['ZIII', 'IZII', 'IIZI', 'IIIZ'], [1,2,3,4])
# %%
sampler = StatevectorSampler(default_shots = 1000000)
estim = StatevectorEstimator()

spo = inner_product_spo(state, estim, 1e-6)
smart_spo = smart_inner_product_spo(state, sampler, 1e-6)
#%%
a = spo(A,B)
b = smart_spo(A,B)
#print(b)
# %%
