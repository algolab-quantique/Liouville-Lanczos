#%%
from LiouvilleLanczos.Quantum_computer.QC_lanczos import Liouvillian_spo, inner_product_spo, sum_spo
from LiouvilleLanczos.Quantum_computer.sqd_lanczos import inner_product_spo_sqd
from LiouvilleLanczos.Lanczos import Lanczos
from LiouvilleLanczos.matrix_impl import MatrixState_inner_product,Matrix_Liouvillian,Matrix_sum
from LiouvilleLanczos.Green import CF_Green
from qiskit.primitives import StatevectorEstimator as pEstimator
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit import QuantumCircuit
import qiskit
import numpy as np
from qiskit.circuit.library import real_amplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import VQE
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians.lattices import LineLattice
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import Statevector
import time
import matplotlib.pyplot as plt
#%%
mapper = JordanWignerMapper()
doubble_occup = FermionicOp(
    {
        "+_0 +_4 -_4 -_0": 1,
        "+_1 +_5 -_5 -_1": 1,
        "+_2 +_6 -_6 -_2": 1,
        "+_3 +_7 -_7 -_3": 1,
    },
    num_spin_orbitals=8,
)
first_hop = FermionicOp(
    {
        "+_0 -_1": 1,
        "+_1 -_0": 1,
        "+_1 -_2": 1,
        "+_2 -_1": 1,
        "+_2 -_3": 1,
        "+_3 -_2": 1,
        "+_4 -_5": 1,
        "+_5 -_4": 1,
        "+_5 -_6": 1,
        "+_6 -_5": 1,
        "+_6 -_7": 1,
        "+_7 -_6": 1,
    },
    num_spin_orbitals=8,
)
Number_op = FermionicOp(
    {
    '+_0 -_0':1,
    '+_1 -_1':1,
    '+_2 -_2':1,
    '+_3 -_3':1,
    '+_4 -_4':1,
    '+_5 -_5':1,
    '+_6 -_6':1,
    '+_7 -_7':1,
    },
    num_spin_orbitals=8
    )
C0u= FermionicOp(
    {
        "+_0": 1,
    },
    num_spin_orbitals=8,
)
C1u= FermionicOp(
    {
        "+_1": 1,
    },
    num_spin_orbitals=8,
)
C2u= FermionicOp(
    {
        "+_2": 1,
    },
    num_spin_orbitals=8,
)
C3u= FermionicOp(
    {
        "+_3": 1,
    },
    num_spin_orbitals=8,
)

C0_spo = mapper.map(C0u)
C1_spo = mapper.map(C1u)
C2_spo = mapper.map(C2u)
C3_spo = mapper.map(C3u)
C0_mat = C0_spo.to_matrix()
C1_mat = C1_spo.to_matrix()
C2_mat = C2_spo.to_matrix()
C3_mat = C3_spo.to_matrix()
t = -1
U = 4
mu = U/2
Hubbard_FOP = t*first_hop-mu*Number_op+U*doubble_occup
Hubbard_FOP
HAM = mapper.map(Hubbard_FOP)
Hubbard_matrix = HAM.to_matrix()
E,S = np.linalg.eigh(Hubbard_matrix)
print("Exact", E[0])
#%%
hub_spo = HAM
Hmat = Hubbard_matrix

#%%
# States from hamiltonian ground state
e,v = np.linalg.eig(Hmat)
gs_id = np.argmin(e)
sv = Statevector(v[:,gs_id])
n = sv.num_qubits

shots = 1000
samples = sv.sample_counts(shots)
bitstrings = list(samples.keys())
filtered_states = np.array(
    [[int(b) for b in s] for s in bitstrings],
    dtype=str
)
indices = np.array([int(s, 2) for s in bitstrings])
filtered_coeffs = sv.data[indices]
print("Bitstrings mesurés :")
print(filtered_states)
print("Coefficients mesurés :")
print(filtered_coeffs)

#%% Classical green's function
start = time.perf_counter()
matrix_lanczos = Lanczos(MatrixState_inner_product(sv.data),Matrix_Liouvillian(),Matrix_sum())
a_ed,b_ed,mu_ed = matrix_lanczos.polynomial_hybrid(Hmat, C0_mat,[C1_mat,C2_mat,C3_mat],30)
green_ed = CF_Green(a_ed,b_ed)
end = time.perf_counter()
print("Classical green time:", f"{end - start:.6f} s")

#%% SQD's green's function

iterations_list = [10]
times = []
eps = 1e-3
for i in iterations_list:
    start = time.perf_counter()
    avg_op = inner_product_spo_sqd(filtered_states,filtered_coeffs,eps)
    SQ_Liou_avg = Liouvillian_spo(eps)
    lanczos_avg = Lanczos(avg_op,SQ_Liou_avg,sum_spo(eps))
    a_avg,b_avg,mu_avg = lanczos_avg.polynomial_hybrid(hub_spo, C0_spo,[C1_spo,C2_spo,C3_spo],i)
    green_sqd = CF_Green(a_avg,b_avg)
    end = time.perf_counter()
    times.append(end - start)
    print(f"Iterations = {i} | Temps = {end - start:.6f} s")

# Plot
plt.figure()
plt.plot(iterations_list, times, marker='o')
plt.xlabel("Nombre d'itérations Lanczos")
plt.ylabel("Temps de calcul (s)")
plt.title("Temps de calcul vs nombre d'itérations")
plt.grid(True)
plt.show()
#%%
import matplotlib.pyplot as plt
w = np.linspace(-5.5,5.5,1000)-1e-1j
#%%
plt.plot(w,np.imag(green_ed(w)), label='Exact', color='blue')

plt.plot(w,np.imag(green_sqd(w)),'--', label='SQD', color='red')
#%%