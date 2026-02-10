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
#%%
line = LineLattice(4)
hub = FermiHubbardModel(line, onsite_interaction = 1)
mapper = JordanWignerMapper()
hub = hub.second_q_op()
hub_spo = mapper.map(hub)

C0= FermionicOp(
    {
        "+_0": 1,
    },
    num_spin_orbitals=8,
)
C1= FermionicOp(
    {
        "+_2": 1,
    },
    num_spin_orbitals=8,
)
C2= FermionicOp(
    {
        "+_4": 1,
    },
    num_spin_orbitals=8,
)
C3= FermionicOp(
    {
        "+_6": 1,
    },
    num_spin_orbitals=8,
)
C0_spo = mapper.map(C0)
C1_spo = mapper.map(C1)
C2_spo = mapper.map(C2)
C3_spo = mapper.map(C3)
C0_mat = C0_spo.to_matrix()
C1_mat = C1_spo.to_matrix()
C2_mat = C2_spo.to_matrix()
C3_mat = C3_spo.to_matrix()

Hmat = hub_spo.to_matrix()
E,S = np.linalg.eigh(Hmat)
#%%
GS_4site = QuantumCircuit(8)
for q in [0,1,2,3]:
    GS_4site.x(q)
ansatz = real_amplitudes(num_qubits=8, entanglement='linear')
GS_4site.compose(ansatz, inplace=True)

#%% VQE to find ground state
estimator = pEstimator()
optimiser = COBYLA(maxiter=300)
vqe = VQE(estimator=estimator, ansatz=GS_4site, optimizer=optimiser)
res = vqe.compute_minimum_eigenvalue(hub_spo)

print("Exact GS energy:", E[0])
print("VQE energy:", res.eigenvalue.real)
GS_opt = GS_4site.assign_parameters(res.optimal_parameters)
GS_vec = Statevector.from_instruction(GS_opt).data


#%% Classical green's function
start = time.perf_counter()
matrix_lanczos = Lanczos(MatrixState_inner_product(GS_vec),Matrix_Liouvillian(),Matrix_sum())
a_ed,b_ed,mu_ed = matrix_lanczos.polynomial_hybrid(Hmat, C0_mat,[C1_mat,C2_mat,C3_mat],10)
green_ed = CF_Green(a_ed,b_ed)
end = time.perf_counter()
print("Classical green time:", f"{end - start:.6f} s")
#%% Quantum green's function
start = time.perf_counter()
eps = 1e-6
SQ_inpro = inner_product_spo(GS_opt,estimator,eps)
SQ_Liou = Liouvillian_spo(eps)
lanczos = Lanczos(SQ_inpro,SQ_Liou,sum_spo(eps))
a_sim5,b_sim5,mu_sim5 = lanczos.polynomial_hybrid(hub_spo, C0_spo,[C1_spo,C2_spo,C3_spo],10,5e-3)
green_sim = CF_Green(a_sim5,b_sim5)
end = time.perf_counter()
print("Quantum green time:", f"{end - start:.6f} s")
#%% SQD's green's function
start = time.perf_counter()
eps = 1e-6
statevector = qiskit.quantum_info.Statevector.from_instruction(GS_opt)
n = statevector.num_qubits
coeffs = statevector.data.copy()
idx = np.arange(2**n)
states = np.array([list(format(i,f'0{n}b')) for i in idx],dtype=str)
avg_op = inner_product_spo_sqd(states,coeffs,eps)
SQ_Liou_avg = Liouvillian_spo(eps)
lanczos_avg = Lanczos(avg_op,SQ_Liou_avg,sum_spo(eps))
a_avg,b_avg,mu_avg = lanczos_avg.polynomial_hybrid(hub_spo, C0_spo,[C1_spo,C2_spo,C3_spo],4,5e-3)
green_sqd = CF_Green(a_avg,b_avg)
end = time.perf_counter()
print("SQD's green time:", f"{end - start:.6f} s")
#%%
import matplotlib.pyplot as plt
w = np.linspace(-5.5,5.5,1000)-1e-1j
plt.plot(w,np.imag(green_sim(w)))

plt.plot(w,np.imag(green_ed(w)))

plt.plot(w,np.imag(green_sqd(w)),'--')
#%%