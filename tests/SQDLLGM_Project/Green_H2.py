#%%
import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from qiskit.primitives import StatevectorEstimator as pEstimator
from qiskit.circuit.library import real_amplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import VQE
from qiskit import QuantumCircuit, transpile
from LiouvilleLanczos.Quantum_computer.QC_lanczos import Liouvillian_spo, sum_spo
from LiouvilleLanczos.Quantum_computer.sqd_lanczos import SampledSubspaceProjector
from LiouvilleLanczos.Lanczos import Lanczos
from LiouvilleLanczos.matrix_impl import MatrixState_inner_product,Matrix_Liouvillian,Matrix_sum
from LiouvilleLanczos.Green import CF_Green
import time
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
#%%
mapper = JordanWignerMapper()

# ===== H2 molecule 4 qubits =====
# num_qubits = 4
# driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.74', basis='sto3g', spin=0, charge=0)
# problem = driver.run()
# ferm_ham = problem.hamiltonian.second_q_op()

# ===== H2 molecule 8 qubits =====
num_qubits = 8
driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.74', basis='6-31g', spin=0, charge=0)
problem = driver.run()
transformer = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=4)
problem = transformer.transform(problem)
ferm_ham = problem.hamiltonian.second_q_op()

C0= FermionicOp(
    {
        "+_0": 1,
    },
    num_spin_orbitals=num_qubits,
)
C2= FermionicOp(
    {
        "+_2": 1,
    },
    num_spin_orbitals=num_qubits,
)

C0_spo = mapper.map(C0)
C2_spo = mapper.map(C2)
C0_mat = C0_spo.to_matrix()
C2_mat = C2_spo.to_matrix()
# mapping JW
HAM = mapper.map(ferm_ham)

# matrice
Hmat = HAM.to_matrix()
E, S = np.linalg.eigh(Hmat)
print("Ground state:", E[0])

gs_coeffs = S[:,0]
treshold = 1e-5
mask = np.abs(gs_coeffs) > treshold
print(f"Number of significant bitstrings in the ground state: {np.sum(mask)}")

# %%
gs_id = np.argmin(E)
sv = Statevector(S[:,gs_id])
#%%
# Sample states on quantum computer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

service = QiskitRuntimeService()
backend = service.backends()[0]
print(f"Using backend: {backend.name}")
#%%
ansatz = QuantumCircuit(num_qubits)
for q in range(num_qubits):
    ansatz.x(q)
ansatz = real_amplitudes(num_qubits=num_qubits, entanglement='linear')
ansatz.compose(ansatz, inplace=True)
optimizer = COBYLA(maxiter=1000)
estimator = pEstimator()
vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
result = vqe.compute_minimum_eigenvalue(HAM)
optimal_params = result.optimal_parameters
energie = result.eigenvalue.real
print("VQE Energy:", energie)
#%%
qc = ansatz.assign_parameters(optimal_params)
qc.measure_all()
qc = transpile(qc, backend)
sampler = Sampler(mode=backend)
job = sampler.run([qc], shots=100000)
result = job.result()

# === 4 qubits ===
# job = service.job("d6jlejo60irc7394m6hg")
# result = job.result()

# === 8 qubits ===
# job = service.job("d6j51je33pjc73dlinbg")
# result = job.result()

counts = result[0].data.meas.get_counts()     
#%%
bitstrings = list(counts.keys())
# check nb de 1 et rejeter les états qui en ont pas 2
bitstrings = [s for s in bitstrings if s.count('1') == 2]
bitstrings = list(set(bitstrings))
states = np.array(
    [[int(b) for b in s] for s in bitstrings],
    dtype=np.int8
)
def H_tilde_from_bitstring(states, hub_spo):

    P = np.array([list(label[::-1]) for label in hub_spo.paulis.to_labels()])
    p = hub_spo.coeffs
    N, q = states.shape
    k = P.shape[0]

    b_iq = states[:, None, None, :]
    b_jq = states[None, :, None, :]
    P = P[None, None, :, :]

    container = np.zeros((N,N,k,q), dtype=complex)
    container += ( (b_iq == 0) & (b_jq == 0) & (P == 'I') ) * 1 
    container += ( (b_iq == 1) & (b_jq == 1) & (P == 'I') ) * 1
    container += ( (b_iq == 0) & (b_jq == 1) & (P == 'X') ) * 1
    container += ( (b_iq == 1) & (b_jq == 0) & (P == 'X') ) * 1
    container += ( (b_iq == 0) & (b_jq == 1) & (P == 'Y') ) * (-1j)
    container += ( (b_iq == 1) & (b_jq == 0) & (P == 'Y') ) * (1j)
    container += ( (b_iq == 0) & (b_jq == 0) & (P == 'Z') ) * 1
    container += ( (b_iq == 1) & (b_jq == 1) & (P == 'Z') ) * (-1)

    container = np.prod(container, axis=-1)
    H_tilde = np.sum(container * p[None, None, :], axis=-1)
    return H_tilde 

H_tilde = H_tilde_from_bitstring(states, HAM)
e, v = np.linalg.eig(H_tilde)
gs_energie = np.min(e).real
coeffs = v[:, np.argmin(e)]
print("Energy from sampled states:", gs_energie)
print(len(bitstrings))
#%% Classical green's function

start = time.perf_counter()
matrix_lanczos = Lanczos(MatrixState_inner_product(sv.data),Matrix_Liouvillian(),Matrix_sum())
a_ed,b_ed,mu_ed = matrix_lanczos.polynomial_hybrid(Hmat, C0_mat,[C2_mat],10)
green_ed = CF_Green(a_ed,b_ed)
end = time.perf_counter()
print("Classical green time:", f"{end - start:.6f} s")
#%%
iterations_list = [10]
times = []
eps = 1e-3
for i in iterations_list:
    start = time.perf_counter()
    eval = SampledSubspaceProjector(states, coeffs, eps)
    avg_op = eval.inner_product_sqd
    SQ_Liou_avg = Liouvillian_spo(eps)
    lanczos_avg = Lanczos(avg_op,SQ_Liou_avg,sum_spo(eps))
    a_avg,b_avg,mu_avg = lanczos_avg.polynomial_hybrid(HAM, C0_spo,[C2_spo],i)
    green_sqd = CF_Green(a_avg,b_avg)
    end = time.perf_counter()
    times.append(end - start)
    print(f"Iterations = {i} | Temps = {end - start:.6f} s")

#%%
import matplotlib.pyplot as plt
w = np.linspace(-5.5,5.5,1000)-1e-1j
#%%
plt.plot(w,np.imag(green_ed(w)), label='Exact', color='blue')

plt.plot(w,np.imag(green_sqd(w)),'--', label='SQD', color='red')
# %%
list1 = ['00010001',
        '00010100',
        '00100010',
        '00101000', 
        '01000001',
        '01000100',
        '10000010',
        '10001000']
set2 = set(bitstrings)
comparison = [b in set2 for b in list1]
print(comparison)

# %%
