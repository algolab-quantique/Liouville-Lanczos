# %%
import sys
sys.path.append('..')
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, PauliEvolutionGate  # TwoLocal, ZZFeatureMap, etc
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeQuebec
from qiskit_ibm_runtime import QiskitRuntimeService
from numbers import Number
from qiskit.synthesis import LieTrotter
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from qiskit_algorithms import VQE
from qiskit.primitives import StatevectorEstimator, StatevectorSampler, Estimator
from qiskit_algorithms.optimizers import SPSA, L_BFGS_B
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal,polar
from LiouvilleLanczos.Quantum_computer.VQE_stuff.ansatz import ControllableHEA, Real_NP_ansatz

 #%%
backend = FakeQuebec()
target = backend.target
#cm = target.build_coupling_map()
cm = backend.coupling_map
deg3_qubits = [
    idx for idx, row in enumerate(cm.distance_matrix) if list(row).count(1) == 3
]
edge_qubits = [0, 2, 6, 10, 18, 32, 37, 51, 56, 70, 75, 89, 94, 108, 116, 120, 124]
full_layering = [
    [edge for edge in cm.get_edges() if d3q in edge]
    for d3q in deg3_qubits + edge_qubits
]
ent_map = sum(
    [[layer[idx] for layer in full_layering if idx < len(layer)] for idx in range(3)],
    [],
)

def Heisenberg(J, n, ent_map):
    if isinstance(J, Number):
        J = np.ones(len(ent_map)) * J
    else:
        assert len(J) == len(ent_map)
    String = "I" * n
    H = SparsePauliOp("I" * n, 0)
    for j, c in zip(J, ent_map):
        c = np.sort(c)
        XX = String[: c[0]] + "X" + String[c[0] + 1 : c[1]] + "X" + String[c[1] + 1 :]
        H += SparsePauliOp(XX, j)
        YY = String[: c[0]] + "Y" + String[c[0] + 1 : c[1]] + "Y" + String[c[1] + 1 :]
        H += SparsePauliOp(YY, j)
        ZZ = String[: c[0]] + "Z" + String[c[0] + 1 : c[1]] + "Z" + String[c[1] + 1 :]
        H += SparsePauliOp(ZZ, j)
    return H.chop()

qubit_subset = [0, 1, 2, 3, 4, 14, 15, 18, 19, 20, 21, 22]
remap = {q: i for q, i in zip(qubit_subset, range(len(qubit_subset)))}
# filtering the entanglement map such that only the desired qubits are present.
f_ent_map = []
for a, b in ent_map:
    if a in qubit_subset and b in qubit_subset:
        f_ent_map.append((remap[a], remap[b]))
H = Heisenberg(1, 12, f_ent_map)
#GSE: -21.549563669780856
#%%

def prep_psi_0_with_checkpoints(qc: QuantumCircuit):
    qc.cx(12, 9)
    #
    qc.cx(9,10)
    #
    qc.cx(10,11)
    qc.cx(9,8)
    #
    qc.cx(11,6)
    qc.cx(8,7)
    #
    qc.cx(7,5)
    qc.cx(6,4)
    #
    qc.cx(4,3)
    qc.cx(5,0)
    #
    qc.cx(3,2)
    #make checkpoints
    qc.cx(11, 17)
    qc.cx(4, 16)
    qc.cx(2, 15)
    qc.cx(0, 14)
    qc.cx(7, 13)
    #undo entanglement on undesired qubits
    qc.cx(2,3)
    qc.cx(4,6)
    qc.cx(11,10)
    qc.cx(9,8)
    qc.cx(7,5)
    return qc

def prep_psi_0_by_0_with_checkpoints(qc: QuantumCircuit):
    qc.cx(12, 9, ctrl_state='0')
    qc.cx(17, 11, ctrl_state='0')
    qc.cx(16,4, ctrl_state='0')
    qc.cx(15,2, ctrl_state='0')
    qc.cx(14,0, ctrl_state='0')
    qc.cx(13,7, ctrl_state='0')
    return qc

#%%
#basis state has energy -12
from qiskit_aer.primitives import EstimatorV2 as mps_estimator
estim = mps_estimator(options={"backend_options": {"method":"matrix_product_state"}})
D = 10
circuits = np.ndarray([D,D], dtype= QuantumCircuit)
H_tilde = np.ndarray([D,D], dtype=complex)
S_tilde = np.ndarray([D,D], dtype=complex)
synth = LieTrotter(reps = 2)
delta_t = np.pi/40
time_evol = PauliEvolutionGate(H, delta_t, synthesis = synth)

real_obs_H = SparsePauliOp('XXXXXX', 1) ^ H
imag_obs_H = SparsePauliOp('YXXXXX', 1) ^ H
real_obs_S = SparsePauliOp('XXXXXX', 1) ^ SparsePauliOp('I'*12, 1)
imag_obs_S = SparsePauliOp('YXXXXX', 1) ^ SparsePauliOp('I'*12, 1)


for j in range(D):
    for i in range(j+1):
        m = i
        n = j - i
        qc = QuantumCircuit(18)
        qc.h(12)
        qc = prep_psi_0_with_checkpoints(qc)
        qc.append(PauliEvolutionGate(H, n*delta_t, synthesis=synth), range(12))  
        qc = prep_psi_0_by_0_with_checkpoints(qc)
        qc.append(PauliEvolutionGate(H, m*delta_t, synthesis=synth), range(12))          
        circuits[i][j] = qc.copy()

for i in range(D):
    for j in range(D):
        if circuits[i][j] is not None:
            res = estim.run([(circuits[i][j].decompose(), [real_obs_H, imag_obs_H, real_obs_S, imag_obs_S])])
            H_tilde[i,j] = res.result()[0].data.evs[0] + 1j * res.result()[0].data.evs[1]
            S_tilde[i,j] = res.result()[0].data.evs[2] + 1j * res.result()[0].data.evs[3]
        else:
            res = estim.run([(circuits[j][i].decompose(), [real_obs_H, imag_obs_H, real_obs_S, imag_obs_S])])
            H_tilde[i,j] = res.result()[0].data.evs[0] - 1j * res.result()[0].data.evs[1]
            S_tilde[i,j] = res.result()[0].data.evs[2] - 1j * res.result()[0].data.evs[3]
#%%
def truncation(threshold, S):
    eigvals, eigvecs = np.linalg.eig(S)
    idx = eigvals.argsort()[::-1]   
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    D = len(eigvals)
    truncated_eigvals = []
    truncated_eigvecs = []
    #make truncated matrix whose rows are eigenvectors of S with eigval above threshold
    for i in range(D):
        if eigvals[i] >= threshold:
            truncated_eigvals.append(eigvals[i])
            truncated_eigvecs.append(eigvecs[i])
    truncated_eigvals = np.array(truncated_eigvals)
    truncated_eigvecs = np.array(truncated_eigvecs).T    
    return truncated_eigvecs

#solve diagonalization problem, find dependence on dimension of krylov subspace. 
def E_vs_D(H_tilde, S_tilde, threshold_slope):
    GSEs = []
    ground_states = []
    for d in range(1, len(S_tilde)+1):
        epsilon = threshold_slope*d
        H_temp = H_tilde[0:d, 0:d]
        S_temp = S_tilde[0:d, 0:d]
        V_eps = truncation(epsilon, S_temp)
        #solve GEVP
        A = V_eps.conj().T @ H_temp @ V_eps
        B = V_eps.conj().T @ S_temp @ V_eps
        print(A.shape[0])
        E, c = scipy.linalg.eig(a = A, b = B)
        idx = E.argsort()[::-1]   
        eigvals = E[idx]
        eigvecs = c[:,idx]
        GSEs.append(eigvals[-1])
        ground_states.append(eigvecs[:,-1])
    return GSEs, ground_states, V_eps
GSEs, vecs, V_eps = E_vs_D(H_tilde, S_tilde, 0)
c = vecs[-1]
gammas = V_eps @ c
plt.plot(GSEs)
plt.title('Ground state energy vs Krylov dimension for fixed ')

#%%
from LiouvilleLanczos.Quantum_computer.QC_lanczos import krylov_inner_product_spo, Liouvillian_spo, sum_spo
from LiouvilleLanczos.Lanczos_components import QKB_geometry
def make_circuit_1(i, j, H, delta_t):
    m = i
    n = j - i
    qc = QuantumCircuit(18)
    qc.h(12)
    qc = prep_psi_0_with_checkpoints(qc)
    qc.append(PauliEvolutionGate(H, n*delta_t, synthesis=synth), range(12))  
    qc = prep_psi_0_by_0_with_checkpoints(qc)
    qc.append(PauliEvolutionGate(H, m*delta_t, synthesis=synth), range(12))          
    return qc.decompose()

class QKB_1Loop(QKB_geometry):
    def __init__(self, H, delta_t, D, gammas, S):
        self.system_pos = [i for i in range(12)]
        self.ancillas_pos = [i for i in range(12, 18)]
        self.num_qubits = 18
        self.H = H
        self.delta_t = delta_t
        self.D = D
        self.gammas = gammas
        self.S = S
    def make_circuit(self, i: int, j: int):
        return make_circuit_1(i, j, self.H, self.delta_t)

A = H
B = SparsePauliOp('I' * 12, 1/2)
qkb = QKB_1Loop(H, np.pi/40, 10, gammas, S_tilde)
kip = krylov_inner_product_spo(qkb, estim, 1e-6)
a = kip(A,B)
#get back krylov GSE estimation, as expected
#Now that we have Krylov basis decomposition, want to calculate Green's function
# %%
#Now, let's use this to calculate the Green's function. Let's simulate it first
from LiouvilleLanczos.Lanczos import Lanczos
from LiouvilleLanczos.matrix_impl import MatrixState_inner_product,Matrix_Liouvillian,Matrix_sum
from LiouvilleLanczos.Green import CF_Green
from qiskit.quantum_info import Statevector, random_unitary

w = np.linspace(-5.5,5.5,1000)-1e-1j
eps = 1e-6
SQ_inpro = kip
SQ_Liou = Liouvillian_spo(eps)
lanczos = Lanczos(SQ_inpro,SQ_Liou,sum_spo(eps))
a_sim5,b_sim5,mu_sim5 = lanczos.polynomial_hybrid(H,A,B,3,5e-3)
green_sim = CF_Green(a_sim5,b_sim5)
plt.plot(w,np.imag(green_sim(w)))
#%%
Hmat = H.to_matrix()
#generate ground state vectors
psi_mat = np.zeros(2**12, dtype = complex)
qc = QuantumCircuit(12)
for qb in [0, 2, 4, 7, 9, 11]:
    qc.x(qb)
psi_mat += gammas[0] * np.array(Statevector(qc))
for i in range(9):
    qc.append(time_evol, range(12))
    psi_mat += gammas[i+1] * np.array(Statevector(qc))
#%%
matrix_lanczos = Lanczos(MatrixState_inner_product(psi_mat),Matrix_Liouvillian(),Matrix_sum())
A = SparsePauliOp('I'*11 + 'Z', 1)
A_mat = A.to_matrix()
B = SparsePauliOp('Z' + 'I' * 11, 1)
B_mat = B.to_matrix()
a_ed,b_ed,mu_ed = matrix_lanczos.polynomial_hybrid(Hmat,A_mat,[B_mat],3)
green_ed = CF_Green(a_ed,b_ed)
plt.plot(w,np.imag(green_ed(w)))
# %%
#New Geometry
#make for marrakesh for now
#assume 156 qubit circuit input
def prep_psi_0_with_checkpoints_2_loops(qc: QuantumCircuit):
    #1
    qc.cx(24, 4)
    #2
    qc.cx(24,15)
    qc.cx(4,5)
    #3
    qc.cx(15,16)
    qc.cx(5,6)
    qc.cx(4,3)
    #4
    qc.cx(15,14)
    qc.cx(6,7)
    qc.cx(3,2)
    qc.cx(16,17)
    #5
    qc.cx(14,13)
    qc.cx(6,21)
    qc.cx(7,8)
    qc.cx(2,1)
    qc.cx(17,18)
    #6
    qc.cx(13,12)
    qc.cx(2,20)
    qc.cx(1,0)
    qc.cx(17,28)
    qc.cx(18,19)
    #7
    qc.cx(13,27)
    qc.cx(12,11)
    qc.cx(8,23)
    qc.cx(19,26)
    qc.cx(0,22)
    #8
    qc.cx(11,25)
    #undo entanglement on undesired qubits in parallel
    qc.cx(13,12)
    qc.cx(15,14)
    qc.cx(17,16)
    qc.cx(19,18)
    qc.cx(8,7)
    qc.cx(6,5)
    qc.cx(4,3)
    qc.cx(2,1)
    return qc

def prep_psi_0_by_0_with_checkpoints_2_loops(qc: QuantumCircuit):
    qc.cx(22, 0, ctrl_state='0')
    qc.cx(20,2, ctrl_state='0')
    qc.cx(21,6, ctrl_state='0')
    qc.cx(23,8, ctrl_state='0')
    qc.cx(26,19, ctrl_state='0')
    qc.cx(28,17, ctrl_state='0')
    qc.cx(27,13, ctrl_state='0')
    qc.cx(25,11, ctrl_state='0')
    qc.cx(24,4, ctrl_state='0')
    qc.cx(24,15, ctrl_state='0')
    return qc
# %%
#generate Hamiltonian for big loop geometry:
ent_map = []
remap = dict()
for i in range(65, 73):
    ent_map.append((i, i+1))
    remap[i] = i-65
remap[73] = 8
ent_map.append((77,65))
remap[77] = 9
ent_map.append((79,73))
remap[79] = 10
ent_map.append((84,77))
ent_map.append((92,79))
for i in range(84, 92):
    ent_map.append((i, i+1))
    remap[i] = i-73
remap[92] = 19
for i, qb in enumerate([57, 58, 64, 74, 78, 83, 93, 97, 98]):
    remap[qb] = i + 20
ent_map_sim = [(remap[ent_map[i][0]], remap[ent_map[i][1]]) for i in range(len(ent_map))]
H_sim = Heisenberg(1, 20, ent_map_sim)
#%%
#TEST INITIAL STATE ENERGY
from qiskit_aer.primitives import EstimatorV2 as mps_estimator
estim = mps_estimator(options={"backend_options": {"method":"matrix_product_state"}})
qc = QuantumCircuit(20)
for qb in [0, 2, 4, 6, 8, 11, 13, 15, 17, 19]:
    qc.x(qb)
res = estim.run([(qc, H_sim)])
print(res.result()[0].data.evs) #energy is -20
#test krylov method
#%%
def make_circuit_2(i: int, j: int, H, delta_t):
    qc = QuantumCircuit(21)
    qc.h(20)
    for target in [4, 15]:
        qc.cx(20, target)
    m = i
    n = j - i
    time_evol = PauliEvolutionGate(H, delta_t, synthesis = LieTrotter(reps = 1))
    for t in range(n):
        qc.append(time_evol, range(20))
    for target in [4, 15]:
        qc.cx(20, target, ctrl_state='0')
    for t in range(m):
        qc.append(time_evol, range(20))
    return qc

real_obs_H = SparsePauliOp('X', 1) ^ H_sim
imag_obs_H = SparsePauliOp('Y', 1) ^ H_sim
real_obs_S = SparsePauliOp('X', 1) ^ SparsePauliOp('I' * 20)
imag_obs_S = SparsePauliOp('Y', 1) ^ SparsePauliOp('I' * 20)
#%%
D = 10
H_tilde = np.zeros([D,D], dtype = complex)
S_tilde = np.zeros([D,D], dtype = complex)
for i in range(D):
    for j in range(i+1):
        qc = make_circuit_2(i, j, H_sim, np.pi/40)
        res = estim.run([(qc.decompose(), [real_obs_H, imag_obs_H, real_obs_S, imag_obs_S])])
        H_tilde[i,j] = res.result()[0].data.evs[0] + 1j * res.result()[0].data.evs[1]
        S_tilde[i,j] = res.result()[0].data.evs[2] + 1j * res.result()[0].data.evs[3]
        H_tilde[j,i] = res.result()[0].data.evs[0] - 1j * res.result()[0].data.evs[1]
        S_tilde[j,i] = res.result()[0].data.evs[2] - 1j * res.result()[0].data.evs[3]

# %%
GSEs, vecs, V_eps = E_vs_D(H_tilde, S_tilde, 0)
plt.plot(GSEs)
plt.title('Ground state energy vs Krylov dimension for fixed ')
#%%
#TEST new GFN protocol
from LiouvilleLanczos.Lanczos_components import QKB_geometry
from LiouvilleLanczos.Quantum_computer import krylov_inner_product_spo
from functools import partial
class QKB_2_loops(QKB_geometry):
    def __init__(self, H, delta_t, D, gammas, S):
        self.system_pos = [i for i in range(20)]
        self.ancillas_pos = [20]
        self.num_qubits = 20
        self.H = H
        self.delta_t = delta_t
        self.D = D
        self.gammas = gammas
        self.S = S
    def make_circuit(self, i: int, j: int):
        return make_circuit_2(i, j, self.H, self.delta_t)

qkb = QKB_2_loops(H_sim)
# %%
