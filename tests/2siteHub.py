#%%
from LiouvilleLanczos.Quantum_computer.QC_lanczos import Liouvillian_slo,inner_product_slo,sum_slo

from LiouvilleLanczos.Quantum_computer.Hamiltonian import Line_Hubbard,BoundaryCondition
from importlib import reload

import LiouvilleLanczos.Lanczos
from LiouvilleLanczos.Lanczos import Lanczos
from LiouvilleLanczos.matrix_impl import MatrixState_inner_product,Matrix_Liouvillian,Matrix_sum
from LiouvilleLanczos.Quantum_computer.Mapping import find_best_layout
from LiouvilleLanczos.Green import CF_Green

reload(LiouvilleLanczos.Lanczos)
Lanczos = LiouvilleLanczos.Lanczos.Lanczos

from qiskit import qpy

from qiskit.primitives import Estimator as pEstimator
from qiskit.primitives import BackendEstimator

from qiskit_nature.second_q.mappers import JordanWignerMapper,QubitConverter
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit import QuantumCircuit
import numpy as np
from LiouvilleLanczos.Quantum_computer.err_mitig_inprod import twirled_inner_product 
from qiskit_research.utils.convenience import add_pauli_twirls,add_dynamical_decoupling,scale_cr_pulses,attach_cr_pulses
from qiskit_research.utils.pulse_scaling import BASIS_GATES
from LiouvilleLanczos.Quantum_computer.LayoutJordanWigner import JordanWignerlayout
from qiskit import transpile

#%%

U = 4
Ham = Line_Hubbard(-1,U/2,U,2,boundary_condition=BoundaryCondition.OPEN)
C0 = FermionicOp(
    {
        "+_0": 1,
    },
    num_spin_orbitals=4,
)
C0_mat = C0.to_matrix().toarray()
C0101 = FermionicOp(
    {
        "+_0 +_1": 1,
    },
    num_spin_orbitals=4,
).to_matrix().toarray()
C0110 = FermionicOp(
    {
        "+_0 +_3": 1,
    },
    num_spin_orbitals=4,
).to_matrix().toarray()
C1001 = FermionicOp(
    {
        "+_1 +_2": 1,
    },
    num_spin_orbitals=4,
).to_matrix().toarray()
C1010 = FermionicOp(
    {
        "+_2 +_3": 1,
    },
    num_spin_orbitals=4,
).to_matrix().toarray()
Hmat = Ham.to_matrix().toarray()
estimator = pEstimator()

qubit_converter = QubitConverter(JordanWignerMapper())
HHam = qubit_converter.convert(Ham)
void_state = np.zeros(2**4)
os2 = 1/np.sqrt(2)
void_state[0] = 1
States = [C0101@void_state,C0110@void_state,C1001@void_state,C1010@void_state]
Pn2s0 = np.concatenate([s.reshape(1,2**4) for s in States])
Hn2s0 = Pn2s0@Hmat@Pn2s0.T
Statesp0 = [ (States[1]-States[2])*os2,(States[0]+States[3])*os2]
Pn2s0p0 = np.concatenate([s.reshape(1,2**4) for s in Statesp0])
Hn2s0p0 = Pn2s0p0@Hmat@Pn2s0p0.T
E,S = np.linalg.eigh(Hmat)
Ep0, Sp0 = np.linalg.eigh(Hn2s0p0)
GS_mat = S[:,0]
print(E[0])
t = np.arctan2(Sp0[1,0].real,Sp0[0,0].real)
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
print(estimator.run(GS_analytical,HHam).result().values)
#%%
matrix_lanczos = Lanczos(MatrixState_inner_product(GS_mat),Matrix_Liouvillian(),Matrix_sum())
a_mat,b_mat = matrix_lanczos(Hmat,C0_mat,10)

#%%

# ansatz = SCA(Ham,2)
# powell = POWELL(200,2000)
# vqe = VQE(estimator,ansatz,powell)
# vqer = vqe.compute_minimum_eigenvalue(HHam)

# varqite = VarQITE(ansatz,vqer.optimal_parameters,estimator=estimator,num_timesteps=20,ode_solver=RK23)
# vqite_result = varqite.evolve(TimeEvolutionProblem(HHam,8.0,aux_operators=[HHam]))
# print(vqite_result.aux_ops_evaluated[-1])
# print(E[0])
# #%%
# with open("optimal_2sites.qpy",'wb') as qpyfile:
#     qpy.dump(vqer.optimal_circuit.assign_parameters(vqite_result.parameter_values[-1]),qpyfile)
#%%
with open("optimal_2sites.qpy",'rb') as qpyfile:
    GS = qpy.load(qpyfile)[0]
#%%
print(estimator.run(GS,HHam).result().values)
print(estimator.run(GS_analytical,HHam).result().values)
#%%
eps = 1e-6
lanczos = Lanczos(inner_product_slo(GS_analytical,estimator,qubit_converter,eps),Liouvillian_slo(eps),sum_slo(eps))
#%%
a_sim5,b_sim5 = lanczos(Ham,C0,10,5e-3)

# %%
# green = CF_Green(a,b)
# green_ed = CF_Green(a_mat,b_mat)

#%%
#On constate que ça marhe très bien en simulation
# import matplotlib.pyplot as plt
# w = np.linspace(-5.5,5.5,1000)-1e-1j
# plt.plot(w,np.imag(green(w)))
# plt.savefig("hubu4mu2.pdf")
# plt.plot(w,np.imag(green_ed(w)))
#%%
from qiskit_ibm_runtime import (
    Session,
    Sampler,
    QiskitRuntimeService,
    Options,
    Estimator,
)
from qiskit_ibm_runtime.options import (
    ExecutionOptions,
    EnvironmentOptions,
    ResilienceOptions,
    TranspilationOptions,
    
)

#<First time only>
# QiskitRuntimeService.save_account(channel="ibm_quantum", token="MY_IBM_QUANTUM_TOKEN")
#<\First time only>
service_charlebois = QiskitRuntimeService(
    channel="ibm_quantum", instance="ibm-q-qida/iq-quantum/charlebois"
)
service_algolab = QiskitRuntimeService(
    channel="ibm_quantum", instance="ibm-q-qida/iq-quantum/algolab"
)
#%%
Sher = service_algolab.backend("ibm_sherbrooke")
cair =  service_algolab.backend("ibm_cairo")
kolk =  service_algolab.backend("ibmq_kolkata")
hanoi =  service_algolab.backend("ibm_hanoi")
auck =  service_algolab.backend("ibm_auckland")
backends = {"ibmq_kolkata":kolk,"ibm_sherbrooke":Sher}


#%%
init_layout,GS_opt = find_best_layout(GS_analytical,Sher,10,seed = 50)
options = Options()
options.transpilation.initial_layout=init_layout
options.optimization_level = 3
options.resilience_level = 1
estim = Estimator(session=Sher,options=options)
job1 = estim.run([GS_analytical],[HHam])
job1.result().values
#%% exemple avec une session
options = Options()
options.optimization_level = 3
options.resilience_level = 1
with Session(backend=Sher) as session:
    backend = backends[session.backend()]
    init_layout,GS_opt = find_best_layout(GS_analytical,backend,10,seed = 50)
    options.transpilation.initial_layout=init_layout
    circuit = GS_analytical.copy()
    num_qubits = circuit.num_qubits
    options.environment.job_tags = ["resil1","Hubbard","opt_layout"]
    estim = Estimator(session=session,options=options)
    observable = HHam
    job1 = estim.run([circuit],[observable])
    job2 = estim.run([circuit]*2,[observable,observable-E*SparsePauliOp("I"*num_qubits)])
    E = job1.result().values[0]
    E2 = job2.result().values[0]
#%%
#%% compute eigenenergy with QC
options = Options()
options.optimization_level = 3
options.transpilation.approximation_degree=1.0
options.transpilation.seed = 0
Ntwirl = 20
with Session(backend=Sher) as session:
    backend = backends[session.backend()]
    init_layout,GS_opt = find_best_layout(GS_analytical,backend,10,seed = 50)
    session_qubit_converter = QubitConverter(JordanWignerlayout(init_layout,backend.configuration().n_qubits))
    options.resilience_level = 1
    # options.transpilation.initial_layout=init_layout
    options.environment.job_tags = ["resil1","twirled","E"]
    estim = Estimator(session=session,options=options)
    circuit = transpile(GS_opt,basis_gates=['sx','rz','cx','x'])
    PT_circs = add_pauli_twirls(circuit,Ntwirl)
    PT_circs = transpile(PT_circs, backend,initial_layout=init_layout)
    observable = session_qubit_converter.convert(Ham)
    job1 = estim.run(PT_circs,[observable]*Ntwirl)

#%% Manual ZNE

def Noise_amplification(circuit:QuantumCircuit,oddfactor:int):
    out = circuit.copy()
    assert(oddfactor%2 == 1)
    nqbit = circuit.num_qubits
    for i in range(oddfactor//2):
        out.barrier(range(nqbit))
        out.compose(circuit.inverse(),inplace=True)
        out.barrier(range(nqbit))
        out.compose(circuit,inplace=True)
    return out
#%%
options = Options()
options.optimization_level = 3
options.transpilation.approximation_degree=1.0
options.transpilation.seed = 0
Ntwirl = 20
with Session(backend=Sher) as session:
    backend = backends[session.backend()]
    init_layout,GS_opt = find_best_layout(GS_analytical,backend,10,seed = 50)
    options.transpilation.initial_layout=init_layout
    circuit = GS_analytical.copy()
    nqbit = circuit.num_qubits
    circuit1 = circuit.copy()
    circuit3 = Noise_amplification(circuit,3)
    circuit5 = Noise_amplification(circuit,5)
    circuit7 = Noise_amplification(circuit,7)
    options.resilience_level = 1
    # options.transpilation.initial_layout=init_layout
    options.environment.job_tags = ["resil1","ZNE","E"]
    estim = Estimator(session=session,options=options)
    observable = (HHam)
    job1 = estim.run([circuit1,circuit3,circuit5,circuit7],[observable]*4)
#%%
# with Session(backend=Sher) as session:
#     options = Options()
#     backend = backends[session.backend()]
#     options.optimization_level = 3
#     options.resilience_level = 2
#     options.environment.job_tags = ["Lanczos","ZNE","Test_output_structure"]
#     options.transpilation.approximation_degree=1.0
#     options.transpilation.seed = 0
#     options.execution.shots=10000
#     init_layout,GS_opt = find_best_layout(GS_analytical,backend,10,seed = 50)
#     options.transpilation.initial_layout=init_layout
#     circuit = GS_analytical.copy()
#     estim = Estimator(session=session,options=options)
#     eps = 1e-3 
#     lanczos = Lanczos(inner_product(circuit,estim,qubit_converter,eps),Liouvillian(eps),sum(eps))
#     # a,b = lanczos(Ham,C0,10,5e-2)
#     out_res2 = estim.run(GS_analytical,HHam)
#     options.resilience_level = 1
#     estim = Estimator(session=session,options=options)
#     out_res1 = estim.run(GS_analytical,HHam)

jobs = ['chjrdi5nopt07g2419ag','chjrdic6f7i49rouaq7g']

# %%
