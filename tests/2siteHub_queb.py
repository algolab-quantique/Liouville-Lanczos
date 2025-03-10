#%%
from LiouvilleLanczos.Quantum_computer.QC_lanczos import Liouvillian_slo,inner_product_slo,sum_slo
from LiouvilleLanczos.Quantum_computer.Hamiltonian import Line_Hubbard,BoundaryCondition
from LiouvilleLanczos.Lanczos import Lanczos
from LiouvilleLanczos.matrix_impl import MatrixState_inner_product,Matrix_Liouvillian,Matrix_sum
from LiouvilleLanczos.Quantum_computer.Mapping import find_best_layout
from LiouvilleLanczos.Green import CF_Green,Green_matrix,Lehmann_Green,PolyCF_Green,PolyLehmann_Green
from qiskit.primitives import StatevectorEstimator as pEstimator

from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit import QuantumCircuit
import numpy as np
from qiskit import transpile
from datetime import datetime
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

#%% problem hamiltonian and other operators setup.
U = 4
mapper = JordanWignerMapper()
Ham = Line_Hubbard(-1,U/2,U,2,boundary_condition=BoundaryCondition.OPEN)
#up spin site 1
C2 = FermionicOp(
    {
        "+_2": 1,
    },
    num_spin_orbitals=4,
)
#up spin site 0
C0 = FermionicOp(
    {
        "+_0": 1,
    },
    num_spin_orbitals=4,
)
C0_mat = mapper.map(C0).to_matrix()
C2_mat = mapper.map(C2).to_matrix()
#%% Ground state circuit, obtained by inspection of analytical wavefunction
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

#%% Sanity check: compare matrix ground energy with simulated estimator ground energy.
Hmat = mapper.map(Ham).to_matrix()
estimator = pEstimator()
qubit_converter = (JordanWignerMapper())
HHam = qubit_converter.map(Ham)
E,S = np.linalg.eigh(Hmat)
GS_mat = S[:,0]
print(E[0])
print(estimator.run([(GS_analytical,HHam)]).result()[0].data.evs)
#%% classical computation of the Green's function at site 0
matrix_lanczos = Lanczos(MatrixState_inner_product(GS_mat),Matrix_Liouvillian(),Matrix_sum())
a_ed,b_ed,mu_ed = matrix_lanczos.polynomial_hybrid(Hmat,C0_mat,[C2_mat],10)
green_ed = CF_Green(a_ed,b_ed)
# %% Quantum computer simulation
eps = 1e-6
SQ_inpro = inner_product_slo(GS_analytical,estimator,qubit_converter,eps)
SQ_Liou = Liouvillian_slo(eps)
lanczos = Lanczos(SQ_inpro,SQ_Liou,sum_slo(eps))
a_sim5,b_sim5,mu_sim5 = lanczos.polynomial_hybrid(Ham,C0,[C2],10,5e-3)
green_sim = CF_Green(a_sim5,b_sim5)
#%% We observe that the result are coherent.
import matplotlib.pyplot as plt
w = np.linspace(-5.5,5.5,1000)-1e-1j
plt.plot(w,np.imag(green_sim(w)))
# # plt.savefig("hubu4mu2.pdf")
plt.plot(w,np.imag(green_ed(w)))
#%%
from qiskit_ibm_runtime import (
    Session,
    Sampler,
    QiskitRuntimeService,
    Options,
    Estimator,
    EstimatorV2
)
from qiskit_ibm_runtime.options import (
    ExecutionOptions,
    EnvironmentOptions,
    ResilienceOptions,
    TranspilationOptions,
    EstimatorOptions
    
)

#<First time only>
#QiskitRuntimeService.save_account(channel="ibm_quantum", token="IBM_TOKEN")
#<\First time only>
#If you want to reproduce my result you will have to modify this to your own provider, 
# you may not have access to the same hardware, it might even not be in service anymore.
service_algolab = QiskitRuntimeService(
    channel="ibm_quantum",
    instance = "pinq-quebec-hub/iq-quantum/algolab"
)
#%%
#Sher = service_algolab.backend("ibm_sherbrooke")
Queb =  service_algolab.backend("ibm_quebec")
# kolk =  service_algolab.backend("ibmq_kolkata")
# hanoi =  service_algolab.backend("ibm_hanoi")
# auck =  service_algolab.backend("ibm_auckland")
#backends = {"ibm_sherbrooke":Sher,"ibm_quebec":Queb}
backends = {'ibm_quebec':Queb}

class logger:
    def __init__(self):
        self.operator_sizes = []
    def __call__(self,iteration,recursion_operator:SparsePauliOp,a_i,b_i):
        self.operator_sizes.append(len(recursion_operator))
log= logger()
#%%
#resilience=3 (PEC) doesn't work with ibm_sherbrooke because its ECR based instead of CNOT, and it's a significantly better quantum computer than the other options.
#resilience=2 (ZNE) implentation is very crappy: There's no readout error mitigation leading to typically worst result than resilience=1.That's because readout error is considerable and mostly independent of circuit details. ZNE can do little to nothing about that error source.
#resilience=1 is only readout error mitigation.
bkd = Queb.name
now = datetime.now()
with Session(backend=backends[bkd]) as session:
    estim_options = EstimatorOptions()
    backend = backends[session.backend()]
    estim_options.resilience_level = 1
    estim_options.default_shots=10000 #shot noise 10000 -> ~0.01, 100000 -> ~0.003
    estim_options.environment.job_tags = []
    estim_options.dynamical_decoupling.enable = True
    estim_options.dynamical_decoupling.sequence_type = 'XX'
    estim = EstimatorV2(backend = backend, session=session,options=estim_options)
    init_layout,GS_opt = find_best_layout(GS_analytical,backend,10,seed = 50)
    circuit = GS_analytical.copy()
    eps = 1e-5
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(circuit)
    isa_obs = HHam.apply_layout(isa_circuit.layout)
    Ground_state_E_job = estim.run([(isa_circuit, isa_obs)])
    produit_interne = inner_product_slo(isa_circuit, estim, qubit_converter, eps)
    Liou = Liouvillian_slo(eps)
    Summation = sum_slo(eps)
    lanczos = Lanczos(produit_interne,Liou,Summation,logger=log)
    a,b,mu = lanczos.polynomial_hybrid(Ham,C0,[C2],10,5e-2)
    E = Ground_state_E_job.result()[0].data.evs
now = datetime.now()
hour=now.hour
min =now.minute
sec = now.second
with open(f"V2_QC_hubbard2site_data_{bkd}_{hour}_{min}_{sec}.py",'w') as txt_file:
    txt_file.write(f"#executed on {now} \n")
    txt_file.write(f"#resilience level: {estim_options.resilience_level} \n")
    txt_file.write(f"#Dynamical decoupling: {estim_options.dynamical_decoupling.enable}, sequence: {estim_options.dynamical_decoupling.sequence_type} \n")    
    txt_file.write(f"#total execution time, with wait :{datetime.now()-now}\n")
    txt_file.write(f"#GS energy result = {E}\n")
    txt_file.write(f'"""\n {estim.options} \n"""\n')
    txt_file.writelines([f"a = {a}\n",f"b={b}\n",f"mu={mu}\n"])
    txt_file.writelines([f"operator_sizes={log.operator_sizes}\n"])


# %%
