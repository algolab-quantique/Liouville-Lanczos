# %%
from lanczos import Lanczos as LCZ
import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from LiouvilleLanczos.matrix_impl import MatrixState_inner_product, Matrix_Liouvillian, Matrix_sum
from QC_lanczos import Liouvillian_spo, sum_spo, inner_product_spo, relative_simplify_spo
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library import real_amplitudes
# %%

kmax = 6
eps = 1e-9

def apply_adjacent_fswap_permutation(qc):
    FSWAP = np.array([
            [1, 0, 0,  0],
            [0, 0, -1,  0],
            [0, 1, 0,  0],
            [0, 0, 0, 1],
        ], dtype=complex)
    
    fswap_gate = UnitaryGate(FSWAP, label="FSWAP")
    
    perm = [5,6,7,8,9,0,1,2,3,4]
    current = list(range(len(perm)))                # this just makes a list [0,1,2,3,4,5,...]
    for target_pos, desired_mode in enumerate(perm):        # this chooses the desired position and target position
        j = current.index(desired_mode)                     # find the index of the desired mode

        while j > target_pos:                               # while the index of the desired mode is larger than the target position
            qc.append(fswap_gate, [j - 1, j])               # swap it with the qubit to the left of it 
            current[j - 1], current[j] = current[j], current[j - 1] #update indices
            j -= 1    

vqe_hubbard_5sites = real_amplitudes(num_qubits=10, reps=10, entanglement='linear')
vqe_hubbard_5sites = vqe_hubbard_5sites.assign_parameters(np.array([
    2.3447832964919617, 1.4474302559938388, 4.713514444810458, 4.483509407584665, 4.400570018376737,
    3.1599828310535765, 1.4811927345460485, 4.401225733572901, 4.9385281687492535, 4.712960988556388,
    3.9345915095063786, 4.798725490293983, 1.6957962772935746, 4.18431690489243, 1.4314151293812927,
    3.6586171333767994, 2.0421913767282662, 4.861557641736541, 4.7384698726472285, 3.734752714234186,
    5.6471486574706375, 4.711408611937339, 4.747181489931575, 4.785705709657881, 5.773891260485987,
    3.164130952306159, 5.87030802541217, 4.62641468296399, 1.5715680039711153, 3.17431807296486,
    0.5479225386262592, 4.7125194894458895, 1.5707646540418294, 1.5708020948979795, 4.712386411886244,
    1.5707956120001854, 1.570800746348143, 4.712386246353408, 6.283183592137927, 6.283185024020641,
    6.283108640215091, -0.31483874554879887, 2.5684582292450258, 6.641278834606595, 1.595865508133264,
    3.5586234337709435, 1.6436999282262008, 1.5707985348741194, -2.0956374394721885e-06, -0.3551216158824308,
    3.1415901265735933, 4.413018428808857, 5.313840134725547, 4.478621591268556, 3.109855048054034,
    6.519653639112034, -0.14634612324162952, 4.712390805311075, 0.2152617755934342, 7.820990358428639,
    3.141596566559733, 4.315707796787452, 1.591058317291354, 1.5511314469306399, 4.654357447200541,
    4.708047365972247, 1.5707966504313826, 4.712389377489952, 4.710279668850846, 4.739490027393471,
    3.14158387406988, 4.71821903393061, 4.752704332942756, 4.680290507834347, 1.3224139337263787,
    3.6174009369047324, 1.5707939141135263, 1.5335737771548246, 4.690684139651661, 1.5408585947576945,
    9.324079134959437e-06, 1.6182910007664684, 1.6703109721089433, 1.4147301236346137, 1.5740252318995949,
    1.5707960492817332, 1.570798283653272, 1.5191349714558515, 1.6161172741980936, -0.31862288861427424,
    1.5707879571326253, 6.165949907011652, 6.6176881469443485, 1.2673449170192668, 3.6129609342647377,
    3.1415932962911555, 5.9963476937642755, 2.8726739404577013, 4.769961554828166, 2.9322930511089185,
    1.5707945163585162, 4.712389541091441, 4.712391063980809, 1.5707945358999216, 1.5707993205185902,
    4.7123863681724245, 1.5707962675025136, 4.712390122552536, 4.712389940305301, 4.712388510339969
]))

def to_sparse_pauli(op):
    MAPPER = JordanWignerMapper()
    """
    Convert FermionicOp -> SparsePauliOp.
    Leave SparsePauliOp unchanged.
    """
    if isinstance(op, SparsePauliOp):
        return op

    if isinstance(op, FermionicOp):
        return relative_simplify_spo(MAPPER.map(op),eps = eps)
    del MAPPER
    raise TypeError(f"Unsupported operator type: {type(op)}")

def hubbard(hop=np.array([[0, 1], [1, 0]]), u=4, mu=None):
    n = len(hop)
    if mu is None:
        mu = u / 2
    hopping = FermionicOp(
        {
            f"+_{i+m} -_{j+m}": hop[i, j]
            for m in [0, n]
            for i in range(n)
            for j in range(n)
            if hop[i, j] != 0
        },
        num_spin_orbitals=2 * n,
    )
    occupation = FermionicOp(
        {f"+_{i} -_{i}": 1 for i in range(2 * n)}, num_spin_orbitals=2 * n
    )
    interaction = FermionicOp(
        {f"+_{i} +_{i+n} -_{i+n} -_{i}": 1 for i in range(n)}, num_spin_orbitals=2 * n
    )
    HH = -hopping + u * interaction - mu * occupation
    
    C0 = FermionicOp(
        {
            "+_0": 1,
        },
        num_spin_orbitals=2*n,
    )
    return to_sparse_pauli(HH), to_sparse_pauli(C0)


hopping = np.diag(np.ones(5 - 1), 1) + np.diag(np.ones(5 - 1), -1)
hamiltonian, C0 = hubbard(hopping, 4, mu=2)     #|up, up, up, up, up, down, down, down, down, down>

eigvals, eigvecs = np.linalg.eigh(hamiltonian.to_matrix())

true_gs_energy = eigvals[0]
true_gs_vector = eigvecs[:, 0]
degen_gs_vectors = eigvecs[:, (eigvals - eigvals.min()) < 1e-9].T

def annihilation_operators(n = 5):
    MAPPER = JordanWignerMapper()
    c_fermi_list = [
        FermionicOp({f"-_{i}": 1}, num_spin_orbitals=2 * n) for i in range(n)
    ]
    return [MAPPER.map(c) for c in c_fermi_list]

def annihilation_operators_down(n = 5):
    MAPPER = JordanWignerMapper()
    c_fermi_list = [
        FermionicOp({f"-_{i+n}": 1}, num_spin_orbitals=2 * n) for i in range(n)
    ]
    return [MAPPER.map(c) for c in c_fermi_list]

opset = ["0-1234","1-23","2-"]

USE_IBM_BACKEND = False

#%%
if not USE_IBM_BACKEND and True:
    c_fermi_up = annihilation_operators_down()
    c_fermi_down = annihilation_operators()
    for i , c_fermi in enumerate([c_fermi_up, c_fermi_down]):
        fold = "matrix_up_five" if i == 0 else "matrix_down_five"
        for j, GS in enumerate(degen_gs_vectors):
            lanczos = LCZ(MatrixState_inner_product(GS),Matrix_Liouvillian(), Matrix_sum(), folder = fold, degen = j)
            for op in opset:
                main_i, other_i = op.split("-")
                i = int(main_i)
                main_op = c_fermi[i]
                other_ops = [c_fermi[int(i)] for i in other_i]
                lanczos.polynomial_hybrid(hamiltonian.to_matrix(),main_op.to_matrix(),[o.to_matrix() for o in other_ops],kmax)


#%%


GS1 = vqe_hubbard_5sites.copy()
GS2 = GS1.copy()
apply_adjacent_fswap_permutation(GS2)

if USE_IBM_BACKEND:
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as IBMEstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService()

    ibm_backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=vqe_hubbard_5sites.num_qubits,
    )

    print(f"Using IBM backend: {ibm_backend.name}")

    pm = generate_preset_pass_manager(
        backend=ibm_backend,
        optimization_level=1,
    )

    estimator = IBMEstimatorV2(mode=ibm_backend)
    estimator.options.resilience_level = 1
    estimator.options.default_precision = 0.03
    print(service.active_account())
    print(ibm_backend.name)
    print(ibm_backend.num_qubits)
    GS1_run = pm.run(GS1)
    GS2_run = pm.run(GS2)
else:
    ibm_backend = None
    pm = None
    estimator = AerEstimatorV2()
    GS1_run = GS1
    GS2_run = GS2

#%%
for i , c_fermi in enumerate([c_fermi_up, c_fermi_down]):
    fold = "inner_up_five" if i == 0 else "inner_down_five"
    for j, GS in enumerate([GS1,GS2]):
        lanczos = LCZ(inner_product_spo(GS,estimator,eps),Liouvillian_spo(eps),sum_spo(eps), folder = fold, degen = j)
        for op in opset:
            main_i, other_i = op.split("-")
            idx = int(main_i)
            main_op = c_fermi[idx]
            other_ops = [c_fermi[int(k)] for k in other_i]
            lanczos.polynomial_hybrid(hamiltonian ,main_op ,[o for o in other_ops],kmax)
            del main_i, other_i, idx, main_op, other_ops


# %%
