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

kmax = 12
eps = 1e-9

def apply_adjacent_fswap_permutation(qc):
    FSWAP = np.array([
            [1, 0, 0,  0],
            [0, 0, -1,  0],
            [0, 1, 0,  0],
            [0, 0, 0, 1],
        ], dtype=complex)
    
    fswap_gate = UnitaryGate(FSWAP, label="FSWAP")
    
    perm = [3,4,5,0,1,2]
    current = list(range(len(perm)))                # this just makes a list [0,1,2,3,4,5,...]
    for target_pos, desired_mode in enumerate(perm):        # this chooses the desired position and target position
        j = current.index(desired_mode)                     # find the index of the desired mode

        while j > target_pos:                               # while the index of the desired mode is larger than the target position
            qc.append(fswap_gate, [j - 1, j])               # swap it with the qubit to the left of it 
            current[j - 1], current[j] = current[j], current[j - 1] #update indices
            j -= 1    

vqe_hubbard_3sites = real_amplitudes(num_qubits=6, reps=11, entanglement='linear')
vqe_hubbard_3sites = vqe_hubbard_3sites.assign_parameters(np.array([
    5.246829583426229, 1.8445563598975319, 6.249341224922228, 5.547426630865061, 4.204112886091582,
    4.7123853975763605, 0.27755320293694075, 0.32924783165591026, 1.5501725153481376, 0.354654347037912,
    1.4126950505240836, 7.524339982913663e-06, 1.5454883723953319, 4.008676262413297, 0.5471153440483539,
    0.6439116484454839, 5.416150750408826, 3.141590029386429, 1.121133004167295, 4.919561698711213,
    4.354025416990712, 4.348236205036469, 0.1627483139705849, 6.890458045940891e-06, 1.5750620500544459,
    3.925552761663296, 2.4735512439001406, 3.564021121427497, 2.1847082492649674, 3.1415875290047737,
    5.145406231818936, 3.082671747231, 4.755880471423207, 1.5314536798535008, 5.823307617043694,
    3.1416008591215223, 5.901892324799543, 1.4437881974849553, 4.645825634069647, 1.6387913496744664,
    5.2749649006268555, 1.1584242198623043e-05, 2.4788427008935474, 5.261826591438184, 5.982045041499495,
    2.24084148657242, 1.3918515401467277, 3.141590524117431, 2.6212585750135275, 0.0, 4.140488402169957,
    0.6887497275343819, 0.16071147559368984, 4.712382212081382, 5.6131238667553776, 0.2442058266515827,
    2.815716430300995, 1.696363607754504, 4.71239011907928, 6.283185307179586, 2.0669264858298853,
    2.441456446485303, 1.273233624886327, 2.646377569674503, 0.957628396457077, 1.8447208439653136,
    5.224543370694376, 1.583011829806925, 6.116083873171027, 6.0304071284479774e-06, 1.570785721175754,
    2.9229147479846883e-06
]))

def to_sparse_pauli(op):
    MAPPER = JordanWignerMapper()

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
    
    return to_sparse_pauli(HH)


hopping = np.diag(np.ones(3 - 1), 1) + np.diag(np.ones(3 - 1), -1)
hamiltonian = hubbard(hopping, 4, mu=2)     #|up, up, up, up, up, down, down, down, down, down>

eigvals, eigvecs = np.linalg.eigh(hamiltonian.to_matrix())

true_gs_energy = eigvals[0]
true_gs_vector = eigvecs[:, 0]
degen_gs_vectors = eigvecs[:, (eigvals - eigvals.min()) < 1e-9].T

c_fermi_down = [to_sparse_pauli(FermionicOp({f"-_{i+3}": 1}, num_spin_orbitals=2 * 3)) for i in range(3)]
c_fermi_up = [to_sparse_pauli(FermionicOp({f"+_{i+3}": 1}, num_spin_orbitals=2 * 3)) for i in range(3)]
opset = ["0-12", "1-"]



#%%
if True:
    for i , c_fermi in enumerate([c_fermi_up, c_fermi_down]):
        fold = "matrix_up_three" if i == 0 else "matrix_down_three"
        for j, GS in enumerate(degen_gs_vectors):
            lanczos = LCZ(MatrixState_inner_product(GS),Matrix_Liouvillian(), Matrix_sum(), folder = fold, degen = j)
            for op in opset:
                main_i, other_i = op.split("-")
                i = int(main_i)
                main_op = c_fermi[i]
                other_ops = [c_fermi[int(i)] for i in other_i]
                a,b,_ = lanczos.polynomial_hybrid(hamiltonian.to_matrix(),main_op.to_matrix(),[o.to_matrix() for o in other_ops],kmax)


#%%

GS1 = vqe_hubbard_3sites.copy()
GS2 = GS1.copy()
apply_adjacent_fswap_permutation(GS2)

greens = []
for i , c_fermi in enumerate([c_fermi_up, c_fermi_down]):
    fold = "inner_up_three" if i == 0 else "inner_down_three"
    fermi = []
    for j, GS in enumerate([GS1,GS2]):
        GROUND = []
        lanczos = LCZ(inner_product_spo(GS,AerEstimatorV2(),eps),Liouvillian_spo(eps),sum_spo(eps), folder = fold, degen = j)
        for op in opset:
            main_i, other_i = op.split("-")
            i = int(main_i)
            main_op = c_fermi[i]
            other_ops = [c_fermi[int(i)] for i in other_i]
            lanczos.polynomial_hybrid(hamiltonian ,main_op ,[o for o in other_ops],kmax)
            