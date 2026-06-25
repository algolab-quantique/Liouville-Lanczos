# %%
from qiskit_nature.second_q.operators import FermionicOp
from LiouvilleLanczos.Quantum_computer.QC_lanczos import Liouvillian_spo, sum_spo, inner_product_spo
from LiouvilleLanczos.Lanczos import Lanczos as LCZ
import numpy as np
from qiskit.circuit.library import real_amplitudes
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import SparsePauliOp
from LiouvilleLanczos.Green import CF_Green, PolyLehmann_Green
from qiskit.circuit.library import UnitaryGate
import matplotlib.pyplot as plt
from LiouvilleLanczos.matrix_impl import (
    MatrixState_inner_product,
    Matrix_Liouvillian,
    Matrix_sum,
)
import sys
from scipy.sparse import csr_array
from pathlib import Path
import csv
# %%
def to_sparse_pauli(op):
    MAPPER = JordanWignerMapper()
    """
    Convert FermionicOp -> SparsePauliOp.
    Leave SparsePauliOp unchanged.
    """
    if isinstance(op, SparsePauliOp):
        return op

    if isinstance(op, FermionicOp):
        return MAPPER.map(op).simplify(atol=1e-12)

    raise TypeError(f"Unsupported operator type: {type(op)}")

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

hamiltonian, C0 = hubbard(hopping, 4, mu=2)

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

# %%
def analytical_greens(hamiltonian):
    n = 5       #3 sites
    
    def annihilation_operators(n):
        c_fermi_list = [
            FermionicOp({f"-_{i}": 1}, num_spin_orbitals=2 * n) for i in range(n)
        ]
        return [to_sparse_pauli(c) for c in c_fermi_list]

    def annihilation_operators_down(n):
        c_fermi_list = [
            FermionicOp({f"-_{i+n}": 1}, num_spin_orbitals=2 * n) for i in range(n)
        ]
        return [to_sparse_pauli(c) for c in c_fermi_list]

    def get_abmu(ground_state, max_iter=10, n=4, opset="0-123", down=False):
        if down:
            c_operators = annihilation_operators_down(n)
        else:
            c_operators = annihilation_operators(n)
        
        main_i, other_i = opset.split("-")
        main_op = c_operators[int(main_i)]
        other_ops = [c_operators[int(i)] for i in other_i]

        lanczos = LCZ(
            MatrixState_inner_product(ground_state),
            Matrix_Liouvillian(),
            Matrix_sum(),
            logger=None,
        )
        a, b, mm = lanczos.polynomial_hybrid(
            hamiltonian.to_matrix(),
            main_op.to_matrix(),
            [o.to_matrix() for o in other_ops],
            max_iter,
        )
        return a, b, mm

    
    def make_green_list(a, b, mm, min_iter=1):
        mm = np.asarray(mm)
        num_iterations = len(a)
        num_k = max(num_iterations, min_iter)
        num_moments = len(mm[0])
        green_list = np.empty((num_k, num_moments + 1), dtype=object)
        for k in range(1, num_k + 1):
            kk = min(
                k, num_iterations
            )  ## fill the k > num_iteration with the converged CF_Green
            green_list[k - 1, 0] = CF_Green(a[:kk], b[:kk])
            green0 = green_list[k - 1, 0].to_Lehmann()
            for m in range(num_moments):
                green_list[k - 1, m + 1] = PolyLehmann_Green(
                    a[:k], b[:k], mm[:k, m], green0
                )
        return green_list

    def make_green_stack(ground_state, opset_list=["0-123", "1-2"], max_iter=10, n=4, down=False):
        stacked_green = None
        abm_dict = {}
        min_iter = 1
        opset_list = ["0-12", "1-"]
        for opset in opset_list:
            a, b, m = get_abmu(ground_state, max_iter, n, opset, down=down)
            abm_dict[opset] = (a, b, m)
            min_iter = max(min_iter, len(a))

        for opset, (a, b, m) in abm_dict.items():
            green_list = make_green_list(a, b, m, min_iter)
            if stacked_green is None:
                stacked_green = green_list
            else:
                stacked_green = np.hstack(
                    [stacked_green, green_list]
                )  # si ça converge plus vite que le critere on a un probleme
        return stacked_green  # 6 Green par 30it
    
    eigvals, eigvecs = np.linalg.eigh(hamiltonian.to_matrix())
    degen_gs_vectors = eigvecs[:,(eigvals-eigvals.min())<1e-9].T

    opset = opset = ["0-1234", "1-23", "2-"]

    greens = []
    for down in [True, False]:
        for gs_vector in degen_gs_vectors:
            true_stack_green = make_green_stack(gs_vector, opset, 31, n, down=down)
            greens.append(true_stack_green[-1,0])
    return greens, eigvals[0]


#%%
def galitski(hamiltonian, kmax, GS1, GS2):
    n = 5       #3 sites
    GS1 = GS1
    GS2 = GS2
    def annihilation_operators(n):
        c_fermi_list = [
            FermionicOp({f"-_{i}": 1}, num_spin_orbitals=2 * n) for i in range(n)
        ]
        return [to_sparse_pauli(c) for c in c_fermi_list]

    def annihilation_operators_down(n):
        c_fermi_list = [
            FermionicOp({f"-_{i+n}": 1}, num_spin_orbitals=2 * n) for i in range(n)
        ]
        return [to_sparse_pauli(c) for c in c_fermi_list]
    
    def fermi(w):
        return w <= 0.0

    def get_abmu(ground_state, max_iter=10, n=4, opset="0-123", down=False):
        if down:
            c_operators = annihilation_operators_down(n)
        else:
            c_operators = annihilation_operators(n)
        
        main_i, other_i = opset.split("-")
        main_op = c_operators[int(main_i)]
        other_ops = [c_operators[int(i)] for i in other_i]

        lanczos = LCZ(inner_product_spo(ground_state,AerEstimatorV2(),eps),Liouvillian_spo(eps),sum_spo(eps))

        a, b, mm = lanczos.polynomial_hybrid(hamiltonian,main_op, other_ops ,max_iter,1e-8)
        return a, b, mm

    
    def make_green_list(a, b, mm, min_iter=1):
        mm = np.asarray(mm)
        num_iterations = len(a)
        num_k = max(num_iterations, min_iter)
        num_moments = len(mm[0])
        green_list = np.empty((num_k, num_moments + 1), dtype=object)
        for k in range(1, num_k + 1):
            kk = min(
                k, num_iterations
            )  ## fill the k > num_iteration with the converged CF_Green
            green_list[k - 1, 0] = CF_Green(a[:kk], b[:kk])
            green0 = green_list[k - 1, 0].to_Lehmann()
            for m in range(num_moments):
                green_list[k - 1, m + 1] = PolyLehmann_Green(
                    a[:k], b[:k], mm[:k, m], green0
                )
        return green_list

    def make_green_stack(ground_state, opset_list=["0-123", "1-2"], max_iter=10, n=4, down=False):
        stacked_green = None
        abm_dict = {}
        min_iter = 1
        for opset in opset_list:
            a, b, m = get_abmu(ground_state, max_iter, n, opset, down=down)
            abm_dict[opset] = (a, b, m)
            min_iter = max(min_iter, len(a))

        for opset, (a, b, m) in abm_dict.items():
            green_list = make_green_list(a, b, m, min_iter)
            if stacked_green is None:
                stacked_green = green_list
            else:
                stacked_green = np.hstack(
                    [stacked_green, green_list]
                )  # si ça converge plus vite que le critere on a un probleme
        return stacked_green  # 6 Green par 30it
    
    degen_gs_vectors = [GS1,GS2]
    eigvals, eigvecs = np.linalg.eigh(hamiltonian.to_matrix())
    true_gs_energy = eigvals[0]
    h_mu = (2) * np.diag(np.ones(n))
    h_0 = -hopping - h_mu
    opset = ["0-1234", "1-23", "2-"]
    mapping_line = [
        (0, 0, 0, 1),
        (4, 4, 0, 1),
        (0, 1, 1, 1),
        (0, 1, 1, 1),
        (3, 4, 1, 1),
        (4, 3, 1, 1),
        (0, 2, 2, 1),
        (2, 0, 2, 1),
        (2, 4, 2, 1),
        (4, 2, 2, 1),
        (0, 3, 3, 1),
        (3, 0, 3, 1),
        (1, 4, 3, 1),
        (4, 1, 3, 1),
        (0, 4, 4, 1),
        (4, 0, 4, 1),
        (1, 1, 5, 1),
        (3, 3, 5, 1),
        (1, 2, 6, 1),
        (2, 1, 6, 1),
        (2, 3, 6, 1),
        (3, 2, 6, 1),
        (1, 3, 7, 1),
        (3, 1, 7, 1),
        (2, 2, 8, 1),
    ]
    greens = []

    spin_gm_energies = []

    for down in [True, False]:

        degen_gm_energies = []

        for gs_vector in degen_gs_vectors:
            true_stack_green = make_green_stack(gs_vector, opset, kmax, n, down=down)
            green = true_stack_green
            num_iterations = len(green)
            greens.append(true_stack_green[-1,0])
            energy = []
            for k in range(0, num_iterations):
                green_list = [g for g in green[k, :]]

                # prepare_position_maps
                def convert_matpos(mat_pos):
                    r, c, k = mat_pos
                    assert r < n
                    assert c < n
                    assert k < len(green_list)
                    return r + c * n, k

                R, C, D = (
                    np.zeros(len(mapping_line), dtype=np.int64),
                    np.zeros(len(mapping_line), dtype=np.int64),
                    np.zeros(len(mapping_line), dtype=np.complex128),
                )
                for i, (il, jl, kl, coeff) in enumerate(mapping_line):
                    R[i], C[i] = convert_matpos((il, jl, kl))
                    D[i] = coeff
                position_maps = csr_array((D, (R, C)), shape=(n**2, len(green_list)))  # (16, 6)

                scalar_frequency_weights_function = lambda x: fermi(x)
                matrix_frequency_constant = h_0
                GI = np.zeros(len(green_list))
                for i, g in enumerate(green_list):
                    if not hasattr(g, "integrate"):
                        g = g.to_Lehmann()
                    GI[i] = g.integrate(scalar_frequency_weights_function)
                fm = matrix_frequency_constant.flatten()
                Kq = fm @ position_maps @ GI

                scalar_frequency_weights_function = lambda x: x * fermi(x)
                matrix_frequency_constant = np.eye(n)
                GI = np.zeros(len(green_list))
                for i, g in enumerate(green_list):
                    if not hasattr(g, "integrate"):
                        g = g.to_Lehmann()
                    GI[i] = g.integrate(scalar_frequency_weights_function)
                fm = matrix_frequency_constant.flatten()
                wG = fm @ position_maps @ GI

                energy.append((wG + Kq).real/2) #missing because spin degeneracy

            degen_gm_energies.append(energy)
        spin_gm_energies.append(degen_gm_energies)
    spin_gm_energies = np.asarray(spin_gm_energies)

    return greens, spin_gm_energies
# %%

GS1 = vqe_hubbard_5sites.copy()

GS2 = GS1.copy()
apply_adjacent_fswap_permutation(GS2)

eps = 1e-6
kmax = int(sys.argv[1])
green, spin_gm_energies= galitski(hamiltonian=hamiltonian, kmax=kmax, GS1 = GS1, GS2 = GS2)
analytical, true_gs_energy = analytical_greens(hamiltonian=hamiltonian)


# %%

script_dir = Path(__file__).resolve().parent
plots_dir = script_dir / "plots" / "five_sites"
out_dir = script_dir / "outputs" / "five_sites"
plots_dir.mkdir(exist_ok=True)
out_dir.mkdir(exist_ok=True)

omega = np.linspace(0, kmax + 1, 200)
real = omega
imag = np.full_like(omega, true_gs_energy)
spin_energy = spin_gm_energies.sum(0).mean(0)

plt.figure()

plt.plot(real, imag, label="true", color="black")
plt.plot(spin_energy, color="red", label=f"sum")
plt.ylabel("energy")
plt.xlabel("iteration")

plt.legend()
plt.savefig(plots_dir / "energies.png", dpi=300, bbox_inches="tight")


w = np.linspace(-5.5,5.5,1000)-1e-1j
x = np.real(w)

green1 = green[0](w)
green2 = green[1](w)
analytical1 = analytical[0](w)
analytical2 = analytical[1](w)

fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

axs[0].plot(x, (np.imag(green1)+ np.imag(green2)), color="blue", label = "Lanczos")
axs[0].plot(w, np.imag(analytical1)+np.imag(analytical2), label = "GM", color = "red", linestyle = "--")
axs[0].set_title("green1 + green2")
axs[0].set_ylabel("Im G")
axs[0].legend()

axs[1].plot(x, np.imag(green1), color="blue", label = "Lanczos")
axs[1].plot(w, np.imag(analytical2), label = "GM", color = "red", linestyle = "--")
axs[1].set_title("green1")
axs[1].set_ylabel("Im G")
axs[1].legend()

axs[2].plot(x, np.imag(green2), color="blue", label = "Lanczos")
axs[2].plot(w, np.imag(analytical1), label = "GM", color = "red", linestyle = "--")
axs[2].set_title("green2")
axs[2].set_ylabel("Im G")
axs[2].set_xlabel(r"$\omega$")
axs[2].legend()
plt.tight_layout()
plt.savefig(plots_dir / "greens.png", dpi=300, bbox_inches="tight")



n_rows = max(len(omega), len(w))
path = out_dir / f"{kmax}_iterations.csv"
with open(path, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "energy_iteration",
        "true_gs_energy",
        "spin_gm_energy",

        "w_real",
        "w_imag",

        "green1_real",
        "green1_imag",
        "green2_real",
        "green2_imag",

        "analytical0_real",
        "analytical0_imag",
        "analytical1_real",
        "analytical1_imag",
    ])

    for i in range(n_rows):
        writer.writerow([
            omega[i] if i < len(omega) else np.nan,
            true_gs_energy if i < len(omega) else np.nan,
            spin_energy[i] if i < len(spin_energy) else np.nan,

            np.real(w[i]) if i < len(w) else np.nan,
            np.imag(w[i]) if i < len(w) else np.nan,

            np.real(green1[i]) if i < len(green1) else np.nan,
            np.imag(green1[i]) if i < len(green1) else np.nan,
            np.real(green2[i]) if i < len(green2) else np.nan,
            np.imag(green2[i]) if i < len(green2) else np.nan,

            np.real(analytical1[i]) if i < len(analytical1) else np.nan,
            np.imag(analytical1[i]) if i < len(analytical1) else np.nan,
            np.real(analytical2[i]) if i < len(analytical2) else np.nan,
            np.imag(analytical2[i]) if i < len(analytical2) else np.nan,
        ])
