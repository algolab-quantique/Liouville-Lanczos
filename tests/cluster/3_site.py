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
    
    perm = [3,4,5,0,1,2]
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

hopping = np.diag(np.ones(3 - 1), 1) + np.diag(np.ones(3 - 1), -1)

hamiltonian, C0 = hubbard(hopping, 4, mu=2)

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

# %%
def analytical_greens(hamiltonian):
    n = 3       #3 sites
    
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

    opset = ["0-12", "1-"]

    greens = []
    for down in [True, False]:
        for gs_vector in degen_gs_vectors:
            true_stack_green = make_green_stack(gs_vector, opset, 31, n, down=down)
            greens.append(true_stack_green[-1,0])
    return greens, eigvals[0]


#%%
def galitski(hamiltonian, kmax, GS1, GS2):
    n = 3       #3 sites
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
    
    degen_gs_vectors = [GS1,GS2]
    eigvals, eigvecs = np.linalg.eigh(hamiltonian.to_matrix())
    true_gs_energy = eigvals[0]
    h_mu = (2) * np.diag(np.ones(n))
    h_0 = -hopping - h_mu
    opset = ["0-12", "1-"]
    mapping_line =[  # 0-12 1    (4 Green)
        (0, 0, 0, 1),
        (2, 2, 0, 1),
        (0, 1, 1, 1),
        (1, 0, 1, 1),
        (1, 2, 1, 1),
        (2, 1, 1, 1),
        (0, 2, 2, 1),
        (2, 0, 2, 1),
        (1, 1, 3, 1),
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
GS1 = vqe_hubbard_3sites.copy()

GS2 = GS1.copy()
apply_adjacent_fswap_permutation(GS2)

eps = 1e-6
kmax = int(sys.argv[1])
green, spin_gm_energies= galitski(hamiltonian=hamiltonian, kmax=kmax, GS1 = GS1, GS2 = GS2)
analytical, true_gs_energy = analytical_greens(hamiltonian=hamiltonian)


# %%


omega = np.linspace(0, 11, 200)
real = omega
imag = np.full_like(omega, true_gs_energy)

plt.figure()

plt.plot(real, imag, label="true", color="black")
plt.plot(spin_gm_energies.sum(0).mean(0), color="red", label=f"sum")
plt.ylabel("energy")
plt.xlabel("iteration")

plt.legend()


green1 = green[0]
green2 = green[1]
w = np.linspace(-5.5,5.5,1000)-1e-1j
x = np.real(w)

fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

axs[0].plot(x, (np.imag(green1(w))+ np.imag(green2(w))), color="blue", label = "Lanczos")
axs[0].plot(w, np.imag(analytical[0](w))+np.imag(analytical[1](w)), label = "GM", color = "red", linestyle = "--")
axs[0].set_title("green1 + green2")
axs[0].set_ylabel("Im G")
axs[0].legend()

axs[1].plot(x, np.imag(green1(w)), color="blue", label = "Lanczos")
axs[1].plot(w, np.imag(analytical[1](w)), label = "GM", color = "red", linestyle = "--")
axs[1].set_title("green1")
axs[1].set_ylabel("Im G")
axs[1].legend()

axs[2].plot(x, np.imag(green2(w)), color="blue", label = "Lanczos")
axs[2].plot(w, np.imag(analytical[0](w)), label = "GM", color = "red", linestyle = "--")
axs[2].set_title("green2")
axs[2].set_ylabel("Im G")
axs[2].set_xlabel(r"$\omega$")
axs[2].legend()
plt.tight_layout()
script_dir = Path(__file__).resolve().parent
plots_dir = script_dir / "plots"
plots_dir.mkdir(exist_ok=True)
plt.savefig(plots_dir / "greens.png", dpi=300, bbox_inches="tight")
