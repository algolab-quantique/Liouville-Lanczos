# ===== System =====
import numpy as np
import matplotlib.pyplot as plt
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.drivers import PySCFDriver
import os
import warnings

warnings.filterwarnings("ignore")

MAPPER = JordanWignerMapper()

n = 7
driver = PySCFDriver(
    atom="O .0 .0 0.117790; H .0 0.755453 -0.471161; H .0 -0.755453 -0.471161",
    basis="sto3g",
    spin=0,
    charge=0,
)
problem = driver.run()
ferm_ham = problem.hamiltonian.second_q_op()

# mapping JW
hamiltonian = MAPPER.map(ferm_ham)

# matrice
HAM = problem.hamiltonian
h0_tensor = HAM.electronic_integrals.one_body.alpha.to_dense()
h0 = h0_tensor["+-"]
h0 = np.array(h0)

if os.path.exists("h2o_sto3g_gs.npz"):
    data = np.load("h2o_sto3g_gs.npz")
    Hmat = data["Hmat"]
    true_gs_energy = data["E0"]
    true_gs_vector = data["psi0"]
else:
    Hmat = hamiltonian.to_matrix()
    E, S = np.linalg.eigh(Hmat)
    true_gs_energy = E[0]
    true_gs_vector = S[:, 0]

    np.savez("h2o_sto3g_gs.npz", Hmat=Hmat, E0=true_gs_energy, psi0=true_gs_vector)
# %%
# ===== SQD =====
from qiskit.quantum_info import Statevector
from LiouvilleLanczos.Quantum_computer.sqd_lanczos import SampledSubspaceProjector

sv = Statevector(true_gs_vector)
sv.seed(42)
shots = 1000
samples = sv.sample_counts(shots)
bitstrings = list(samples.keys())
states = np.array([[int(b) for b in s] for s in bitstrings], dtype=int)
print(states)

eval = SampledSubspaceProjector(states)
H_tilde = eval.H_tilde_matrix(hamiltonian)
e, v = np.linalg.eig(H_tilde)
gs_energie = np.min(e).real
coeffs = v[:, np.argmin(e)]
print("Energy from sampled states:", gs_energie)


# ===== Liouville-Lanczos =====
from LiouvilleLanczos.Lanczos import Lanczos
from LiouvilleLanczos.Green import CF_Green, PolyLehmann_Green
from LiouvilleLanczos.matrix_impl import (
    MatrixState_inner_product,
    Matrix_Liouvillian,
    Matrix_sum,
)
from LiouvilleLanczos.Quantum_computer.QC_lanczos import Liouvillian_spo, sum_spo
from LiouvilleLanczos.Lanczos import Lanczos
from LiouvilleLanczos.Green import CF_Green, PolyLehmann_Green


def make_green_stack(
    opset_list=["0-123", "1-2"],
    max_iter=10,
    n=4,
    u=4,
    mu=None,
    energy=None,
    backend="exact",
    shots=np.inf,
    eps=1e-17,
    overwrite=False,
):
    stacked_green = None
    abm_dict = {}
    min_iter = 1
    for opset in opset_list:
        a, b, m = get_abmu(
            max_iter, n, u, mu, energy, backend, shots, eps, opset, overwrite
        )
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


def annihilation_operators(n):
    c_fermi_list = [
        FermionicOp({f"-_{i}": 1}, num_spin_orbitals=2 * n) for i in range(n)
    ]
    return [MAPPER.map(c) for c in c_fermi_list]


def get_abmu(
    max_iter=10,
    n=4,
    u=4,
    mu=None,
    energy=None,
    backend="exact",
    shots=np.inf,
    eps=1e-17,
    opset="0-123",
    overwrite=False,
):

    c_operators = annihilation_operators(n)

    main_i, other_i = opset.split("-")
    main_op = c_operators[int(main_i)]
    other_ops = [c_operators[int(i)] for i in other_i]

    if backend == "exact":
        _, eigvecs = np.linalg.eigh(hamiltonian.to_matrix())
        ground_state = eigvecs[:, 0]
        lanczos = Lanczos(
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
    if backend == "sqd":

        eval = SampledSubspaceProjector(states, coeffs, eps)
        avg_op = eval.inner_product_sqd
        SQ_Liou_avg = Liouvillian_spo(eps)
        lanczos = Lanczos(avg_op, SQ_Liou_avg, sum_spo(eps))

        a, b, mm = lanczos.polynomial_hybrid(
            hamiltonian,
            main_op,
            [o for o in other_ops],
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


true_stack_green = make_green_stack(
    [
        "0-123456",
        "1-023456",
        "2-013456",
        "3-012456",
        "4-012356",
        "5-012346",
        "6-012345",
    ],
    31,
    n,
    backend="exact",
    eps=1e-17,
)
sqd_stack_green = make_green_stack(
    [
        "0-123456",
        "1-023456",
        "2-013456",
        "3-012456",
        "4-012356",
        "5-012346",
        "6-012345",
    ],
    31,
    n,
    backend="sqd",
    eps=1e-17,
)

# ===== Plot Green fonction =====
green_ed = true_stack_green[0][0]
green_sqd = sqd_stack_green[0][0]

w = np.linspace(-5.5, 5.5, 1000) - 1e-1j
w_real = np.real(w)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    w_real,
    np.imag(green_ed(w)),
    color="steelblue",
    linewidth=1.8,
    label=r"Exact diagonalization",
)

ax.plot(
    w_real,
    np.imag(green_sqd(w)),
    color="firebrick",
    linewidth=1.5,
    linestyle="--",
    label=r"SQD approximation",
)

ax.set_xlabel(r"Frequency ($\omega$)", fontsize=13)
ax.set_ylabel(r"$\mathrm{Im}\, G_{00}(\omega)$", fontsize=13)
ax.set_title("Green Function — Exact vs SQD", fontsize=14, fontweight="bold")


ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)


ax.legend(fontsize=11, framealpha=0.9)


ax.margins(x=0.02)

fig.tight_layout()
# plt.savefig("green_function_comparison.pdf", dpi=300, bbox_inches="tight")
plt.show()


# ===== Galistkii-Migdal =====
from scipy.sparse import csr_array


def fermi(w):
    return w <= 0.0


green = sqd_stack_green


green_mapping_h2o = [
    (0, 0, 0, 1),
    (0, 1, 1, 1),
    (0, 2, 2, 1),
    (0, 3, 3, 1),
    (0, 4, 4, 1),
    (0, 5, 5, 1),
    (0, 6, 6, 1),
    (1, 0, 8, 1),
    (1, 1, 7, 1),
    (1, 2, 9, 1),
    (1, 3, 10, 1),
    (1, 4, 11, 1),
    (1, 5, 12, 1),
    (1, 6, 13, 1),
    (2, 0, 15, 1),
    (2, 1, 16, 1),
    (2, 2, 14, 1),
    (2, 3, 17, 1),
    (2, 4, 18, 1),
    (2, 5, 19, 1),
    (2, 6, 20, 1),
    (3, 0, 22, 1),
    (3, 1, 23, 1),
    (3, 2, 24, 1),
    (3, 3, 21, 1),
    (3, 4, 25, 1),
    (3, 5, 26, 1),
    (3, 6, 27, 1),
    (4, 0, 29, 1),
    (4, 1, 30, 1),
    (4, 2, 31, 1),
    (4, 3, 32, 1),
    (4, 4, 28, 1),
    (4, 5, 33, 1),
    (4, 6, 34, 1),
    (5, 0, 36, 1),
    (5, 1, 37, 1),
    (5, 2, 38, 1),
    (5, 3, 39, 1),
    (5, 4, 40, 1),
    (5, 5, 35, 1),
    (5, 6, 41, 1),
    (6, 0, 43, 1),
    (6, 1, 44, 1),
    (6, 2, 45, 1),
    (6, 3, 46, 1),
    (6, 4, 47, 1),
    (6, 5, 48, 1),
    (6, 6, 42, 1),
]

# Galitskii_Migdal_energy
n = len(h0)
num_iterations = len(green)
energy = []
for k in range(0, num_iterations):
    green_list = [g for g in green[k, :]]
    mapping_line = green_mapping_h2o

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
    position_maps = csr_array((D, (R, C)), shape=(n**2, len(green_list)))

    scalar_frequency_weights_function = lambda x: fermi(x)
    matrix_frequency_constant = h0
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

    energy.append((wG + Kq).real)  # /2 missing because spin degeneracy

true_gm_energy = energy


# ===== Plot G-M =====
iterations = np.arange(len(true_gm_energy))

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    iterations,
    true_gm_energy,
    color="steelblue",
    linewidth=1.8,
    marker="o",
    markersize=4,
    label=r"Computed energy $E_{\mathrm{SQD}}$",
)

ax.axhline(
    y=true_gs_energy,
    color="firebrick",
    linewidth=1.5,
    linestyle="--",
    label=r"Exact ground state $E_0 = $" + f"{true_gs_energy:.4f} Ha",
)

ax.set_xlabel("Iteration", fontsize=13)
ax.set_ylabel("Energy (Ha)", fontsize=13)
ax.set_title("Convergence of Ground State Energy", fontsize=14, fontweight="bold")

ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)

ax.legend(fontsize=11, framealpha=0.9)

ax.margins(x=0.02)

fig.tight_layout()
# plt.savefig("energy_convergence.pdf", dpi=300, bbox_inches="tight")  # pour le rapport
plt.show()


sv = true_gs_vector
probs = np.abs(sv) ** 2

# indices des 100 plus grandes contributions
top_k = 100
top_indices = np.argsort(probs)[-top_k:][::-1]

top_probs = probs[top_indices]
top_amplitudes = sv[top_indices]

n_qubits = 14
bitstrings = [format(i, f"0{n_qubits}b") for i in top_indices]

plt.figure()
plt.bar(range(top_k), top_probs)
plt.xlabel("Top 100 indices ")
plt.ylabel("Probabilité ")
plt.title("Top 100 contributions du statevector")
plt.show()

for i in range(10):
    print(
        f"{bitstrings[i]} : amplitude = {top_amplitudes[i]:.4f}, prob = {top_probs[i]:.4e}"
    )
