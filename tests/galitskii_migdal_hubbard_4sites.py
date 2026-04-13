# %%
from math import ceil

import numpy as np
import matplotlib.pyplot as plt
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

MAPPER = JordanWignerMapper()


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
    return MAPPER.map(-hopping + u * interaction - mu * occupation)


n = 6
u = 4
mu = u/2
max_iter = None

hopping = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
hamiltonian = hubbard(hopping, u, mu=mu)
h_mu = (mu) * np.diag(np.ones(n))
h_0 = -hopping - h_mu
eigvals, eigvecs = np.linalg.eigh(hamiltonian.to_matrix())
true_gs_energy = eigvals[0]
true_gs_vector = eigvecs[:, 0]

degen_gs_vectors = eigvecs[:,(eigvals-eigvals.min())<1e-9].T


from LiouvilleLanczos.Lanczos import Lanczos
from LiouvilleLanczos.Green import CF_Green, PolyLehmann_Green, Green_matrix
from LiouvilleLanczos.matrix_impl import (
    MatrixState_inner_product,
    Matrix_Liouvillian,
    Matrix_sum,
)


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


def annihilation_operators(n):
    c_fermi_list = [
        FermionicOp({f"-_{i}": 1}, num_spin_orbitals=2 * n) for i in range(n)
    ]
    return [MAPPER.map(c) for c in c_fermi_list]

def annihilation_operators_down(n):
    c_fermi_list = [
        FermionicOp({f"-_{i+n}": 1}, num_spin_orbitals=2 * n) for i in range(n)
    ]
    return [MAPPER.map(c) for c in c_fermi_list]


def get_abmu(ground_state, max_iter=10, n=4, opset="0-123", down=False):
    if down:
        c_operators = annihilation_operators_down(n)
    else:
        c_operators = annihilation_operators(n)
    
    main_i, other_i = opset.split("-")
    main_op = c_operators[int(main_i)]
    other_ops = [c_operators[int(i)] for i in other_i]

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

from scipy.sparse import csr_array
from typing import Callable
from LiouvilleLanczos.Green import integrable_Green_function_base


def fermi(w):
    return w <= 0.0



# Hardcode Green mapping lines for 2-3-4-5 sites
def green_mapping_line(n):
    if n == 2:
        opset = ["0-1"]
        green_mapping = [  # 0-1
            (0, 0, 0, 1),
            (1, 1, 0, 1),
            (0, 1, 1, 1),
            (1, 0, 1, 1),
        ]
    if n == 3:
        opset = ["0-12", "1-"]
        green_mapping = [  # 0-12 1    (4 Green)
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
    if n == 4:
        opset = ["0-123", "1-2"]
        green_mapping = [
            (0, 0, 0, 1),
            (3, 3, 0, 1),
            (0, 1, 1, 1),
            (2, 3, 1, 1),
            (3, 2, 1, 1),
            (1, 0, 1, 1),
            (0, 2, 2, 1),
            (1, 3, 2, 1),
            (3, 1, 2, 1),
            (2, 0, 2, 1),
            (0, 3, 3, 1),
            (3, 0, 3, 1),
            (1, 1, 4, 1),
            (2, 2, 4, 1),
            (1, 2, 5, 1),
            (2, 1, 5, 1),
        ]
    if n == 5:
        opset = ["0-1234", "1-23", "2-"]
        green_mapping = [
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
    if n == 6:
        opset = ["0-12345", "1-234", "2-3"]  ## 12 greens
        green_mapping = [
            #0-
            (0, 0, 0, 1),
            (5, 5, 0, 1),
            #0-1
            (0, 1, 1, 1),
            (1, 0, 1, 1),
            (4, 5, 1, 1),
            (5, 4, 1, 1),
            #0-2
            (0, 2, 2, 1),
            (2, 0, 2, 1),
            (3, 5, 2, 1),
            (5, 3, 2, 1),
            #0-3
            (0, 3, 3, 1),
            (3, 0, 3, 1),
            (5, 2, 3, 1),
            (2, 5, 3, 1),
            #0-4
            (0, 4, 4, 1),
            (4, 0, 4, 1),
            (5, 1, 4, 1),
            (1, 5, 4, 1),
            #0-5
            (0, 5, 5, 1),
            (5, 0, 5, 1),
            #1-
            (1, 1, 6, 1),
            (4, 4, 6, 1),
            #1-2
            (1, 2, 7, 1),
            (2, 1, 7, 1),
            (4, 3, 7, 1),
            (3, 4, 7, 1),
            #1-3
            (1, 3, 8, 1),
            (3, 1, 8, 1),
            (4, 2, 8, 1),
            (2, 4, 8, 1),
            #1-4
            (1, 4, 9, 1),
            (4, 1, 9, 1),
            #2-
            (2, 2, 10, 1),
            (3, 3, 10, 1),
            #2-3
            (2, 3, 11, 1),
            (3, 2, 11, 1),
        ]
    return opset, green_mapping


spin_gm_energies = []
for down in [True, False]:

    degen_gm_energies = []
    opset, mapping_line = green_mapping_line(n)
    for gs_vector in degen_gs_vectors:

        true_stack_green = make_green_stack(gs_vector, opset, 31, n, down=down)
        green = true_stack_green
        num_iterations = len(green)

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


omega = np.linspace(0, 31, 200)
real = omega
imag = np.full_like(omega, true_gs_energy)

plt.figure()

plt.plot(real, imag, label="true", color="black")
for s, spin in enumerate(['up', 'down']):
    for i, gs_vec in enumerate(degen_gs_vectors):
        plt.plot(spin_gm_energies[s,i], label=f"degen gs {i} {spin}", lw=3-0.4*(i+s*2))

plt.plot(spin_gm_energies.sum(0).mean(0), color="red", label=f"sum")
plt.ylabel("energy")
plt.xlabel("iteration")
plt.title(f"{n} sites")

plt.legend()
plt.show()

