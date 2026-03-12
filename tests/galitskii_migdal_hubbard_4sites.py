#%%
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
    hopping = FermionicOp({
            f"+_{i+m} -_{j+m}": hop[i, j] for m in [0, n]
            for i in range(n) for j in range(n) if hop[i, j] != 0
        }, num_spin_orbitals=2 * n)
    occupation = FermionicOp({
            f"+_{i} -_{i}": 1 for i in range(2 * n)
        }, num_spin_orbitals=2 * n)
    interaction = FermionicOp({
            f"+_{i} +_{i+n} -_{i+n} -_{i}": 1
            for i in range(n)
        }, num_spin_orbitals=2 * n)
    return MAPPER.map(-hopping + u * interaction - mu * occupation)

n=4
u=4
mu = None
max_iter = None

hopping = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
hamiltonian = hubbard(hopping, u)
h_mu = (u / 2) * np.diag(np.ones(n))
h_0 = -hopping - h_mu
eigvals, eigvecs = np.linalg.eigh(hamiltonian.to_matrix())
true_gs_energy = eigvals[0]
true_gs_vector = eigvecs[:, 0]


#%%

from LiouvilleLanczos.Lanczos import Lanczos
from LiouvilleLanczos.Green import CF_Green, PolyLehmann_Green, Green_matrix
from LiouvilleLanczos.matrix_impl import MatrixState_inner_product, Matrix_Liouvillian, Matrix_sum


def make_green_stack(opset_list=['0-123', '1-2'], max_iter=10, n=4, u=4, mu=None, energy=None, backend='exact', shots=np.inf, eps=1e-17, overwrite=False):
    stacked_green = None
    abm_dict = {}
    min_iter = 1
    for opset in opset_list:
        a, b, m = get_abmu(max_iter, n, u, mu, energy, backend, shots, eps, opset, overwrite)
        abm_dict[opset] = (a,b,m)
        min_iter = max(min_iter, len(a))
    
    for opset, (a, b, m) in abm_dict.items():
        green_list = make_green_list(a, b, m, min_iter)
        if stacked_green is None:
            stacked_green = green_list
        else:
            stacked_green = np.hstack([stacked_green, green_list]) # si ça converge plus vite que le critere on a un probleme
    return stacked_green # 6 Green par 30it

def annihilation_operators(n):
    c_fermi_list = [FermionicOp({f"-_{i}": 1}, num_spin_orbitals=2 * n) for i in range(n)]
    return [MAPPER.map(c) for c in c_fermi_list]

def get_abmu(max_iter=10, n=4, u=4, mu=None, energy=None, backend='exact', shots=np.inf, eps=1e-17, opset="0-123", overwrite=False):
    
    c_operators = annihilation_operators(n)
    
    
    main_i, other_i = opset.split('-')
    main_op = c_operators[int(main_i)]
    other_ops = [c_operators[int(i)] for i in other_i]

    if backend == 'exact': 
        _, eigvecs = np.linalg.eigh(hamiltonian.to_matrix())
        ground_state = eigvecs[:, 0]
        lanczos = Lanczos(
            MatrixState_inner_product(ground_state), 
            Matrix_Liouvillian(), 
            Matrix_sum(),
            logger=None)
        a, b, mm = lanczos.polynomial_hybrid(
            hamiltonian.to_matrix(), 
            main_op.to_matrix(), 
            [o.to_matrix() for o in other_ops], 
            max_iter)
        return a, b, mm
    
def make_green_list(a, b, mm, min_iter=1):
    mm = np.asarray(mm)
    num_iterations = len(a)
    num_k = max(num_iterations, min_iter)
    num_moments = len(mm[0])
    green_list = np.empty((num_k, num_moments + 1), dtype=object)
    for k in range(1, num_k + 1):
        kk = min(k, num_iterations)  ## fill the k > num_iteration with the converged CF_Green
        green_list[k - 1, 0] = CF_Green(a[:kk], b[:kk])
        green0 = green_list[k - 1, 0].to_Lehmann()
        for m in range(num_moments):
            green_list[k - 1, m + 1] = PolyLehmann_Green(a[:k], b[:k], mm[:k, m], green0)
    return green_list

true_stack_green = make_green_stack(['0-123', '1-2'], 31, n, u, backend='exact', eps=1e-17)
#%%
from scipy.sparse import csr_array
from typing import Callable
from LiouvilleLanczos.Green import integrable_Green_function_base


def fermi(w):
    return w <= 0.0

green = true_stack_green
h0 = h_0

def green_mapping_line(n):
    green_mapping = []
    l = 0
    for i in range(ceil(n / 2)):  # row in matrix
        for j in range(i, n - i):  # column in matrix
            green_mapping.append((i, j, l, 1))
            green_mapping.append((n - 1 - j, n - 1 - i, l, 1))
            if i != j and i != n - 1 - j:
                green_mapping.append((n - 1 - i, n - 1 - j, l, 1))
                green_mapping.append((j, i, l, 1))
            l += 1  # position in green list

    return green_mapping

# Galitskii_Migdal_energy
n = len(h0)
num_iterations = len(green)
energy = []
for k in range(0, num_iterations):
    green_list = [g for g in green[k, :]]
    mapping_line = green_mapping_line(n)
    
    # prepare_position_maps
    def convert_matpos(mat_pos):
            r,c,k = mat_pos
            assert r < n
            assert c < n
            assert k < len(green_list)
            return r+c*n,k
    R,C,D = np.zeros(len(mapping_line),dtype=np.int64),np.zeros(len(mapping_line),dtype=np.int64),np.zeros(len(mapping_line),dtype=np.complex128)
    for i,(il,jl,kl,coeff) in enumerate(mapping_line):
        R[i],C[i] = convert_matpos((il,jl,kl))
        D[i] = coeff
    position_maps = csr_array((D,(R,C)) ,shape=(n**2,len(green_list))) # (16, 6)

    scalar_frequency_weights_function = lambda x: fermi(x)
    matrix_frequency_constant = h0       
    GI = np.zeros(len(green_list))
    for i,g in enumerate(green_list):
        if not hasattr(g, "integrate"):
            g = g.to_Lehmann()
        GI[i] = g.integrate(scalar_frequency_weights_function)
    fm = matrix_frequency_constant.flatten()
    Kq = fm@position_maps@GI

    scalar_frequency_weights_function = lambda x: x * fermi(x)
    matrix_frequency_constant = np.eye(n)
    GI = np.zeros(len(green_list))
    for i,g in enumerate(green_list):
        if not hasattr(g, "integrate"):
            g = g.to_Lehmann()
        GI[i] = g.integrate(scalar_frequency_weights_function)
    fm = matrix_frequency_constant.flatten()
    wG = fm@position_maps@GI
    
    energy.append((wG + Kq).real)  # /2 missing because spin degeneracy

true_gm_energy = energy

plt.plot(true_gm_energy)

# %%
green_mapping_line(4)