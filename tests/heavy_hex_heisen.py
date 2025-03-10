# %%
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2  # TwoLocal, ZZFeatureMap, etc
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeQuebec
from numbers import Number
import numpy as np
from qiskit_algorithms import VQE
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import SPSA, L_BFGS_B
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal,polar
from LiouvilleLanczos.Quantum_computer.VQE_stuff.ansatz import ControllableHEA, Real_NP_ansatz

def Lanczos_valh(H, psi0, eps, iter=200):
    psi_n = psi0 / np.sqrt(psi0 @ psi0)
    psi_nm = 0
    alpha_s = []
    beta_s = []
    alpha = 0
    beta = 0
    for i in range(iter):
        psi_nm = H @ psi_n - beta * psi_nm
        alpha = psi_n @ psi_nm
        alpha_s.append(alpha)
        psi_nm -= alpha * psi_n
        beta = np.sqrt(psi_nm @ psi_nm)
        psi_n, psi_nm = psi_nm / beta, psi_n
        Evals, Evecs = eigh_tridiagonal(alpha_s, beta_s)
        # print(Evals,beta)
        if abs(beta * Evecs[-1, -1]) < eps:
            break
        beta_s.append(beta)
    return Evals[0], beta * Evecs[-1, -1]


# %%
backend = FakeQuebec()
target = backend.target


# %%
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


# %%


from qiskit.visualization import plot_gate_map

# need to manually add edge qubits
plot_gate_map(backend, figsize=(7, 7), font_size=40, font_color="black")
# %% Nick's code to produce the ansatz for an entire Qcomp.
cm = target.build_coupling_map()
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
# create ansatz

ansatz_HEA = EfficientSU2(
    target.num_qubits, entanglement=ent_map, reps=3, skip_final_rotation_layer=True
)
print(f"Circuit has {ansatz_HEA.num_parameters} parameters")
ansatz_HEA.decompose().draw(fold=-1)

ansatz_CHEA = ControllableHEA(target.num_qubits, ent_map, reps = 2)
print(f"Circuit has {ansatz_CHEA.num_parameters} parameters")
ansatz_CHEA.decompose().draw(fold=-1)
# %%
# The subset of qubits i want to use.
qubit_subset = [0, 1, 2, 3, 4, 14, 15, 18, 19, 20, 21, 22]
remap = {q: i for q, i in zip(qubit_subset, range(len(qubit_subset)))}
# filtering the entanglement map such that only the desired qubits are present.
f_ent_map = []
for a, b in ent_map:
    if a in qubit_subset and b in qubit_subset:
        f_ent_map.append((remap[a], remap[b]))
H = Heisenberg(1, 12, f_ent_map)

# %%
estimator = Estimator()
spsa = SPSA(maxiter=600)
lbfgs = L_BFGS_B()
E_dict_ESU2 = {}
E_dict_NPA = {}
param_dict_ESU2 = {}
param_dict_NPA = {}

for rep in range(1, 10, 2):
    print(f"reps: {rep}")
    ansatz = EfficientSU2(
        len(qubit_subset), "ry", entanglement=f_ent_map, reps=rep, flatten=True
    )
    print(f"Circuit has {ansatz.num_parameters} parameters")
    # ansatz.draw(fold=-1)
    vqe = VQE(estimator, ansatz, spsa)
    sol = vqe.compute_minimum_eigenvalue(H)
    print("spsa Energy ", sol.optimal_value)
    vqe = VQE(estimator, ansatz, lbfgs, initial_point=sol.optimal_point)
    sol = vqe.compute_minimum_eigenvalue(H)
    E_dict_ESU2[rep] = sol.optimal_value
    param_dict_ESU2[rep] = sol.optimal_parameters
    print("bfgs energy ", sol.optimal_value)

    ansatz_NP = QuantumCircuit(12)
    for i in range(6):
        ansatz_NP.x(2 * i)
    ansatz_NP.append(Real_NP_ansatz(12, rep, f_ent_map), range(12))
    vqe = VQE(estimator, ansatz_NP, spsa)
    sol = vqe.compute_minimum_eigenvalue(H)
    print("spsa Energy ", sol.optimal_value)
    vqe = VQE(estimator, ansatz_NP, lbfgs, initial_point=sol.optimal_point)
    sol = vqe.compute_minimum_eigenvalue(H)
    E_dict_NPA[rep] = sol.optimal_value
    param_dict_NPA[rep] = sol.optimal_parameters
    print("bfgs energy ", sol.optimal_value)
# %%
E_dict_c_hea = {}
for rep in range(1, 5):
    print(f"reps: {rep}")
    ansatz = ControllableHEA(
        len(qubit_subset), ent_map=f_ent_map, reps=rep
    )
    print(f"Circuit has {ansatz.num_parameters} parameters")
    # ansatz.draw(fold=-1)
    vqe = VQE(estimator, ansatz, spsa)
    sol = vqe.compute_minimum_eigenvalue(H)
    print("spsa Energy ", sol.optimal_value)
    vqe = VQE(estimator, ansatz, lbfgs, initial_point=sol.optimal_point)
    sol = vqe.compute_minimum_eigenvalue(H)
    E_dict_c_hea[rep] = sol.optimal_value
    print("bfgs energy ", sol.optimal_value)
# %%
Sparse_H = H.to_matrix(True)
psi_0 = np.random.rand(Sparse_H.shape[0])
print(Lanczos_valh(Sparse_H.real, psi_0, 1e-4))
# %%
from qiskit import QuantumCircuit


def Qrylov(H: SparsePauliOp, qc: QuantumCircuit, order, tol=1e-4):
    """Very Slow, cannot do more than order 1 (2x2 matrix) for 12 site heisenberg"""
    from scipy.linalg import hankel

    Hn = H
    T = [1]
    estim = Estimator()
    for i in range(1, 2 * order + 1):
        T.append(estim.run(qc, Hn).result().values[0])
        # print(T[-1])
        Hn = (Hn @ H).chop(tol)
        print(len(Hn))
    estim.run(qc, Hn)
    T.append(estim.run(qc, Hn).result().values[0])
    S = hankel(T[:-1])[: order + 1, : order + 1]
    Ht = hankel(T[1:])[: order + 1, : order + 1]
    return S, Ht


aansatz = ansatz.assign_parameters(sol.optimal_point)
print(Estimator().run(aansatz, H).result().values[0])
S, KH = Qrylov(H / abs(sol.optimal_value), aansatz, 2)
# %%
import scipy as sp

sp.linalg.eigh(KH[:2, :2] * abs(sol.optimal_value), S[:2, :2])


# %%
def TQ_T(H: SparsePauliOp, qc, order, tol=1e-4):
    Hn = H
    Hnm = SparsePauliOp("I" * H.num_qubits)
    T = [1]
    estim = Estimator()
    for i in range(1, 2 * order + 1):
        T.append(estim.run(qc, Hn).result().values[0])
        Hnm = (2 * H @ Hn - Hnm).chop(tol)
        print(len(Hnm))
        Hn, Hnm = Hnm, Hn
    T.append(estim.run(qc, Hn).result().values[0])
    return T


def TQ_proj(T, order):
    # prepare the first row of the hamiltonian (beyond the actual full size we're able to fill)
    """
    run into numerical accuracy issue at 4x4 size projection.
    Filtering the weight matrix accomplishes nothing (GS energy estimation becomes
    worse than 3x3 projection).
    """
    Hp = np.zeros((order + 1, order + 1))
    Sp = np.zeros((order + 1, order + 1))
    for i in range(order + 1):
        for j in range(order + 1):
            Hp[i, j] = 0.25 * (
                T[i + j + 1]
                + T[abs(i + j - 1)]
                + T[abs(abs(i - j) - 1)]
                + T[abs(i - j) + 1]
            )
            Sp[i, j] = 0.5 * (T[i + j] + T[abs(i - j)])
    return Hp, Sp, T


def TQ(H: SparsePauliOp, qc, order, tol=1e-4):
    return TQ_proj(TQ_T(H, qc, order, tol), order)


# %%
Hp, Sp, T = TQ(H / (abs(sol.optimal_value) + 2), aansatz, 3, 1e-5)
# %%
print("overlap matrix:\n", Sp)
# %%
E, S = sp.linalg.eigh(Hp[:3, :3], Sp[:3, :3])
print(E[0] * (abs(sol.optimal_value) + 1))

# %%
import quimb.tensor as qtn
def rdn_b4_unitary_MPO():
    Xr = np.random.rand(4, 4)
    Xr, _, _ = np.linalg.svd(Xr)
    Xr = Xr.reshape(2,2,2,2)
    idxf= qtn.rand_uuid()+'{}'
    inid = 'in{}{}'
    outid = 'out{}{}'
    idxs = [inid.format(0,0),inid.format(0,1),outid.format(0,0),idxf.format(0)]
    Xr = qtn.Tensor(Xr,idxs)
    Xa,Xb = qtn.tensor_split(Xr,left_inds=idxs[0:3:2])
    Xa.add_tag('T0')
    Xb.add_tag('T1')
    Xr = Xa&Xb
    c_idx=[idxs[-1]]
    for i in range(1,4):
        idxs = [idxs[-1],inid.format(i,0),outid.format(i,0),idxf.format(i)]
        c_idx.append(idxs[-1])
        X = np.random.rand(4, 4)
        X, _, _ = np.linalg.svd(X)
        X = X.reshape(2,2,2,2)
        X = qtn.Tensor(X,idxs)
        Xa,Xb = qtn.tensor_split(X,left_inds=idxs[0:3:2])
        Xb.add_tag(f'T{i+1}')
        Xr = Xr&Xa&Xb
    Xr.reindex_({idxs[-1]:outid.format(3,1)})
    c_idx = c_idx[:-1]
    for id in c_idx:
        Xr.contract_ind(id)
    Xr.reindex_(dict({f'in{i}0':f'l{i+(i>0)}' for i in range(4)},**{'in01':'l1'}))
    Xr.reindex_(dict({f'out{i}0':f'u{i}' for i in range(4)},**{'out31':'u4'}))
    Xr = qtn.MatrixProductOperator([x.data for x in reversed(Xr.tensors_sorted())],'rdul')
    return Xr
Xr = rdn_b4_unitary_MPO()
Xr.draw()

# %%
def to_isometric_gauge(mpo:qtn.MatrixProductOperator,ic:int=None,fi = 'upper'):
    """
    Bring to the isometric gauge a unitary MPO. 
    Enforce that all tensor within a MPO are right isometric on the left of the 
    isometry center (ic) and left isometric on the right of the ic.
    A tensor is left(right) isometric when the contraction with its own conjugate
    of the selected free index (fi) and left(right) bond index gives the identity.
    If the overall MPO is a unitary, the center will be unitary or isometric in 
    this gauge.
    The local tensors can be projectingwhen split in half, those case can cause failure.
    """
    mpo= mpo.copy()
    if ic == None:
        ic  = mpo.L-1
    else:
        assert (ic < mpo.L and ic>0), ("the isometry tensor must be in "+
                                       f"[0, {mpo.L}[")
    if fi == "upper":
        fi = mpo.upper_ind_id
        oi = mpo.lower_ind_id
    elif fi == 'lower':
        fi = mpo.lower_ind_id
        oi = mpo.upper_ind_id
    else:
        assert (False), ("free index must be either upper or lower")
    r2 = np.sqrt(2)
    ir2 = 1/r2
    for i in range(ic):
        right_bond = set(mpo[i].inds).intersection(mpo[i+1].inds).pop()
        W,V = qtn.tensor_split(mpo[i],left_inds = None,right_inds=[right_bond],absorb='right')
        Wp,s,Vp = qtn.tensor_split(W,left_inds = [oi.format(i)],absorb=None)
        # factor = np.sqrt(W.ind_size(set(W.inds).intersection(V.inds).pop()))
        # factor= 1
        assert (np.allclose(s.data,s.data[0])),"failure to isometrize, is the MPO unitary?"

        V.drop_tags()
        # Wp = Wp@Vp
        mpo[i] = W
        mpo[i+1] = mpo[i+1]@V
    # for i in range(mpo.L-1,ic+1,-1):
    #     ...
    return mpo
Xri = to_isometric_gauge(Xr)
def to_qubit_operator(tensor: qtn.Tensor,bond_inds:list[str], left_inds: list[str], right_inds=None):
    """
    convert an isometric tensor in the bond direction to an unitary in the quantum hilbert space
    """
    if left_inds is None:
        assert (
            right_inds is not None
        ), "value must be supplied for right_inds or left_inds"
        left_inds = set(tensor.inds).difference(right_inds)
    elif right_inds is None:
        right_inds = set(tensor.inds).difference(left_inds)
    else:
        assert (
            len(set(tensor.inds).difference(left_inds).difference(right_inds)) == 0
        ), "When both left_inds and right_inds are supplied, all of the tensor's index must be present"
    assert(set(bond_inds).issubset(right_inds))
    Data = tensor.transpose(*left_inds,*right_inds).data
    left_sizes = [tensor.ind_size(ind) for ind in left_inds]
    right_sizes = [tensor.ind_size(ind) for ind in right_inds]
    left_size = np.prod(left_sizes)
    right_size = np.prod(right_sizes)
    Data = Data.reshape(left_size,right_size)
    Q,R = np.linalg.qr(Data.conj().T,'complete')
    Q = Q.conj().T
    R = R.conj().T
    for r in np.diag(R):
        assert abs(r-R[0,0])<1e-14,"Failure to complete to an unitary"
    f = abs(R[0,0])
    R/=f #I should be able to throw that out in the end...
    R2 = np.eye(max(R.shape))
    R2[:R.shape[0],:R.shape[1]] = R #contains only phase factors
    Q = R2@Q
    #reshape Q into a qubit operator
    #for testing purpose, just reshape it to somthing close to original form.
    new_inds = [ind+'in' for ind in bond_inds]
    new_sizes = [tensor.ind_size(ind) for ind in bond_inds]
    Q = qtn.Tensor(Q.reshape([*left_sizes,*new_sizes,*right_sizes]),[*left_inds,*new_inds,*right_inds])
    return Q,f


# %%
def unitary_MPO_to_circuit(mpo:qtn.MatrixProductOperator):
    nqbit = mpo.L
    target_2q_op = np.array(mpo.bond_sizes())/2
    # I assume that a suitably optimized two qubit operator
    # is able to generate an additive 2 worth of bond dimension.
    # This roughly match the exponential depth expected for 
    # an arbitrary unitary, because bond dimension for an arbirary MPO
    # unitary is up to 4^{n/2} for the middle bond. 
    # A simple staircase circuit produces a MPO with uniform bond dimension 4.
    # A X shaped base structure seems called for.
    # hopefully bond dimension over/2 is always more than needed.à
    