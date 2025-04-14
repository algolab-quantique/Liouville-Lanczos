# %%
import sys
import pathlib
import subprocess
from math import pi
from numbers import Number
from datetime import datetime

import numpy as np
import pandas as pd
import scipy as scipy
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.synthesis import LieTrotter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp


def get_heisenberg_hamiltonian_12_qbits(J=1, n=12, ent_map=[(1,0),(2,1),(3,2),(3,4),(4,6),(5,0),(5,7),(8,7),(8,9),(10,9),(10,11),(11,6)]):
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
    H = H.chop()
    return H


def prep_psi_0_with_checkpoints(qc: QuantumCircuit):
    qc.cx(12, 9)
    #
    qc.cx(9,10)
    #
    qc.cx(10,11)
    qc.cx(9,8)
    #
    qc.cx(11,6)
    qc.cx(8,7)
    #
    qc.cx(7,5)
    qc.cx(6,4)
    #
    qc.cx(4,3)
    qc.cx(5,0)
    #
    qc.cx(3,2)
    #make checkpoints
    qc.cx(11, 17)
    qc.cx(4, 16)
    qc.cx(2, 15)
    qc.cx(0, 14)
    qc.cx(7, 13)
    #undo entanglement on undesired qubits
    qc.cx(2,3)
    qc.cx(4,6)
    qc.cx(11,10)
    qc.cx(9,8)
    qc.cx(7,5)
    return qc


def prep_psi_0_by_0_with_checkpoints(qc: QuantumCircuit):
    qc.cx(12, 9, ctrl_state='0')
    qc.cx(17, 11, ctrl_state='0')
    qc.cx(16,4, ctrl_state='0')
    qc.cx(15,2, ctrl_state='0')
    qc.cx(14,0, ctrl_state='0')
    qc.cx(13,7, ctrl_state='0')
    return qc


def original_naive_circuit(H, i, j):
    time_evol = PauliEvolutionGate(H, pi/40, synthesis=LieTrotter(reps = 1))
    qc = QuantumCircuit(18)
    m = i
    n = j - i    
    qc.h(12)
    qc = prep_psi_0_with_checkpoints(qc)
    for t in range(n):
        qc.append(time_evol, range(12))
    qc = prep_psi_0_by_0_with_checkpoints(qc)
    for t in range(m):
        qc.append(time_evol, range(12))
    return qc.copy()


def general_krylov_circuit(H, n_reps, m_reps, n_time, m_time):
    qc = QuantumCircuit(18)
    qc.h(12)
    qc = prep_psi_0_with_checkpoints(qc)
    qc.append(PauliEvolutionGate(H, n_time, synthesis=LieTrotter(reps=max(n_reps,1))), range(12))
    qc = prep_psi_0_by_0_with_checkpoints(qc)
    qc.append(PauliEvolutionGate(H, m_time, synthesis=LieTrotter(reps=max(m_reps,1))), range(12))
    return qc


def naive_circuit(H, i, j):
    return general_krylov_circuit(H ,j-i, i, pi*(j-i)/40, pi*i/40)


def fixed_n_circuit(H, i, j, N, dt):
    return general_krylov_circuit(H, N//2, N//2, (j-i)*dt, i*dt)


def switching_n_circuit(H, i, j, N, dt):
    n, m = balancing_favors_low(i, j, N)
    return general_krylov_circuit(H, n, m, (j-i)*dt, i*dt)


def switching_n_high_circuit(H, i, j, N, dt):
    n, m = balancing_favors_high(i, j, N)
    return general_krylov_circuit(H, n, m, (j-i)*dt, i*dt)


def balancing_favors_low(i, j, N, verbose=False):
    nn = j-i
    mm = i
    if nn==0 and mm==0:
        n=m=N//2
    else:
        n = nn*N//(nn+mm)
        m = mm*N//(nn+mm)
    while n+m < N:
        if nn<mm : n += 1
        elif nn>mm : m += 1
    if verbose: print(f"{i},{j} ->   {nn},{mm} -> {n},{m}  total={n+m}")
    return n,m


def balancing_favors_high(i, j, N, verbose=False):
    nn = j-i
    mm = i
    if nn==0 and mm==0:
        n=m=N//2
    else:
        n = nn*N//(nn+mm)
        m = mm*N//(nn+mm)
    while n+m < N: 
        if nn > mm : n += 1    ## ONLY difference from favor low is the inequality here
        elif nn < mm : m += 1
    if verbose: print(f"{i},{j} ->   {nn},{mm} -> {n},{m}  total={n+m}")
    return n,m


def print_balancing_strategy(strategy, N, dim):
    print(f"balancing {N} gates for {dim} krylov states")
    print(f"i,j -> i-j,i -> n,m")
    for j in range(dim):
        for i in range(j+1):
            strategy(i, j, N, verbose=True)
    print()


def krylov_hamiltonian_and_overlap(H, krylov_circuit, dim=10):
    estimator = StatevectorEstimator()
    real_obs_H = SparsePauliOp('XXXXXX', 1) ^ H
    imag_obs_H = SparsePauliOp('YXXXXX', 1) ^ H
    real_obs_S = SparsePauliOp('XXXXXX', 1) ^ SparsePauliOp('I'*12, 1)
    imag_obs_S = SparsePauliOp('YXXXXX', 1) ^ SparsePauliOp('I'*12, 1)

    # result containers
    h = np.ndarray([10,10], dtype=complex)
    s = np.ndarray([10,10], dtype=complex)
    
    # measurement loop
    for j in range(dim):
        for i in range(j+1):
            print(f"{j}/{dim}, {i}/{j+1}", end="\r")
            qc = krylov_circuit(H, i, j)
            res = estimator.run([(qc, [real_obs_H, imag_obs_H, real_obs_S, imag_obs_S])])
            h[i,j] = res.result()[0].data.evs[0] + 1j * res.result()[0].data.evs[1]
            h[j,i] = res.result()[0].data.evs[0] - 1j * res.result()[0].data.evs[1]
            s[i,j] = res.result()[0].data.evs[2] + 1j * res.result()[0].data.evs[3]
            s[j,i] = res.result()[0].data.evs[2] - 1j * res.result()[0].data.evs[3]

    return h, s

def solve_generalized_eigenvalue(H, S, threshold=1e-9):
    #truncate useful subspace
    s_eigvals, s_eigvecs = scipy.linalg.eig(S)
    P = s_eigvecs[s_eigvals>threshold].T  
    
    #project un useful subspace and solve GEVP
    A = P.conj().T @ H @ P
    B = P.conj().T @ S @ P
    ab_eigvals, ab_eigvecs = scipy.linalg.eig(A, B)
    
    #return ground state
    i_min = ab_eigvals.argmin()
    gs_energy = ab_eigvals[i_min]
    gs_vector = ab_eigvecs[:,i_min]
    return gs_energy, gs_vector, P


def save_gs_vs_d_figure(gs_vs_d, title, true_gs=None):
    
    saved_gs_vs_d = [-12.0, -16.898, -19.042, -20.017, -20.498, -20.735, -20.862, -21.026, -21.21, -21.293]
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("krylov dimension")
    ax.set_ylabel("lowest energy")
    ax.set_ylim(-22,-12)    
    ax.plot(list(range(1,11)), saved_gs_vs_d, color='0.6', linestyle="dashed")
    ax.plot(list(range(1,11)), gs_vs_d)

    if true_gs is not None:
        ax.hlines(true_gs, 0, 10, color='k', linestyles="dotted", label="true GS")
    
    plt.savefig(f"{RESULTS_DIR}/{title}.jpg")
    plt.clf()


def evaluate_krylov_circuit(H, krylov_circuit=naive_circuit, dim=10, true_gs=-21.5496, treshold_factor=1e-8):
    if isinstance(true_gs, Number):
        true_gs = true_gs
    elif true_gs:
        true_eigvals, true_eigvecs = scipy.linalg.eig(H.to_matrix()) 
        true_imin = true_eigvals.argmin() 
        true_gs = true_eigvals[true_imin]

    H_tilde, S_tilde = krylov_hamiltonian_and_overlap(H, krylov_circuit, dim=dim)    
    gs_vs_d = [
        solve_generalized_eigenvalue(
            H_tilde[0:d,0:d], 
            S_tilde[0:d,0:d], 
            threshold=treshold_factor*d
        )[0].real
        for d in range(1, dim+1)
    ]
    gap = min(gs_vs_d) - true_gs if true_gs else None
    return gs_vs_d, gap


# Experiment
RESULTS_DIR = "trotter_balancing_results"

def main():
    # Setup
    date_str = datetime.today().strftime('%Y-%m-%d')
    time_str = datetime.today().strftime('%H:%M:%S')
    git_str = "git" + subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()    

    rundir = pathlib.Path(__file__).parent/RESULTS_DIR/"-".join([date_str, time_str, git_str]) 
    rundir.mkdir(parents=True, exist_ok=True)
    
    stdout = open(rundir/'stdout.txt', "a")
    sys.stdout = stdout  # redirect print to the log file

    def save_to_csv(strategy, gap, denom, treshold, num_gates, gs_vs_d, path=rundir/"results.csv"):
        df = pd.DataFrame({
            'strategy': [strategy],
            'gap': [gap],
            'dt_denom': [denom],
            'num_gates': [num_gates],
            'gs_vs_d': [gs_vs_d],
            'treshold': [treshold],
        })
        if path.is_file():
            df.to_csv(path, header=None, mode="a")  
        else: 
            df.to_csv(path)


    def check_existance(dict_of_values, df):
        v = df.iloc[:, 0] == df.iloc[:, 0]
        for key, value in dict_of_values.items():
            v &= (df[key] == value)
        return v.any()


    # Experiment
    
    H = get_heisenberg_hamiltonian_12_qbits()
        
    for treshold in [1e-8, 1e-2]:
        for denom in [30,40,60,80]:
            for num_gates in [2,4,6,8,10]:
                for strategy, strategy_circ in {
                    "switch" : switching_n_circuit,
                    "high" : switching_n_high_circuit,
                    "fixed" : fixed_n_circuit,
                }.items():
                    trial_circ = lambda H,i,j: strategy_circ(H, i, j, num_gates, dt=pi/denom)
                    gs_vs_d, gap = evaluate_krylov_circuit(H, trial_circ, dim=10, treshold_factor=treshold)
                    save_to_csv(strategy, gap, denom, treshold, num_gates, gs_vs_d)

    gs_vs_d, gap = evaluate_krylov_circuit(H, original_naive_circuit)
    save_to_csv("original", gap, 40, 1e-8, 10, gs_vs_d)
        
    gs_vs_d, gap = evaluate_krylov_circuit(H, naive_circuit)
    save_to_csv("naive", gap, 40, 1e-8, 10, gs_vs_d)


if __name__ == "__main__":
    # main()

    results = pathlib.Path(__file__).parent/RESULTS_DIR/"latest"/"results.csv"
    allres = pd.read_csv(results)

    allres.loc[allres['strategy']=='fixed'].groupby('treshold').plot(x='num_gates', y='gap')
    plt.show()