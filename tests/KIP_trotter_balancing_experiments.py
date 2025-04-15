from math import pi
import shutil
import sys
import subprocess
from datetime import datetime
import pathlib
from copy import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from KIP_trotter_balancing import (
    get_heisenberg_hamiltonian_12_qbits,
    evaluate_krylov_circuit,
    
    original_naive_circuit,
    naive_circuit,
    switching_n_circuit,
    switching_n_high_circuit,
    fixed_n_circuit,
    
    LieTrotter,
    SuzukiTrotter,
)

RESULTS_DIR = pathlib.Path(__file__).parent/"trotter_balancing_results"

CONFIG_SPACE = {
        'num_gates': [4, 6, 8],
        'treshold': [1e-1, 1e-4, 1e-8, 0],
        'dt_denom': [20, 30, 40, 50],
        'strategy': {
            "low": switching_n_circuit,
            "high": switching_n_high_circuit,
            "fixed": fixed_n_circuit,
        },
        'synthesis': {
            "LieTrotter": LieTrotter,
            "SuzukiTrotter": SuzukiTrotter,
        },
    }

RESULT_KEYS = [ 
    'gap',
    'gs_vs_d',
    'git',
]

DEFAULT_VALUE = {
    'num_gates': 10,
    'treshold': 1e-8,
    'dt_denom': 40,
    'strategy': "fixed",
    'synthesis': "LieTrotter",
    'gap': None,
    'gs_vs_d' : None,
    'git': None,
}


def generate_all_combinations(config_space=CONFIG_SPACE):
    list_of_config_dicts = [{}]
    for key in config_space:
        new_list_of_config_dicts = []
        for config in list_of_config_dicts:
            for value in config_space[key]:
                new_config = copy(config)
                new_config[key] = value
                new_list_of_config_dicts.append(new_config)
        list_of_config_dicts = new_list_of_config_dicts
    return list_of_config_dicts


def get_git_hash():
    return "git" + subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


def new_rundir():
    date_str = datetime.today().strftime('%Y-%m-%d')
    time_str = datetime.today().strftime('%H-%M-%S')
    git_str = "git" + subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()    

    rundir = RESULTS_DIR/"-".join([date_str, time_str, git_str]) 
    rundir.mkdir(parents=True, exist_ok=True)
    
    stdout_path = open(rundir/'stdout.txt', "a")
    sys.stdout = stdout_path  # redirect print to the log file

    return rundir


def save_to_csv(path, **kwargs):
    df = pd.DataFrame({k:[v] for k,v in kwargs.items()})
    if path.is_file():
        df.to_csv(path, index=False, sep="\t", header=None, mode="a")  
    else: 
        df.to_csv(path, index=False, sep="\t")


# Experiment
def run_original_experiment(csv_path):
    H = get_heisenberg_hamiltonian_12_qbits()

    gs_vs_d, gap = evaluate_krylov_circuit(H, original_naive_circuit)
    save_to_csv(csv_path, strategy="original", dt_denom=40, num_gates=10, treshold=1e-8, gap=gap, gs_vs_d=gs_vs_d)

    gs_vs_d, gap = evaluate_krylov_circuit(H, naive_circuit)
    save_to_csv(csv_path, strategy="original", dt_denom=40, num_gates=10, treshold=1e-8, gap=gap, gs_vs_d=gs_vs_d)


def run_simulations(
    csv_path,
    continue_from=None,
    config_space=CONFIG_SPACE,
):
    H = get_heisenberg_hamiltonian_12_qbits()

    if continue_from:
        prev_df = update_csv(continue_from/"results.csv", csv_path)

    config_list = generate_all_combinations(config_space)
    print(f"Running {len(config_list)} krylov subspace evaluation")

    for config_dict in config_list:
        is_already_done = False
        if continue_from:
            is_already_done = check_existance(config_dict, prev_df)
        if continue_from and is_already_done:
            print(f"Skipping already available {config_dict}")
        else:
            num_gates = config_dict['num_gates']
            denom = config_dict['dt_denom']
            treshold = config_dict['treshold']
            strategy_circ = config_space['strategy'][config_dict['strategy']]
            synth = config_space['synthesis'][config_dict['synthesis']]

            trial_circ = lambda H,i,j: strategy_circ(H, i, j, num_gates, dt=pi/denom, synthesis=synth)
            gs_vs_d, gap = evaluate_krylov_circuit(H, trial_circ, dim=10, treshold_factor=treshold)

            save_to_csv(csv_path, **config_dict, gap=gap, gs_vs_d=gs_vs_d, git=get_git_hash())


def update_csv(prev_csv, new_csv):
    prev_df = pd.read_csv(prev_csv, sep='\t')
    columns = list(CONFIG_SPACE.keys()) + RESULT_KEYS
    for key in columns:
        if not key in prev_df.columns:
            prev_df = prev_df.assign(**{key: lambda x: DEFAULT_VALUE[key]})

    prev_df.to_csv(
            new_csv, index=False, mode="x", sep="\t",
            columns=columns, 
        )
    return prev_df


def check_existance(id_dict, df):
    v = df.iloc[:, 0] == df.iloc[:, 0]
    for key, value in id_dict.items():
        v &= (df[key] == value)
    return v.any()


def save_gs_vs_d_figure(gs_vs_d, path, gap=None):
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("krylov dimension")
    ax.set_ylabel("lowest energy")
    ax.set_ylim(-22,-12)    
    ax.plot(list(range(1,11)), gs_vs_d)

    if gap is not None:
        true_gs = min(gs_vs_d) - gap
        ax.hlines(true_gs, 0, 10, color='k', linestyles="dotted", label="true GS")
    
    plt.savefig(path)
    plt.close()


def make_figures(path, config_space=CONFIG_SPACE):
    csv_path = path/"results.csv"
    df = pd.read_csv(csv_path, sep="\t")
    df_as_list_of_dict = df.to_dict(orient='records', index=config_space.keys())

    config_list = generate_all_combinations(config_space)

    for row_dict in df_as_list_of_dict:
        will_plot = is_included(row_dict, config_list)
        
        if will_plot:
            name = row_dict['strategy']
            name += f"{row_dict['num_gates']}"
            name += f"_pi{row_dict['dt_denom']}"
            name += f"_tresh{row_dict['treshold']}"
            
            gs_vs_d = np.asarray(eval(row_dict['gs_vs_d']))
            gap = row_dict['gap']
            
            if gap:
                filename = path/f"{gap.real:2.3f}_{name}.jpg"
            else:
                filename = path/f"{name}.jpg"
            
            if not filename.exists() and will_plot:
                save_gs_vs_d_figure(gs_vs_d, filename, gap)
            else:
                print(f"skipping {filename}")


def is_included(row_dict, config_list):
    will_plot = False
    for config in config_list:
        is_same = True
        for k, v in config.items():
            is_same = is_same and row_dict[k]==v
        will_plot = will_plot or is_same
    return will_plot


if __name__ == "__main__":

    restricted_space = {
        'num_gates': [4, 6, 8],
        'treshold': [1e-1, 1e-4, 1e-8, 0],
        'dt_denom': [20, 30, 40, 50],
        'strategy': {
            "low": switching_n_circuit,
            "high": switching_n_high_circuit,
            "fixed": fixed_n_circuit,
        },
        'synthesis': {
            "LieTrotter": LieTrotter,
            "SuzukiTrotter": SuzukiTrotter,
        },
    }

    rundir = new_rundir()
    latest = RESULTS_DIR/"latest"

    run_simulations(
        csv_path=rundir/"results.csv",
        continue_from=latest,
        config_space=restricted_space,
    )

    make_figures(
        path=rundir,
        config_space=restricted_space,
    )

    shutil.copy(latest/"results.csv", latest/"results_bkp.csv")
    shutil.copy(rundir/"results.csv", latest/"results.csv")
