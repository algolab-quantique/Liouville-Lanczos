from math import pi
import shutil
import sys
import subprocess
from datetime import datetime
from pathlib import Path
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



# Parameters


RESULTS_DIR = Path(__file__).parent/"trotter_balancing_results"

SEP = "\t"

CONFIG_SPACE = {
        'num_gates': [4, 6, 8],
        'treshold': [1e-1, 1e-4, 1e-8, 0],
        'dt_denom': [20, 30, 40, 50],
        'strategy': ["low", "high", "fixed"],
        'synthesis': ["LieTrotter", "SuzukiTrotter"],
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

STRATEGY_MAP = {
    "low": switching_n_circuit,
    "high": switching_n_high_circuit,
    "fixed": fixed_n_circuit,
    "naive": naive_circuit,
}

SYNTHESIS_MAP = {
    "LieTrotter": LieTrotter,
    "SuzukiTrotter": SuzukiTrotter,
}


def generate_all_combinations(config_space: dict[list] = CONFIG_SPACE) -> list[dict]:
    '''Generates a list of dict covering all combinations of key values in 

    Args:
        config_space (dict, optional): Keys should map to lists of possible values. Defaults to CONFIG_SPACE.

    Returns:
        list of dict: List of all combinations where keys map to single values
    '''    
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


def config_name(config: dict) -> str:
    '''returns a name based on the content of `config`'''
    name = config['strategy']
    name += f"{config['num_gates']}"
    name += f"_pi{config['dt_denom']}"
    name += f"_tresh{config['treshold']}"
    name += f"_synt{config['synthesis']}"
    return name

# Utilities


def get_git_hash() -> str:
    '''Retrieves the current commit's short hash'''    
    return "git" + subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


def new_rundir(parent: Path = RESULTS_DIR) -> Path:
    '''Create directory named from datetime and git hash as "yyyy-mm-dd-hh-mm-ss-hash"

    Args:
        parent (Path, optional): Target location. Defaults to this file's location/trotter_balancing_results/
    
    Returns:
        Path: New directory
    '''    
    date_str = datetime.today().strftime('%Y-%m-%d')
    time_str = datetime.today().strftime('%H-%M-%S')
    git_str = "git" + subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()    

    rundir = parent/"-".join([date_str, time_str, git_str]) 
    rundir.mkdir(parents=True, exist_ok=True)
    
    return rundir


def save_to_csv(path: Path, **kwargs):
    '''Append all keywords argument as a row to a .csv file

    Args:
        path (pathlib.Path): location of the .csv file
        **kwargs (Any): key-value pairs to fill the row
    '''    
    df = pd.DataFrame({k:[v] for k,v in kwargs.items()})
    if path.is_file():
        df.to_csv(path, index=False, sep=SEP, header=None, mode="a")  
    else: 
        df.to_csv(path, index=False, sep=SEP)


def recover_previous_data(prev_csv: Path, new_csv: Path) -> pd.DataFrame:
    '''generate `new_csv` from `prev_csv` assigning missing fields in the process. Returns the new_csv DataFrame'''
    prev_df = pd.read_csv(prev_csv, sep=SEP)
    columns = list(CONFIG_SPACE.keys()) + RESULT_KEYS
    for key in columns:
        if not key in prev_df.columns:
            prev_df = prev_df.assign(**{key: lambda x: DEFAULT_VALUE[key]})

    prev_df.to_csv(
            new_csv, index=False, mode="x", sep=SEP,
            columns=columns, 
        )
    return prev_df


def check_existance(config: dict, df: pd.DataFrame) -> bool:
    '''Returns True if the `config` correspond to a row of `df`'''    
    v = df.iloc[:, 0] == df.iloc[:, 0]
    for key, value in config.items():
        v &= (df[key] == value)
    return v.any()


def is_included(row_dict: dict, config_list: list[dict]) -> bool:
    '''Returns True if part of `row_dict` matches one full dict in `config_list`'''
    enable = False
    for config in config_list:
        is_same = True
        for k, v in config.items():
            is_same = is_same and row_dict[k]==v
        enable = enable or is_same
    return enable


# Simulations


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
        prev_df = recover_previous_data(continue_from/"results.csv", csv_path)

    config_list = generate_all_combinations(config_space)
    print(f"Running {len(config_list)} krylov subspace evaluation")

    for config in config_list:
        is_already_done = False
        if continue_from:
            is_already_done = check_existance(config, prev_df)
        if continue_from and is_already_done:
            print(f"Skipping already available {config}")
        else:
            num_gates = config['num_gates']
            denom = config['dt_denom']
            treshold = config['treshold']
            strategy_circ = STRATEGY_MAP[config['strategy']]
            synth = SYNTHESIS_MAP[config['synthesis']]

            trial_circ = lambda H,i,j: strategy_circ(H, i, j, num_gates, dt=pi/denom, synthesis=synth)
            gs_vs_d, gap = evaluate_krylov_circuit(H, trial_circ, dim=10, treshold_factor=treshold)

            save_to_csv(csv_path, **config, gap=gap, gs_vs_d=gs_vs_d, git=get_git_hash())


## Figures


def make_figures(path, config_space=CONFIG_SPACE, dim=10, true_gs=-21.5496):
    config_list = generate_all_combinations(config_space)
    
    csv_path = path/"results.csv"
    df = pd.read_csv(csv_path, sep=SEP)
    df_as_list_of_dict = df.to_dict(orient='records', index=config_space.keys())

    for config in df_as_list_of_dict:
        if is_included(config, config_list):
            name = config_name(config)
            gs_vs_d = np.asarray(eval(config['gs_vs_d']))
            gap = min(gs_vs_d[:dim]) - true_gs
            name = f"{gap.real:2.3f}_{name}"
            filename = path/f"{name}.jpg"
            
            if not filename.exists():
                save_gs_vs_d_figure(gs_vs_d[:dim], filename, true_gs)
            else:
                print(f"skipping {filename}")


def save_gs_vs_d_figure(gs_vs_d, path, true_gs=None):
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("krylov dimension")
    ax.set_ylabel("lowest energy")
    ax.set_ylim(-22,-12)    
    ax.plot(list(range(1,len(gs_vs_d)+1)), gs_vs_d)

    if true_gs is not None:
        ax.hlines(true_gs, 0, 10, color='k', linestyles="dotted", label="true GS")
    
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":

    restricted_space = {
        'num_gates': [10],
        'treshold': [0],
        'dt_denom': [40],
        'strategy': {
            # "low",
            # "high",
            # "fixed",
            "naive",
        },
        'synthesis': [
            "LieTrotter", 
            # "SuzukiTrotter"
        ],
    }


    rundir = new_rundir()
    latest = RESULTS_DIR/"latest"
    stdout_path = open(rundir/'stdout.txt', "a")
    sys.stdout = stdout_path  # redirect print to the log file

    print(restricted_space, flush=True)

    run_simulations(
        csv_path=rundir/"results.csv",
        # continue_from=latest,
        config_space=restricted_space,
    )

    make_figures(
        path=rundir,
        config_space=restricted_space,
        dim=10,
    )

    # shutil.copy(latest/"results.csv", latest/"results_bkp.csv")
    # shutil.copy(rundir/"results.csv", latest/"results.csv")
