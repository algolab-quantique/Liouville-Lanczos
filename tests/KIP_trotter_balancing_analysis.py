#%%
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CONFIG_SPACE = {
        'num_gates': [4, 6, 8],
        'treshold': [1e-1, 1e-4, 1e-8, 0],
        'dt_denom': [20, 30, 40, 50],
        'strategy': ["low", "high", "fixed"],
        'synthesis': ["LieTrotter", "SuzukiTrotter"],
}

RESULTS_DIR = Path(__file__).parent/"trotter_balancing_results"
path = RESULTS_DIR/"latest"/"results.csv"
df = pd.read_csv(path, sep="\t")

new_df = df.copy()

# for i in range(10):
#     new_df[f'e{i}'] = df.apply(lambda x:  eval(x.gs_vs_d)[i], axis=1)

def make_array(x):
    gs_vs_d = np.array(eval( x.gs_vs_d ))
    return gs_vs_d

def remove_outlier(x):
    gs_vs_d = np.array(eval( x.gs_vs_d ))
    return gs_vs_d[gs_vs_d > -21.5496]

def remove_dim_gt_num(x):
    gs_vs_d = np.array(eval(x.gs_vs_d))[:x.num_gates+1]
    return gs_vs_d

def remove_both(x):
    gs_vs_d = np.array(eval(x.gs_vs_d))[:x.num_gates]
    return gs_vs_d[gs_vs_d > -21.5496]

for strategy in CONFIG_SPACE['strategy']:
    for num_gates in CONFIG_SPACE['num_gates']:
        for treshold in CONFIG_SPACE['treshold']:

            new_df = df.copy()
            new_df = new_df[new_df['strategy']==strategy]
            new_df = new_df[new_df['num_gates']==num_gates]
            new_df = new_df[new_df['treshold']==treshold]
            new_df['gs_vs_d'] = df.apply(make_array, axis=1)


            suzuki_df = new_df[new_df['synthesis']=='SuzukiTrotter']
            lie_df = new_df[new_df['synthesis']=='LieTrotter']

            plt.figure(figsize=(12, 9), dpi=80)
            plt.rc('font', size=22)
            plt.title(f"{strategy} balancing, {num_gates} gates, $\epsilon={treshold}$")

            for gs in suzuki_df[suzuki_df['dt_denom']==40]['gs_vs_d']:
                plt.plot(gs, color=f'C3')
            # for gs in suzuki_df[suzuki_df['dt_denom']==30]['gs_vs_d']:
                # plt.plot(gs, color=f'C3')
            for gs in suzuki_df[suzuki_df['dt_denom']==20]['gs_vs_d']:
                plt.plot(gs, color=f'C1')
            for gs in lie_df[lie_df['dt_denom']==40]['gs_vs_d']:
                plt.plot(gs, color=f'C2')
            # for gs in lie_df[lie_df['dt_denom']==30]['gs_vs_d']:
                # plt.plot(gs, color=f'C2')
            for gs in lie_df[lie_df['dt_denom']==20]['gs_vs_d']:
                plt.plot(gs, color=f'C0')
            line0 = plt.hlines(y=-21.5496, linestyles='dashed', color='k', xmin=0, xmax=9, label="Exact")

            plt.ylim(-22,-17)
            line4 = plt.Line2D([0], [0], label='Suzuki pi/40', color='C3')
            line3 = plt.Line2D([0], [0], label='Suzuki pi/20', color='C1')
            line2 = plt.Line2D([0], [0], label='Lie pi/40', color='C2')
            line1 = plt.Line2D([0], [0], label='Lie pi/20', color='C0')
            plt.xlabel("Krylov dimension")
            plt.ylabel("$E_0$")
            plt.legend(handles=[line0, line1, line2, line3, line4])
            plt.savefig(RESULTS_DIR/f"{num_gates}{strategy}{treshold}.jpg")
            plt.close()

