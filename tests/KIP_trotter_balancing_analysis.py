#%%
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CONFIG_SPACE = {
        'num_gates': [10],# [4, 6, 8, 10],
        'treshold': [0],# [1e-1, 1e-4, 1e-5, 1e-5, 1e-8, 0],
        'dt_denom': [10, 15, 20, 25, 30, 40, 50, 80],
        'strategy': ["fixed"], # ["low", "high", "fixed", "naive"],
        'synthesis': ["SuzukiTrotter"],# ["LieTrotter", "SuzukiTrotter"],
}
RESULTS_DIR = Path(__file__).parent/"trotter_balancing_results"
path = RESULTS_DIR/"latest"/"results.csv"
df = pd.read_csv(path, sep="\t")

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
            for synthesis in CONFIG_SPACE['synthesis']:
                
                plt.figure(figsize=(11, 7), dpi=80)
                plt.rc('font', size=20)
                plt.title(f"{strategy} balancing, {num_gates} gates, $\epsilon={treshold}$")

                for i, dt_denom in enumerate(CONFIG_SPACE['dt_denom']):

                    new_df = df.copy()
                    new_df['gs_vs_d'] = df.apply(make_array, axis=1)
                    new_df = new_df[new_df['strategy']==strategy]
                    new_df = new_df[new_df['num_gates']==num_gates]
                    new_df = new_df[new_df['treshold']==treshold]
                    new_df = new_df[new_df['synthesis']==synthesis]
                    new_df = new_df[new_df['dt_denom']==dt_denom]


                    for gs_v_d in new_df['gs_vs_d']:
                        if dt_denom == 30:
                            plt.plot(range(1, len(gs_v_d)+1), gs_v_d, label=str(dt_denom), color='red', lw=3, marker=".")
                        else:
                            plt.plot(range(1, len(gs_v_d)+1), gs_v_d, label=str(dt_denom), color=str(1-i/12), lw=2, marker=".")

                line0 = plt.hlines(y=-21.5496, linestyles='dashed', color='k', xmin=0, xmax=10, label="Exact")

                plt.ylim(-22,-17)
                plt.xlabel("Krylov dimension")
                plt.ylabel("$E_0$")
                plt.legend()
                plt.savefig(RESULTS_DIR/f"{num_gates}{strategy}{treshold}s{synthesis}.jpg")
                plt.show()
                plt.close()

new_df
# %%
