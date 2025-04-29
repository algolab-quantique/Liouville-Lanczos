#%%
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent/"trotter_balancing_results"

def make_plot_general(df, config, dim=10, gs=-21.5496, save=False):
    
    fixed = {}
    show = {}
    for k, v in config.items():
        if isinstance(v, list):
            show[k] = v
        else:
            fixed[k] = v

    new_df = df.copy()
    new_df['gs_vs_d'] = df.apply(lambda x : np.array(eval(x.gs_vs_d)), axis=1)
    new_df['min'] = new_df.apply(lambda x : x.gs_vs_d[x.gs_vs_d>-21.5496].min(), axis=1)
    new_df['gs_vs_d'] = new_df.apply(lambda x : x.gs_vs_d[:dim], axis=1)

    for key, val in fixed.items():
        new_df = new_df[new_df[key]==val]


    fixed_values = [str(k)+str(v) for k,v in fixed.items()]
    if dim != 10: fixed_values.append(f"dim{dim}")
    
    title = f"_".join(fixed_values)

    plt.figure(figsize=(11, 7), dpi=80)
    plt.rc('font', size=20) 
    plt.title(f" ".join(fixed_values)) 

    for key, val in show.items():
        new_df = new_df[new_df[key].isin(val)]

    legend_handles = []
    key = next(iter(show))
    for i, val in enumerate(show[key]):
        this_val_df = new_df[new_df[key]==val]

        for gs_v_d in this_val_df['gs_vs_d']:
            plt.plot(
                range(1, len(gs_v_d)+1),
                gs_v_d, label=str(),
                color=f'C{i}', alpha=0.4, lw=1, marker='.'
            )

        highlight_gs_v_d = this_val_df.nsmallest(1,'min')['gs_vs_d'].item()

        line = plt.plot(
            range(1, len(highlight_gs_v_d)+1),
            highlight_gs_v_d,
            label=str(val), color=f'C{i}', lw=2, marker="o", markerfacecolor="white", markeredgewidth=2
        )[0]
        legend_handles.append(line)

    line = plt.hlines([gs], xmin=0, xmax=10, linestyles="dashed", color='k', label="exact")
    legend_handles.append(line)
    
    plt.ylim(-22,-17)
    plt.xlim(0,10)
    plt.xlabel("Krylov dimension")
    plt.ylabel("$E_0$")
    plt.legend(handles=legend_handles)
    plt.tight_layout()

    if save: 
        title += ".jpg"
        plt.savefig(RESULTS_DIR / title)
    else:
        plt.show()
    plt.close()

#%%
if __name__ == "__main__":

    path = RESULTS_DIR/"latest"/"results.csv"
    df = pd.read_csv(path, sep="\t")

    make_plot_general(df,{
        'num_gates': 4,# [4, 6, 8, 10],
        'strategy': "fixed",
        'synthesis': "LieTrotter",
        'dt_denom': [20, 40],
        'treshold': [1e-4, 1e-5, 1e-6, 1e-8, 0],
    }, save=True)
    
    make_plot_general(df,{
        'num_gates': 4,# [4, 6, 8, 10],
        'strategy': "fixed",
        'synthesis': "LieTrotter",
        'dt_denom': 40,
        'treshold': [1e-4, 1e-5, 1e-6, 1e-8, 0],
    }, save=True)

    make_plot_general(df,{
        'strategy': "fixed",
        'synthesis': ["LieTrotter", "SuzukiTrotter"],
        'num_gates':[4],
        'dt_denom': 40,
        'treshold': [1e-4, 1e-5, 1e-6, 1e-8, 0],
    }, save=True)

    make_plot_general(df,{
        'strategy': "fixed",
        'synthesis': ["LieTrotter", "SuzukiTrotter"],
        'num_gates':[6],
        'dt_denom': 40,
        'treshold': [1e-4, 1e-5, 1e-6, 1e-8, 0],
    }, save=True)

    make_plot_general(df,{
        'strategy': "high",
        'synthesis': ["LieTrotter", "SuzukiTrotter"],
        'num_gates':[4, 6],
        'dt_denom': [20,30,40,50],
        'treshold': 1e-8,
    }, save=True)
    
    make_plot_general(df,{
        'strategy': "high",
        'synthesis': ["LieTrotter", "SuzukiTrotter"],
        'num_gates':[4, 6],
        'dt_denom': [20,30,40,50],
        'treshold': 1e-8,
    }, save=True)

    make_plot_general(df,{
        'strategy': "high",
        'synthesis': "LieTrotter",
        'num_gates':[4, 6],
        'dt_denom': [20,30,40,50],
        'treshold': 1e-8,
    }, save=True)

    make_plot_general(df,{
        'strategy': "high",
        'synthesis': "LieTrotter",
        'num_gates':[4, 6],
        'dt_denom': [20,30,40,50],
        'treshold': 1e-4,
    }, save=True)

    make_plot_general(df,{
        'strategy': "high",
        'synthesis': "SuzukiTrotter",
        'num_gates':[4, 6],
        'dt_denom': [20,30,40,50],
        'treshold': 1e-8,
    }, save=True)

    make_plot_general(df,{
        'strategy': "high",
        'synthesis': "SuzukiTrotter",
        'num_gates':[4, 6],
        'dt_denom': [20,30,40,50],
        'treshold': 1e-4,
    }, save=True)

    make_plot_general(df,{
        'strategy': "high",
        'synthesis': "LieTrotter",
        'num_gates':4,
        'dt_denom': [20,30,40,50],
        'treshold': 1e-8,
    }, save=True)

    make_plot_general(df,{
        'strategy': ["naive", "fixed"],
        'synthesis': ["LieTrotter", "SuzukiTrotter"],
        'num_gates': 10,
        'dt_denom': [20,30,40,50,80],
        'treshold': 0,
    }, save=True)

    make_plot_general(df,{
        'strategy': "naive",
        'synthesis': "LieTrotter",
        'num_gates': 10,
        'dt_denom': [20,30,40,50,80],
        'treshold': 0,
    }, save=True)

    make_plot_general(df,{
        'strategy': "naive",
        'synthesis': "SuzukiTrotter",
        'num_gates': 10,
        'dt_denom': [20,30,40,50,80],
        'treshold': 0,
    }, save=True)
#%%


make_plot_general(df,{
        'strategy': "high",
        'synthesis': ["LieTrotter", "SuzukiTrotter"],
        'num_gates':[4, 6],
        'dt_denom': [20,30,40,50],
        'treshold': 1e-8,
    }, save=True, dim=5)
# %%
