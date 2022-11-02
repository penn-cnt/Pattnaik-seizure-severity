#%%
'''
try different baseline methods function
'''
# pylint: disable-msg=C0103
%load_ext autoreload
%autoreload 2
import os
from os.path import join as ospj
import json
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import iqr, pearsonr
from sklearn import metrics
from tqdm import tqdm
import tools
from bct import *

import matplotlib.pyplot as plt
data_path = "../data"
n_ii = 20

band_names = ["delta", "theta", "alpha", "beta", "gamma", "broad"]
n_bands = len(band_names)

def zscore(ii_feature, pi_feature):
    pi_z = pi_feature - np.nanmean(ii_feature, axis=0)
    pi_z = pi_z / np.nanstd(ii_feature, axis=0)

    return pi_z


def plot_pi_z(pt_pi_z, sz_severity, feature_name):
    sort_idx = np.argsort(pt_pi_z.mean(axis=0))
    sort_idx = np.argsort(sz_severity)

    # make plot
    fig, ax = plt.subplots()
    
    sns.boxplot(data=pt_pi_z.iloc[:, sort_idx], ax=ax, boxprops={'facecolor':'None'}, showfliers=False)
    sns.stripplot(data=pt_pi_z.iloc[:, sort_idx], ax=ax)

    sns.despine(ax=ax)
    ax.set_ylabel(f"{feature_name} (z)")
    ax.set_xlabel("")
    ax.set_xticklabels(
        [f"ss = {i:.2f}" for i in sz_severity.values[sort_idx]],
        rotation=45,
        ha='right'
        )

    return fig, ax

# %%
# set variables
sz_metadata = pd.read_excel(ospj(data_path, "metadata", "seizure_metadata_with_severity.xlsx"), index_col=0)
pt_metadata = pd.read_excel(ospj(data_path, "metadata", "patient_table.xlsx"), index_col=0)
sz_metadata = sz_metadata[sz_metadata['Patient'] != "HUP125"]

power_arr = np.empty(len(sz_metadata), dtype=object)
coherence_arr = np.empty(len(sz_metadata), dtype=object)

# get variance of severity, want to have patients with most variance
pt_sz_var = sz_metadata.groupby('Patient').apply(lambda grp: np.var(grp['Seizure severity (NMF)']))

high_var_pt = pt_sz_var[pt_sz_var > 0.1].index
sz_metadata = sz_metadata[sz_metadata['Patient'].isin(high_var_pt)]

# %% Broadband network
# calculate z for all channels
band_idx = band_names.index("broad")
all_pi_z = {}
all_pi_z['broadband coherence'] = {}

patient_inter_corr = []

ind = 0
for pt, group in sz_metadata.groupby('Patient'):
    # look at patients with more than one seziure only
    if len(group) <= 2:
        continue

    # get interictal features
    ii_network = []
    for i_ii in range(n_ii):
        ii_network.append(np.load(ospj(data_path, pt, f"interictal_networks_{i_ii}.npy")))

    ii_network = np.array(ii_network)
    # ii_network = ii_network[:, band_idx, :, :]
    ii_node_str = np.array([[strengths_und(i) for i in j] for j in ii_network])

    pt_pi_z = []
    sz_num = group['Seizure number']
    for i_sz in sz_num:
        pi_network = np.load(ospj(data_path, pt, f"preictal_networks_sz-{i_sz}.npy"))
        pi_node_str = np.array([strengths_und(i) for i in pi_network])
        n_channels = pi_node_str.shape[-1]

        pi_z = np.zeros_like(pi_node_str)
        for i_band, band in enumerate(band_names):
            pi_z[i_band, :] = zscore(ii_node_str[:, i_band], pi_node_str[i_band, :])
        coherence_arr[ind] = pi_z
        ind += 1
        pt_pi_z.append(pi_z)
    
    pt_pi_z = np.array(pt_pi_z)
    pt_pi_z = pd.DataFrame(pt_pi_z[:, band_idx, :].T, columns=group.index)
    pt_pi_z = np.abs(pt_pi_z)

    # make plot
    # fig, ax = plot_pi_z(pt_pi_z, all_sz_sev[pt], "broadband coherence")
    # ax.set_title(pt)
    # plt.tight_layout()
    # plt.savefig(f"../figures/{pt}/preictal_z_broadband_coherence.pdf")
    # plt.close()

    all_pi_z['broadband coherence'][pt] = pt_pi_z

    # patient_inter_corr.append(pearsonr(pt_pi_z.mean(axis=0), all_sz_sev[pt])[0])

# %%
# save bandpower and coherence for matlab
from scipy.io import savemat
savemat(
    "../data/pre_ictal_features.mat",
    {
        "bandpower": power_arr,
        "coherence": coherence_arr,
        "severity": sz_metadata['Seizure severity (NMF)'].to_numpy()
    }
)
sz_metadata.to_csv("../data/pre_ictal_metadata.csv")

# %%
all_univar = []
all_bivar = []
all_sev = []
all_pt = []
for pt in all_sz_sev.keys():
    all_sev.extend(all_sz_sev[pt].values)
    all_pt.extend([pt] * len(all_sz_sev[pt]))

    all_univar.extend(all_pi_z['broadband power'][pt].mean(axis=0).values)
    all_bivar.extend(all_pi_z['broadband coherence'][pt].mean(axis=0).values)

df = pd.DataFrame(
    [all_pt, all_sev, all_univar, all_bivar]
).T
df.columns = ['pt', 'sz_sev', 'univar_z', 'bivar_z']

df['pt_num'], pt_num2str = df['pt'].factorize()
df['univar_z'] = pd.to_numeric(df['univar_z'])
df['bivar_z'] = pd.to_numeric(df['bivar_z'])
df['sz_sev'] = pd.to_numeric(df['sz_sev'])

# get variance of severity, want to have patients with most variance
pt_sz_var = sz_metadata.groupby('Patient').apply(lambda grp: np.var(grp['Seizure severity (NMF)']))

high_var_pt = pt_sz_var[pt_sz_var > 0.1].index
df = df[df['pt'].isin(high_var_pt)]

# df['univar_z'] = np.log(df['univar_z'])
# df['bivar_z'] = np.log(df['bivar_z'])
# %%
palettes = sns.palettes.mpl_palette('viridis', n_colors=len(rand_effs))

def abline(slope, intercept, xlims=None):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    if xlims is None:
        x_vals = np.array(axes.get_xlim())
    else:
        x_vals = np.array(xlims)
    y_vals = intercept + slope * x_vals
    return x_vals, y_vals

import statsmodels as sm
import statsmodels.formula.api as smf

mod_type = 'rsri'

statmodel_df = pd.DataFrame({})

for feature in ['univar_z', 'bivar_z']:
    if mod_type == 'rs':
        mod = smf.mixedlm(
            "sz_sev ~ univar_z + bivar_z",
            df,
            re_formula="0 + univar_z + bivar_z",
            groups=df['pt_num']
            )
    if mod_type == 'ri':
        mod = smf.mixedlm(
            "sz_sev ~ mean_pi",
            df,
            groups=df['pt_num']
            )
    if mod_type == 'rsri':
        mod = smf.mixedlm(
            f"sz_sev ~ {feature}",
            df,
            re_formula=f"~{feature}",
            groups=df['pt_num']
            )

    mod = mod.fit(method=["lbfgs"], reml=True)
    rand_effs = mod.random_effects
    slope = mod.fe_params[feature]
    intercept = mod.fe_params['Intercept']

    display(mod.summary())
    # ax.plot(x, y, color='mediumorchid', alpha=1, lw=2)

    for (pt, randef), col in zip(rand_effs.items(), palettes):
        xmin = df[df['pt_num'] == pt][feature].min()
        xmax = df[df['pt_num'] == pt][feature].max()
        if mod_type == 'rs':
            pt_randef = randef['bivar_z']
            x, y = abline(pt_randef + slope, intercept, xlims=[xmin, xmax])
            statmodel_df.at[pt_num2str[pt], f"{feature}_rand_slope"] = pt_randef

        if mod_type == 'ri':
            pt_randef = randef['Group']
            x, y = abline(slope, pt_randef + intercept, xlims=[xmin, xmax])
            statmodel_df.at[pt_num2str[pt], f"{feature}_rand_int"] = pt_randef

        if mod_type == 'rsri':
            pt_randsl = randef[feature]
            pt_randint = randef['Group']
            x, y = abline(pt_randsl + slope, pt_randint + intercept,
            # xlims=[xmin, xmax],
            xlims=[0, 10]
            )

            statmodel_df.at[pt_num2str[pt], f"{feature}_rand_slope"] = pt_randsl
            statmodel_df.at[pt_num2str[pt], f"{feature}_rand_int"] = pt_randint
    

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim([0, 7])
    ax.set_ylim([0, 2.5])

    ax.set_xlabel("Pre-ictal change from interictal baseline")
    ax.set_ylabel("Seizure severity (a.u.)")


    for pt_num, group in df.groupby('pt_num'):
        ax.scatter(
            group[feature],
            group['sz_sev'],
            color='grey',
            alpha=0.3,
        )

    for (pt, row), col in zip(statmodel_df.sort_values(by=f"{feature}_rand_slope").iterrows(), palettes):
        pt_randsl = row[f"{feature}_rand_slope"]
        pt_randint = row[f"{feature}_rand_int"]
        x, y = abline(pt_randsl + slope, pt_randint + intercept, xlims=[0, 10])
        ax.plot(x, y, c=col, alpha=0.6)

    x, y = abline(slope, intercept)
    ax.plot(x, y, color='k', alpha=1, lw=2.5, ls='--')

    fig.savefig(f"../figures/pi_rand_eff_{feature}.pdf")
# statmodel_df = pd.DataFrame(statmodel_df).T


# %%
# positive and negative separately
pos_pt = statmodel_df[(statmodel_df['bivar_z_rand_slope'] + slope > 0)].index
neg_pt = statmodel_df[(statmodel_df['bivar_z_rand_slope'] + slope < 0)].index

for subgroup in [pos_pt, neg_pt]:
    subgroup_df = df[df['pt'].isin(subgroup)]
    subgroup_statmodel_df = statmodel_df[statmodel_df.index.isin(subgroup)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim([0, 7])
    ax.set_ylim([0, 2.5])

    ax.set_xlabel("Pre-ictal change from interictal baseline")
    ax.set_ylabel("Seizure severity (a.u.)")


    for pt_num, group in subgroup_df.groupby('pt_num'):
        ax.scatter(
            group[feature],
            group['sz_sev'],
            color='grey',
            alpha=0.3,
        )

    palettes = sns.palettes.mpl_palette('viridis', n_colors=len(rand_effs))

    for (pt, row), col in zip(subgroup_statmodel_df.sort_values(by=f"{feature}_rand_slope").iterrows(), palettes):
        pt_randsl = row[f"{feature}_rand_slope"]
        pt_randint = row[f"{feature}_rand_int"]
        x, y = abline(pt_randsl + slope, pt_randint + intercept, xlims=[0, 10])
        ax.plot(x, y, 
            # c=col, 
            alpha=0.6)

        group = subgroup_df[subgroup_df["pt"] == pt]
        ax.scatter(
            group[feature],
            group['sz_sev'],
            # color='grey',
            alpha=0.3,
        )

    x, y = abline(slope, intercept)
    ax.plot(x, y, color='k', alpha=1, lw=2.5, ls='--')

# %%
import plotly.express as px

high_var_pt = pt_sz_var.index[pt_sz_var > 0.1]

tab = statmodel_df
tab = statmodel_df[statmodel_df.index.isin(high_var_pt)]

px.scatter(
    tab,
    x="univar_z_rand_int",
    y="bivar_z_rand_int",
    symbol=tab.index,
)

px.scatter(
    tab,
    x="univar_z_rand_slope",
    y="bivar_z_rand_slope",
    symbol=tab.index,
)
# %%
# to figure out which patients have positive and negative slopes
display(statmodel_df.sort_values(by='bivar_z_rand_slope')['bivar_z_rand_slope'] + slope)
# %%
# Look at differences within patients
all_sev_diff = []
all_ft_diff = []

for pt, group in df.groupby('pt'):
    if len(group) < 2:
        continue
    print(pt)
    triu_inds = np.triu_indices(len(group), k=1)

    sev_diff = np.abs((group['sz_sev'].values - group['sz_sev'].values[:, None])[triu_inds])
    ft_diff = np.abs((group['univar_z'].values - group['univar_z'].values[:, None])[triu_inds])

    plt.figure()
    plt.scatter(ft_diff, sev_diff)
    plt.title(pt)
    all_sev_diff.extend(sev_diff)
    all_ft_diff.extend(ft_diff)

plt.figure()
plt.scatter(all_ft_diff, all_sev_diff)
plt.title('all')
# %%


# %%
