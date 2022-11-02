# pylint: disable-msg=C0103
'''
This function uses duration and spread to calculate seizure severity
'''
# %%
import json
from os.path import join as ospj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, NMF
from sklearn import metrics

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

sns.set_context('paper')

ANDY_MODELS = False
data_path = "../data/metadata"
# load in seizure metadata
if ANDY_MODELS:
    sz_metadata_original = pd.read_excel(ospj(data_path, "seizure_metadata_with_atlas_spread.xlsx"), index_col=0)
    sz_metadata = pd.read_excel(ospj(data_path, "sz_metadata_with_spread_andy.xlsx"), index_col=0)
    sz_metadata = pd.merge(sz_metadata, sz_metadata_original[['Patient', 'Seizure number', 'Number of Regions', 'Total Volume (cm^3)']], on=['Patient', 'Seizure number'])

    sz_metadata_nhs3 = pd.read_excel(ospj(data_path, "seizure_metadata_with_nhs3.xlsx"), index_col=0)
    sz_metadata = sz_metadata.merge(sz_metadata_nhs3[['Patient', 'Seizure number', 'NHS3']], on=['Patient', 'Seizure number'])

else:
    sz_metadata = pd.read_excel(ospj(data_path, "seizure_metadata_with_nhs3.xlsx"), index_col=0)

sz_metadata["Log duration"] = np.log10(sz_metadata['Seizure duration'])
sz_metadata = sz_metadata.dropna()

# %%
def calc_severity(duration_col, spread_col, nhs3_col, out=None):
    '''
    This function calculates severity in a df given spread and duration columns
    '''
    X = np.vstack((
        sz_metadata[duration_col], 
        sz_metadata[spread_col],
        sz_metadata[nhs3_col]
        )).T
    X = MinMaxScaler().fit_transform(X)

    sev = np.sqrt(np.power(X[:, 0], 2) + np.power(X[:, 1], 2) + np.power(X[:, 2], 2))
    # sev = np.sqrt(np.power(X[:, 0], 2) + np.power(X[:, 1], 2))

    if out is not None:
        sz_metadata[out] = sev
        return X
    return sev, X

def calc_severity2(duration_col, spread_col, nhs3_col, out=None):
    '''
    This function calculates severity in a df given spread and duration columns
    '''
    pca = PCA(n_components=1)
    nmf = NMF(
        n_components=1, 
        init='nndsvda', 
        max_iter=1000, 
        beta_loss='frobenius',
        solver='mu'
        )

    X = np.vstack((
        sz_metadata[duration_col], 
        sz_metadata[spread_col],
        sz_metadata[nhs3_col]
        )).T

    # X = X[:, 1:]

    sev = nmf.fit_transform((X))

    print(nmf.components_)  
    print(nmf.inverse_transform(sev).shape)
    print(X.shape)

    # print(metrics.r2_score(X[:, 2], nmf.inverse_transform(sev)[:, 2]))
    print(metrics.r2_score(X, nmf.inverse_transform(sev), multioutput='raw_values'))

    if out is not None:
        sz_metadata[out] = sev
        return X
    return sev, X

# %% 
# Calculate severity for different spread models
duration_col = "Seizure duration"
duration_col = "Log duration"
nhs3_col = "NHS3"
if ANDY_MODELS:
    models = ['Absolute Slope ', '', 'Wavenet ', 'CNN ', 'LSTM ']

    for model in models:
        spread_col = f"{model}Number of Regions"
        X_norm = calc_severity2(duration_col, spread_col, nhs3_col, out=f'Seizure severity ({model})')

    sz_metadata.to_excel(ospj(data_path, "seizure_metadata_with_severity_all_models.xlsx"))
else:
    spread_col = "Number of Regions"
    X_norm = calc_severity(duration_col, spread_col, nhs3_col, out='Seizure severity')
    X_norm = calc_severity2(duration_col, spread_col, nhs3_col, out='Seizure severity (NMF)')

    sz_metadata.to_excel(ospj(data_path, "seizure_metadata_with_severity.xlsx"))

# %%

# %%
def nmf_robustness(duration_col, spread_col, nhs3_col, n=1000):
    r2_vals = np.zeros((n, 3))
    for i in range(n):
        nmf = NMF(
            n_components=1, 
            init='nndsvda', 
            max_iter=1000, 
            beta_loss='frobenius',
            solver='mu'
            )

        X = np.vstack((
            sz_metadata[duration_col], 
            sz_metadata[spread_col],
            sz_metadata[nhs3_col]
            )).T

        sev = nmf.fit_transform((X))
        X_hat = nmf.inverse_transform(sev)
        r2_vals[i] = metrics.r2_score(X, X_hat, multioutput='raw_values')

    return r2_vals
r2_vals = nmf_robustness(duration_col, spread_col, nhs3_col)

# %%
sns.pairplot(sz_metadata[[duration_col, spread_col, nhs3_col]])
# %%
# sz_to_highlight = 49

# fig, ax = plt.subplots(figsize=(4, 4))
# c_means = ['tab:blue', 'tab:orange']
# c_means = ['pink', 'purple']
# i = 0
# palette = sns.color_palette("flare", n_colors=3)
# types = ['FAS', 'FIAS', 'FBTC']
# for index, row in sz_metadata.iterrows():
#     if row['Seizure type'] == 'Focal':
#         c = c_means[0]
#     else:
#         c = c_means[1]

#     c = 'lightgrey'
#     s = 25
#     zord = -1
#     if i == sz_to_highlight:
#         c = 'mediumslateblue'
#         s = 50
#         zord = 1

#     # c = palette[types.index(row['Seizure type'])]
#     ax.scatter(X_norm[i, 0], X_norm[i, 1], c=c, s=s, zorder=zord)
#     i += 1

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.grid(False)

# # ax.plot(
# #     [0, X_norm[sz_to_highlight, 0]], 
# #     [0, X_norm[sz_to_highlight, 1]],
# #     lw=2,
# #     color='green',
# #     ls = '--',
# #     alpha=0.5
# #     )

# ax.arrow(
#     x=0,
#     y=0,
#     dx=X_norm[sz_to_highlight, 0]*.9,
#     dy=X_norm[sz_to_highlight, 1]*.9,
#     lw=3,
#     color='purple',
#     alpha=0.5,
#     head_width=0.02
#     )

# ax.set_xlim([-0.01, X_norm[:, 0].max() * 1.1])
# ax.set_ylim([-0.01, X_norm[:, 1].max() * 1.1])

# ax.set_ylabel('# of regions with spread')
# ax.set_xlabel('Duration (log, s)')
# plt.savefig(ospj(figure_path, 'severity_scatter.png'), bbox_inches='tight', transparent='true')
# plt.savefig(ospj(figure_path, 'severity_scatter.svg'), bbox_inches='tight', transparent='true')
# fig.show()

# %%
pt_groups = sz_metadata.groupby("Patient")

n_pt = len(pt_groups)
new_tab = []
for pt, group in pt_groups:
    pt_severities = group[sev_col]

    new_tab.append({
        'Patient': pt,
        "Severities": np.array(pt_severities),
        "Min severity": pt_severities.min()
    })

new_tab = pd.DataFrame(new_tab).sort_values(by='Min severity', ascending=False).reset_index(drop=True)

color = cm.turbo(np.linspace(0, 1, n_pt))
fig, ax = plt.subplots(figsize=(2, n_pt/5))

for ind, row in new_tab.iterrows():
    pt_severities = row['Severities']
    n_sz = len(row['Severities'])
    ax.scatter(pt_severities, np.ones(n_sz)*ind, color='grey')

ax.set_xlim([-0.1, 2.5])
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.set_xlabel("Seizure severity")
ax.set_yticks(np.arange(len(pt_groups)))
ax.set_yticklabels(new_tab['Patient'])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.grid(True)
# ax.yaxis.set_major_locator(MultipleLocator(1))

fname = "patient_severity"
plt.savefig(ospj(figure_path, f"{fname}.pdf"), bbox_inches='tight', transparent='true', dpi=600)
plt.savefig(ospj(figure_path, f"{fname}.png"), bbox_inches='tight', transparent='true')

# %%
