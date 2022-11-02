
#%%
from os.path import join as ospj
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster')

data_path = "../data/metadata"
sz_metadata = pd.read_excel(ospj(data_path, "seizure_metadata_with_atlas_spread.xlsx"), index_col=0)
nhs3 = pd.read_excel(ospj(data_path, "seizure_metadata_with_nhs3.xlsx"), index_col=0)

##

# This script parses the NHS3 scoring and saves it as a dataframe.
##
# #%%
# for index, row in nhs3.iterrows():
#     pt = row['Patient']
#     sz_ueo = row['Seizure UEO']
#     score = row['NHS3']
     
#     if pt == 'HUP097':
#         continue
#     table_ind = sz_metadata[np.logical_and(sz_metadata.Patient == pt, sz_metadata['Seizure UEO'] == sz_ueo)].index[0]
#     sz_metadata.at[table_ind, "NHS3"] = score

# severity = 'Seizure severity'

# sc_idx = sz_metadata[sz_metadata['Seizure type'] == 'Subclinical'].index
# sz_metadata.at[sc_idx, 'NHS3'] = 1
# sz_metadata.dropna(subset=['NHS3'], inplace=True)
# sz_metadata = sz_metadata[sz_metadata['NHS3'] != 0]

# sz_metadata.to_excel(ospj(data_path, 'seizure_metadata_with_nhs3.xlsx'))

# # %%
# p = sns.lmplot(
#     x='NHS3',
#     y=severity,
#     data=sz_metadata,
#     line_kws={
#         'label':'Linear Reg',
#         'color': 'black',
#         }, 
#     scatter_kws={
#         'color': 'mediumslateblue'
#     },
#     legend=True,
#     height=8,
#     aspect=1
#     )
# ax = p.axes[0, 0]

# # Hide the right and top spines
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.grid(False)
# # Only show ticks on the left and bottom spines
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# ax.set_xlim([0.5, 14])
# ax.set_xlabel("NHS3 score")
# ax.set_ylabel('Seizure severity')
# ax.set_xticks(np.arange(18, step=2))

# ax.set_xlim([.5, 16])

# r, pvalue = spearmanr(sz_metadata['NHS3'], sz_metadata[severity])

# ax.legend()
# leg = ax.get_legend()
# L_labels = leg.get_texts()
# label_line = f'$r={r:.2f}$\n$p={pvalue:.1e}$'
# L_labels[0].set_text(label_line)

# FNAME = "NHS3_severity"
# plt.savefig(ospj(figure_path, f"{FNAME}.svg"), bbox_inches='tight', transparent='true')
# plt.savefig(ospj(figure_path, f"{FNAME}.png"), bbox_inches='tight', transparent='true')

# %%
