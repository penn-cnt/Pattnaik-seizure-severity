# %%
from os.path import join as ospj
import json
import pickle

import pandas as pd
from scipy.io import loadmat
# %% Query electrode level data
elec_ROI = loadmat("../data/electrode_localization/elecs2ROI_with_hemisphere.mat")['elec2ROIOut']
n_patients = elec_ROI.shape[0]

elec_ROI_dict = {}

# parse matlab file
for i_pt in range(n_patients):
    pt = elec_ROI[i_pt, 0][0][0][0]
    coords_and_roi = elec_ROI[i_pt, 1]
    n_coords = coords_and_roi.shape[0]

    elec_label = []
    elec_coords = []
    roi_allregions = []
    roi_maxregions = []
    for i_coords in range(n_coords):
        elec_label.append(coords_and_roi[i_coords, 0][0][0])
        elec_coords.append(coords_and_roi[i_coords, 0][1][0])
        roi_allregions.append([i[0] for i in coords_and_roi[i_coords, 0][2]])
        roi_maxregions.append(coords_and_roi[i_coords, 0][3][0][0])

    col_names = ['Labels', 'Coordinates', 'All Regions', 'Max Region']
    df = pd.DataFrame([elec_label, elec_coords, roi_allregions, roi_maxregions], index = col_names).T
    elec_ROI_dict[pt] = df

elec_ROI = elec_ROI_dict
del elec_ROI_dict

with open("../data/electrode_localization/elec_ROI.pkl", 'wb') as handle:
    pickle.dump(elec_ROI, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
data_path = "../data"
dkt_mni_labels = pd.read_csv(ospj(data_path, 'electrode_localization', 'dkt_mni_labels.csv'))

atlas_cortical_info = pd.read_excel(ospj(data_path, 'electrode_localization', 'FreeSurferDKT_Cortical.nii.xlsx'))
atlas_subcortical_info = pd.read_excel(ospj(data_path, 'electrode_localization', 'FreeSurferDKT_Subcortical.xlsx'))

atlas_cortical_info = atlas_cortical_info.iloc[0:2, :].set_index('Name').T
atlas_subcortical_info = atlas_subcortical_info.iloc[0:2, :].set_index('Name').T

atlas_info = atlas_cortical_info.append(atlas_subcortical_info)

for index, row in atlas_info.iterrows():
    dkt_row = dkt_mni_labels[dkt_mni_labels['roiNames'] == index]

    atlas_info.at[index, 'isLeft'] = bool(dkt_row['isLeft'].values)
    atlas_info.at[index, 'parcels'] = int(dkt_row['parcels'].values)
atlas_info['regions'] = atlas_info.index
atlas_info.set_index('parcels', inplace=True)
atlas_info.rename_axis(None, axis=1, inplace=True)
atlas_info.index.name = 'regionID'

atlas_info.to_csv(ospj(data_path, 'electrode_localization', 'dkt_atlas_info.csv'))
# %%

# %%
