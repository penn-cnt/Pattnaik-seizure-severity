'''
This script selects electrodes
'''
# %%
# pylint: disable-msg=C0103
import os
import json
from os.path import join as ospj
import warnings

import numpy as np
# from ieeg.auth import Session
import pandas as pd
from scipy.io import savemat
import tools

warnings.filterwarnings("ignore")

# %%
patients, labels, ignore, resect, gm_wm, coords, region, soz = \
    tools.pull_patient_localization("../data/metadata/patient_localization_final.mat")


# %% Load metadata file
metadata = pd.read_excel("../data/metadata/patient_table.xlsx")

for index, row in metadata.iterrows():
    patient = row['Patient']
    iEEG_filename = row['portal_ID']

    localization_ind = patients.index(patient)
    pt_labels = labels[localization_ind]
    pt_ignore = ignore[localization_ind].T[0]
    pt_resect = resect[localization_ind].T[0]
    pt_gm_wm = gm_wm[localization_ind].T[0]

    # Set up region list
    pt_region = []
    for i in region[localization_ind]:
        if len(i[0]) == 0:
            pt_region.append('')
        else:
            pt_region.append(i[0][0])


    pt_soz = soz[localization_ind].T[0]

    df_data = {
        'labels': pt_labels,
        'ignore': pt_ignore,
        'resect': pt_resect,
        'gm_wm': pt_gm_wm,
        'region': pt_region,
        'soz': pt_soz
        }

    # print(f"Starting pipeline for {patient}, iEEG filename is {iEEG_filename}")
    df = pd.DataFrame(df_data).reset_index()

    df_filtered = df[df['ignore'] != 1]

    # Remove white matter and non-localized electrodes
    df_filtered = df_filtered[df_filtered['gm_wm'] != -1]
    # Sort rows in alphabetical order by electrode name, easier to read with iEEG.org
    df_filtered.sort_values(by=['labels'], inplace=True)

    mdic = {
        "iEEGFilename": iEEG_filename,
        "targetElectrodesRegionInds": np.array(df_filtered['index']),
        "Regions": list(df_filtered['region']),
        "electrodeNames": list(df_filtered['labels'])
    }

    patient_data_path = f"../data/{patient}"
    if not os.path.exists(patient_data_path):
        os.makedirs(patient_data_path)

    save_path = ospj(patient_data_path, "selected_electrodes_elec-all.mat")
    savemat(save_path, mdic)

    # print(f"\t{patient} has {len(mdic['Regions'])} channels after filtering")
    # print(f"\tResults are saved in {save_path}")

# %%
