#%%
'''
get_preictal features function
'''
# pylint: disable-msg=C0103
from os.path import join as ospj
import os
import json
import glob

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import tools

data_path = "../data/"

# credentials
USERNAME = json.load(open("../ieeg_credentials.json", "rb"))['usr']
PWD_BIN_FILE = json.load(open("../ieeg_credentials.json", "rb"))['pwd_bin']

NEW_FS = 200

#%%
sz_metadata = pd.read_excel(ospj(data_path, "metadata", "seizure_metadata_with_severity.xlsx"), index_col=0)
table = sz_metadata
# table = sz_metadata.iloc[65:]
for ind, row in tqdm(table.iterrows(), total=table.shape[0]):
    target_electrodes_vars = loadmat(
        ospj(data_path, row['Patient'], "selected_electrodes_elec-all.mat")
        )
    electrodes = list(target_electrodes_vars['targetElectrodesRegionInds'][0])

    if os.path.exists(ospj(data_path, row['Patient'], f"preictal_networks_sz-{row['Seizure number']}.npy")):
        continue
    
    eeg, bandpower, networks = tools.get_features(
        USERNAME,
        PWD_BIN_FILE,
        row['iEEG Filename'],
        row['Seizure EEC'] - 60,
        row['Seizure EEC'],
        electrodes,
        # existing_data=ospj(data_path, row['Patient'], f"preictal_eeg_sz-{row['Seizure number']}.csv")
    )

    # eeg.to_csv(
    #     ospj(data_path, row['Patient'], f"preictal_eeg_sz-{row['Seizure number']}.csv")
    # )
    # np.save(
    #     ospj(data_path, row['Patient'], f"preictal_bandpower_sz-{row['Seizure number']}.npy"),
    #     bandpower
    # )
    np.save(
        ospj(data_path, row['Patient'], f"preictal_networks_sz-{row['Seizure number']}.npy"),
        networks
    )

# %%
