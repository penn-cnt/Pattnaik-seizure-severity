#%%
'''
get interictal features
'''
# pylint: disable-msg=C0103
import os
from os.path import join as ospj
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

#%%
sz_metadata = pd.read_excel(ospj(data_path, "metadata", "seizure_metadata_with_severity.xlsx"), index_col=0)
table = sz_metadata[['Patient', 'iEEG Filename']].drop_duplicates()
# table = table.iloc[44:]

for ind, row in tqdm(table.iterrows(), total=table.shape[0]):
    target_electrodes_vars = loadmat(
        ospj(data_path, row['Patient'], "selected_electrodes_elec-all.mat")
        )
    electrodes = list(target_electrodes_vars['targetElectrodesRegionInds'][0])

    interictal_clips = np.load(ospj(data_path, row['Patient'], "interictal_clip_times.npy"))

    for i_ii, clip_start in enumerate(interictal_clips):
        if os.path.exists(ospj(data_path, row['Patient'], f"interictal_networks_{i_ii}.npy")):
            continue

        eeg, bandpower, networks = tools.get_features(
            USERNAME,
            PWD_BIN_FILE,
            row['iEEG Filename'],
            clip_start,
            clip_start + 60,
            electrodes,
            # existing_data=ospj(data_path, row['Patient'], f"interictal_eeg_{i_ii}.csv")
        )

        # eeg.to_csv(
        #     ospj(data_path, row['Patient'], f"interictal_eeg_{i_ii}.csv")
        # )
        # np.save(
        #     ospj(data_path, row['Patient'], f"interictal_bandpower_{i_ii}.npy"),
        #     bandpower
        # )
        np.save(
            ospj(data_path, row['Patient'], f"interictal_networks_{i_ii}.npy"),
            networks
            )
