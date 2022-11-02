#%%
import os
import json
import warnings
from os.path import join as ospj
import glob

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import iirnotch, filtfilt, butter
from tqdm import tqdm

import tools

warnings.filterwarnings(action='ignore', category=RuntimeWarning)

# Constants
F0 = 60.0  # Frequency to be removed from signal (Hz)
Q = 30.0  # Quality factor

# credentials
USERNAME = json.load(open("../ieeg_credentials.json", "rb"))['usr']
PWD_BIN_FILE = glob.glob("../*ieeglogin.bin")[0]
ELECTRODES_FNAME = "selected_electrodes_elec-all.mat"

PREICTAL_WINDOW_USEC = 30 * 1e6
# Read metadata
seizure_table = pd.read_excel("../data/metadata/seizure_table.xlsx")

# %%
def _common_average_reference(data_arg):
    data_arg = data_arg.subtract(data_arg.mean(axis=1), axis=0)
    return data_arg

# %%
for index, row in tqdm(seizure_table.iterrows(), total=len(seizure_table)):
    pt = row['Patient']

    iEEG_filename = row['iEEG Filename']
    sz_num = row['Seizure number']
    sz_EEC = row['Seizure EEC']
    sz_end = row['Seizure end']

    pt_data_path = f"../data/{pt}"

    if os.path.exists(ospj(pt_data_path, f"raw_signal_period-ictal_sz-{sz_num}.pkl")):
        continue

    target_electrodes_vars = loadmat(ospj(pt_data_path, ELECTRODES_FNAME))
    electrodes = list(target_electrodes_vars['targetElectrodesRegionInds'][0])

    sz_start_usec = sz_EEC * 1e6
    sz_end_usec = sz_end * 1e6

    # extend pull time to the nearest second, double for preictal
    duration_usec = (sz_end_usec - sz_start_usec) + PREICTAL_WINDOW_USEC
    duration_usec = np.ceil(duration_usec / 1e6) * 1e6
    duration_min = duration_usec / (1e6 * 60)

    # start clip at matched pre-ictal
    clip_start_usec = sz_start_usec - PREICTAL_WINDOW_USEC

    start_usec = clip_start_usec
    data_duration_usec = duration_usec

    data, fs = tools.get_iEEG_data(USERNAME, PWD_BIN_FILE, iEEG_filename, start_usec, start_usec + data_duration_usec, select_electrodes=electrodes)

    # extract dims
    n_samples = np.size(data, axis=0)
    n_channels = np.size(data, axis=1)

    # set time array
    t_usec = np.linspace(start_usec, start_usec + data_duration_usec, n_samples)
    t_sec = t_usec / 1e6

    # indices for 5 second non-overlapping windows
    win_size = int(1 * fs)
    ind_overlap = np.reshape(np.arange(len(t_sec)), (-1, int(win_size)))
    n_windows = np.size(ind_overlap, axis=0)

    # nan check
    # nan_mask = np.ones(n_samples, dtype=bool)
    # for win_inds in ind_overlap:
    #     if np.sum(np.isnan(data.iloc[win_inds, :]), axis=0).any():
    #         nan_mask[win_inds] = False
    #     if (np.sum(np.abs(data.iloc[win_inds, :]), axis=0) < 1/12).any():
    #         nan_mask[win_inds] = False
    #     if (np.sqrt(np.sum(np.power(np.diff(data.iloc[win_inds, :], axis=0), 2), axis=0)) > 15000).any():
    #         nan_mask[win_inds] = False

    # print(np.sum(~nan_mask))
    # signal_nan = data[nan_mask]
    # t_sec_nan = t_sec[nan_mask]

    # if len(t_sec_nan) == 0:
    #     continue

    # remove 60Hz noise
    b, a = iirnotch(F0, Q, fs)
    signal_filt = filtfilt(b, a, data, axis=0)

    # bandpass between 1 and 120Hz
    bandpass_b, bandpass_a = butter(3, [1, 120], btype='bandpass', fs=fs)
    signal_filt = filtfilt(bandpass_b, bandpass_a, signal_filt, axis=0)

    # format resulting data into pandas DataFrame
    signal_filt = pd.DataFrame(signal_filt, columns=data.columns)
    signal_filt.index = pd.to_timedelta(t_sec, unit="S")

    # re-reference the signals using common average referencing
    signal_ref = _common_average_reference(signal_filt)

    ictal_signal = signal_ref.iloc[(signal_ref.index > pd.to_timedelta(sz_EEC, unit='S')), :]
    preictal_signal = signal_ref.iloc[(signal_ref.index < pd.to_timedelta(sz_EEC, unit='S')), :]

    ictal_t_sec = t_sec[(signal_ref.index > pd.to_timedelta(sz_EEC, unit='S'))]
    preictal_t_sec = t_sec[(signal_ref.index < pd.to_timedelta(sz_EEC, unit='S'))]

    pt_signal = pd.DataFrame(ictal_signal, index=pd.to_timedelta(ictal_t_sec, unit='S'), columns=signal_ref.columns)

    signal_fname = f"raw_signal_period-ictal_sz-{sz_num}"
    pt_signal.to_pickle(ospj(pt_data_path, f"{signal_fname}.pkl"))
    # pt_signal.to_csv(ospj(pt_data_path, f"{signal_fname}.csv"))

    pt_signal = pd.DataFrame(preictal_signal, index=pd.to_timedelta(preictal_t_sec, unit='S'), columns=signal_ref.columns)

    signal_fname = f"raw_signal_period-preictal_sz-{sz_num}"

    pt_signal.to_pickle(ospj(pt_data_path, f"{signal_fname}.pkl"))
    # pt_signal.to_csv(ospj(pt_data_path, f"{signal_fname}.csv"))

# %%
