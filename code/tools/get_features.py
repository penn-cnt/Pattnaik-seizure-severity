"""
    This script gets coherence networks and bandpower for a clip of iEEG
"""
from fractions import Fraction
import itertools

import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt, get_window, welch, coherence, resample_poly
from scipy.integrate import simpson

from .get_iEEG_data import get_iEEG_data
from .common_avg_reference import common_avg_reference
from .butter_bp_filter import butter_bp_filter
from .format_network import format_network

bands = [
    [0.5, 4], # delta
    [4, 8], # theta
    [8, 12], # alpha
    [12, 30], # beta
    [30, 80], # gamma
    [0.5, 80] # broad
]
band_names = ["delta", "theta", "alpha", "beta", "gamma", "broad"]
N_BANDS = len(bands)

#%%
def get_features(usr, pwd_bin_file, ieeg_fname, clip_start_sec, clip_end_sec, electrodes, time='relative', new_fs=200, existing_data=None):
    """This function generates preictal features

    Args:
        patient (str): Patient HUP ID ex. "HUPXXX"
        ieeg_fname (str): Patient iEEG.org filename ex. "HUPXXX_phaseII"
        clip_start_sec (float): Start time of the clip in seconds
        clip_end_sec (float): End time of the clip in seconds
        time (str): options are "real" and "relative"
    """

    if existing_data is None:
        data, fs = get_iEEG_data(
            usr,
            pwd_bin_file,
            ieeg_fname,
            clip_start_sec*1e6,
            clip_end_sec*1e6,
            select_electrodes=electrodes
        )

        data_ref = common_avg_reference(data)

        # bandpass between 0.5 and 80 and notch filter 60Hz
        data_bandpass = butter_bp_filter(data_ref, 0.5, 80, fs)
        b, a = iirnotch(60.0, 30.0, fs)
        data_filtered = filtfilt(b, a, data_bandpass, axis=0)

        # downsample to 200 hz
        if new_fs is not None:
            frac = Fraction(new_fs, int(fs))
            data_resampled = resample_poly(data_filtered, up=frac.numerator, down=frac.denominator)
            fs = new_fs
        else:
            data_resampled = data_filtered
        (n_samples, n_channels) = data_resampled.shape

        # set time array
        t_sec = np.linspace(clip_start_sec, clip_end_sec, n_samples, endpoint=False)
        if time == 'relative':
            t_sec = t_sec - clip_end_sec

        data_resampled = pd.DataFrame(
            data_resampled,
            index=t_sec,
            columns=data.columns
        )
    else:
        data_resampled = pd.read_csv(existing_data, index_col=0)
        fs = int(np.around(1 / (data_resampled.index[1] - data_resampled.index[0])))
        (n_samples, n_channels) = data_resampled.shape
        
    # calculate psd
    window = get_window('hamming', fs * 2)
    freq, pxx = welch(
        x=data_resampled,
        fs=fs,
        window=window,
        noverlap=fs,
        axis=0
    )

    n_edges = sum(1 for i in itertools.combinations(range(n_channels), 2))

    cohers = np.zeros((len(freq), n_edges))

    for i_pair, (ch1, ch2) in enumerate(itertools.combinations(range(n_channels), 2)):
        _, pair_coher = coherence(
            data_resampled.iloc[:, ch1],
            data_resampled.iloc[:, ch2],
            fs=fs,
            window='hamming',
            nperseg=fs * 2,
            noverlap=fs
            )

        cohers[:, i_pair] = pair_coher

    # keep only between originally filtered range
    filter_idx = np.logical_and(freq >= 0.5, freq <= 80)
    freq = freq[filter_idx]
    pxx = pxx[filter_idx]
    cohers = cohers[filter_idx]

    pxx_bands = np.empty((N_BANDS, n_channels))
    coher_bands = np.empty((N_BANDS, n_edges))

    pxx_bands[-1] = np.log10(simpson(pxx, dx=freq[1] - freq[0], axis=0) + 1)
    coher_bands[-1] = np.mean(cohers, axis=0)

    # format all frequency bands
    for i_band, (lower, upper) in enumerate(bands[:-1]):
        filter_idx = np.logical_and(freq >= lower, freq <= upper)

        pxx_bands[i_band] = simpson(pxx[filter_idx], dx=freq[1] - freq[0], axis=0)
        pxx_bands[i_band] = np.log10(pxx_bands[i_band] + 1)

        coher_bands[i_band] = np.mean(cohers[filter_idx], axis=0)

    pxx_bands[:-1] = pxx_bands[:-1] /  np.sum(pxx_bands[:-1], axis=0)

    network_bands = format_network(coher_bands, n_channels)

    return data_resampled, pxx_bands, network_bands
