"""
This script finds times to use for 30 second interictal clips based on sleep,
proximity to seizure times
"""
# pylint: disable-msg=C0103

# %% Imports
import os
from os.path import join as ospj
import json

import pandas as pd
import numpy as np
from scipy.io import loadmat

# buffers (in seconds)
IMPL_EFF_BUFF = 72 * 60 * 60
SCS_BUFF = 2 * 60 * 60
FOC_BUFF = 6 * 60 * 60
GEN_BUFF = 12 * 60 * 60
BEFORE_BUFF = 2 * 60 * 60

# seed for getting clips
rng = np.random.default_rng(2021)
data_path = "../data"
# %%
sz_metadata = pd.read_excel(ospj(data_path, "metadata", "seizure_metadata_with_severity.xlsx"), index_col=0)
# %%
# Read sleep times mat file
sleep_times = loadmat(ospj(data_path, "metadata", "sleeptimes.mat"))['sleepdata']

all_patients = np.squeeze(sleep_times['name'])
all_files = np.squeeze(sleep_times['file'])
all_times = np.squeeze(sleep_times['t'])
all_sleep = np.squeeze(sleep_times['sleep'])

n_patients = all_patients.shape[0]

# %% function to get longest segments of interictal
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    # idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] - 1 # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

# %%
for i_pt in range(n_patients):
    patient = all_patients[i_pt][0]
    file = np.squeeze(all_files[i_pt])
    time = np.squeeze(all_times[i_pt])

    # remove this line to run on all patients (not just the ones in the atlas cohort)
    if patient not in sz_metadata['Patient'].values:
        continue

    # times that are during wake
    is_wake = ~np.squeeze(all_sleep[i_pt]).astype(bool)

    # times that are outside of seizures
    is_outside_sz = np.ones(time.shape, dtype=bool)
    seizures = sz_metadata[sz_metadata['Patient'] == patient]

    if np.unique(file).size == 1:
        # times that are not in implant effect window
        is_not_impl = time > IMPL_EFF_BUFF

        for i_sz, row in seizures.iterrows():
            sz_start = row['Seizure EEC']
            sz_end = row['Seizure end']
            sz_type = row['Seizure type']

            is_before = np.logical_and(sz_start - BEFORE_BUFF < time, time < sz_start)

            if sz_type == "Subclinical":
                AFTER_BUFF = SCS_BUFF
            if sz_type in ['FAS', 'FIAS']:
                AFTER_BUFF = FOC_BUFF
            if sz_type == "FBTCS":
                AFTER_BUFF = GEN_BUFF
            is_after = np.logical_and(sz_end < time, time < sz_end + AFTER_BUFF)

            is_outside_sz = np.logical_and(
                is_outside_sz,
                np.logical_and(
                    ~is_before,
                    ~is_after
                )
            )
    else:
        # times that are not in implant effect window
        is_not_impl = np.logical_or(time > IMPL_EFF_BUFF, file != 1)

        for i_sz, row in seizures.iterrows():
            sz_start = row['Seizure EEC']
            sz_end = row['Seizure end']
            sz_type = row['Seizure type']
            file_num = int(row['iEEG Filename'][-1])

            is_correct_file = file == file_num

            is_before = np.logical_and(sz_start - BEFORE_BUFF < time, time < sz_start)

            if sz_type == "Subclinical":
                AFTER_BUFF = SCS_BUFF
            if sz_type in ['FAS', 'FIAS']:
                AFTER_BUFF = FOC_BUFF
            if sz_type == "FBTCS":
                AFTER_BUFF = GEN_BUFF
            is_after = np.logical_and(sz_end < time, time < sz_end + AFTER_BUFF)

            is_outside_sz = np.logical_and(
                is_outside_sz,
                np.logical_and(
                    ~is_before,
                    ~is_after
                )
            )
        is_outside_sz = np.logical_and(
            is_outside_sz,
            is_correct_file
        )

    # put all the conditions together
    interictal_mask = np.logical_and(
        is_wake,
        np.logical_and(
            is_not_impl,
            is_outside_sz
        )
    )

    # get segments where all values are true and sort by longest
    true_segments = contiguous_regions(interictal_mask)
    best_interictal_times = time[true_segments]
    best_interictal_files = file[true_segments]

    # keep only segments that are within one file
    keep_segments = best_interictal_files[:, 0] - best_interictal_files[:, 1] == 0

    best_interictal_times = best_interictal_times[keep_segments]
    interictal_files = best_interictal_files[keep_segments][:, 0]

    longest_segments = np.argsort(-1*(best_interictal_times[:, 1] - best_interictal_times[:, 0]), kind='stable')
    best_interictal_times = best_interictal_times[longest_segments]
    interictal_files = interictal_files[longest_segments]

    # get timestamps to search for interictal clips and save
    np.save(ospj(data_path, "metadata", "best_interictal_times.npy"), best_interictal_times)
    np.save(ospj(data_path, "metadata", "best_interictal_times_filenum.npy"), interictal_files)


    ii_win_len = 30
    clips = set()
    file_nums = []

    n_periods = 2
    n_clips_per_period = 10
    if len(best_interictal_times) == 1:
        n_periods = 1
        n_clips_per_period = 20

    for ii_period in range(n_periods):
        start = best_interictal_times[ii_period, 0]
        end = best_interictal_times[ii_period, 1]

        file_num = interictal_files[ii_period]

        for _ in range(n_clips_per_period):
            rng = np.random.RandomState(2021)
            temp = rng.uniform(start, end)
            while any(temp >= existing_st and temp <= existing_st + ii_win_len for existing_st in clips):
                temp = rng.uniform(start, end)
            clips.add(temp)
            file_nums.append(file_num)

    clips = np.array(list(clips))
    file_nums = np.array(file_nums)

    # print(len(clips))
    # print(len(file_nums))
    # print(file_nums)
    np.save(ospj(data_path, patient, "interictal_clip_times.npy"), clips)
    np.save(ospj(data_path, patient, "interictal_clip_files.npy"), file_nums)

# %%
