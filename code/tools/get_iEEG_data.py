# pylint: disable-msg=C0103
from ieeg.auth import Session
import pandas as pd
import pickle
from numbers import Number
import numpy as np
from .clean_labels import clean_labels
import time

def _pull_iEEG(ds, start_usec, duration_usec, channel_ids):
    '''
    Pull data while handling iEEGConnectionError
    '''
    while True:
        try:
            data = ds.get_data(start_usec, duration_usec, channel_ids)
            return data
        except Exception as e:
            time.sleep(1)

def get_iEEG_data(username, password_bin_file, iEEG_filename, start_time_usec, stop_time_usec, select_electrodes=None, ignore_electrodes=None, outputfile=None):
    """"
    2020.04.06. Python 3.7
    Andy Revell, adapted by Akash Pattnaik (2021.06.23)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Purpose:
    To get iEEG data from iEEG.org. Note, you must download iEEG python package from GitHub - instructions are below
    1. Gets time series data and sampling frequency information. Specified electrodes are removed.
    2. Saves as a pickle format
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Input
        username: your iEEG.org username
        password_bin_file: your iEEG.org password bin_file
        iEEG_filename: The file name on iEEG.org you want to download from
        start_time_usec: the start time in the iEEG_filename. In microseconds
        stop_time_usec: the stop time in the iEEG_filename. In microseconds.
            iEEG.org needs a duration input: this is calculated by stop_time_usec - start_time_usec
        ignore_electrodes: the electrode/channel names you want to exclude. EXACT MATCH on iEEG.org. Caution: some may be LA08 or LA8
        outputfile: the path and filename you want to save.
            PLEASE INCLUDE EXTENSION .pickle.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Output:
        Saves file outputfile as a pickle. For more info on pickling, see https://docs.python.org/3/library/pickle.html
        Briefly: it is a way to save + compress data. it is useful for saving lists, as in a list of time series data and sampling frequency together along with channel names
        List index 0: Pandas dataframe. T x C (rows x columns). T is time. C is channels.
        List index 1: float. Sampling frequency. Single number
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Example usage:
    username = 'arevell'
    password = 'password'
    iEEG_filename='HUP138_phaseII'
    start_time_usec = 248432340000
    stop_time_usec = 248525740000
    removed_channels = ['EKG1', 'EKG2', 'CZ', 'C3', 'C4', 'F3', 'F7', 'FZ', 'F4', 'F8', 'LF04', 'RC03', 'RE07', 'RC05', 'RF01', 'RF03', 'RB07', 'RG03', 'RF11', 'RF12']
    outputfile = '/Users/andyrevell/mount/DATA/Human_Data/BIDS_processed/sub-RID0278/eeg/sub-RID0278_HUP138_phaseII_248432340000_248525740000_EEG.pickle'
    get_iEEG_data(username, password, iEEG_filename, start_time_usec, stop_time_usec, removed_channels, outputfile)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    To run from command line:
    python3.6 -c 'import get_iEEG_data; get_iEEG_data.get_iEEG_data("arevell", "password", "HUP138_phaseII", 248432340000, 248525740000, ["EKG1", "EKG2", "CZ", "C3", "C4", "F3", "F7", "FZ", "F4", "F8", "LF04", "RC03", "RE07", "RC05", "RF01", "RF03", "RB07", "RG03", "RF11", "RF12"], "/gdrive/public/DATA/Human_Data/BIDS_processed/sub-RID0278/eeg/sub-RID0278_HUP138_phaseII_D01_248432340000_248525740000_EEG.pickle")'
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #How to get back pickled files
    with open(outputfile, 'rb') as f: data, fs = pickle.load(f)
    """

    # print("\n\nGetting data from iEEG.org:")
    # print("iEEG_filename: {0}".format(iEEG_filename))
    # print("start_time_usec: {0}".format(start_time_usec))
    # print("stop_time_usec: {0}".format(stop_time_usec))
    # print("ignore_electrodes: {0}".format(ignore_electrodes))
    # if outputfile:
    #     print("Saving to: {0}".format(outputfile))
    # else:
    #     print("Not saving, returning data and sampling frequency")

    # Pull and format metadata from patient_localization_mat
    
    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec

    pwd = open(password_bin_file, 'r').read()

    while True:
        try:
            s = Session(username, pwd)
            ds = s.open_dataset(iEEG_filename)
            all_channel_labels = ds.get_channel_labels()
            break
        except Exception as e:
            time.sleep(1)

    all_channel_labels = clean_labels(all_channel_labels)

    if select_electrodes is not None:
        if isinstance(select_electrodes[0], Number):
            channel_ids = select_electrodes
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(select_electrodes[0], str):
            select_electrodes = clean_labels(select_electrodes)
            
            channel_ids = [i for i, e in enumerate(all_channel_labels) if e in select_electrodes]
            channel_names = select_electrodes
        else:
            print("Electrodes not given as a list of ints or strings")

    elif ignore_electrodes is not None:
        if isinstance(ignore_electrodes[0], int):
            channel_ids = [i for i in np.arange(len(all_channel_labels)) if i not in ignore_electrodes]
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(ignore_electrodes[0], str):
            ignore_electrodes = clean_labels(ignore_electrodes)

            channel_ids = [i for i, e in enumerate(all_channel_labels) if e not in ignore_electrodes]
            channel_names = [e for e in all_channel_labels if e not in ignore_electrodes]
        else:
            print("Electrodes not given as a list of ints or strings")

    else:
        channel_ids = np.arange(len(all_channel_labels))
        channel_names = all_channel_labels

    try:
        # data = ds.get_data(start_time_usec, duration, channel_ids)
        data =  _pull_iEEG(ds, start_time_usec, duration, channel_ids)
    except Exception as e:
        # clip is probably too big, pull chunks and concatenate
        clip_size = 60 * 1e6

        clip_start = start_time_usec
        data = None
        while clip_start + clip_size < stop_time_usec:
            if data is None:
                # data = ds.get_data(clip_start, clip_size, channel_ids)
                data = _pull_iEEG(ds, clip_start, clip_size, channel_ids)
            else:
                # data = np.concatenate(([data, ds.get_data(clip_start, clip_size, channel_ids)]), axis=0)
                data = np.concatenate(([data, _pull_iEEG(ds, clip_start, clip_size, channel_ids)]), axis=0)
            clip_start = clip_start + clip_size
        # data = np.concatenate(([data, ds.get_data(clip_start, stop_time_usec - clip_start, channel_ids)]), axis=0)

    df = pd.DataFrame(data, columns=channel_names)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate #get sample rate

    if outputfile:
        with open(outputfile, 'wb') as f: pickle.dump([df, fs], f)
    else:
        return df, fs

    # session.delete
    # clear variables

""""
Download and install iEEG python package - ieegpy
GitHub repository: https://github.com/ieeg-portal/ieegpy
If you downloaded this code from https://github.com/andyrevell/paper001.git then skip to step 2
1. Download/clone ieepy. 
    git clone https://github.com/ieeg-portal/ieegpy.git
2. Change directory to the GitHub repo
3. Install libraries to your python Path. If you are using a virtual environment (ex. conda), make sure you are in it
    a. Run:
        python setup.py build
    b. Run: 
        python setup.py install
              
"""