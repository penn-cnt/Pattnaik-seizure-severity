'''
    clean labels function
'''
import re
import numpy as np

def clean_labels(channel_li):
    '''
    This function cleans a list of channels and returns the new channels
    '''

    new_channels = []
    keep_channels = np.ones(len(channel_li), dtype=bool)
    for i in channel_li:
        # standardizes channel names
        regex_match = re.match(r"(\D+)(\d+)", i)
        if regex_match is None:
            new_channels.append(i)
            continue
        lead = regex_match.group(1).replace("EEG", "").strip()
        contact = int(regex_match.group(2))
        new_channels.append(f"{lead}{contact:02d}")

    return new_channels