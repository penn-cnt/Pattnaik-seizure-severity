"""
butter_bp_filter
"""
from scipy.signal import filtfilt, butter


def butter_bp_filter(data, lowcut, highcut, fs, order=3):
    """This function bandpasses data

    Args:
        data (pandas.DataFrame): Pandas dataframe with channels in columns
        lowcut (float): Lower bound of band (Hz)
        highcut (float): Higher bound of band (Hz)
        fs (int): Sample frequency
        order (int, optional): Filter order. Defaults to 3.

    Returns:
        pandas.DataFrame: Filtered data
    """
    bandpass_b, bandpass_a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
    signal_bp = filtfilt(bandpass_b, bandpass_a, data, axis=0)
    return signal_bp
