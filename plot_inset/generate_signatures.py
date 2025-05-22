import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pickle
import os
from tqdm import tqdm  # Import tqdm for progress bar
plt.rcParams.update({'font.size': 18})  # Set global font size to 14

def find_extreme_window(signal, mean_y, peak_detection_params=None):
    """
    Automatically find the most extreme feature (peak or valley) in a signal
    that has the maximum vertical distance from the mean value,
    and determine the window boundaries where the signal crosses the mean value.
    
    Parameters:
    -----------
    signal : numpy.ndarray
        The input signal array
    peak_detection_params : dict, optional
        Additional parameters to pass to scipy.signal.find_peaks
        
    Returns:
    --------
    dict
        A dictionary containing:
        - 'left_idx': Left boundary index of the window
        - 'right_idx': Right boundary index of the window
        - 'extremum_idx': Index of the extremum (peak or valley)
        - 'extremum_value': Value of the extremum
        - 'mean_value': Mean value of the signal
        - 'window_mask': Boolean mask of points within the window
        - 'is_minimum': Boolean indicating if the extremum is a minimum (valley)
        - 'vertical_distance': Vertical distance from mean to extremum
    """
    import numpy as np
    from scipy.signal import find_peaks
    
    # Set default peak detection parameters if none provided
    if peak_detection_params is None:
        peak_detection_params = {}
    
    # Create x indices
    x = np.arange(len(signal))
    
    # Calculate the mean
    # mean_y = np.mean(signal)
    
    # Find all peaks (maxima)
    peaks, _ = find_peaks(signal, **peak_detection_params)
    
    # Find all valleys (minima) by inverting the signal
    valleys, _ = find_peaks(-signal, **peak_detection_params)
    
    # Calculate vertical distances from mean for peaks
    peak_distances = np.abs(signal[peaks] - mean_y) if len(peaks) > 0 else np.array([0])
    
    # Calculate vertical distances from mean for valleys
    valley_distances = np.abs(signal[valleys] - mean_y) if len(valleys) > 0 else np.array([0])
    
    # Find maximum distance and whether it's a peak or valley
    max_peak_distance = np.max(peak_distances) if len(peaks) > 0 else 0
    max_valley_distance = np.max(valley_distances) if len(valleys) > 0 else 0
    
    # Determine if the most extreme point is a peak or valley
    is_minimum = max_valley_distance > max_peak_distance
    
    if is_minimum and len(valleys) > 0:
        # It's a valley
        extrema = valleys
        extremum_idx = valleys[np.argmax(valley_distances)]
        vertical_distance = max_valley_distance
        # For a valley, we're looking for points BELOW the mean
        across_mean = signal < mean_y
    elif not is_minimum and len(peaks) > 0:
        # It's a peak
        extrema = peaks
        extremum_idx = peaks[np.argmax(peak_distances)]
        vertical_distance = max_peak_distance
        # For a peak, we're looking for points ABOVE the mean
        across_mean = signal > mean_y
    else:
        # No peaks or valleys found
        return {
            'left_idx': None,
            'right_idx': None,
            'extremum_idx': None,
            'extremum_value': None,
            'mean_value': mean_y,
            'window_mask': np.zeros_like(signal, dtype=bool),
            'is_minimum': None,
            'vertical_distance': 0
        }
    
    extremum_value = signal[extremum_idx]
    
    # Find the window boundaries around the extremum
    # Start from the extremum and move left until we cross the mean boundary
    left_idx = extremum_idx
    while left_idx > 0 and across_mean[left_idx]:
        left_idx -= 1
    
    # Start from the extremum and move right until we cross the mean boundary
    right_idx = extremum_idx
    while right_idx < len(signal) - 1 and across_mean[right_idx]:
        right_idx += 1
    
    # Create the window mask
    window_mask = np.zeros_like(signal, dtype=bool)
    window_mask[left_idx:right_idx+1] = True
    
    # Combine with across-mean mask to get final window points across the mean
    points_across_mean = window_mask & across_mean
    
    #return {
    #    'left_idx': left_idx,
    #    'right_idx': right_idx,
    #    'extremum_idx': extremum_idx,
    #    'extremum_value': extremum_value,
    #    'mean_value': mean_y,
    #    'window_mask': window_mask,
    #    'points_across_mean': points_across_mean,
    #    'is_minimum': is_minimum,
    #    'vertical_distance': vertical_distance
    #}
    return window_mask

#with open('simdata_tip.pkl', 'rb') as f:
#    simdata_tip = pickle.load(f)
#
#base_values = np.loadtxt('base_values.txt',delimiter=',')
#
## Create a list of lists to store all the signatures
## Each sim has signatures of cx and cy. Here we must make two separate lists of lists
#signatures_x = []
#signatures_y = []
#ncilia = 6
#
#for isim in range(len(simdata_tip)):
#    for icilia in range(ncilia):
#        signal = simdata_tip[isim][:,icilia,0]  # simdata_tip shape: [ntime,ncilia,ndim]
#          

