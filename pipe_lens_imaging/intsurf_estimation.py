import numpy as np
from scipy.signal import find_peaks

def img_line_first_echoes(img, threshold=0.5, height = 0.005):
    idx = np.argmax(img, 0).astype(np.int32)
    w = np.max(img, 0)
    good_idx = np.argwhere(w > threshold * w.max())[:, 0]
    bad_idx = np.argwhere(w <= threshold * w.max())[:, 0]
    # img = img/w

    first_peaks, factor = [], 1
    for i in np.array(bad_idx):
        while len(first_peaks) == 0:
            first_peaks = find_peaks(img[:, i], height=height[i] * factor)[0]
            if len(first_peaks) > 0:
                idx[i] = first_peaks[0]
            else:
                factor = factor * .9
        first_peaks, factor = [], 1

    w = w/np.max(w)
    return idx, w