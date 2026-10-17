import numpy as np

from .qrs_detection import PanTompkinsDetector


def peak_window_mask(peaks, n_samples, half_width):
    mask = np.zeros(n_samples, dtype=bool)
    for p in peaks:
        mask[max(0, p - half_width):min(n_samples, p + half_width + 1)] = True
    return mask


def oracle_mask(r_peaks, n_samples, fs=250.0):
    return peak_window_mask(r_peaks, n_samples, int(0.04 * fs))


def random_burst_mask(n_samples, seed, duty=0.045, hold=8):
    mask = np.zeros(n_samples, dtype=bool)
    rng = np.random.RandomState(seed)
    n_bursts = int(round(duty * n_samples / hold))
    for start in rng.choice(max(1, n_samples - hold), size=n_bursts, replace=False):
        mask[start:start + hold] = True
    return mask


def shifted_peak_mask(r_peaks, n_samples, fs=250.0, shift_sec=0.15, half_width=4):
    mask = np.zeros(n_samples, dtype=bool)
    shift = int(shift_sec * fs)
    for p in r_peaks:
        center = p + shift
        mask[max(0, center - half_width):min(n_samples, center + half_width)] = True
    return mask


def pan_tompkins_buffered_mask(d, fs=250.0):
    peaks = PanTompkinsDetector(fs=fs).detect(d)
    return peak_window_mask(peaks, len(d), int(0.04 * fs))


def pan_tompkins_causal_mask(d, fs=250.0):
    n_samples = len(d)
    peaks = PanTompkinsDetector(fs=fs).detect(d)
    half_width = int(0.04 * fs)
    if len(peaks) <= 1:
        return peak_window_mask(peaks, n_samples, half_width)
    mask = np.zeros(n_samples, dtype=bool)
    rr_history = []
    for i, p in enumerate(peaks):
        if i > 0:
            rr = p - peaks[i - 1]
            if int(0.3 * fs) < rr < int(2.0 * fs):
                rr_history.append(rr)
        predicted = p + (int(np.median(rr_history)) if rr_history else int(0.8 * fs))
        if predicted < n_samples:
            mask[max(0, predicted - half_width):min(n_samples, predicted + half_width + 1)] = True
    return mask


def depinto_buffered_mask(d, fs=250.0, theta=0.20, hold=10, delay_sec=0.046, decay=0.998):
    n_samples = len(d)
    delay = int(round(delay_sec * fs))
    mask = np.zeros(n_samples, dtype=bool)
    peak_follower = 0.5
    for n in range(1, n_samples):
        peak_follower = max(abs(d[n]), decay * peak_follower)
        if abs(d[n] - d[n - 1]) / (peak_follower + 1e-4) >= theta:
            mask[max(0, n - delay):min(n_samples, n + hold)] = True
    return mask
