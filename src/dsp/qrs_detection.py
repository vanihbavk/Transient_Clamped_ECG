import numpy as np
import scipy.signal as signal


class PanTompkinsDetector:
    def __init__(self, fs=250.0):
        self.fs = fs
        nyquist = 0.5 * fs
        self.b_band, self.a_band = signal.butter(2, [5.0 / nyquist, 15.0 / nyquist], btype='band')
        self.integration_window = max(1, int(0.15 * fs))

    def bandpass(self, ecg):
        return signal.lfilter(self.b_band, self.a_band, ecg)

    def differentiate(self, x):
        kernel = np.array([2.0, 1.0, 0.0, -1.0, -2.0]) / (8.0 / self.fs)
        return signal.lfilter(kernel, [1.0], x)

    def integrate(self, x):
        kernel = np.ones(self.integration_window) / self.integration_window
        return signal.lfilter(kernel, [1.0], x)

    def detect(self, ecg):
        mwi = self.integrate(self.differentiate(self.bandpass(ecg)) ** 2)
        n_samples = len(mwi)
        refractory = int(0.2 * self.fs)
        delay = int(0.12 * self.fs)
        search_radius = int(0.06 * self.fs)
        init_samples = int(min(2.0 * self.fs, n_samples))
        signal_level = float(np.percentile(mwi[:init_samples], 85)) if init_samples > 0 else 0.5
        noise_level = float(np.percentile(mwi[:init_samples], 20)) if init_samples > 0 else 0.1
        threshold = noise_level + 0.25 * (signal_level - noise_level)
        last_peak = -refractory
        peaks = []
        for n in range(2, n_samples):
            if not (mwi[n - 1] > mwi[n - 2] and mwi[n - 1] >= mwi[n]):
                continue
            value = mwi[n - 1]
            if value > threshold and n - 1 - last_peak > refractory:
                r_sample = max(0, n - 1 - delay)
                lo = max(0, r_sample - search_radius)
                hi = min(len(ecg), r_sample + search_radius + 1)
                peaks.append(lo + int(np.argmax(np.abs(ecg[lo:hi]))) if hi > lo else r_sample)
                last_peak = n - 1
                signal_level = 0.125 * value + 0.875 * signal_level
            else:
                noise_level = 0.125 * value + 0.875 * noise_level
            threshold = noise_level + 0.25 * (signal_level - noise_level)
        return np.array(peaks, dtype=int)
