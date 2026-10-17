import json
from pathlib import Path

import numpy as np
import scipy.signal as signal
import wfdb

CODEBASE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = CODEBASE_DIR.parent
DATA_DIR = CODEBASE_DIR / 'data'
RESULTS_DIR = CODEBASE_DIR / 'docs'

FS = 250.0
TAPS = 16
SEGMENT_SEC = 60.0
SEGMENT_LEN = int(SEGMENT_SEC * FS)
EVAL_SLICE = slice(int(2.0 * FS), SEGMENT_LEN)
RECORDS = ['100', '101', '103', '105', '106', '115', '215']
NUM_TRIALS = 28
SNR_LEVELS = [-6.0, 0.0, 6.0, 12.0]
BODY_COUPLING = np.array([0.65, 0.45, -0.25, 0.15, -0.08, 0.04])
BEAT_SYMBOLS = {'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/'}


def load_ecg_record(record_id, fs=FS, max_seconds=None):
    path = str(DATA_DIR / 'mitdb' / record_id)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, 'atr')
    b, a = signal.butter(2, 0.5 / (0.5 * rec.fs), btype='high')
    ecg_native = signal.filtfilt(b, a, rec.p_signal[:, 0])
    if max_seconds is not None:
        ecg_native = ecg_native[:int(max_seconds * rec.fs)]
    ecg = signal.resample(ecg_native, int(len(ecg_native) * fs / rec.fs))
    r_peaks = []
    for sample, symbol in zip(ann.sample, ann.symbol):
        p = int(round(sample * fs / rec.fs))
        if symbol in BEAT_SYMBOLS and p < len(ecg):
            r_peaks.append(p)
    return ecg, r_peaks


def load_ecg_records(fs=FS):
    return {r: load_ecg_record(r, fs=fs) for r in RECORDS}


def load_motion_artifact(fs=FS, max_samples=None):
    rec = wfdb.rdrecord(str(DATA_DIR / 'nstdb' / 'em'))
    b_hp, a_hp = signal.butter(2, 0.2 / (0.5 * rec.fs), btype='high')
    motion = signal.resample_poly(signal.filtfilt(b_hp, a_hp, rec.p_signal[:, 0]), 25, 36)
    if max_samples is not None:
        motion = motion[:max_samples]
    b_lp, a_lp = signal.butter(3, 8.0 / (0.5 * fs), btype='low')
    source = signal.filtfilt(b_lp, a_lp, motion)
    source = source / (np.std(source) + 1e-12)
    coupled = signal.lfilter(BODY_COUPLING, 1.0, source)
    x_ref = source + 0.08 * np.random.RandomState(42).randn(len(source))
    return coupled, x_ref


def load_muscle_artifact(fs=FS):
    rec = wfdb.rdrecord(str(DATA_DIR / 'nstdb' / 'ma'))
    b, a = signal.butter(2, 10.0 / (0.5 * rec.fs), btype='high')
    ma = signal.filtfilt(b, a, rec.p_signal[:, 0])
    ma = signal.resample(ma, int(len(ma) * fs / rec.fs))
    return ma / (np.std(ma) + 1e-12)


def artifact_scale(clean, artifact, snr_db=0.0):
    return np.sqrt(np.mean(clean ** 2) / ((np.mean(artifact ** 2) + 1e-12) * 10.0 ** (snr_db / 10.0)))


def input_snr(clean, noise, eval_slice=EVAL_SLICE):
    return 10.0 * np.log10(np.sum(clean[eval_slice] ** 2) / (np.sum(noise[eval_slice] ** 2) + 1e-12))


class Trial:
    def __init__(self, index, records, coupled, x_ref, snr_db=0.0):
        self.index = index
        self.record = RECORDS[index % len(RECORDS)]
        ecg, r_peaks = records[self.record]
        ecg_start = index // len(RECORDS) * SEGMENT_LEN * 3 + 1000
        noise_start = index * SEGMENT_LEN
        self.ecg_start = ecg_start
        self.clean = ecg[ecg_start:ecg_start + SEGMENT_LEN]
        self.r_peaks = [p - ecg_start for p in r_peaks if ecg_start <= p < ecg_start + SEGMENT_LEN]
        self.artifact = coupled[noise_start:noise_start + SEGMENT_LEN] * artifact_scale(
            self.clean, coupled[noise_start:noise_start + SEGMENT_LEN], snr_db)
        self.x_ref = x_ref[noise_start:noise_start + SEGMENT_LEN]
        self.d = self.clean + self.artifact
        self.snr_in = input_snr(self.clean, self.artifact)


def iter_trials(records, coupled, x_ref, snr_db=0.0, num_trials=NUM_TRIALS):
    for i in range(num_trials):
        yield Trial(i, records, coupled, x_ref, snr_db)


def trial_metrics(clean, est, r_peaks, eval_slice=EVAL_SLICE):
    c = clean[eval_slice]
    noise = est[eval_slice] - c
    p_signal = np.sum(c ** 2)
    p_noise = np.sum(noise ** 2) + 1e-12
    metrics = {
        'snr_out': float(10.0 * np.log10(p_signal / p_noise)),
        'prd': float(np.sqrt(p_noise / (p_signal + 1e-12)) * 100.0),
        'rho_r': 100.0,
        'mae_r': 0.0,
        'r_qrs': 1.0,
    }
    peaks = [p for p in r_peaks if eval_slice.start <= p < eval_slice.stop]
    if not peaks:
        return metrics
    rho, mae, corr = [], [], []
    for p in peaks:
        lo, hi = max(0, p - 3), min(len(clean), p + 4)
        apex = lo + np.argmax(np.abs(clean[lo:hi]))
        ref, val = clean[apex], est[apex]
        if abs(ref) > 0.1:
            rho.append(val / ref * 100.0)
            mae.append(abs(val - ref) / abs(ref) * 100.0)
        lo, hi = max(0, p - 9), min(len(clean), p + 10)
        cs, es = clean[lo:hi], est[lo:hi]
        if np.std(cs) > 1e-06 and np.std(es) > 1e-06:
            corr.append(np.corrcoef(cs, es)[0, 1])
    if rho:
        metrics['rho_r'] = float(np.mean(rho))
        metrics['mae_r'] = float(np.mean(mae))
    if corr:
        metrics['r_qrs'] = float(np.mean(corr))
    return metrics


def summarize(values):
    return {'mean': float(np.mean(values)), 'std': float(np.std(values))}


def write_json(name, payload):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / name, 'w') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')


def read_json(name):
    with open(RESULTS_DIR / name) as f:
        return json.load(f)
