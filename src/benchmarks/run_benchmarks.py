import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.common import (
    EVAL_SLICE,
    FS,
    NUM_TRIALS,
    RECORDS,
    SEGMENT_LEN,
    SNR_LEVELS,
    TAPS,
    Trial,
    artifact_scale,
    iter_trials,
    load_ecg_record,
    load_ecg_records,
    load_motion_artifact,
    trial_metrics,
    write_json,
)
from dsp.adaptive_filters import (
    CLAMPED,
    ADAPTING,
    STILL,
    ClampTrigger,
    FixedStepNLMS,
    ImuGatedNLMS,
    KwongJohnstonVSS,
    MaskClampedNLMS,
    RLS,
    TriggerClampedNLMS,
    run_filter,
)
from dsp.gating import (
    depinto_buffered_mask,
    oracle_mask,
    pan_tompkins_buffered_mask,
    pan_tompkins_causal_mask,
    random_burst_mask,
    shifted_peak_mask,
)

ALGORITHMS = [
    ('raw', 'Raw Input', 0, 0.0),
    ('nlms_stat_tuned', 'Plain NLMS (mu=0.002)', 48, 0.0),
    ('random_clamp', 'Random Clamp (4.5% duty)', 48, 0.0),
    ('shifted_clamp', 'Shifted Clamp (+150ms)', 48, 0.0),
    ('nlms_clamped_amp', 'Proposed Amplitude Clamp', 48, 0.0),
    ('nlms_clamped_diff', 'First-Difference Clamp', 48, 0.0),
    ('nlms_clamped_curv', 'Curvature Clamp', 48, 8.0),
    ('pt_look_nlms', 'Pan-Tompkins (Buffered 150ms)', 48, 150.0),
    ('pt_pred_nlms', 'Pan-Tompkins (Causal)', 48, 0.0),
    ('depinto_slope_delay', 'DePinto Slope Gate (Buffered 46ms)', 48, 46.0),
    ('oracle_gated', 'Oracle Beat Gating', 48, 0.0),
    ('nlms_tuned', 'Fast NLMS (mu=0.02)', 48, 0.0),
    ('vss_kwong', 'Kwong-Johnston VSS', 50, 0.0),
    ('rls', 'Plain RLS (lambda=0.998)', 256, 0.0),
    ('ablation_imu', 'IMU Kinetic Gate Only', 48, 0.0),
    ('imu_amp', 'NLMS + IMU + Amp Clamp', 50, 0.0),
    ('imu_fd', 'NLMS + IMU + First-Diff', 50, 0.0),
    ('imu_curv', 'NLMS + IMU + Curvature', 50, 0.0),
]


def build_rule(key, trial, x_imu, fs=FS):
    n_samples = len(trial.d)
    if key == 'nlms_stat_tuned':
        return FixedStepNLMS(mu=0.002)
    if key == 'random_clamp':
        return MaskClampedNLMS(random_burst_mask(n_samples, seed=42 + trial.index, duty=0.045, hold=8), mu=0.002)
    if key == 'shifted_clamp':
        return MaskClampedNLMS(shifted_peak_mask(trial.r_peaks, n_samples, fs=fs), mu=0.002)
    if key == 'nlms_clamped_amp':
        return TriggerClampedNLMS(mu=0.002, trigger='amplitude', theta=3.0, hold=8)
    if key == 'nlms_clamped_diff':
        return TriggerClampedNLMS(mu=0.002, trigger='first_difference', theta=0.7, hold=8)
    if key == 'nlms_clamped_curv':
        return TriggerClampedNLMS(mu=0.002, trigger='curvature', theta=0.38, hold=8)
    if key == 'pt_look_nlms':
        return MaskClampedNLMS(pan_tompkins_buffered_mask(trial.d, fs=fs), mu=0.002)
    if key == 'pt_pred_nlms':
        return MaskClampedNLMS(pan_tompkins_causal_mask(trial.d, fs=fs), mu=0.002)
    if key == 'depinto_slope_delay':
        return MaskClampedNLMS(depinto_buffered_mask(trial.d, fs=fs, theta=0.20, hold=10), mu=0.002)
    if key == 'oracle_gated':
        return MaskClampedNLMS(oracle_mask(trial.r_peaks, n_samples, fs=fs), mu=0.002)
    if key == 'nlms_tuned':
        return FixedStepNLMS(mu=0.02)
    if key == 'vss_kwong':
        return KwongJohnstonVSS()
    if key == 'rls':
        return RLS(lam=0.998, delta=0.1)
    if key == 'ablation_imu':
        return ImuGatedNLMS(x_imu, mu0=0.002, trigger=None)
    if key == 'imu_amp':
        return ImuGatedNLMS(x_imu, mu0=0.002, trigger='amplitude', theta=3.0, hold=8)
    if key == 'imu_fd':
        return ImuGatedNLMS(x_imu, mu0=0.002, trigger='first_difference', theta=0.7, hold=8)
    if key == 'imu_curv':
        return ImuGatedNLMS(x_imu, mu0=0.002, trigger='curvature', theta=0.38, hold=8)
    raise ValueError(f'unknown algorithm key: {key}')


def compute_isoelectric_snr(clean, est, r_peaks, eval_slice, snr_out):
    valid_peaks = [p for p in r_peaks if eval_slice.start <= p < eval_slice.stop]
    if not valid_peaks:
        return snr_out
    c = clean[eval_slice]
    noise = est[eval_slice] - c
    mask = np.ones(len(c), dtype=bool)
    for p in valid_peaks:
        local_p = p - eval_slice.start
        mask[max(0, local_p - 15):min(len(c), local_p + 15)] = False
    if np.sum(mask) <= 50:
        return snr_out
    c_iso = c[mask]
    noise_iso = noise[mask]
    return float(10.0 * np.log10((np.sum(c_iso ** 2) + 1e-12) / (np.sum(noise_iso ** 2) + 1e-12)))


def holm_bonferroni(p_values):
    sorted_indices = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.zeros(m)
    for rank, idx in enumerate(sorted_indices):
        adjusted[idx] = min(1.0, (m - rank) * p_values[idx])
    for i in range(1, m):
        curr_idx = sorted_indices[i]
        prev_idx = sorted_indices[i - 1]
        adjusted[curr_idx] = max(adjusted[curr_idx], adjusted[prev_idx])
    return adjusted


def evaluate_significance(results_raw, snr_key):
    ref_data = results_raw[snr_key]['nlms_clamped_amp']['delta_snr']
    sig_map = {}
    raw_p_vals = []
    keys_tested = []
    for key, _, _, _ in ALGORITHMS:
        if key == 'nlms_clamped_amp':
            continue
        base_data = results_raw[snr_key][key]['delta_snr']
        try:
            stat, p_val = wilcoxon(ref_data, base_data)
            ref_rec = [float(np.mean(ref_data[i::len(RECORDS)])) for i in range(len(RECORDS))]
            base_rec = [float(np.mean(base_data[i::len(RECORDS)])) for i in range(len(RECORDS))]
            stat_rec, p_val_rec = wilcoxon(ref_rec, base_rec)
            sig_map[key] = {
                'p_value': float(p_val),
                'statistic': float(stat),
                'p_value_record': float(p_val_rec),
                'statistic_record': float(stat_rec),
                'significant': bool(p_val < 0.05),
                'significant_record': bool(p_val_rec < 0.05),
            }
            raw_p_vals.append(float(p_val))
            keys_tested.append(key)
        except Exception as err:
            sig_map[key] = {
                'p_value': 1.0,
                'statistic': 0.0,
                'p_value_record': 1.0,
                'statistic_record': 0.0,
                'significant': False,
                'significant_record': False,
                'error': str(err),
            }
    if raw_p_vals:
        holm_p = holm_bonferroni(raw_p_vals)
        for idx, k_test in enumerate(keys_tested):
            sig_map[k_test]['p_value_holm'] = float(holm_p[idx])
    return sig_map


def evaluate_rls_generalization(records, coupled, x_ref):
    out = {}
    for lam in [0.985, 0.99, 0.998]:
        res_u, res_c, res_o = [], [], []
        rq_u, rq_c, rq_o = [], [], []
        rho_u, rho_c, rho_o = [], [], []
        duties = []
        for trial in iter_trials(records, coupled, x_ref, snr_db=0.0):
            e_u, _, _ = run_filter(RLS(lam=lam), trial.d, trial.x_ref, taps=TAPS)
            m_u = trial_metrics(trial.clean, e_u, trial.r_peaks)
            res_u.append(m_u['snr_out'] - trial.snr_in)
            rq_u.append(m_u['r_qrs'])
            rho_u.append(m_u['rho_r'])

            e_c, _, ph_c = run_filter(RLS(lam=lam, trigger='amplitude', theta=3.0, hold=8), trial.d, trial.x_ref, taps=TAPS)
            m_c = trial_metrics(trial.clean, e_c, trial.r_peaks)
            res_c.append(m_c['snr_out'] - trial.snr_in)
            rq_c.append(m_c['r_qrs'])
            rho_c.append(m_c['rho_r'])
            duties.append(float(np.mean(ph_c[EVAL_SLICE] == CLAMPED) * 100.0))

            mask_o = oracle_mask(trial.r_peaks, len(trial.d), fs=FS)
            e_o, _, _ = run_filter(RLS(lam=lam, mask=mask_o), trial.d, trial.x_ref, taps=TAPS)
            m_o = trial_metrics(trial.clean, e_o, trial.r_peaks)
            res_o.append(m_o['snr_out'] - trial.snr_in)
            rq_o.append(m_o['r_qrs'])
            rho_o.append(m_o['rho_r'])

        boost = float(np.mean(res_c) - np.mean(res_u))
        oracle_boost = float(np.mean(res_o) - np.mean(res_u))
        out[f'{lam:.3f}'] = {
            'lambda': lam,
            'ungated': {
                'delta_snr_mean': float(np.mean(res_u)),
                'delta_snr_std': float(np.std(res_u)),
                'r_qrs_mean': float(np.mean(rq_u)),
                'r_qrs_std': float(np.std(rq_u)),
                'rho_r_mean': float(np.mean(rho_u)),
                'rho_r_std': float(np.std(rho_u)),
            },
            'clamped': {
                'delta_snr_mean': float(np.mean(res_c)),
                'delta_snr_std': float(np.std(res_c)),
                'r_qrs_mean': float(np.mean(rq_c)),
                'r_qrs_std': float(np.std(rq_c)),
                'rho_r_mean': float(np.mean(rho_c)),
                'rho_r_std': float(np.std(rho_c)),
                'clamp_duty_pct': float(np.mean(duties)),
            },
            'oracle': {
                'delta_snr_mean': float(np.mean(res_o)),
                'delta_snr_std': float(np.std(res_o)),
                'r_qrs_mean': float(np.mean(rq_o)),
                'r_qrs_std': float(np.std(rq_o)),
                'rho_r_mean': float(np.mean(rho_o)),
                'rho_r_std': float(np.std(rho_o)),
            },
            'boost_snr_mean': boost,
            'oracle_boost_mean': oracle_boost,
            'oracle_capture_pct': boost / oracle_boost * 100.0 if oracle_boost > 0 else 100.0,
        }
    return out


def evaluate_multi_regime_clamped_rls(records, coupled, x_ref):
    out = {}
    lam = 0.998
    for snr_in in SNR_LEVELS:
        res_u, res_c, res_o = [], [], []
        rq_u, rq_c, rq_o = [], [], []
        rho_u, rho_c, rho_o = [], [], []
        for trial in iter_trials(records, coupled, x_ref, snr_db=snr_in):
            e_u, _, _ = run_filter(RLS(lam=lam), trial.d, trial.x_ref, taps=TAPS)
            m_u = trial_metrics(trial.clean, e_u, trial.r_peaks)
            res_u.append(m_u['snr_out'] - trial.snr_in)
            rq_u.append(m_u['r_qrs'])
            rho_u.append(m_u['rho_r'])

            e_c, _, _ = run_filter(RLS(lam=lam, trigger='amplitude', theta=3.0, hold=8), trial.d, trial.x_ref, taps=TAPS)
            m_c = trial_metrics(trial.clean, e_c, trial.r_peaks)
            res_c.append(m_c['snr_out'] - trial.snr_in)
            rq_c.append(m_c['r_qrs'])
            rho_c.append(m_c['rho_r'])

            mask_o = oracle_mask(trial.r_peaks, len(trial.d), fs=FS)
            e_o, _, _ = run_filter(RLS(lam=lam, mask=mask_o), trial.d, trial.x_ref, taps=TAPS)
            m_o = trial_metrics(trial.clean, e_o, trial.r_peaks)
            res_o.append(m_o['snr_out'] - trial.snr_in)
            rq_o.append(m_o['r_qrs'])
            rho_o.append(m_o['rho_r'])

        out[str(snr_in)] = {
            'snr_in': snr_in,
            'ungated': {
                'delta_snr_mean': float(np.mean(res_u)),
                'delta_snr_std': float(np.std(res_u)),
                'r_qrs_mean': float(np.mean(rq_u)),
                'rho_r_mean': float(np.mean(rho_u)),
            },
            'clamped': {
                'delta_snr_mean': float(np.mean(res_c)),
                'delta_snr_std': float(np.std(res_c)),
                'r_qrs_mean': float(np.mean(rq_c)),
                'rho_r_mean': float(np.mean(rho_c)),
            },
            'oracle': {
                'delta_snr_mean': float(np.mean(res_o)),
                'delta_snr_std': float(np.std(res_o)),
                'r_qrs_mean': float(np.mean(rq_o)),
                'rho_r_mean': float(np.mean(rho_o)),
            },
            'boost_snr_mean': float(np.mean(res_c) - np.mean(res_u)),
        }
    return out


def evaluate_reference_noise_sweep(records, coupled, x_ref):
    out = {}
    for extra_mult in [0.0, 0.5, 1.0, 2.0]:
        res_plain, res_clamp, res_rls = [], [], []
        rho_plain, rho_clamp, rho_rls = [], [], []
        rq_plain, rq_clamp, rq_rls = [], [], []
        for trial in iter_trials(records, coupled, x_ref, snr_db=0.0):
            if extra_mult > 0.0:
                rng = np.random.RandomState(3000 + trial.index + int(extra_mult * 100))
                corrupted_ref = trial.x_ref + extra_mult * np.std(trial.x_ref) * rng.randn(len(trial.x_ref))
            else:
                corrupted_ref = trial.x_ref

            e_p, _, _ = run_filter(FixedStepNLMS(mu=0.002), trial.d, corrupted_ref, taps=TAPS)
            m_p = trial_metrics(trial.clean, e_p, trial.r_peaks)
            res_plain.append(m_p['snr_out'] - trial.snr_in)
            rho_plain.append(m_p['rho_r'])
            rq_plain.append(m_p['r_qrs'])

            e_c, _, _ = run_filter(TriggerClampedNLMS(mu=0.002, trigger='amplitude', theta=3.0, hold=8), trial.d, corrupted_ref, taps=TAPS)
            m_c = trial_metrics(trial.clean, e_c, trial.r_peaks)
            res_clamp.append(m_c['snr_out'] - trial.snr_in)
            rho_clamp.append(m_c['rho_r'])
            rq_clamp.append(m_c['r_qrs'])

            e_r, _, _ = run_filter(RLS(lam=0.998), trial.d, corrupted_ref, taps=TAPS)
            m_r = trial_metrics(trial.clean, e_r, trial.r_peaks)
            res_rls.append(m_r['snr_out'] - trial.snr_in)
            rho_rls.append(m_r['rho_r'])
            rq_rls.append(m_r['r_qrs'])

        out[str(extra_mult)] = {
            'extra_noise_mult': extra_mult,
            'plain_nlms': {
                'delta_snr_mean': float(np.mean(res_plain)),
                'delta_snr_std': float(np.std(res_plain)),
                'rho_r_mean': float(np.mean(rho_plain)),
                'r_qrs_mean': float(np.mean(rq_plain)),
            },
            'clamped_nlms': {
                'delta_snr_mean': float(np.mean(res_clamp)),
                'delta_snr_std': float(np.std(res_clamp)),
                'rho_r_mean': float(np.mean(rho_clamp)),
                'r_qrs_mean': float(np.mean(rq_clamp)),
            },
            'rls': {
                'delta_snr_mean': float(np.mean(res_rls)),
                'delta_snr_std': float(np.std(res_rls)),
                'rho_r_mean': float(np.mean(rho_rls)),
                'r_qrs_mean': float(np.mean(rq_rls)),
            },
        }
    return out


def run_mu_sweep(records, coupled, x_ref):
    sweep_data = {}
    for mu in [0.001, 0.002, 0.005, 0.01, 0.02]:
        plain_snrs, clamped_snrs = [], []
        for trial in iter_trials(records, coupled, x_ref, snr_db=0.0):
            e_p, _, _ = run_filter(FixedStepNLMS(mu=mu), trial.d, trial.x_ref, taps=TAPS)
            m_p = trial_metrics(trial.clean, e_p, trial.r_peaks)
            plain_snrs.append(m_p['snr_out'] - trial.snr_in)

            e_c, _, _ = run_filter(TriggerClampedNLMS(mu=mu, trigger='amplitude', theta=3.0, hold=8), trial.d, trial.x_ref, taps=TAPS)
            m_c = trial_metrics(trial.clean, e_c, trial.r_peaks)
            clamped_snrs.append(m_c['snr_out'] - trial.snr_in)

        sweep_data[str(mu)] = {
            'mu': mu,
            'plain_delta_snr_mean': float(np.mean(plain_snrs)),
            'plain_delta_snr_std': float(np.std(plain_snrs)),
            'clamped_delta_snr_mean': float(np.mean(clamped_snrs)),
            'clamped_delta_snr_std': float(np.std(clamped_snrs)),
            'gain_delta_snr': float(np.mean(clamped_snrs) - np.mean(plain_snrs)),
        }
    write_json('mu_sweep_results.json', sweep_data)
    return sweep_data


def run_rls_lambda_sweep(records, coupled, x_ref):
    sweep_data = {}
    for snr_in in SNR_LEVELS:
        snr_key = str(snr_in)
        sweep_data[snr_key] = {}
        for lam in [0.990, 0.995, 0.998, 0.999]:
            snrs = []
            for trial in iter_trials(records, coupled, x_ref, snr_db=snr_in):
                e_r, _, _ = run_filter(RLS(lam=lam, delta=0.1), trial.d, trial.x_ref, taps=TAPS)
                m_r = trial_metrics(trial.clean, e_r, trial.r_peaks)
                snrs.append(float(m_r['snr_out'] - trial.snr_in))
            sweep_data[snr_key][f'{lam:.3f}'] = {
                'mean': float(np.mean(snrs)),
                'std': float(np.std(snrs)),
            }
    write_json('rls_lambda_sweep.json', sweep_data)
    return sweep_data


def run_trigger_evaluation(records, coupled):
    rec_stats = {r_id: {'tp': 0, 'total': 0} for r_id in RECORDS}
    total_clamp_samples = 0
    total_qrs_active_clamp = 0
    for trial in range(NUM_TRIALS):
        r_id = RECORDS[trial % len(RECORDS)]
        ecg_data, r_peaks = records[r_id]
        noise_start = trial * SEGMENT_LEN
        ecg_start = trial // len(RECORDS) * SEGMENT_LEN * 3 + 1000
        clean_seg = ecg_data[ecg_start:ecg_start + SEGMENT_LEN]
        art_seg = coupled[noise_start:noise_start + SEGMENT_LEN]
        d_contaminated = clean_seg + art_seg * artifact_scale(clean_seg, art_seg, 0.0)
        sub_r_peaks = [p - ecg_start for p in r_peaks if ecg_start + int(2.0 * FS) <= p < ecg_start + SEGMENT_LEN]

        trigger = ClampTrigger(mode='amplitude', theta=3.0, hold=8)
        clamped = np.zeros(SEGMENT_LEN, dtype=bool)
        for n in range(SEGMENT_LEN):
            _, is_c = trigger.step(d_contaminated[n])
            clamped[n] = is_c

        for p in sub_r_peaks:
            if np.any(clamped[max(0, p - 3):min(SEGMENT_LEN, p + 5)]):
                rec_stats[r_id]['tp'] += 1
            rec_stats[r_id]['total'] += 1

        qrs_zone = np.zeros(SEGMENT_LEN, dtype=bool)
        for p in sub_r_peaks:
            qrs_zone[max(0, p - 8):min(SEGMENT_LEN, p + 9)] = True
        clamped_ev = clamped[EVAL_SLICE]
        qrs_zone_ev = qrs_zone[EVAL_SLICE]
        total_clamp_samples += int(np.sum(clamped_ev))
        total_qrs_active_clamp += int(np.sum(clamped_ev & qrs_zone_ev))

    rec_recalls = {r_id: float(rec_stats[r_id]['tp'] / rec_stats[r_id]['total'] * 100.0) for r_id in RECORDS}
    tot_tp = sum(rec_stats[r]['tp'] for r in RECORDS)
    tot_all = sum(rec_stats[r]['total'] for r in RECORDS)
    min_r = min(rec_recalls, key=rec_recalls.get)
    max_r = max(rec_recalls, key=rec_recalls.get)
    res = {
        'per_record_recall': rec_recalls,
        'pooled_recall': float(tot_tp / tot_all * 100.0),
        'mean_recall': float(np.mean(list(rec_recalls.values()))),
        'min_recall': rec_recalls[min_r],
        'min_record': min_r,
        'max_recall': rec_recalls[max_r],
        'max_record': max_r,
        'sample_precision': float(total_qrs_active_clamp / total_clamp_samples * 100.0) if total_clamp_samples > 0 else 0.0,
    }
    write_json('trigger_detection_results.json', res)
    return res


def compute_blunting_distribution(records, coupled, x_ref):
    blunting_list = []
    for trial in iter_trials(records, coupled, x_ref, snr_db=0.0):
        sub_r_peaks = [p for p in trial.r_peaks if int(2.0 * FS) <= p < SEGMENT_LEN]
        e_fast, _, _ = run_filter(FixedStepNLMS(mu=0.02), trial.d, trial.x_ref, taps=TAPS)
        for p in sub_r_peaks:
            lo, hi = max(0, p - 3), min(len(trial.clean), p + 4)
            apex = lo + np.argmax(np.abs(trial.clean[lo:hi]))
            val_c = trial.clean[apex]
            val_f = e_fast[apex]
            if abs(val_c) > 0.1:
                blunting_list.append((val_c - val_f) / abs(val_c) * 100.0)

    clean_100, _ = load_ecg_record('100', fs=FS, max_seconds=60.0)
    art_100, ref_100 = load_motion_artifact(fs=FS, max_samples=SEGMENT_LEN)
    d_100 = clean_100 + art_100 * artifact_scale(clean_100, art_100, 0.0)
    e_fast_100, _, _ = run_filter(FixedStepNLMS(mu=0.02), d_100, ref_100, taps=TAPS)
    p_beat = 4315
    z_s = p_beat - int(0.045 * FS)
    z_e = p_beat + int(0.045 * FS)
    clean_peak = float(np.max(clean_100[z_s:z_e]))
    fast_peak = float(np.max(e_fast_100[z_s:z_e]))
    fast_apex_diff_pct = float((fast_peak - clean_peak) / clean_peak * 100.0)

    blunting_stats = {
        'total_beats': len(blunting_list),
        'mean_blunting_pct': float(np.mean(blunting_list)),
        'median_blunting_pct': float(np.median(blunting_list)),
        'p90_blunting_pct': float(np.percentile(blunting_list, 90)),
        'p95_blunting_pct': float(np.percentile(blunting_list, 95)),
        'max_blunting_pct': float(np.max(blunting_list)),
        'fast_apex_blunting_pct': abs(fast_apex_diff_pct),
        'fast_apex_diff_pct': fast_apex_diff_pct,
    }
    write_json('blunting_results.json', blunting_stats)
    return blunting_stats


def run_full_benchmark():
    coupled, x_ref = load_motion_artifact(fs=FS)
    records = load_ecg_records(fs=FS)
    results_raw = {}
    results_summary = {}

    for snr_in in SNR_LEVELS:
        snr_key = str(snr_in)
        results_raw[snr_key] = {}
        results_summary[snr_key] = {}
        for key, name, macs, lat in ALGORITHMS:
            results_raw[snr_key][key] = {
                'name': name,
                'macs': macs,
                'latency_ms': lat,
                'delta_snr': [],
                'prd': [],
                'rho_r': [],
                'mae_r': [],
                'r_qrs': [],
                'delta_snr_iso': [],
                'duty_p1': [],
                'duty_p2': [],
                'duty_p3': [],
            }
        for trial in iter_trials(records, coupled, x_ref, snr_db=snr_in):
            rng_imu = np.random.RandomState(1000 + trial.index)
            x_imu = np.abs(trial.x_ref) + 0.02 * rng_imu.randn(len(trial.x_ref))
            for key, name, macs, lat in ALGORITHMS:
                if key == 'raw':
                    e_out = trial.d.copy()
                    phase_hist = np.zeros(len(trial.d), dtype=int)
                else:
                    rule = build_rule(key, trial, x_imu, fs=FS)
                    e_out, _, phase_hist = run_filter(rule, trial.d, trial.x_ref, taps=TAPS)
                m = trial_metrics(trial.clean, e_out, trial.r_peaks)
                snr_iso = compute_isoelectric_snr(trial.clean, e_out, trial.r_peaks, EVAL_SLICE, m['snr_out'])
                res = results_raw[snr_key][key]
                res['delta_snr'].append(float(m['snr_out'] - trial.snr_in))
                res['prd'].append(m['prd'])
                res['rho_r'].append(m['rho_r'])
                res['mae_r'].append(m['mae_r'])
                res['r_qrs'].append(m['r_qrs'])
                res['delta_snr_iso'].append(float(snr_iso - trial.snr_in))
                res['duty_p1'].append(float(np.mean(phase_hist[EVAL_SLICE] == CLAMPED) * 100.0))
                res['duty_p2'].append(float(np.mean(phase_hist[EVAL_SLICE] == ADAPTING) * 100.0))
                res['duty_p3'].append(float(np.mean(phase_hist[EVAL_SLICE] == STILL) * 100.0))

        for key, name, macs, lat in ALGORITHMS:
            res = results_raw[snr_key][key]
            results_summary[snr_key][key] = {
                'name': name,
                'macs': macs,
                'latency_ms': lat,
                'delta_snr_mean': float(np.mean(res['delta_snr'])),
                'delta_snr_std': float(np.std(res['delta_snr'])),
                'prd_mean': float(np.mean(res['prd'])),
                'prd_std': float(np.std(res['prd'])),
                'rho_r_mean': float(np.mean(res['rho_r'])),
                'rho_r_std': float(np.std(res['rho_r'])),
                'mae_r_mean': float(np.mean(res['mae_r'])),
                'mae_r_std': float(np.std(res['mae_r'])),
                'r_qrs_mean': float(np.mean(res['r_qrs'])),
                'r_qrs_std': float(np.std(res['r_qrs'])),
                'delta_snr_iso_mean': float(np.mean(res['delta_snr_iso'])),
                'duty_p1_mean': float(np.mean(res['duty_p1'])),
                'duty_p2_mean': float(np.mean(res['duty_p2'])),
                'duty_p3_mean': float(np.mean(res['duty_p3'])),
            }
        results_summary[snr_key]['significance_clamped_amp'] = evaluate_significance(results_raw, snr_key)

    results_summary['rls_generalization'] = evaluate_rls_generalization(records, coupled, x_ref)
    results_summary['rls_multi_regime'] = evaluate_multi_regime_clamped_rls(records, coupled, x_ref)
    results_summary['reference_noise_sweep'] = evaluate_reference_noise_sweep(records, coupled, x_ref)

    write_json('benchmark_results.json', results_summary)
    write_json('benchmark_raw_trials.json', results_raw)
    run_mu_sweep(records, coupled, x_ref)
    run_rls_lambda_sweep(records, coupled, x_ref)
    run_trigger_evaluation(records, coupled)
    compute_blunting_distribution(records, coupled, x_ref)
    return results_summary, results_raw


if __name__ == '__main__':
    run_full_benchmark()
