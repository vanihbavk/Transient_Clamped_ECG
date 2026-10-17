import os
import sys

import numpy as np
import scipy.signal as signal
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.common import (
    BODY_COUPLING,
    EVAL_SLICE,
    FS,
    RECORDS,
    SEGMENT_LEN,
    TAPS,
    artifact_scale,
    input_snr,
    load_ecg_records,
    load_motion_artifact,
    load_muscle_artifact,
    trial_metrics,
    write_json,
)
from dsp.adaptive_filters import (
    CLAMPED,
    ClampTrigger,
    FixedStepNLMS,
    ImuGatedNLMS,
    KwongJohnstonVSS,
    MaskClampedNLMS,
    RLS,
    TriggerClampedNLMS,
    run_filter,
)


def run_error_amplitude_gate(d, x_ref, taps=TAPS, mu=0.04, mu_min=1e-4, eps=1e-4):
    n_samples = len(d)
    x_buf = np.zeros(taps)
    w = np.zeros(taps)
    e = np.zeros(n_samples)
    sigma_e = 0.5
    hold_cnt = 0
    for n in range(n_samples):
        x_buf[1:] = x_buf[:-1]
        x_buf[0] = x_ref[n]
        norm_x = np.dot(x_buf, x_buf) + eps
        e_n = d[n] - np.dot(w, x_buf)
        e[n] = e_n
        sigma_e = 0.95 * sigma_e + 0.05 * abs(e_n)
        if abs(e_n) >= 3.0 * sigma_e:
            hold_cnt = 8
        mu_n = mu_min if hold_cnt > 0 else mu
        if hold_cnt > 0:
            hold_cnt -= 1
        w += mu_n / norm_x * e_n * x_buf
    return e


def run_dynamic_plant_experiments(records, x_ref_all):
    step_samples = int(10.0 * FS)
    methods = [
        'nlms_def',
        'nlms_tuned_002',
        'nlms_clamped_002',
        'nlms_tuned_005',
        'nlms_clamped_amp',
        'kwong',
        'amplitude_gate',
        'random_clamp',
        'imu_curv',
        'imu_fd',
        'imu_amp',
        'plain_rls',
        'clamped_rls',
        'clamped_rls_099',
    ]
    results = {m: {'delta_snr': [], 'rho_r': [], 'r_qrs': [], 'duty': []} for m in methods}
    grid_mus = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    grid_scores = {str(mu): [] for mu in grid_mus}

    for r_idx, rec_id in enumerate(RECORDS):
        clean_full, r_peaks_full = records[rec_id]
        for trial_sub in range(2):
            ecg_start = (trial_sub * 2 + 1) * SEGMENT_LEN
            noise_start = (r_idx * 2 + trial_sub) * SEGMENT_LEN
            clean = clean_full[ecg_start:ecg_start + SEGMENT_LEN]
            x_ref = x_ref_all[noise_start:noise_start + SEGMENT_LEN]

            art_tv = np.zeros(SEGMENT_LEN)
            n_chunks = SEGMENT_LEN // step_samples
            for chunk in range(n_chunks):
                gain = 1.0 if chunk % 2 == 0 else 0.3
                idx = slice(chunk * step_samples, (chunk + 1) * step_samples)
                art_tv[idx] = gain * signal.lfilter(BODY_COUPLING, 1.0, x_ref[idx])
            art_tv *= artifact_scale(clean, art_tv, 0.0)
            d = clean + art_tv
            sub_r_peaks = [p - ecg_start for p in r_peaks_full if ecg_start <= p < ecg_start + SEGMENT_LEN]
            actual_snr_in = input_snr(clean, art_tv)

            def record_method(name, est, phase=None):
                m = trial_metrics(clean, est, sub_r_peaks)
                results[name]['delta_snr'].append(m['snr_out'] - actual_snr_in)
                results[name]['rho_r'].append(m['rho_r'])
                results[name]['r_qrs'].append(m['r_qrs'])
                if phase is not None:
                    results[name]['duty'].append(float(np.mean(phase == CLAMPED) * 100.0))

            e_def, _, _ = run_filter(FixedStepNLMS(mu=0.02), d, x_ref, taps=TAPS)
            record_method('nlms_def', e_def)

            for g_mu in grid_mus:
                e_g, _, _ = run_filter(FixedStepNLMS(mu=g_mu), d, x_ref, taps=TAPS)
                m_g = trial_metrics(clean, e_g, sub_r_peaks)
                grid_scores[str(g_mu)].append(m_g['snr_out'] - actual_snr_in)

            e_002, _, _ = run_filter(FixedStepNLMS(mu=0.002), d, x_ref, taps=TAPS)
            record_method('nlms_tuned_002', e_002)

            e_nc002, _, _ = run_filter(TriggerClampedNLMS(mu=0.002, trigger='amplitude'), d, x_ref, taps=TAPS)
            record_method('nlms_clamped_002', e_nc002)

            e_005, _, _ = run_filter(FixedStepNLMS(mu=0.005), d, x_ref, taps=TAPS)
            record_method('nlms_tuned_005', e_005)

            e_nc005, _, _ = run_filter(TriggerClampedNLMS(mu=0.005, trigger='amplitude'), d, x_ref, taps=TAPS)
            record_method('nlms_clamped_amp', e_nc005)

            e_kw, _, _ = run_filter(KwongJohnstonVSS(), d, x_ref, taps=TAPS)
            record_method('kwong', e_kw)

            e_amp = run_error_amplitude_gate(d, x_ref, taps=TAPS)
            record_method('amplitude_gate', e_amp)

            rng_placebo = np.random.RandomState(42 + r_idx * 10 + trial_sub)
            rand_mask = rng_placebo.rand(SEGMENT_LEN) < 0.18
            e_rnd, _, _ = run_filter(MaskClampedNLMS(rand_mask, mu=0.04, mu_min=1e-4), d, x_ref, taps=TAPS)
            record_method('random_clamp', e_rnd)

            rng_imu = np.random.RandomState(1000 + r_idx * 2 + trial_sub)
            x_imu = np.abs(x_ref) + 0.02 * rng_imu.randn(len(x_ref))

            e_ic, _, ph_ic = run_filter(ImuGatedNLMS(x_imu, mu0=0.005, trigger='curvature'), d, x_ref, taps=TAPS)
            record_method('imu_curv', e_ic, ph_ic)

            e_ifd, _, ph_ifd = run_filter(ImuGatedNLMS(x_imu, mu0=0.005, trigger='first_difference'), d, x_ref, taps=TAPS)
            record_method('imu_fd', e_ifd, ph_ifd)

            e_ia, _, ph_ia = run_filter(ImuGatedNLMS(x_imu, mu0=0.005, trigger='amplitude'), d, x_ref, taps=TAPS)
            record_method('imu_amp', e_ia, ph_ia)

            e_rls, _, _ = run_filter(RLS(lam=0.998, delta=0.1), d, x_ref, taps=TAPS)
            record_method('plain_rls', e_rls)

            e_crls, _, _ = run_filter(RLS(lam=0.985, delta=0.1, trigger='amplitude', theta=3.0, hold=8), d, x_ref, taps=TAPS)
            record_method('clamped_rls', e_crls)

            e_crls99, _, _ = run_filter(RLS(lam=0.99, delta=0.1, trigger='amplitude', theta=3.0, hold=8), d, x_ref, taps=TAPS)
            record_method('clamped_rls_099', e_crls99)

    summary = {}
    for m in methods:
        summary[m] = {
            'delta_snr_mean': float(np.mean(results[m]['delta_snr'])),
            'delta_snr_std': float(np.std(results[m]['delta_snr'])),
            'rho_r_mean': float(np.mean(results[m]['rho_r'])),
            'rho_r_std': float(np.std(results[m]['rho_r'])),
            'r_qrs_mean': float(np.mean(results[m]['r_qrs'])),
            'duty_mean': float(np.mean(results[m]['duty'])) if results[m]['duty'] else 0.0,
        }

    pairs = [
        ('wilcoxon_curv_vs_nlms', 'imu_curv', 'nlms_tuned_005'),
        ('wilcoxon_amp_vs_nlms', 'imu_amp', 'nlms_tuned_005'),
        ('wilcoxon_amp_vs_clamped', 'imu_amp', 'nlms_clamped_amp'),
        ('wilcoxon_clamp_vs_tuned_nlms', 'nlms_clamped_amp', 'nlms_tuned_005'),
        ('wilcoxon_clamp_vs_tuned_002', 'nlms_clamped_002', 'nlms_tuned_002'),
        ('wilcoxon_clamp_vs_def_nlms', 'nlms_clamped_amp', 'nlms_def'),
    ]
    for label, k1, k2 in pairs:
        stat, p_val = wilcoxon(results[k1]['delta_snr'], results[k2]['delta_snr'])
        summary[label] = {'stat': float(stat), 'p_value': float(p_val)}

    grid_results = {mu_str: round(float(np.mean(scores)), 2) for mu_str, scores in grid_scores.items()}
    best_mu = max(grid_results, key=grid_results.get)
    summary['nlms_grid_sweep'] = {
        'grid': grid_results,
        'best_mu': float(best_mu),
        'best_delta_snr': grid_results[best_mu],
    }
    return summary


def synthesize_stress_scenario(s_id, trial, clean, motion, ma_all):
    noise_start = trial * SEGMENT_LEN
    rng = np.random.RandomState(42 + trial * 7)
    if s_id == 'S0':
        c = signal.lfilter(BODY_COUPLING, 1.0, motion)
        d = clean + c * artifact_scale(clean, c, 0.0)
        x_ref = motion + 0.08 * rng.randn(SEGMENT_LEN)
        x_imu = np.abs(x_ref) + 0.02 * rng.randn(SEGMENT_LEN)
        return d, x_ref, x_imu
    if s_id == 'S1':
        drift = 1.0 + 0.2 * np.sin(2.0 * np.pi * 0.05 * np.arange(SEGMENT_LEN) / FS)
        c_lin = signal.lfilter(BODY_COUPLING, 1.0, motion)
        c = (c_lin + 0.15 * (c_lin ** 2 / (np.std(c_lin) + 1e-12))) * drift
        d = clean + c * artifact_scale(clean, c, 0.0)
        x_ref = motion + 0.08 * rng.randn(SEGMENT_LEN)
        x_imu = np.abs(x_ref) + 0.02 * rng.randn(SEGMENT_LEN)
        return d, x_ref, x_imu
    if s_id == 'S2':
        c_corr = signal.lfilter(BODY_COUPLING, 1.0, motion)
        u_raw = np.random.RandomState(5000 + trial).randn(SEGMENT_LEN)
        b_lp, a_lp = signal.butter(3, 8.0 / (0.5 * FS), btype='low')
        u_filt = signal.filtfilt(b_lp, a_lp, u_raw)
        u_filt /= np.std(u_filt) + 1e-12
        c_uncorr = signal.lfilter(BODY_COUPLING, 1.0, u_filt)
        c = np.sqrt(0.5) * c_corr + np.sqrt(0.5) * c_uncorr
        d = clean + c * artifact_scale(clean, c, 0.0)
        x_ref = motion + 0.08 * rng.randn(SEGMENT_LEN)
        x_imu = np.abs(x_ref) + 0.02 * rng.randn(SEGMENT_LEN)
        return d, x_ref, x_imu
    if s_id == 'S3':
        drift = 1.0 + 0.2 * np.sin(2.0 * np.pi * 0.05 * np.arange(SEGMENT_LEN) / FS)
        c_lin = signal.lfilter(BODY_COUPLING, 1.0, motion)
        c_corr = (c_lin + 0.15 * (c_lin ** 2 / (np.std(c_lin) + 1e-12))) * drift
        u_raw = np.random.RandomState(6000 + trial).randn(SEGMENT_LEN)
        b_lp, a_lp = signal.butter(3, 8.0 / (0.5 * FS), btype='low')
        u_filt = signal.filtfilt(b_lp, a_lp, u_raw)
        u_filt /= np.std(u_filt) + 1e-12
        c_uncorr = signal.lfilter(BODY_COUPLING, 1.0, u_filt)
        c = np.sqrt(0.5) * c_corr + np.sqrt(0.5) * c_uncorr
        d = clean + c * artifact_scale(clean, c, 0.0)
        x_ref = motion + 0.3 * rng.randn(SEGMENT_LEN)
        x_imu = np.abs(x_ref) + 0.05 * rng.randn(SEGMENT_LEN)
        return d, x_ref, x_imu
    if s_id in ('S4', 'S5'):
        c = signal.lfilter(BODY_COUPLING, 1.0, motion)
        d_s0 = clean + c * artifact_scale(clean, c, 0.0)
        ma_seg = ma_all[noise_start:noise_start + SEGMENT_LEN]
        snr_ma = 6.0 if s_id == 'S4' else 0.0
        d = d_s0 + ma_seg * artifact_scale(clean, ma_seg, snr_ma)
        x_ref = motion + 0.08 * rng.randn(SEGMENT_LEN)
        x_imu = np.abs(x_ref) + 0.02 * rng.randn(SEGMENT_LEN)
        return d, x_ref, x_imu
    raise ValueError(f'unknown scenario: {s_id}')


def run_stress_scenarios(records, x_ref_all, ma_all):
    descriptions = {
        'S0': 'Stationary S0 (0 dB input SNR)',
        'S1': 'S1: Nonlinear coupling + tap drift',
        'S2': 'S2: 50% artifact power unseen by reference',
        'S3': 'S3: Harsh (S1 + S2 + Noisy sensor 0.3)',
        'S4': 'S4: Unreferenced EMG (NSTDB ma at 6 dB)',
        'S5': 'S5: Severe unreferenced EMG (NSTDB ma at 0 dB)',
    }
    stress_results = {}
    for s_id, desc in descriptions.items():
        s_data = {
            'nlms': [],
            'nlms_tuned': [],
            'nlms_clamped': [],
            'kwong': [],
            'rls': [],
            'imu_only': [],
            'curv': [],
            'amp': [],
            'diff': [],
            'curv_duty': [],
            'amp_duty': [],
            'diff_duty': [],
        }
        for trial in range(14):
            rec_id = RECORDS[trial // 2]
            clean_full, r_peaks_full = records[rec_id]
            clean_start = 1000 + (trial % 2) * SEGMENT_LEN
            noise_start = trial * SEGMENT_LEN
            clean = clean_full[clean_start:clean_start + SEGMENT_LEN]
            motion = x_ref_all[noise_start:noise_start + SEGMENT_LEN]
            sub_r_peaks = [p - clean_start for p in r_peaks_full if clean_start <= p < clean_start + SEGMENT_LEN]
            d, x_ref, x_imu = synthesize_stress_scenario(s_id, trial, clean, motion, ma_all)
            actual_snr_in = input_snr(clean, d - clean)

            def eval_snr(rule):
                e_out, _, ph = run_filter(rule, d, x_ref, taps=TAPS)
                m = trial_metrics(clean, e_out, sub_r_peaks)
                return m['snr_out'] - actual_snr_in, ph

            snr_nlms, _ = eval_snr(FixedStepNLMS(mu=0.02))
            s_data['nlms'].append(snr_nlms)

            snr_tuned, _ = eval_snr(FixedStepNLMS(mu=0.005))
            s_data['nlms_tuned'].append(snr_tuned)

            snr_clamp, _ = eval_snr(TriggerClampedNLMS(mu=0.005, trigger='amplitude'))
            s_data['nlms_clamped'].append(snr_clamp)

            snr_kw, _ = eval_snr(KwongJohnstonVSS())
            s_data['kwong'].append(snr_kw)

            snr_rls, _ = eval_snr(RLS(lam=0.998, delta=0.1))
            s_data['rls'].append(snr_rls)

            snr_imu, _ = eval_snr(ImuGatedNLMS(x_imu, mu0=0.002, trigger=None))
            s_data['imu_only'].append(snr_imu)

            snr_curv, ph_c = eval_snr(ImuGatedNLMS(x_imu, mu0=0.002, trigger='curvature', theta=0.38))
            s_data['curv'].append(snr_curv)
            s_data['curv_duty'].append(float(np.mean(ph_c == CLAMPED) * 100.0))

            snr_amp, ph_a = eval_snr(ImuGatedNLMS(x_imu, mu0=0.002, trigger='amplitude', theta=3.0))
            s_data['amp'].append(snr_amp)
            s_data['amp_duty'].append(float(np.mean(ph_a == CLAMPED) * 100.0))

            snr_diff, ph_d = eval_snr(ImuGatedNLMS(x_imu, mu0=0.002, trigger='first_difference', theta=0.7))
            s_data['diff'].append(snr_diff)
            s_data['diff_duty'].append(float(np.mean(ph_d == CLAMPED) * 100.0))

        entry = {'description': desc}
        for k, vals in s_data.items():
            entry[k] = float(np.mean(vals))
        stat_nlms, p_nlms = wilcoxon(s_data['nlms_clamped'], s_data['nlms_tuned'])
        stat_rls, p_rls = wilcoxon(s_data['nlms_clamped'], s_data['rls'])
        entry['wilcoxon_clamp_vs_tuned'] = {'stat': float(stat_nlms), 'p_value': float(p_nlms)}
        entry['wilcoxon_clamp_vs_rls'] = {'stat': float(stat_rls), 'p_value': float(p_rls)}
        if s_id == 'S5':
            _, p_val = wilcoxon(s_data['amp'], s_data['curv'])
            entry['p_val_amp_vs_curv'] = float(p_val)
        stress_results[s_id] = entry
    return stress_results


def run_gate_statistics_evaluation(records, coupled_all):
    total_qrs = 0
    detected_qrs = 0
    total_clamped_samples = 0
    total_samples = 0
    true_positive_clamp_samples = 0
    trigger = ClampTrigger(mode='curvature', theta=0.38, hold=8)

    for rec_id in RECORDS:
        clean_full, r_peaks_full = records[rec_id]
        clean = clean_full[:SEGMENT_LEN]
        coupled = coupled_all[:SEGMENT_LEN]
        d = clean + coupled * artifact_scale(clean, coupled, 0.0)
        sub_r_peaks = [p for p in r_peaks_full if 0 <= p < SEGMENT_LEN]

        qrs_mask = np.zeros(SEGMENT_LEN, dtype=bool)
        for p in sub_r_peaks:
            qrs_mask[max(0, p - 10):min(SEGMENT_LEN, p + 11)] = True

        clamp_mask = np.zeros(SEGMENT_LEN, dtype=bool)
        trigger.reset()
        for n in range(SEGMENT_LEN):
            _, is_clamp = trigger.step(d[n])
            clamp_mask[n] = is_clamp

        total_samples += SEGMENT_LEN
        total_clamped_samples += int(np.sum(clamp_mask))
        true_positive_clamp_samples += int(np.sum(clamp_mask & qrs_mask))
        for p in sub_r_peaks:
            total_qrs += 1
            if np.any(clamp_mask[max(0, p - 10):min(SEGMENT_LEN, p + 11)]):
                detected_qrs += 1

    return {
        'duty_cycle_pct': round(total_clamped_samples / total_samples * 100.0, 1),
        'beat_recall_pct': round(detected_qrs / max(1, total_qrs) * 100.0, 1),
        'precision_pct': round(true_positive_clamp_samples / max(1, total_clamped_samples) * 100.0, 1),
        'total_beats': total_qrs,
        'detected_beats': detected_qrs,
    }


def run_biopotential_gated_noise_sweep(records, coupled_all, x_ref_all):
    sweep_results = {}
    for sigma in [0.08, 0.25, 0.5, 1.0]:
        plain_snrs, clamped_e_snrs, clamped_d_snrs = [], [], []
        duty_e_list, duty_d_list = [], []
        for r_idx, rec_id in enumerate(RECORDS):
            clean_full, r_peaks_full = records[rec_id]
            clean = clean_full[:SEGMENT_LEN]
            x_ref = x_ref_all[:SEGMENT_LEN]
            coupled = coupled_all[:SEGMENT_LEN]
            scaled_coupled = coupled * artifact_scale(clean, coupled, 0.0)
            d = clean + scaled_coupled
            rng = np.random.RandomState(42 + int(sigma * 100) + r_idx)
            x_noisy = x_ref + sigma * rng.randn(len(x_ref))
            sub_peaks = [p for p in r_peaks_full if 0 <= p < SEGMENT_LEN]
            actual_snr_in = input_snr(clean, scaled_coupled)

            e_plain, _, _ = run_filter(RLS(lam=0.998, delta=0.1), d, x_noisy, taps=TAPS)
            m_p = trial_metrics(clean, e_plain, sub_peaks)
            plain_snrs.append(m_p['snr_out'] - actual_snr_in)

            P = np.eye(TAPS) / 0.1
            w = np.zeros(TAPS)
            x_buf = np.zeros(TAPS)
            e_ge = np.zeros(SEGMENT_LEN)
            trig_e = ClampTrigger(mode='amplitude', theta=3.0, hold=8)
            clamps_e = 0
            for n in range(SEGMENT_LEN):
                x_buf[1:] = x_buf[:-1]
                x_buf[0] = x_noisy[n]
                e_n = d[n] - np.dot(w, x_buf)
                e_ge[n] = e_n
                _, is_clamp = trig_e.step(e_n)
                if is_clamp:
                    clamps_e += 1
                else:
                    Px = np.dot(P, x_buf)
                    k = Px / (0.998 + np.dot(x_buf, Px))
                    w += k * e_n
                    P = (P - np.outer(k, np.dot(x_buf, P))) / 0.998
            m_ge = trial_metrics(clean, e_ge, sub_peaks)
            clamped_e_snrs.append(m_ge['snr_out'] - actual_snr_in)
            duty_e_list.append(clamps_e / SEGMENT_LEN * 100.0)

            e_gd, _, ph_d = run_filter(RLS(lam=0.998, delta=0.1, trigger='amplitude', theta=3.0, hold=8), d, x_noisy, taps=TAPS)
            m_gd = trial_metrics(clean, e_gd, sub_peaks)
            clamped_d_snrs.append(m_gd['snr_out'] - actual_snr_in)
            duty_d_list.append(float(np.mean(ph_d == CLAMPED) * 100.0))

        sweep_results[str(sigma)] = {
            'plain_rls_mean': float(np.mean(plain_snrs)),
            'clamped_e_mean': float(np.mean(clamped_e_snrs)),
            'clamped_d_mean': float(np.mean(clamped_d_snrs)),
            'duty_e_mean': float(np.mean(duty_e_list)),
            'duty_d_mean': float(np.mean(duty_d_list)),
        }
    return sweep_results


def main():
    records = load_ecg_records(fs=FS)
    coupled_all, x_ref_all = load_motion_artifact(fs=FS)
    ma_all = load_muscle_artifact(fs=FS)
    combined = {
        'dynamic_plant_benchmarks': run_dynamic_plant_experiments(records, x_ref_all),
        'stress_tests': run_stress_scenarios(records, x_ref_all, ma_all),
        'gate_statistics': run_gate_statistics_evaluation(records, coupled_all),
        'biopotential_gated_noise_sweep': run_biopotential_gated_noise_sweep(records, coupled_all, x_ref_all),
    }
    write_json('dynamic_benchmark_results.json', combined)


if __name__ == '__main__':
    main()
