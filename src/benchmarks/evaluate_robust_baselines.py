import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.common import (
    FS,
    TAPS,
    iter_trials,
    load_ecg_records,
    load_motion_artifact,
    trial_metrics,
    write_json,
)
from dsp.adaptive_filters import (
    AboulnasrMayyasVSS,
    FixedStepNLMS,
    HuberNLMS,
    KwongJohnstonVSS,
    RLS,
    SetMembershipNLMS,
    TriggerClampedNLMS,
    run_filter,
)

PRIMARY_ALGORITHMS = [
    ('nlms_clamped_amp', 'Proposed NLMS+Amp', lambda: TriggerClampedNLMS(mu=0.002, trigger='amplitude', theta=3.0, hold=8)),
    ('nlms_stat_tuned', 'Plain NLMS (mu=0.002)', lambda: FixedStepNLMS(mu=0.002)),
    ('nlms_clipped_error', 'Clipped-Error (Huber) NLMS', lambda: HuberNLMS(mu=0.002, delta=0.5)),
    ('sm_nlms', 'Set-Membership NLMS', lambda: SetMembershipNLMS(gamma=0.2)),
    ('vss_aboulnasr', 'Aboulnasr-Mayyas VSS', lambda: AboulnasrMayyasVSS(mu_init=0.02, alpha=0.97, beta=0.95, gamma=0.001)),
    ('vss_kwong', 'Kwong-Johnston VSS', lambda: KwongJohnstonVSS(mu_init=0.02, alpha=0.97, gamma=0.001)),
    ('rls', 'Tuned Plain RLS', lambda: RLS(lam=0.998)),
]
SM_GAMMA_GRID = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
HUBER_DELTA_GRID = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
GAIN_FACTORS = [0.5, 1.0, 2.0]


def run_evaluation():
    coupled, x_ref = load_motion_artifact(fs=FS)
    records = load_ecg_records(fs=FS)
    trials = list(iter_trials(records, coupled, x_ref, snr_db=0.0))

    primary_results = {
        k: {'delta_snr': [], 'rho_r': [], 'prd': [], 'mae_r': [], 'r_qrs': []}
        for k, _, _ in PRIMARY_ALGORITHMS
    }

    for trial in trials:
        for key, _, make_rule in PRIMARY_ALGORITHMS:
            e_out, _, _ = run_filter(make_rule(), trial.d, trial.x_ref, taps=TAPS)
            m = trial_metrics(trial.clean, e_out, trial.r_peaks)
            primary_results[key]['delta_snr'].append(float(m['snr_out'] - trial.snr_in))
            primary_results[key]['rho_r'].append(m['rho_r'])
            primary_results[key]['prd'].append(m['prd'])
            primary_results[key]['mae_r'].append(m['mae_r'])
            primary_results[key]['r_qrs'].append(m['r_qrs'])

    summary = {}
    for key, name, _ in PRIMARY_ALGORITHMS:
        res = primary_results[key]
        summary[key] = {
            'name': name,
            'delta_snr_mean': float(np.mean(res['delta_snr'])),
            'delta_snr_std': float(np.std(res['delta_snr'])),
            'rho_r_mean': float(np.mean(res['rho_r'])),
            'rho_r_std': float(np.std(res['rho_r'])),
            'prd_mean': float(np.mean(res['prd'])),
            'prd_std': float(np.std(res['prd'])),
            'mae_r_mean': float(np.mean(res['mae_r'])),
            'mae_r_std': float(np.std(res['mae_r'])),
            'r_qrs_mean': float(np.mean(res['r_qrs'])),
            'r_qrs_std': float(np.std(res['r_qrs'])),
        }

    clamp_snr = primary_results['nlms_clamped_amp']['delta_snr']
    for key, _, _ in PRIMARY_ALGORITHMS:
        if key == 'nlms_clamped_amp':
            continue
        stat, p_val = wilcoxon(clamp_snr, primary_results[key]['delta_snr'])
        summary[key]['wilcoxon_vs_clamp'] = {'stat': float(stat), 'p_value': float(p_val)}

    gain_summary = {}
    gain_rules = [
        ('plain_nlms', lambda: FixedStepNLMS(mu=0.002)),
        ('clamped_amp', lambda: TriggerClampedNLMS(mu=0.002, trigger='amplitude', theta=3.0, hold=8)),
        ('huber_fixed', lambda: HuberNLMS(mu=0.002, delta=0.5, adaptive=False)),
        ('huber_adaptive', lambda: HuberNLMS(mu=0.002, adaptive=True, k=1.4)),
    ]
    for g in GAIN_FACTORS:
        g_str = f'{g:.1f}'
        g_res = {m_key: {'delta_snr': [], 'rho_r': []} for m_key, _ in gain_rules}
        for trial in trials:
            clean_g = trial.clean * g
            d_g = clean_g + trial.artifact * g
            for m_key, make_rule in gain_rules:
                e_out, _, _ = run_filter(make_rule(), d_g, trial.x_ref, taps=TAPS)
                m = trial_metrics(clean_g, e_out, trial.r_peaks)
                g_res[m_key]['delta_snr'].append(float(m['snr_out'] - trial.snr_in))
                g_res[m_key]['rho_r'].append(m['rho_r'])

        gain_summary[g_str] = {}
        for m_key, _ in gain_rules:
            ds = g_res[m_key]['delta_snr']
            rr = g_res[m_key]['rho_r']
            gain_summary[g_str][m_key] = {
                'delta_snr_mean': float(np.mean(ds)),
                'delta_snr_std': float(np.std(ds)),
                'rho_r_mean': float(np.mean(rr)),
                'rho_r_std': float(np.std(rr)),
            }
        c_snr = g_res['clamped_amp']['delta_snr']
        for m_key in ['plain_nlms', 'huber_fixed', 'huber_adaptive']:
            st, pv = wilcoxon(c_snr, g_res[m_key]['delta_snr'])
            gain_summary[g_str][m_key]['wilcoxon_vs_clamp'] = {'stat': float(st), 'p_value': float(pv)}
    summary['gain_mismatch_experiment'] = gain_summary

    sm_sweep = {}
    for gamma in SM_GAMMA_GRID:
        snrs, rhos = [], []
        for trial in trials:
            e_sm, _, _ = run_filter(SetMembershipNLMS(gamma=gamma), trial.d, trial.x_ref, taps=TAPS)
            m = trial_metrics(trial.clean, e_sm, trial.r_peaks)
            snrs.append(float(m['snr_out'] - trial.snr_in))
            rhos.append(m['rho_r'])
        sm_sweep[f'{gamma:.2f}'] = {
            'delta_snr_mean': float(np.mean(snrs)),
            'delta_snr_std': float(np.std(snrs)),
            'rho_r_mean': float(np.mean(rhos)),
            'rho_r_std': float(np.std(rhos)),
        }
    best_sm = max(sm_sweep, key=lambda k: sm_sweep[k]['delta_snr_mean'])
    summary['sm_nlms_sweep'] = {
        'grid': sm_sweep,
        'best_gamma': float(best_sm),
        'best_delta_snr': sm_sweep[best_sm]['delta_snr_mean'],
        'best_rho_r': sm_sweep[best_sm]['rho_r_mean'],
    }

    huber_sweep = {}
    for delta in HUBER_DELTA_GRID:
        snrs, rhos = [], []
        for trial in trials:
            e_h, _, _ = run_filter(HuberNLMS(mu=0.002, delta=delta, adaptive=False), trial.d, trial.x_ref, taps=TAPS)
            m = trial_metrics(trial.clean, e_h, trial.r_peaks)
            snrs.append(float(m['snr_out'] - trial.snr_in))
            rhos.append(m['rho_r'])
        huber_sweep[f'{delta:.2f}'] = {
            'delta_snr_mean': float(np.mean(snrs)),
            'delta_snr_std': float(np.std(snrs)),
            'rho_r_mean': float(np.mean(rhos)),
            'rho_r_std': float(np.std(rhos)),
        }
    best_huber = max(huber_sweep, key=lambda k: huber_sweep[k]['delta_snr_mean'])
    summary['huber_delta_sweep'] = {
        'grid': huber_sweep,
        'best_delta': float(best_huber),
        'best_delta_snr': huber_sweep[best_huber]['delta_snr_mean'],
        'best_rho_r': huber_sweep[best_huber]['rho_r_mean'],
    }

    write_json('robust_baselines_results.json', summary)
    return summary


if __name__ == '__main__':
    run_evaluation()
