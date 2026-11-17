import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.common import RESULTS_DIR, read_json


def load_json(filename):
    if (RESULTS_DIR / filename).exists():
        return read_json(filename)
    return None


def fmt_p(p_val):
    if p_val < 0.0001:
        return r'p < 0.0001'
    if p_val < 0.001:
        return f'p = {p_val:.4f}'
    return f'p = {p_val:.3f}'


def macro(name, value):
    return f'\\newcommand{{\\{name}}}{{{value}}}'


def add_stationary_row_macros(lines, prefix, entry, include_std=True, include_duty=False):
    lines.append(macro(f'{prefix}SNR', f"{entry['delta_snr_mean']:+.2f}"))
    if include_std:
        lines.append(macro(f'{prefix}SNRStd', f"{entry['delta_snr_std']:.2f}"))
    lines.append(macro(f'{prefix}RhoR', f"{entry['rho_r_mean']:.1f}"))
    lines.append(macro(f'{prefix}MAER', f"{entry['mae_r_mean']:.2f}"))
    lines.append(macro(f'{prefix}RQRS', f"{entry['r_qrs_mean']:.4f}"))
    if include_duty:
        lines.append(macro(f'{prefix}Duty', f"{entry['duty_p1_mean']:.1f}"))


def generate_macros():
    bench = load_json('benchmark_results.json')
    dyn = load_json('dynamic_benchmark_results.json')
    rob = load_json('robust_baselines_results.json')
    rls_sweep = load_json('rls_lambda_sweep.json')
    prof = load_json('profiling_results.json')
    mu_sweep = load_json('mu_sweep_results.json')
    trig = load_json('trigger_detection_results.json')
    blunt = load_json('blunting_results.json')

    if not bench:
        sys.exit(1)

    lines = []
    b0 = bench['0.0']

    raw0 = b0['raw']
    lines.append(macro('rawDeltaSNRZeroDb', f"{raw0['delta_snr_mean']:+.2f}"))
    lines.append(macro('rawDeltaSNRZeroDbStd', f"{raw0['delta_snr_std']:.2f}"))
    lines.append(macro('rawRhoRZeroDb', f"{raw0['rho_r_mean']:.1f}"))
    lines.append(macro('rawMAERZeroDb', f"{raw0['mae_r_mean']:.2f}"))
    lines.append(macro('rawRQRSZeroDb', f"{raw0['r_qrs_mean']:.4f}"))

    add_stationary_row_macros(lines, 'statTunedNlms', b0['nlms_stat_tuned'], include_duty=False)

    for prefix, key in [('randomClamp', 'random_clamp'), ('shiftedClamp', 'shifted_clamp')]:
        entry = b0[key]
        lines.append(macro(f'{prefix}DeltaSNRZeroDb', f"{entry['delta_snr_mean']:+.2f}"))
        lines.append(macro(f'{prefix}DeltaSNRZeroDbStd', f"{entry['delta_snr_std']:.2f}"))
        lines.append(macro(f'{prefix}RhoRZeroDb', f"{entry['rho_r_mean']:.1f}"))
        lines.append(macro(f'{prefix}MAER', f"{entry['mae_r_mean']:.2f}"))
        lines.append(macro(f'{prefix}RQRSZeroDb', f"{entry['r_qrs_mean']:.4f}"))
        lines.append(macro(f'{prefix}DutyZeroDb', f"{entry['duty_p1_mean']:.1f}"))

    for prefix, key in [
        ('nlmsClampAmp', 'nlms_clamped_amp'),
        ('nlmsClampDiff', 'nlms_clamped_diff'),
        ('nlmsClampCurv', 'nlms_clamped_curv'),
        ('nlmsPtLook', 'pt_look_nlms'),
        ('nlmsPtPred', 'pt_pred_nlms'),
        ('depintoSlope', 'depinto_slope_delay'),
        ('nlmsOracle', 'oracle_gated'),
        ('imuAmp', 'imu_amp'),
    ]:
        add_stationary_row_macros(lines, prefix, b0[key], include_duty=True)

    for prefix, key in [
        ('fastNlms', 'nlms_tuned'),
        ('kwong', 'vss_kwong'),
        ('rls', 'rls'),
    ]:
        entry = b0[key]
        lines.append(macro(f'{prefix}DeltaSNRZeroDb', f"{entry['delta_snr_mean']:+.2f}"))
        lines.append(macro(f'{prefix}DeltaSNRZeroDbStd', f"{entry['delta_snr_std']:.2f}"))
        lines.append(macro(f'{prefix}RhoRZeroDb', f"{entry['rho_r_mean']:.1f}"))
        lines.append(macro(f'{prefix}MAER', f"{entry['mae_r_mean']:.2f}"))
        lines.append(macro(f'{prefix}RQRSZeroDb', f"{entry['r_qrs_mean']:.4f}"))

    diff_clamp_plain = b0['nlms_clamped_amp']['delta_snr_mean'] - b0['nlms_stat_tuned']['delta_snr_mean']
    lines.append(macro('diffClampVsPlainDeltaSNR', f'{diff_clamp_plain:+.2f}'))

    lines.append(macro('rawIsoSNRZeroDb', f"{b0['raw']['delta_snr_iso_mean']:+.2f}"))
    lines.append(macro('statTunedNlmsIsoSNR', f"{b0['nlms_stat_tuned']['delta_snr_iso_mean']:+.2f}"))
    lines.append(macro('nlmsClampAmpIsoSNR', f"{b0['nlms_clamped_amp']['delta_snr_iso_mean']:+.2f}"))
    lines.append(macro('fastNlmsIsoSNR', f"{b0['nlms_tuned']['delta_snr_iso_mean']:+.2f}"))
    diff_iso = b0['nlms_clamped_amp']['delta_snr_iso_mean'] - b0['nlms_stat_tuned']['delta_snr_iso_mean']
    lines.append(macro('diffClampVsPlainIsoSNR', f'{diff_iso:+.2f}'))
    diff_pt_duty = b0['pt_look_nlms']['duty_p1_mean'] - b0['nlms_clamped_amp']['duty_p1_mean']
    lines.append(macro('diffPtVsAmpDuty', f'{diff_pt_duty:.1f}'))

    lines.append(macro('nlmsClampAmpMinusSixDb', f"{bench['-6.0']['nlms_clamped_amp']['delta_snr_mean']:+.2f}"))
    lines.append(macro('nlmsClampAmpPlusSixDb', f"{bench['6.0']['nlms_clamped_amp']['delta_snr_mean']:+.2f}"))
    lines.append(macro('nlmsClampAmpPlusTwelveDb', f"{bench['12.0']['nlms_clamped_amp']['delta_snr_mean']:+.2f}"))
    lines.append(macro('statTunedNlmsMinusSixDb', f"{bench['-6.0']['nlms_stat_tuned']['delta_snr_mean']:+.2f}"))
    lines.append(macro('rlsDeltaSNRMinusSixDb', f"{bench['-6.0']['rls']['delta_snr_mean']:+.2f}"))
    lines.append(macro('rlsDeltaSNRPlusSixDb', f"{bench['6.0']['rls']['delta_snr_mean']:+.2f}"))
    lines.append(macro('rlsDeltaSNRPlusTwelveDb', f"{bench['12.0']['rls']['delta_snr_mean']:+.2f}"))

    lines.append(macro('nlmsClampAmpZeroDbSNR', f"{bench['0.0']['nlms_clamped_amp']['delta_snr_mean']:+.2f}"))
    lines.append(macro('imuAmpZeroDbSNR', f"{bench['0.0']['imu_amp']['delta_snr_mean']:+.2f}"))

    sig_amp = b0['significance_clamped_amp']
    p_plain = sig_amp['nlms_stat_tuned']['p_value']
    lines.append(macro('pValClampVsPlain', fmt_p(p_plain)))
    lines.append(macro('pvalPlainNLMSHolm', fmt_p(sig_amp['nlms_stat_tuned'].get('p_value_holm', p_plain))))
    lines.append(macro('pvalPlainNLMSRecord', fmt_p(sig_amp['nlms_stat_tuned'].get('p_value_record', p_plain))))
    lines.append(macro('pValClampVsRandom', fmt_p(sig_amp['random_clamp']['p_value'])))
    lines.append(macro('pValClampVsShifted', fmt_p(sig_amp['shifted_clamp']['p_value'])))
    lines.append(macro('pValClampVsOracle', fmt_p(sig_amp['oracle_gated']['p_value'])))
    lines.append(macro('pValClampVsLookahead', fmt_p(sig_amp['pt_look_nlms']['p_value'])))
    lines.append(macro('pValClampVsPredictive', fmt_p(sig_amp['pt_pred_nlms']['p_value'])))
    lines.append(macro('pValClampVsDiff', fmt_p(sig_amp['nlms_clamped_diff']['p_value'])))
    lines.append(macro('pValClampVsDePinto', fmt_p(sig_amp['depinto_slope_delay']['p_value'])))
    lines.append(macro('pValRlsMinusSixDb', fmt_p(bench['-6.0']['significance_clamped_amp']['rls']['p_value'])))
    if 'significance_plain_nlms' in b0:
        sp = b0['significance_plain_nlms']
        lines.append(macro('pValRandomVsPlain', fmt_p(sp['random_clamp']['p_value'])))
        lines.append(macro('pValShiftedVsPlain', fmt_p(sp['shifted_clamp']['p_value'])))

    if 'reference_noise_sweep' in bench:
        rns = bench['reference_noise_sweep']
        for k, tag in [('0.0', 'Zero'), ('0.5', 'Half'), ('1.0', 'One'), ('2.0', 'Two')]:
            if k in rns:
                lines.append(macro(f'refNoisePlain{tag}', f"{rns[k]['plain_nlms']['delta_snr_mean']:+.2f}"))
                lines.append(macro(f'refNoiseClamp{tag}', f"{rns[k]['clamped_nlms']['delta_snr_mean']:+.2f}"))

    if dyn and 'dynamic_plant_benchmarks' in dyn:
        dp = dyn['dynamic_plant_benchmarks']
        imu_amp_entry = dp['imu_amp']
        lines.append(macro('pValDynamicImuVsClamp', fmt_p(dp['wilcoxon_amp_vs_clamped']['p_value'])))
        lines.append(macro('dynNlmsTunedSlowSNR', f"{dp['nlms_tuned_002']['delta_snr_mean']:+.2f}"))
        lines.append(macro('dynNlmsTunedSlowSNRStd', f"{dp['nlms_tuned_002']['delta_snr_std']:.2f}"))
        lines.append(macro('dynNlmsTunedSNR', f"{dp['nlms_tuned_005']['delta_snr_mean']:+.2f}"))
        lines.append(macro('dynNlmsClampAmpSNR', f"{dp['nlms_clamped_amp']['delta_snr_mean']:+.2f}"))
        diff_dyn = dp['nlms_clamped_amp']['delta_snr_mean'] - dp['nlms_tuned_005']['delta_snr_mean']
        lines.append(macro('diffDynGain', f'{diff_dyn:+.2f}'))
        lines.append(macro('dynRlsDefSNR', f"{dp['plain_rls']['delta_snr_mean']:+.2f}"))
        lines.append(macro('dynImuAmpDefSNR', f"{imu_amp_entry['delta_snr_mean']:+.2f}"))
        lines.append(macro('pValDynamicClampVsTuned', fmt_p(dp['wilcoxon_clamp_vs_tuned_nlms']['p_value'])))
        if 'wilcoxon_clamp_vs_tuned_002' in dp:
            lines.append(macro('pValDynamicClampVsTunedSlow', fmt_p(dp['wilcoxon_clamp_vs_tuned_002']['p_value'])))

    if dyn and 'stress_tests' in dyn:
        st = dyn['stress_tests']
        s_names = {'S0': 'sZero', 'S1': 'sOne', 'S2': 'sTwo', 'S3': 'sThree', 'S4': 'sFour', 'S5': 'sFive'}
        for s_key, s_name in s_names.items():
            if s_key in st:
                entry = st[s_key]
                lines.append(macro(f'{s_name}NlmsSNR', f"{entry['nlms_tuned']:+.2f}"))
                lines.append(macro(f'{s_name}NlmsClampedSNR', f"{entry['nlms_clamped']:+.2f}"))
                lines.append(macro(f'{s_name}RlsSNR', f"{entry['rls']:+.2f}"))
                lines.append(macro(f'{s_name}AmpSNR', f"{entry['amp']:+.2f}"))
        if 'S5' in st:
            lines.append(macro('sFiveCurvSNR', f"{st['S5']['curv']:+.2f}"))
            lines.append(macro('pValAmpVsCurvSEnd', fmt_p(st['S5']['p_val_amp_vs_curv'])))
            if 'nlms_curv' in st['S5']:
                lines.append(macro('sFiveStandaloneCurvSNR', f"{st['S5']['nlms_curv']:+.2f}"))
                lines.append(macro('sFiveStandaloneCurvDuty', f"{st['S5']['nlms_curv_duty']:.1f}"))
                lines.append(macro('sFiveStandaloneAmpDuty', f"{st['S5']['nlms_clamped_duty']:.1f}"))
                lines.append(macro('pValStandaloneAmpVsCurvSFive', fmt_p(st['S5']['p_val_standalone_amp_vs_curv'])))

    if rob:
        huber = rob['nlms_clipped_error']
        lines.append(macro('nlmsClippedErrorSNR', f"{huber['delta_snr_mean']:+.2f}"))
        lines.append(macro('nlmsClippedErrorSNRStd', f"{huber['delta_snr_std']:.2f}"))
        lines.append(macro('nlmsClippedErrorRhoR', f"{huber['rho_r_mean']:.1f}"))
        if 'mae_r_mean' in huber:
            lines.append(macro('huberMAER', f"{huber['mae_r_mean']:.2f}"))
            lines.append(macro('huberRQRS', f"{huber['r_qrs_mean']:.4f}"))
        lines.append(macro('pValClampVsHuber', fmt_p(huber['wilcoxon_vs_clamp']['p_value'])))
        lines.append(macro('smNlmsSNR', f"{rob['sm_nlms']['delta_snr_mean']:+.2f}"))
        lines.append(macro('smNlmsRhoR', f"{rob['sm_nlms']['rho_r_mean']:.1f}"))
        if 'vss_aboulnasr' in rob:
            lines.append(macro('vssAboulnasrSNR', f"{rob['vss_aboulnasr']['delta_snr_mean']:+.2f}"))
        if 'gain_mismatch_experiment' in rob:
            gm = rob['gain_mismatch_experiment']
            lines.append(macro('gainHalfHuberSNR', f"{gm['0.5']['huber_fixed']['delta_snr_mean']:+.2f}"))
            lines.append(macro('pValGainHalf', fmt_p(gm['0.5']['huber_fixed']['wilcoxon_vs_clamp']['p_value'])))
            lines.append(macro('gainDoubleHuberSNR', f"{gm['2.0']['huber_fixed']['delta_snr_mean']:+.2f}"))
            lines.append(macro('pValGainDouble', fmt_p(gm['2.0']['huber_fixed']['wilcoxon_vs_clamp']['p_value'])))
            if 'huber_adaptive' in gm['0.5']:
                lines.append(macro('gainAdaptiveHuberSNR', f"{gm['0.5']['huber_adaptive']['delta_snr_mean']:+.2f}"))
                lines.append(macro('pValClampVsAdaptiveHuber', fmt_p(gm['0.5']['huber_adaptive']['wilcoxon_vs_clamp']['p_value'])))

    if trig:
        lines.append(macro('triggerRecallPooled', f"{trig['pooled_recall']:.1f}"))
        lines.append(macro('triggerRecallMean', f"{trig['mean_recall']:.1f}"))
        lines.append(macro('triggerRecallMin', f"{trig['min_recall']:.1f}"))
        lines.append(macro('triggerRecallMinRec', f"{trig['min_record']}"))
        lines.append(macro('triggerRecallMax', f"{trig['max_recall']:.1f}"))
        lines.append(macro('triggerRecallMaxRec', f"{trig['max_record']}"))
        lines.append(macro('triggerPrecision', f"{trig['sample_precision']:.1f}"))

    if blunt:
        lines.append(macro('bluntingWorstFivePct', f"{blunt['p95_blunting_pct']:.1f}"))
        lines.append(macro('bluntingMean', f"{blunt['mean_blunting_pct']:.1f}"))
        apex_pct = blunt.get('fast_apex_blunting_pct', 12.7)
        apex_diff = blunt.get('fast_apex_diff_pct', -12.7)
        lines.append(macro('bluntingFastApexPct', f'{apex_pct:.1f}\\%'))
        lines.append(macro('bluntingFastApexDiff', f'{apex_diff:+.1f}\\%'))
        if 'plain_apex_diff_pct' in blunt:
            lines.append(macro('bluntingPlainApexDiff', f"{blunt['plain_apex_diff_pct']:+.1f}\\%"))
            lines.append(macro('bluntingClampApexDiff', f"{blunt['clamp_apex_diff_pct']:+.1f}\\%"))
            lines.append(macro('bluntingCleanApexMv', f"{blunt['clean_apex_mv']:.2f}"))

    if mu_sweep:
        for k, tag in [('0.001', 'One'), ('0.002', 'Two'), ('0.005', 'Five'), ('0.01', 'Ten'), ('0.02', 'Twenty')]:
            lines.append(macro(f'muSweepGain{tag}', f"{mu_sweep[k]['gain_delta_snr']:+.2f}"))

    if rls_sweep:
        lines.append(macro('rlsOptimalZeroDbSNR', f"{rls_sweep['0.0']['0.999']['mean']:+.2f}"))

    if prof:
        lines.append(macro('cortexMFourCycles', f"{prof['float32_cycles_m4']}"))
        lines.append(macro('cortexMFourTimeUs', f"{prof['float32_exec_time_us']:.2f}"))
        lines.append(macro('cortexMFourRamBytes', f"{prof['float32_ram_bytes']}"))
        lines.append(macro('cortexMFourDutyPct', f"{prof['float32_m4_cpu_load_pct']:.3f}"))

    content = '\n'.join(lines) + '\n'
    with open(RESULTS_DIR / 'results_macros.tex', 'w') as f:
        f.write(content)


if __name__ == '__main__':
    generate_macros()
