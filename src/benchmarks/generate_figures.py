import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.common import CODEBASE_DIR, FS, TAPS, artifact_scale, load_ecg_record, load_motion_artifact, write_json
from dsp.adaptive_filters import FixedStepNLMS, TriggerClampedNLMS, run_filter

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 9.0,
    'axes.labelsize': 8.8,
    'axes.titlesize': 8.8,
    'xtick.labelsize': 8.2,
    'ytick.labelsize': 8.2,
    'legend.fontsize': 7.2,
    'figure.titlesize': 9.5,
    'lines.linewidth': 1.1,
    'grid.alpha': 0.35,
    'grid.linestyle': ':',
    'grid.color': '#94a3b8',
    'axes.grid': True,
    'axes.edgecolor': '#0f172a',
    'axes.linewidth': 0.8,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
})


def generate_system_block_diagram(output_dir):
    fig, ax = plt.subplots(figsize=(9.8, 5.0), dpi=300)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 5.0)
    ax.axis('off')

    c_dark = '#0f172a'
    c_blue_tint = '#f8fafc'
    c_clamp_tint = '#f8fafc'
    c_border = '#334155'
    c_wire = '#0f172a'
    c_gate = '#0284c7'

    def draw_node_box(x, y, w, h, title, formulas, bg=c_blue_tint, border=c_border, lw=1.1, ls='-', fontsize=6.5):
        cx = x + w / 2.0
        p = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.03,rounding_size=0.08',
            facecolor=bg, edgecolor=border, linewidth=lw, linestyle=ls, zorder=3,
        )
        ax.add_patch(p)
        ax.text(cx, y + h - 0.16, title, ha='center', va='top', fontsize=7.4, fontweight='bold', color=c_dark, zorder=4)
        lines = '\n'.join(formulas)
        ax.text(cx, y + (h - 0.28) / 2.0, lines, ha='center', va='center', fontsize=fontsize, color=c_dark, zorder=4)

    y_top = 4.35
    x_sum = 7.7
    r_sum = 0.22

    ax.text(0.3, y_top + 0.28, r'Primary Biopotential $d(n) = s(n) + \eta(n)$', fontsize=8.2, fontweight='bold', color=c_dark)
    ax.plot([0.3, x_sum - r_sum], [y_top, y_top], color=c_wire, lw=1.3, zorder=2)
    ax.annotate('', xy=(x_sum - r_sum, y_top), xytext=(x_sum - r_sum - 0.15, y_top),
                arrowprops=dict(arrowstyle='->', lw=1.3, color=c_wire), zorder=3)

    sum_circle = patches.Circle((x_sum, y_top), r_sum, facecolor='#ffffff', edgecolor=c_border, linewidth=1.3, zorder=4)
    ax.add_patch(sum_circle)
    ax.text(x_sum, y_top, r'$\mathbf{\Sigma}$', ha='center', va='center', fontsize=11, fontweight='bold', zorder=5)
    ax.text(x_sum - 0.30, y_top + 0.14, r'$+$', fontsize=8.5, fontweight='bold', color=c_dark, zorder=5)
    ax.text(x_sum - 0.16, y_top - 0.32, r'$-$', fontsize=9.5, fontweight='bold', color=c_dark, zorder=5)

    ax.plot([x_sum + r_sum, 9.6], [y_top, y_top], color=c_wire, lw=1.3, zorder=2)
    ax.annotate('', xy=(9.6, y_top), xytext=(9.4, y_top),
                arrowprops=dict(arrowstyle='->', lw=1.3, color=c_wire), zorder=3)
    ax.text(9.65, y_top + 0.15, r'Denoised Output', fontsize=7.8, fontweight='bold', color=c_dark)
    ax.text(9.65, y_top - 0.12, r'$e(n) = d(n) - \hat{\eta}(n)$', fontsize=7.2, fontstyle='italic', color=c_dark)

    clamp_x = 0.85
    clamp_y = 2.35
    clamp_w = 4.75
    clamp_h = 1.35
    clamp_frame = patches.FancyBboxPatch(
        (clamp_x, clamp_y), clamp_w, clamp_h,
        boxstyle='round,pad=0.04,rounding_size=0.09',
        facecolor='#fafafa', edgecolor='#94a3b8', linewidth=0.9, linestyle='--', zorder=1,
    )
    ax.add_patch(clamp_frame)
    ax.text(
        clamp_x + 0.20, clamp_y + clamp_h - 0.16,
        r'Causal Zero-Lookahead Clamping Path ($\tau_g = 0$, Multiplier-Free)',
        fontsize=7.4, fontweight='bold', fontstyle='italic', color='#334155', zorder=2,
    )

    ax.plot(0.50, y_top, 'o', color=c_wire, markersize=4, zorder=4)
    ax.plot([0.50, 0.50, 1.05], [y_top, 2.90, 2.90], color=c_wire, lw=1.2, zorder=2)
    ax.annotate('', xy=(1.05, 2.90), xytext=(0.90, 2.90),
                arrowprops=dict(arrowstyle='->', lw=1.2, color=c_wire), zorder=3)

    draw_node_box(
        1.05, 2.45, 1.30, 0.90,
        r'Envelope Tracker',
        [r'$\sigma_d(n) = \sigma_d(n-1) +$', r'$\left(|d(n)| - \sigma_d\right) \gg 4$', r'($\lambda = 1/16$, shift-only)'],
        bg=c_clamp_tint, lw=1.0, fontsize=6.2,
    )

    ax.annotate('', xy=(2.70, 2.90), xytext=(2.35, 2.90),
                arrowprops=dict(arrowstyle='->', lw=1.2, color=c_wire), zorder=3)

    draw_node_box(
        2.70, 2.45, 1.30, 0.90,
        r'Ratio Comparator',
        [r'$\mathcal{A}(n) = \frac{|d(n)|}{\sigma_d(n) + \varepsilon}$', r'Apex trigger:', r'$\mathcal{A}(n) \geq \Theta_{\mathrm{amp}} = 3.0$'],
        bg=c_clamp_tint, lw=1.0, fontsize=6.2,
    )

    ax.annotate('', xy=(4.35, 2.90), xytext=(4.00, 2.90),
                arrowprops=dict(arrowstyle='->', lw=1.2, color=c_wire), zorder=3)

    draw_node_box(
        4.35, 2.45, 1.05, 0.90,
        r'Hold Logic',
        [r'$H(n) \leftarrow 8$ on trigger', r'Hold: $32\text{ ms}$', r'Gate $G(n) \in \{0, 1\}$'],
        bg=c_clamp_tint, lw=1.0, fontsize=6.2,
    )

    w_tf, h_tf = 2.4, 1.0
    x_tf, y_tf = x_sum - w_tf / 2.0, 2.45
    draw_node_box(
        x_tf, y_tf, w_tf, h_tf,
        r'Transversal Filter ($M = 16$)',
        [r'$\mathbf{x}(n) = [x(n), \dots, x(n-M+1)]^T$', r'$\hat{\eta}(n) = \mathbf{w}^T(n)\mathbf{x}(n) = \sum_{k=0}^{M-1} w_k x(n-k)$'],
        bg=c_blue_tint, lw=1.2, fontsize=6.3,
    )

    ax.plot([x_sum, x_sum], [y_tf + h_tf, y_top - r_sum], color=c_wire, lw=1.3, zorder=2)
    ax.annotate('', xy=(x_sum, y_top - r_sum), xytext=(x_sum, y_top - r_sum - 0.15),
                arrowprops=dict(arrowstyle='->', lw=1.3, color=c_wire), zorder=3)
    ax.text(x_sum + 0.12, (y_tf + h_tf + y_top) / 2.0 - 0.05, r'Artifact Estimate $\hat{\eta}(n)$',
            va='center', fontsize=7.2, fontstyle='italic', color=c_dark)

    w_up, h_up = 2.8, 1.15
    x_up, y_up = x_sum - w_up / 2.0, 0.45
    draw_node_box(
        x_up, y_up, w_up, h_up,
        r'Gradient Tap-Weight Update Unit',
        [r'$\mathbf{w}(n+1) = \mathbf{w}(n) + \frac{\mu(n)}{\|\mathbf{x}(n)\|^2 + \delta}\, e(n)\,\mathbf{x}(n)$',
         r'Step: $\mu(n) = \mu_{\min}\ (10^{-4})$ if $G(n)=1$',
         r'else $\mu_0\ (0.002)$ (unclamped)'],
        bg=c_blue_tint, lw=1.2, fontsize=6.2,
    )

    ax.plot([x_sum, x_sum], [y_up + h_up, y_tf], color=c_wire, lw=1.3, zorder=2)
    ax.annotate('', xy=(x_sum, y_tf), xytext=(x_sum, y_tf - 0.15),
                arrowprops=dict(arrowstyle='->', lw=1.3, color=c_wire), zorder=3)
    ax.text(x_sum + 0.12, (y_up + h_up + y_tf) / 2.0, r'Taps $\mathbf{w}(n+1)$',
            va='center', fontsize=7.0, fontstyle='italic', color=c_dark)

    ax.text(0.3, 1.35, r'Motion Reference $x(n)$', fontsize=8.0, fontweight='bold', color=c_dark)
    ax.text(0.3, 1.15, r'(Chest Accelerometer)', fontsize=7.0, fontstyle='italic', color='#475569')

    ax.plot([0.3, x_up], [0.95, 0.95], color=c_wire, lw=1.3, zorder=2)
    ax.annotate('', xy=(x_up, 0.95), xytext=(x_up - 0.15, 0.95),
                arrowprops=dict(arrowstyle='->', lw=1.3, color=c_wire), zorder=3)

    x_branch = 6.15
    ax.plot(x_branch, 0.95, 'o', color=c_wire, markersize=3.5, zorder=4)
    ax.plot([x_branch, x_branch, x_tf], [0.95, 2.95, 2.95], color=c_wire, lw=1.2, zorder=2)
    ax.annotate('', xy=(x_tf, 2.95), xytext=(x_tf - 0.15, 2.95),
                arrowprops=dict(arrowstyle='->', lw=1.2, color=c_wire), zorder=3)
    ax.text(x_branch + 0.08, 1.95, r'Reference $x(n)$', fontsize=7.0, fontstyle='italic', color=c_dark, va='center')

    x_gate = 5.65
    ax.plot([5.40, x_gate, x_gate, x_up], [2.90, 2.90, 1.35, 1.35], color=c_gate, lw=1.3, zorder=2)
    ax.annotate('', xy=(x_up, 1.35), xytext=(x_up - 0.15, 1.35),
                arrowprops=dict(arrowstyle='->', lw=1.3, color=c_gate), zorder=3)
    ax.text(x_gate - 0.08, 2.15, r'Gate Flag $G(n)$' + '\n' + r'$\mu \in \{10^{-4}, 0.002\}$',
            fontsize=6.8, fontweight='bold', color=c_gate, ha='right', va='center')

    x_err = 9.40
    ax.plot(x_err, y_top, 'o', color=c_wire, markersize=4, zorder=4)
    ax.plot([x_err, x_err, x_up + w_up], [y_top, 0.95, 0.95], color=c_wire, lw=1.2, linestyle=':', zorder=2)
    ax.annotate('', xy=(x_up + w_up, 0.95), xytext=(x_up + w_up + 0.15, 0.95),
                arrowprops=dict(arrowstyle='->', lw=1.2, color=c_wire), zorder=3)
    ax.text(x_err + 0.08, 2.5, r'Error Feedback $e(n)$', fontsize=7.2, fontstyle='italic', color=c_dark)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'system_block_diagram.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'system_block_diagram.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_waveform_comparison(output_dir):
    n_samples = int(60.0 * FS)
    clean_ecg, _ = load_ecg_record('100', fs=FS, max_seconds=60.0)
    coupled_artifact, x_ref = load_motion_artifact(fs=FS, max_samples=n_samples)
    d_noisy = clean_ecg + coupled_artifact * artifact_scale(clean_ecg, coupled_artifact, 0.0)

    e_fast, _, _ = run_filter(FixedStepNLMS(mu=0.02), d_noisy, x_ref, taps=TAPS)
    e_plain, _, _ = run_filter(FixedStepNLMS(mu=0.002), d_noisy, x_ref, taps=TAPS)
    e_clamp, mu_clamp, phase_clamp = run_filter(
        TriggerClampedNLMS(mu=0.002, trigger='amplitude', theta=3.0, hold=8),
        d_noisy, x_ref, taps=TAPS,
    )

    ratio_full = np.zeros(n_samples)
    sig_env = 0.5
    for n in range(n_samples):
        sig_env = (1.0 - 1.0 / 16.0) * sig_env + (1.0 / 16.0) * abs(d_noisy[n])
        ratio_full[n] = abs(d_noisy[n]) / (sig_env + 1e-4)

    p_beat = 4315
    w_start = p_beat - int(0.55 * FS)
    w_end = w_start + int(1.8 * FS)
    t = (np.arange(w_start, w_end) - w_start) / FS

    z_s = p_beat - int(0.045 * FS)
    z_e = p_beat + int(0.045 * FS)
    t_z = (np.arange(z_s, z_e) - p_beat) / FS * 1000.0

    clean_b1_peak = float(np.max(clean_ecg[z_s:z_e]))
    fast_b1_peak = float(np.max(e_fast[z_s:z_e]))
    plain_b1_peak = float(np.max(e_plain[z_s:z_e]))
    clamp_b1_peak = float(np.max(e_clamp[z_s:z_e]))
    diff_fast = (fast_b1_peak - clean_b1_peak) / clean_b1_peak * 100.0
    diff_plain = (plain_b1_peak - clean_b1_peak) / clean_b1_peak * 100.0
    diff_clamp = (clamp_b1_peak - clean_b1_peak) / clean_b1_peak * 100.0

    fig = plt.figure(figsize=(7.6, 5.2), dpi=300)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.65, 1.0], hspace=0.38, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)

    ax_zoom = fig.add_subplot(gs[0:2, 1])
    ax_ratio = fig.add_subplot(gs[2:4, 1], sharex=ax_zoom)

    ax0.tick_params(labelbottom=False)
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)
    ax3.tick_params(labelbottom=True)
    ax_zoom.tick_params(labelbottom=False)
    ax_ratio.tick_params(labelbottom=True)

    ax0.plot(t, d_noisy[w_start:w_end], color='#64748b', linewidth=0.9, linestyle='-', label=r'Contaminated $d(n)$')
    ax0.plot(t, clean_ecg[w_start:w_end], color='#000000', linewidth=1.1, linestyle='-', label=r'Clean ECG $s(n)$')
    ax0.set_ylabel('Ampl. (mV)', fontsize=8.0)
    ax0.set_title('(a) Primary Biopotential with Motion Artifact (0 dB NSTDB)', loc='left', fontweight='bold', fontsize=8.0, pad=3)
    ax0.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=6.6, ncol=2)
    ax0.set_ylim(-1.5, 2.5)

    ax1.plot(t, clean_ecg[w_start:w_end], color='#94a3b8', linewidth=0.9, linestyle=':', label=r'Clean $s(n)$')
    ax1.plot(t, e_fast[w_start:w_end], color='#000000', linewidth=1.1, linestyle='--', label=r'Fast NLMS ($\mu = 0.02$)')
    ax1.set_ylabel('Ampl. (mV)', fontsize=8.0)
    ax1.set_title(r'(b) Fast NLMS ($\mu = 0.02$): Gradient Leakage', loc='left', fontweight='bold', fontsize=8.0, pad=3)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=6.6, ncol=2)
    ax1.set_ylim(-1.5, 2.5)

    ax2.plot(t, clean_ecg[w_start:w_end], color='#94a3b8', linewidth=0.9, linestyle=':', label=r'Clean $s(n)$')
    ax2.plot(t, e_clamp[w_start:w_end], color='#000000', linewidth=1.1, linestyle='-', label=r'Proposed Clamped ($\mu = 0.002, \Theta=3.0$)')
    ax2.set_ylabel('Ampl. (mV)', fontsize=8.0)
    ax2.set_title(r'(c) Proposed Clamped NLMS ($\mu = 0.002$, Zero Lookahead)', loc='left', fontweight='bold', fontsize=8.0, pad=3)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=6.6, ncol=2)
    ax2.set_ylim(-1.5, 2.5)

    t_b1_start = (z_s - w_start) / FS
    t_b1_end = (z_e - w_start) / FS
    rect = patches.Rectangle((t_b1_start, -0.3), t_b1_end - t_b1_start, 1.8,
                             linewidth=0.8, edgecolor='#0f172a', facecolor='none', linestyle='--')
    ax2.add_patch(rect)
    ax2.text(t_b1_end + 0.02, 1.1, r'Beat 1 $\rightarrow$', fontsize=6.8, fontstyle='italic', color='#0f172a')

    p_seg = phase_clamp[w_start:w_end]
    mu_seg = mu_clamp[w_start:w_end]
    in_clamp = False
    clamp_start = 0
    labeled = False
    for idx_t in range(len(t)):
        if p_seg[idx_t] == 1 and (not in_clamp):
            in_clamp = True
            clamp_start = t[idx_t]
        elif p_seg[idx_t] != 1 and in_clamp:
            in_clamp = False
            lbl = r'Hold ($\mu_{\min}$)' if not labeled else ''
            ax3.axvspan(clamp_start, t[idx_t], color='#e2e8f0', ec='#94a3b8', linestyle='-', alpha=0.9, label=lbl)
            labeled = True
    if in_clamp:
        lbl = r'Hold ($\mu_{\min}$)' if not labeled else ''
        ax3.axvspan(clamp_start, t[-1], color='#e2e8f0', ec='#94a3b8', linestyle='-', alpha=0.9, label=lbl)

    ax3.plot(t, mu_seg, color='#000000', linewidth=1.1, linestyle='-', label=r'Step $\mu(n)$')
    ax3.set_ylabel(r'$\mu(n)$', fontsize=8.0)
    ax3.set_xlabel('Time (seconds)', fontsize=8.2)
    ax3.set_title(r'(d) Causal Adaptation Gating Dynamics (8-Sample Hold)', loc='left', fontweight='bold', fontsize=8.0, pad=3)
    ax3.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=6.6, ncol=2)
    ax3.set_ylim(-0.0005, 0.0035)

    ax_zoom.plot(t_z, clean_ecg[z_s:z_e], color='#94a3b8', linewidth=1.3, linestyle=':', label=f'Clean ({clean_b1_peak:.2f} mV)')
    ax_zoom.plot(t_z, e_plain[z_s:z_e], color='#475569', linewidth=1.1, linestyle='-.', label=f'Plain $\\mu=0.002$ ({diff_plain:+.1f}%)')
    ax_zoom.plot(t_z, e_fast[z_s:z_e], color='#000000', linewidth=1.2, linestyle='--', marker='s', markersize=3.0, label=f'Fast $\\mu=0.02$ ({diff_fast:+.1f}%)')
    ax_zoom.plot(t_z, e_clamp[z_s:z_e], color='#000000', linewidth=1.4, linestyle='-', marker='o', markersize=3.5, label=f'Clamped $\\mu=0.002$ ({diff_clamp:+.1f}%)')
    ax_zoom.set_ylabel('Ampl. (mV)', fontsize=8.0)
    ax_zoom.set_title(r'(e) Ventricular Apex Detail ($\pm 45$ ms)', loc='left', fontweight='bold', fontsize=8.0, pad=3)
    ax_zoom.set_ylim(-0.25, 1.55)
    ax_zoom.grid(True, linestyle=':', alpha=0.5)
    ax_zoom.legend(loc='lower right', fontsize=6.0, frameon=True, framealpha=0.95, edgecolor='#cbd5e1')

    ratio_seg = ratio_full[z_s:z_e]

    ax_ratio.plot(t_z, ratio_seg, color='#000000', linewidth=1.2, linestyle='-', label=r'Ratio $\mathcal{A}(n)$')
    ax_ratio.axhline(3.0, color='#64748b', linewidth=1.0, linestyle='--', label=r'Threshold $\Theta = 3.0$')
    ax_ratio.fill_between(t_z, 0, ratio_seg, where=(ratio_seg >= 3.0), color='#e2e8f0', alpha=0.8, label='Clamped Hold')
    ax_ratio.set_ylabel(r'Ratio $\mathcal{A}(n)$', fontsize=8.0)
    ax_ratio.set_xlabel('Relative Time (ms)', fontsize=8.0)
    ax_ratio.set_title(r'(f) Zero-Lookahead Ratio Comparator', loc='left', fontweight='bold', fontsize=8.0, pad=3)
    ax_ratio.set_ylim(0, 6.8)
    ax_ratio.grid(True, linestyle=':', alpha=0.5)
    ax_ratio.legend(loc='upper right', fontsize=6.6, frameon=True, framealpha=0.95, edgecolor='#cbd5e1')

    fig.align_ylabels([ax0, ax1, ax2, ax3])
    fig.align_ylabels([ax_zoom, ax_ratio])
    fig.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.98, hspace=0.38, wspace=0.28)

    plt.savefig(os.path.join(output_dir, 'waveform_comparison.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'waveform_comparison.png'), dpi=300)
    plt.close()


def generate_ablation_breakdown(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    snrs = ['-6.0', '0.0', '6.0', '12.0']
    snr_labels = ['-6 dB', '0 dB', '+6 dB', '+12 dB']

    styles = [
        ('raw', 'Raw Contaminated', '#f8fafc', '..', '#0f172a'),
        ('nlms_stat_tuned', 'Plain NLMS', '#cbd5e1', '--', '#0f172a'),
        ('rls', 'Plain RLS', '#64748b', '//', '#0f172a'),
        ('nlms_clamped_amp', 'Proposed NLMS+Amp', '#000000', '', '#000000'),
    ]

    n_groups = len(snrs)
    n_bars = len(styles)
    bar_width = 0.18
    index = np.arange(n_groups)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 4.4), dpi=300, sharex=True)

    bars_list = []
    labels_list = []

    for i, (k, name, col, hatch, ec) in enumerate(styles):
        means = [data[s][k]['delta_snr_mean'] for s in snrs]
        stds = [data[s][k]['delta_snr_std'] for s in snrs]
        b = ax1.bar(index + i * bar_width, means, bar_width, yerr=stds, capsize=2.0,
                    color=col, hatch=hatch, edgecolor=ec, linewidth=0.9, label=name)
        bars_list.append(b)
        labels_list.append(name)

    ax1.set_ylabel(r'$\Delta\mathrm{SNR}$ (dB)')
    ax1.set_title(r'(a) Denoising Performance Across SNR Regimes (Mean $\pm$ SD, N=28)', loc='left', fontweight='bold', pad=4)
    ax1.axhline(0, color='#64748b', linestyle='--', linewidth=0.8)
    ax1.set_ylim(-2.0, 24.0)

    for i, (k, name, col, hatch, ec) in enumerate(styles):
        means = [data[s][k]['rho_r_mean'] for s in snrs]
        stds = [data[s][k]['rho_r_std'] for s in snrs]
        ax2.bar(index + i * bar_width, means, bar_width, yerr=stds, capsize=2.0,
                color=col, hatch=hatch, edgecolor=ec, linewidth=0.9, label=name)

    ax2.set_xlabel('Input SNR Regime')
    ax2.set_ylabel(r'R-Peak Amplitude $\rho_R$ (%)')
    ax2.set_title(r'(b) R-Peak Amplitude Preservation $\rho_R$ Across Regimes', loc='left', fontweight='bold', pad=4)
    ax2.set_xticks(index + bar_width * (n_bars - 1) / 2)
    ax2.set_xticklabels(snr_labels)
    ax2.axhline(100.0, color='#0f172a', linestyle=':', linewidth=0.9)
    ax2.set_ylim(82.0, 107.0)

    fig.legend(bars_list, labels_list, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=4, fontsize=8.0, frameon=True, framealpha=0.95, edgecolor='#94a3b8')
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout(rect=[0, 0, 1, 0.93], pad=0.5)

    plt.savefig(os.path.join(output_dir, 'ablation_breakdown.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'ablation_breakdown.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    docs_dir = str(CODEBASE_DIR / 'docs')
    fig_dir = str(CODEBASE_DIR / 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    generate_system_block_diagram(fig_dir)
    generate_waveform_comparison(fig_dir)

    json_summary = os.path.join(docs_dir, 'benchmark_results.json')
    if os.path.exists(json_summary):
        generate_ablation_breakdown(json_summary, fig_dir)
