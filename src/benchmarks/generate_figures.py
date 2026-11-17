import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.common import CODEBASE_DIR, FS, TAPS, artifact_scale, load_ecg_record, load_motion_artifact
from dsp.adaptive_filters import FixedStepNLMS, TriggerClampedNLMS, run_filter

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 9.0,
    'axes.labelsize': 8.8,
    'axes.titlesize': 8.8,
    'xtick.labelsize': 8.2,
    'ytick.labelsize': 8.2,
    'legend.fontsize': 7.8,
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
    # Drawn at true print size (1 data unit = 1 inch) so that, when included at
    # \textwidth (~6.0 in) in the journal template, all text renders at ~7.5-8.5 pt.
    W, H, y0 = 6.15, 4.50, 0.30
    fig = plt.figure(figsize=(W, H - y0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(y0, H)
    ax.axis('off')

    c_dark = '#0f172a'
    c_tint = '#f8fafc'
    c_border = '#334155'
    c_wire = '#0f172a'
    c_gate = '#0284c7'
    fs_title = 8.4
    fs_body = 7.8
    fs_label = 8.0
    lw_wire = 1.1

    def draw_node_box(x, y, w, h, title, formulas, lw=1.0, fontsize=fs_body, linespacing=1.38):
        p = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.02,rounding_size=0.06',
            facecolor=c_tint, edgecolor=c_border, linewidth=lw, zorder=3,
        )
        ax.add_patch(p)
        cx = x + w / 2.0
        ax.text(cx, y + h - 0.07, title, ha='center', va='top', fontsize=fs_title,
                fontweight='bold', color=c_dark, zorder=4)
        ax.text(cx, y + h - 0.30, '\n'.join(formulas), ha='center', va='top',
                fontsize=fontsize, color=c_dark, linespacing=linespacing, zorder=4)

    def wire(xs, ys, color=c_wire, ls='-', arrow=True):
        ax.plot(xs, ys, color=color, lw=lw_wire, linestyle=ls, zorder=2, solid_capstyle='butt')
        if arrow:
            ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                        arrowprops=dict(arrowstyle='-|>', lw=lw_wire, color=color,
                                        mutation_scale=8, shrinkA=0, shrinkB=0), zorder=3)

    def node(x, y):
        ax.plot(x, y, 'o', color=c_wire, markersize=3.2, zorder=4)

    # ---- Primary path: d(n) -> summing junction -> e(n) ----------------------
    y_top = 4.12
    x_sum, r_sum = 4.55, 0.16
    x_err = 5.98

    ax.text(0.08, y_top + 0.10, r'Primary biopotential $d(n) = s(n) + \eta(n)$',
            fontsize=fs_title, fontweight='bold', color=c_dark, va='bottom')
    wire([0.08, x_sum - r_sum], [y_top, y_top])

    ax.add_patch(patches.Circle((x_sum, y_top), r_sum, facecolor='#ffffff',
                                edgecolor=c_border, linewidth=1.1, zorder=4))
    ax.text(x_sum, y_top, r'$\mathbf{\Sigma}$', ha='center', va='center', fontsize=10, zorder=5)
    ax.text(x_sum - r_sum - 0.13, y_top + 0.05, r'$+$', fontsize=9, color=c_dark, va='bottom', zorder=5)
    ax.text(x_sum + 0.06, y_top - r_sum - 0.05, r'$-$', fontsize=9.5, color=c_dark, va='top', zorder=5)

    wire([x_sum + r_sum, W - 0.02], [y_top, y_top])
    ax.text(x_sum + r_sum + 0.08, y_top + 0.10, r'Denoised output', fontsize=fs_label,
            fontweight='bold', color=c_dark, va='bottom')
    ax.text(x_sum + r_sum + 0.08, y_top - 0.10, r'$e(n) = d(n) - \hat{\eta}(n)$', fontsize=fs_label,
            color=c_dark, va='top')

    # ---- Causal clamping path (left column) ----------------------------------
    fx, fy, fw, fh = 0.24, 0.92, 2.56, 2.98
    ax.add_patch(patches.FancyBboxPatch(
        (fx, fy), fw, fh, boxstyle='round,pad=0.02,rounding_size=0.08',
        facecolor='#fafafa', edgecolor='#94a3b8', linewidth=0.9, linestyle='--', zorder=1,
    ))
    ax.text(fx + fw / 2.0, fy + fh - 0.06,
            'Causal zero-lookahead clamping path\n' + r'($\tau_g = 0$, multiplier-free)',
            ha='center', va='top', fontsize=fs_label, fontweight='bold', fontstyle='italic',
            color='#334155', linespacing=1.25, zorder=2)

    bx, bw, bh = 0.40, 2.24, 0.66
    y_env, y_rat, y_hold = 2.74, 1.93, 1.12

    node(0.12, y_top)
    wire([0.12, 0.12, bx], [y_top, y_env + bh / 2.0, y_env + bh / 2.0])

    draw_node_box(bx, y_env, bw, bh, r'Envelope Tracker',
                  [r'$\sigma_d(n) = \sigma_d(n-1) + \left(|d(n)| - \sigma_d\right) \gg 4$',
                   r'($\lambda = 1/16$, shift-only)'])
    wire([bx + bw / 2.0, bx + bw / 2.0], [y_env, y_rat + bh])

    draw_node_box(bx, y_rat, bw, bh, r'Ratio Comparator',
                  [r'$\mathcal{A}(n) = |d(n)| \,/\, (\sigma_d(n) + \varepsilon)$',
                   r'Apex trigger: $\mathcal{A}(n) \geq \Theta_{\mathrm{amp}} = 3.0$'])
    wire([bx + bw / 2.0, bx + bw / 2.0], [y_rat, y_hold + bh])

    draw_node_box(bx, y_hold, bw, bh, r'Hold Logic',
                  [r'$H(n) \leftarrow 8$ on trigger (hold $32$ ms)',
                   r'Gate $G(n) \in \{0, 1\}$'])

    # ---- Adaptive filter (right column) --------------------------------------
    rx, rw = 3.42, 2.34
    y_tf, h_tf = 2.72, 0.68
    y_up, h_up = 0.42, 0.98

    draw_node_box(rx, y_tf, rw, h_tf, r'Transversal Filter ($M = 16$)',
                  [r'$\mathbf{x}(n) = [x(n), \dots, x(n-M+1)]^T$',
                   r'$\hat{\eta}(n) = \mathbf{w}^T(n)\mathbf{x}(n) = \Sigma_{k=0}^{M-1}\, w_k x(n-k)$'],
                  lw=1.1, fontsize=7.6, linespacing=1.42)
    wire([x_sum, x_sum], [y_tf + h_tf, y_top - r_sum])
    ax.text(x_sum + 0.07, (y_tf + h_tf + y_top - r_sum) / 2.0, r'Estimate $\hat{\eta}(n)$',
            va='center', fontsize=fs_label, fontstyle='italic', color=c_dark)

    draw_node_box(rx, y_up, rw, h_up, r'Tap-Weight Update (NLMS)',
                  [r'$\mathbf{w}(n+1) = \mathbf{w}(n) + \frac{\mu(n)}{\|\mathbf{x}(n)\|^2 + \delta}\, e(n)\,\mathbf{x}(n)$',
                   r'$\mu(n) = \mu_{\min}\ (10^{-4})$ if $G(n)=1$',
                   r'else $\mu_0\ (0.002)$'],
                  lw=1.1, fontsize=7.6, linespacing=1.55)
    wire([x_sum, x_sum], [y_up + h_up, y_tf])
    ax.text(x_sum + 0.07, (y_up + h_up + y_tf) / 2.0, r'Taps $\mathbf{w}(n+1)$',
            va='center', fontsize=fs_label, fontstyle='italic', color=c_dark)

    # Gate flag G(n): Hold Logic -> update unit
    y_g_out = y_hold + bh / 2.0
    y_g_in = y_up + h_up - 0.22
    x_g = 3.08
    wire([bx + bw, x_g, x_g, rx], [y_g_out, y_g_out, y_g_in, y_g_in], color=c_gate)
    ax.text(2.94, y_g_out + 0.06, r'$G(n)$', ha='center', va='bottom',
            fontsize=fs_label, fontweight='bold', color=c_gate,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.6), zorder=5)

    # Motion reference x(n): -> update unit, branch -> transversal filter
    y_ref = 0.62
    ax.text(0.08, y_ref - 0.08, r'Motion reference $x(n)$ (chest accelerometer)',
            fontsize=fs_label, fontweight='bold', color=c_dark, va='top')
    wire([0.08, rx], [y_ref, y_ref])
    x_br = 3.25
    node(x_br, y_ref)
    wire([x_br, x_br, rx], [y_ref, y_tf + 0.22, y_tf + 0.22])
    ax.text(x_br - 0.05, (y_g_out + y_tf) / 2.0 + 0.05, r'$x(n)$', ha='right', va='center',
            fontsize=fs_label, fontstyle='italic', color=c_dark)

    # Error feedback e(n): output -> update unit
    node(x_err, y_top)
    wire([x_err, x_err, rx + rw], [y_top, y_up + 0.30, y_up + 0.30], ls=':')
    ax.text(x_err + 0.06, 2.0, r'Error feedback $e(n)$', rotation=90, ha='left', va='center',
            fontsize=fs_label, fontstyle='italic', color=c_dark)

    plt.savefig(os.path.join(output_dir, 'system_block_diagram.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(output_dir, 'system_block_diagram.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
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

    fig = plt.figure(figsize=(6.0, 6.2), dpi=300)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.55, 1.0], hspace=0.42, wspace=0.30)

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
    ax0.set_ylabel('Ampl. (mV)', fontsize=8.8)
    ax0.set_title('(a) Contaminated Input (0 dB NSTDB)', loc='left', fontweight='bold', fontsize=8.2, pad=3)
    ax0.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=7.6, ncol=2)
    ax0.set_ylim(-1.5, 2.5)

    ax1.plot(t, clean_ecg[w_start:w_end], color='#94a3b8', linewidth=0.9, linestyle=':', label=r'Clean $s(n)$')
    ax1.plot(t, e_fast[w_start:w_end], color='#000000', linewidth=1.1, linestyle='--', label=r'Fast NLMS ($\mu = 0.02$)')
    ax1.set_ylabel('Ampl. (mV)', fontsize=8.8)
    ax1.set_title(r'(b) Fast NLMS ($\mu = 0.02$): Gradient Leakage', loc='left', fontweight='bold', fontsize=8.2, pad=3)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=7.6, ncol=2)
    ax1.set_ylim(-1.5, 2.5)

    ax2.plot(t, clean_ecg[w_start:w_end], color='#94a3b8', linewidth=0.9, linestyle=':', label=r'Clean $s(n)$')
    ax2.plot(t, e_clamp[w_start:w_end], color='#000000', linewidth=1.1, linestyle='-', label=r'Proposed clamped')
    ax2.set_ylabel('Ampl. (mV)', fontsize=8.8)
    ax2.set_title(r'(c) Proposed Clamped NLMS ($\mu = 0.002$)', loc='left', fontweight='bold', fontsize=8.2, pad=3)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=7.6, ncol=2)
    ax2.set_ylim(-1.5, 2.5)

    t_b1_start = (z_s - w_start) / FS
    t_b1_end = (z_e - w_start) / FS
    rect = patches.Rectangle((t_b1_start, -0.3), t_b1_end - t_b1_start, 1.8,
                             linewidth=0.8, edgecolor='#0f172a', facecolor='none', linestyle='--')
    ax2.add_patch(rect)
    ax2.text(t_b1_end + 0.02, 1.1, r'Beat 1 $\rightarrow$', fontsize=7.8, fontstyle='italic', color='#0f172a')

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
    ax3.set_ylabel(r'$\mu(n)$', fontsize=8.8)
    ax3.set_xlabel('Time (s)', fontsize=8.8)
    ax3.set_title(r'(d) Step-Size Gating $\mu(n)$ (8-Sample Hold)', loc='left', fontweight='bold', fontsize=8.2, pad=3)
    ax3.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=7.6, ncol=2)
    ax3.set_ylim(-0.0005, 0.0042)

    ax_zoom.plot(t_z, clean_ecg[z_s:z_e], color='#94a3b8', linewidth=1.3, linestyle=':', label=f'Clean ({clean_b1_peak:.2f} mV)')
    ax_zoom.plot(t_z, e_plain[z_s:z_e], color='#475569', linewidth=1.1, linestyle='-.', label=f'Plain $\\mu=0.002$ ({diff_plain:+.1f}%)')
    ax_zoom.plot(t_z, e_fast[z_s:z_e], color='#000000', linewidth=1.2, linestyle='--', marker='s', markersize=3.0, label=f'Fast $\\mu=0.02$ ({diff_fast:+.1f}%)')
    ax_zoom.plot(t_z, e_clamp[z_s:z_e], color='#000000', linewidth=1.4, linestyle='-', marker='o', markersize=3.5, label=f'Clamped $\\mu=0.002$ ({diff_clamp:+.1f}%)')
    ax_zoom.set_ylabel('Ampl. (mV)', fontsize=8.8)
    ax_zoom.set_title(r'(e) R-Peak Apex ($\pm 45$ ms)', loc='left', fontweight='bold', fontsize=8.2, pad=3)
    ax_zoom.set_ylim(-0.25, 2.15)
    ax_zoom.grid(True, linestyle=':', alpha=0.5)
    ax_zoom.legend(loc='upper center', fontsize=7.2, frameon=True, framealpha=0.95, edgecolor='#cbd5e1')

    ratio_seg = ratio_full[z_s:z_e]

    ax_ratio.plot(t_z, ratio_seg, color='#000000', linewidth=1.2, linestyle='-', label=r'Ratio $\mathcal{A}(n)$')
    ax_ratio.axhline(3.0, color='#64748b', linewidth=1.0, linestyle='--', label=r'Threshold $\Theta = 3.0$')
    ax_ratio.fill_between(t_z, 0, ratio_seg, where=(ratio_seg >= 3.0), color='#e2e8f0', alpha=0.8, label='Clamped Hold')
    ax_ratio.set_ylabel(r'Ratio $\mathcal{A}(n)$', fontsize=8.8)
    ax_ratio.set_xlabel('Relative Time (ms)', fontsize=8.8)
    ax_ratio.set_title(r'(f) Ratio Comparator $\mathcal{A}(n)$', loc='left', fontweight='bold', fontsize=8.2, pad=3)
    ax_ratio.set_ylim(0, 8.6)
    ax_ratio.grid(True, linestyle=':', alpha=0.5)
    ax_ratio.legend(loc='upper right', fontsize=7.4, frameon=True, framealpha=0.95, edgecolor='#cbd5e1')

    fig.align_ylabels([ax0, ax1, ax2, ax3])
    fig.align_ylabels([ax_zoom, ax_ratio])
    fig.subplots_adjust(top=0.96, bottom=0.07, left=0.10, right=0.99, hspace=0.42, wspace=0.32)

    plt.savefig(os.path.join(output_dir, 'waveform_comparison.pdf'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(output_dir, 'waveform_comparison.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 4.4), dpi=300, sharex=True)

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
               ncol=4, fontsize=8.4, frameon=True, framealpha=0.95, edgecolor='#94a3b8')
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout(rect=[0, 0, 1, 0.92], pad=0.5)

    plt.savefig(os.path.join(output_dir, 'ablation_breakdown.pdf'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(output_dir, 'ablation_breakdown.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
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
