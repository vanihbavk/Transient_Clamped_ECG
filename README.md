# Zero-Lookahead Amplitude-Clamped Adaptive Filtering for Ambulatory ECG Telemetry

**Open-Source DSP Implementation, Clinical Benchmark Suite & Embedded Firmware Profiler**  
**Author:** Abhinav Varma Kosuri  
**Preprint (Zenodo):** [https://doi.org/10.5281/zenodo.24303181](https://doi.org/10.5281/zenodo.24303181)

---

## 1. Overview

This repository provides the complete, self-contained source code, benchmark evaluation suite, and embedded C99 firmware profiling harness for the preprint:

> **"Zero-Lookahead Amplitude-Clamped Adaptive Filtering for Ambulatory Electrocardiogram Telemetry"**  
> Abhinav Varma Kosuri — [DOI: 10.5281/zenodo.24303181](https://doi.org/10.5281/zenodo.24303181)

### Key Contribution
Standard normalized least mean squares (NLMS) adaptive filters distort diagnostic QRS complexes via gradient leakage, attenuating clinical $R$-peak amplitudes by $3.4\%$ on average across the dataset (and up to $12.7\%$ during rapid transients at $\mu=0.02$, exceeding $13.9\%$ on the worst $5\%$ of depolarizations). Conventional beat-gated remedies introduce 150–200 ms FIFO buffer latency to compensate for QRS detector group delay. 

This work introduces a **causal, zero-lookahead amplitude-clamped NLMS filter** that eliminates gradient leakage without buffer latency or auxiliary sensor reliance:
- **Bit-shift envelope tracker:** Causal running envelope tracker using power-of-two decay ($\lambda = 1/16$, implemented via arithmetic bit-shift `>> 4` in Q15 fixed-point or `* 0.0625f` in single-precision FPU).
- **Self-normalizing thresholding:** Clamps tap adaptation ($\mu(n) = 10^{-4}$) during steep ventricular depolarizations when $\mathcal{A}(n) \ge \Theta_{\text{amp}} = 3.0$ for an 8-sample hold interval (32 ms at 250 Hz).
- **Embedded efficiency:** Requires only 48 direct MAC operations per sample (reduced to 34 MACs / 35 FPU multiplications in C99 firmware via $O(1)$ recursive sliding-window energy tracking), modeled analytically at **97 cycles on ARM Cortex-M4F** ($1.52\ \mu\text{s}$ at 64 MHz, $0.038\%$ CPU utilization at 250 Hz) and occupying **160 bytes of static RAM**.
- **Clinical validation:** Evaluated across 28 non-overlapping 60-second segments from the PhysioNet MIT-BIH Arrhythmia Database (`mitdb`) and Noise Stress Test Database (`nstdb`) at $-6, 0, +6, +12$ dB input SNR, plus non-stationary coupling transitions and five physical stress regimes.

---

## 2. Repository Structure

```text
.
├── Makefile                                    # Top-level build and evaluation targets
├── README.md                                   # Documentation and usage guide
├── requirements.txt                            # Python dependencies
├── LICENSE                                     # MIT License
│
├── src/
│   ├── dsp/                                    # Core Signal Processing Implementations
│   │   ├── adaptive_filters.py                 # Clamped NLMS, plain NLMS, VSS, Huber, SM-NLMS, IMU-gated, RLS
│   │   ├── gating.py                           # Annotation-gated, random/shifted controls, DePinto & Pan-Tompkins masks
│   │   └── qrs_detection.py                    # Pan-Tompkins QRS detector
│   ├── benchmarks/                             # Clinical Benchmarking & Evaluation Suite
│   │   ├── common.py                           # Auto-downloading dataset loaders, trial builders, and metrics
│   │   ├── run_benchmarks.py                   # 28-segment MIT-BIH + NSTDB evaluation & parameter sweeps
│   │   ├── run_dynamic_benchmarks.py           # Multi-sensor IMU & non-stationary coupling transitions
│   │   ├── evaluate_robust_baselines.py        # Baseline comparisons across noise types & gain mismatch
│   │   ├── generate_figures.py                 # Vector and raster figure generation
│   │   └── generate_paper_macros.py            # Export of benchmark metrics to LaTeX macros
│   └── profiling/                              # Embedded Firmware Execution Profiler
│       ├── profile_harness.c                   # C99 bare-metal implementation, verification & cycle model
│       └── Makefile                            # C harness build file (gcc/clang)
├── docs/                                       # Precomputed Benchmark & Profiling Outputs (JSON & TeX macros)
│   ├── benchmark_results.json
│   ├── benchmark_raw_trials.json
│   ├── dynamic_benchmark_results.json
│   ├── robust_baselines_results.json
│   ├── mu_sweep_results.json
│   ├── rls_lambda_sweep.json
│   ├── trigger_detection_results.json
│   ├── blunting_results.json
│   ├── profiling_results.json
│   └── results_macros.tex
└── figures/                                    # Generated Figures (PDF & PNG)
    ├── system_block_diagram.pdf                # Figure 1: DSP signal flow & zero-lookahead clamp
    ├── waveform_comparison.pdf                 # Figure 2: Denoising waveforms & R-peak fidelity
    └── ablation_breakdown.pdf                  # Figure 3: Performance trade-offs & ablation
```

*(Note: Required PhysioNet `mitdb` and `nstdb` records are automatically downloaded into `data/` via `wfdb` on first execution if not already cached locally.)*

---

## 3. Quick Start

### Prerequisites
- **Python**: 3.9+
- **C Compiler**: `gcc` or `clang` (C99 compatible)

### Step 1: Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Run Embedded Firmware Profiling
Compiles and runs the bare-metal C99 harness (`src/profiling/profile_harness.c`), verifying the `Q15` bit-shift envelope against `float32`, checking the $O(1)$ recursive sliding-window energy tracker against direct dot-product energy, and computing the analytical ARM Cortex-M4F instruction cycle model:
```bash
make profile
```
**Output:**
```text
Validation: samples=10000, clamp_duty=3.4%, cycles_per_sample=97, exec_time_us=1.52, cpu_util_250hz=0.038%
Q15 bit-shift envelope verification: max_error=0.000332, max_single_step_error=0.000031, verified=true
```

### Step 3: Run Clinical Benchmarks
Executes the evaluation pipeline across all 28 MIT-BIH and NSTDB segments across SNR regimes, non-stationary coupling transitions, and stress scenarios:
```bash
make benchmarks
```
This updates the JSON result files and `results_macros.tex` in `docs/`.

### Step 4: Regenerate Figures
Generates the figures in vector PDF and PNG formats into `figures/`:
```bash
make figures
```

---

## 4. Key Quantitative Metrics Verified (0 dB Input SNR, N=28)

| Metric | Clamped NLMS (Ours) | Plain NLMS ($\mu=0.002$) | Random Clamp Control | Shifted Window Control | DePinto Slope Gate | Pan-Tompkins (Buffered) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\Delta\text{SNR}$ (dB)** | **$+14.15 \pm 1.08$** | $+12.84 \pm 1.40$ | $+12.76 \pm 1.37$ | $+12.86 \pm 1.45$ | $+13.84 \pm 1.14$ | $+13.98 \pm 1.18$ |
| **$R$-Peak Preservation ($\rho_R$)** | **$99.8\%$** | $99.6\%$ | $99.6\%$ | $99.6\%$ | $99.8\%$ | $99.8\%$ |
| **Peak Absolute Error ($\text{MAE}_R$)** | **$2.45\%$** | $2.67\%$ | $2.69\%$ | $2.67\%$ | $2.54\%$ | $2.52\%$ |
| **QRS Correlation ($r_{\text{QRS}}$)** | **$0.9992$** | $0.9990$ | $0.9990$ | $0.9990$ | $0.9991$ | $0.9992$ |
| **Non-QRS Isoelectric $\text{SNR}_{\text{iso}}$ (dB)** | **$+7.03$** | $+5.67$ | $+5.60$ | $+5.68$ | $+6.74$ | $+6.87$ |
| **Clamp Duty Cycle** | **$4.5\%$** | --- | $4.4\%$ | $4.1\%$ | $15.0\%$ | $10.6\%$ |
| **Telemetry Buffer Latency** | **$0\ \text{ms}$** | $0\ \text{ms}$ | $0\ \text{ms}$ | $0\ \text{ms}$ | $46\ \text{ms}$ | $150\ \text{ms}$ |
| **Direct Vector Cost** | **$48\ \text{MACs}$** ($34$ recursive) | $48\ \text{MACs}$ | $48\ \text{MACs}$ | $48\ \text{MACs}$ | $48\ \text{MACs}$ | $48\ \text{MACs} + \text{QRS bank}$ |

---

## 5. License

This project is released under the **MIT License** (see [LICENSE](LICENSE)).
