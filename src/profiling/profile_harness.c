#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define FILTER_TAPS 16
#define NUM_ITERATIONS 100000
#define N_SAMPLES 10000
#define SAMPLE_RATE 250.0f

#define CYCLES_FIR         (16 * 2 + 2)
#define CYCLES_ENERGY      6
#define CYCLES_ENVELOPE    5
#define CYCLES_COMPARATOR  4
#define CYCLES_VDIV        14
#define CYCLES_TAP_UPDATE  (16 * 2 + 2)

typedef struct {
    float w[FILTER_TAPS];
    float x_buf[FILTER_TAPS];
    float sigma_d;
    float norm_x2;
    int   hold_counter;
    float mu_nominal;
    float mu_min;
    float theta_amp;
    int   hold_samples;
    float delta;
} ClampedNLMS_F32;

static inline int32_t update_envelope_q15_bitshift(int32_t sigma_q15, int32_t abs_d_q15) {
    return sigma_q15 + ((abs_d_q15 - sigma_q15) >> 4);
}

void clamped_nlms_f32_init(ClampedNLMS_F32 *f) {
    memset(f, 0, sizeof(ClampedNLMS_F32));
    f->sigma_d      = 0.5f;
    f->norm_x2      = 0.0f;
    f->delta        = 1e-4f;
    f->theta_amp    = 3.0f;
    f->hold_samples = 8;
    f->mu_min       = 1e-4f;
    f->mu_nominal   = 0.002f;
}

static inline float clamped_nlms_f32_process_sample(ClampedNLMS_F32 *f,
                                                    float d_n, float x_n,
                                                    int *is_clamped_out) {
    float old_x = f->x_buf[FILTER_TAPS - 1];

    for (int i = FILTER_TAPS - 1; i > 0; i--) {
        f->x_buf[i] = f->x_buf[i - 1];
    }
    f->x_buf[0] = x_n;

    f->norm_x2 += (x_n * x_n) - (old_x * old_x);
    if (f->norm_x2 < 0.0f) f->norm_x2 = 0.0f;

    float y_n = 0.0f;
    for (int i = 0; i < FILTER_TAPS; i++) {
        y_n += f->w[i] * f->x_buf[i];
    }

    float e_n = d_n - y_n;

    float abs_d = fabsf(d_n);
    f->sigma_d += (abs_d - f->sigma_d) * 0.0625f;

    if (abs_d >= f->theta_amp * (f->sigma_d + f->delta)) {
        f->hold_counter = f->hold_samples;
    }

    float mu_n;
    int is_clamped = 0;
    if (f->hold_counter > 0) {
        mu_n = f->mu_min;
        is_clamped = 1;
        f->hold_counter--;
    } else {
        mu_n = f->mu_nominal;
        is_clamped = 0;
    }

    float scale = (mu_n / (f->norm_x2 + f->delta)) * e_n;
    for (int i = 0; i < FILTER_TAPS; i++) {
        f->w[i] += scale * f->x_buf[i];
    }

    if (is_clamped_out) *is_clamped_out = is_clamped;
    return e_n;
}

static double get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

int main(void) {
    ClampedNLMS_F32 filter;
    clamped_nlms_f32_init(&filter);

    int test_samples = N_SAMPLES;
    int clamp_count = 0;
    int dummy_clamped = 0;

    int32_t sigma_q15 = (int32_t)(0.25f * 32768.0f);
    float   sigma_f32 = 0.25f;
    float   max_q15_error = 0.0f;
    float   max_single_step_q15_error = 0.0f;
    float   max_energy_error = 0.0f;

    for (int i = 0; i < test_samples; i++) {
        int beat_phase = i % 250;
        float ecg_clean = 0.0f;
        if (beat_phase >= 0 && beat_phase < 4) {
            ecg_clean = (float)beat_phase * 0.375f;
        } else if (beat_phase >= 4 && beat_phase < 8) {
            ecg_clean = 1.5f - (float)(beat_phase - 4) * 0.45f;
        } else if (beat_phase >= 8 && beat_phase < 12) {
            ecg_clean = -0.3f + (float)(beat_phase - 8) * 0.075f;
        } else if (beat_phase >= 30 && beat_phase < 70) {
            ecg_clean = 0.20f * sinf(3.14159f * (float)(beat_phase - 30) / 40.0f);
        }

        float x_ref = 0.35f * sinf(0.05f * (float)i);
        float art_d = 0.35f * sinf(0.05f * (float)i + 0.2f);
        float d_in = ecg_clean + art_d;

        int is_c = 0;
        clamped_nlms_f32_process_sample(&filter, d_in, x_ref, &is_c);
        if (is_c) clamp_count++;

        float direct_energy = 0.0f;
        for (int k = 0; k < FILTER_TAPS; k++) {
            direct_energy += filter.x_buf[k] * filter.x_buf[k];
        }
        float energy_err = fabsf(filter.norm_x2 - direct_energy);
        if (energy_err > max_energy_error) max_energy_error = energy_err;

        float abs_d_f = 0.5f * fabsf(d_in);
        int32_t abs_d_q15 = (int32_t)(abs_d_f * 32768.0f);
        if (abs_d_q15 > 32767) abs_d_q15 = 32767;
        if (abs_d_q15 < 0) abs_d_q15 = 0;

        int32_t diff_q15 = abs_d_q15 - sigma_q15;
        int32_t next_sigma_q15 = update_envelope_q15_bitshift(sigma_q15, abs_d_q15);
        int32_t shift_step = next_sigma_q15 - sigma_q15;
        int32_t exact_div_step = (int32_t)((float)diff_q15 * (1.0f / 16.0f));
        float single_step_err = fabsf((float)(shift_step - exact_div_step)) / 32768.0f;
        if (single_step_err > max_single_step_q15_error) {
            max_single_step_q15_error = single_step_err;
        }

        sigma_q15 = next_sigma_q15;
        sigma_f32 += (abs_d_f - sigma_f32) * 0.0625f;

        float q15_as_float = (float)sigma_q15 / 32768.0f;
        float err = fabsf(q15_as_float - sigma_f32);
        if (err > max_q15_error) max_q15_error = err;
    }

    int q15_verified = (max_q15_error < 0.0005f && max_single_step_q15_error <= 3.1e-5f) ? 1 : 0;
    int energy_verified = (max_energy_error < 1e-4f) ? 1 : 0;

    double t_start = get_time_ns();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        float d_in = 0.8f * sinf(0.05f * (float)i) + 0.2f * cosf(0.02f * (float)i);
        float x_in = 0.3f * sinf(0.05f * (float)i);
        clamped_nlms_f32_process_sample(&filter, d_in, x_in, &dummy_clamped);
    }
    double t_end = get_time_ns();
    double ns_per_sample = (t_end - t_start) / NUM_ITERATIONS;

    int macs_direct_3m     = 48;
    int muls_recursive_o1  = 35;
    int cycles_fir         = CYCLES_FIR;
    int cycles_energy      = CYCLES_ENERGY;
    int cycles_envelope    = CYCLES_ENVELOPE;
    int cycles_comparator  = CYCLES_COMPARATOR;
    int cycles_vdiv        = CYCLES_VDIV;
    int cycles_tap_update  = CYCLES_TAP_UPDATE;
    int cycles_m4f = cycles_fir + cycles_energy + cycles_envelope
                   + cycles_comparator + cycles_vdiv + cycles_tap_update;
    double us_m4f_64mhz = (double)cycles_m4f / 64.0;
    double cpu_duty_pct = (us_m4f_64mhz / 4000.0) * 100.0;
    float clamp_duty_pct = (float)clamp_count / (float)test_samples * 100.0f;

    printf("Validation: samples=%d, clamp_duty=%.1f%%, cycles_per_sample=%d, "
           "exec_time_us=%.2f, cpu_util_250hz=%.3f%%\n",
           test_samples, clamp_duty_pct, cycles_m4f, us_m4f_64mhz, cpu_duty_pct);
    printf("Q15 bit-shift envelope verification: max_error=%.6f, max_single_step_error=%.6f, verified=%s\n",
           max_q15_error, max_single_step_q15_error, q15_verified ? "true" : "false");

    FILE *fp = fopen("docs/profiling_results.json", "w");
    if (!fp) fp = fopen("../docs/profiling_results.json", "w");
    if (!fp) fp = fopen("../../docs/profiling_results.json", "w");
    if (fp) {
        fprintf(fp, "{\n");
        fprintf(fp, "  \"cycle_estimation_method\": \"modeled_instruction_counts_analytical\",\n");
        fprintf(fp, "  \"cycle_disclaimer\": \"Cycles are modeled analytical instruction-level estimates from ARM Cortex-M4 technical reference manuals summing constituent kernels.\",\n");
        fprintf(fp, "  \"macs_direct_3m\": %d,\n", macs_direct_3m);
        fprintf(fp, "  \"muls_recursive_o1\": %d,\n", muls_recursive_o1);
        fprintf(fp, "  \"cycles_fir\": %d,\n", cycles_fir);
        fprintf(fp, "  \"cycles_energy\": %d,\n", cycles_energy);
        fprintf(fp, "  \"cycles_envelope\": %d,\n", cycles_envelope);
        fprintf(fp, "  \"cycles_comparator\": %d,\n", cycles_comparator);
        fprintf(fp, "  \"cycles_vdiv\": %d,\n", cycles_vdiv);
        fprintf(fp, "  \"cycles_tap_update\": %d,\n", cycles_tap_update);
        fprintf(fp, "  \"float32_cycles_m4\": %d,\n", cycles_m4f);
        fprintf(fp, "  \"float32_exec_time_us\": %.2f,\n", us_m4f_64mhz);
        fprintf(fp, "  \"float32_m4_cpu_load_pct\": %.3f,\n", cpu_duty_pct);
        fprintf(fp, "  \"float32_ram_bytes\": %zu,\n", sizeof(ClampedNLMS_F32));
        fprintf(fp, "  \"host_latency_ns_f32\": %.1f,\n", ns_per_sample);
        fprintf(fp, "  \"clamp_duty_validation_pct\": %.1f,\n", clamp_duty_pct);
        fprintf(fp, "  \"recursive_energy_max_error\": %.8f,\n", max_energy_error);
        fprintf(fp, "  \"recursive_energy_verified\": %s,\n", energy_verified ? "true" : "false");
        fprintf(fp, "  \"q15_bitshift_max_single_step_error\": %.6f,\n", max_single_step_q15_error);
        fprintf(fp, "  \"q15_bitshift_max_error\": %.6f,\n", max_q15_error);
        fprintf(fp, "  \"q15_bitshift_verified\": %s\n", q15_verified ? "true" : "false");
        fprintf(fp, "}\n");
        fclose(fp);
    }

    return 0;
}
