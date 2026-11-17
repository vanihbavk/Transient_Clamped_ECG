import numpy as np

CLAMPED = 1
ADAPTING = 2
STILL = 3

DEFAULT_THRESHOLDS = {
    'amplitude': 3.0,
    'first_difference': 0.7,
    'curvature': 0.38,
}


class ClampTrigger:
    def __init__(self, mode='amplitude', theta=None, hold=8, alpha=1.0 / 16.0, eps=1e-4, initial_envelope=0.5):
        if mode not in DEFAULT_THRESHOLDS:
            raise ValueError(f'unknown trigger mode: {mode}')
        self.mode = mode
        self.theta = DEFAULT_THRESHOLDS[mode] if theta is None else theta
        self.hold = hold
        self.alpha = alpha
        self.eps = eps
        self.initial_envelope = initial_envelope
        self.reset()

    def reset(self):
        self.envelope = self.initial_envelope
        self.history = np.zeros(5)
        self.counter = 0

    def step(self, value):
        self.envelope = (1.0 - self.alpha) * self.envelope + self.alpha * abs(value)
        self.history[1:] = self.history[:-1]
        self.history[0] = value
        if self.mode == 'amplitude':
            numerator = abs(value)
        elif self.mode == 'first_difference':
            numerator = abs(self.history[0] - self.history[1])
        else:
            numerator = abs(0.25 * self.history[0] - 0.5 * self.history[2] + 0.25 * self.history[4])
        ratio = numerator / (self.envelope + self.eps)
        if ratio >= self.theta:
            self.counter = self.hold
        clamped = self.counter > 0
        if clamped:
            self.counter -= 1
        return ratio, clamped


def run_filter(rule, d, x_ref, taps=16, eps=1e-4):
    n_samples = len(d)
    w = np.zeros(taps)
    x_buf = np.zeros(taps)
    e = np.zeros(n_samples)
    mu_hist = np.zeros(n_samples)
    phase_hist = np.zeros(n_samples, dtype=int)
    rule.reset(taps)
    for n in range(n_samples):
        x_buf[1:] = x_buf[:-1]
        x_buf[0] = x_ref[n]
        norm_x2 = np.dot(x_buf, x_buf) + eps
        e_n = d[n] - np.dot(w, x_buf)
        e[n] = e_n
        mu_hist[n], phase_hist[n] = rule.update(n, w, x_buf, norm_x2, e_n, d[n])
    return e, mu_hist, phase_hist


class UpdateRule:
    def reset(self, taps):
        pass


class FixedStepNLMS(UpdateRule):
    def __init__(self, mu=0.002):
        self.mu = mu

    def update(self, n, w, x_buf, norm_x2, e_n, d_n):
        w += self.mu / norm_x2 * e_n * x_buf
        return self.mu, ADAPTING


class TriggerClampedNLMS(UpdateRule):
    def __init__(self, mu=0.002, mu_min=1e-4, trigger='amplitude', theta=None, hold=8, source='primary'):
        self.mu = mu
        self.mu_min = mu_min
        self.trigger = ClampTrigger(trigger, theta=theta, hold=hold)
        self.source = source

    def reset(self, taps):
        self.trigger.reset()

    def update(self, n, w, x_buf, norm_x2, e_n, d_n):
        _, clamped = self.trigger.step(d_n if self.source == 'primary' else e_n)
        mu = self.mu_min if clamped else self.mu
        w += mu / norm_x2 * e_n * x_buf
        return mu, CLAMPED if clamped else ADAPTING


class MaskClampedNLMS(UpdateRule):
    def __init__(self, mask, mu=0.002, mu_min=1e-4):
        self.mask = mask
        self.mu = mu
        self.mu_min = mu_min

    def update(self, n, w, x_buf, norm_x2, e_n, d_n):
        clamped = self.mask[n]
        mu = self.mu_min if clamped else self.mu
        w += mu / norm_x2 * e_n * x_buf
        return mu, CLAMPED if clamped else ADAPTING


class KwongJohnstonVSS(UpdateRule):
    def __init__(self, mu_init=0.02, alpha=0.97, gamma=0.001, mu_min=1e-4, mu_max=0.08):
        self.mu_init = mu_init
        self.alpha = alpha
        self.gamma = gamma
        self.mu_min = mu_min
        self.mu_max = mu_max

    def reset(self, taps):
        self.mu = self.mu_init

    def update(self, n, w, x_buf, norm_x2, e_n, d_n):
        self.mu = np.clip(self.alpha * self.mu + self.gamma * e_n ** 2, self.mu_min, self.mu_max)
        w += self.mu / norm_x2 * e_n * x_buf
        return self.mu, ADAPTING


class AboulnasrMayyasVSS(UpdateRule):
    def __init__(self, mu_init=0.02, alpha=0.97, beta=0.95, gamma=0.001, mu_min=1e-4, mu_max=0.08):
        self.mu_init = mu_init
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu_min = mu_min
        self.mu_max = mu_max

    def reset(self, taps):
        self.mu = self.mu_init
        self.correlation = 0.0
        self.e_prev = 0.0

    def update(self, n, w, x_buf, norm_x2, e_n, d_n):
        self.correlation = self.beta * self.correlation + (1.0 - self.beta) * e_n * self.e_prev
        self.e_prev = e_n
        self.mu = np.clip(self.alpha * self.mu + self.gamma * self.correlation ** 2, self.mu_min, self.mu_max)
        w += self.mu / norm_x2 * e_n * x_buf
        return self.mu, ADAPTING


class HuberNLMS(UpdateRule):
    def __init__(self, mu=0.002, delta=0.5, adaptive=False, k=1.4, alpha=1.0 / 16.0, initial_envelope=0.35):
        self.mu = mu
        self.delta = delta
        self.adaptive = adaptive
        self.k = k
        self.alpha = alpha
        self.initial_envelope = initial_envelope

    def reset(self, taps):
        self.envelope = self.initial_envelope

    def update(self, n, w, x_buf, norm_x2, e_n, d_n):
        if self.adaptive:
            self.envelope = (1.0 - self.alpha) * self.envelope + self.alpha * abs(d_n)
            delta = self.k * self.envelope
        else:
            delta = self.delta
        w += self.mu / norm_x2 * np.clip(e_n, -delta, delta) * x_buf
        return self.mu, ADAPTING


class SetMembershipNLMS(UpdateRule):
    def __init__(self, gamma=0.2):
        self.gamma = gamma

    def update(self, n, w, x_buf, norm_x2, e_n, d_n):
        abs_e = abs(e_n)
        if abs_e <= self.gamma:
            return 0.0, CLAMPED
        w += (1.0 - self.gamma / abs_e) / norm_x2 * e_n * x_buf
        return 1.0 - self.gamma / abs_e, ADAPTING


class ImuGatedNLMS(UpdateRule):
    def __init__(self, x_imu, mu0=0.002, mu_min=1e-4, beta=0.1, gamma_still=0.02, window=16,
                 trigger=None, theta=None, hold=8, source='primary'):
        self.x_imu = x_imu
        self.mu0 = mu0
        self.mu_min = mu_min
        self.beta = beta
        self.gamma_still = gamma_still
        self.window = window
        self.trigger = None if trigger is None else ClampTrigger(trigger, theta=theta, hold=hold)
        self.source = source

    def reset(self, taps):
        self.imu_buf = np.zeros(self.window)
        if self.trigger is not None:
            self.trigger.reset()

    def update(self, n, w, x_buf, norm_x2, e_n, d_n):
        self.imu_buf[1:] = self.imu_buf[:-1]
        self.imu_buf[0] = self.x_imu[n] if n < len(self.x_imu) else 0.0
        variance = np.var(self.imu_buf)
        clamped = False
        if self.trigger is not None:
            _, clamped = self.trigger.step(d_n if self.source == 'primary' else e_n)
        if clamped:
            mu, phase = self.mu_min, CLAMPED
        elif variance >= self.gamma_still:
            mu, phase = self.mu0 * variance / (variance + self.beta), ADAPTING
        else:
            mu, phase = self.mu_min, STILL
        w += mu / norm_x2 * e_n * x_buf
        return mu, phase


class RLS(UpdateRule):
    def __init__(self, lam=0.998, delta=0.1, trigger=None, theta=None, hold=8, mask=None):
        self.lam = lam
        self.delta = delta
        self.trigger = None if trigger is None else ClampTrigger(trigger, theta=theta, hold=hold)
        self.mask = mask

    def reset(self, taps):
        self.taps = taps
        self.P = np.eye(taps) / self.delta
        if self.trigger is not None:
            self.trigger.reset()

    def update(self, n, w, x_buf, norm_x2, e_n, d_n):
        clamped = False
        if self.trigger is not None:
            _, clamped = self.trigger.step(d_n)
        elif self.mask is not None:
            clamped = self.mask[n]
        if not clamped:
            Px = np.dot(self.P, x_buf)
            gain = Px / (self.lam + np.dot(x_buf, Px))
            w += gain * e_n
            self.P = (self.P - np.outer(gain, np.dot(x_buf, self.P))) / self.lam
        return float(np.trace(self.P) / self.taps), CLAMPED if clamped else ADAPTING
