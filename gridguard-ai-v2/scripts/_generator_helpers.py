"""Shared synthetic-data generators for GridGuard AI v2.

Used by the dataset generator (step 1) and the manual-scenario tester
(step 7) so that both use exactly the same behavioural model.

IMPORTANT: identical draw order as step 1; reproducible with seed 42.
"""
import numpy as np

from config import START_TEMP_MAX, START_TEMP_MIN, WINDOW_LEN

W = WINDOW_LEN


def gen_baseline(rng, start_temp):
    """20 samples around a random environmental temperature."""
    trend = rng.uniform(-0.25, 0.25)
    noise_sigma = rng.uniform(0.03, 0.15)
    t = np.linspace(0.0, 1.0, W)
    values = start_temp + trend * t
    values = values + rng.normal(0.0, noise_sigma, W)
    return values


def gen_normal(rng, start_temp):
    """Stable or naturally varying temperature, staying near the baseline."""
    baseline = gen_baseline(rng, start_temp)
    dev = np.zeros(W)
    sigma = rng.uniform(0.03, 0.10)
    dev[0] = rng.uniform(-0.25, 0.25)
    for i in range(1, W):
        dev[i] = np.clip(dev[i - 1] + rng.normal(0.0, sigma, 1)[0], -0.55, 0.55)
    if rng.random() < 0.35:
        dev = dev + rng.uniform(0.0, 0.9) * np.linspace(0.0, 1.0, W)
    noise = rng.normal(0.0, rng.uniform(0.03, 0.12), W)
    return baseline, baseline + dev + noise


def gen_rising(rng, start_temp):
    """Smooth gradual upward trend, relatively low curvature/noise."""
    baseline = gen_baseline(rng, start_temp)
    slope = rng.uniform(0.09, 0.24)
    curvature = rng.uniform(-0.008, 0.008)
    idx = np.arange(W)
    rise = slope * idx + curvature * idx * (idx - (W - 1))
    noise = rng.normal(0.0, rng.uniform(0.02, 0.08), W)
    return baseline, baseline[-1] + rise + noise


def gen_drift(rng, start_temp):
    """Slow persistent movement away from the learned baseline."""
    baseline = gen_baseline(rng, start_temp)
    direction = 1.0 if rng.random() < 0.70 else -1.0
    displacement = direction * rng.uniform(1.8, 6.5)
    power = rng.uniform(0.7, 1.6)
    progress = np.linspace(0.0, 1.0, W) ** power
    profile = displacement * progress
    wobble = np.cumsum(rng.normal(0.0, rng.uniform(0.02, 0.09), W))
    wobble = wobble - wobble * np.linspace(0.0, 1.0, W) ** 2
    noise = rng.normal(0.0, rng.uniform(0.04, 0.14), W)
    return baseline, baseline[-1] + profile + wobble + noise


def gen_rapid(rng, start_temp):
    """Fast abnormal rise with randomised shape, rate and onset index."""
    baseline = gen_baseline(rng, start_temp)
    shape = rng.choice(["linear", "accelerating", "exponential", "delayed"])
    idx = np.arange(W)
    noise = rng.normal(0.0, rng.uniform(0.04, 0.15), W)

    if shape == "linear":
        rate = rng.uniform(0.28, 0.80)
        profile = rate * idx
    elif shape == "accelerating":
        final_rise = rng.uniform(5.0, 16.0)
        k = final_rise / (W - 1) ** 2
        profile = k * idx ** 2
    elif shape == "exponential":
        final_rise = rng.uniform(5.0, 16.0)
        tc = rng.uniform(2.0, 5.0)
        profile = final_rise * (np.exp(idx / tc) - 1.0) / (np.exp((W - 1) / tc) - 1.0)
    else:
        onset = rng.integers(4, 14)
        rate = rng.uniform(0.35, 0.90)
        profile = np.where(idx >= onset, rate * (idx - onset), 0.0)

    return baseline, baseline[-1] + profile + noise


def gen_instability(rng, start_temp):
    """Irregular unstable behaviour: incommensurate oscillations, spikes,
    random amplitudes/frequencies - never a clean periodic sine."""
    baseline = gen_baseline(rng, start_temp)
    amp = rng.uniform(0.5, 2.6)
    t = np.arange(W)
    osc = np.zeros(W)
    for _ in range(int(rng.integers(2, 4))):
        freq = rng.uniform(0.10, 0.45)
        a = rng.uniform(0.2, 1.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        osc = osc + a * np.sin(2.0 * np.pi * freq * t + phase)
    peak = np.max(np.abs(osc))
    if peak > 1e-9:
        osc = osc * (amp / peak)
    for _ in range(int(rng.integers(0, 4))):
        i = int(rng.integers(0, W))
        osc[i] += rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.8)
    osc = osc + rng.uniform(-1.0, 1.0) * np.linspace(0.0, 1.0, W)
    noise = rng.normal(0.0, rng.uniform(0.05, 0.18), W)
    return baseline, baseline + osc + noise


GENERATORS = {
    0: ("NORMAL_OPERATION", gen_normal),
    1: ("TEMPERATURE_RISING", gen_rising),
    2: ("THERMAL_DRIFT", gen_drift),
    3: ("RAPID_TEMPERATURE_RISE", gen_rapid),
    4: ("THERMAL_INSTABILITY", gen_instability),
}


def generate_class_examples(rng, class_id, n, start_range=(START_TEMP_MIN, START_TEMP_MAX)):
    rows = []
    for _ in range(n):
        start_temp = rng.uniform(*start_range)
        baseline = gen_baseline(rng, start_temp)
        _, current = GENERATORS[class_id][1](rng, start_temp)
        rows.append(np.concatenate([baseline, current]))
    return np.asarray(rows)