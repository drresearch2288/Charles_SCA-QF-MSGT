"""Synthetic 12-lead cohort for smoke tests and the synthetic fallback of Section 3.3.4.

Records are built with NeuroKit2 (realistic P-QRS-T morphology). Pre-arrest ECGs get
lower heart-rate variability and a growing ST-T change as the arrest approaches, so the
pipeline has something to learn. These data are not clinical and must never be used to
report results."""
import os

import numpy as np
import pandas as pd


def _ecg12(duration, fs, hr, hr_std, st_shift, rng):
    import neurokit2 as nk
    df = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=hr, heart_rate_std=hr_std,
                         method="multileads", random_state=int(rng.integers(1e9)))
    sig = df.to_numpy().T                                   # (12, T) in I..V6 order
    if st_shift:
        t = np.arange(sig.shape[1]) / fs
        sig = sig + st_shift * (0.5 + 0.5 * np.sin(2 * np.pi * hr / 60 * t))[None] * rng.uniform(0.5, 1.5, (12, 1))
    return sig + rng.normal(0, 0.01, sig.shape)


def make_cohort(out_dir, n_patients=60, arrest_frac=0.3, ecgs_per_patient=4, fs=500, duration=10,
                sources=("SCAOS", "MIMIC-IV"), seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for src in sources:
        for p in range(n_patients):
            arrest = p < int(arrest_frac * n_patients)
            pid = "%s_%04d" % (src[:2], p)
            t_arrest = 48.0 if arrest else np.nan
            times = np.sort(rng.uniform(30.0, 47.9, ecgs_per_patient)) if arrest else np.sort(rng.uniform(0, 48, ecgs_per_patient))
            for k, t in enumerate(times):
                if arrest:
                    closeness = np.clip(1 - (t_arrest - t) / 18.0, 0, 1)
                    hr, hr_std, st = 70 + 25 * closeness, max(0.5, 3 - 2.5 * closeness), 0.15 * closeness
                else:
                    hr, hr_std, st = rng.uniform(60, 85), rng.uniform(2.5, 5), 0.0
                sig = _ecg12(duration, fs, hr, hr_std, st, rng)
                rel = os.path.join("records", "%s_%d.npy" % (pid, k))
                os.makedirs(os.path.join(out_dir, "records"), exist_ok=True)
                np.save(os.path.join(out_dir, rel), sig.astype(np.float32))
                rows.append(dict(source=src, patient_id=pid, ecg_id="%s_%d" % (pid, k), path=rel, fs=fs,
                                 ecg_time_h=round(float(t), 3), arrest_time_h=t_arrest))
    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    return man
