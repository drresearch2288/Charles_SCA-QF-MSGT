"""Metrics and alarm rules (Section 3.12) plus the patient-level bootstrap (Section 4.7)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score


def best_f1_threshold(y, s):
    """Operating threshold delta: maximum F1 on the validation set (Table 4)."""
    o = np.argsort(-s); ys, ss = y[o], s[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    f1 = 2 * tp / (tp + fp + P)
    k = int(np.argmax(f1))
    return float(ss[k] - 1e-9) if k < len(ss) else 0.5


def ecg_metrics(y, s, delta):
    pred = (s > delta).astype(int)
    out = dict(AUPRC=average_precision_score(y, s) if y.any() else np.nan,
               F1=f1_score(y, pred, zero_division=0), Recall=recall_score(y, pred, zero_division=0))
    try:
        out["AUROC"] = roc_auc_score(y, s)
    except ValueError:
        out["AUROC"] = np.nan
    neg = y == 0
    out["Specificity"] = float((pred[neg] == 0).mean()) if neg.any() else np.nan
    return out


def patient_alarms(df, delta, horizon_h=12.0):
    """df: one row per ECG with patient_id, ecg_time_h, hours_before_arrest, is_arrest_patient, score.
    Alarm at the first ECG whose score exceeds delta (Section 3.12)."""
    rows = []
    for (src, pid), g in df.sort_values("ecg_time_h").groupby(["source", "patient_id"], sort=False):
        hit = g[g["score"] > delta]
        arrest = bool(g["is_arrest_patient"].iloc[0])
        first_dt = g["hours_before_arrest"].max() if arrest else np.nan
        if arrest:
            ta = hit[hit["hours_before_arrest"] <= horizon_h]
            lead = float(ta["hours_before_arrest"].iloc[0]) if len(ta) else np.nan
            rows.append(dict(source=src, patient_id=pid, arrest=1, true_alarm=int(len(ta) > 0), lead_h=lead,
                             first_ecg_h=first_dt, false_alarm=0))
        else:
            rows.append(dict(source=src, patient_id=pid, arrest=0, true_alarm=0, lead_h=np.nan,
                             first_ecg_h=np.nan, false_alarm=int(len(hit) > 0)))
    return pd.DataFrame(rows)


def alarm_summary(pa, target_h=6.0):
    a, c = pa[pa.arrest == 1], pa[pa.arrest == 0]
    lt = a["lead_h"].dropna()
    return dict(mean_lead_h=lt.mean() if len(lt) else np.nan, median_lead_h=lt.median() if len(lt) else np.nan,
                q1_lead_h=lt.quantile(0.25) if len(lt) else np.nan, q3_lead_h=lt.quantile(0.75) if len(lt) else np.nan,
                detected_ge_target=(lt >= target_h).mean() if len(lt) else np.nan,
                missed_arrest=1 - a["true_alarm"].mean() if len(a) else np.nan,
                controls_false_alarm=c["false_alarm"].mean() if len(c) else np.nan,
                n_first_ecg_lt_1h=int((a["first_ecg_h"] < 1).sum()))


def horizon_table(df, delta, horizons=(1, 3, 6, 12)):
    """ECG-level horizon analysis (Section 3.12, Table S5)."""
    rows = []
    for h in horizons:
        keep = (df["is_arrest_patient"] == 0) | (df["hours_before_arrest"] <= h)
        d = df[keep]
        y = (d["is_arrest_patient"] == 1).astype(int).to_numpy()
        m = ecg_metrics(y, d["score"].to_numpy(), delta)
        a = df[df.is_arrest_patient == 1]
        n_in = a[a.hours_before_arrest <= h]["patient_id"].nunique()
        rows.append(dict(horizon_h=h, arrest_patients_in_window=n_in,
                         excluded=a["patient_id"].nunique() - n_in, Sensitivity=m["Recall"],
                         Specificity=m["Specificity"], AUPRC=m["AUPRC"]))
    return pd.DataFrame(rows)


def _resample_index(df, rng):
    """Patient-level resample: draw patients with replacement, keep all their ECGs."""
    groups = df.groupby(["source", "patient_id"]).indices
    keys = list(groups)
    pick = rng.integers(0, len(keys), len(keys))
    return np.concatenate([groups[keys[k]] for k in pick])


def bootstrap(df, delta, B=2000, seed=0, ci=0.95):
    """95% percentile CIs for ECG metrics and mean lead time (Section 4.7)."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(B):
        d = df.iloc[_resample_index(df, rng)]
        stats.append(ecg_metrics(d["y"].to_numpy(), d["score"].to_numpy(), delta))
    st = pd.DataFrame(stats)
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    return {k: (st[k].quantile(lo), st[k].quantile(hi)) for k in st.columns}


def lead_time_ci(df, delta, B=2000, seed=0, ci=0.95):
    """Bootstrap CI of the mean lead time of true alarms, resampling patients."""
    pa = patient_alarms(df, delta)
    lt = pa.loc[pa.arrest == 1, "lead_h"].to_numpy()
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(B):
        s = lt[rng.integers(0, len(lt), len(lt))]
        means.append(np.nanmean(s) if np.isfinite(s).any() else np.nan)
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    return np.nanquantile(means, lo), np.nanquantile(means, hi)


def paired_bootstrap(df_a, df_b, delta_a, delta_b, metric="AUPRC", B=2000, seed=0):
    """Difference A - B on identical patient resamples; two-sided p = 2 x the share of
    differences on the far side of zero (reported as p < 0.001 when none cross)."""
    key = ["source", "patient_id", "ecg_id"]
    m = df_a.merge(df_b[key + ["score"]], on=key, suffixes=("_a", "_b"))
    rng = np.random.default_rng(seed)
    point = ecg_metrics(m["y"].to_numpy(), m["score_a"].to_numpy(), delta_a)[metric] - \
        ecg_metrics(m["y"].to_numpy(), m["score_b"].to_numpy(), delta_b)[metric]
    diffs = []
    for _ in range(B):
        d = m.iloc[_resample_index(m, rng)]
        y = d["y"].to_numpy()
        diffs.append(ecg_metrics(y, d["score_a"].to_numpy(), delta_a)[metric] -
                     ecg_metrics(y, d["score_b"].to_numpy(), delta_b)[metric])
    diffs = np.array(diffs)
    far = (diffs <= 0).mean() if point > 0 else (diffs >= 0).mean()
    return dict(diff=point, ci=(np.quantile(diffs, 0.025), np.quantile(diffs, 0.975)), p=min(1.0, 2 * far))
