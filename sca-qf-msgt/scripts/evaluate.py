"""Evaluates test predictions under the rules of Section 3.12 with patient-level bootstrap CIs.

  python scripts/evaluate.py --pred runs/sca_qf_msgt/seed*/test_predictions.csv --out results/sca_qf_msgt
  python scripts/evaluate.py --pred A.csv --compare B.csv          # paired bootstrap (Table 7, Table S9)

With several seeds, ECG scores are averaged over seeds (the seed ensemble) and delta is the
mean of the per-seed validation thresholds. Writes metrics.json (Table 5 / S7 rows),
horizons.csv (Table S5) and predictions_for_figures.csv (input of figures/from_predictions.py).
"""
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sca_qf_msgt.evaluation.metrics import (alarm_summary, bootstrap, ecg_metrics, horizon_table,
                                            lead_time_ci, paired_bootstrap, patient_alarms)

warnings.filterwarnings("ignore", category=UserWarning)
KEY = ["source", "patient_id", "ecg_id"]


def load(patterns):
    files = sorted(f for p in patterns for f in glob.glob(p))
    if not files:
        sys.exit("no prediction files match %s" % patterns)
    dfs = [pd.read_csv(f, dtype={"patient_id": str, "ecg_id": str}) for f in files]
    base = dfs[0].drop(columns=["score", "delta"])
    s = np.mean([d.set_index(KEY).loc[base.set_index(KEY).index, "score"].to_numpy() for d in dfs], axis=0)
    delta = float(np.mean([d["delta"].iloc[0] for d in dfs]))
    return base.assign(score=s), delta, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--sources", nargs="*", help="restrict to these sources, e.g. MIMIC-IV")
    a = ap.parse_args()
    df, delta, files = load(a.pred)
    if a.sources:
        df = df[df.source.isin(a.sources)]
    os.makedirs(a.out, exist_ok=True)
    m = ecg_metrics(df["y"].to_numpy(), df["score"].to_numpy(), delta)
    ci = bootstrap(df, delta, a.B)
    pa = patient_alarms(df, delta)
    al = alarm_summary(pa)
    al["mean_lead_ci"] = lead_time_ci(df, delta, a.B)
    res = {"files": files, "delta": delta, "n_ecg": len(df), "n_patients": int(df.patient_id.nunique()),
           "ecg_level": {k: [m[k], ci[k][0], ci[k][1]] for k in m}, "patient_level": al}
    if a.compare:
        db, delta_b, _ = load(a.compare)
        res["paired_vs_compare"] = {k: paired_bootstrap(df, db, delta, delta_b, k, a.B) for k in ("AUPRC", "F1", "Recall")}
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1, default=float)
    horizon_table(df, delta).to_csv(os.path.join(a.out, "horizons.csv"), index=False)
    pa.to_csv(os.path.join(a.out, "patient_alarms.csv"), index=False)
    model = df["model"].iloc[0] if "model" in df else "SCA-QF-MSGT"
    df.assign(model=model, y_true=df["y"], delta=delta)[["model", "source", "patient_id", "hours_before_arrest",
                                                         "y_true", "score", "delta"]] \
        .to_csv(os.path.join(a.out, "predictions_for_figures.csv"), index=False)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
