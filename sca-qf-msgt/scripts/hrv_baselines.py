"""Clinical baselines of Table 5: logistic regression and XGBoost on mean HR, SDNN, RMSSD,
pNN50, QRS duration, QTc and PR interval (Section 4.5.1), trained on the same splits.

  python scripts/hrv_baselines.py --labelled data/processed/bundle_labelled.csv \
      --hrv data/processed/bundle_hrv.csv --out runs/hrv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sca_qf_msgt.data.hrv import HRV_COLUMNS
from sca_qf_msgt.evaluation.metrics import best_f1_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True); ap.add_argument("--hrv", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_csv(a.labelled, dtype={"patient_id": str, "ecg_id": str}).merge(
        pd.read_csv(a.hrv, dtype={"ecg_id": str}), on="ecg_id")
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    pos = max(tr.y.sum(), 1)
    models = {
        "Logistic regression (HRV + rhythm)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                            LogisticRegression(max_iter=2000, class_weight="balanced")),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost (HRV + rhythm)"] = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                                         subsample=0.8, colsample_bytree=0.8,
                                                         scale_pos_weight=(len(tr) - pos) / pos,
                                                         eval_metric="aucpr", random_state=a.seed)
    except ImportError:
        print("xgboost not installed; only logistic regression is run")
    for name, m in models.items():
        m.fit(tr[HRV_COLUMNS], tr.y)
        delta = best_f1_threshold(va.y.to_numpy(), m.predict_proba(va[HRV_COLUMNS])[:, 1])
        out = te[["source", "patient_id", "ecg_id", "ecg_time_h", "hours_before_arrest", "is_arrest_patient", "y"]].copy()
        out["score"] = m.predict_proba(te[HRV_COLUMNS])[:, 1]
        out["model"], out["delta"] = name, delta
        d = os.path.join(a.out, name.split(" ")[0].lower()); os.makedirs(d, exist_ok=True)
        out.to_csv(os.path.join(d, "test_predictions.csv"), index=False)
        print(name, "delta %.3f ->" % delta, d)


if __name__ == "__main__":
    main()
