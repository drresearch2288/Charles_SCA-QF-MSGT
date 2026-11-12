"""Rebuilds the data behind Figures 7-10 from the saved test-set predictions.

Input: one CSV with a row per test ECG and per model, with these columns:
  model                one of: ResNet-Attention, Multi-Scale GCN, CNN-LSTM, SCA-QF-MSGT
  source               SCAOS or MIMIC-IV
  patient_id           patient identifier
  hours_before_arrest  hours from the ECG to the arrest (empty for control patients)
  y_true               label of Section 3.3.8 (1 = within 12 h before arrest, 0 = control)
  score                predicted risk score (ŷ) of that ECG
  delta                (optional) the model's validation threshold; 0.42 if absent

Usage (one CSV per model, as written by scripts/evaluate.py):
  python figures/from_predictions.py results/*/predictions_for_figures.csv --example-patient <id>
  python make_figures.py

It overwrites the traced curves in digit.json and writes data_override.json
(confusion matrix, AUPRC per test set, lead-time summaries), which
make_figures.py reads. All numbers it computes are printed so they can be
checked against Tables 5 and S7.
"""
import argparse, csv, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DELTA = 0.42                                   # used only when the CSV has no delta column
SRC_KEY = {'SCAOS': 'scaos', 'MIMIC-IV': 'mimic'}


def pr_curve(y, s):
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / (tp + fp); rec = tp / max(y.sum(), 1)
    auprc = float(np.sum(np.diff(np.r_[0, rec]) * prec))   # average precision
    keep = np.linspace(0, len(rec) - 1, min(len(rec), 400)).astype(int)
    return rec[keep].tolist(), prec[keep].tolist(), auprc


def lead_times(rows, delta=DELTA):
    """hard first crossing per arrest patient: earliest ECG with score > delta."""
    out = {}
    for r in rows:
        if r['y_true'] == 1 and r['score'] > delta:
            out[r['pid']] = max(out.get(r['pid'], 0.0), r['h'])
    return np.array(list(out.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='+'); ap.add_argument('--example-patient', required=True)
    ap.add_argument('--hrv', help='bundle_hrv.csv from prepare_data.py --hrv, for Figure 3')
    ap.add_argument('--labelled', help='bundle_labelled.csv from prepare_data.py, for Figure 3')
    ap.add_argument('--hrv-metric', default='sdnn_ms')
    a = ap.parse_args()
    rows = []
    for r in (r for f in a.csv for r in csv.DictReader(open(f, newline='', encoding='utf-8-sig'))):
        rows.append(dict(model=r['model'].strip(), src=r['source'].strip(), pid=r['patient_id'].strip(),
                         h=float(r['hours_before_arrest']) if r['hours_before_arrest'].strip() else np.nan,
                         y_true=int(float(r['y_true'])), score=float(r['score']),
                         delta=float(r['delta']) if r.get('delta') not in (None, '') else DELTA))
    dgp = os.path.join(HERE, 'digit.json')
    dg = json.load(open(dgp)) if os.path.exists(dgp) else {}
    ov = {'AUPRC_SRC': {}, 'LEAD': {}}
    models = sorted({r['model'] for r in rows})
    for m in models:
        mr = [r for r in rows if r['model'] == m]
        d_m = mr[0]['delta']                       # validation threshold of this model
        if m == 'SCA-QF-MSGT':
            ov['DELTA'] = d_m
        aus = []
        for src in ('SCAOS', 'MIMIC-IV'):
            sr = [r for r in mr if r['src'] == src]
            y = np.array([r['y_true'] for r in sr]); s = np.array([r['score'] for r in sr])
            rec, prec, au = pr_curve(y, s); aus.append(round(au, 3))
            if m == 'ResNet-Attention': dg['f7_pr_' + SRC_KEY[src]] = [rec, prec]
            if m == 'Multi-Scale GCN': dg['f8_pr_' + SRC_KEY[src]] = [rec, prec]
            if m == 'ResNet-Attention' and src == 'MIMIC-IV':
                p = s > d_m
                cm = [[int(((y == 0) & ~p).sum()), int(((y == 0) & p).sum())],
                      [int(((y == 1) & ~p).sum()), int(((y == 1) & p).sum())]]
                ov['CM_RESNET'] = cm
            if m == 'Multi-Scale GCN' and src == 'MIMIC-IV':
                lt = lead_times(sr, d_m)
                dg['f8_scatter'] = [[i + 1, float(v)] for i, v in enumerate(lt)]
        ov['AUPRC_SRC'][m] = aus
        lt = lead_times(mr, d_m)
        if len(lt):
            ov['LEAD'][m] = [round(float(lt.mean()), 1), round(float(np.median(lt)), 1),
                             round(float(np.percentile(lt, 25)), 1), round(float(np.percentile(lt, 75)), 1)]
        # example-patient trajectories
        er = sorted((r for r in mr if r['pid'] == a.example_patient), key=lambda r: -r['h'])
        if er:
            xy = [[r['h'] for r in er], [r['score'] for r in er]]
            key = {'SCA-QF-MSGT': ['f9_prop', 'f10_prop'], 'CNN-LSTM': ['f9_cnnlstm', 'f10_cnnlstm'],
                   'ResNet-Attention': ['f10_resnet'], 'Multi-Scale GCN': ['f10_gcn']}.get(m, [])
            for k in key: dg[k] = xy
    if a.hrv and a.labelled:                     # Figure 3: density of one HRV metric by class
        import pandas as pd
        from scipy.stats import gaussian_kde
        h = pd.read_csv(a.hrv, dtype={'ecg_id': str}).merge(pd.read_csv(a.labelled, dtype={'ecg_id': str})[['ecg_id', 'y']], on='ecg_id')
        for key, yv in (('normal', 0), ('pre', 1)):
            v = h.loc[h.y == yv, a.hrv_metric].dropna().to_numpy()
            grid = np.linspace(max(0, v.min()), v.max(), 200)
            dg.setdefault('f3', {})[key] = [[float(g), float(d)] for g, d in zip(grid, gaussian_kde(v)(grid))]
        ov['HRV_METRIC'] = a.hrv_metric
    json.dump(dg, open(os.path.join(HERE, 'digit.json'), 'w'))
    json.dump(ov, open(os.path.join(HERE, 'data_override.json'), 'w'), indent=1)
    print(json.dumps(ov, indent=1))


if __name__ == '__main__':
    main()
