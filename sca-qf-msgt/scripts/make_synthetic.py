"""Writes a small synthetic 12-lead cohort (NeuroKit2) and its manifest, for smoke tests.

  python scripts/make_synthetic.py --out data/synthetic --patients 60
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sca_qf_msgt.data.synthetic import make_cohort

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic"); ap.add_argument("--patients", type=int, default=60)
ap.add_argument("--ecgs", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
m = make_cohort(a.out, a.patients, ecgs_per_patient=a.ecgs, seed=a.seed)
print("%d ECGs written to %s" % (len(m), a.out))
