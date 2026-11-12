# SCA-QF-MSGT

Code for **"SCA-QF-MSGT: A Multi-Scale Graph Transformer with Hybrid Quantum-Classical Feature Transformation for Early Prediction of Sudden Cardiac Arrest from ECG"** (Charles P, Sountharrajan S).

Each heartbeat of a 12-lead ECG becomes a graph with one node per lead and Pearson-correlation edges. An edge-weighted multi-head graph attention encoder produces a graph embedding. A four-qubit parameterized circuit, simulated exactly on a classical computer, transforms four angles derived from that embedding. A learned sigmoid gate fuses the two streams. The model is trained with a focal loss plus a differentiable lead-time loss that rewards earlier alarms.

> The circuit is a classically simulated feature transformation. No part of this work runs on quantum hardware, and we make no claim of quantum advantage (manuscript, Sections 3.8 and 4.6.1).

## Repository layout

```
configs/default.yaml           all settings of Table 4
sca_qf_msgt/
  data/preprocessing.py        Eqs. 1-6: resampling, band-pass, z-score, R-peaks, beat windows, node features, edges
  data/labels.py               labelling protocol (Section 3.3.8, Algorithm S1) and patient-level splits
  data/io.py                   reading .npy and WFDB records, lead ordering, SCD segments
  data/hrv.py                  HRV and interval features of the clinical baselines
  data/dataset.py              patient-sequence batches (complete sequences per update)
  data/synthetic.py            NeuroKit2 synthetic cohort (smoke tests, Section 3.3.4)
  models/gat.py                edge-weighted graph attention and multi-scale readout (Eqs. 7-12)
  models/quantum.py            4-qubit circuit (Eqs. 15-17), torch and Qiskit backends, classical controls
  models/sca_qf_msgt.py        full model, gated fusion and head (Eqs. 13-23), ablation switches
  models/baselines.py          ResNet-Attention, Multi-Scale GCN, CNN-LSTM
  training/losses.py           focal loss, soft first-crossing lead time, L2 (Eqs. 24-26)
  training/trainer.py          Adam training, validation threshold, early stopping
  evaluation/metrics.py        AUPRC/F1/recall, alarm rules (Section 3.12), horizons, patient-level bootstrap
scripts/                       command-line entry points (see below)
figures/                       Figures 1-3 and 7-10 (make_figures.py, from_predictions.py, Figure1.drawio)
tests/                         unit and end-to-end tests
```

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/<user>/sca-qf-msgt.git
cd sca-qf-msgt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests          # 13 tests, about 5 s on a CPU
```

The default circuit backend (`model.quantum_backend: torch`) simulates the 16-amplitude state vector in PyTorch. Its outputs and gradients match the Qiskit backend (Qiskit Machine Learning `EstimatorQNN` with `TorchConnector` and parameter-shift gradients) to within 1e-7, as checked in `tests/test_quantum.py`. The Qiskit backend is much slower. Use it with `--set model.quantum_backend=qiskit` to reproduce the reported software stack exactly.

## Quick start on synthetic data

This runs the whole pipeline in a few minutes on a CPU. The data are synthetic, so the numbers mean nothing.

```bash
python scripts/make_synthetic.py --out data/synthetic --patients 30 --ecgs 3
python scripts/prepare_data.py --manifest data/synthetic/manifest.csv --root data/synthetic \
       --out data/processed_syn --keep-windows --hrv
python scripts/train.py --data data/processed_syn/bundle.pt --out runs/syn --seeds 0 --set train.epochs=3
python scripts/evaluate.py --pred "runs/syn/seed*/test_predictions.csv" --out results/syn --B 200
```

## Data

None of the datasets can be redistributed, so you must obtain each one from its provider.

| Source | Role | Access |
|---|---|---|
| SCAOS (Nightingale Open Science, Subtyping Cardiac Arrest) | train / validation / test | Nightingale Open Science terms |
| MIMIC-IV-ECG with the MIMIC-IV clinical modules | train / validation / test | PhysioNet credentialed access and data use agreement |
| SCD Holter Database (PhysioNet `sddb`) | external two-channel test only | open, PhysioNet |

### Manifest format

Every source is described by a CSV manifest with one row per ECG:

| column | meaning |
|---|---|
| `source` | `SCAOS`, `MIMIC-IV` or `SCD` |
| `patient_id` | patient (admission) identifier |
| `ecg_id` | unique ECG identifier |
| `path` | `.npy` array (leads x samples) or WFDB record path without extension, relative to `--root` |
| `fs` | sampling rate in Hz (read from the header for WFDB) |
| `ecg_time_h` | time of the ECG in hours on the admission clock |
| `arrest_time_h` | time of the arrest in hours on the same clock; empty for patients without an arrest |
| `leads` | optional, lead names joined by `\|` for `.npy` files (default I, II, III, aVR, aVL, aVF, V1-V6) |
| `seg_start_s`, `seg_end_s` | optional, crop of a continuous record (SCD 5-min segments) |

Helpers:

* `scripts/build_manifest_mimic.py` links `record_list.csv` of MIMIC-IV-ECG to admissions and to a table of timestamped arrests that you extract from the clinical modules (for example charted cardiopulmonary resuscitation events, Section 3.3.3).
* `scripts/build_manifest_scd.py` cuts each SCD record into consecutive 5-min segments up to the arrest onset and excludes the paced record.
* For SCAOS, write the manifest from the Nightingale metadata in the same format.

### Labels and splits (`sca_qf_msgt/data/labels.py`)

* An ECG of an arrest patient is labelled y = 1 if it was recorded 0 to H = 12 h before the arrest. Earlier ECGs from arrest patients are excluded, and so are post-arrest ECGs.
* Every ECG of a patient without an arrest is labelled y = 0.
* Patients are split 70/15/15 within each source, stratified by outcome and with a fixed seed, so no patient appears in two splits. SCD is not split.
* `prepare_data.py` writes `bundle_table_S1.csv` with patients, events, ECGs and prevalence for each split (Table S1).

## Reproducing the experiments

`scripts/run_all.sh` runs everything in order. The individual steps are:

| Manuscript item | Command |
|---|---|
| Preprocessing, graphs, Table S1 | `prepare_data.py --manifest ... --keep-windows --hrv` |
| Table 5, proposed model (5 seeds) | `train.py --data bundle.pt --out runs/sca_qf_msgt` then `evaluate.py` |
| Table 5, deep baselines | `train.py --baseline resnet_attention \| multiscale_gcn \| cnn_lstm` |
| Table 5, HRV baselines | `hrv_baselines.py --labelled ... --hrv ...` |
| Table 6, ablations | `--set model.quantum_block=none`, `model.fusion=concat`, `model.edge_prior=false`, `loss.use_time_loss=false` |
| Table 7, paired test vs Multi-Scale GCN | `evaluate.py --pred <proposed> --compare <gcn>` |
| Table S5, horizons 1/3/6/12 h | `evaluate.py` writes `horizons.csv` |
| Table S6, leave-one-dataset-out | `train.py --train-sources MIMIC-IV` (test on SCAOS) and the reverse; `evaluate.py --sources ...` |
| Table S6, domain classifier | `domain_classifier.py --model ... --data ... --external scd.pt` |
| Table S7, alarms and lead time | `evaluate.py` writes `metrics.json` (`patient_level`) and `patient_alarms.csv` |
| Table S8, external SCD test | `prepare_data.py --external --name scd` then `predict.py --models ... --data scd.pt` |
| Table S9, matched classical controls | `--set model.quantum_block=bottleneck_mlp \| random_features \| wide_mlp \| identity` |
| Table S10, latency | `latency.py --model ... --manifest ...` |

`evaluate.py` averages ECG scores over the seeds given. It uses the mean of the per-seed validation thresholds and reports point estimates with 95% patient-level bootstrap intervals (B = 2,000). The paired test resamples the same patients for both models and reports p as twice the share of differences on the far side of zero.

## Implementation notes

These choices are not fully fixed by the equations, so they are recorded here:

* **Neighbourhood.** N(i) in Eq. 9 contains the leads with a retained edge (|ρ| > 0.5) plus lead i itself. The self-loop carries edge prior 0.
* **Heads.** Eq. 11 concatenates M = 5 heads of width 64. A linear layer projects the 320 concatenated features back to 64, so every layer has the working width of Table 4.
* **Readout.** z_G = Σ β_l mean_i h_i^(l) over the nodes present, with β_l learnable and initialized to 1/L.
* **ECG score and alarms.** The score of an ECG is the mean of its beat scores. An alarm is raised at the first ECG of a patient whose score exceeds δ (Section 3.12).
* **Threshold.** δ is the maximum-F1 threshold on the validation set. It is updated after every epoch and held fixed within the next epoch, which is when the lead-time loss uses it.
* **Lead-time loss.** Each update uses 5 complete patient sequences. The lead-time loss uses ECG-level scores in time order, exactly as in Listing S1.
* **Missing leads.** Leads that were not recorded are left out of the graph rather than zero-filled (Section 3.3.7). SCD beats therefore become 2-node graphs.
* **Window length.** Signals are resampled to 510 Hz, so a 0.6 s window has 306 samples.

## Figures

* **Figure 1** is a diagram (`figures/Figure1.drawio`, editable at app.diagrams.net). `figures/fig1_drawio.py` regenerates it.
* **Figures 2, 3 and 7-10** are drawn from your own results.

```bash
python figures/from_predictions.py results/*/predictions_for_figures.csv --example-patient <patient_id> \
       --hrv data/processed/bundle_hrv.csv --labelled data/processed/bundle_labelled.csv
python figures/make_figures.py figures/out
```

`from_predictions.py` computes the precision-recall curves, the confusion matrix at each model's δ, AUPRC per test set, lead-time summaries, the single-patient trajectories and the HRV densities. It writes them to `figures/digit.json` and `figures/data_override.json`, which `make_figures.py` reads. The class counts of Figure 2 come from Table S1 and are set in the `DATA` block of `make_figures.py`.

## Citation

See `CITATION.cff`. Please cite the article when you use this code.

## License

MIT (see `LICENSE`). The license covers the code only. Each dataset keeps its own terms of use.
