#!/usr/bin/env bash
# Runs every experiment of Section 4 in order. Edit the manifest paths first (README, step 2).
set -euo pipefail
DATA=data/processed
python scripts/prepare_data.py --manifest data/manifests/scaos.csv data/manifests/mimic.csv \
    --root data/raw --out $DATA --keep-windows --hrv
python scripts/prepare_data.py --manifest data/manifests/scd.csv --root data/raw --out $DATA --name scd --external

# Table 5: proposed model (5 seeds) and baselines
python scripts/train.py --data $DATA/bundle.pt --out runs/sca_qf_msgt
for b in resnet_attention multiscale_gcn cnn_lstm; do
  python scripts/train.py --data $DATA/bundle.pt --out runs/$b --baseline $b
done
python scripts/hrv_baselines.py --labelled $DATA/bundle_labelled.csv --hrv $DATA/bundle_hrv.csv --out runs/hrv

# Table 6: ablations
python scripts/train.py --data $DATA/bundle.pt --out runs/abl_no_quantum   --set model.quantum_block=none
python scripts/train.py --data $DATA/bundle.pt --out runs/abl_no_gate      --set model.fusion=concat
python scripts/train.py --data $DATA/bundle.pt --out runs/abl_no_edge      --set model.edge_prior=false
python scripts/train.py --data $DATA/bundle.pt --out runs/abl_no_timeloss  --set loss.use_time_loss=false
python scripts/train.py --data $DATA/bundle.pt --out runs/abl_standard_gat --set model.edge_prior=false model.quantum_block=none

# Table S9: parameter-matched classical controls
for q in bottleneck_mlp random_features wide_mlp identity; do
  python scripts/train.py --data $DATA/bundle.pt --out runs/ctrl_$q --set model.quantum_block=$q
done

# Table S6: leave-one-dataset-out
python scripts/train.py --data $DATA/bundle.pt --out runs/lodo_train_mimic --train-sources MIMIC-IV
python scripts/train.py --data $DATA/bundle.pt --out runs/lodo_train_scaos --train-sources SCAOS

# Table S8: external two-channel test on SCD, no retraining
python scripts/predict.py --models "runs/sca_qf_msgt/seed*/model.pt" --data $DATA/scd.pt --out runs/scd_external

# Evaluation with patient-level bootstrap (B = 2,000)
for r in runs/*/; do
  n=$(basename $r)
  if [ "$n" = hrv ]; then
    for m in logistic xgboost; do python scripts/evaluate.py --pred "runs/hrv/$m/test_predictions.csv" --out results/$m; done
  elif [[ "$n" == lodo_train_mimic ]]; then
    python scripts/evaluate.py --pred "$r/seed*/test_predictions.csv" --sources SCAOS --out results/$n
  elif [[ "$n" == lodo_train_scaos ]]; then
    python scripts/evaluate.py --pred "$r/seed*/test_predictions.csv" --sources MIMIC-IV --out results/$n
  else
    python scripts/evaluate.py --pred "$r/seed*/test_predictions.csv" --out results/$n
  fi
done
# Table 7 and Table S9 paired tests
python scripts/evaluate.py --pred "runs/sca_qf_msgt/seed*/test_predictions.csv" \
    --compare "runs/multiscale_gcn/seed*/test_predictions.csv" --out results/paired_vs_gcn
python scripts/evaluate.py --pred "runs/sca_qf_msgt/seed*/test_predictions.csv" \
    --compare "runs/ctrl_bottleneck_mlp/seed*/test_predictions.csv" --out results/paired_vs_mlp
