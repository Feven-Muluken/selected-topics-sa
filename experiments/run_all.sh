#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 "$ROOT/data/download_data.py"
python3 -m experiments.tune_parameters
python3 -m experiments.make_tuning_heatmaps
python3 -m experiments.run_sa
python3 -m experiments.run_baseline
python3 -m experiments.analyze_results
python3 -m experiments.make_plots
