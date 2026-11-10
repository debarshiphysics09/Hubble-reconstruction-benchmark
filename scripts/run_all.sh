#!/usr/bin/env bash
# Runs every analysis in the paper, then regenerates figures and tables.
# Single CPU: ~2-3 h.  Set HZ_FAST=1 for a ~5-minute smoke test.
set -e
cd "$(dirname "$0")/.."
for s in run_real_data run_nodal_diagnostics run_tier1 run_tier2 run_covariance_gls \
         run_synthetic_control run_nn_variability run_tuned; do
  echo "=== $s ($(date +%H:%M:%S))"
  python scripts/$s.py
done
python scripts/make_figures.py
python scripts/make_tables.py
echo "=== all done ($(date +%H:%M:%S))"
