#!/usr/bin/env bash
#
# run_measurements.sh — regenerate the results/*.jsonl committed measurement data
# (Block B + ICM + rollout + headline + pool) that scripts/make_figures.py renders
# into figures/. No re-training is needed to redraw a plot once these JSONs exist.
#
# This covers the .jsonl figures only. Three standalone results are produced by
# their own measure scripts (see GUIDE.md "How to reproduce"):
#   results/variance_reduction.json  <- python -m scripts.measure_variance_reduction
#   results/exploitability.json      <- python -m scripts.measure_exploitability
#   results/tilt_realdata.json       <- python -m scripts.measure_tilt_realdata
#
# Thread-pinned (OMP_NUM_THREADS=1 — the small MLP only thrashes under parallel
# BLAS). Per-(script, cell) jobs run PAR at a time; each writes
# its own part file (atomic, no interleaving) and the parts are concatenated.
#
# Scale matches the committed Block B / A5 measurements (see results/ + GUIDE.md):
# init_seed varies the torch weight-init; the trainer RNG is held at seed=1, so
# every A/B is PAIRED per init_seed (the rl_multihand_sweep convention).
#
# Usage:  PY=python3.11 PAR=8 bash scripts/run_measurements.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
PAR="${PAR:-8}"
export OMP_NUM_THREADS=1

# Preflight: every measurement cell trains the DQN, which requires torch
# (commented out in requirements.txt). Fail early with a clear message rather
# than letting each job die on an opaque ImportError.
if ! "$PY" -c "import torch" 2>/dev/null; then
  echo "ERROR: '$PY' cannot import torch. Install it first: pip install \"torch>=2.0\"" >&2
  echo "       (the measurement scripts train the DQN; torch is required)" >&2
  exit 1
fi

mkdir -p results/_parts
rm -f results/_parts/*.jsonl results/_parts/*.json

JOBS="$(mktemp)"

# --- Block B per-cell jobs (B2 grid, B3 clip, B4 self-play, B5 tilt-decouple) -
for seed in 0 1 2 3 4 5; do
  for grid in five seven; do
    echo "$PY -m scripts.measure_action_grid --grid $grid --seed $seed --steps 1500 --out results/_parts/action_grid_${grid}_${seed}.jsonl" >> "$JOBS"
  done
  for clip in old wide 4.6; do
    tag="${clip//./}"
    echo "$PY -m scripts.measure_bust_clip --clip $clip --seed $seed --steps 1500 --out results/_parts/bust_clip_${tag}_${seed}.jsonl" >> "$JOBS"
  done
done
for seed in 0 1 2; do
  for config in fixed snapshot; do
    echo "$PY -m scripts.measure_selfplay --config $config --seed $seed --steps 1500 --out results/_parts/selfplay_${config}_${seed}.jsonl" >> "$JOBS"
  done
  for config in pnl_nobonus nopnl_bonus pnl_naive pnl_decouple; do
    echo "$PY -m scripts.measure_tilt_decouple --config $config --seed $seed --steps 6000 --out results/_parts/tilt_decouple_${config}_${seed}.jsonl" >> "$JOBS"
  done
done

# --- whole-sweep jobs (ICM mild+bubble ladders, rollout-FE, headline curve,
#     and the belief+mix generalist pool/sweep) ----------------------------------
echo "$PY -m scripts.measure_icm --steps 1500 --train-seeds 1 2 3 4 5 6 --prize-fracs 0.5 0.3 0.2 --out results/_parts/icm_mild.jsonl" >> "$JOBS"
echo "$PY -m scripts.measure_icm --steps 1500 --train-seeds 1 2 3 4 5 6 --prize-fracs 0.65 0.35 0.0 --out results/_parts/icm_bubble.jsonl" >> "$JOBS"
echo "$PY -m scripts.measure_rollout_fe --seeds 60 --hands 200 --out results/_parts/rollout_fe.jsonl" >> "$JOBS"
echo "$PY -m scripts.measure_headline --steps 1500 --eval-every 250 --eval-seeds 50 --eval-hands 150 --out results/_parts/headline_history.json" >> "$JOBS"
# The generalist is a single heavy 12k-step train + roster + MC sweep (~20 min).
echo "$PY -m scripts.measure_pool --steps 12000 --seeds 16 --hands 100 --out results/_parts/pool.json" >> "$JOBS"

echo "Launching $(wc -l < "$JOBS") jobs, $PAR at a time (thread-pinned)..."
# shellcheck disable=SC2002
if ! cat "$JOBS" | xargs -P "$PAR" -I CMD bash -c CMD; then
  echo "ERROR: one or more measurement cells failed; results/ NOT updated" >&2
  rm -f "$JOBS"
  exit 1
fi
rm -f "$JOBS"

# --- concatenate parts into the committed per-metric files --------------------
cat results/_parts/action_grid_*.jsonl   > results/action_grid.jsonl
cat results/_parts/bust_clip_*.jsonl     > results/bust_clip.jsonl
cat results/_parts/selfplay_*.jsonl      > results/selfplay.jsonl
cat results/_parts/tilt_decouple_*.jsonl > results/tilt_decouple.jsonl
cat results/_parts/icm_*.jsonl           > results/icm.jsonl
cp  results/_parts/rollout_fe.jsonl      results/rollout_fe.jsonl
cp  results/_parts/headline_history.json results/headline_history.json
cp  results/_parts/pool.json             results/pool.json
rm -rf results/_parts

echo "DONE -> results/"
wc -l results/*.jsonl
