"""
measure_tilt_realdata.py
------------------------
Validate the HMM tilt detector on REAL human hands and write the committed
result (results/tilt_realdata.json) the figure layer reads.

Pipeline (see src/real_data_tilt.py for the methodology and honesty notes):
  1. parse the fetched PHH hands -> per-(hand, player) observations;
  2. build per-(player, table) session sequences;
  3. PHENOMENON test  -- post-loss aggression / VPIP shift, with a shuffled-label
     placebo as the negative control;
  4. DETECTOR test    -- the project's HMMBeliefState forward filter (emission-
     only) P(tilted) separation, also with a placebo;
  5. REGIME-FIT       -- 2-state vs 1-state HMM on binary aggression (honest BIC).

Real hands feed the OPPONENT-MODEL VALIDATION ONLY — never the DQN policy.

    python -m scripts.fetch_phh                  # fetch the subset first
    python -m scripts.measure_tilt_realdata      # -> results/tilt_realdata.json
"""

import argparse
import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")

from src.real_data_tilt import (parse_phhs, build_sequences, phenomenon_test,
                                within_player_loss_vs_win, detector_separation,
                                fit_regime_hmm, _aggr_rate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "phh")
RESULTS = os.path.join(ROOT, "results")

PLACEBO_SEED = 12345


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--loss-bb", type=float, default=10.0)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-gap-s", type=int, default=3600)
    ap.add_argument("--subset", default="PokerStars 25NL (2009 HandHQ scrape)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.phhs")))
    if not files:
        print(f"No .phhs files in {args.data_dir} — run scripts/fetch_phh first.")
        return

    print(f"parsing {len(files)} files...")
    records = []
    for f in files:
        records.extend(parse_phhs(f))
    sequences = build_sequences(records, min_len=args.min_len,
                                max_gap_s=args.max_gap_s)
    print(f"  {len(records)} (hand,player) rows -> {len(sequences)} sessions")

    # Population calibration of the detector's emission means (descriptive
    # statistics fixed BEFORE measuring separation; not tuned to the outcome).
    rates = [_aggr_rate(o) for s in sequences for o in s
             if _aggr_rate(o) is not None]
    mu_normal = round(sum(rates) / len(rates), 3)
    pos = [r for r in rates if r > 0]
    mu_tilted = round(sum(pos) / len(pos), 3)            # mean rate | aggressive
    frac_aggr = round(len(pos) / len(rates), 3)

    lb = args.loss_bb
    out = {
        "source": {
            "dataset": "A Dataset of Poker Hand Histories (Kim, J., 2024)",
            "doi": "10.5281/zenodo.13997158",
            "license": "CC-BY-4.0",
            "subset": args.subset,
            "provenance": ("NLHE hands originate from a July 2009 HandHQ scrape, "
                           "redistributed under CC-BY-4.0; see references.md"),
            "use": ("opponent-model validation only — these hands never train "
                    "the self-play DQN policy"),
            "parser": "pokerkit (canonical PHH replayer; chip-conserving net)",
        },
        "config": {
            "n_files": len(files), "loss_bb": lb, "min_len": args.min_len,
            "max_gap_s": args.max_gap_s, "mu_normal": mu_normal,
            "mu_tilted": mu_tilted, "recover": 0.15,
            "emission_calibration_note": (
                "mu_normal = global mean per-hand aggression rate; mu_tilted = "
                "mean rate over AGGRESSIVE hands only (rate>0), used as the "
                "detector's tilted-state emission anchor — NOT a regime mean. "
                "Both are population descriptive statistics fixed BEFORE the "
                "separation is measured, not tuned to the outcome (the "
                "post-hoc-tuning footgun, references.md §2)."
            ),
            "session_note": (
                "all hands carry one placeholder date (2009-07-01); sessions are "
                "split by the within-day time-of-day gap (max_gap_s), which "
                "reproduces PokerStars hand-id order on ~99.9% of adjacent pairs."
            ),
        },
        "n_rows": len(records),
        "n_sequences": len(sequences),
        "global_mean_rate": mu_normal,
        "frac_aggressive_hands": frac_aggr,
        "phenomenon": {
            "real": phenomenon_test(sequences, loss_bb=lb),
            "placebo": phenomenon_test(sequences, loss_bb=lb,
                                       placebo_seed=PLACEBO_SEED),
        },
        # The SYMMETRIC within-player control: post-(big-)loss vs post-(big-)WIN
        # of the SAME magnitude, so player type / big-pot arousal / event size
        # are matched and only the swing SIGN differs (the loss-aversion test).
        "within_player": {
            "real": within_player_loss_vs_win(sequences, swing_bb=lb),
            "placebo": within_player_loss_vs_win(sequences, swing_bb=lb,
                                                 placebo_seed=PLACEBO_SEED),
        },
        "detector": {
            "real": detector_separation(sequences, loss_bb=lb,
                                        mu_normal=mu_normal, mu_tilted=mu_tilted),
            "placebo": detector_separation(sequences, loss_bb=lb,
                                           mu_normal=mu_normal,
                                           mu_tilted=mu_tilted,
                                           placebo_seed=PLACEBO_SEED),
        },
        "regime": fit_regime_hmm(sequences, loss_bb=lb, seed=0),
    }

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "tilt_realdata.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    ph, det, reg = out["phenomenon"], out["detector"], out["regime"]
    print(f"\nPHENOMENON (loss>={lb:.0f}bb, {ph['real']['n_players']} players):")
    for k in ("aggr", "vpip"):
        r, p = ph["real"][k], ph["placebo"][k]
        print(f"  {k}: real {r['mean']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]   "
              f"placebo {p['mean']:+.4f} [{p['lo']:+.4f},{p['hi']:+.4f}]")
    wp = out["within_player"]
    print(f"WITHIN-PLAYER loss vs win (swing>={lb:.0f}bb, "
          f"{wp['real']['n_players']} matched players):")
    for k in ("aggr", "vpip"):
        r, p, dd = wp["real"][k], wp["placebo"][k], wp["real"][f"{k}_cohen_d"]
        print(f"  {k}: real {r['mean']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] "
              f"d={dd:.3f}   placebo {p['mean']:+.4f} "
              f"[{p['lo']:+.4f},{p['hi']:+.4f}]")
    r, p = det["real"]["separation"], det["placebo"]["separation"]
    print(f"DETECTOR P(tilted) separation: real {r['mean']:+.4f} "
          f"[{r['lo']:+.4f},{r['hi']:+.4f}]   placebo {p['mean']:+.4f} "
          f"[{p['lo']:+.4f},{p['hi']:+.4f}]")
    if reg.get("ok"):
        print(f"REGIME-FIT: BIC gain (2-state − 1-state) = {reg['bic_gain']:+.1f} "
              f"({'2-state preferred' if reg['bic_gain'] > 0 else '1-state preferred'})")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
