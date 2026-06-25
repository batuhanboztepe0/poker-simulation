"""
test_results_schema.py
----------------------
Guard the committed measurement data under results/ — the single source of truth
the figure layer (scripts/make_figures.py) reads. Each file must parse and carry
the documented schema, so a schema drift in a measure_* script (the --out writers)
is caught here rather than silently breaking a figure.

Files are validated when present; a file that is absent (e.g. a checkout that has
not run scripts/run_measurements.sh) skips its case rather than failing.
"""

import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# filename -> required keys per JSON-line row (the --out schemas).
JSONL_SCHEMAS = {
    "action_grid.jsonl": {"grid", "n_actions", "init_seed", "wins", "n",
                          "mean", "p_value"},
    "bust_clip.jsonl": {"clip", "reward_clip", "init_seed", "wins", "n",
                        "mean", "p_value", "bust_rate"},
    "selfplay.jsonl": {"config", "init_seed", "n", "myopic", "tilt", "random"},
    "tilt_decouple.jsonl": {"config", "init_seed", "myopic", "tilt", "random"},
    "icm.jsonl": {"init_seed", "n_players", "ladder", "icm_mean_prize",
                  "chips_mean_prize", "myopic_mean_prize", "icm_minus_chips",
                  "p_value"},
    "rollout_fe.jsonl": {"config", "opponent", "mean_diff", "p_value",
                         "n_seeds", "n_hands"},
}


def _load_jsonl(path):
    return [json.loads(ln) for ln in open(path) if ln.strip()]


class TestResultsSchema(unittest.TestCase):
    def test_jsonl_files_match_schema(self):
        checked = 0
        for fname, required in JSONL_SCHEMAS.items():
            path = os.path.join(RESULTS, fname)
            if not os.path.exists(path):
                continue
            rows = _load_jsonl(path)
            self.assertTrue(rows, f"{fname} is empty")
            for row in rows:
                missing = required - set(row)
                self.assertFalse(missing, f"{fname} row missing {missing}")
            checked += 1
        if checked == 0:
            self.skipTest("no results/*.jsonl present (run run_measurements.sh)")

    def test_headline_history_schema(self):
        path = os.path.join(RESULTS, "headline_history.json")
        if not os.path.exists(path):
            self.skipTest("no results/headline_history.json present")
        with open(path) as f:
            d = json.load(f)
        self.assertLessEqual({"history", "final", "torch_seed", "steps",
                              "eval_seeds", "eval_hands"}, set(d))
        self.assertTrue(d["history"], "history is empty")
        for snap in d["history"]:
            self.assertLessEqual(
                {"step", "wins", "n_seeds", "mean_chip_diff", "per_seed_diffs"},
                set(snap))
        self.assertLessEqual(
            {"wins", "n_seeds", "mean_chip_diff", "p_value", "binom_p",
             "binom", "ci95"},
            set(d["final"]))
        self.assertLessEqual({"lo", "hi", "mean"}, set(d["final"]["ci95"]))
        # The exact binomial sign test is the headline's correct test (binary
        # bust matches); guard its presence so a measure_* drift is caught here.
        self.assertLessEqual({"wins", "losses", "n", "p_value"},
                             set(d["final"]["binom"]))

    def test_variance_reduction_schema(self):
        path = os.path.join(RESULTS, "variance_reduction.json")
        if not os.path.exists(path):
            self.skipTest("no results/variance_reduction.json present")
        with open(path) as f:
            d = json.load(f)
        self.assertIn("arms", d)
        for arm in d["arms"]:
            self.assertLessEqual(
                {"arm", "mean", "lo", "hi", "ci_width", "ci_width_vs_raw"},
                set(arm))

    def test_exploitability_schema(self):
        path = os.path.join(RESULTS, "exploitability.json")
        if not os.path.exists(path):
            self.skipTest("no results/exploitability.json present")
        with open(path) as f:
            d = json.load(f)
        self.assertIn("curve", d)
        for pt in d["curve"]:
            self.assertLessEqual(
                {"iters", "avg_exploitability", "last_iterate_exploitability"},
                set(pt))
        # NFSP average-policy curve (the learned averaging method), when present
        if "nfsp_curve" in d:
            for pt in d["nfsp_curve"]:
                self.assertLessEqual({"episodes", "exploitability"}, set(pt))

    def test_pool_schema(self):
        path = os.path.join(RESULTS, "pool.json")
        if not os.path.exists(path):
            self.skipTest("no results/pool.json present")
        with open(path) as f:
            d = json.load(f)
        self.assertLessEqual(
            {"leaderboard", "per_agent_nets", "win_matrix", "grid", "rl_rank",
             "n_agents_in_sweep", "best_static"}, set(d))
        for entry in d["leaderboard"]:
            self.assertLessEqual({"name", "mean_net_chips", "ci95"}, set(entry))
            self.assertLessEqual({"lo", "hi", "mean"}, set(entry["ci95"]))
        for cell in d["grid"]:
            self.assertLessEqual({"tight", "aggr", "mean_net_chips"}, set(cell))

    def test_tilt_realdata_schema(self):
        path = os.path.join(RESULTS, "tilt_realdata.json")
        if not os.path.exists(path):
            self.skipTest("no results/tilt_realdata.json present")
        with open(path) as f:
            d = json.load(f)
        self.assertLessEqual(
            {"source", "config", "n_rows", "n_sequences", "phenomenon",
             "within_player", "detector", "regime"}, set(d))
        # Phenomenon: each of real/placebo carries vpip + aggr CIs and n_players.
        for arm in ("real", "placebo"):
            ph = d["phenomenon"][arm]
            self.assertIn("n_players", ph)
            for metric in ("vpip", "aggr"):
                self.assertLessEqual({"mean", "lo", "hi"}, set(ph[metric]))
        # Symmetric within-player control (post-loss vs post-WIN): CIs, the
        # matched-player count, and the Cohen's d effect size.
        for arm in ("real", "placebo"):
            wp = d["within_player"][arm]
            self.assertLessEqual({"n_players", "n_loss", "n_win"}, set(wp))
            for metric in ("vpip", "aggr"):
                self.assertLessEqual({"mean", "lo", "hi"}, set(wp[metric]))
            self.assertIn("aggr_cohen_d", wp)
        # Detector: each arm carries a separation CI (the figure reads its mean).
        for arm in ("real", "placebo"):
            self.assertLessEqual(
                {"mean", "lo", "hi"}, set(d["detector"][arm]["separation"]))
        # Regime: the keys fig_tilt_realdata reads when two_state_found is true.
        reg = d["regime"]
        self.assertIn("two_state_found", reg)
        if reg.get("two_state_found"):
            self.assertLessEqual(
                {"bic_gain", "heldout_ll_gain", "p_aggr_low", "p_aggr_high",
                 "p_loss_given_high", "p_loss_base"}, set(reg))
        # Config fields the caption interpolates.
        self.assertLessEqual(
            {"loss_bb", "n_files", "mu_normal", "mu_tilted"}, set(d["config"]))


    def test_seed_sweep_schema(self):
        # Phase 0 multi-seed robustness (PREREGISTRATION.md §10). The figure layer
        # (fig_seed_sweep) reads per_seed[*].mirror.{mean_diff,ci95} and the
        # across-seed summary; guard those keys so a measure_seed_sweep drift is
        # caught here.
        path = os.path.join(RESULTS, "seed_sweep.json")
        if not os.path.exists(path):
            self.skipTest("no results/seed_sweep.json present")
        with open(path) as f:
            d = json.load(f)
        self.assertLessEqual({"protocol", "per_seed", "summary"}, set(d))
        self.assertTrue(d["per_seed"], "per_seed is empty")
        for row in d["per_seed"]:
            self.assertLessEqual({"torch_seed", "mirror"}, set(row))
            mir = row["mirror"]
            self.assertLessEqual({"mean_diff", "ci95", "resolved", "edge_sign",
                                  "per_seed_diffs"}, set(mir))
            self.assertLessEqual({"lo", "hi", "mean"}, set(mir["ci95"]))
        s = d["summary"]
        self.assertLessEqual(
            {"n_seeds_trained", "mean_edge", "median_edge", "sd_edge",
             "min_edge", "max_edge", "across_seed_ci95", "n_resolved_positive",
             "n_resolved_negative", "n_unresolved", "seed0_edge",
             "seed0_percentile", "verdict"}, set(s))
        self.assertLessEqual({"lo", "hi", "mean"}, set(s["across_seed_ci95"]))
        self.assertIn(s["verdict"], {"robust", "seed-dependent"})
        # The reproducibility gate the script records when confirmatory.json is
        # present: seed 0's edge equals the committed confirmatory edge.
        if "seed0_reproduces_committed" in s:
            self.assertEqual(s["seed0_edge"], s["committed_seed0_edge"])
            self.assertTrue(s["seed0_reproduces_committed"],
                            "seed 0 must reproduce the committed confirmatory edge")


    def test_neural_nfsp_schema(self):
        # Phase 2 neural NFSP convergence (PREREGISTRATION.md §11). The figure
        # layer (fig_neural_nfsp) reads the across-seed curve, the tabular
        # reference, and the head-to-head; guard those keys.
        path = os.path.join(RESULTS, "neural_nfsp.json")
        if not os.path.exists(path):
            self.skipTest("no results/neural_nfsp.json present")
        with open(path) as f:
            d = json.load(f)
        self.assertLessEqual(
            {"method", "feature_dim", "config", "n_seeds", "checkpoints",
             "uniform_exploitability", "per_seed", "curve", "head_to_head"},
            set(d))
        self.assertTrue(d["curve"], "curve is empty")
        for pt in d["curve"]:
            self.assertLessEqual(
                {"episodes", "mean_exploitability", "min", "max"}, set(pt))
        for ps in d["per_seed"]:
            self.assertLessEqual({"seed", "curve"}, set(ps))
        for h in d["head_to_head"]:
            self.assertLessEqual(
                {"episodes", "neural_mean", "tabular", "neural_beats_tabular"},
                set(h))


    def test_scale_experiment_schema(self):
        # Phase 2 scaling experiment (PREREGISTRATION.md §12). The figure layer
        # (fig_scale) reads the head-to-head, the cost curve, and the per-method
        # exact exploitability; guard those keys.
        path = os.path.join(RESULTS, "scale_experiment.json")
        if not os.path.exists(path):
            self.skipTest("no results/scale_experiment.json present")
        with open(path) as f:
            d = json.load(f)
        self.assertLessEqual(
            {"ranks", "deals_per_cfr_iter", "info_sets", "cost_curve",
             "cfr_secs_per_iter", "cfr_converge_hours_est", "uniform",
             "tabular_cfr", "neural_nfsp", "head_to_head", "lbr_le_exact_check"},
            set(d))
        self.assertLessEqual({"exact", "lbr"}, set(d["uniform"]))
        self.assertLessEqual({"iters", "exact", "lbr"}, set(d["tabular_cfr"]))
        self.assertLessEqual({"seeds", "per_seed", "exact_mean", "exact_min",
                              "exact_max"}, set(d["neural_nfsp"]))
        self.assertLessEqual(
            {"neural_exact_mean", "tabular_exact",
             "neural_beats_tabular_at_matched_budget"}, set(d["head_to_head"]))
        # The LBR lower-bound guarantee must hold at scale too.
        self.assertTrue(d["lbr_le_exact_check"], "LBR must be <= exact at scale")


if __name__ == "__main__":
    unittest.main()
