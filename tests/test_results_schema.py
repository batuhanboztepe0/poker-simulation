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
            {"wins", "n_seeds", "mean_chip_diff", "p_value"}, set(d["final"]))

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
            self.assertLessEqual({"name", "mean_net_chips"}, set(entry))
        for cell in d["grid"]:
            self.assertLessEqual({"tight", "aggr", "mean_net_chips"}, set(cell))


if __name__ == "__main__":
    unittest.main()
