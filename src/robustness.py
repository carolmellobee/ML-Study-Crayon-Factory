"""Repeat the whole study on many freshly generated datasets.

One dataset is one draw of luck: which stores happened to carry what, which
weeks were noisy. Regenerating the sales with different random seeds (same
company, same true demand, different assortment, promotions and noise) shows
whether the conclusions hold, and whether the model's 95% intervals really
contain the truth about 95% of the time.

Run:  python src/robustness.py          (20 datasets, a few minutes)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as A  # noqa: E402
import generate_data as G  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "robustness.csv"


def one_run(seed: int) -> dict:
    bundles, stores, sales, truth = G.generate(seed)
    t = truth["true_bundle_parameters"].set_index("bundle_id")
    traffic = truth["true_store_traffic"].set_index("store_id")
    s = A.prepare(sales, bundles, stores)
    naive = A.naive_table(s, bundles)
    true_roi = A.true_promo_roi(s, bundles, t, traffic)
    best = t.baseline_velocity.idxmax()
    row = {
        "seed": seed,
        "rank_corr_total_units": A.spearman(naive.units, t.baseline_velocity),
        "rank_corr_velocity": A.spearman(naive.velocity, t.baseline_velocity),
        "top_pick_total_units": naive.units.idxmax() == best,
        "top_pick_velocity": naive.velocity.idxmax() == best,
    }
    for label, cb in [("basic", False), ("channel", True)]:
        m = A.DemandModel(s, channel_bundle=cb)
        ci = m.baseline_velocity_ci(400, seed=seed)
        el = m.elasticities()
        roi = A.promo_roi(m, s, bundles).roi
        row |= {
            f"{label}_rank_corr": A.spearman(ci.estimate, t.baseline_velocity),
            f"{label}_top_pick": ci.estimate.idxmax() == best,
            f"{label}_baseline_ci_coverage": float(((ci.low <= t.baseline_velocity)
                                                   & (t.baseline_velocity <= ci.high)).mean()),
            f"{label}_elasticity_ci_coverage": float(((el.low <= t.elasticity)
                                                     & (t.elasticity <= el.high)).mean()),
            f"{label}_elasticity_mean_abs_error": float((el.estimate - t.elasticity).abs().mean()),
            f"{label}_roi_sign_agreement": float((np.sign(roi)
                                                  == np.sign(true_roi.reindex(roi.index))).mean()),
            f"{label}_roi_mean_abs_error": float((roi - true_roi.reindex(roi.index)).abs().mean()),
        }
    return row


def run(seeds=range(1, 21), workers: int = 2) -> pd.DataFrame:
    """Each dataset runs in a fresh worker process, so memory is returned
    between runs and two datasets are processed at once."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        rows = list(pool.map(one_run, seeds))
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c != "seed"]
    return df[cols].astype(float).agg(["mean", "min", "max"]).T


if __name__ == "__main__":
    df = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(summarize(df).round(3))
