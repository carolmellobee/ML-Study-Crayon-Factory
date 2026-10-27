"""Which crayon bundles sell more? Three answers, from naive to fair.

1. Total units: what a sales report shows. Rewards whatever is on the most
   shelves for the longest, and whatever was promoted hardest.
2. Velocity: units per store-week where the bundle was on the shelf. Fixes
   distribution, but still rewards bundles placed in busy stores or promoted.
3. Baseline velocity from a model: what each bundle would sell in the same
   stores, the same weeks, at its list price, with no promotion. This is the
   like-for-like answer.

The model is a Poisson regression (log link) on weekly units with:
    store fixed effects      every store's traffic, whatever bundles it carries
    week fixed effects       the crayon calendar, shared by all bundles
    bundle effects           the thing we want
    bundle x season          bundles that peak harder at back-to-school or holidays
    bundle x log(price)      a price elasticity for each bundle
    display                  the lift from an end-cap display
Standard errors are clustered by store, because weeks in the same store are not
independent and the counts are noisier than Poisson assumes.

Nothing in this file reads data/answer_key. That is used only in validate().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REF = "B01"          # reference bundle for the dummy coding; the choice does not change results
REF_CHANNEL = "Grocery"  # channel the others are compared with (see channel_bundle below)


# --------------------------------------------------------------------------- #
# Loading and season windows
# --------------------------------------------------------------------------- #
def season_flags(weeks: pd.Series) -> pd.DataFrame:
    """Back-to-school (Jul 15 - Sep 15) and holiday (Nov 15 - Dec 24) windows."""
    w = pd.to_datetime(weeks)
    md = w.dt.month * 100 + w.dt.day
    return pd.DataFrame({"bts": ((md >= 715) & (md <= 915)).astype(int),
                         "holiday": ((md >= 1115) & (md <= 1224)).astype(int)},
                        index=weeks.index)


def load(data_dir: Path = DATA_DIR):
    sales = pd.read_csv(data_dir / "weekly_sales.csv", parse_dates=["week_start"])
    bundles = pd.read_csv(data_dir / "bundles.csv")
    stores = pd.read_csv(data_dir / "stores.csv")
    return prepare(sales, bundles, stores), bundles, stores


def prepare(sales: pd.DataFrame, bundles: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    s["channel"] = s.store_id.map(stores.set_index("store_id").channel)
    s["week_start"] = pd.to_datetime(s["week_start"])
    s = s.join(season_flags(s.week_start))
    ref_price = s.bundle_id.map(bundles.set_index("bundle_id").list_price)
    # price relative to the 2024 list price, so "list price" means the same
    # thing in both years
    s["log_price_ratio"] = np.log(s.shelf_price / ref_price)
    return s


# --------------------------------------------------------------------------- #
# The two naive answers
# --------------------------------------------------------------------------- #
def naive_table(s: pd.DataFrame, bundles: pd.DataFrame) -> pd.DataFrame:
    b = bundles.set_index("bundle_id")
    g = s.groupby("bundle_id").agg(units=("units", "sum"),
                                   store_weeks=("units", "size"),
                                   stores=("store_id", "nunique"),
                                   promo_share=("on_promo", "mean"),
                                   trade_spend=("trade_spend", "sum"))
    g["velocity"] = g.units / g.store_weeks
    g["unit_margin"] = b.wholesale_price - b.unit_cost
    g["net_revenue"] = g.units * b.wholesale_price - g.trade_spend   # ASC 606: trade spend reduces revenue
    g["gross_profit"] = g.net_revenue - g.units * b.unit_cost
    g.insert(0, "name", b.name)
    return g


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
def design(s: pd.DataFrame, stores, weeks, bundle_ids, channel_bundle=False) -> pd.DataFrame:
    """Build the regression matrix explicitly, so every column is visible.

    channel_bundle=True adds a term for each bundle in each channel, measured
    against grocery. Where a bundle was never sold in a channel there is no data
    to learn from, and the model assumes it would sell there as it does in grocery."""
    X = pd.DataFrame(index=s.index)
    X["const"] = 1.0
    st = pd.Categorical(s.store_id, categories=stores)
    wk = pd.Categorical(pd.to_datetime(s.week_start).dt.strftime("%Y-%m-%d"), categories=weeks)
    bd = pd.Categorical(s.bundle_id, categories=bundle_ids)
    X = X.join(pd.get_dummies(st, prefix="store", drop_first=True, dtype=float).set_index(s.index))
    X = X.join(pd.get_dummies(wk, prefix="week", drop_first=True, dtype=float).set_index(s.index))
    bd_d = pd.get_dummies(bd, dtype=float).set_index(s.index)
    for b in bundle_ids:
        if b != REF:
            X[f"bundle_{b}"] = bd_d[b]
            # the reference bundle's seasonality is absorbed by the week effects
            X[f"bts_{b}"] = bd_d[b] * s.bts
            X[f"holiday_{b}"] = bd_d[b] * s.holiday
    for b in bundle_ids:
        X[f"elasticity_{b}"] = bd_d[b] * s.log_price_ratio
    X["display"] = s.on_display.astype(float)
    if channel_bundle:
        for ch in CHANNELS:
            if ch == REF_CHANNEL:
                continue
            for b in bundle_ids:
                if b != REF:
                    X[f"cb_{ch}_{b}"] = ((s.channel == ch) & (s.bundle_id == b)).astype(float)
    return X


CHANNELS = ["Big-box", "Drugstore", "Grocery", "School & office"]


def drop_redundant(X: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are exact combinations of others.

    This happens when a bundle was never sold in the reference channel (say,
    no grocery store stocked it): its channel terms then add up to its bundle
    term. Dropping one of them makes another channel that bundle's reference.
    Fitted values for the observed data are unchanged."""
    from scipy.linalg import qr
    A = X.to_numpy()
    _, r, piv = qr(A, mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    rank = int((d > d.max() * 1e-9).sum())
    if rank == A.shape[1]:
        return X
    dropped = X.columns[piv[rank:]]
    assert all(c.startswith("cb_") for c in dropped), f"unexpected collinearity: {list(dropped)}"
    return X.drop(columns=dropped)


class DemandModel:
    def __init__(self, s: pd.DataFrame, channel_bundle: bool = False):
        self.s = s
        self.channel_bundle = channel_bundle
        self.stores = sorted(s.store_id.unique())
        self.store_channel = s.groupby("store_id").channel.first().reindex(self.stores)
        self.weeks = sorted(s.week_start.dt.strftime("%Y-%m-%d").unique())
        self.bundles = sorted(s.bundle_id.unique())
        X = design(s, self.stores, self.weeks, self.bundles, channel_bundle)
        X = X.loc[:, X.abs().sum() > 0]     # drop bundle-channel pairs that never happened
        X = drop_redundant(X)
        self.columns = X.columns
        groups = pd.Categorical(s.store_id).codes
        self.fit = sm.GLM(s.units.to_numpy(), X.to_numpy(), family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
        self.params = pd.Series(self.fit.params, index=self.columns)
        self.cov = pd.DataFrame(self.fit.cov_params(), index=self.columns, columns=self.columns)

    # ---- pieces of the linear predictor ----------------------------------
    def _store_effects(self, p):
        return np.array([0.0] + [p[f"store_{x}"] for x in self.stores[1:]])

    def _week_effects(self, p):
        return np.array([0.0] + [p[f"week_{x}"] for x in self.weeks[1:]])

    def baseline_velocity(self, params: pd.Series | None = None) -> pd.Series:
        """Expected units per store-week in every store, every week, at the 2024
        list price, no promotion and no display. The model splits into a store
        part and a week part, so the average over the 60 x 104 grid is exact."""
        p = self.params if params is None else params
        store_fx = self._store_effects(p)
        flags = season_flags(pd.Series(pd.to_datetime(self.weeks)))
        week = self._week_effects(p)
        out = {}
        for b in self.bundles:
            cb = np.array([p.get(f"cb_{ch}_{b}", 0.0) for ch in self.store_channel])
            store = np.exp(store_fx + cb).mean()
            lin = week.copy()
            if b != REF:
                lin = lin + p[f"bundle_{b}"] + flags.bts.to_numpy() * p[f"bts_{b}"] \
                      + flags.holiday.to_numpy() * p[f"holiday_{b}"]
            out[b] = float(np.exp(p["const"]) * store * np.exp(lin).mean())
        return pd.Series(out, name="baseline_velocity")

    def baseline_velocity_ci(self, n_draws: int = 1000, seed: int = 0) -> pd.DataFrame:
        """95% intervals by simulating coefficients from their estimated
        (cluster-robust) sampling distribution."""
        rng = np.random.default_rng(seed)
        # With 60 store clusters the robust covariance of ~190 coefficients has
        # rank of at most 60. That is normal and does not bias any single
        # interval, but it rules out a Cholesky factor, so use eigenvalues.
        draws = rng.multivariate_normal(self.params.to_numpy(), self.cov.to_numpy(), n_draws,
                                        method="eigh", check_valid="ignore")
        sims = pd.DataFrame([self.baseline_velocity(pd.Series(d, index=self.columns))
                             for d in draws])
        est = self.baseline_velocity()
        return pd.DataFrame({"estimate": est, "low": sims.quantile(0.025),
                             "high": sims.quantile(0.975)})

    def elasticities(self) -> pd.DataFrame:
        rows = []
        for b in self.bundles:
            c = f"elasticity_{b}"
            se = np.sqrt(self.cov.at[c, c])
            rows.append({"bundle_id": b, "estimate": self.params[c],
                         "low": self.params[c] - 1.96 * se, "high": self.params[c] + 1.96 * se})
        return pd.DataFrame(rows).set_index("bundle_id")

    def display_lift(self) -> tuple[float, float, float]:
        se = np.sqrt(self.cov.at["display", "display"])
        b = self.params["display"]
        return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Expected units for any rows shaped like the sales table."""
        f = frame.copy()
        if "channel" not in f:
            f["channel"] = f.store_id.map(self.store_channel)
        X = design(f, self.stores, self.weeks, self.bundles, self.channel_bundle)
        X = X.reindex(columns=self.columns, fill_value=0.0)
        return np.exp(X.to_numpy() @ self.params.to_numpy())


# --------------------------------------------------------------------------- #
# Promotions: what they cost and what they bought
# --------------------------------------------------------------------------- #
def naive_promo_lift(s: pd.DataFrame) -> pd.Series:
    """Average units on promotion divided by average units off it."""
    m = s.groupby(["bundle_id", "on_promo"]).units.mean().unstack()
    return (m[1] / m[0]).rename("naive_lift")


def promo_roi(model: DemandModel, s: pd.DataFrame, bundles: pd.DataFrame) -> pd.DataFrame:
    """For every store-week with a promotion or display, compare what sold with
    what the model says would have sold at the regular price, off display.
    The factory earns its unit margin on the extra units and pays the trade spend."""
    b = bundles.set_index("bundle_id")
    ev = s[(s.on_promo == 1) | (s.on_display == 1)].copy()
    cf = ev.copy()
    cf["log_price_ratio"] = np.log(cf.regular_price / cf.bundle_id.map(b.list_price))
    cf["on_display"] = 0
    ev["baseline_units"] = model.predict(cf)
    ev["model_units"] = model.predict(ev)
    ev["incremental_units"] = ev.units - ev.baseline_units
    margin = ev.bundle_id.map(b.wholesale_price - b.unit_cost)
    ev["incremental_profit"] = ev.incremental_units * margin
    g = ev.groupby("bundle_id").agg(events=("units", "size"), units=("units", "sum"),
                                    baseline_units=("baseline_units", "sum"),
                                    incremental_units=("incremental_units", "sum"),
                                    incremental_profit=("incremental_profit", "sum"),
                                    trade_spend=("trade_spend", "sum"))
    g["model_lift"] = g.units / g.baseline_units
    g["net_return"] = g.incremental_profit - g.trade_spend
    g["roi"] = g.net_return / g.trade_spend
    g.insert(0, "name", b.name)
    return g


def roi_under_terms(model: DemandModel, s: pd.DataFrame, bundles: pd.DataFrame,
                    factory_share: float, display_fee: float) -> pd.Series:
    """Promotion ROI if the same promotions had run under different trade terms.
    The units sold do not change, only what the factory pays for them."""
    base = promo_roi(model, s, bundles)
    ev = s[(s.on_promo == 1) | (s.on_display == 1)]
    spend = (factory_share * (ev.regular_price - ev.shelf_price) * ev.units * ev.on_promo
             + display_fee * ev.on_display).groupby(ev.bundle_id).sum()
    return (base.incremental_profit - spend) / spend


# --------------------------------------------------------------------------- #
# What-if: carry a bundle everywhere
# --------------------------------------------------------------------------- #
def channel_residuals(model: DemandModel) -> pd.DataFrame:
    """Actual units divided by the model's predicted units, for every bundle in
    every channel. A well-specified model gives about 1.00 everywhere."""
    s = model.s.assign(pred=model.predict(model.s))
    g = s.groupby(["bundle_id", "channel"])[["units", "pred"]].sum()
    return (g.units / g.pred).unstack()


def full_distribution_gain(model: DemandModel, s: pd.DataFrame, bundles: pd.DataFrame,
                           bundle_id: str, year: int = 2025) -> pd.Series:
    """Extra units and gross profit if every store had carried the bundle for
    the whole year, at regular price, no promotion. Ignores cannibalisation."""
    b = bundles.set_index("bundle_id")
    weeks = [pd.Timestamp(w) for w in model.weeks if pd.Timestamp(w).year == year]
    have = s[(s.bundle_id == bundle_id) & (s.week_start.dt.year == year)]
    grid = pd.MultiIndex.from_product([model.stores, weeks], names=["store_id", "week_start"])
    missing = grid.difference(pd.MultiIndex.from_frame(have[["store_id", "week_start"]]))
    f = missing.to_frame(index=False)
    f["bundle_id"] = bundle_id
    f["week_start"] = pd.to_datetime(f.week_start)
    f = f.join(season_flags(f.week_start))
    list_now = b.at[bundle_id, "list_price_2025" if year == 2025 else "list_price"]
    f["log_price_ratio"] = np.log(list_now / b.at[bundle_id, "list_price"])
    f["on_display"] = 0
    units = model.predict(f).sum()
    margin = b.at[bundle_id, "wholesale_price"] - b.at[bundle_id, "unit_cost"]
    return pd.Series({"new_store_weeks": len(f), "extra_units": units,
                      "extra_gross_profit": units * margin,
                      "stores_carrying_now": have.store_id.nunique()})


# --------------------------------------------------------------------------- #
# Validation against the answer key
# --------------------------------------------------------------------------- #
def load_truth(data_dir: Path = DATA_DIR):
    k = data_dir / "answer_key"
    return (pd.read_csv(k / "true_bundle_parameters.csv").set_index("bundle_id"),
            pd.read_csv(k / "true_global_parameters.csv").set_index("parameter").value,
            pd.read_csv(k / "true_store_traffic.csv").set_index("store_id"))


def true_promo_roi(s: pd.DataFrame, bundles: pd.DataFrame, truth, traffic):
    """The same promo calculation, using the true demand instead of the model's.
    Needs the generator's season curve, so it imports it."""
    import generate_data as G
    b = bundles.set_index("bundle_id")
    ev = s[(s.on_promo == 1) | (s.on_display == 1)].copy()
    season = pd.Series(G.shared_season(G.week_starts()), index=G.week_starts())
    t = truth.loc[ev.bundle_id]
    appeal = ev.bundle_id.map(G.BUNDLES.set_index("bundle_id").appeal).to_numpy()
    channel = ev.store_id.map(traffic.channel)
    affinity = np.array([G.CHANNEL_AFFINITY.get((c, bid), 1.0)
                         for c, bid in zip(channel, ev.bundle_id)])
    base = G.true_mean(appeal * affinity, ev.store_id.map(traffic.traffic).to_numpy(),
                       ev.week_start.map(season).to_numpy(), ev.bts.to_numpy(),
                       ev.holiday.to_numpy(), t.bts_mult.to_numpy(), t.holiday_mult.to_numpy(),
                       t.elasticity.to_numpy(),
                       (ev.regular_price / ev.bundle_id.map(b.list_price)).to_numpy(), 0)
    margin = ev.bundle_id.map(b.wholesale_price - b.unit_cost)
    ev["incremental_profit"] = (ev.units - base) * margin
    g = ev.groupby("bundle_id")[["incremental_profit", "trade_spend"]].sum()
    return ((g.incremental_profit - g.trade_spend) / g.trade_spend).rename("true_roi")


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.reindex(a.index).rank()))


if __name__ == "__main__":
    s, bundles, stores = load()
    print(naive_table(s, bundles).round(2))
    m = DemandModel(s)
    print(m.baseline_velocity_ci(300).round(2))
    print(m.elasticities().round(2))
    print("display lift", np.round(m.display_lift(), 3))
    print(promo_roi(m, s, bundles).round(2))
