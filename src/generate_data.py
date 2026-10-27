"""Generate two years of weekly store sales for Tallow Creek Crayon Co., a
fictional crayon maker, with the true demand behind them saved separately.

Everything is invented: the company, the stores and every number. The point of
fake data with a known truth is that an analysis can be checked, not just shown.

The factory sells eight crayon bundles through 60 retail stores in four
channels. Retailers send weekly point-of-sale data back to the factory: units
sold, shelf price, whether the bundle was on promotion or on an end-cap display.
The factory pays for promotions: it refunds half of each discount to the
retailer for every unit sold on promotion (a "scan-down"; the retailer funds
the other half) and pays $3 for each store-week on display. That is trade spend.

Outputs:
    data/bundles.csv              the eight bundles, with prices and unit cost
    data/stores.csv               the 60 stores
    data/weekly_sales.csv         one row per store, bundle and week it was on the shelf
    data/answer_key/true_*.csv    the demand parameters the sales were drawn from.
                                  The analysis never reads these. They are only
                                  used to check its answers at the end.

Run:  python src/generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N_WEEKS = 104
START = pd.Timestamp("2024-01-01")          # a Monday
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DISPLAY_FEE = 3.00           # paid to the retailer per store-week on display
FACTORY_SHARE = 0.50         # share of each promotional discount the factory refunds
DISPERSION = 6.0             # negative binomial size; lower = noisier weeks
PRICE_NOISE = 0.04           # week-to-week shelf price wobble (log sd)

# --------------------------------------------------------------------------- #
# Bundles. `appeal` is the true baseline demand: units per week in an average
# store, at the 2024 list price, with no promotion and no display, outside the
# two peak seasons. The analysis is not told these numbers.
# --------------------------------------------------------------------------- #
BUNDLES = pd.DataFrame([
    # id,  name,                count, theme,       list,  wholesale, cost, appeal, elasticity, bts,  holiday, promo_rate, list_2025
    ("B01", "Classic 8",            8, "primary",    1.99, 1.05, 0.38, 5.0, -2.2, 1.40, 1.00, 0.07, 2.09),
    ("B02", "Classic 24",          24, "primary",    3.49, 1.85, 0.80, 5.5, -1.8, 1.50, 1.00, 0.08, 3.69),
    ("B03", "Pastel Garden 16",    16, "themed",     2.99, 1.60, 0.62, 2.6, -1.6, 1.00, 1.10, 0.05, 2.99),
    ("B04", "Ocean & Sky 16",      16, "themed",     2.99, 1.60, 0.62, 3.2, -1.7, 1.00, 1.00, 0.05, 2.99),
    ("B05", "Neon Pop 12",         12, "novelty",    2.79, 1.45, 0.66, 2.4, -2.6, 1.20, 1.00, 0.13, 2.79),
    ("B06", "Metallic Shine 12",   12, "novelty",    4.49, 2.45, 0.95, 2.2, -1.1, 1.00, 1.60, 0.01, 4.49),
    ("B07", "Skin Tones 24",       24, "skin tones", 3.99, 2.15, 0.85, 6.5, -1.3, 1.30, 1.00, 0.02, 3.99),
    ("B08", "Big Box 64",          64, "assorted",   7.99, 4.30, 2.10, 1.6, -1.5, 1.10, 2.20, 0.04, 8.49),
], columns=["bundle_id", "name", "crayons", "theme", "list_price", "wholesale_price",
            "unit_cost", "appeal", "elasticity", "bts_mult", "holiday_mult",
            "promo_rate", "list_price_2025"])

DISPLAY_LIFT = 1.5           # true multiplier on demand while on an end-cap

# Some bundles sell better in some channels. The analysis model is deliberately
# NOT told about this, so it is fitted to a world slightly richer than it can
# describe, the way a real model always is.
CHANNEL_AFFINITY = {
    ("School & office", "B02"): 1.25, ("School & office", "B07"): 1.25,
    ("School & office", "B08"): 0.80, ("Drugstore", "B05"): 1.25,
    ("Drugstore", "B06"): 0.80, ("Big-box", "B08"): 1.20,
}
LAUNCH = {"B07": 36}         # Skin Tones 24 launches in week 36 (early Sept 2024)

CHANNELS = {                 # n stores, traffic multiplier, shelf price vs list
    "Big-box":               (12, 1.8, 0.95),
    "School & office":       (12, 1.4, 1.00),
    "Grocery":               (20, 0.8, 1.04),
    "Drugstore":             (16, 0.6, 1.10),
}
REGIONS = ["Northeast", "Midwest", "South", "West"]


def week_starts():
    return pd.date_range(START, periods=N_WEEKS, freq="7D")


def season_flags(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Back-to-school runs mid-July to mid-September, holidays mid-November to
    Christmas. Both are known to anyone in the business, so the analysis uses
    these same windows."""
    md = weeks.month * 100 + weeks.day
    bts = (md >= 715) & (md <= 915)
    holiday = (md >= 1115) & (md <= 1224)
    return pd.DataFrame({"bts": bts.astype(int), "holiday": holiday.astype(int)}, index=weeks)


def shared_season(weeks: pd.DatetimeIndex) -> np.ndarray:
    """How busy the crayon aisle is overall, for every bundle alike."""
    doy = weeks.dayofyear.to_numpy()
    f = season_flags(weeks)
    curve = 1 + 0.10 * np.sin(2 * np.pi * (doy - 80) / 365)       # gentle spring/fall swell
    curve = curve * np.where(f.bts, 2.0, 1.0) * np.where(f.holiday, 1.3, 1.0)
    curve = curve * np.where(weeks.month == 1, 0.8, 1.0)            # January lull
    return curve


def make_stores(rng) -> pd.DataFrame:
    rows, n = [], 1
    for channel, (count, traffic, price_factor) in CHANNELS.items():
        for _ in range(count):
            rows.append({"store_id": f"S{n:03d}", "channel": channel,
                         "region": REGIONS[int(rng.integers(0, 4))],
                         "traffic": traffic * float(rng.lognormal(0, 0.25)),
                         "price_factor": price_factor})
            n += 1
    s = pd.DataFrame(rows)
    # normalise so the average store has traffic 1.0; "appeal" is per average store
    s["traffic"] = s["traffic"] / s["traffic"].mean()
    return s


def make_assortment(stores: pd.DataFrame, rng) -> dict:
    """Which stores carry which bundle. Deliberately uneven."""
    ids = stores.store_id.to_numpy()
    ch = stores.set_index("store_id").channel
    big = ch.isin(["Big-box", "School & office"])
    a = {
        "B01": set(ids),
        "B02": set(ids),
        "B03": set(rng.choice(ids, 42, replace=False)),
        "B04": set(rng.choice(ids, 36, replace=False)),
        "B05": set(rng.choice(ids, 51, replace=False)),
        # Metallic sits in the high-traffic stores, plus a few others
        "B06": set(ch.index[big]) | set(rng.choice(ch.index[~big], 3, replace=False)),
        # Big Box 64 needs shelf space: big stores and half the grocers
        "B08": set(ch.index[big]) | set(rng.choice(ch.index[ch == "Grocery"], 10, replace=False)),
    }
    # Skin Tones 24: launched in 18 stores (mostly smaller ones), 12 more join in week 70
    small = ch.index[~big]
    first = list(rng.choice(small, 14, replace=False)) + list(rng.choice(ch.index[big], 4, replace=False))
    later = list(rng.choice([s for s in ids if s not in first], 12, replace=False))
    a["B07"] = {s: LAUNCH["B07"] for s in first} | {s: 70 for s in later}
    return a


def true_mean(appeal, traffic, season, bts, holiday, bts_mult, holiday_mult,
              elasticity, price_ratio, display):
    """Expected weekly units. The single source of truth for demand."""
    return (appeal * traffic * season
            * np.where(bts == 1, bts_mult, 1.0)
            * np.where(holiday == 1, holiday_mult, 1.0)
            * price_ratio ** elasticity
            * np.where(display == 1, DISPLAY_LIFT, 1.0))


def generate(seed: int = SEED):
    rng = np.random.default_rng(seed)
    weeks = week_starts()
    flags = season_flags(weeks)
    season = shared_season(weeks)
    stores = make_stores(rng)
    assortment = make_assortment(stores, rng)
    b = BUNDLES.set_index("bundle_id")

    rows = []
    for bid in b.index:
        carried = assortment[bid]
        for _, st in stores.iterrows():
            if st.store_id not in carried:
                continue
            first_week = carried[st.store_id] if isinstance(carried, dict) else 0
            for w in range(first_week, N_WEEKS):
                wk = weeks[w]
                list_now = b.at[bid, "list_price"] if wk.year == 2024 else b.at[bid, "list_price_2025"]
                regular = round(list_now * st.price_factor * float(rng.lognormal(0, PRICE_NOISE)), 2)
                # promotions are planned into the busy seasons, which is what
                # makes a naive "sales were higher on promo" comparison misleading
                busy = flags.bts.iat[w] if bid != "B08" else flags.holiday.iat[w]
                p_promo = b.at[bid, "promo_rate"] * (2.5 if busy else 1.0)
                on_promo = int(rng.random() < p_promo)
                depth = float(rng.choice([0.20, 0.25, 0.30])) if on_promo else 0.0
                price = round(regular * (1 - depth), 2)
                on_display = int(rng.random() < (0.6 if on_promo else 0.02))
                affinity = CHANNEL_AFFINITY.get((st.channel, bid), 1.0)
                mu = true_mean(b.at[bid, "appeal"] * affinity, st.traffic, season[w], flags.bts.iat[w],
                               flags.holiday.iat[w], b.at[bid, "bts_mult"],
                               b.at[bid, "holiday_mult"], b.at[bid, "elasticity"],
                               price / b.at[bid, "list_price"], on_display)
                units = int(rng.negative_binomial(DISPERSION, DISPERSION / (DISPERSION + mu)))
                scan_down = round(FACTORY_SHARE * (regular - price) * units, 2) if on_promo else 0.0
                rows.append({
                    "week_start": wk.date(), "store_id": st.store_id, "bundle_id": bid,
                    "regular_price": regular, "shelf_price": price,
                    "on_promo": on_promo, "on_display": on_display, "units": units,
                    "trade_spend": round(scan_down + (DISPLAY_FEE if on_display else 0.0), 2),
                })
    sales = pd.DataFrame(rows).sort_values(["week_start", "store_id", "bundle_id"])
    sales = sales.reset_index(drop=True)

    bundles_pub = BUNDLES[["bundle_id", "name", "crayons", "theme", "list_price",
                           "list_price_2025", "wholesale_price", "unit_cost"]].copy()
    bundles_pub["launch_week"] = bundles_pub.bundle_id.map(
        lambda x: weeks[LAUNCH[x]].date() if x in LAUNCH else weeks[0].date())
    stores_pub = stores[["store_id", "channel", "region"]]

    truth = answer_key(stores, weeks)
    return bundles_pub, stores_pub, sales, truth


def answer_key(stores: pd.DataFrame, weeks: pd.DatetimeIndex) -> dict:
    """The true quantities the analysis should recover.

    baseline_velocity: expected units per store-week if all 60 stores had carried
    the bundle for all 104 weeks at the 2024 list price, with no promotion and
    no display. It is the fair, like-for-like measure of how well a bundle sells.
    """
    flags = season_flags(weeks)
    season = shared_season(weeks)
    out = []
    for _, r in BUNDLES.iterrows():
        affinity = stores.channel.map(lambda c: CHANNEL_AFFINITY.get((c, r.bundle_id), 1.0))
        grid = true_mean(r.appeal * affinity.to_numpy()[:, None],
                         stores.traffic.to_numpy()[:, None], season[None, :],
                         flags.bts.to_numpy()[None, :], flags.holiday.to_numpy()[None, :],
                         r.bts_mult, r.holiday_mult, r.elasticity, 1.0, 0)
        out.append({"bundle_id": r.bundle_id, "name": r.name,
                    "baseline_velocity": float(grid.mean()),
                    "elasticity": r.elasticity,
                    "bts_mult": r.bts_mult, "holiday_mult": r.holiday_mult})
    params = pd.DataFrame(out)
    params["baseline_margin_per_store_week"] = (
        params.baseline_velocity
        * (BUNDLES.wholesale_price - BUNDLES.unit_cost).to_numpy())
    globals_ = pd.DataFrame([{"parameter": "display_lift", "value": DISPLAY_LIFT},
                             {"parameter": "dispersion", "value": DISPERSION}])
    return {"true_bundle_parameters": params, "true_global_parameters": globals_,
            "true_store_traffic": stores[["store_id", "channel", "traffic"]]}


def main():
    bundles, stores, sales, truth = generate(SEED)
    (DATA_DIR / "answer_key").mkdir(parents=True, exist_ok=True)
    bundles.to_csv(DATA_DIR / "bundles.csv", index=False)
    stores.to_csv(DATA_DIR / "stores.csv", index=False)
    sales.to_csv(DATA_DIR / "weekly_sales.csv", index=False)
    for name, df in truth.items():
        df.to_csv(DATA_DIR / "answer_key" / f"{name}.csv", index=False)
    print(f"{len(sales):,} store-bundle-weeks, {sales.units.sum():,} units, "
          f"${sales.trade_spend.sum():,.0f} trade spend")


if __name__ == "__main__":
    main()
