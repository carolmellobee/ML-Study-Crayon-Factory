"""Charts for the README and notebook. One quiet style; color only where it means something."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
MUTED = "#b9b8b3"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#e34948"
NEUTRAL = "#f0efec"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.major.size": 0, "ytick.major.size": 0,
    "font.size": 10, "legend.frameon": False, "legend.labelcolor": TEXT_2,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def _subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=TEXT_2, fontsize=9.5, va="bottom")


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def weekly_units(s: pd.DataFrame, path=None):
    w = s.groupby("week_start").units.sum()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.grid(axis="x", visible=False)
    for flag, label in [("bts", "Back-to-school"), ("holiday", "Holidays")]:
        on = s.groupby("week_start")[flag].first()
        for start in on.index[(on == 1) & (on.shift(fill_value=0) == 0)]:
            after = on[(on.index > start) & (on == 0)]
            end = after.index[0] if len(after) else on.index[-1] + pd.Timedelta(days=7)
            ax.axvspan(start, end, color=NEUTRAL, zorder=0, lw=0)
            if start.year == 2024:
                ax.text(start, w.max() * 1.02, label, color=TEXT_2, fontsize=8.5, va="bottom")
    ax.plot(w.index, w.values, color=BLUE, zorder=3)
    ax.set_ylim(0, w.max() * 1.12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0")
    ax.set_title("Crayon boxes sold per week, all stores", pad=26)
    _subtitle(ax, "Back-to-school triples the aisle. Any fair comparison has to account for when a bundle was on sale.")
    fig.tight_layout()
    return _save(fig, path)


def three_rankings(table: pd.DataFrame, highlight: dict, path=None):
    """table: index bundle name; columns are the three measures, left to right.
    highlight: name -> color for the bundles the story is about; others are gray."""
    cols = list(table.columns)
    ranks = table.rank(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.grid(False)
    x = np.arange(len(cols))
    for name, r in ranks.iterrows():
        c = highlight.get(name, MUTED)
        z = 3 if name in highlight else 2
        lw = 2.5 if name in highlight else 1.5
        ax.plot(x, r.values, color=c, lw=lw, zorder=z, marker="o", ms=7,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(x[0] - 0.06, r.iloc[0], name, ha="right", va="center", fontsize=9,
                color=TEXT if name in highlight else TEXT_2)
        ax.text(x[-1] + 0.06, r.iloc[-1], f"#{int(r.iloc[-1])}  {name}", ha="left",
                va="center", fontsize=9, color=TEXT if name in highlight else TEXT_2)
    ax.set_xticks(x, cols)
    ax.xaxis.tick_top()
    ax.set_ylim(len(table) + 0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlim(-0.9, len(cols) - 1 + 0.9)
    ax.set_title("Which bundle sells best depends on what you divide by", pad=40)
    ax.text(0, 1.11, "Rank of each bundle under three measures. 1 is best.",
            transform=ax.transAxes, color=TEXT_2, fontsize=9.5)
    fig.tight_layout()
    return _save(fig, path)


def channel_heatmap(resid: pd.DataFrame, names: pd.Series, path=None):
    """Observed / predicted units by bundle and channel, on a diverging scale."""
    cmap = LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
    data = resid.copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    im = ax.imshow(np.log(data.to_numpy(dtype=float)), cmap=cmap, vmin=-0.3, vmax=0.3,
                   aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.iat[i, j]
            txt = "not sold" if pd.isna(v) else f"{v:.2f}"
            strong = (not pd.isna(v)) and abs(np.log(v)) > 0.18
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="#ffffff" if strong else (TEXT_2 if pd.isna(v) else TEXT))
    ax.set_xticks(range(data.shape[1]), data.columns)
    ax.set_yticks(range(data.shape[0]), names.reindex(data.index))
    ax.xaxis.tick_top()
    ax.set_title("Where the basic model is wrong", pad=44)
    ax.text(0, 1.13, "Units sold divided by units the model predicted. 1.00 means the model was right.",
            transform=ax.transAxes, color=TEXT_2, fontsize=9.5)
    fig.tight_layout()
    return _save(fig, path)


def elasticity_dots(est: dict, truth: pd.Series, names: pd.Series, path=None):
    """est: label -> DataFrame(estimate, low, high) indexed by bundle_id."""
    order = truth.sort_values().index
    colors = {list(est)[0]: ORANGE, list(est)[1]: BLUE}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.grid(axis="y", visible=False)
    y = np.arange(len(order))
    off = {list(est)[0]: -0.16, list(est)[1]: 0.16}
    for label, df in est.items():
        d = df.reindex(order)
        ax.hlines(y + off[label], d.low, d.high, color=colors[label], lw=2, zorder=2)
        ax.plot(d.estimate, y + off[label], "o", color=colors[label], ms=7, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{label} (95% interval)")
    ax.plot(truth.reindex(order), y, "D", color=TEXT, ms=5, zorder=4, label="True value")
    ax.set_yticks(y, names.reindex(order))
    ax.set_xlabel("Price elasticity: % change in units for a 1% change in price")
    ax.set_title("How price-sensitive is each bundle?", pad=26)
    _subtitle(ax, "Ignoring channel preferences makes Big Box 64 and Skin Tones 24 look more price-sensitive than they are.")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=3)
    fig.tight_layout()
    return _save(fig, path)


def promo_roi_bars(roi: pd.Series, names: pd.Series, path=None):
    r = roi.sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.grid(axis="y", visible=False)
    colors = [BLUE if v >= 0 else RED for v in r.values]
    y = np.arange(len(r))
    ax.barh(y, r.values * 100, height=0.55, color=colors, zorder=2)
    ax.axvline(0, color=TEXT_2, lw=0.8, zorder=3)
    for yi, v in zip(y, r.values):
        ax.text(v * 100 + (1 if v >= 0 else -1), yi, f"{v:+.0%}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=TEXT)
    ax.set_yticks(y, names.reindex(r.index))
    lim = max(abs(r.values * 100)) + 8
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0")
    ax.set_title("Return on promotion spend, by bundle", pad=26)
    _subtitle(ax, "Profit on the extra units sold, minus what the promotions cost, per dollar of trade spend.")
    fig.tight_layout()
    return _save(fig, path)


def velocity_vs_truth(ci: pd.DataFrame, truth: pd.Series, names: pd.Series, path=None):
    order = ci.estimate.sort_values().index
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.grid(axis="y", visible=False)
    y = np.arange(len(order))
    d = ci.reindex(order)
    ax.barh(y, d.estimate, height=0.55, color=BLUE, zorder=2, label="Model estimate")
    ax.hlines(y, d.low, d.high, color=TEXT_2, lw=1.2, zorder=3)
    ax.plot(truth.reindex(order), y, "D", color=TEXT, ms=5, zorder=4, label="True value")
    ax.set_yticks(y, names.reindex(order))
    ax.set_xlabel("Boxes per store per week, like for like")
    ax.set_title("Baseline sales: the model against the answer key", pad=26)
    _subtitle(ax, "Every store, every week, list price, no promotion. Lines show 95% intervals.")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)
