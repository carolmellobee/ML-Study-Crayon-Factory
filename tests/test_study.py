import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analysis as A  # noqa: E402
import generate_data as G  # noqa: E402


@pytest.fixture(scope="module")
def data():
    s, bundles, stores = A.load()
    return s, bundles, stores


def test_one_row_per_store_bundle_week(data):
    s = data[0]
    assert not s.duplicated(["week_start", "store_id", "bundle_id"]).any()


def test_units_are_non_negative_whole_numbers(data):
    s = data[0]
    assert (s.units >= 0).all() and (s.units == s.units.round()).all()


def test_promo_price_is_below_regular_price(data):
    s = data[0]
    promo = s[s.on_promo == 1]
    assert (promo.shelf_price < promo.regular_price).all()
    assert (s[s.on_promo == 0].shelf_price == s[s.on_promo == 0].regular_price).all()


def test_trade_spend_matches_the_terms(data):
    s = data[0]
    expected = (G.FACTORY_SHARE * (s.regular_price - s.shelf_price) * s.units
                + G.DISPLAY_FEE * s.on_display)
    assert np.allclose(s.trade_spend, expected, atol=0.011)


def test_skin_tones_only_sold_after_launch(data):
    s, bundles = data[0], data[1]
    launch = pd.Timestamp(bundles.set_index("bundle_id").at["B07", "launch_week"])
    assert s[s.bundle_id == "B07"].week_start.min() == launch


def test_season_windows():
    w = pd.Series(pd.to_datetime(["2024-07-15", "2024-07-08", "2024-09-15", "2024-12-23",
                                  "2024-12-30"]))
    f = A.season_flags(w)
    assert f.bts.tolist() == [1, 0, 1, 0, 0]
    assert f.holiday.tolist() == [0, 0, 0, 1, 0]


def test_generator_is_reproducible():
    a = G.generate(3)[2]
    b = G.generate(3)[2]
    pd.testing.assert_frame_equal(a, b)


def test_model_recovers_a_world_it_can_describe():
    """With channel effects switched off in the generator, the basic model has
    exactly the right form and should land close to the truth."""
    saved = dict(G.CHANNEL_AFFINITY)
    G.CHANNEL_AFFINITY.clear()
    try:
        bundles, stores, sales, truth = G.generate(11)
    finally:
        G.CHANNEL_AFFINITY.update(saved)
    t = truth["true_bundle_parameters"].set_index("bundle_id")
    m = A.DemandModel(A.prepare(sales, bundles, stores))
    est = m.baseline_velocity()
    assert (abs(est / t.baseline_velocity - 1) < 0.05).all()
    assert (abs(m.elasticities().estimate - t.elasticity) < 0.35).all()
    assert abs(m.display_lift()[0] - G.DISPLAY_LIFT) < 0.1


def test_redundant_channel_terms_are_dropped():
    """If a bundle is never stocked in the reference channel, its channel terms
    add up to its bundle term; the fit must still work."""
    X = pd.DataFrame({"const": 1.0,
                      "bundle_B06": [1, 1, 1, 1, 0, 0, 0, 0],
                      "cb_Big-box_B06": [1, 1, 0, 0, 0, 0, 0, 0],
                      "cb_School & office_B06": [0, 0, 1, 1, 0, 0, 0, 0],
                      "x": [0.1, 0.5, 0.2, 0.9, 0.3, 0.7, 0.4, 0.8]})
    out = A.drop_redundant(X)
    assert out.shape[1] == 4
    assert {"const", "bundle_B06", "x"} <= set(out.columns)
    assert np.linalg.matrix_rank(out.to_numpy()) == 4
