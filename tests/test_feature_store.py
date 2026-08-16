from finmas.data.feature_store import FeatureStore


def test_feature_store_market_snapshot():
    store = FeatureStore()
    features = store.market_snapshot("801780", "2024-07-22")
    assert features["n_hist"] > 20
    assert "momentum_20" in features


def test_feature_store_is_strictly_pre_event():
    store = FeatureStore()
    before = store.market_snapshot("801780", "2024-07-22")
    after = store.market_snapshot("801780", "2024-08-01")
    # The later event has more history but the early event must not see post-date data.
    assert before["n_hist"] <= after["n_hist"]


def test_valuation_snapshot_history_is_pre_event():
    store = FeatureStore()
    early = store.valuation_snapshot("801780", "2021-01-10")
    late = store.valuation_snapshot("801780", "2024-07-22")
    if early and late:
        assert early["n_hist"] < late["n_hist"]


def test_sentiment_snapshot_history_is_pre_event():
    store = FeatureStore()
    early = store.sentiment_snapshot("2021-01-10")
    late = store.sentiment_snapshot("2024-07-22")
    if early and late:
        assert early["n_hist"] < late["n_hist"]
