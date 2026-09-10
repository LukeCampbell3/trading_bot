"""Causality/fill tests for V14.5.1 option shadow labels."""

from datetime import datetime, timedelta, timezone

from replay.v14_5_1_shadow_labeler import (
    OptionSpreadShadowLabeler,
    ShadowCandidate,
    VerticalQuoteSnapshot,
)
from strategy.empirical_option_edge import EmpiricalOptionEdgeMemory


def _q(ts, lb, la, sb, sa):
    return VerticalQuoteSnapshot(
        timestamp=ts, symbol="COIN", long_contract="L", short_contract="S",
        long_bid=lb, long_ask=la, short_bid=sb, short_ask=sa,
    )


def _candidate(ts, watch_mid=1.00, score=.70):
    return ShadowCandidate(
        timestamp=ts, symbol="COIN", route="VWAP_PULLBACK", side="CALL",
        route_score=score, long_contract="L", short_contract="S",
        watch_spread_mid=watch_mid, regime="NORMAL_IV",
    )


def test_entry_quote_must_be_strictly_after_candidate(tmp_path):
    t0 = datetime(2026, 9, 10, 14, 0, tzinfo=timezone.utc)
    labeler = OptionSpreadShadowLabeler(
        max_limit_chase_pct=.10, horizons_minutes=(30,), output_dir=str(tmp_path)
    )
    quotes = [
        _q(t0, 2.0, 2.05, 1.05, 1.10),  # same timestamp: forbidden
        _q(t0 + timedelta(seconds=1), 2.0, 2.04, 1.04, 1.09),
        _q(t0 + timedelta(minutes=30, seconds=1), 2.2, 2.25, 1.05, 1.10),
    ]
    labels = labeler.label_candidate(_candidate(t0), quotes)
    assert len(labels) == 1
    assert labels[0].entry_timestamp == t0 + timedelta(seconds=1)
    assert labels[0].entry_timestamp > t0
    assert labels[0].label_valid is True


def test_unmarketable_limit_is_counted_as_missed_fill(tmp_path):
    t0 = datetime(2026, 9, 10, 14, 0, tzinfo=timezone.utc)
    labeler = OptionSpreadShadowLabeler(
        max_limit_chase_pct=.03, horizons_minutes=(30,), output_dir=str(tmp_path)
    )
    # Natural debit = 2.00 - .50 = 1.50, much greater than 1.03 limit.
    labels = labeler.label_candidate(
        _candidate(t0, watch_mid=1.00),
        [_q(t0 + timedelta(seconds=1), 1.9, 2.0, .5, .6)],
    )
    assert labels[0].filled is False
    assert labels[0].label_valid is False
    assert labels[0].missed_fill_reason == "natural_ask_above_limit"


def test_spread_return_uses_natural_debit_and_natural_credit(tmp_path):
    t0 = datetime(2026, 9, 10, 14, 0, tzinfo=timezone.utc)
    labeler = OptionSpreadShadowLabeler(
        max_limit_chase_pct=.20, horizons_minutes=(30,), output_dir=str(tmp_path)
    )
    # Entry natural debit 2.00 - 1.00 = 1.00.
    # Exit natural credit 2.40 - 1.10 = 1.30 => +30%.
    labels = labeler.label_candidate(
        _candidate(t0, watch_mid=1.00),
        [
            _q(t0 + timedelta(seconds=1), 1.95, 2.00, 1.00, 1.05),
            _q(t0 + timedelta(minutes=30, seconds=1), 2.40, 2.45, 1.05, 1.10),
        ],
    )
    x = labels[0]
    assert x.entry_debit == 1.0
    assert abs(x.exit_credit - 1.30) < 1e-12
    assert abs(x.spread_return - .30) < 1e-12


def test_primary_shadow_labels_feed_independent_edge_memory(tmp_path):
    t0 = datetime(2026, 9, 10, 14, 0, tzinfo=timezone.utc)
    labeler = OptionSpreadShadowLabeler(
        max_limit_chase_pct=.20, horizons_minutes=(60,), primary_horizon_minutes=60,
        output_dir=str(tmp_path),
    )
    mem = EmpiricalOptionEdgeMemory(
        min_samples=3, prior_strength=0, log_path=str(tmp_path / "edge.csv")
    )
    # Three observations are the minimum needed for a useful correlation test.
    # Route score and executable spread return are intentionally separate inputs.
    for i, (score, exit_credit) in enumerate(((.60, .90), (.70, 1.10), (.80, 1.30))):
        ts = t0 + timedelta(hours=i * 2)
        labels = labeler.label_candidate(
            _candidate(ts, watch_mid=1.0, score=score),
            [
                _q(ts + timedelta(seconds=1), 1.95, 2.00, 1.00, 1.05),
                _q(ts + timedelta(minutes=60, seconds=1), exit_credit + 1.1, exit_credit + 1.15, 1.0, 1.1),
            ],
        )
        assert labeler.feed_primary_labels(labels, mem) == 1
    stats = mem.stats("COIN", "VWAP_PULLBACK", "NORMAL_IV")
    assert stats.ready is True
    assert stats.ic_spread > 0
    assert stats.ev_over_debit > 0
