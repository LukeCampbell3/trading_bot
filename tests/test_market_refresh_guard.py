"""Regression tests for the live market-data freshness guard."""

import unittest

from execution_controller import FreshMarketSignalEngine


class _FakeBuffer:
    def __init__(self, ts):
        self.last_timestamp = ts


class _FakeState:
    def __init__(self, ts):
        self.buf = _FakeBuffer(ts)


class _FakeEngine:
    def __init__(self):
        self.sym_states = {
            "AAPL": _FakeState(100),
            "NVDA": _FakeState(100),
        }
        self.poll_calls = 0
        self.generate_calls = 0
        self.advance_aapl = True

    def poll_latest(self):
        self.poll_calls += 1
        if self.advance_aapl:
            self.sym_states["AAPL"].buf.last_timestamp += 1
            self.advance_aapl = False

    def generate_signals(self):
        self.generate_calls += 1
        return list(self.sym_states.keys())

    def get_market_state(self, symbol):
        return {"symbol": symbol, "ts": self.sym_states[symbol].buf.last_timestamp}

    def estimate_costs(self, symbol):
        return {"symbol": symbol}


class _FailingEngine(_FakeEngine):
    def poll_latest(self):
        self.poll_calls += 1
        raise RuntimeError("market data unavailable")


class FreshMarketSignalEngineTests(unittest.TestCase):
    def test_only_symbols_with_advanced_bars_emit(self):
        engine = _FakeEngine()
        guard = FreshMarketSignalEngine(engine, refresh_interval_seconds=0)

        signals = guard.generate_signals()

        self.assertEqual(signals, ["AAPL"])
        self.assertEqual(engine.poll_calls, 1)
        self.assertEqual(engine.generate_calls, 1)
        self.assertEqual(set(engine.sym_states), {"AAPL", "NVDA"})

    def test_same_bar_is_not_reprocessed(self):
        engine = _FakeEngine()
        guard = FreshMarketSignalEngine(engine, refresh_interval_seconds=0)

        self.assertEqual(guard.generate_signals(), ["AAPL"])
        self.assertEqual(guard.generate_signals(), [])
        self.assertEqual(engine.poll_calls, 2)
        self.assertEqual(engine.generate_calls, 1)

    def test_refresh_failure_fails_closed(self):
        engine = _FailingEngine()
        guard = FreshMarketSignalEngine(engine, refresh_interval_seconds=0)

        with self.assertRaises(RuntimeError):
            guard.generate_signals()

        self.assertEqual(engine.generate_calls, 0)

    def test_market_state_refreshes_before_read(self):
        engine = _FakeEngine()
        guard = FreshMarketSignalEngine(engine, refresh_interval_seconds=0)

        state = guard.get_market_state("AAPL")

        self.assertEqual(state["ts"], 101)
        self.assertEqual(engine.poll_calls, 1)


if __name__ == "__main__":
    unittest.main()
