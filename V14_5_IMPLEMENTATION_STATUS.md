# V14.5 Implementation Status

Policy: `V14_5_STABLE_OPTIONS_TRADER`

Current status: **paper/replay validation candidate; not live-proven**.

Implemented:
- V14.4 zero-wait/session-aware feature initialization retained.
- V14.3 CALL/PUT debit-spread route metrics retained.
- Watch-only base signal and confirmation retained.
- Real spread quote quality gate retained.
- Limit-only Alpaca MLEG execution retained.
- Core/runner package retained.
- Stateful CALL/PUT directional hysteresis added.
- Opposite-side entry blocked while filled/confirmed/pending risk exists.
- Real reversals require stronger route score, score edge, VWAP/ATR separation, and persistence.
- Stop/loss exits trigger longer directional cooldown.
- One live option strategy per symbol enforced.
- Continuation-decay exits now use ATR-buffered VWAP hysteresis and consecutive-failure confirmation.
- Alpaca option snapshots used to enrich quotes with IV and Greeks when available.
- Option-native delta, delta-separation, theta burden, IV-skew, and quality checks added.
- Structural reversal validator added.
- Regression tests added.

Not claimed:
- Live profitability.
- Live fill-adjusted win rate.
- Historical option replay improvement versus V14.3 until credential-backed validation is run.
- Guaranteed prevention of all reversals. The policy suppresses transient/noisy reversals while preserving confirmed regime changes.
