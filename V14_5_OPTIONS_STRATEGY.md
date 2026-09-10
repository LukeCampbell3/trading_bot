# V14.5 Stable Options Strategy

Status: **PAPER / REPLAY VALIDATION CANDIDATE**. Live trading remains disabled.

V14.5 keeps V14.3's high-volatility debit-spread economics and V14.4's zero-wait warm start, but treats option direction as a stateful position decision rather than a one-bar classification.

## Pipeline

1. V14.4 reconstructs session/cross-session features without a long live warm-up.
2. Existing V14 route scores generate CALL/PUT candidates.
3. Directional hysteresis establishes a CALL or PUT bias after short consensus.
4. Once a bias is established, an opposite route must be materially stronger, on the correct side of VWAP by an ATR buffer, and persist for multiple observations before the bias may reverse.
5. Filled, confirmed, or pending risk blocks an opposite-side entry. V14.5 does not accidentally hedge or flip an open spread.
6. Real option quotes/snapshots are fetched for the chosen contracts. When available, IV and Greeks enrich the spread-quality decision.
7. Existing spread-quality checks remain mandatory: bid/ask validity, composite width, mid inflation, chase amount, move consumed, remaining reward/risk, and debit limit.
8. V14.5 additionally checks long-leg delta, long-vs-short delta separation, net theta burden, IV skew, and an option-native quality score when Greeks are available.
9. Execution remains limit-only Alpaca MLEG through the single execution manager.
10. Only one option strategy per symbol may carry risk at a time.
11. Hard spread stops and profit targets are immediate. VWAP/continuation-decay exits require a buffered, persistent structural failure so a one-bar cross cannot churn the position.
12. After a stop/loss, reversal permission has a longer cooldown than after a target/normal close.

## Direction defaults

- Initial bias: 2 qualifying observations inside a 4-observation window.
- Initial CALL/PUT score edge when both exist: 0.06.
- Reversal route score: >= 0.68.
- Reversal edge over current side: >= 0.12.
- Reversal VWAP separation: >= 0.10 ATR in the new direction.
- Reversal persistence: 3 consecutive observations.
- Post-reversal cooldown: 4 observations.
- Stop/loss cooldown: 7 observations.
- Continuation-decay exit persistence: 2 observations beyond a 0.05 ATR VWAP buffer.

The asymmetric thresholds are intentional: entering a direction and reversing an established direction are not the same decision.

## Option-native defaults

- Long-leg |delta| target: 0.55.
- Accepted long-leg |delta| band: 0.40-0.72.
- Minimum long-short delta separation: 0.07.
- Maximum net theta burden / spread mid: 0.12.
- Maximum IV difference between legs: 0.15.
- Minimum composite option-native quality score: 0.55.

These checks augment V14.3. They do not replace quote-width, fill-quality, reward/risk, or limit-price controls.

## Validation meaning

The deterministic reversal validator measures structural whipsaw suppression versus a naive raw best-side selector. It is not a profitability backtest. Profitability, win rate, fill quality, and missed fills must be evaluated separately using timestamp-safe historical option data and paper fills.
