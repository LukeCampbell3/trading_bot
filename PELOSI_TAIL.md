# Pelosi Disclosure Tail

`feature/pelosi-tail` adds a disclosure-driven stock signal service that tracks Nancy Pelosi transactions as soon as they appear in Quiver's live congressional trading feed.

## Important timing rule

The service never treats the disclosed `TransactionDate` as an observable trading signal. House Periodic Transaction Reports can be filed days or weeks after the underlying transaction. The bot records:

- transaction date — when the disclosed trade occurred;
- report date — filing/public-disclosure date supplied by the data source;
- first-seen timestamp — when this process first observed the disclosure.

Backtests and production monitoring must key signal availability to disclosure/first-seen time, not transaction time.

## Source

Primary source:

`https://api.quiverquant.com/beta/live/congresstrading`

Authentication is `Authorization: Bearer $QUIVER_API_KEY`.

Quiver is an ingestion source, not the legal origin of the disclosure. House Clerk Periodic Transaction Reports remain the authoritative source for later verification.

## Behavior

The service polls every 60 seconds by default. A first run baseline-seeds the current live API window and does not emit trades from old disclosures. After that, only unseen Pelosi transactions generate decisions.

Bullish stock purchases and clearly identified purchased calls can create `BUY` signals. Sales and clearly bearish option disclosures create `EXIT_ONLY` signals; this feature never opens a naked short. Ambiguous option disclosures are `WATCH` only.

The tail score discounts old disclosures, smaller reported ranges, and option ambiguity. If Alpaca credentials are available, the runner also estimates how far the underlying has already moved since the disclosed transaction date. Large run-ups are not chased and large adverse moves are treated as a changed thesis rather than an averaging-down opportunity.

Default maximum suggested notional is 8% of account equity, multiplied by the tail score. This is a signal budget, not an automatic live order.

## Run

```bash
python run_pelosi_tail.py
```

One source check:

```bash
python run_pelosi_tail.py --once
```

Replay the currently returned Quiver window instead of baseline-seeding it:

```bash
python run_pelosi_tail.py --once --replay-existing
```

The runner is deliberately `SHADOW_ONLY`. It writes:

- `HFT/logs/pelosi_tail/state.json`
- `HFT/logs/pelosi_tail/disclosures.jsonl`
- `HFT/logs/pelosi_tail/latest_signal.json`
- `HFT/logs/pelosi_tail/signals.jsonl`

## Environment

```text
QUIVER_API_KEY=...
QUIVER_CONGRESS_URL=https://api.quiverquant.com/beta/live/congresstrading
PELOSI_POLL_SECONDS=60
PELOSI_MAX_NOTIONAL_PCT=0.08
```

Alpaca credentials are optional for the disclosure tracker itself and are used only to estimate residual price opportunity in the current feature branch.

## Promotion path

Do not wire this directly to live execution until disclosure-time backtests have measured win rate, profit factor, drawdown, signal lag, and residual-return behavior. The intended next integration is as an external-prior input to the existing multi-symbol stock trader, where Pelosi signals can promote/rank an existing technical opportunity rather than bypass portfolio risk controls.
