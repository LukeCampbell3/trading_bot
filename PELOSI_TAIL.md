# Pelosi Disclosure Tail

`feature/pelosi-tail` tracks Nancy Pelosi transactions as soon as they appear in Quiver's live congressional-trading feed and can automatically translate eligible public disclosures into Alpaca stock orders.

## Timing rule

The service never treats the disclosed `TransactionDate` as an observable trading signal. House Periodic Transaction Reports can be filed days or weeks after the underlying transaction. The bot records transaction date, report date, and first-seen timestamp separately. Backtests and execution availability must key to disclosure/first-seen time, never the original transaction date.

## Source

Primary ingestion source:

`https://api.quiverquant.com/beta/live/congresstrading`

Authentication is `Authorization: Bearer $QUIVER_API_KEY`. Quiver is the fast ingestion source; House Clerk PTR filings remain the authoritative disclosure source for verification.

## Signal policy

The poller defaults to 60 seconds and baseline-seeds the current Quiver window on first launch so historical disclosures are not mistaken for new signals. Bullish stock purchases and clearly identified purchased calls can create `BUY` decisions. Sales and clearly bearish option disclosures create `EXIT_ONLY` decisions. Ambiguous options remain `WATCH`.

The tail score discounts disclosure lag, smaller reported ranges, and less-certain instruments. Alpaca market data can also estimate the move since the disclosed transaction date; large run-ups are not chased and large adverse moves are not averaged down.

## Automated execution

`political_signals/pelosi_execution.py` is the broker-write layer. It is independent from the parser and signal policy so detection, decision-making, and execution are auditable separately.

Execution modes:

- `shadow`: no broker writes.
- `paper`: eligible decisions automatically submit actual orders to the Alpaca paper account.
- `live`: eligible decisions automatically submit real-money orders to the Alpaca live brokerage account.

Live execution has a deliberate multi-gate interlock. All three settings must agree:

```text
PELOSI_EXECUTION_MODE=live
PELOSI_ALLOW_LIVE=true
ALPACA_PAPER=false
```

The executor will refuse to start live if the second explicit gate is absent or the account configuration still points to paper.

### Order behavior

Bullish `BUY` decisions submit regular-hours Alpaca market orders using notional dollars, which permits fractional stock sizing for small accounts. Bearish `EXIT_ONLY` decisions never create a naked short: they sell only the quantity recorded in the strategy-owned Pelosi ledger.

The strategy is restart-safe and idempotent. A disclosure fingerprint can submit at most one entry action even after a process restart. Broker fills are reconciled by Alpaca order id and strategy-owned filled quantity is persisted separately from any unrelated shares in the brokerage account.

If a new disclosure is detected outside regular market hours, it is queued instead of using extended-hours trading. The default queue lifetime is 18 hours; stale pending signals expire rather than being executed indefinitely later.

### Execution risk caps

Defaults are intentionally bounded independently of the signal score:

```text
single disclosure/order cap: 8% equity
single Pelosi-tail symbol cap: 10% equity
aggregate Pelosi-tail cap: 20% equity
minimum order: $5
```

Actual buy notional is the minimum of the signal's suggested allocation, these portfolio caps, and available buying power.

## Run

Signal-only mode:

```bash
python run_pelosi_tail.py --execution-mode shadow
```

Automatic paper orders:

```bash
PELOSI_EXECUTION_MODE=paper ALPACA_PAPER=true python run_pelosi_tail.py
```

Automatic live orders after deliberate live-account configuration:

```bash
PELOSI_EXECUTION_MODE=live PELOSI_ALLOW_LIVE=true ALPACA_PAPER=false python run_pelosi_tail.py
```

One source/execution-cycle check:

```bash
python run_pelosi_tail.py --once --execution-mode shadow
```

The first run baseline-seeds existing disclosures. `--replay-existing` exists for testing/research; do not use it casually with paper/live execution because it intentionally treats the current returned window as processable.

## Persistent evidence

The service writes under `HFT/logs/pelosi_tail/`:

- `state.json` — disclosure deduplication state
- `disclosures.jsonl` — observed public filings
- `latest_signal.json` / `signals.jsonl` — policy decisions
- `execution_state.json` — broker order and strategy-owned position ledger
- `execution_audit.jsonl` — execution state changes
- `execution_results.jsonl` — runner-level broker decisions/results

## Environment

```text
QUIVER_API_KEY=...
QUIVER_CONGRESS_URL=https://api.quiverquant.com/beta/live/congresstrading
PELOSI_POLL_SECONDS=60
PELOSI_MAX_NOTIONAL_PCT=0.08

PELOSI_EXECUTION_MODE=shadow
PELOSI_ALLOW_LIVE=false
PELOSI_MAX_ORDER_PCT=0.08
PELOSI_MAX_SYMBOL_PCT=0.10
PELOSI_MAX_TOTAL_PCT=0.20
PELOSI_MIN_ORDER_NOTIONAL=5
PELOSI_MAX_PENDING_HOURS=18
```

## Validation status

The automated execution layer has passed the credential-free broker simulation suite: 17 focused tests total across the Pelosi source/policy and execution modules. These verify automatic paper-order construction, fill ownership, restart idempotency, market-closed queueing, risk caps, strategy-only exits, shadow isolation, and the explicit live-mode lock.

The available ChatGPT Alpaca connection exposes market data but not brokerage order submission, and the repository currently has no Quiver/Alpaca secrets available to CI. Therefore no real Alpaca paper or live order was submitted during this implementation. Before switching `PELOSI_EXECUTION_MODE` to live, run the paper mode against the intended Alpaca account and verify the resulting order/fill ledger.

Execution correctness does not imply strategy profitability. Disclosure-time residual-return validation should continue in parallel.
