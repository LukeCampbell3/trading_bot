# Pelosi Disclosure Tail

`feature/pelosi-tail` tracks Nancy Pelosi transactions as soon as they appear in Quiver's live congressional-trading feed and can automatically translate eligible public disclosures into Alpaca stock orders.

## Timing rule

The service never treats the disclosed `TransactionDate` as an observable trading signal. House Periodic Transaction Reports can be filed days or weeks after the underlying transaction. The bot records transaction date, report date, and first-seen timestamp separately. Backtests and execution availability must key to disclosure/first-seen time, never the original transaction date.

## Automated execution

`political_signals/pelosi_execution.py` separates broker writes from the disclosure parser/policy. Modes are `shadow`, `paper`, and `live`.

`paper` submits actual Alpaca paper-account stock orders. `live` submits real-money Alpaca orders only when all three gates agree:

```text
PELOSI_EXECUTION_MODE=live
PELOSI_ALLOW_LIVE=true
ALPACA_PAPER=false
```

Bullish eligible disclosures submit regular-session notional market BUY orders. Bearish `EXIT_ONLY` decisions never open naked shorts; they sell only quantity recorded as owned by this Pelosi strategy. Orders and fills are reconciled by Alpaca order id, persisted across restarts, and deduplicated by disclosure fingerprint.

Signals detected outside regular market hours are queued for the next regular session and expire after 18 hours by default. `--replay-existing` should remain off in automated execution because it intentionally allows the current source window to be reprocessed.

Default independent execution caps are 8% equity per order, 10% per Pelosi-tail symbol, 20% aggregate Pelosi-tail exposure, and $5 minimum notional. Actual allocation is the minimum of the policy's suggested notional, those caps, and available buying power.

## Run

```bash
# signal only
python run_pelosi_tail.py --execution-mode shadow

# automatically place Alpaca paper orders
PELOSI_EXECUTION_MODE=paper ALPACA_PAPER=true python run_pelosi_tail.py

# automatically place real-money Alpaca orders
PELOSI_EXECUTION_MODE=live PELOSI_ALLOW_LIVE=true ALPACA_PAPER=false python run_pelosi_tail.py
```

## Persistent state

The service writes disclosure state, signal history, execution state, execution audit records, and execution results under `HFT/logs/pelosi_tail/`. The execution ledger tracks only strategy-owned shares so a Pelosi sale cannot accidentally liquidate unrelated holdings in the same ticker.

## Validation status

CI passes 17 focused tests covering disclosure policy plus automatic execution: paper-order construction, fill ownership, restart idempotency, market-closed queueing, exposure caps, strategy-only exits, shadow isolation, and the explicit live-mode lock.

The available ChatGPT Alpaca integration exposes market data but not brokerage order submission, and repository CI does not currently have Quiver/Alpaca credentials. Therefore no real paper/live order was submitted during this implementation. Before live activation, run the exact branch in `paper` mode against the intended Alpaca account and verify that `execution_state.json` matches the paper account's orders/fills.

## Activation sequence

1. Configure `QUIVER_API_KEY` and Alpaca paper credentials.
2. Run `python run_pelosi_tail.py --execution-mode paper` continuously and verify broker reconciliation.
3. Keep `--replay-existing` off.
4. After the paper execution ledger is verified, move to live credentials and deliberately enable all three live gates.
5. Host the runner as a persistent daemon/process; GitHub Actions is validation CI, not the trading runtime.
