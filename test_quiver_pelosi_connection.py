"""Read-only connectivity probe for the Pelosi disclosure source.

No broker order is submitted.  This verifies Quiver authentication, response shape,
and Nancy Pelosi filtering only.
"""

from political_signals.pelosi_tail import QuiverCongressClient


def main() -> int:
    client = QuiverCongressClient()
    rows = client.fetch_recent()
    print(f"Quiver Congress API: PASS | Pelosi rows in current live window={len(rows)}")
    for row in rows[:5]:
        print(
            f"{row.report_date} traded={row.transaction_date} {row.ticker} "
            f"{row.transaction} {row.amount_range} lag={row.disclosure_lag_days}d"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
