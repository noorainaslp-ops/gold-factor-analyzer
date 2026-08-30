"""
V6.2 Historical Telegram Alert Auditor
======================================

Usage:

    python audit_alerts_v6_2.py alerts.txt

The script extracts stock recommendations from historical Telegram
messages and evaluates their subsequent 1-, 3- and 5-session returns.

It is intentionally separate from the live engine.

This allows historical alerts to be audited without affecting the
live prediction model.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf


# ------------------------------------------------------------
# PARSE ALERTS
# ------------------------------------------------------------

def parse_alerts(
    path="alerts.txt",
):

    text = Path(
        path
    ).read_text(
        encoding="utf-8"
    )

    date_pattern = re.compile(
        r"(\d{1,2}\s+[A-Za-z]{3}\s+2026)"
    )

    stock_pattern = re.compile(
        r"^\s*\d+\.\s*"
        r"([A-Z0-9&._-]+)"
        r"\s*\|\s*"
        r"Rs\.?\s*"
        r"([0-9,.]+)",
        re.MULTILINE,
    )

    current_date = None

    rows = []

    for line in text.splitlines():

        date_match = (
            date_pattern.search(line)
        )

        if date_match:

            current_date = (
                pd.to_datetime(
                    date_match.group(1),
                    dayfirst=True,
                ).date()
            )

        stock_match = (
            stock_pattern.search(line)
        )

        if (
            stock_match
            and current_date
        ):

            ticker = (
                stock_match
                .group(1)
                .replace(
                    "&",
                    "-",
                )
                + ".NS"
            )

            entry = float(
                stock_match
                .group(2)
                .replace(
                    ",",
                    "",
                )
            )

            # Ignore obviously broken prices such as
            # the historical RELIANCE ₹0.0 alert.
            if entry <= 0:
                continue

            rows.append(
                {
                    "date":
                        current_date,

                    "ticker":
                        ticker,

                    "entry":
                        entry,
                }
            )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates(
            [
                "date",
                "ticker",
            ]
        )
    )


# ------------------------------------------------------------
# DOWNLOAD AND AUDIT
# ------------------------------------------------------------

def audit_alerts(
    alerts,
):

    results = []

    for _, alert in alerts.iterrows():

        ticker = alert["ticker"]

        alert_date = (
            pd.Timestamp(
                alert["date"]
            )
        )

        entry = float(
            alert["entry"]
        )

        try:

            history = yf.download(
                ticker,
                start=str(
                    alert_date.date()
                ),
                end=str(
                    (
                        alert_date
                        + pd.Timedelta(
                            days=14
                        )
                    ).date()
                ),
                auto_adjust=False,
                progress=False,
            )

            if history.empty:
                continue

            close = history[
                "Close"
            ]

            if isinstance(
                close,
                pd.DataFrame,
            ):

                close = (
                    close.iloc[:, 0]
                )

            close = (
                close
                .dropna()
                .tolist()
            )

            if len(close) < 2:
                continue

            next_return = (
                close[
                    min(
                        1,
                        len(close) - 1,
                    )
                ]
                / entry
                - 1
            ) * 100

            return_3d = (
                close[
                    min(
                        3,
                        len(close) - 1,
                    )
                ]
                / entry
                - 1
            ) * 100

            return_5d = (
                close[
                    min(
                        5,
                        len(close) - 1,
                    )
                ]
                / entry
                - 1
            ) * 100

            results.append(
                {
                    "date":
                        alert["date"],

                    "ticker":
                        ticker,

                    "entry":
                        entry,

                    "next_session_pct":
                        next_return,

                    "3_session_pct":
                        return_3d,

                    "5_session_pct":
                        return_5d,
                }
            )

        except Exception as exc:

            print(
                "Skipping",
                ticker,
                alert["date"],
                exc,
            )

    return pd.DataFrame(
        results
    )


# ------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------

def print_statistics(
    result,
):

    if result.empty:

        print(
            "No historical alerts could be evaluated."
        )

        return

    print()
    print(
        "=" * 65
    )
    print(
        "V6.2 HISTORICAL ALERT AUDIT"
    )
    print(
        "=" * 65
    )

    print(
        f"Evaluated recommendations: "
        f"{len(result)}"
    )

    horizons = [
        (
            "next_session_pct",
            "1-session",
        ),
        (
            "3_session_pct",
            "3-session",
        ),
        (
            "5_session_pct",
            "5-session",
        ),
    ]

    for column, label in horizons:

        values = result[
            column
        ].dropna()

        if values.empty:
            continue

        wins = (
            values > 0
        ).mean() * 100

        print()
        print(
            f"{label}"
        )

        print(
            f"  Win rate: "
            f"{wins:.1f}%"
        )

        print(
            f"  Average return: "
            f"{values.mean():.2f}%"
        )

        print(
            f"  Median return: "
            f"{values.median():.2f}%"
        )

        print(
            f"  Best return: "
            f"{values.max():.2f}%"
        )

        print(
            f"  Worst return: "
            f"{values.min():.2f}%"
        )

        positive = (
            values[values > 0]
        )

        negative = (
            values[values < 0]
        )

        if not negative.empty:

            profit_factor = (
                positive.sum()
                / abs(
                    negative.sum()
                )
            )

            print(
                f"  Profit factor: "
                f"{profit_factor:.2f}"
            )

    print()
    print(
        "=" * 65
    )

    # --------------------------------------------------------
    # STOCK-BY-STOCK
    # --------------------------------------------------------

    print()
    print(
        "RESULTS BY STOCK — 3 SESSION"
    )

    grouped = (
        result
        .groupby("ticker")
        .agg(
            alerts=(
                "ticker",
                "size",
            ),
            avg_3d=(
                "3_session_pct",
                "mean",
            ),
            win_rate=(
                "3_session_pct",
                lambda x:
                    (x > 0).mean()
                    * 100,
            ),
        )
        .sort_values(
            "avg_3d",
            ascending=False,
        )
    )

    print(
        grouped.to_string(
            float_format=lambda x:
                f"{x:.2f}"
        )
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "alerts.txt"
    )

    alerts = parse_alerts(
        path
    )

    if alerts.empty:

        raise SystemExit(
            "No stock recommendations found."
        )

    print(
        f"Found {len(alerts)} "
        f"historical stock recommendations."
    )

    result = audit_alerts(
        alerts
    )

    if result.empty:

        raise SystemExit(
            "No recommendations could be evaluated."
        )

    output_file = (
        "alert_audit_v6_2.csv"
    )

    result.to_csv(
        output_file,
        index=False,
    )

    print_statistics(
        result
    )

    print()
    print(
        f"Detailed audit saved to "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()
