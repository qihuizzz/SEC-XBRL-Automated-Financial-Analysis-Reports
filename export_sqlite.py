from __future__ import annotations
import argparse
import os
import sqlite3
from datetime import datetime, timezone
import pandas as pd
from sec_client import make_default_client
from xbrl_normalize import normalize_pipeline
from financials import build_annual_financials_table, format_financials_for_display


DB_DEFAULT = os.path.join("output_database", "sec_fin.db")


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS annual_long (
            ticker TEXT NOT NULL,
            fy INTEGER NOT NULL,
            fiscal_year_end TEXT,
            metric TEXT NOT NULL,
            value REAL,
            unit TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (ticker, fy, metric)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS company_meta (
            ticker TEXT PRIMARY KEY,
            cik TEXT,
            company_name TEXT,
            latest_fy INTEGER,
            latest_fy_end TEXT,
            concept_map_json TEXT,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_log (
            run_id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            status TEXT NOT NULL,
            message TEXT
        );
        """
    )
    conn.commit()


def to_long_table(display_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    d = display_df.copy()

    d["fy"] = pd.to_numeric(d.get("fy"), errors="coerce")
    d = d.dropna(subset=["fy"]).copy()
    d["fy"] = d["fy"].astype(int)

    id_cols = [c for c in ["fy", "fiscal_year_end"] if c in d.columns]
    metric_cols = [c for c in d.columns if c not in id_cols]

    long = d.melt(id_vars=id_cols, value_vars=metric_cols, var_name="metric", value_name="value")
    long.insert(0, "ticker", ticker.upper())

    def infer_unit(metric: str) -> str:
        if metric.endswith("_margin") or metric.endswith("_yoy") or metric in ["roe"]:
            return "ratio"
        return "usd_billions"

    long["unit"] = long["metric"].map(infer_unit)

    now = datetime.now(timezone.utc).isoformat()
    long["updated_at"] = now

    if "fiscal_year_end" in long.columns:
        long["fiscal_year_end"] = long["fiscal_year_end"].astype(str)

    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long


def upsert_annual_long(conn: sqlite3.Connection, long_df: pd.DataFrame) -> None:
    rows = long_df[
        ["ticker", "fy", "fiscal_year_end", "metric", "value", "unit", "updated_at"]
    ].to_records(index=False)

    conn.executemany(
        """
        INSERT INTO annual_long (ticker, fy, fiscal_year_end, metric, value, unit, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, fy, metric) DO UPDATE SET
            fiscal_year_end=excluded.fiscal_year_end,
            value=excluded.value,
            unit=excluded.unit,
            updated_at=excluded.updated_at;
        """,
        list(rows),
    )
    conn.commit()


def upsert_company_meta(
    conn: sqlite3.Connection,
    ticker: str,
    cik: str,
    company_name: str,
    latest_fy: int | None,
    latest_fy_end: str | None,
    concept_map_json: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO company_meta (ticker, cik, company_name, latest_fy, latest_fy_end, concept_map_json, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            cik=excluded.cik,
            company_name=excluded.company_name,
            latest_fy=excluded.latest_fy,
            latest_fy_end=excluded.latest_fy_end,
            concept_map_json=excluded.concept_map_json,
            updated_at=excluded.updated_at;
        """,
        (ticker.upper(), cik, company_name, latest_fy, latest_fy_end, concept_map_json, now),
    )
    conn.commit()


def export_one_ticker(conn: sqlite3.Connection, ticker: str, years: int) -> None:
    ticker = ticker.strip().upper()
    if not ticker:
        return

    run_id = f"{ticker}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    started = datetime.now(timezone.utc).isoformat()

    conn.execute(
        "INSERT INTO run_log (run_id, ticker, started_at, finished_at, status, message) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, ticker, started, started, "RUNNING", None),
    )
    conn.commit()

    try:
        c = make_default_client()
        cik = c.ticker_to_cik(ticker)
        subs = c.get_submissions(cik)
        company_name = subs.get("name") or subs.get("companyName") or ticker

        facts = c.get_companyfacts(cik)
        df = normalize_pipeline(facts)

        annual = build_annual_financials_table(df)
        display = format_financials_for_display(annual)

        # keep last N fiscal years
        display = display.copy()
        display["fy"] = pd.to_numeric(display.get("fy"), errors="coerce")
        display = display.dropna(subset=["fy"]).copy()
        display["fy"] = display["fy"].astype(int)
        display = display.sort_values("fy", ascending=False).head(years)
        display = display.sort_values("fy", ascending=True).reset_index(drop=True)

        long_df = to_long_table(display, ticker=ticker)
        upsert_annual_long(conn, long_df)

        latest_fy = int(display["fy"].max()) if len(display) else None
        latest_fy_end = None
        if "fiscal_year_end" in display.columns and len(display):
            latest_fy_end = str(display.loc[display["fy"].idxmax(), "fiscal_year_end"])

        concept_map_json = str(annual.attrs.get("concept_map", {}))
        upsert_company_meta(
            conn,
            ticker=ticker,
            cik=str(cik),
            company_name=str(company_name),
            latest_fy=latest_fy,
            latest_fy_end=latest_fy_end,
            concept_map_json=concept_map_json,
        )

        finished = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE run_log SET finished_at=?, status=?, message=? WHERE run_id=?",
            (finished, "OK", f"exported last {years} FY", run_id),
        )
        conn.commit()

        print(f"OK: {ticker}")
    except Exception as e:
        finished = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE run_log SET finished_at=?, status=?, message=? WHERE run_id=?",
            (finished, "ERROR", str(e), run_id),
        )
        conn.commit()
        print(f"ERROR: {ticker} -> {e}")


def parse_tickers(args) -> list[str]:
    if args.tickers:
        # allow: --tickers AAPL,MSFT,NVDA or --tickers AAPL MSFT NVDA
        raw = []
        for item in args.tickers:
            raw.extend([x.strip() for x in item.split(",") if x.strip()])
        return [t.upper() for t in raw if t]
    if args.ticker:
        return [args.ticker.strip().upper()]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SEC annual financials to SQLite for Tableau.")
    parser.add_argument("--ticker", help="Single ticker symbol, e.g., AAPL")
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Multiple tickers: --tickers AAPL,MSFT,NVDA  OR  --tickers AAPL MSFT NVDA",
    )
    parser.add_argument("--years", type=int, default=5, help="How many fiscal years to keep (default 5)")
    parser.add_argument("--db", default=DB_DEFAULT, help=f"SQLite db path (default {DB_DEFAULT})")
    args = parser.parse_args()

    tickers = parse_tickers(args)
    if not tickers:
        raise SystemExit("Please provide --ticker or --tickers.")

    _ensure_parent(args.db)
    conn = sqlite3.connect(args.db)
    try:
        init_db(conn)
        for t in tickers:
            export_one_ticker(conn, t, args.years)
    finally:
        conn.close()

    print(f"Done. SQLite saved at: {args.db}")


if __name__ == "__main__":
    main()