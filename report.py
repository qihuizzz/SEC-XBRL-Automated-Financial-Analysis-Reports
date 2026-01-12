# report.py
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from sec_client import make_default_client
from xbrl_normalize import normalize_pipeline
from financials import (
    FinancialsConfig,
    build_annual_financials_table,
    format_financials_for_display,
)


# -----------------------------
# Formatting helpers
# -----------------------------
def _pct(x: Optional[float], digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "NA"
    try:
        return f"{x * 100:.{digits}f}%"
    except Exception:
        return "NA"


def _num(x: Optional[float], digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "NA"
    try:
        return f"{x:.{digits}f}"
    except Exception:
        return "NA"


def _safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _as_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    return df.to_markdown(index=False)


# -----------------------------
# Insights
# -----------------------------
def _make_insights(display_df: pd.DataFrame, raw_df: pd.DataFrame) -> str:
    """
    display_df: scaled to billions for numeric lines
    raw_df: unscaled, used for comparisons if needed
    """
    if display_df.empty:
        return "- No data available."

    # Use the last row as "latest"
    latest = display_df.iloc[-1]
    prev = display_df.iloc[-2] if len(display_df) >= 2 else None

    fy = _safe_float(latest.get("fy"))
    end = latest.get("fiscal_year_end")
    end_str = str(end.date()) if isinstance(end, pd.Timestamp) else str(end)

    rev = _safe_float(latest.get("revenue"))
    rev_yoy = _safe_float(latest.get("revenue_yoy"))
    gm = _safe_float(latest.get("gross_margin"))
    om = _safe_float(latest.get("operating_margin"))
    nm = _safe_float(latest.get("net_margin"))
    fcf = _safe_float(latest.get("fcf"))
    fcfm = _safe_float(latest.get("fcf_margin"))

    lines = []
    if fy is not None:
        lines.append(f"- Latest fiscal year: **FY{int(fy)}** (ended {end_str}).")
    if rev is not None:
        if rev_yoy is not None:
            lines.append(f"- Revenue: **{_num(rev)}B** ({_pct(rev_yoy)} YoY).")
        else:
            lines.append(f"- Revenue: **{_num(rev)}B**.")
    if gm is not None:
        lines.append(f"- Gross margin: **{_pct(gm)}**.")
    if om is not None:
        lines.append(f"- Operating margin: **{_pct(om)}**.")
    if nm is not None:
        lines.append(f"- Net margin: **{_pct(nm)}**.")
    if fcf is not None:
        if fcfm is not None:
            lines.append(f"- Free cash flow: **{_num(fcf)}B** ({_pct(fcfm)} of revenue).")
        else:
            lines.append(f"- Free cash flow: **{_num(fcf)}B**.")

    # A simple trend comment if we have at least 2 years of revenue
    if prev is not None:
        prev_rev = _safe_float(prev.get("revenue"))
        if rev is not None and prev_rev is not None:
            if rev > prev_rev:
                lines.append("- Revenue increased vs prior year.")
            elif rev < prev_rev:
                lines.append("- Revenue decreased vs prior year.")
            else:
                lines.append("- Revenue was flat vs prior year.")

    if not lines:
        return "- No insights could be generated."
    return "\n".join(lines)


# -----------------------------
# Report builder
# -----------------------------
def build_report_markdown(
    ticker: str,
    years: int = 5,
    out_dir: str = "reports",
    include_concept_map: bool = True,
) -> Tuple[str, str]:
    """
    Returns (md_text, output_path)
    """
    ticker = ticker.strip().upper()
    out_path = Path(out_dir) / f"{ticker}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    c = make_default_client()
    cik = c.ticker_to_cik(ticker)
    submissions = c.get_submissions(cik)
    company_name = submissions.get("name", ticker)

    facts = c.get_companyfacts(cik)
    df = normalize_pipeline(facts)

    cfg = FinancialsConfig(last_n_years=years)
    raw_table = build_annual_financials_table(df, cfg=cfg)

    # For the markdown table we want readability
    display_table = format_financials_for_display(raw_table)

    # Keep a clean column order if present
    preferred_cols = [
        "fy",
        "fiscal_year_end",
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "cfo",
        "capex",
        "fcf",
        "revenue_yoy",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "fcf_margin",
        "cash",
        "equity",
    ]
    cols = [c for c in preferred_cols if c in display_table.columns] + [
        c for c in display_table.columns if c not in preferred_cols
    ]
    display_table = display_table[cols].copy()

    # Format display columns for markdown readability
    # Numbers are already scaled for USD lines; we still format percent columns.
    percent_cols = ["revenue_yoy", "gross_margin", "operating_margin", "net_margin", "fcf_margin"]
    for col in percent_cols:
        if col in display_table.columns:
            display_table[col] = display_table[col].apply(lambda x: _pct(_safe_float(x)) if x is not None else "NA")

    # Keep FY as int string
    if "fy" in display_table.columns:
        display_table["fy"] = display_table["fy"].apply(
            lambda x: f"{int(x)}" if _safe_float(x) is not None else "NA"
        )

    # Fiscal year end as YYYY-MM-DD
    if "fiscal_year_end" in display_table.columns:
        def _fmt_date(x):
            if isinstance(x, pd.Timestamp):
                return str(x.date())
            return "NA" if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x)

        display_table["fiscal_year_end"] = display_table["fiscal_year_end"].apply(_fmt_date)

    # Numeric columns (billions) formatting
    billions_cols = ["revenue", "gross_profit", "operating_income", "net_income", "cfo", "capex", "fcf", "cash", "equity"]
    for col in billions_cols:
        if col in display_table.columns:
            display_table[col] = display_table[col].apply(lambda x: _num(_safe_float(x), digits=1))

    md_lines = []
    md_lines.append(f"# {company_name} ({ticker}) Automated Financial Analysis Report")
    md_lines.append("")
    md_lines.append(f"- CIK: `{cik}`")
    md_lines.append(f"- Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"- Coverage: last {years} fiscal years")
    md_lines.append("")

    md_lines.append("## Highlights")
    md_lines.append(_make_insights(display_table, raw_table))
    md_lines.append("")

    md_lines.append("## Annual Financials Table (USD in billions)")
    md_lines.append(_as_markdown_table(display_table))
    md_lines.append("")

    if include_concept_map:
        concept_map: Dict[str, str] = raw_table.attrs.get("concept_map", {}) if hasattr(raw_table, "attrs") else {}
        if concept_map:
            md_lines.append("## XBRL Concept Map")
            md_lines.append("")
            md_lines.append("| Metric | XBRL Concept |")
            md_lines.append("|---|---|")
            for k, v in concept_map.items():
                md_lines.append(f"| {k} | `{v}` |")
            md_lines.append("")

    md_text = "\n".join(md_lines)
    out_path.write_text(md_text, encoding="utf-8")
    return md_text, str(out_path)


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate automated financial analysis report from SEC XBRL company facts."
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--years", type=int, default=5, help="Number of fiscal years to include")
    parser.add_argument("--out", default="reports", help="Output directory")
    parser.add_argument("--no-concept-map", action="store_true", help="Do not include concept map section")

    args = parser.parse_args()
    _, path = build_report_markdown(
        ticker=args.ticker,
        years=args.years,
        out_dir=args.out,
        include_concept_map=not args.no_concept_map,
    )
    print(path)


if __name__ == "__main__":
    main()