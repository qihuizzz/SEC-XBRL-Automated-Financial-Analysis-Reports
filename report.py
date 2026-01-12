# report.py
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from sec_client import make_default_client
from xbrl_normalize import normalize_pipeline
from financials import FinancialsConfig, build_annual_financials_table, format_financials_for_display
from viz import save_financial_charts


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


def _fmt_date(x) -> str:
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "NA"
    return str(x)


def _as_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    return df.to_markdown(index=False)


# -----------------------------
# Insights
# -----------------------------
def _make_insights(display_df: pd.DataFrame) -> str:
    if display_df.empty:
        return "- No data available."

    latest = display_df.iloc[-1]
    prev = display_df.iloc[-2] if len(display_df) >= 2 else None

    fy = _safe_float(latest.get("fy"))
    end = _fmt_date(latest.get("fiscal_year_end"))

    rev = _safe_float(latest.get("revenue"))
    rev_yoy = _safe_float(latest.get("revenue_yoy"))
    gm = _safe_float(latest.get("gross_margin"))
    om = _safe_float(latest.get("operating_margin"))
    nm = _safe_float(latest.get("net_margin"))
    fcf = _safe_float(latest.get("fcf"))
    fcfm = _safe_float(latest.get("fcf_margin"))

    lines = []
    if fy is not None:
        lines.append(f"- Latest fiscal year: **FY{int(fy)}** (ended {end}).")
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
    generate_charts: bool = True,
) -> Tuple[str, str]:
    """
    Returns (md_text, output_path)
    """
    ticker = ticker.strip().upper()
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    out_path = out_root / f"{ticker}.md"

    c = make_default_client()
    cik = c.ticker_to_cik(ticker)
    submissions = c.get_submissions(cik)
    company_name = submissions.get("name", ticker)

    facts = c.get_companyfacts(cik)
    df = normalize_pipeline(facts)

    cfg = FinancialsConfig(last_n_years=years)
    raw_table = build_annual_financials_table(df, cfg=cfg)
    display_table = format_financials_for_display(raw_table)

    # Generate charts and compute relative paths for markdown
    charts: Dict[str, str] = {}
    if generate_charts and not display_table.empty:
        asset_dir = out_root / "assets" / ticker
        charts = save_financial_charts(display_table, str(asset_dir))

    def _rel_path(p: str) -> str:
        # markdown file is in out_root, so we want relative paths from out_root
        return str(Path(p).relative_to(out_root)).replace("\\", "/")

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
    cols = [col for col in preferred_cols if col in display_table.columns] + [
        col for col in display_table.columns if col not in preferred_cols
    ]
    display_table = display_table[cols].copy()

    # Convert to markdown-friendly strings
    if "fy" in display_table.columns:
        display_table["fy"] = display_table["fy"].apply(
            lambda x: f"{int(x)}" if _safe_float(x) is not None else "NA"
        )
    if "fiscal_year_end" in display_table.columns:
        display_table["fiscal_year_end"] = display_table["fiscal_year_end"].apply(_fmt_date)

    # Numeric columns in billions
    billions_cols = [
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "cfo",
        "capex",
        "fcf",
        "cash",
        "equity",
    ]
    for col in billions_cols:
        if col in display_table.columns:
            display_table[col] = display_table[col].apply(lambda x: _num(_safe_float(x), digits=1))

    # Percent columns
    percent_cols = ["revenue_yoy", "gross_margin", "operating_margin", "net_margin", "fcf_margin"]
    for col in percent_cols:
        if col in display_table.columns:
            display_table[col] = display_table[col].apply(lambda x: _pct(_safe_float(x), digits=1))

    md_lines = []
    md_lines.append(f"# {company_name} ({ticker}) Automated Financial Analysis Report")
    md_lines.append("")
    md_lines.append(f"- CIK: `{cik}`")
    md_lines.append(f"- Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"- Coverage: last {years} fiscal years")
    md_lines.append("")

    md_lines.append("## Highlights")
    md_lines.append(_make_insights(format_financials_for_display(raw_table)))
    md_lines.append("")

    if charts:
        md_lines.append("## Charts")
        md_lines.append("")
        order = ["revenue", "margins", "cash_flow", "revenue_yoy"]
        titles = {
            "revenue": "Revenue",
            "margins": "Margins",
            "cash_flow": "Cash flow",
            "revenue_yoy": "Revenue YoY",
        }
        for k in order:
            if k in charts:
                title = titles[k]
                md_lines.append(f"### {title}")
                md_lines.append(f"![{title}]({_rel_path(charts[k])})")
                md_lines.append("")
    else:
        md_lines.append("## Charts")
        md_lines.append("")
        md_lines.append("_No charts generated._")
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
    parser.add_argument("--no-charts", action="store_true", help="Do not generate charts")

    args = parser.parse_args()

    _, path = build_report_markdown(
        ticker=args.ticker,
        years=args.years,
        out_dir=args.out,
        include_concept_map=not args.no_concept_map,
        generate_charts=not args.no_charts,
    )
    print(path)


if __name__ == "__main__":
    main()