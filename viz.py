# viz.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_financial_charts(display_df: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    """
    display_df is the output of format_financials_for_display
    It contains USD lines scaled to billions, margin columns as decimals

    Returns a dict: chart_key -> relative_path
    """
    out_path = Path(out_dir)
    _ensure_dir(out_path)

    df = display_df.copy()
    if df.empty:
        return {}

    df = df.sort_values("fy", ascending=True)
    x = df["fy"].astype(int).tolist()

    saved: Dict[str, str] = {}

    # Chart 1 revenue
    if "revenue" in df.columns and df["revenue"].notna().any():
        plt.figure()
        plt.plot(x, df["revenue"])
        plt.title("Revenue")
        plt.xlabel("Fiscal year")
        plt.ylabel("USD billions")
        plt.tight_layout()

        fn = out_path / "revenue.png"
        plt.savefig(fn, dpi=160)
        plt.close()
        saved["revenue"] = str(fn).replace("\\", "/")

    # Chart 2 margins
    margin_cols = [
        ("gross_margin", "Gross margin"),
        ("operating_margin", "Operating margin"),
        ("net_margin", "Net margin"),
    ]
    have_any = any(c in df.columns and df[c].notna().any() for c, _ in margin_cols)
    if have_any:
        plt.figure()
        for c, label in margin_cols:
            if c in df.columns and df[c].notna().any():
                plt.plot(x, df[c], label=label)
        plt.title("Margins")
        plt.xlabel("Fiscal year")
        plt.ylabel("Ratio")
        plt.legend()
        plt.tight_layout()

        fn = out_path / "margins.png"
        plt.savefig(fn, dpi=160)
        plt.close()
        saved["margins"] = str(fn).replace("\\", "/")

    # Chart 3 cash flow
    cf_cols = [
        ("cfo", "CFO"),
        ("capex", "Capex"),
        ("fcf", "FCF"),
    ]
    have_any = any(c in df.columns and df[c].notna().any() for c, _ in cf_cols)
    if have_any:
        plt.figure()
        for c, label in cf_cols:
            if c in df.columns and df[c].notna().any():
                plt.plot(x, df[c], label=label)
        plt.title("Cash flow")
        plt.xlabel("Fiscal year")
        plt.ylabel("USD billions")
        plt.legend()
        plt.tight_layout()

        fn = out_path / "cash_flow.png"
        plt.savefig(fn, dpi=160)
        plt.close()
        saved["cash_flow"] = str(fn).replace("\\", "/")

    # Chart 4 yoy
    if "revenue_yoy" in df.columns and df["revenue_yoy"].notna().any():
        plt.figure()
        plt.bar(x, df["revenue_yoy"])
        plt.title("Revenue YoY")
        plt.xlabel("Fiscal year")
        plt.ylabel("YoY")
        plt.tight_layout()

        fn = out_path / "revenue_yoy.png"
        plt.savefig(fn, dpi=160)
        plt.close()
        saved["revenue_yoy"] = str(fn).replace("\\", "/")

    return saved