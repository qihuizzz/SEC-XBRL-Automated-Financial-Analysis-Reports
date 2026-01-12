from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _prep_df(display_df: pd.DataFrame) -> Tuple[List[int], pd.DataFrame]:
    d = display_df.copy()
    d = d.sort_values("fy", ascending=True).reset_index(drop=True)
    x = d["fy"].astype(int).tolist()
    return x, d


def _apply_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 170,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.titlepad": 10,
        }
    )


def _beautify_axis(ax) -> None:
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _series_has_data(s: pd.Series) -> bool:
    return pd.to_numeric(s, errors="coerce").notna().any()


def _annotate_last(ax, x: List[int], y: pd.Series, fmt: str) -> None:
    s = pd.to_numeric(y, errors="coerce")
    if s.notna().sum() == 0:
        return

    last_i = int(s.last_valid_index())
    last_x = x[last_i]
    last_y = float(s.iloc[last_i])

    ax.scatter([last_x], [last_y], zorder=5)
    ax.annotate(
        fmt.format(last_y),
        xy=(last_x, last_y),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.8},
    )


def _save(fig, out_path: Path, filename: str) -> str:
    fp = out_path / filename
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    return str(fp).replace("\\", "/")


def save_financial_charts(display_df: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    out_path = Path(out_dir)
    _ensure_dir(out_path)

    if display_df is None or display_df.empty:
        return {}

    _apply_style()
    x, df = _prep_df(display_df)

    saved: Dict[str, str] = {}

    billions_fmt = mtick.FormatStrFormatter("%.1f")
    pct_fmt_0 = mtick.PercentFormatter(xmax=1.0, decimals=0)

    def line_chart(
        key: str,
        title: str,
        series_list: List[Tuple[str, str]],
        y_label: str,
        y_formatter=None,
        annotate_suffix: str = "",
        fill_first: bool = False,
        filename: str = "",
    ) -> None:
        have_any = any(col in df.columns and _series_has_data(df[col]) for col, _ in series_list)
        if not have_any:
            return

        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for idx, (col, label) in enumerate(series_list):
            if col not in df.columns or not _series_has_data(df[col]):
                continue
            ax.plot(x, df[col], marker="o", linewidth=2.2, label=label)
            if fill_first and idx == 0:
                ax.fill_between(x, pd.to_numeric(df[col], errors="coerce"), alpha=0.10)

        ax.set_title(title)
        ax.set_xlabel("Fiscal year")
        ax.set_ylabel(y_label)
        if y_formatter is not None:
            ax.yaxis.set_major_formatter(y_formatter)

        if len(series_list) > 1:
            ax.legend(loc="best")

        _beautify_axis(ax)

        first_col = series_list[0][0]
        if first_col in df.columns and _series_has_data(df[first_col]):
            if annotate_suffix:
                _annotate_last(ax, x, df[first_col], "{:.1f}" + annotate_suffix)
            else:
                _annotate_last(ax, x, df[first_col], "{:.1f}")

        saved[key] = _save(fig, out_path, filename or f"{key}.png")

    def bar_chart(
        key: str,
        title: str,
        col: str,
        y_label: str,
        y_formatter=None,
        filename: str = "",
        zero_line: bool = False,
    ) -> None:
        if col not in df.columns or not _series_has_data(df[col]):
            return

        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        ax.bar(x, df[col])
        if zero_line:
            ax.axhline(0, linewidth=1)

        ax.set_title(title)
        ax.set_xlabel("Fiscal year")
        ax.set_ylabel(y_label)
        if y_formatter is not None:
            ax.yaxis.set_major_formatter(y_formatter)

        _beautify_axis(ax)
        saved[key] = _save(fig, out_path, filename or f"{key}.png")

    line_chart(
        key="revenue",
        title="Revenue",
        series_list=[("revenue", "Revenue")],
        y_label="USD billions",
        y_formatter=billions_fmt,
        annotate_suffix="B",
        fill_first=True,
        filename="revenue.png",
    )

    line_chart(
        key="margins",
        title="Margins",
        series_list=[
            ("gross_margin", "Gross margin"),
            ("operating_margin", "Operating margin"),
            ("net_margin", "Net margin"),
            ("fcf_margin", "FCF margin"),
        ],
        y_label="Percent",
        y_formatter=pct_fmt_0,
        filename="margins.png",
    )

    line_chart(
        key="cash_flow",
        title="Cash flow",
        series_list=[("cfo", "CFO"), ("capex", "Capex"), ("fcf", "FCF")],
        y_label="USD billions",
        y_formatter=billions_fmt,
        filename="cash_flow.png",
    )

    bar_chart(
        key="revenue_yoy",
        title="Revenue YoY",
        col="revenue_yoy",
        y_label="Percent",
        y_formatter=pct_fmt_0,
        filename="revenue_yoy.png",
        zero_line=True,
    )

    line_chart(
        key="income_statement",
        title="Income statement levels",
        series_list=[
            ("revenue", "Revenue"),
            ("gross_profit", "Gross profit"),
            ("operating_income", "Operating income"),
            ("net_income", "Net income"),
        ],
        y_label="USD billions",
        y_formatter=billions_fmt,
        filename="income_statement.png",
    )

    line_chart(
        key="balance_sheet",
        title="Balance sheet snapshot",
        series_list=[("cash", "Cash"), ("equity", "Equity")],
        y_label="USD billions",
        y_formatter=billions_fmt,
        filename="balance_sheet.png",
    )

    if "cfo" in df.columns and "net_income" in df.columns and _series_has_data(df["cfo"]) and _series_has_data(df["net_income"]):
        diff = pd.to_numeric(df["cfo"], errors="coerce") - pd.to_numeric(df["net_income"], errors="coerce")
        df["cfo_minus_net_income"] = diff

        bar_chart(
            key="cash_quality",
            title="Cash flow quality",
            col="cfo_minus_net_income",
            y_label="CFO minus net income, USD billions",
            y_formatter=billions_fmt,
            filename="cash_quality.png",
            zero_line=True,
        )

    if "net_income" in df.columns and "equity" in df.columns and _series_has_data(df["net_income"]) and _series_has_data(df["equity"]):
        roe = pd.to_numeric(df["net_income"], errors="coerce") / pd.to_numeric(df["equity"], errors="coerce")
        df["roe"] = roe

        line_chart(
            key="roe",
            title="Return on equity",
            series_list=[("roe", "ROE")],
            y_label="Percent",
            y_formatter=pct_fmt_0,
            filename="roe.png",
        )

    return saved