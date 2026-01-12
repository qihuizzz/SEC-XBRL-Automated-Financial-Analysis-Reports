from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd


@dataclass
class ConceptSpec:
    name: str
    candidates: List[str]
    unit: str = "USD"
    period_type: Optional[str] = None
    forms: Optional[List[str]] = None


@dataclass
class FinancialsConfig:
    forms_annual: List[str] = None
    last_n_years: int = 5

    def __post_init__(self) -> None:
        if self.forms_annual is None:
            self.forms_annual = ["10-K"]


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def _filter_forms(df: pd.DataFrame, forms: Iterable[str]) -> pd.DataFrame:
    forms_set = set(forms)
    return df[df["form"].isin(forms_set)].copy()


def _filter_unit(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    return df[df["unit"] == unit].copy()


def _filter_period_type(df: pd.DataFrame, period_type: Optional[str]) -> pd.DataFrame:
    if period_type is None:
        return df
    return df[df["period_type"] == period_type].copy()


def _annualize_duration_series(d: pd.DataFrame) -> pd.DataFrame:
    """
    Keep annual like duration facts.

    Preference order
    1 qtrs equals 4 when present
    2 else fp equals FY when present
    3 then duration about one year using start and end
    """
    if d.empty:
        return d

    out = d.copy()
    out["start"] = _ensure_datetime(out, "start") if "start" in out.columns else pd.NaT
    out["end"] = _ensure_datetime(out, "end")

    used_filter = False
    if "qtrs" in out.columns:
        q = pd.to_numeric(out["qtrs"], errors="coerce")
        q4 = out[q == 4].copy()
        if not q4.empty:
            out = q4
            used_filter = True

    if not used_filter and "fp" in out.columns:
        fp = out["fp"].astype(str).str.upper()
        fy_rows = out[fp == "FY"].copy()
        if not fy_rows.empty:
            out = fy_rows

    if out["start"].notna().any():
        dur_days = (out["end"] - out["start"]).dt.days
        out = out[(dur_days >= 330) & (dur_days <= 400)].copy()

    return out


def _select_best_row_per_fy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one best record per fy.

    Best means
    - end year closest to fy
    - prefer qtrs equals 4
    - prefer longer duration for duration facts
    - prefer latest filed
    - prefer later end
    """
    if df.empty:
        return df

    d = df.copy()
    d["end"] = _ensure_datetime(d, "end")
    d["start"] = _ensure_datetime(d, "start") if "start" in d.columns else pd.NaT
    d["filed"] = _ensure_datetime(d, "filed") if "filed" in d.columns else pd.NaT
    d["fy"] = pd.to_numeric(d.get("fy"), errors="coerce")

    d = d.dropna(subset=["fy", "end", "value"])
    if d.empty:
        return d

    d["fy_int"] = d["fy"].astype(int)
    d["end_year"] = d["end"].dt.year.astype("Int64")
    d["year_gap"] = (d["end_year"] - d["fy_int"]).abs()

    if "qtrs" in d.columns:
        q = pd.to_numeric(d["qtrs"], errors="coerce")
    else:
        q = pd.Series([pd.NA] * len(d), index=d.index)
    d["is_q4"] = q == 4

    dur_days = (d["end"] - d["start"]).dt.days
    dur_days = dur_days.where(d["start"].notna(), -1)
    d["dur_days"] = dur_days

    d = d.sort_values(
        ["fy_int", "year_gap", "is_q4", "dur_days", "filed", "end"],
        ascending=[True, True, False, False, False, False],
    )
    d = d.drop_duplicates(subset=["fy_int"], keep="first")

    d = d.drop(columns=["fy_int", "end_year", "year_gap", "is_q4", "dur_days"], errors="ignore")
    return d


def default_mvp_specs() -> List[ConceptSpec]:
    return [
        ConceptSpec(
            name="revenue",
            candidates=[
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues",
                "SalesRevenueNet",
            ],
            unit="USD",
            period_type="duration",
            forms=["10-K"],
        ),
        ConceptSpec(
            name="gross_profit",
            candidates=["GrossProfit"],
            unit="USD",
            period_type="duration",
            forms=["10-K"],
        ),
        ConceptSpec(
            name="operating_income",
            candidates=["OperatingIncomeLoss"],
            unit="USD",
            period_type="duration",
            forms=["10-K"],
        ),
        ConceptSpec(
            name="net_income",
            candidates=["NetIncomeLoss"],
            unit="USD",
            period_type="duration",
            forms=["10-K"],
        ),
        ConceptSpec(
            name="cfo",
            candidates=[
                "NetCashProvidedByUsedInOperatingActivities",
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            ],
            unit="USD",
            period_type="duration",
            forms=["10-K"],
        ),
        ConceptSpec(
            name="capex",
            candidates=[
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "PaymentsToAcquireProductiveAssets",
            ],
            unit="USD",
            period_type="duration",
            forms=["10-K"],
        ),
        ConceptSpec(
            name="cash",
            candidates=["CashAndCashEquivalentsAtCarryingValue"],
            unit="USD",
            period_type="instant",
            forms=["10-K"],
        ),
        ConceptSpec(
            name="equity",
            candidates=["StockholdersEquity"],
            unit="USD",
            period_type="instant",
            forms=["10-K"],
        ),
    ]


def _prepare_base(df: pd.DataFrame, spec: ConceptSpec) -> pd.DataFrame:
    base = df.copy()
    if spec.forms:
        base = _filter_forms(base, spec.forms)
    base = _filter_unit(base, spec.unit)
    base = _filter_period_type(base, spec.period_type)
    return base


def _extract_series_for_concept(base: pd.DataFrame, spec: ConceptSpec, concept: str) -> pd.DataFrame:
    d = base[base["concept"] == concept].copy()
    d = d.dropna(subset=["value", "end"])
    if d.empty:
        return d

    d["fy"] = pd.to_numeric(d.get("fy"), errors="coerce")
    d["end"] = _ensure_datetime(d, "end")
    d = d.dropna(subset=["fy", "end", "value"])
    if d.empty:
        return d

    if spec.period_type == "duration":
        d = _annualize_duration_series(d)

    d = _select_best_row_per_fy(d)
    return d


def _choose_best_concept(base: pd.DataFrame, spec: ConceptSpec) -> Tuple[Optional[str], pd.DataFrame]:
    """
    Choose the concept that provides the freshest fiscal year coverage.

    Score
    - max fy higher is better
    - row count higher is better
    - max end later is better
    """
    best_concept: Optional[str] = None
    best_df = pd.DataFrame()
    best_score: Optional[Tuple[int, int, pd.Timestamp]] = None

    for concept in spec.candidates:
        d = _extract_series_for_concept(base, spec, concept)
        if d.empty:
            continue

        max_fy = int(pd.to_numeric(d["fy"], errors="coerce").max())
        row_count = int(d["fy"].nunique())
        max_end = pd.to_datetime(d["end"], errors="coerce").max()

        score = (max_fy, row_count, max_end)

        if best_score is None or score > best_score:
            best_score = score
            best_concept = concept
            best_df = d

    return best_concept, best_df


def extract_annual_series(df: pd.DataFrame, spec: ConceptSpec, last_n_years: int = 5) -> pd.DataFrame:
    """
    Returns tidy series with columns fy fiscal_year_end value concept_used name
    """
    base = _prepare_base(df, spec)
    concept_used, d = _choose_best_concept(base, spec)

    if concept_used is None or d.empty:
        return pd.DataFrame(columns=["fy", "fiscal_year_end", "value", "concept_used", "name"])

    d = d.sort_values("fy", ascending=True).tail(last_n_years)

    out = d[["fy", "end", "value"]].copy()
    out = out.rename(columns={"end": "fiscal_year_end"})
    out["concept_used"] = concept_used
    out["name"] = spec.name
    return out


def build_annual_financials_table(
    df: pd.DataFrame,
    specs: Optional[List[ConceptSpec]] = None,
    cfg: Optional[FinancialsConfig] = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = FinancialsConfig()
    if specs is None:
        specs = default_mvp_specs()

    required_cols = {"concept", "unit", "value", "end", "form", "period_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input df missing required columns: {sorted(missing)}")

    all_series: List[pd.DataFrame] = []
    concept_map: Dict[str, str] = {}

    for spec in specs:
        s = extract_annual_series(df, spec, last_n_years=cfg.last_n_years)
        if s.empty:
            concept_map[spec.name] = ""
            continue
        concept_map[spec.name] = str(s["concept_used"].iloc[-1])
        all_series.append(s)

    if not all_series:
        return pd.DataFrame()

    long = pd.concat(all_series, ignore_index=True)
    long["fy"] = pd.to_numeric(long["fy"], errors="coerce")
    long["fiscal_year_end"] = pd.to_datetime(long["fiscal_year_end"], errors="coerce")
    long = long.dropna(subset=["fy"])

    wide = long.pivot_table(index="fy", columns="name", values="value", aggfunc="last").reset_index()
    wide = wide.sort_values("fy", ascending=True)

    fy_end = long.groupby("fy")["fiscal_year_end"].max().reset_index()
    wide = wide.merge(fy_end, on="fy", how="left")

    if "cfo" in wide.columns and "capex" in wide.columns:
        wide["fcf"] = wide["cfo"] - wide["capex"]
    else:
        wide["fcf"] = pd.NA

    if "revenue" in wide.columns:
        wide["revenue_yoy"] = wide["revenue"].pct_change(fill_method=None)
    else:
        wide["revenue_yoy"] = pd.NA

    if "gross_profit" in wide.columns and "revenue" in wide.columns:
        wide["gross_margin"] = wide["gross_profit"] / wide["revenue"]
    else:
        wide["gross_margin"] = pd.NA

    if "operating_income" in wide.columns and "revenue" in wide.columns:
        wide["operating_margin"] = wide["operating_income"] / wide["revenue"]
    else:
        wide["operating_margin"] = pd.NA

    if "net_income" in wide.columns and "revenue" in wide.columns:
        wide["net_margin"] = wide["net_income"] / wide["revenue"]
    else:
        wide["net_margin"] = pd.NA

    if "fcf" in wide.columns and "revenue" in wide.columns:
        wide["fcf_margin"] = wide["fcf"] / wide["revenue"]
    else:
        wide["fcf_margin"] = pd.NA

    wide.attrs["concept_map"] = concept_map
    return wide


def format_financials_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    scale_cols = ["revenue", "gross_profit", "operating_income", "net_income", "cfo", "capex", "fcf", "cash", "equity"]
    for col in scale_cols:
        if col in out.columns:
            out[col] = out[col] / 1e9
    return out