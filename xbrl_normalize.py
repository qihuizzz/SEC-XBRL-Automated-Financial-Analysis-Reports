from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import pandas as pd


@dataclass
class NormalizeConfig:
    taxonomy: str = "us-gaap"
    preferred_units: List[str] = None
    keep_dimensions: bool = False
    keep_forms: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.preferred_units is None:
            self.preferred_units = ["USD", "shares", "pure"]


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default)


def normalize_companyfacts_to_df(companyfacts: Dict[str, Any], cfg: Optional[NormalizeConfig] = None) -> pd.DataFrame:
    """
    Convert SEC companyfacts JSON into a flat DataFrame.

    Output columns
    taxonomy, concept, label, description, unit, value, start, end, fy, fp, form, filed, frame, accn, qtrs, has_dimensions
    """
    if cfg is None:
        cfg = NormalizeConfig()

    facts_root = _safe_get(companyfacts, "facts", {})
    tax = _safe_get(facts_root, cfg.taxonomy, {})

    rows: List[Dict[str, Any]] = []

    for concept, concept_obj in tax.items():
        label = _safe_get(concept_obj, "label", "")
        description = _safe_get(concept_obj, "description", "")
        units_obj = _safe_get(concept_obj, "units", {})

        for unit, facts_list in units_obj.items():
            if not isinstance(facts_list, list):
                continue

            for item in facts_list:
                if not isinstance(item, dict):
                    continue

                form = _as_str(_safe_get(item, "form", ""))
                if cfg.keep_forms is not None and form not in cfg.keep_forms:
                    continue

                has_dims = False
                if "dims" in item and item["dims"]:
                    has_dims = True
                    if not cfg.keep_dimensions:
                        continue

                rows.append(
                    {
                        "taxonomy": cfg.taxonomy,
                        "concept": concept,
                        "label": label,
                        "description": description,
                        "unit": unit,
                        "value": _safe_get(item, "val", None),
                        "start": _safe_get(item, "start", None),
                        "end": _safe_get(item, "end", None),
                        "fy": _safe_get(item, "fy", None),
                        "fp": _safe_get(item, "fp", None),
                        "form": form,
                        "filed": _safe_get(item, "filed", None),
                        "frame": _safe_get(item, "frame", None),
                        "accn": _safe_get(item, "accn", None),
                        "qtrs": _safe_get(item, "qtrs", None),
                        "has_dimensions": has_dims,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df["period_type"] = df["start"].isna().map(lambda x: "instant" if x else "duration")

    return df


def choose_preferred_unit(df: pd.DataFrame, preferred_units: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Keep rows only for units in preferred_units and return the best available unit per concept.
    Strategy
    1 filter to preferred units if present
    2 if multiple preferred units exist for a concept keep the earliest in preferred_units list
    """
    if df.empty:
        return df

    if preferred_units is None:
        preferred_units = ["USD", "shares", "pure"]

    df2 = df.copy()

    df2["unit_rank"] = df2["unit"].map(lambda u: preferred_units.index(u) if u in preferred_units else 10_000)

    df2 = df2[df2["unit_rank"] < 10_000]
    if df2.empty:
        return df2

    best_rank = df2.groupby("concept")["unit_rank"].transform("min")
    df2 = df2[df2["unit_rank"] == best_rank].drop(columns=["unit_rank"])

    return df2


def dedupe_keep_latest_filed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dedupe facts by keeping the latest filed record for the same concept, unit, start, end, fy, fp, form.
    """
    if df.empty:
        return df

    df2 = df.copy()

    df2 = df2.sort_values(["concept", "unit", "start", "end", "fy", "fp", "form", "filed"], ascending=True)

    key_cols = ["concept", "unit", "start", "end", "fy", "fp", "form"]
    df2 = df2.drop_duplicates(subset=key_cols, keep="last")

    return df2


def filter_forms(df: pd.DataFrame, forms: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return df
    forms_set = set(forms)
    return df[df["form"].isin(forms_set)].copy()


def summarize_concepts(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """
    Quick summary table to help you see what concepts exist and how many records you got.
    """
    if df.empty:
        return pd.DataFrame(columns=["concept", "rows", "units"])

    g = df.groupby("concept").agg(
        rows=("value", "size"),
        units=("unit", lambda x: sorted(set(x))),
    )
    g = g.sort_values("rows", ascending=False).head(top_n).reset_index()
    return g


def extract_concept_series(df: pd.DataFrame, concept: str, unit: str = "USD") -> pd.DataFrame:
    """
    Extract a single concept time series.
    Returns rows sorted by end date.
    """
    if df.empty:
        return pd.DataFrame()

    d = df[(df["concept"] == concept) & (df["unit"] == unit)].copy()
    d = d.sort_values(["end", "filed"], ascending=True)
    return d


def normalize_pipeline(companyfacts: Dict[str, Any], cfg: Optional[NormalizeConfig] = None) -> pd.DataFrame:
    """
    One call pipeline
    normalize to df, choose preferred unit, dedupe latest filed
    """
    if cfg is None:
        cfg = NormalizeConfig()

    df = normalize_companyfacts_to_df(companyfacts, cfg=cfg)
    df = choose_preferred_unit(df, preferred_units=cfg.preferred_units)
    df = dedupe_keep_latest_filed(df)
    return df