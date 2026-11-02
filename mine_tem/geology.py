"""Validated borehole/geology exchange and geometry-only candidate links."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io import _read_table


ALIASES = {
    "line_id": ["line_id", "line", "测线", "测线号", "测线编号"],
    "borehole_id": ["borehole_id", "borehole", "钻孔", "钻孔编号", "孔号"],
    "x_m": ["x_m", "x", "横坐标", "x坐标", "里程m", "钻孔位置m"],
    "y_m": ["y_m", "y", "纵坐标", "y坐标"],
    "elevation_m": ["elevation_m", "elevation", "z_m", "高程m", "高程", "孔口高程"],
    "depth_top_m": ["depth_top_m", "depth_top", "顶深m", "区间顶深m", "起始深度m"],
    "depth_bottom_m": ["depth_bottom_m", "depth_bottom", "底深m", "区间底深m", "终止深度m"],
    "lithology": ["lithology", "岩性", "地层", "岩性描述"],
    "water_evidence": ["water_evidence", "water", "含水证据", "水文证据", "涌水记录"],
    "confidence": ["confidence", "置信度", "证据置信度"],
    "notes": ["notes", "备注", "说明"],
}
LOOKUP = {alias.lower().strip().replace(" ", ""): canonical
          for canonical, aliases in ALIASES.items() for alias in aliases}

BOREHOLE_REQUIRED = (
    "line_id", "borehole_id", "x_m", "y_m", "elevation_m", "depth_top_m",
    "depth_bottom_m", "lithology",
)
LINK_COLUMNS = [
    "line_id", "borehole_id", "candidate_id", "x_distance_m", "depth_overlap_m",
    "candidate_x_start_m", "candidate_x_end_m", "candidate_depth_top_m",
    "candidate_depth_bottom_m", "lithology", "water_evidence", "confidence",
    "notes", "spatial_relation", "evidence_level",
]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip().lower().replace(" ", "") for c in d.columns]
    d = d.rename(columns={c: LOOKUP.get(c, c) for c in d.columns})
    if d.columns.duplicated().any():
        raise ValueError("多个原始地质列被识别为同一字段，请使用标准表头")
    return d


def canonicalize_boreholes(df: pd.DataFrame, source_name: str = "地质/钻孔") -> pd.DataFrame:
    """Map documented Chinese/English headers and validate interval records."""
    d = _normalise_columns(df)
    if "line_id" not in d:
        d["line_id"] = "L1"
    for name, default in (("y_m", 0.0), ("elevation_m", 0.0)):
        if name not in d:
            d[name] = default
    for name, default in (("water_evidence", ""), ("confidence", np.nan), ("notes", "")):
        if name not in d:
            d[name] = default
    missing = [name for name in BOREHOLE_REQUIRED if name not in d]
    if missing:
        raise ValueError(f"{source_name} 缺少字段：" + ", ".join(missing))
    d["line_id"] = d["line_id"].fillna("").astype(str).str.strip()
    d["borehole_id"] = d["borehole_id"].fillna("").astype(str).str.strip()
    d["lithology"] = d["lithology"].fillna("").astype(str).str.strip()
    for name in ("water_evidence", "notes"):
        d[name] = d[name].fillna("").astype(str).str.strip()
    for name in ("x_m", "y_m", "elevation_m", "depth_top_m", "depth_bottom_m", "confidence"):
        d[name] = pd.to_numeric(d[name], errors="coerce")
    return validate_boreholes(d, source_name=source_name)


def validate_boreholes(df: pd.DataFrame, source_name: str = "地质/钻孔") -> pd.DataFrame:
    missing = [name for name in BOREHOLE_REQUIRED if name not in df.columns]
    if missing:
        raise ValueError(f"{source_name} 缺少字段：" + ", ".join(missing))
    d = df.copy()
    for name in ("line_id", "borehole_id", "lithology"):
        d[name] = d[name].fillna("").astype(str).str.strip()
    if (d["line_id"] == "").any() or (d["borehole_id"] == "").any():
        raise ValueError(f"{source_name} 的测线和钻孔编号不得为空")
    if (d["lithology"] == "").any():
        raise ValueError(f"{source_name} 的岩性描述不得为空")
    numeric = ("x_m", "y_m", "elevation_m", "depth_top_m", "depth_bottom_m")
    for name in numeric:
        d[name] = pd.to_numeric(d[name], errors="coerce")
    if d[list(numeric)].isna().any().any() or not np.isfinite(d[list(numeric)]).all().all():
        raise ValueError(f"{source_name} 坐标、高程和深度必须是有限数值")
    if (d["depth_top_m"] < 0).any() or (d["depth_bottom_m"] <= d["depth_top_m"]).any():
        raise ValueError(f"{source_name} 顶深必须小于底深，且深度不能为负")
    if "confidence" in d:
        d["confidence"] = pd.to_numeric(d["confidence"], errors="coerce")
        bad = d["confidence"].notna() & ((d["confidence"] < 0) | (d["confidence"] > 1))
        if bad.any():
            raise ValueError(f"{source_name} 置信度必须在 0 到 1 之间")
    for (line_id, borehole_id), g in d.groupby(["line_id", "borehole_id"], sort=False):
        for coordinate in ("x_m", "y_m", "elevation_m"):
            if (g[coordinate].max() - g[coordinate].min()) > 1e-6:
                raise ValueError(f"钻孔 {borehole_id} 的 {coordinate} 坐标不一致")
        ordered = g.sort_values("depth_top_m")
        if len(ordered) > 1 and (ordered["depth_top_m"].to_numpy()[1:] <
                                  ordered["depth_bottom_m"].to_numpy()[:-1] - 1e-9).any():
            raise ValueError(f"钻孔 {borehole_id} 的区间重叠")
    return d.reset_index(drop=True)


def read_boreholes(paths: Iterable[str | Path]) -> pd.DataFrame:
    paths = list(paths)
    if not paths:
        raise ValueError("请选择至少一个地质/钻孔文件")
    frames = []
    for item in paths:
        path = Path(item)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        frames.append(canonicalize_boreholes(_read_table(path), source_name=path.name))
    return validate_boreholes(pd.concat(frames, ignore_index=True))


def _empty_links() -> pd.DataFrame:
    return pd.DataFrame(columns=LINK_COLUMNS)


def link_boreholes_to_candidates(boreholes: pd.DataFrame, result, x_tolerance: float | None = None) -> pd.DataFrame:
    """Return nearby candidate pairs as geometry-only evidence records.

    Candidate depths are pseudo-depths. A pair can be useful even when its
    depth intervals do not overlap, so those pairs are retained with an
    explicit relation label instead of being silently discarded.
    """
    b = validate_boreholes(boreholes)
    candidates = result.anomalies
    if b.empty or candidates is None or candidates.empty:
        return _empty_links()
    if "x_m" not in result.profile or len(result.profile["x_m"]) == 0:
        return _empty_links()
    xs = np.asarray(result.profile["x_m"], dtype=float)
    dx = np.diff(np.sort(xs))
    dx = dx[np.isfinite(dx) & (dx > 0)]
    if x_tolerance is None:
        x_tolerance = float(np.median(dx) / 2) if len(dx) else 0.5
    if not np.isfinite(x_tolerance) or x_tolerance < 0:
        raise ValueError("钻孔 X 关联容差必须是有限非负数")
    rows = []
    current_line = str(result.line_id)
    for borehole in b[b["line_id"].astype(str) == current_line].itertuples(index=False):
        for candidate in candidates.itertuples(index=False):
            x_start = float(candidate.x_start_m); x_end = float(candidate.x_end_m)
            x = float(borehole.x_m)
            distance = max(x_start - x, 0.0, x - x_end)
            if distance > x_tolerance:
                continue
            top = max(float(borehole.depth_top_m), float(candidate.depth_top_m))
            bottom = min(float(borehole.depth_bottom_m), float(candidate.depth_bottom_m))
            overlap = max(0.0, bottom - top)
            relation = "伪深度区间重叠" if overlap > 0 else "X邻近但深度无重叠"
            rows.append({
                "line_id": current_line,
                "borehole_id": str(borehole.borehole_id),
                "candidate_id": str(candidate.id),
                "x_distance_m": round(float(distance), 6),
                "depth_overlap_m": round(float(overlap), 6),
                "candidate_x_start_m": float(candidate.x_start_m),
                "candidate_x_end_m": float(candidate.x_end_m),
                "candidate_depth_top_m": float(candidate.depth_top_m),
                "candidate_depth_bottom_m": float(candidate.depth_bottom_m),
                "lithology": str(borehole.lithology),
                "water_evidence": str(getattr(borehole, "water_evidence", "")),
                "confidence": getattr(borehole, "confidence", np.nan),
                "notes": str(getattr(borehole, "notes", "")),
                "spatial_relation": relation,
                "evidence_level": "几何辅助证据，需人工复核",
            })
    if not rows:
        return _empty_links()
    return pd.DataFrame(rows, columns=LINK_COLUMNS).sort_values(
        ["candidate_id", "x_distance_m", "borehole_id"], kind="stable").reset_index(drop=True)
