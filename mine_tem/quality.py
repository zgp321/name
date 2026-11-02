"""Auditable summaries for the processed gate-level QC records."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_QC_COLUMNS = (
    "station",
    "x_m",
    "y_m",
    "elevation_m",
    "time_s",
    "qc",
    "accepted",
    "snr_min",
)


def _empty_station_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "station", "x_m", "y_m", "elevation_m", "total_gates", "accepted_gates",
        "rejected_gates", "acceptance_rate", "snr_median", "snr_min",
        "first_valid_time_s", "last_valid_time_s", "qc_breakdown",
    ])


def _empty_reason_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=["qc", "gate_count", "accepted_gates", "rejected_gates", "share"])


def build_quality_report(readings: pd.DataFrame) -> dict[str, Any]:
    """Build station, reason and gate-level views without changing QC decisions.

    The returned ``gate_matrix`` is intentionally a long table. A time gate can
    be absent for a station, so pivoting it to a dense matrix would make a
    missing measurement look like a rejected measurement.
    """
    missing = [c for c in REQUIRED_QC_COLUMNS if c not in readings.columns]
    if missing:
        raise ValueError("质量诊断缺少字段：" + ", ".join(missing))
    d = readings.copy()
    if d.empty:
        return {
            "totals": {"total_gates": 0, "accepted_gates": 0, "rejected_gates": 0,
                       "acceptance_rate": 0.0},
            "station_summary": _empty_station_summary(),
            "reason_summary": _empty_reason_summary(),
            "gate_matrix": d.loc[:, list(REQUIRED_QC_COLUMNS)].copy(),
        }
    accepted = d["accepted"].astype("boolean")
    if accepted.isna().any():
        raise ValueError("质量诊断 accepted 字段包含空值")
    d["accepted"] = accepted.astype(bool)
    d["station"] = d["station"].astype(str).str.strip()
    d["qc"] = d["qc"].astype(str).str.strip()
    for col in ("x_m", "y_m", "elevation_m", "time_s", "snr_min"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if d[list(REQUIRED_QC_COLUMNS)].isna().any().any():
        raise ValueError("质量诊断字段包含无法解析的数值或空值")
    d = d.sort_values(["x_m", "station", "time_s"], kind="stable").reset_index(drop=True)

    station_rows = []
    for station, g in d.groupby("station", sort=False):
        valid_snr = g.loc[np.isfinite(g["snr_min"]), "snr_min"]
        valid_times = g.loc[g["accepted"], "time_s"]
        reason_counts = Counter(str(v) for v in g["qc"])
        total = int(len(g)); accepted_count = int(g["accepted"].sum())
        station_rows.append({
            "station": station,
            "x_m": float(g["x_m"].iloc[0]),
            "y_m": float(g["y_m"].iloc[0]),
            "elevation_m": float(g["elevation_m"].iloc[0]),
            "total_gates": total,
            "accepted_gates": accepted_count,
            "rejected_gates": total - accepted_count,
            "acceptance_rate": accepted_count / total if total else 0.0,
            "snr_median": float(valid_snr.median()) if len(valid_snr) else np.nan,
            "snr_min": float(valid_snr.min()) if len(valid_snr) else np.nan,
            "first_valid_time_s": float(valid_times.min()) if len(valid_times) else np.nan,
            "last_valid_time_s": float(valid_times.max()) if len(valid_times) else np.nan,
            "qc_breakdown": "; ".join(f"{key}: {value}" for key, value in sorted(reason_counts.items())),
        })
    stations = pd.DataFrame(station_rows, columns=_empty_station_summary().columns)
    reason = (d.groupby("qc", sort=False, dropna=False)
              .agg(gate_count=("qc", "size"), accepted_gates=("accepted", "sum"))
              .reset_index())
    reason["accepted_gates"] = reason["accepted_gates"].astype(int)
    reason["rejected_gates"] = reason["gate_count"] - reason["accepted_gates"]
    reason["share"] = reason["gate_count"] / len(d)
    reason = reason.loc[:, _empty_reason_summary().columns]
    gates = d.loc[:, [
        c for c in ("line_id", "station", "x_m", "y_m", "elevation_m", "time_s",
                    "qc", "accepted", "snr_min", "raw_dbdt_t_s", "dbdt_clean_t_s",
                    "rho_apparent_ohm_m", "pseudo_depth_m") if c in d.columns
    ]].copy()
    total = int(len(d)); accepted_count = int(d["accepted"].sum())
    return {
        "totals": {
            "total_gates": total,
            "accepted_gates": accepted_count,
            "rejected_gates": total - accepted_count,
            "acceptance_rate": accepted_count / total if total else 0.0,
        },
        "station_summary": stations,
        "reason_summary": reason,
        "gate_matrix": gates,
    }
