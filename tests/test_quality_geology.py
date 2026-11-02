import numpy as np
import pandas as pd
import pytest

from mine_tem.demo import make_demo
from mine_tem.geology import (
    canonicalize_boreholes,
    link_boreholes_to_candidates,
    validate_boreholes,
)
from mine_tem.processing import process
from mine_tem.quality import build_quality_report


def test_quality_report_preserves_gate_qc_and_station_totals():
    result = process(make_demo(n_stations=4))
    report = build_quality_report(result.readings)

    stations = report["station_summary"]
    gates = report["gate_matrix"]
    reasons = report["reason_summary"]
    assert len(stations) == 4
    assert len(gates) == len(result.readings)
    assert int(stations["total_gates"].sum()) == len(result.readings)
    assert int(stations["accepted_gates"].sum()) == int(result.readings["accepted"].sum())
    assert set(reasons["qc"]) == set(result.readings["qc"])
    assert stations["acceptance_rate"].between(0, 1).all()


def test_quality_report_rejects_missing_qc_columns():
    with pytest.raises(ValueError, match="质量诊断缺少字段"):
        build_quality_report(pd.DataFrame({"station": ["P1"]}))


def test_borehole_aliases_and_interval_validation():
    raw = pd.DataFrame(
        {
            "测线": ["L1"],
            "钻孔编号": ["ZK-01"],
            "横坐标": [100.0],
            "纵坐标": [0.0],
            "高程": [512.0],
            "顶深m": [10.0],
            "底深m": [80.0],
            "岩性": ["砂岩"],
            "水文证据": ["井壁潮湿"],
            "置信度": [0.8],
        }
    )
    boreholes = canonicalize_boreholes(raw)
    assert boreholes.loc[0, "borehole_id"] == "ZK-01"
    assert boreholes.loc[0, "depth_top_m"] == 10
    assert boreholes.loc[0, "lithology"] == "砂岩"
    assert validate_boreholes(boreholes).equals(boreholes)

    bad = boreholes.copy()
    bad.loc[0, "depth_bottom_m"] = bad.loc[0, "depth_top_m"]
    with pytest.raises(ValueError, match="顶深必须小于底深"):
        validate_boreholes(bad)


def test_borehole_overlapping_intervals_are_rejected():
    data = pd.DataFrame(
        {
            "line_id": ["L1", "L1"],
            "borehole_id": ["ZK-01", "ZK-01"],
            "x_m": [100.0, 100.0],
            "y_m": [0.0, 0.0],
            "elevation_m": [0.0, 0.0],
            "depth_top_m": [0.0, 20.0],
            "depth_bottom_m": [30.0, 40.0],
            "lithology": ["砂岩", "泥岩"],
        }
    )
    with pytest.raises(ValueError, match="区间重叠"):
        validate_boreholes(data)


def test_borehole_links_are_geometry_only_and_keep_external_evidence():
    result = process(make_demo(n_stations=8))
    anomaly = result.anomalies.iloc[0]
    boreholes = pd.DataFrame(
        {
            "line_id": [result.line_id],
            "borehole_id": ["ZK-01"],
            "x_m": [(anomaly.x_start_m + anomaly.x_end_m) / 2],
            "y_m": [0.0],
            "elevation_m": [0.0],
            "depth_top_m": [max(0.0, anomaly.depth_top_m - 5)],
            "depth_bottom_m": [anomaly.depth_bottom_m + 5],
            "lithology": ["砂岩"],
            "water_evidence": ["人工记录，未独立核验"],
            "confidence": [0.4],
        }
    )
    links = link_boreholes_to_candidates(boreholes, result)
    assert len(links) == 1
    assert links.loc[0, "borehole_id"] == "ZK-01"
    assert links.loc[0, "water_evidence"] == "人工记录，未独立核验"
    assert links.loc[0, "spatial_relation"] == "伪深度区间重叠"
    assert "含水" not in str(links.loc[0, "spatial_relation"])


def test_borehole_without_candidate_returns_empty_link_table():
    result = process(make_demo(n_stations=4))
    boreholes = pd.DataFrame(
        {
            "line_id": [result.line_id],
            "borehole_id": ["ZK-OUT"],
            "x_m": [9999.0],
            "y_m": [0.0],
            "elevation_m": [0.0],
            "depth_top_m": [1.0],
            "depth_bottom_m": [2.0],
            "lithology": ["未知"],
        }
    )
    links = link_boreholes_to_candidates(boreholes, result)
    assert links.empty
    assert list(links.columns)[:3] == ["line_id", "borehole_id", "candidate_id"]
