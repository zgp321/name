from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import pandas as pd

MU0 = 4 * np.pi * 1e-7

@dataclass
class Settings:
    # Equipment configuration: ALL must match field instruments before engineering use.
    tx_area_m2: float = 100.0
    tx_turns: int = 1
    current_a: float = 1.0
    rx_area_m2: float = 1.0
    rx_turns: int = 1
    noise_floor_t_per_s: float = 1e-13
    spike_log10_threshold: float = 0.48
    smooth_window: int = 5
    min_snr: float = 3.0
    depth_factor: float = 1.0
    low_rho_ohm_m: float = 35.0
    contrast_ratio: float = 0.7
    min_area_m2: float = 80.0
    geology: str = "未知（需结合地质资料）"
    # 2D-only 3D depiction is a display assumption, NOT measured volume.
    display_extrusion_width_m: float = 20.0
    late_offset_correction: bool = False

    def validate(self) -> None:
        for name in ("tx_area_m2", "current_a", "rx_area_m2", "noise_floor_t_per_s", "min_snr",
                     "depth_factor", "low_rho_ohm_m", "min_area_m2", "display_extrusion_width_m"):
            if not (np.isfinite(getattr(self, name)) and getattr(self, name) > 0):
                raise ValueError(f"参数 {name} 必须是有限正数")
        if self.tx_turns < 1 or self.rx_turns < 1:
            raise ValueError("发射/接收匝数必须大于或等于1")
        if not 0 < self.contrast_ratio <= 1:
            raise ValueError("contrast_ratio 必须在(0,1]之间")
        if self.spike_log10_threshold <= 0 or self.smooth_window < 1:
            raise ValueError("去尖峰参数无效")

    @property
    def tx_moment(self) -> float:
        return self.tx_area_m2 * self.tx_turns * self.current_a

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Result:
    readings: pd.DataFrame
    profile: dict[str, Any]
    anomalies: pd.DataFrame
    quality: dict[str, Any]
    settings: Settings
    line_id: str


REQUIRED_COMMON = ["line_id", "station", "x_m", "y_m", "elevation_m", "time_s"]


def validate_measurements(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COMMON if c not in df.columns]
    if missing:
        raise ValueError("缺少必要列：" + ", ".join(missing))
    if "voltage_v" not in df.columns and "dbdt_t_s" not in df.columns:
        raise ValueError("至少需要 voltage_v（接收电压，V）或 dbdt_t_s（磁场变化率，T/s）列")
    result = df.copy()
    result["line_id"] = result["line_id"].fillna("L1").astype(str).str.strip()
    result["station"] = result["station"].astype(str).str.strip()
    if (result["station"] == "").any() or (result["line_id"] == "").any():
        raise ValueError("测线/测点标识不得为空")
    numeric = ["x_m", "y_m", "elevation_m", "time_s"]
    numeric += [c for c in ("voltage_v", "dbdt_t_s") if c in result]
    for c in numeric:
        result[c] = pd.to_numeric(result[c], errors="coerce")
    if result[["x_m", "y_m", "elevation_m", "time_s"]].isna().any().any():
        raise ValueError("坐标、高程和时间列存在缺失或非数字")
    if (result["time_s"] <= 0).any() or not np.isfinite(result["time_s"]).all():
        raise ValueError("时间门必须为正、有限值（单位秒）")
    # Explicit dB/dt values take priority, allow missing if voltage is provided.
    if "dbdt_t_s" in result and "voltage_v" in result:
        bad = result["dbdt_t_s"].isna() & result["voltage_v"].isna()
    else:
        col = "dbdt_t_s" if "dbdt_t_s" in result else "voltage_v"
        bad = result[col].isna()
    if bad.any():
        raise ValueError(f"存在 {int(bad.sum())} 行缺少有效接收信号")
    if result.duplicated(["line_id", "station", "time_s"]).any():
        raise ValueError("同一测线、测点出现重复时间门；请核查并在导入前合并")
    # A station should have one physical location.
    grouping = result.groupby(["line_id", "station"], sort=False)
    for coord in ("x_m", "y_m", "elevation_m"):
        if (grouping[coord].max() - grouping[coord].min()).abs().gt(1e-6).any():
            raise ValueError(f"同一测点的 {coord} 坐标不一致")
    if len(result) > 2_000_000:
        raise ValueError("单批次最多200万数据行；请拆分批次")
    return result.sort_values(["line_id", "x_m", "station", "time_s"]).reset_index(drop=True)
