"""Documented exchange format, Chinese/English headers and text/Excel inputs."""
from pathlib import Path
import re
import pandas as pd
from .model import validate_measurements

ALIASES = {
    "line_id": ["line_id", "line", "测线", "测线号", "剖面", "测线编号"],
    "station": ["station", "station_id", "point", "测点", "点号", "测点号"],
    "x_m": ["x_m", "x", "横坐标", "x坐标", "里程m", "测点位置m"],
    "y_m": ["y_m", "y", "纵坐标", "y坐标"],
    "elevation_m": ["elevation_m", "elevation", "z_m", "高程m", "地表高程", "高程"],
    "time_s": ["time_s", "time", "时间s", "时间_s"],
    "time_ms": ["time_ms", "时间ms", "时间_ms", "时间毫秒"],
    "time_us": ["time_us", "时间us", "时间微秒"],
    "voltage_v": ["voltage_v", "voltage", "电压v", "接收电压v", "感应电压v"],
    "voltage_mv": ["voltage_mv", "电压mv", "接收电压mv"],
    "voltage_uv": ["voltage_uv", "电压uv", "接收电压uv"],
    "dbdt_t_s": ["dbdt_t_s", "dbdt", "db/dt", "磁场变化率t/s", "磁场变化率"],
}
LOOKUP = {alias.lower().strip(): canonical for canonical, aliases in ALIASES.items() for alias in aliases}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return pd.read_excel(path)
    if path.suffix.lower() not in (".csv", ".txt", ".tsv"):
        raise ValueError(f"不支持文件格式: {path.suffix}")
    err = None
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, comment="#")
        except (UnicodeError, pd.errors.ParserError) as exc:
            err = exc
    raise ValueError(f"无法解析 {path.name}: {err}")


def canonicalize(df: pd.DataFrame, source_name: str = "import") -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip().replace(" ", "").lower() for c in d.columns]
    d = d.rename(columns={c: LOOKUP.get(c, c) for c in d.columns})
    if d.columns.duplicated().any():
        raise ValueError("多个原始列被识别为同一字段，请使用标准表头")
    if "line_id" not in d:
        d["line_id"] = "L1"
    if "station" not in d:
        raise ValueError(f"{source_name} 缺少 station/测点 列")
    if "x_m" not in d:
        raise ValueError(f"{source_name} 缺少 x_m/横坐标 列")
    if "y_m" not in d:
        d["y_m"] = 0.0
    if "elevation_m" not in d:
        d["elevation_m"] = 0.0
    if "time_s" not in d:
        if "time_ms" in d:
            d["time_s"] = pd.to_numeric(d.pop("time_ms"), errors="coerce") / 1000.0
        elif "time_us" in d:
            d["time_s"] = pd.to_numeric(d.pop("time_us"), errors="coerce") / 1e6
        else:
            # Standard wide-form e.g. t_0.05ms, t_0.1ms (signal voltage in V)
            wide = []
            for col in d:
                match = re.fullmatch(r"t_(\d+(?:\.\d+)?)(us|ms|s)", col)
                if match:
                    v, unit = match.groups()
                    wide.append((col, float(v) * {"us": 1e-6, "ms": 1e-3, "s": 1.0}[unit]))
            if not wide:
                raise ValueError(f"{source_name} 缺少 time_s / time_ms 或 t_...ms 时间门列")
            ids = [c for c in d if c not in {c for c, _ in wide}]
            out = d.melt(id_vars=ids, value_vars=[c for c, _ in wide], var_name="_gate", value_name="voltage_v")
            gate_map = dict(wide)
            out["time_s"] = out.pop("_gate").map(gate_map)
            d = out
    if "voltage_v" not in d:
        if "voltage_mv" in d:
            d["voltage_v"] = pd.to_numeric(d.pop("voltage_mv"), errors="coerce") / 1e3
        elif "voltage_uv" in d:
            d["voltage_v"] = pd.to_numeric(d.pop("voltage_uv"), errors="coerce") / 1e6
    return d


def read_files(paths: list[str | Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("请选择至少一个数据文件")
    dfs = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        dfs.append(canonicalize(_read_table(path), source_name=path.name))
    return validate_measurements(pd.concat(dfs, ignore_index=True))
