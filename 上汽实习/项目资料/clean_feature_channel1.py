from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HEADER_MARKER = "Date"
DATA_START_ROW = 20
EXPECTED_CHANNEL1_PARAMS = {
    "Channel": 1,
    "Program": 1,
    "Operation": 0,
    "Study": 0,
    "Time VexFilling": 12,
    "Time PartFilling": 60,
    "Time Prefilling": 1.5,
    "Time Filling": 2,
    "Time Balancing": 2,
    "Time Measuring": 6,
    "Time Venting": 1,
    "Pressure Filling": 2.3,
    "Pressure Measuring": 2,
    "Min. Pressure Measuring": 1.9,
    "Max. Pressure Measuring": 2.4,
    "Factor": 65,
    "Offset": 0.1,
    "Limit1": 2,
    "Limit2": 2,
    "Limit3": -3,
}


def normalize_column_name(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def convert_xls_with_excel(src: Path, dst: Path) -> Path:
    """Convert legacy .XLS through local Excel when pandas cannot read it."""
    try:
        import win32com.client  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Cannot read legacy .XLS directly. Install xlrd, or run on Windows with Excel installed."
        ) from exc

    dst.parent.mkdir(parents=True, exist_ok=True)
    excel = win32com.client.DispatchEx("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False
    try:
        wb = excel.Workbooks.Open(str(src.resolve()), 0, True)
        wb.SaveAs(str(dst.resolve()), 51)  # 51 = .xlsx
        wb.Close(False)
    finally:
        excel.Quit()
    return dst


def load_channel_export(path: Path) -> pd.DataFrame:
    """Load a FROEHLICH/MPS400 exported workbook and return raw data rows."""
    source = path
    if path.suffix.lower() == ".xls":
        cached = path.parent / "_analysis_converted" / f"{path.stem}.xlsx"
        if cached.exists() and cached.stat().st_mtime >= path.stat().st_mtime:
            source = cached
        else:
            source = convert_xls_with_excel(path, cached)

    preview = pd.read_excel(source, sheet_name=0, header=None, nrows=40, engine="openpyxl")
    header_row = None
    for i, row in preview.iterrows():
        if normalize_column_name(row.iloc[0]) == HEADER_MARKER:
            header_row = i
            break
    if header_row is None:
        # The known export puts field names at Excel row 20, but keep the scan above for robustness.
        header_row = DATA_START_ROW - 1

    df = pd.read_excel(source, sheet_name=0, header=header_row, engine="openpyxl")
    df.columns = [normalize_column_name(c) for c in df.columns]
    df = df[df["Date"].notna()].copy()
    df = df.dropna(axis=1, how="all")
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DateTime"] = pd.to_datetime(out["Date"], errors="coerce")

    numeric_cols = [
        "Device",
        "Channel",
        "Program",
        "Operation",
        "Study",
        "Pressure Filled",
        "Pressure Measured",
        "Leakrate",
        "Reject",
        "Pressure Difference",
        "Temperature Environment",
        "Temperature Part",
        "Difference Temperature",
        "Leakrate Uncompensated",
        "Time VexFilling",
        "Time PartFilling",
        "Time Prefilling",
        "Time Filling",
        "Time Balancing",
        "Time Measuring",
        "Time Venting",
        "Pressure Filling",
        "Pressure Measuring",
        "Min. Pressure Measuring",
        "Max. Pressure Measuring",
        "Factor",
        "Offset",
        "Limit1",
        "Limit2",
        "Limit3",
        "Flowsensor",
        "Transmitter",
        "PO",
        "FO",
        "RO1",
        "RO2",
        "LeakrateRaw",
        "LeakrateAfterSmoothing",
        "LeakrateAfterOC",
        "TestRun",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.sort_values("DateTime").reset_index(drop=True)


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = [
        "DateTime",
        "Leakrate",
        "Pressure Difference",
        "Pressure Measured",
        "Factor",
        "Offset",
        "Limit1",
        "Limit3",
    ]
    out["missing_required"] = out[required].isna().any(axis=1)

    for col, expected in EXPECTED_CHANNEL1_PARAMS.items():
        if col in out.columns:
            out[f"param_mismatch_{safe_name(col)}"] = ~np.isclose(out[col], expected, equal_nan=False)

    mismatch_cols = [c for c in out.columns if c.startswith("param_mismatch_")]
    out["param_mismatch_any"] = out[mismatch_cols].any(axis=1) if mismatch_cols else False

    out["pressure_out_of_range"] = (
        (out["Pressure Measured"] < out["Min. Pressure Measuring"])
        | (out["Pressure Measured"] > out["Max. Pressure Measuring"])
    )
    out["leak_over_upper_limit"] = out["Leakrate"] > out["Limit1"]
    out["leak_under_lower_limit"] = out["Leakrate"] < out["Limit3"]
    out["leak_out_of_spec"] = out["leak_over_upper_limit"] | out["leak_under_lower_limit"]

    expected_leak = -out["Pressure Difference"] * out["Factor"] / 10000 - out["Offset"]
    out["leakrate_recomputed"] = expected_leak
    out["leakrate_calc_error"] = out["Leakrate"] - out["leakrate_recomputed"]
    out["leakrate_calc_error_abs"] = out["leakrate_calc_error"].abs()
    out["calc_mismatch"] = out["leakrate_calc_error_abs"] > 1e-4

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["DateTime"].dt.date.astype("string")
    out["hour"] = out["DateTime"].dt.hour
    out["weekday"] = out["DateTime"].dt.dayofweek
    out["shift"] = np.select(
        [out["hour"].between(6, 13), out["hour"].between(14, 21)],
        ["day", "middle"],
        default="night",
    )
    out["seconds_since_prev"] = out["DateTime"].diff().dt.total_seconds()
    out["long_gap_flag"] = out["seconds_since_prev"] > 600
    return out


def rolling_slope(values: pd.Series) -> float:
    y = values.to_numpy(dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)[valid]
    y = y[valid]
    return float(np.polyfit(x, y, 1)[0])


def add_leak_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    leak = out["Leakrate"]
    upper = out["Limit1"]
    lower = out["Limit3"]
    spec_width = upper - lower

    out["leak_margin_to_upper"] = upper - leak
    out["leak_margin_to_lower"] = leak - lower
    out["leak_nearest_margin"] = np.minimum(out["leak_margin_to_upper"], out["leak_margin_to_lower"])
    out["leak_spec_position"] = (leak - lower) / spec_width.replace(0, np.nan)
    out["near_upper_80pct"] = out["leak_spec_position"] >= 0.8
    out["near_upper_90pct"] = out["leak_spec_position"] >= 0.9

    for window in (5, 20, 50, 100):
        roll = leak.rolling(window=window, min_periods=max(3, window // 3))
        out[f"leak_roll_mean_{window}"] = roll.mean()
        out[f"leak_roll_std_{window}"] = roll.std()
        out[f"leak_roll_min_{window}"] = roll.min()
        out[f"leak_roll_max_{window}"] = roll.max()
        out[f"leak_roll_range_{window}"] = out[f"leak_roll_max_{window}"] - out[f"leak_roll_min_{window}"]
        out[f"leak_roll_slope_{window}"] = roll.apply(rolling_slope, raw=False)

    out["leak_diff_1"] = leak.diff()
    out["leak_diff_5"] = leak.diff(5)
    baseline_mean = leak.rolling(window=200, min_periods=50).mean()
    baseline_std = leak.rolling(window=200, min_periods=50).std()
    out["leak_rolling_zscore_200"] = (leak - baseline_mean) / baseline_std.replace(0, np.nan)
    out["leak_spike_flag"] = out["leak_rolling_zscore_200"].abs() >= 3

    out["pressure_diff_abs"] = out["Pressure Difference"].abs()
    out["pressure_diff_delta_1"] = out["Pressure Difference"].diff()
    out["pressure_measured_delta_1"] = out["Pressure Measured"].diff()
    out["pressure_fill_minus_measure"] = out["Pressure Filled"] - out["Pressure Measured"]

    # These are sequence-level shape proxies. The export does not contain each test's raw leak curve.
    out["trend_up_short"] = out["leak_roll_slope_20"] > 0
    out["trend_up_mid"] = out["leak_roll_slope_100"] > 0
    out["possible_inflection"] = np.sign(out["leak_roll_slope_20"]).diff().fillna(0).ne(0)

    return out


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def clean_and_engineer(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    raw = load_channel_export(input_path)
    df = coerce_types(raw)
    df = add_quality_flags(df)
    df = add_time_features(df)
    df = add_leak_features(df)

    before = len(df)
    cleaned = df[
        ~df["missing_required"]
        & ~df["param_mismatch_any"]
        & ~df["calc_mismatch"]
    ].copy()
    after = len(cleaned)

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "channel1_clean_features.csv"
    summary_path = output_dir / "channel1_cleaning_summary.txt"
    cleaned.to_csv(feature_path, index=False, encoding="utf-8-sig")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("channel1 cleaning and feature engineering summary\n")
        f.write(f"input_rows: {before}\n")
        f.write(f"output_rows: {after}\n")
        f.write(f"dropped_rows: {before - after}\n")
        f.write(f"date_min: {df['DateTime'].min()}\n")
        f.write(f"date_max: {df['DateTime'].max()}\n")
        f.write(f"leak_out_of_spec_rows: {int(df['leak_out_of_spec'].sum())}\n")
        f.write(f"pressure_out_of_range_rows: {int(df['pressure_out_of_range'].sum())}\n")
        f.write(f"param_mismatch_rows: {int(df['param_mismatch_any'].sum())}\n")
        f.write(f"calc_mismatch_rows: {int(df['calc_mismatch'].sum())}\n")
        f.write("\nGenerated feature groups:\n")
        f.write("- basic cleaned numeric fields\n")
        f.write("- parameter consistency flags\n")
        f.write("- leak limit margins and near-limit flags\n")
        f.write("- rolling mean/std/range/slope features\n")
        f.write("- pressure delta and pressure gap features\n")
        f.write("- time, shift, long-gap features\n")

    return feature_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean channel1.XLS and build analysis features.")
    parser.add_argument("--input", default="channel1.XLS", help="Path to channel1.XLS or converted .xlsx")
    parser.add_argument("--output-dir", default="outputs/channel1_features", help="Output directory")
    args = parser.parse_args()

    feature_path, summary_path = clean_and_engineer(Path(args.input), Path(args.output_dir))
    print(f"feature_csv={feature_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
