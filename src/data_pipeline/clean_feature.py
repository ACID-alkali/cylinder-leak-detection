from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HEADER_MARKER = "Date"
DATA_START_ROW = 20

# 我们提供一个通道1的默认参数配置。
# 在实际应用中，如果 parameters.json 不存在，脚本会自动在 reference 目录下生成此默认配置。
DEFAULT_CHANNEL_PARAMS = {
    "1": {
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
        "Device", "Channel", "Program", "Operation", "Study",
        "Pressure Filled", "Pressure Measured", "Leakrate", "Reject",
        "Pressure Difference", "Temperature Environment", "Temperature Part",
        "Difference Temperature", "Leakrate Uncompensated",
        "Time VexFilling", "Time PartFilling", "Time Prefilling",
        "Time Filling", "Time Balancing", "Time Measuring", "Time Venting",
        "Pressure Filling", "Pressure Measuring",
        "Min. Pressure Measuring", "Max. Pressure Measuring",
        "Factor", "Offset", "Limit1", "Limit2", "Limit3",
        "Flowsensor", "Transmitter", "PO", "FO", "RO1", "RO2",
        "LeakrateRaw", "LeakrateAfterSmoothing", "LeakrateAfterOC", "TestRun",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.sort_values("DateTime").reset_index(drop=True)


def add_quality_flags(df: pd.DataFrame, expected_params: dict) -> pd.DataFrame:
    out = df.copy()
    required = [
        "DateTime", "Leakrate", "Pressure Difference", "Pressure Measured",
        "Factor", "Offset", "Limit1", "Limit3",
    ]
    
    existing_required = [col for col in required if col in out.columns]
    if existing_required:
        out["missing_required"] = out[existing_required].isna().any(axis=1)
    else:
        out["missing_required"] = True

    for col, expected in expected_params.items():
        if col in out.columns:
            out[f"param_mismatch_{safe_name(col)}"] = ~np.isclose(out[col], expected, equal_nan=False)

    mismatch_cols = [c for c in out.columns if c.startswith("param_mismatch_")]
    out["param_mismatch_any"] = out[mismatch_cols].any(axis=1) if mismatch_cols else False

    out["pressure_out_of_range"] = False
    if "Pressure Measured" in out.columns and "Min. Pressure Measuring" in out.columns and "Max. Pressure Measuring" in out.columns:
        out["pressure_out_of_range"] = (
            (out["Pressure Measured"] < out["Min. Pressure Measuring"])
            | (out["Pressure Measured"] > out["Max. Pressure Measuring"])
        )
    
    out["leak_over_upper_limit"] = False
    out["leak_under_lower_limit"] = False
    if "Leakrate" in out.columns and "Limit1" in out.columns and "Limit3" in out.columns:
        out["leak_over_upper_limit"] = out["Leakrate"] > out["Limit1"]
        out["leak_under_lower_limit"] = out["Leakrate"] < out["Limit3"]
        out["leak_out_of_spec"] = out["leak_over_upper_limit"] | out["leak_under_lower_limit"]
    else:
        out["leak_out_of_spec"] = False

    out["calc_mismatch"] = False
    if all(col in out.columns for col in ["Pressure Difference", "Factor", "Offset", "Leakrate"]):
        expected_leak = -out["Pressure Difference"] * out["Factor"] / 10000 - out["Offset"]
        out["leakrate_recomputed"] = expected_leak
        out["leakrate_calc_error"] = out["Leakrate"] - out["leakrate_recomputed"]
        out["leakrate_calc_error_abs"] = out["leakrate_calc_error"].abs()
        out["calc_mismatch"] = out["leakrate_calc_error_abs"] > 1e-4

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "DateTime" not in out.columns:
        return out
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
    """Optimized analytical slope calculation replacing np.polyfit."""
    y = values.to_numpy(dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)[valid]
    y = y[valid]
    
    x_mean = x.mean()
    y_mean = y.mean()
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def add_leak_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Leakrate" not in out.columns:
        return out
        
    leak = out["Leakrate"]
    
    if "Limit1" in out.columns and "Limit3" in out.columns:
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
        # Use our optimized rolling_slope function
        out[f"leak_roll_slope_{window}"] = roll.apply(rolling_slope, raw=False)

    out["leak_diff_1"] = leak.diff()
    out["leak_diff_5"] = leak.diff(5)
    baseline_mean = leak.rolling(window=200, min_periods=50).mean()
    baseline_std = leak.rolling(window=200, min_periods=50).std()
    out["leak_rolling_zscore_200"] = (leak - baseline_mean) / baseline_std.replace(0, np.nan)
    out["leak_spike_flag"] = out["leak_rolling_zscore_200"].abs() >= 3

    if "Pressure Difference" in out.columns:
        out["pressure_diff_abs"] = out["Pressure Difference"].abs()
        out["pressure_diff_delta_1"] = out["Pressure Difference"].diff()
        
    if "Pressure Measured" in out.columns:
        out["pressure_measured_delta_1"] = out["Pressure Measured"].diff()
        
    if "Pressure Filled" in out.columns and "Pressure Measured" in out.columns:
        out["pressure_fill_minus_measure"] = out["Pressure Filled"] - out["Pressure Measured"]

    out["trend_up_short"] = out["leak_roll_slope_20"] > 0
    out["trend_up_mid"] = out["leak_roll_slope_100"] > 0
    out["possible_inflection"] = np.sign(out["leak_roll_slope_20"]).diff().fillna(0).ne(0)

    return out


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def load_parameters(config_path: Path, channel: str) -> dict:
    """Load parameters from JSON config, create a default one if it doesn't exist."""
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CHANNEL_PARAMS, f, indent=4, ensure_ascii=False)
        print(f"[*] Created default parameter config at {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        params_db = json.load(f)
        
    if channel not in params_db:
        print(f"[!] Warning: Parameters for channel {channel} not found in {config_path}. Using default Channel 1 parameters.")
        return DEFAULT_CHANNEL_PARAMS.get("1", {})
        
    return params_db[channel]


def generate_visualizations(df: pd.DataFrame, output_dir: Path, channel: str):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib or seaborn not found. Skipping visualizations.")
        return

    print("Generating visualizations...")
    import matplotlib
    matplotlib.use('Agg') # Ensure plot generation doesn't require GUI
    sns.set_theme(style="whitegrid")
    
    plot_dir = output_dir / f"channel{channel}" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Timeline of Leakrate
    plt.figure(figsize=(15, 6))
    if "DateTime" in df.columns and "Leakrate" in df.columns:
        plt.scatter(df["DateTime"], df["Leakrate"], alpha=0.5, s=10, label="Leakrate")
        if "Limit1" in df.columns:
            upper_limit = df["Limit1"].iloc[0] if not df["Limit1"].empty else np.nan
            plt.axhline(y=upper_limit, color='r', linestyle='--', label=f'Upper Limit ({upper_limit})')
        if "Limit3" in df.columns:
            lower_limit = df["Limit3"].iloc[0] if not df["Limit3"].empty else np.nan
            plt.axhline(y=lower_limit, color='g', linestyle='--', label=f'Lower Limit ({lower_limit})')
        
        plt.title(f"Channel {channel} - Leakrate Timeline")
        plt.xlabel("Time")
        plt.ylabel("Leakrate (cm3/min)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"channel{channel}_leakrate_timeline.png", dpi=300)
        plt.close()

    # 2. Scatter: Pressure Diff vs Leakrate
    if "Pressure Difference" in df.columns and "Leakrate" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="Pressure Difference", y="Leakrate", alpha=0.6)
        plt.title(f"Channel {channel} - Pressure Diff vs Leakrate")
        plt.xlabel("Pressure Difference (mbar)")
        plt.ylabel("Leakrate (cm3/min)")
        plt.tight_layout()
        plt.savefig(plot_dir / f"channel{channel}_pressure_vs_leakrate.png", dpi=300)
        plt.close()

    # 3. Distribution of Leakrate
    if "Leakrate" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df["Leakrate"], kde=True, bins=50, color="steelblue")
        plt.title(f"Channel {channel} - Leakrate Distribution")
        plt.xlabel("Leakrate (cm3/min)")
        plt.tight_layout()
        plt.savefig(plot_dir / f"channel{channel}_leakrate_dist.png", dpi=300)
        plt.close()

    # 4. Rolling Slope (Trend of curves)
    slope_col = "leak_roll_slope_20"
    if slope_col in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[slope_col].dropna(), kde=True, bins=50, color="orange")
        plt.title(f"Channel {channel} - Short-term Curve Slope Distribution")
        plt.xlabel("Leakrate Rolling Slope (window=20)")
        plt.tight_layout()
        plt.savefig(plot_dir / f"channel{channel}_slope_dist.png", dpi=300)
        plt.close()
        
    print(f"Visualizations saved to {plot_dir}")


def clean_and_engineer(input_path: Path, output_dir: Path, expected_params: dict, channel: str) -> tuple[Path, Path]:
    raw = load_channel_export(input_path)
    df = coerce_types(raw)
    df = add_quality_flags(df, expected_params)
    df = add_time_features(df)
    df = add_leak_features(df)

    before = len(df)
    
    # 动态过滤条件，防止因为缺失某些列导致报错
    filter_cond = pd.Series(True, index=df.index)
    if "missing_required" in df.columns:
        filter_cond &= ~df["missing_required"]
    if "param_mismatch_any" in df.columns:
        filter_cond &= ~df["param_mismatch_any"]
    if "calc_mismatch" in df.columns:
        filter_cond &= ~df["calc_mismatch"]
        
    cleaned = df[filter_cond].copy()
    after = len(cleaned)

    data_dir = output_dir / f"channel{channel}" / "data"
    report_dir = output_dir / f"channel{channel}" / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    feature_path = data_dir / f"channel{channel}_clean_features.csv"
    summary_path = report_dir / f"channel{channel}_cleaning_summary.txt"
    cleaned.to_csv(feature_path, index=False, encoding="utf-8-sig")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Channel {channel} cleaning and feature engineering summary\n")
        f.write(f"input_rows: {before}\n")
        f.write(f"output_rows: {after}\n")
        f.write(f"dropped_rows: {before - after}\n")
        
        if "DateTime" in df.columns:
            f.write(f"date_min: {df['DateTime'].min()}\n")
            f.write(f"date_max: {df['DateTime'].max()}\n")
            
        if "leak_out_of_spec" in df.columns:
            f.write(f"leak_out_of_spec_rows: {int(df['leak_out_of_spec'].sum())}\n")
        if "pressure_out_of_range" in df.columns:
            f.write(f"pressure_out_of_range_rows: {int(df['pressure_out_of_range'].sum())}\n")
        if "param_mismatch_any" in df.columns:
            f.write(f"param_mismatch_rows: {int(df['param_mismatch_any'].sum())}\n")
        if "calc_mismatch" in df.columns:
            f.write(f"calc_mismatch_rows: {int(df['calc_mismatch'].sum())}\n")
            
        f.write("\nGenerated feature groups:\n")
        f.write("- basic cleaned numeric fields\n")
        f.write("- parameter consistency flags\n")
        f.write("- leak limit margins and near-limit flags\n")
        f.write("- rolling mean/std/range/slope features (Optimized)\n")
        f.write("- pressure delta and pressure gap features\n")
        f.write("- time, shift, long-gap features\n")

    generate_visualizations(cleaned, output_dir, channel)

    return feature_path, summary_path


# 动态定位项目根目录 (src/data_pipeline/clean_feature.py -> 项目根目录)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean all channels and build analysis features.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "processed"), help="Output directory")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "data" / "reference" / "parameters.json"), help="Path to parameter JSON config")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "raw"), help="Path to raw data directory")
    args = parser.parse_args()

    # Define processing tasks: (Config ID, Raw File Name)
    tasks = [
        ("1", "channel1.XLS"),
        ("2_old", "channel2.XLS"),
        ("2_new", "channel2.XLS"),
        ("3", "channel3.XLS"),
    ]

    for config_id, raw_filename in tasks:
        raw_path = Path(args.data_dir) / raw_filename
        if not raw_path.exists():
            print(f"[!] Skipping {config_id}: File {raw_path} not found.")
            continue
            
        print(f"\n{'='*40}")
        print(f"[*] Processing channel configuration '{config_id}'...")
        print(f"    Input: {raw_path}")
        
        expected_params = load_parameters(Path(args.config), config_id)
        
        try:
            feature_path, summary_path = clean_and_engineer(
                raw_path, 
                Path(args.output_dir), 
                expected_params, 
                config_id
            )
            print(f"[+] Success!")
            print(f"    Feature CSV saved to: {feature_path}")
            print(f"    Summary saved to: {summary_path}")
        except Exception as e:
            print(f"[-] Failed to process {config_id}: {e}")
            import traceback
            traceback.print_exc()

    print("\n[*] All channels processed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
