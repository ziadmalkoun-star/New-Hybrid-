import io
import time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.dates as mdates

HOURS_PER_YEAR = 8760
QH_PER_HOUR = 4
QH_PER_YEAR = 35040
QH_DT_HOURS = 0.25
DEFAULT_YEAR = 2025
PV_ZERO_TOLERANCE_MWH = 1e-6

# Only the standard solar curve and the optional curtailment curve are kept in the script.
# All market and aFRR price datasets must be uploaded by the user in the app.
APP_DIR = Path(__file__).resolve().parent
BUILTIN_CURTAILMENT_CURVE = APP_DIR / "Curtailment_Curve.xlsx"

# Embedded fallback data. The app first looks for files next to the script; if absent, it uses these bundled bytes.

# 15-minute 2025 Spain datasets embedded as fallback bytes.

def _open_builtin_file(path: Path, label: str):
    """Open the optional external curtailment file placed next to this script."""
    if not path.exists():
        raise FileNotFoundError(f"Required file '{path.name}' for {label} was not found next to the script. Please upload it instead.")
    return path.open("rb")



@dataclass
class SimulationInputs:
    batt_power_mw: float
    batt_energy_mwh: float
    pv_dc_mw: float
    productible_kwh_per_kwp: float
    pv_losses_pct: float
    plant_availability_pct: float
    eta_charge: float
    eta_discharge: float

    # Effective prices used by the optimizer/economics
    pv_price: np.ndarray
    batt_sell_price: np.ndarray
    grid_buy_price: np.ndarray

    # PV available for direct sale / standard PV-to-battery charging
    solar_profile: np.ndarray

    # PV curtailed but optionally recoverable into battery only
    curtailed_pv_recoverable_mwh: np.ndarray | None = None

    # BESS availability reporting: batt_energy_mwh is the effective available capacity used by the model.
    nominal_batt_energy_mwh: float = 0.0
    bess_availability_pct: float = 100.0

    nightly_bess_revenue_eur: float = 0.0
    soc_steps: int = 101
    initial_soc_mwh: float = 0.0
    final_soc_mwh: float = 0.0
    min_soc_pct: float = 0.0
    max_soc_pct: float = 100.0
    grid_export_limit_mw: float = 0.0
    cycle_cost_eur_per_mwh: float = 0.0
    charge_quantile: float = 100.0
    discharge_quantile: float = 0.0
    max_cycles_per_year: float = 1.0
    min_spread_arbitrage_eur_per_mwh: float = 0.0
    # Forward-looking cross-market optimization controls
    forward_optimization_horizon_hours: float = 24.0
    afrr_up_cross_market_min_spread_eur_per_mwh: float = 20.0
    afrr_down_to_wholesale_min_spread_eur_per_mwh: float = 20.0

    # Capture rates
    pv_capture_rate_pct: float = 100.0
    bess_capture_rate_pct: float = 100.0

    # aFRR inputs
    enable_afrr: bool = False
    afrr_charge_price_qh: np.ndarray | None = None
    afrr_discharge_price_qh: np.ndarray | None = None
    afrr_min_spread_eur_per_mwh: float = 0.0
    afrr_cycle_cost_eur_per_mwh: float = 0.0
    afrr_max_events_per_day: int = 1
    afrr_night_start_hour: int = 20
    afrr_night_end_hour: int = 8
    afrr_pv_zero_tolerance_mwh: float = PV_ZERO_TOLERANCE_MWH
    afrr_n_qh_per_side: int = 4
    afrr_energy_down_activation_pct: float = 100.0
    afrr_energy_up_activation_pct: float = 100.0

    # aFRR Capacity inputs
    enable_afrr_capacity: bool = False
    afrr_capacity_up_price_h: np.ndarray | None = None
    afrr_capacity_down_price_h: np.ndarray | None = None
    afrr_certified_capacity_pct: float = 100.0
    afrr_capacity_success_rate_pct: float = 80.0
    afrr_capacity_start_hour: int = 0
    afrr_capacity_end_hour: int = 0
    allow_afrr_energy_without_capacity: bool = True
    afrr_certified_capacity_up_mw: float = 0.0
    afrr_certified_capacity_down_mw: float = 0.0
    # Internal quarter-hour market selection used to block wholesale and gate aFRR energy.
    afrr_capacity_selected_market_h: np.ndarray | None = None
    # Expected activated energy arrays from Phase-1 aFRR capacity selection.
    # These are used to keep physical aFRR energy dispatch aligned with the
    # expected MWh used in the capacity value comparison.
    afrr_expected_up_activated_mwh_qh: np.ndarray | None = None
    afrr_expected_down_activated_mwh_qh: np.ndarray | None = None

    # Curtailment
    enable_tso_dso_curtailment: bool = False
    tso_dso_monthly_curtailment_pct: np.ndarray | None = None
    enable_self_curtailment: bool = False
    curtailment_threshold_eur_per_mwh: float = -1.0
    pv_commercial_structure: str = "Fully merchant"  # Fully merchant / With CfD / With PPA
    cfd_price_eur_per_mwh: float = 0.0
    negative_price_rule: bool = False
    consecutive_negative_hours_limit: int = 6
    ppa_price_eur_per_mwh: float = 0.0
    charge_battery_if_curtailment: bool = False
    enable_cfd: bool = False
    cfd_price_standalone_eur_per_mwh: float = 0.0
    enable_ppa: bool = False
    ppa_price_standalone_eur_per_mwh: float = 0.0
    project_lifetime_years: int = 1
    bess_degradation_curve_pct: np.ndarray | None = None
    degraded_bess_energy_by_year_mwh: np.ndarray | None = None

def _validate_array_length(arr: np.ndarray, name: str, expected_len: int = QH_PER_YEAR) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if len(arr) != expected_len:
        raise ValueError(f"{name} doit contenir exactement {expected_len} valeurs. Reçu: {len(arr)}.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contient des valeurs non numériques ou infinies.")
    return arr


def rolling_forward_max(values: np.ndarray, horizon_steps: int) -> np.ndarray:
    """Maximum future value within (t, t + horizon_steps]."""
    values = np.asarray(values, dtype=float).reshape(-1)
    out = np.full(len(values), -1e30, dtype=float)
    h = int(max(1, horizon_steps))
    for t in range(len(values)):
        end = min(len(values), t + h + 1)
        if t + 1 < end:
            out[t] = float(np.nanmax(values[t + 1:end]))
    return out


def compute_forward_cross_market_value_curves(inputs: SimulationInputs) -> Dict[str, np.ndarray]:
    """Build practical forward-looking value curves for cross-market charging.

    This is intentionally lightweight and compatible with the existing DP.
    It does not replace the full dispatch optimizer with LP/MILP; it provides
    forward opportunity signals used as charging gates and audit columns.
    """
    horizon_steps = int(max(1, round(float(inputs.forward_optimization_horizon_hours) * QH_PER_HOUR)))

    wholesale_value = _validate_array_length(inputs.batt_sell_price, "BESS sell price", QH_PER_YEAR) * (float(inputs.bess_capture_rate_pct) / 100.0)

    if inputs.enable_afrr and inputs.afrr_discharge_price_qh is not None:
        afrr_up_energy = _validate_array_length(inputs.afrr_discharge_price_qh, "aFRR UP energy price", QH_PER_YEAR)
    else:
        afrr_up_energy = np.full(QH_PER_YEAR, -1e30, dtype=float)

    if inputs.enable_afrr_capacity and inputs.afrr_capacity_up_price_h is not None:
        cap_up = _validate_array_length(inputs.afrr_capacity_up_price_h, "aFRR UP capacity price", QH_PER_YEAR)
        success = min(max(float(inputs.afrr_capacity_success_rate_pct) / 100.0, 0.0), 1.0)
        activation_up = min(max(float(inputs.afrr_energy_up_activation_pct) / 100.0, 0.0), 1.0)
        if activation_up > 1e-9:
            # Convert capacity availability value into an expected EUR/MWh uplift
            # on activated UP energy. This is an expected-value signal only.
            cap_uplift_per_mwh = cap_up * success / activation_up
        else:
            cap_uplift_per_mwh = np.zeros(QH_PER_YEAR, dtype=float)
        afrr_up_value = afrr_up_energy + cap_uplift_per_mwh
    else:
        afrr_up_value = afrr_up_energy.copy()

    future_wholesale = rolling_forward_max(wholesale_value, horizon_steps)
    future_afrr_up = rolling_forward_max(afrr_up_value, horizon_steps)
    future_best = np.maximum(future_wholesale, future_afrr_up)
    future_type = np.where(future_afrr_up > future_wholesale, "afrr_up", "wholesale")
    future_type[future_best <= -1e20] = "none"

    return {
        "future_expected_wholesale_value_eur_per_mwh": future_wholesale,
        "future_expected_afrr_up_value_eur_per_mwh": future_afrr_up,
        "future_best_market_value_eur_per_mwh": future_best,
        "future_best_market_type": future_type.astype(object),
        "forward_horizon_hours": np.full(QH_PER_YEAR, float(inputs.forward_optimization_horizon_hours), dtype=float),
    }

def build_combined_soc_with_afrr(
    result_hourly: Dict[str, np.ndarray],
    afrr_result: Dict[str, np.ndarray] | None,
    batt_energy_mwh: float,
    initial_soc_mwh: float,
    eta_charge: float,
    eta_discharge: float,
    min_soc_pct: float = 0.0,
    max_soc_pct: float = 100.0,
) -> Dict[str, np.ndarray]:

    # The wholesale dispatch is already calculated at 15-minute resolution.
    # Values are MWh per 15-minute step, so do NOT repeat or divide them again.
    wholesale_pv_to_batt_qh = _validate_array_length(
        result_hourly["pv_to_batt"], "PV vers batterie wholesale 15 min", QH_PER_YEAR
    )
    wholesale_pv_curtailed_to_batt_qh = _validate_array_length(
        result_hourly.get("pv_curtailed_to_battery", np.zeros(QH_PER_YEAR)),
        "PV curtailed vers batterie wholesale 15 min",
        QH_PER_YEAR,
    )
    wholesale_grid_charge_qh = _validate_array_length(
        result_hourly["grid_charge"], "Charge réseau wholesale 15 min", QH_PER_YEAR
    )
    wholesale_discharge_market_qh = _validate_array_length(
        result_hourly["discharge"], "Décharge wholesale 15 min", QH_PER_YEAR
    )

    if afrr_result is not None:
        afrr_charge_market_qh = np.asarray(afrr_result["afrr_charge_qh_mwh"], dtype=float)
        afrr_discharge_market_qh = np.asarray(afrr_result["afrr_discharge_qh_mwh"], dtype=float)
    else:
        afrr_charge_market_qh = np.zeros(QH_PER_YEAR, dtype=float)
        afrr_discharge_market_qh = np.zeros(QH_PER_YEAR, dtype=float)

    # Convert to SOC flows
    wholesale_charge_to_soc_qh = (
        wholesale_pv_to_batt_qh
        + wholesale_pv_curtailed_to_batt_qh
        + wholesale_grid_charge_qh
    ) * eta_charge
    wholesale_discharge_from_soc_qh = wholesale_discharge_market_qh / max(eta_discharge, 1e-12)

    afrr_charge_to_soc_qh = afrr_charge_market_qh * eta_charge
    afrr_discharge_from_soc_qh = afrr_discharge_market_qh / max(eta_discharge, 1e-12)

    combined_charge_to_soc_qh = wholesale_charge_to_soc_qh + afrr_charge_to_soc_qh
    combined_discharge_from_soc_qh = wholesale_discharge_from_soc_qh + afrr_discharge_from_soc_qh

    # SOC simulation
    min_soc_mwh = batt_energy_mwh * min_soc_pct / 100.0
    max_soc_mwh = batt_energy_mwh * max_soc_pct / 100.0

    soc_qh = np.zeros(QH_PER_YEAR + 1, dtype=float)
    soc_qh[0] = min(max(float(initial_soc_mwh), min_soc_mwh), max_soc_mwh)

    for t in range(QH_PER_YEAR):
        soc_next = soc_qh[t] + combined_charge_to_soc_qh[t] - combined_discharge_from_soc_qh[t]
        soc_qh[t + 1] = min(max(soc_next, min_soc_mwh), max_soc_mwh)

    soc_hourly_end = soc_qh[4::4]

    return {
        "combined_soc_qh": soc_qh,
        "combined_soc_hourly_end": soc_hourly_end,
        "combined_charge_to_soc_qh": combined_charge_to_soc_qh,
        "combined_discharge_from_soc_qh": combined_discharge_from_soc_qh,
        "wholesale_charge_to_soc_qh": wholesale_charge_to_soc_qh,
        "wholesale_pv_curtailed_to_batt_qh": wholesale_pv_curtailed_to_batt_qh,
        "wholesale_discharge_from_soc_qh": wholesale_discharge_from_soc_qh,
        "afrr_charge_to_soc_qh": afrr_charge_to_soc_qh,
        "afrr_discharge_from_soc_qh": afrr_discharge_from_soc_qh,
        "afrr_charge_market_qh": afrr_charge_market_qh,
        "afrr_discharge_market_qh": afrr_discharge_market_qh,
    }

def _read_single_column_csv(uploaded_file, expected_len: int = QH_PER_YEAR) -> np.ndarray:
    if uploaded_file is None:
        raise ValueError("Aucun fichier CSV fourni.")

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    filename = str(getattr(uploaded_file, "name", "")).lower()
    if filename.endswith((".xlsx", ".xls")):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        df = pd.read_excel(uploaded_file, header=None)
        numeric_cols = []
        for col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) >= expected_len:
                numeric_cols.append(values)
        if not numeric_cols:
            raise ValueError(f"Le fichier Excel doit contenir exactement {expected_len} valeurs numériques dans une colonne. Reçu: aucune colonne exploitable.")
        arr = np.asarray(numeric_cols[-1][:expected_len], dtype=float)
        if len(arr) != expected_len:
            raise ValueError(f"Le fichier Excel doit contenir exactement {expected_len} valeurs numériques. Reçu: {len(arr)}.")
        if np.any(~np.isfinite(arr)):
            raise ValueError("Le fichier Excel contient des valeurs non finies.")
        return arr

    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8-sig", errors="replace")
    else:
        text = str(raw)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Le CSV est vide.")

    values = []
    bad_rows = []

    for i, line in enumerate(lines):
        cleaned = line.strip().strip('"').strip("'").replace(",", ".")
        try:
            values.append(float(cleaned))
        except ValueError:
            bad_rows.append(i)

    if len(bad_rows) == 1 and bad_rows[0] == 0 and len(values) == expected_len:
        return np.asarray(values, dtype=float)

    if bad_rows:
        raise ValueError(
            f"Le CSV contient des valeurs non numériques dans la première colonne. "
            f"Lignes problématiques: {bad_rows[:10]}"
        )

    if len(values) != expected_len:
        raise ValueError(
            f"Le CSV doit contenir exactement {expected_len} lignes numériques. "
            f"Reçu: {len(values)}."
        )

    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Le CSV contient des valeurs non finies.")
    return arr


def _read_single_column_csv_qh(uploaded_file, expected_len: int = QH_PER_YEAR) -> np.ndarray:
    return _read_single_column_csv(uploaded_file, expected_len=expected_len)


def read_monthly_curtailment_excel(uploaded_file) -> np.ndarray:
    if uploaded_file is None:
        raise ValueError("Aucun fichier Excel de courbe de curtailment fourni.")

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    df = pd.read_excel(uploaded_file, header=None)
    values = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)

    if len(values) != 12:
        raise ValueError(f"La courbe de curtailment mensuelle doit contenir exactement 12 valeurs. Reçu: {len(values)}.")

    return values

def read_bess_degradation_excel(uploaded_file, project_lifetime_years: int, initial_bess_mwh: float) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if uploaded_file is None:
        degradation_pct = np.full(project_lifetime_years, 100.0, dtype=float)
    else:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        df = pd.read_excel(uploaded_file, header=None)
        degradation_pct = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)

        if len(degradation_pct) < project_lifetime_years:
            raise ValueError(
                f"La courbe de dégradation BESS doit contenir au moins {project_lifetime_years} valeurs. "
                f"Reçu: {len(degradation_pct)}."
            )

        degradation_pct = degradation_pct[:project_lifetime_years]

    if len(degradation_pct) == 0:
        raise ValueError("La courbe de dégradation BESS est vide.")

    if degradation_pct[0] <= 1.5:
        degradation_pct = degradation_pct * 100.0

    degraded_mwh = np.zeros(project_lifetime_years, dtype=float)
    degraded_mwh[0] = float(initial_bess_mwh) * degradation_pct[0] / 100.0
    
    for y in range(1, project_lifetime_years):
        degraded_mwh[y] = degraded_mwh[y - 1] * degradation_pct[y] / 100.0

    degradation_df = pd.DataFrame({
        "Year": np.arange(1, project_lifetime_years + 1),
        "Degradation_pct": degradation_pct,
        "BESS_energy_mwh": degraded_mwh,
    })

    return degradation_pct, degraded_mwh, degradation_df
    

def _make_flat_curve(value: float, expected_len: int = QH_PER_YEAR) -> np.ndarray:
    if value is None:
        raise ValueError("La valeur moyenne annuelle n'a pas été renseignée.")
    return np.full(expected_len, float(value), dtype=float)


def build_quarter_hour_index(year: int = DEFAULT_YEAR) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01 00:00:00", periods=QH_PER_YEAR, freq="15min")


def repeat_hourly_to_qh(hourly_arr: np.ndarray) -> np.ndarray:
    hourly_arr = np.asarray(hourly_arr, dtype=float).reshape(-1)
    if len(hourly_arr) != HOURS_PER_YEAR:
        raise ValueError(f"La série horaire doit contenir {HOURS_PER_YEAR} valeurs.")
    return np.repeat(hourly_arr, QH_PER_HOUR)


def build_night_mask_qh(idx_qh: pd.DatetimeIndex, night_start_hour: int, night_end_hour: int) -> np.ndarray:
    hours = idx_qh.hour.to_numpy()

    if night_start_hour == night_end_hour:
        return np.ones(len(idx_qh), dtype=bool)
    if night_start_hour > night_end_hour:
        return (hours >= night_start_hour) | (hours < night_end_hour)
    return (hours >= night_start_hour) & (hours < night_end_hour)


def build_hour_mask(idx_hourly: pd.DatetimeIndex, start_hour: int, end_hour: int) -> np.ndarray:
    """Hourly eligibility window with the same midnight-crossing logic as aFRR energy."""
    hours = idx_hourly.hour.to_numpy()

    if start_hour == end_hour:
        return np.ones(len(idx_hourly), dtype=bool)
    if start_hour > end_hour:
        return (hours >= start_hour) | (hours < end_hour)
    return (hours >= start_hour) & (hours < end_hour)


def read_afrr_capacity_excel(uploaded_file, year: int) -> np.ndarray:
    """Read aFRR Capacity Excel prices for a selected forecast year.

    Expected format:
    - A1 empty
    - A2:A8761 = 0..8759
    - B1 onward = forecast years
    - selected year column = 35040 quarter-hour prices in EUR/MW/h
    """
    if uploaded_file is None:
        raise ValueError("Aucun fichier Excel aFRR Capacity fourni.")

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    df = pd.read_excel(uploaded_file, header=None)
    if df.shape[0] < HOURS_PER_YEAR + 1 or df.shape[1] < 2:
        raise ValueError("Le fichier aFRR Capacity doit contenir A2:A8761 et au moins une colonne année.")

    hours = pd.to_numeric(df.iloc[1:HOURS_PER_YEAR + 1, 0], errors="coerce").to_numpy(dtype=float)
    expected_hours = np.arange(HOURS_PER_YEAR, dtype=float)
    if len(hours) != HOURS_PER_YEAR or np.any(~np.isfinite(hours)) or not np.array_equal(hours.astype(int), expected_hours.astype(int)):
        raise ValueError("La colonne A du fichier aFRR Capacity doit contenir exactement les heures 0 à 8759.")

    raw_years = df.iloc[0, 1:].to_numpy()
    normalized_years = []
    for y in raw_years:
        try:
            normalized_years.append(int(float(y)))
        except Exception:
            normalized_years.append(None)

    if int(year) not in normalized_years:
        raise ValueError(f"Le fichier aFRR Capacity ne contient pas l'année {year}.")

    col_pos = normalized_years.index(int(year)) + 1
    values = pd.to_numeric(df.iloc[1:HOURS_PER_YEAR + 1, col_pos], errors="coerce").to_numpy(dtype=float)

    if len(values) != HOURS_PER_YEAR or np.any(~np.isfinite(values)):
        raise ValueError(f"La colonne {year} du fichier aFRR Capacity doit contenir exactement 35040 valeurs numériques.")

    return values

def read_afrr_capacity_csv(uploaded_file, year: int) -> np.ndarray:
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    df = pd.read_csv(
        uploaded_file,
        header=None,
        sep=";",
        decimal=",",
        engine="python"
    )

    # Remove fully empty columns caused by trailing semicolons
    df = df.dropna(axis=1, how="all")

    if df.shape[0] < HOURS_PER_YEAR + 1 or df.shape[1] < 2:
        raise ValueError("Le CSV aFRR Capacity doit contenir A2:A8761 et au moins une colonne année.")

    hours = pd.to_numeric(df.iloc[1:HOURS_PER_YEAR + 1, 0], errors="coerce").to_numpy()
    expected_hours = np.arange(HOURS_PER_YEAR)

    if len(hours) != HOURS_PER_YEAR or np.any(~np.isfinite(hours)) or not np.array_equal(hours.astype(int), expected_hours):
        raise ValueError("La colonne A doit contenir les heures 0 à 8759.")

    raw_years = df.iloc[0, 1:].to_numpy()
    normalized_years = []

    for y in raw_years:
        try:
            normalized_years.append(int(float(y)))
        except Exception:
            normalized_years.append(None)

    if int(year) not in normalized_years:
        raise ValueError(f"Le CSV ne contient pas l'année {year}.")

    col_pos = normalized_years.index(int(year)) + 1

    values = pd.to_numeric(
        df.iloc[1:HOURS_PER_YEAR + 1, col_pos],
        errors="coerce"
    ).to_numpy(dtype=float)

    if len(values) != HOURS_PER_YEAR or np.any(~np.isfinite(values)):
        raise ValueError(f"La colonne {year} doit contenir exactement 35040 valeurs numériques.")

    return values
    
def simulate_afrr_capacity(
    inputs: SimulationInputs,
    wholesale_reference_result: Dict[str, np.ndarray] | None = None,
) -> Dict[str, np.ndarray]:
    """Phase 1 aFRR capacity co-optimization at 15-minute resolution.

    Exclusive market selection: for each 15-minute timestep the battery chooses
    either wholesale, aFRR UP capacity + expected UP energy, aFRR DOWN capacity +
    expected DOWN energy, or no battery market action.

    This version is forward-SOC-aware: aFRR UP/DOWN capacity is selected only if
    a sequential SOC trajectory can deliver/absorb the expected activation MWh
    after accounting for previously selected aFRR capacity actions.
    """
    zero_f = np.zeros(QH_PER_YEAR, dtype=float)
    zero_i = np.zeros(QH_PER_YEAR, dtype=int)
    none_o = np.full(QH_PER_YEAR, "none", dtype=object)

    audit_zero_bool = np.zeros(QH_PER_YEAR, dtype=int)
    base_return = {
        "afrr_capacity_up_awarded_h": zero_i.copy(),
        "afrr_capacity_down_awarded_h": zero_i.copy(),
        "afrr_capacity_selected_market_h": none_o.copy(),
        "afrr_capacity_up_revenue_h_eur": zero_f.copy(),
        "afrr_capacity_down_revenue_h_eur": zero_f.copy(),
        "afrr_capacity_total_revenue_h_eur": zero_f.copy(),
        "afrr_capacity_eligible_h": zero_i.copy(),
        "afrr_certified_capacity_up_mw_h": zero_f.copy(),
        "afrr_certified_capacity_down_mw_h": zero_f.copy(),
        "wholesale_opportunity_value_eur": zero_f.copy(),
        "wholesale_expected_value_after_capture_rate_eur": zero_f.copy(),
        "raw_up_capacity_revenue_eur": zero_f.copy(),
        "expected_up_capacity_revenue_eur": zero_f.copy(),
        "raw_down_capacity_revenue_eur": zero_f.copy(),
        "expected_down_capacity_revenue_eur": zero_f.copy(),
        "expected_up_activated_mwh": zero_f.copy(),
        "expected_down_activated_mwh": zero_f.copy(),
        "afrr_up_energy_expected_value_eur": zero_f.copy(),
        "afrr_down_energy_expected_value_eur": zero_f.copy(),
        "afrr_up_total_expected_value_eur": zero_f.copy(),
        "afrr_down_total_expected_value_eur": zero_f.copy(),
        "selected_market": none_o.copy(),
        "selected_capacity_direction": none_o.copy(),
        "afrr_capacity_success_rate_pct": np.full(QH_PER_YEAR, float(inputs.afrr_capacity_success_rate_pct), dtype=float),
        "bess_wholesale_capture_rate_pct": np.full(QH_PER_YEAR, float(inputs.bess_capture_rate_pct), dtype=float),
        "afrr_up_activation_pct": np.full(QH_PER_YEAR, float(inputs.afrr_energy_up_activation_pct), dtype=float),
        "afrr_down_activation_pct": np.full(QH_PER_YEAR, float(inputs.afrr_energy_down_activation_pct), dtype=float),
        "available_export_headroom_mwh": zero_f.copy(),
        "available_soc_headroom_mwh": zero_f.copy(),
        "available_discharge_from_soc_mwh": zero_f.copy(),
        "required_up_soc_reserve_mwh": zero_f.copy(),
        "required_down_soc_headroom_mwh": zero_f.copy(),
        "expected_degradation_cost_eur": zero_f.copy(),
        "future_best_market_value_eur_per_mwh": zero_f.copy(),
        "future_best_market_type": none_o.copy(),
        "cross_market_spread_eur_per_mwh": zero_f.copy(),
        "required_min_spread_eur_per_mwh": zero_f.copy(),
        "spread_condition_respected": audit_zero_bool.copy(),
        "charge_reason": none_o.copy(),
        "discharge_reason": none_o.copy(),
        "stored_energy_cost_eur_per_mwh": zero_f.copy(),
        "effective_discharge_value_eur_per_mwh": zero_f.copy(),
        "future_expected_afrr_up_value_eur": zero_f.copy(),
        "future_expected_wholesale_value_eur": zero_f.copy(),
        "future_expected_best_discharge_market": none_o.copy(),
        "wholesale_charge_for_future_afrr_flag": audit_zero_bool.copy(),
        "afrr_down_charge_for_future_wholesale_flag": audit_zero_bool.copy(),
        "afrr_down_charge_for_future_afrr_up_flag": audit_zero_bool.copy(),
        "wholesale_discharge_spread_ok": audit_zero_bool.copy(),
        "afrr_up_discharge_spread_ok": audit_zero_bool.copy(),
        "forward_horizon_hours": zero_f.copy(),
        "future_opportunity_selected": audit_zero_bool.copy(),
        "forward_soc_before_capacity_selection_mwh": zero_f.copy(),
        "forward_soc_after_capacity_selection_mwh": zero_f.copy(),
        "afrr_up_soc_feasible": audit_zero_bool.copy(),
        "afrr_down_soc_feasible": audit_zero_bool.copy(),
        "afrr_up_rejected_due_to_soc": audit_zero_bool.copy(),
        "afrr_down_rejected_due_to_soc": audit_zero_bool.copy(),
        "afrr_up_expected_vs_actual_shortfall_mwh": zero_f.copy(),
        "afrr_down_expected_vs_actual_shortfall_mwh": zero_f.copy(),
        "afrr_up_rejected_due_to_final_combined_soc": audit_zero_bool.copy(),
        "afrr_down_rejected_due_to_final_combined_soc": audit_zero_bool.copy(),
    }

    if not inputs.enable_afrr_capacity:
        return base_return
    if inputs.afrr_capacity_up_price_h is None or inputs.afrr_capacity_down_price_h is None:
        raise ValueError("Les deux courbes aFRR Capacity UP et Down doivent être fournies.")

    up_price = _validate_array_length(inputs.afrr_capacity_up_price_h, "Prix aFRR Capacity UP", QH_PER_YEAR)
    down_price = _validate_array_length(inputs.afrr_capacity_down_price_h, "Prix aFRR Capacity Down", QH_PER_YEAR)

    if inputs.afrr_charge_price_qh is None:
        afrr_down_energy_price = np.zeros(QH_PER_YEAR, dtype=float)
    else:
        afrr_down_energy_price = _validate_array_length(inputs.afrr_charge_price_qh, "Prix aFRR Down Energy", QH_PER_YEAR)
    if inputs.afrr_discharge_price_qh is None:
        afrr_up_energy_price = np.zeros(QH_PER_YEAR, dtype=float)
    else:
        afrr_up_energy_price = _validate_array_length(inputs.afrr_discharge_price_qh, "Prix aFRR Up Energy", QH_PER_YEAR)

    if not (0.0 <= inputs.afrr_certified_capacity_pct <= 100.0):
        raise ValueError("% of Certified Capacity for aFRR doit être compris entre 0 et 100 %.")
    if not (0.0 <= inputs.afrr_capacity_success_rate_pct <= 100.0):
        raise ValueError("aFRR Capacity Bid Success Rate (%) doit être compris entre 0 et 100 %.")

    success = float(inputs.afrr_capacity_success_rate_pct) / 100.0
    activation_up = min(max(float(inputs.afrr_energy_up_activation_pct) / 100.0, 0.0), 1.0)
    activation_down = min(max(float(inputs.afrr_energy_down_activation_pct) / 100.0, 0.0), 1.0)

    certified_up = float(inputs.afrr_certified_capacity_up_mw)
    certified_down = float(inputs.afrr_certified_capacity_down_mw)

    pv_direct = np.zeros(QH_PER_YEAR, dtype=float)
    wholesale_opportunity = np.zeros(QH_PER_YEAR, dtype=float)
    baseline_soc = np.full(QH_PER_YEAR + 1, float(inputs.initial_soc_mwh), dtype=float)
    baseline_soc_delta = np.zeros(QH_PER_YEAR, dtype=float)

    if wholesale_reference_result is not None:
        pv_direct = _validate_array_length(wholesale_reference_result.get("pv_direct", pv_direct), "PV direct référence wholesale", QH_PER_YEAR)
        soc_curve = np.asarray(wholesale_reference_result.get("soc", baseline_soc), dtype=float).reshape(-1)
        if len(soc_curve) >= QH_PER_YEAR + 1:
            baseline_soc = soc_curve[:QH_PER_YEAR + 1].astype(float)
            baseline_soc_delta = np.diff(baseline_soc)
        batt_sale = np.asarray(wholesale_reference_result.get("batt_sale_revenue", zero_f), dtype=float).reshape(-1)
        grid_cost = np.asarray(wholesale_reference_result.get("grid_charge_cost", zero_f), dtype=float).reshape(-1)
        pv_to_batt = np.asarray(wholesale_reference_result.get("pv_to_batt", zero_f), dtype=float).reshape(-1)
        curtailed_to_batt = np.asarray(wholesale_reference_result.get("pv_curtailed_to_battery", zero_f), dtype=float).reshape(-1)
        discharge = np.asarray(wholesale_reference_result.get("discharge", zero_f), dtype=float).reshape(-1)
        future_best_sell = np.maximum.accumulate(np.asarray(inputs.batt_sell_price, dtype=float)[::-1])[::-1]
        charge_future_value = (pv_to_batt + curtailed_to_batt) * np.maximum(future_best_sell * inputs.eta_charge * inputs.eta_discharge - inputs.pv_price, 0.0)
        discharge_value = np.maximum(batt_sale - discharge * inputs.cycle_cost_eur_per_mwh, 0.0)
        grid_charge_value = np.maximum((future_best_sell * inputs.eta_charge * inputs.eta_discharge - inputs.grid_buy_price) * np.asarray(wholesale_reference_result.get("grid_charge", zero_f), dtype=float), 0.0)
        wholesale_opportunity = np.maximum.reduce([discharge_value, charge_future_value, grid_charge_value, np.zeros(QH_PER_YEAR)])

    min_soc_mwh = inputs.batt_energy_mwh * inputs.min_soc_pct / 100.0
    max_soc_mwh = inputs.batt_energy_mwh * inputs.max_soc_pct / 100.0
    export_limit_qh = inputs.grid_export_limit_mw * QH_DT_HOURS
    available_export_headroom = np.maximum(export_limit_qh - pv_direct, 0.0)

    required_up_soc_reserve_mwh = certified_up * QH_DT_HOURS * activation_up
    required_down_soc_headroom_mwh = certified_down * QH_DT_HOURS * activation_down

    raw_up_capacity = up_price * certified_up * QH_DT_HOURS
    raw_down_capacity = down_price * certified_down * QH_DT_HOURS
    expected_up_capacity = raw_up_capacity * success
    expected_down_capacity = raw_down_capacity * success
    wholesale_expected = np.maximum(wholesale_opportunity, 0.0)

    forward_curves = compute_forward_cross_market_value_curves(inputs)
    future_best_value = forward_curves["future_best_market_value_eur_per_mwh"]
    future_best_type = forward_curves["future_best_market_type"]
    future_wholesale_value = forward_curves["future_expected_wholesale_value_eur_per_mwh"]
    future_afrr_up_value = forward_curves["future_expected_afrr_up_value_eur_per_mwh"]

    stored_energy_cost_qh = np.zeros(QH_PER_YEAR, dtype=float)
    if wholesale_reference_result is not None and "avg_stored_charge_price" in wholesale_reference_result:
        avg_cost = np.asarray(wholesale_reference_result["avg_stored_charge_price"], dtype=float).reshape(-1)
        if len(avg_cost) >= QH_PER_YEAR:
            stored_energy_cost_qh = np.nan_to_num(avg_cost[:QH_PER_YEAR], nan=0.0, posinf=0.0, neginf=0.0)

    selected = np.full(QH_PER_YEAR, "none", dtype=object)
    selected_market = np.full(QH_PER_YEAR, "none", dtype=object)
    up_awarded = np.zeros(QH_PER_YEAR, dtype=int)
    down_awarded = np.zeros(QH_PER_YEAR, dtype=int)
    up_revenue = np.zeros(QH_PER_YEAR, dtype=float)
    down_revenue = np.zeros(QH_PER_YEAR, dtype=float)
    expected_up_activated = np.zeros(QH_PER_YEAR, dtype=float)
    expected_down_activated = np.zeros(QH_PER_YEAR, dtype=float)
    up_energy_value_selected = np.zeros(QH_PER_YEAR, dtype=float)
    down_energy_value_selected = np.zeros(QH_PER_YEAR, dtype=float)
    up_total_value_audit = np.zeros(QH_PER_YEAR, dtype=float)
    down_total_value_audit = np.zeros(QH_PER_YEAR, dtype=float)
    expected_degradation_selected = np.zeros(QH_PER_YEAR, dtype=float)
    cross_market_spread = np.zeros(QH_PER_YEAR, dtype=float)
    required_min_spread = np.zeros(QH_PER_YEAR, dtype=float)
    spread_condition_respected = np.zeros(QH_PER_YEAR, dtype=int)
    charge_reason = np.full(QH_PER_YEAR, "none", dtype=object)
    discharge_reason = np.full(QH_PER_YEAR, "none", dtype=object)
    effective_discharge_value = np.zeros(QH_PER_YEAR, dtype=float)
    wholesale_charge_for_future_afrr_flag = np.zeros(QH_PER_YEAR, dtype=int)
    afrr_down_charge_for_future_wholesale_flag = np.zeros(QH_PER_YEAR, dtype=int)
    afrr_down_charge_for_future_afrr_up_flag = np.zeros(QH_PER_YEAR, dtype=int)
    wholesale_discharge_spread_ok = np.zeros(QH_PER_YEAR, dtype=int)
    afrr_up_discharge_spread_ok = np.zeros(QH_PER_YEAR, dtype=int)
    future_opportunity_selected = np.zeros(QH_PER_YEAR, dtype=int)

    forward_soc_before = np.zeros(QH_PER_YEAR, dtype=float)
    forward_soc_after = np.zeros(QH_PER_YEAR, dtype=float)
    forward_soc_mwh = np.zeros(QH_PER_YEAR + 1, dtype=float)
    forward_soc_mwh[0] = min(max(float(inputs.initial_soc_mwh), min_soc_mwh), max_soc_mwh)

    available_soc_headroom_input = np.zeros(QH_PER_YEAR, dtype=float)
    available_discharge_output = np.zeros(QH_PER_YEAR, dtype=float)
    up_soc_feasible = np.zeros(QH_PER_YEAR, dtype=int)
    down_soc_feasible = np.zeros(QH_PER_YEAR, dtype=int)
    up_rejected_due_to_soc = np.zeros(QH_PER_YEAR, dtype=int)
    down_rejected_due_to_soc = np.zeros(QH_PER_YEAR, dtype=int)

    for t in range(QH_PER_YEAR):
        soc_now = min(max(forward_soc_mwh[t], min_soc_mwh), max_soc_mwh)
        forward_soc_before[t] = soc_now

        headroom_input_t = max(max_soc_mwh - soc_now, 0.0) / max(inputs.eta_charge, 1e-12)
        discharge_output_t = max(soc_now - min_soc_mwh, 0.0) * inputs.eta_discharge
        available_soc_headroom_input[t] = headroom_input_t
        available_discharge_output[t] = discharge_output_t

        # Expected activated MWh used for the economic comparison and later physical dispatch.
        # UP also needs export headroom; DOWN only needs SOC headroom in this Phase-1 model.
        up_target_full = certified_up * QH_DT_HOURS * activation_up
        down_target_full = certified_down * QH_DT_HOURS * activation_down
        up_target_t = min(up_target_full, available_export_headroom[t])
        down_target_t = down_target_full

        up_feasible_t = (
            certified_up > 0
            and up_target_t > 1e-12
            and available_export_headroom[t] + 1e-12 >= up_target_t
            and discharge_output_t + 1e-12 >= up_target_t
        )
        down_feasible_t = (
            certified_down > 0
            and down_target_t > 1e-12
            and headroom_input_t + 1e-12 >= down_target_t
        )
        up_soc_feasible[t] = int(up_feasible_t)
        down_soc_feasible[t] = int(down_feasible_t)

        stored_cost_per_output_mwh_t = stored_energy_cost_qh[t] / max(inputs.eta_discharge, 1e-12)
        afrr_up_spread_t = (
            afrr_up_energy_price[t]
            - stored_cost_per_output_mwh_t
            - inputs.afrr_cycle_cost_eur_per_mwh / max(inputs.eta_discharge, 1e-12)
        )
        up_spread_ok_t = afrr_up_spread_t + 1e-12 >= inputs.afrr_min_spread_eur_per_mwh
        afrr_up_discharge_spread_ok[t] = int(up_spread_ok_t)

        up_energy_value_t = up_target_t * afrr_up_energy_price[t]
        up_degradation_t = up_target_t / max(inputs.eta_discharge, 1e-12) * inputs.afrr_cycle_cost_eur_per_mwh
        up_total_t = expected_up_capacity[t] + up_energy_value_t - up_degradation_t
        if not up_spread_ok_t:
            up_total_t = -1e30
        if not up_feasible_t:
            if up_total_t > max(wholesale_expected[t], 0.0):
                up_rejected_due_to_soc[t] = 1
            up_total_t = -1e30

        # DOWN sign convention: positive DOWN price is a charging cost; negative price is revenue/benefit.
        # Add cross-market future value: DOWN charge now can be used later for wholesale or aFRR UP discharge.
        future_output_mwh_t = down_target_t * inputs.eta_charge * inputs.eta_discharge
        down_energy_value_t = -down_target_t * afrr_down_energy_price[t]
        down_future_value_t = 0.0
        down_required_spread_t = inputs.afrr_min_spread_eur_per_mwh
        down_spread_t = -1e30
        if future_best_type[t] == "wholesale":
            down_required_spread_t = inputs.afrr_down_to_wholesale_min_spread_eur_per_mwh
        elif future_best_type[t] == "afrr_up":
            down_required_spread_t = inputs.afrr_min_spread_eur_per_mwh
        if future_best_value[t] > -1e20 and down_target_t > 1e-12:
            input_cost_per_future_output = afrr_down_energy_price[t] / max(inputs.eta_charge * inputs.eta_discharge, 1e-12)
            down_spread_t = future_best_value[t] - input_cost_per_future_output - inputs.afrr_cycle_cost_eur_per_mwh
            if down_spread_t + 1e-12 >= down_required_spread_t:
                down_future_value_t = future_output_mwh_t * future_best_value[t]
                future_opportunity_selected[t] = 1
                if future_best_type[t] == "wholesale":
                    afrr_down_charge_for_future_wholesale_flag[t] = 1
                elif future_best_type[t] == "afrr_up":
                    afrr_down_charge_for_future_afrr_up_flag[t] = 1
        down_total_t = expected_down_capacity[t] + down_energy_value_t + down_future_value_t
        if not down_feasible_t:
            if down_total_t > max(wholesale_expected[t], 0.0):
                down_rejected_due_to_soc[t] = 1
            down_total_t = -1e30

        up_total_value_audit[t] = 0.0 if up_total_t <= -1e20 else up_total_t
        down_total_value_audit[t] = 0.0 if down_total_t <= -1e20 else down_total_t

        best_val = max(float(wholesale_expected[t]), float(up_total_t), float(down_total_t), 0.0)
        if up_total_t == best_val and up_total_t > 0 and up_total_t > wholesale_expected[t] + 1e-9:
            selected[t] = "up"
            selected_market[t] = "afrr_up_capacity"
            up_awarded[t] = 1
            up_revenue[t] = expected_up_capacity[t]
            expected_up_activated[t] = up_target_t
            up_energy_value_selected[t] = up_energy_value_t
            expected_degradation_selected[t] = up_degradation_t
            cross_market_spread[t] = afrr_up_spread_t
            required_min_spread[t] = inputs.afrr_min_spread_eur_per_mwh
            spread_condition_respected[t] = int(up_spread_ok_t)
            discharge_reason[t] = "afrr_up_capacity_activation_spread_ok"
            effective_discharge_value[t] = afrr_up_energy_price[t]
            forward_soc_mwh[t + 1] = min(max(soc_now - up_target_t / max(inputs.eta_discharge, 1e-12), min_soc_mwh), max_soc_mwh)
        elif down_total_t == best_val and down_total_t > 0 and down_total_t > wholesale_expected[t] + 1e-9:
            selected[t] = "down"
            selected_market[t] = "afrr_down_capacity"
            down_awarded[t] = 1
            down_revenue[t] = expected_down_capacity[t]
            expected_down_activated[t] = down_target_t
            down_energy_value_selected[t] = down_energy_value_t + down_future_value_t
            cross_market_spread[t] = down_spread_t if down_spread_t > -1e20 else 0.0
            required_min_spread[t] = down_required_spread_t
            spread_condition_respected[t] = int(down_spread_t + 1e-12 >= down_required_spread_t)
            charge_reason[t] = "afrr_down_charge_for_future_" + str(future_best_type[t])
            effective_discharge_value[t] = max(future_best_value[t], 0.0)
            forward_soc_mwh[t + 1] = min(max(soc_now + down_target_t * inputs.eta_charge, min_soc_mwh), max_soc_mwh)
        elif wholesale_expected[t] == best_val and wholesale_expected[t] > 0:
            selected_market[t] = "wholesale"
            # Apply the baseline wholesale SOC movement, but from the forward SOC state.
            forward_soc_mwh[t + 1] = min(max(soc_now + baseline_soc_delta[t], min_soc_mwh), max_soc_mwh)
        else:
            forward_soc_mwh[t + 1] = soc_now

        forward_soc_after[t] = forward_soc_mwh[t + 1]

    return {
        "afrr_capacity_up_awarded_h": up_awarded,
        "afrr_capacity_down_awarded_h": down_awarded,
        "afrr_capacity_selected_market_h": selected,
        "afrr_capacity_up_revenue_h_eur": up_revenue,
        "afrr_capacity_down_revenue_h_eur": down_revenue,
        "afrr_capacity_total_revenue_h_eur": up_revenue + down_revenue,
        "afrr_capacity_eligible_h": np.ones(QH_PER_YEAR, dtype=int),
        "afrr_certified_capacity_up_mw_h": np.full(QH_PER_YEAR, certified_up, dtype=float),
        "afrr_certified_capacity_down_mw_h": np.full(QH_PER_YEAR, certified_down, dtype=float),
        "wholesale_opportunity_value_eur": wholesale_opportunity,
        "wholesale_expected_value_after_capture_rate_eur": wholesale_expected,
        "raw_up_capacity_revenue_eur": raw_up_capacity,
        "expected_up_capacity_revenue_eur": expected_up_capacity,
        "raw_down_capacity_revenue_eur": raw_down_capacity,
        "expected_down_capacity_revenue_eur": expected_down_capacity,
        "expected_up_activated_mwh": expected_up_activated,
        "expected_down_activated_mwh": expected_down_activated,
        "afrr_up_energy_expected_value_eur": up_energy_value_selected,
        "afrr_down_energy_expected_value_eur": down_energy_value_selected,
        "afrr_up_total_expected_value_eur": up_total_value_audit,
        "afrr_down_total_expected_value_eur": down_total_value_audit,
        "selected_market": selected_market,
        "selected_capacity_direction": selected,
        "afrr_capacity_success_rate_pct": np.full(QH_PER_YEAR, float(inputs.afrr_capacity_success_rate_pct), dtype=float),
        "bess_wholesale_capture_rate_pct": np.full(QH_PER_YEAR, float(inputs.bess_capture_rate_pct), dtype=float),
        "afrr_up_activation_pct": np.full(QH_PER_YEAR, float(inputs.afrr_energy_up_activation_pct), dtype=float),
        "afrr_down_activation_pct": np.full(QH_PER_YEAR, float(inputs.afrr_energy_down_activation_pct), dtype=float),
        "available_export_headroom_mwh": available_export_headroom,
        "available_soc_headroom_mwh": available_soc_headroom_input,
        "available_discharge_from_soc_mwh": available_discharge_output,
        "required_up_soc_reserve_mwh": np.full(QH_PER_YEAR, required_up_soc_reserve_mwh, dtype=float),
        "required_down_soc_headroom_mwh": np.full(QH_PER_YEAR, required_down_soc_headroom_mwh, dtype=float),
        "expected_degradation_cost_eur": expected_degradation_selected,
        "future_best_market_value_eur_per_mwh": future_best_value,
        "future_best_market_type": future_best_type,
        "cross_market_spread_eur_per_mwh": cross_market_spread,
        "required_min_spread_eur_per_mwh": required_min_spread,
        "spread_condition_respected": spread_condition_respected,
        "charge_reason": charge_reason,
        "discharge_reason": discharge_reason,
        "stored_energy_cost_eur_per_mwh": stored_energy_cost_qh,
        "effective_discharge_value_eur_per_mwh": effective_discharge_value,
        "future_expected_afrr_up_value_eur": future_afrr_up_value,
        "future_expected_wholesale_value_eur": future_wholesale_value,
        "future_expected_best_discharge_market": future_best_type,
        "wholesale_charge_for_future_afrr_flag": wholesale_charge_for_future_afrr_flag,
        "afrr_down_charge_for_future_wholesale_flag": afrr_down_charge_for_future_wholesale_flag,
        "afrr_down_charge_for_future_afrr_up_flag": afrr_down_charge_for_future_afrr_up_flag,
        "wholesale_discharge_spread_ok": wholesale_discharge_spread_ok,
        "afrr_up_discharge_spread_ok": afrr_up_discharge_spread_ok,
        "forward_horizon_hours": forward_curves["forward_horizon_hours"],
        "future_opportunity_selected": future_opportunity_selected,
        "forward_soc_before_capacity_selection_mwh": forward_soc_before,
        "forward_soc_after_capacity_selection_mwh": forward_soc_after,
        "afrr_up_soc_feasible": up_soc_feasible,
        "afrr_down_soc_feasible": down_soc_feasible,
        "afrr_up_rejected_due_to_soc": up_rejected_due_to_soc,
        "afrr_down_rejected_due_to_soc": down_rejected_due_to_soc,
        "afrr_up_expected_vs_actual_shortfall_mwh": np.zeros(QH_PER_YEAR, dtype=float),
        "afrr_down_expected_vs_actual_shortfall_mwh": np.zeros(QH_PER_YEAR, dtype=float),
        "afrr_up_rejected_due_to_final_combined_soc": np.zeros(QH_PER_YEAR, dtype=int),
        "afrr_down_rejected_due_to_final_combined_soc": np.zeros(QH_PER_YEAR, dtype=int),
    }

def build_standard_france_solar_profile() -> np.ndarray:
    idx = build_quarter_hour_index(DEFAULT_YEAR)
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    seasonal = 0.18 + 0.82 * (0.5 + 0.5 * np.sin(2 * np.pi * (doy - 81) / 365.0))
    daylight_hours = 8.0 + 8.0 * (0.5 + 0.5 * np.sin(2 * np.pi * (doy - 81) / 365.0))
    sunrise = 12.0 - daylight_hours / 2.0
    sunset = 12.0 + daylight_hours / 2.0
    shape = np.zeros(QH_PER_YEAR, dtype=float)

    for i in range(QH_PER_YEAR):
        if sunrise[i] <= hour[i] <= sunset[i]:
            x = (hour[i] - sunrise[i]) / max(sunset[i] - sunrise[i], 1e-9)
            shape[i] = (np.sin(np.pi * x) ** 1.6) * seasonal[i]

    total = shape.sum()
    if total <= 0:
        raise ValueError("Impossible de générer une courbe solaire standard valide.")
    return shape / total


def build_pv_generation_mwh(
    solar_profile_relative: np.ndarray,
    pv_dc_mw: float,
    productible_kwh_per_kwp: float,
    pv_losses_pct: float,
    plant_availability_pct: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    relative = _validate_array_length(solar_profile_relative, "Le profil solaire")
    relative = np.maximum(relative, 0.0)

    if relative.sum() <= 0:
        raise ValueError("Le profil solaire doit avoir une somme strictement positive.")
    if pv_dc_mw < 0 or productible_kwh_per_kwp < 0:
        raise ValueError("La puissance PV et le productible doivent être positifs.")
    if not (0 <= pv_losses_pct <= 100):
        raise ValueError("Les pertes PV doivent être entre 0 et 100 %.")
    if not (0 <= plant_availability_pct <= 100):
        raise ValueError("La disponibilité doit être entre 0 et 100 %.")

    annual_dc_mwh = pv_dc_mw * productible_kwh_per_kwp
    net_factor = (1.0 - pv_losses_pct / 100.0) * (plant_availability_pct / 100.0)
    annual_net_mwh = annual_dc_mwh * net_factor
    relative = relative / relative.sum()
    qh_net_mwh = annual_net_mwh * relative

    stats = {
        "annual_dc_mwh": float(annual_dc_mwh),
        "annual_net_mwh": float(annual_net_mwh),
        "annual_losses_mwh": float(max(annual_dc_mwh - annual_net_mwh, 0.0)),
    }
    return qh_net_mwh, stats


def apply_tso_dso_curtailment(
    pv_hourly_mwh: np.ndarray,
    monthly_curtailment_pct: np.ndarray,
) -> Dict[str, np.ndarray]:
    pv_hourly_mwh = _validate_array_length(pv_hourly_mwh, "PV horaire avant TSO/DSO")
    monthly_curtailment_pct = np.asarray(monthly_curtailment_pct, dtype=float).reshape(-1)

    if len(monthly_curtailment_pct) != 12:
        raise ValueError("La courbe mensuelle de curtailment TSO/DSO doit avoir 12 valeurs.")

    idx = build_quarter_hour_index(DEFAULT_YEAR)
    month_idx = idx.month.to_numpy() - 1
    pct_hourly = monthly_curtailment_pct[month_idx]

    pv_after = pv_hourly_mwh * (1.0 - pct_hourly)
    pv_after = np.maximum(pv_after, 0.0)
    curtailed = np.maximum(pv_hourly_mwh - pv_after, 0.0)
    flag = curtailed > 1e-12

    return {
        "pv_after_tso_dso_mwh": pv_after,
        "tso_dso_curtailed_mwh": curtailed,
        "tso_dso_curtailment_flag": flag.astype(int),
        "tso_dso_monthly_pct_hourly": pct_hourly,
    }


def apply_self_curtailment(
    pv_hourly_mwh: np.ndarray,
    pv_spot_price_raw: np.ndarray,
    pv_spot_price_effective: np.ndarray,
    enable_self_curtailment: bool,
    pv_commercial_structure: str,
    curtailment_threshold_eur_per_mwh: float,
    cfd_price_eur_per_mwh: float,
    negative_price_rule: bool,
    consecutive_negative_hours_limit: int,
    ppa_price_eur_per_mwh: float,
) -> Dict[str, np.ndarray]:
    pv_hourly_mwh = _validate_array_length(pv_hourly_mwh, "PV avant self curtailment")
    pv_spot_price_raw = _validate_array_length(pv_spot_price_raw, "Prix spot PV raw")
    pv_spot_price_effective = _validate_array_length(pv_spot_price_effective, "Prix spot PV effectif")

    sellable = pv_hourly_mwh.copy()
    pv_effective_price = pv_spot_price_effective.copy()
    self_curtailed = np.zeros(QH_PER_YEAR, dtype=float)
    self_flag = np.zeros(QH_PER_YEAR, dtype=int)
    structure_arr = np.full(QH_PER_YEAR, pv_commercial_structure, dtype=object)
    reason_arr = np.full(QH_PER_YEAR, "", dtype=object)

    if not enable_self_curtailment:
        return {
            "pv_after_self_curtailment_mwh": sellable,
            "self_curtailed_mwh": self_curtailed,
            "self_curtailment_flag": self_flag,
            "pv_effective_price_eur_per_mwh": pv_effective_price,
            "pv_commercial_structure_hourly": structure_arr,
            "self_curtailment_reason": reason_arr,
        }

    if pv_commercial_structure == "Fully merchant":
        mask = pv_spot_price_raw <= curtailment_threshold_eur_per_mwh
        self_curtailed[mask] = sellable[mask]
        sellable[mask] = 0.0
        self_flag[mask] = 1
        reason_arr[mask] = "Merchant threshold curtailment"
        pv_effective_price = pv_spot_price_effective

    elif pv_commercial_structure == "With CfD":
        pv_effective_price[:] = float(cfd_price_eur_per_mwh)

        if negative_price_rule:
            neg_run = 0
            for t in range(QH_PER_YEAR):
                if pv_spot_price_raw[t] < 0:
                    neg_run += 1
                    if neg_run > int(consecutive_negative_hours_limit):
                        self_curtailed[t] = sellable[t]
                        sellable[t] = 0.0
                        self_flag[t] = 1
                        reason_arr[t] = "CfD negative-hours curtailment"
                else:
                    neg_run = 0

    elif pv_commercial_structure == "With PPA":
        pv_effective_price[:] = float(ppa_price_eur_per_mwh)
        mask = pv_spot_price_raw <= curtailment_threshold_eur_per_mwh
        self_curtailed[mask] = sellable[mask]
        sellable[mask] = 0.0
        self_flag[mask] = 1
        reason_arr[mask] = "PPA threshold curtailment"

    else:
        raise ValueError(f"Structure commerciale PV non reconnue: {pv_commercial_structure}")

    return {
        "pv_after_self_curtailment_mwh": sellable,
        "self_curtailed_mwh": self_curtailed,
        "self_curtailment_flag": self_flag,
        "pv_effective_price_eur_per_mwh": pv_effective_price,
        "pv_commercial_structure_hourly": structure_arr,
        "self_curtailment_reason": reason_arr,
    }


def build_pure_pv_benchmark(
    pv_generation_mwh: np.ndarray,
    pv_price: np.ndarray,
    grid_export_limit_mw: float,
) -> Dict[str, np.ndarray]:
    pv_generation_mwh = _validate_array_length(pv_generation_mwh, "Production PV benchmark")
    pv_price = _validate_array_length(pv_price, "Prix PV benchmark")

    pv_only_direct_mwh = np.minimum(np.maximum(pv_generation_mwh, 0.0), float(grid_export_limit_mw) * QH_DT_HOURS)
    pv_only_revenue_eur = pv_only_direct_mwh * pv_price
    total_pv_only_revenue_eur = float(pv_only_revenue_eur.sum())

    return {
        "pv_only_direct_mwh": pv_only_direct_mwh,
        "pv_only_revenue_eur": pv_only_revenue_eur,
        "total_pv_only_revenue_eur": np.array([total_pv_only_revenue_eur]),
    }


def optimize_dispatch_dp(inputs: SimulationInputs) -> Dict[str, np.ndarray]:
    pv_sellable = _validate_array_length(inputs.solar_profile, "La production PV nette 15 minutes sellable")
    pv_sellable = np.maximum(pv_sellable, 0.0)

    if inputs.curtailed_pv_recoverable_mwh is None:
        pv_recoverable = np.zeros(QH_PER_YEAR, dtype=float)
    else:
        pv_recoverable = _validate_array_length(inputs.curtailed_pv_recoverable_mwh, "PV curtailed recoverable")
        pv_recoverable = np.maximum(pv_recoverable, 0.0)

    pv_price = _validate_array_length(inputs.pv_price, "Le prix PV")
    batt_sell = _validate_array_length(inputs.batt_sell_price, "Le prix de vente batterie")
    grid_buy = _validate_array_length(inputs.grid_buy_price, "Le prix d'achat réseau")

    idx = build_quarter_hour_index(DEFAULT_YEAR)

    df_thresholds = pd.DataFrame({
        "datetime": idx,
        "grid_buy": grid_buy,
        "batt_sell": batt_sell,
    })
    df_thresholds["day"] = df_thresholds["datetime"].dt.date

    charge_threshold_series = df_thresholds.groupby("day")["grid_buy"].transform(
        lambda x: np.percentile(x, inputs.charge_quantile)
    ).to_numpy()

    discharge_threshold_series = df_thresholds.groupby("day")["batt_sell"].transform(
        lambda x: np.percentile(x, inputs.discharge_quantile)
    ).to_numpy()

    if np.any(~np.isfinite(pv_sellable)) or np.any(~np.isfinite(pv_price)) or np.any(~np.isfinite(batt_sell)) or np.any(~np.isfinite(grid_buy)):
        raise ValueError("Une ou plusieurs séries contiennent des valeurs invalides.")
    if inputs.batt_power_mw < 0 or inputs.batt_energy_mwh < 0:
        raise ValueError("La puissance et la capacité batterie doivent être positives.")
    if inputs.eta_charge <= 0 or inputs.eta_charge > 1:
        raise ValueError("Le rendement de charge doit être compris entre 0 et 1.")
    if inputs.eta_discharge <= 0 or inputs.eta_discharge > 1:
        raise ValueError("Le rendement de décharge doit être compris entre 0 et 1.")
    if inputs.initial_soc_mwh < 0 or inputs.final_soc_mwh < 0:
        raise ValueError("Les SOC initial et final doivent être positifs.")
    if inputs.initial_soc_mwh > inputs.batt_energy_mwh:
        raise ValueError("Le SOC initial ne peut pas dépasser la capacité batterie.")
    if inputs.final_soc_mwh > inputs.batt_energy_mwh:
        raise ValueError("Le SOC final ne peut pas dépasser la capacité batterie.")
    if not (0.0 <= inputs.min_soc_pct <= 100.0):
        raise ValueError("Minimum SOC batterie (%) doit être compris entre 0 et 100 %.")
    if not (0.0 <= inputs.max_soc_pct <= 100.0):
        raise ValueError("Maximum SOC batterie (%) doit être compris entre 0 et 100 %.")
    if inputs.min_soc_pct >= inputs.max_soc_pct:
        raise ValueError("Minimum SOC batterie (%) doit être strictement inférieur au Maximum SOC batterie (%).")

    min_soc_mwh = inputs.batt_energy_mwh * inputs.min_soc_pct / 100.0
    max_soc_mwh = inputs.batt_energy_mwh * inputs.max_soc_pct / 100.0

    if inputs.initial_soc_mwh < min_soc_mwh or inputs.initial_soc_mwh > max_soc_mwh:
        raise ValueError("Le SOC initial doit être compris dans la plage SOC autorisée.")
    if inputs.final_soc_mwh < min_soc_mwh or inputs.final_soc_mwh > max_soc_mwh:
        raise ValueError("Le SOC final doit être compris dans la plage SOC autorisée.")

    T = len(pv_sellable)
    if T != QH_PER_YEAR:
        raise ValueError("Toutes les séries doivent contenir 35040 pas de 15 minutes.")

    if inputs.max_cycles_per_year < 0:
        raise ValueError("Cycles max / an doit être positif ou nul.")

    max_annual_discharge_mwh = float(inputs.max_cycles_per_year) * float(inputs.batt_energy_mwh)
    minimum_discharge_to_reach_final_mwh = max(inputs.initial_soc_mwh - inputs.final_soc_mwh, 0.0) * inputs.eta_discharge
    if max_annual_discharge_mwh + 1e-9 < minimum_discharge_to_reach_final_mwh:
        raise ValueError(
            "Cycles max / an est trop faible pour atteindre le SOC final demandé. "
            f"Minimum requis: {minimum_discharge_to_reach_final_mwh / max(inputs.batt_energy_mwh, 1e-12):.3f} cycles/an."
        )

    # aFRR Capacity awarded hours reserve the battery and block wholesale battery actions.
    if inputs.afrr_capacity_selected_market_h is None:
        afrr_capacity_selected_market_h = np.full(T, "none", dtype=object)
    else:
        afrr_capacity_selected_market_h = np.asarray(inputs.afrr_capacity_selected_market_h, dtype=object).reshape(-1)
        if len(afrr_capacity_selected_market_h) != T:
            raise ValueError("La courbe de sélection aFRR Capacity doit contenir 35040 pas de 15 minutes.")

    battery_blocked_by_afrr_capacity = np.isin(afrr_capacity_selected_market_h, ["up", "down"])

    soc_steps = int(max(21, inputs.soc_steps))
    soc_grid = np.linspace(min_soc_mwh, max_soc_mwh, soc_steps)

    def nearest_state_index(value: float) -> int:
        value = min(max(value, min_soc_mwh), max_soc_mwh)
        return int(np.argmin(np.abs(soc_grid - value)))

    init_idx = nearest_state_index(inputs.initial_soc_mwh)
    final_idx = nearest_state_index(inputs.final_soc_mwh)

    DT = QH_DT_HOURS
    charge_soc_max = inputs.batt_power_mw * inputs.eta_charge * DT
    discharge_soc_max = inputs.batt_power_mw * DT / inputs.eta_discharge

    transitions = []
    for i, soc in enumerate(soc_grid):
        j_min = np.searchsorted(soc_grid, max(min_soc_mwh, soc - discharge_soc_max), side="left")
        j_max = np.searchsorted(soc_grid, min(max_soc_mwh, soc + charge_soc_max), side="right") - 1
        transitions.append(np.arange(j_min, j_max + 1, dtype=int))

    forward_curves_dp = compute_forward_cross_market_value_curves(inputs)
    future_best_sell_price_from_t = np.maximum(
        forward_curves_dp["future_expected_wholesale_value_eur_per_mwh"],
        forward_curves_dp["future_expected_afrr_up_value_eur_per_mwh"],
    )
    future_best_market_type_from_t = forward_curves_dp["future_best_market_type"]
    future_best_sell_price_from_t = np.nan_to_num(future_best_sell_price_from_t, nan=-1e30, posinf=1e30, neginf=-1e30)
    
    def run_dp_once(
        required_discharge_price_estimate: np.ndarray,
        annual_cycle_budget_penalty_eur_per_mwh: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        neg_inf = -1e30
        value_next = np.full(soc_steps, neg_inf, dtype=float)
        value_next[final_idx] = 0.0
        policy_next = np.full((T, soc_steps), -1, dtype=np.int16 if soc_steps < 32000 else np.int32)

        estimate_gate = np.asarray(required_discharge_price_estimate, dtype=float).reshape(-1)
        if len(estimate_gate) != T:
            raise ValueError("La courbe estimée de prix requis de décharge a une mauvaise longueur.")
        estimate_gate = np.nan_to_num(estimate_gate, nan=-1e30, posinf=1e30, neginf=-1e30)

        for t in range(T - 1, -1, -1):
            value_now = np.full(soc_steps, neg_inf, dtype=float)
            pv_sellable_t = pv_sellable[t]
            pv_recoverable_t = pv_recoverable[t]
            pv_price_t = pv_price[t]
            batt_sell_t = batt_sell[t]
            grid_buy_t = grid_buy[t]

            for i in range(soc_steps):
                best_val = neg_inf
                best_j = -1
                soc_i = soc_grid[i]

                for j in transitions[i]:
                    delta_soc = soc_grid[j] - soc_i

                    # If aFRR Capacity is awarded for this hour, the battery must be reserved:
                    # no PV-to-battery, curtailed-PV-to-battery, grid charge or wholesale discharge.
                    if battery_blocked_by_afrr_capacity[t] and abs(delta_soc) > 1e-12:
                        continue

                    pv_direct_candidate = pv_sellable_t
                    sellable_pv_to_batt = 0.0
                    recoverable_pv_to_batt = 0.0
                    grid_charge = 0.0
                    discharge_candidate = 0.0
                    cycle_penalty = 0.0

                    if delta_soc > 1e-12:
                        charge_input = delta_soc / inputs.eta_charge

                        recoverable_pv_to_batt = min(charge_input, pv_recoverable_t)
                        remaining_after_recoverable = charge_input - recoverable_pv_to_batt

                        sellable_pv_to_batt = min(remaining_after_recoverable, pv_sellable_t)
                        remaining_after_sellable = remaining_after_recoverable - sellable_pv_to_batt

                        grid_charge = max(remaining_after_sellable, 0.0)
                        pv_direct_candidate = pv_sellable_t - sellable_pv_to_batt
                        pv_is_producing = (pv_sellable_t + pv_recoverable_t) > 1e-9
                        
                        # Block grid charging when PV is producing
                        if grid_charge > 1e-9 and pv_sellable_t > 1e-9:
                            continue

                        if grid_charge > 1e-9 and pv_is_producing:
                            continue
                            
                        if grid_charge > 1e-9:
                            if grid_buy_t > charge_threshold_series[t]:
                                continue
                        
                            future_best_sell_price = future_best_sell_price_from_t[t]
                            future_route = future_best_market_type_from_t[t]
                            if future_route == "afrr_up":
                                required_spread = max(inputs.afrr_up_cross_market_min_spread_eur_per_mwh, inputs.afrr_min_spread_eur_per_mwh)
                            else:
                                required_spread = inputs.min_spread_arbitrage_eur_per_mwh
                        
                            required_future_sell_price = (
                                grid_buy_t / max(inputs.eta_charge * inputs.eta_discharge, 1e-12)
                                + required_spread
                                + inputs.cycle_cost_eur_per_mwh
                            )
                        
                            if future_best_sell_price < required_future_sell_price:
                                continue

                    elif delta_soc < -1e-12:
                        # PV priority rule with export headroom:
                        # PV keeps priority on the grid export limit, but BESS may
                        # discharge during PV production if PV does not already fill
                        # the available injection capacity.
                        pv_export_headroom = max(inputs.grid_export_limit_mw * QH_DT_HOURS - pv_sellable_t, 0.0)
                        if pv_export_headroom <= 1e-9:
                            continue

                        discharge_candidate = min((-delta_soc) * inputs.eta_discharge, pv_export_headroom)

                        if discharge_candidate > 1e-9:
                            if batt_sell_t < discharge_threshold_series[t]:
                                continue
                            if batt_sell_t < estimate_gate[t]:
                                continue

                    total_export = pv_direct_candidate + discharge_candidate

                    if total_export > inputs.grid_export_limit_mw * QH_DT_HOURS:
                        excess = total_export - inputs.grid_export_limit_mw * QH_DT_HOURS
                        reduction_pv = min(excess, pv_direct_candidate)
                        pv_direct_candidate -= reduction_pv
                        excess -= reduction_pv

                        if excess > 0:
                            discharge_candidate = max(discharge_candidate - excess, 0.0)

                        # Cycle cost is applied below for every MWh actually discharged,
                        # not only when the grid export limit is binding.

                    if discharge_candidate > 1e-12:
                        # Marginal degradation / wear cost, in EUR per MWh discharged.
                        # This makes cycle cost economically effective in the dispatch decision.
                        cycle_penalty = discharge_candidate * inputs.cycle_cost_eur_per_mwh

                    reward = pv_direct_candidate * pv_price_t

                    if delta_soc > 1e-12:
                        reward -= grid_charge * grid_buy_t
                    elif delta_soc < -1e-12:
                        reward += discharge_candidate * batt_sell_t
                        reward -= cycle_penalty
                        # Shadow price used only by the optimizer to allocate a limited
                        # annual cycle budget to the best spreads over the full year.
                        reward -= annual_cycle_budget_penalty_eur_per_mwh * discharge_candidate

                    total_val = reward + value_next[j]
                    if total_val > best_val:
                        best_val = total_val
                        best_j = int(j)

                value_now[i] = best_val
                policy_next[t, i] = best_j

            value_next = value_now

        if np.all(value_next == neg_inf):
            raise RuntimeError("DP failed: all states unreachable")

        soc = np.zeros(T + 1, dtype=float)
        soc[0] = soc_grid[init_idx]
        state = init_idx

        pv_direct = np.zeros(T, dtype=float)
        pv_to_batt = np.zeros(T, dtype=float)
        pv_curtailed_to_battery = np.zeros(T, dtype=float)
        grid_charge = np.zeros(T, dtype=float)
        discharge = np.zeros(T, dtype=float)
        batt_sale_revenue = np.zeros(T, dtype=float)
        grid_charge_cost = np.zeros(T, dtype=float)
        wholesale_cycle_cost = np.zeros(T, dtype=float)
        pv_direct_revenue = np.zeros(T, dtype=float)
        avg_stored_charge_price = np.full(T + 1, np.nan, dtype=float)
        required_discharge_price = np.full(T, np.nan, dtype=float)
        stored_energy_value_eur = 0.0
        stored_energy_mwh = soc[0]

        avg_stored_charge_price[0] = 0.0 if stored_energy_mwh > 1e-9 else np.nan

        for t in range(T):
            next_state = int(policy_next[t, state])
            if next_state < 0:
                raise RuntimeError(f"Policy failure at t={t}, state={state}")

            delta_soc = soc_grid[next_state] - soc_grid[state]
            if battery_blocked_by_afrr_capacity[t] and abs(delta_soc) > 1e-9:
                raise RuntimeError(f"aFRR Capacity wholesale block violated at t={t}")
            soc[t + 1] = soc_grid[next_state]

            pv_sellable_t = pv_sellable[t]
            pv_recoverable_t = pv_recoverable[t]

            pv_direct_candidate = pv_sellable_t
            sellable_pv_to_batt = 0.0
            recoverable_pv_to_batt = 0.0
            grid_charge[t] = 0.0
            discharge[t] = 0.0

            if delta_soc > 1e-12:
                charge_input = delta_soc / inputs.eta_charge

                recoverable_pv_to_batt = min(charge_input, pv_recoverable_t)
                remaining_after_recoverable = charge_input - recoverable_pv_to_batt

                sellable_pv_to_batt = min(remaining_after_recoverable, pv_sellable_t)
                remaining_after_sellable = remaining_after_recoverable - sellable_pv_to_batt

                grid_charge[t] = max(remaining_after_sellable, 0.0)
                pv_direct_candidate = pv_sellable_t - sellable_pv_to_batt

            elif delta_soc < -1e-12:
                # Safety mirror of the DP rule above: allow discharge only into
                # remaining grid export headroom after PV priority.
                pv_export_headroom = max(inputs.grid_export_limit_mw * QH_DT_HOURS - pv_sellable_t, 0.0)
                discharge[t] = min((-delta_soc) * inputs.eta_discharge, pv_export_headroom)

            pv_to_batt[t] = sellable_pv_to_batt
            pv_curtailed_to_battery[t] = recoverable_pv_to_batt

            if delta_soc > 1e-12:
                charge_cost_eur = (
                    sellable_pv_to_batt * pv_price[t] +
                    grid_charge[t] * grid_buy[t]
                    # recoverable_pv_to_batt enters at zero opportunity cost
                )
                stored_energy_value_eur += charge_cost_eur
                stored_energy_mwh += delta_soc

            elif delta_soc < -1e-12:
                avg_cost_now = stored_energy_value_eur / max(stored_energy_mwh, 1e-9)
                energy_removed_from_soc = -delta_soc
                cost_removed_eur = avg_cost_now * energy_removed_from_soc
                stored_energy_value_eur = max(stored_energy_value_eur - cost_removed_eur, 0.0)
                stored_energy_mwh = max(stored_energy_mwh - energy_removed_from_soc, 0.0)

            if stored_energy_mwh > 1e-9:
                avg_stored_charge_price[t + 1] = stored_energy_value_eur / stored_energy_mwh
            else:
                avg_stored_charge_price[t + 1] = np.nan

            if np.isfinite(avg_stored_charge_price[t]):
                required_discharge_price[t] = (
                    avg_stored_charge_price[t]
                    + inputs.min_spread_arbitrage_eur_per_mwh
                    + inputs.cycle_cost_eur_per_mwh
                )

            total_export = pv_direct_candidate + discharge[t]
            if total_export > inputs.grid_export_limit_mw:
                excess = total_export - inputs.grid_export_limit_mw
                reduction_pv = min(excess, pv_direct_candidate)
                pv_direct_candidate -= reduction_pv
                excess -= reduction_pv
                if excess > 0:
                    discharge[t] = max(discharge[t] - excess, 0.0)

            pv_direct[t] = max(pv_direct_candidate, 0.0)
            pv_direct_revenue[t] = pv_direct[t] * pv_price[t]
            batt_sale_revenue[t] = discharge[t] * batt_sell[t]
            grid_charge_cost[t] = grid_charge[t] * grid_buy[t]
            # Option A: cycle cost is a dispatch hurdle and an informational theoretical degradation metric only.
            # It is NOT deducted from cash revenue.
            wholesale_cycle_cost[t] = discharge[t] * inputs.cycle_cost_eur_per_mwh
            state = next_state

        total_direct_pv_revenue = float(pv_direct_revenue.sum())
        total_batt_sale_revenue = float(batt_sale_revenue.sum())
        total_grid_charge_cost = float(grid_charge_cost.sum())
        total_wholesale_cycle_cost = float(wholesale_cycle_cost.sum())
        nightly_revenue_total = float(inputs.nightly_bess_revenue_eur * (T // 24))
        total_revenue = total_direct_pv_revenue + total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total
        total_discharged_mwh = float(discharge.sum())
        equivalent_cycles = total_discharged_mwh / max(inputs.batt_energy_mwh, 1e-12)
        remaining_cycle_budget_mwh = max(max_annual_discharge_mwh - total_discharged_mwh, 0.0)

        return {
            "soc": soc,
            "pv_direct": pv_direct,
            "pv_to_batt": pv_to_batt,
            "pv_curtailed_to_battery": pv_curtailed_to_battery,
            "grid_charge": grid_charge,
            "discharge": discharge,
            "pv_direct_revenue": pv_direct_revenue,
            "batt_sale_revenue": batt_sale_revenue,
            "grid_charge_cost": grid_charge_cost,
            "wholesale_cycle_cost_eur": wholesale_cycle_cost,
            "total_direct_pv_revenue": np.array([total_direct_pv_revenue]),
            "total_batt_sale_revenue": np.array([total_batt_sale_revenue]),
            "total_grid_charge_cost": np.array([total_grid_charge_cost]),
            "total_wholesale_cycle_cost_eur": np.array([total_wholesale_cycle_cost]),
            "gross_bess_revenue_before_cycle_cost_eur": np.array([total_batt_sale_revenue]),
            "net_bess_revenue_after_cycle_cost_eur": np.array([total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total]),
            "bess_cash_revenue_eur": np.array([total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total]),
            "nightly_revenue_total": np.array([nightly_revenue_total]),
            "total_revenue": np.array([total_revenue]),
            "equivalent_cycles": np.array([equivalent_cycles]),
            "energy_sold_total_mwh": np.array([pv_direct.sum() + total_discharged_mwh]),
            "energy_shifted_mwh": np.array([total_discharged_mwh]),
            "max_cycles_per_year": np.array([float(inputs.max_cycles_per_year)]),
            "annual_discharge_cap_mwh": np.array([max_annual_discharge_mwh]),
            "remaining_cycle_budget_mwh": np.array([remaining_cycle_budget_mwh]),
            "annual_cycle_budget_penalty_eur_per_mwh": np.array([float(annual_cycle_budget_penalty_eur_per_mwh)]),
            "pv_direct_sold_mwh": np.array([pv_direct.sum()]),
            "avg_stored_charge_price": avg_stored_charge_price,
            "required_discharge_price": required_discharge_price,
            "hourly_datetime": idx,
            "required_discharge_price_gate_estimate": estimate_gate,
            "afrr_capacity_selected_market_h": afrr_capacity_selected_market_h,
            "battery_blocked_by_afrr_capacity": battery_blocked_by_afrr_capacity.astype(int),
        }

    def run_dp_with_annual_cycle_cap(required_discharge_price_estimate: np.ndarray) -> Dict[str, np.ndarray]:
        """Run the annual DP with a global annual discharge budget.

        A direct SOC x cycle-budget DP would be very large for 35040 quarter-hour steps.
        This uses the equivalent Lagrangian form: a shadow price is applied to
        every MWh discharged, then found by bisection. Because the DP still sees
        the full 35040-step quarter-hour horizon, it can skip low-value cycles early in the
        year and keep the limited annual cycle budget for better spreads later.
        """
        cap_tolerance_mwh = max(1e-6, 1e-6 * max(inputs.batt_energy_mwh, 1.0))
        uncapped = run_dp_once(required_discharge_price_estimate, 0.0)
        if float(uncapped["energy_shifted_mwh"][0]) <= max_annual_discharge_mwh + cap_tolerance_mwh:
            return uncapped

        low_penalty = 0.0
        high_penalty = max(1.0, float(np.nanmax(batt_sell) - np.nanmin(grid_buy) + inputs.min_spread_arbitrage_eur_per_mwh))
        capped = run_dp_once(required_discharge_price_estimate, high_penalty)

        # Increase the shadow price until the selected annual dispatch respects the cap.
        for _ in range(3):
            if float(capped["energy_shifted_mwh"][0]) <= max_annual_discharge_mwh + cap_tolerance_mwh:
                break
            low_penalty = high_penalty
            high_penalty *= 2.0
            capped = run_dp_once(required_discharge_price_estimate, high_penalty)

        if float(capped["energy_shifted_mwh"][0]) > max_annual_discharge_mwh + cap_tolerance_mwh:
            raise RuntimeError(
                "Impossible de respecter Cycles max / an avec les contraintes SOC initial/final et les pas de SOC choisis. "
                "Augmentez Cycles max / an, réduisez le SOC final requis, ou augmentez le nombre de pas de SOC."
            )

        best_capped = capped
        for _ in range(3):
            mid_penalty = 0.5 * (low_penalty + high_penalty)
            candidate = run_dp_once(required_discharge_price_estimate, mid_penalty)
            if float(candidate["energy_shifted_mwh"][0]) <= max_annual_discharge_mwh + cap_tolerance_mwh:
                high_penalty = mid_penalty
                best_capped = candidate
            else:
                low_penalty = mid_penalty

        return best_capped

    max_passes = 2
    required_estimate = np.full(T, -1e30, dtype=float)
    final_result = None

    for _ in range(max_passes):
        candidate = run_dp_with_annual_cycle_cap(required_estimate)

        new_estimate = np.nan_to_num(
            candidate["required_discharge_price"],
            nan=-1e30,
            posinf=1e30,
            neginf=-1e30,
        )

        # Important: only tighten the gate, never loosen it.
        tightened_estimate = np.maximum(required_estimate, new_estimate)

        discharge_mask = candidate["discharge"] > 1e-6
        valid_required_mask = np.isfinite(candidate["required_discharge_price"])

        violations = (
            discharge_mask
            & valid_required_mask
            & (batt_sell < candidate["required_discharge_price"] - 1e-9)
        )

        final_result = candidate

        if not violations.any() and np.allclose(
            tightened_estimate,
            required_estimate,
            atol=1e-6,
            rtol=0.0,
        ):
            break

        required_estimate = tightened_estimate.copy()

    return final_result


def _afrr_qh_limits(
    batt_power_mw: float,
    eta_charge: float,
    eta_discharge: float,
    dt_hours: float = QH_DT_HOURS,
) -> Dict[str, float]:
    input_per_qh = batt_power_mw * dt_hours
    stored_per_qh = input_per_qh * eta_charge
    output_per_qh = stored_per_qh * eta_discharge
    return {
        "input_per_qh_mwh": float(input_per_qh),
        "stored_per_qh_mwh": float(stored_per_qh),
        "output_per_qh_mwh": float(output_per_qh),
    }


def select_best_daily_afrr_trade_blocks(
    charge_prices_day: np.ndarray,
    discharge_prices_day: np.ndarray,
    eligible_mask_day: np.ndarray,
    batt_power_mw: float,
    batt_energy_mwh: float,
    eta_charge: float,
    eta_discharge: float,
    afrr_cycle_cost_eur_per_mwh: float,
    afrr_min_spread_eur_per_mwh: float,
    n_qh: int = 4,
    dt_hours: float = QH_DT_HOURS,
) -> Dict[str, object]:
    idx_eligible = np.where(eligible_mask_day)[0]

    if len(idx_eligible) < 2 * n_qh:
        return {
            "execute": False,
            "charge_indices": [],
            "discharge_indices": [],
            "avg_charge_price": np.nan,
            "avg_discharge_price": np.nan,
            "expected_net_spread_eur_per_mwh": np.nan,
            "expected_charge_input_mwh": 0.0,
            "expected_stored_mwh": 0.0,
            "expected_discharge_output_mwh": 0.0,
            "reason": "Pas assez de quarts d'heure éligibles.",
        }

    best = {
        "execute": False,
        "charge_indices": [],
        "discharge_indices": [],
        "avg_charge_price": np.nan,
        "avg_discharge_price": np.nan,
        "expected_net_spread_eur_per_mwh": -np.inf,
        "expected_charge_input_mwh": 0.0,
        "expected_stored_mwh": 0.0,
        "expected_discharge_output_mwh": 0.0,
        "reason": "Aucune combinaison valide.",
    }

    power_limited_input_per_qh = batt_power_mw * dt_hours
    total_charge_input_mwh = n_qh * power_limited_input_per_qh
    total_stored_mwh = total_charge_input_mwh * eta_charge
    total_discharge_output_mwh = total_stored_mwh * eta_discharge

    for split_pos in range(1, len(idx_eligible)):
        charge_pool = idx_eligible[:split_pos]
        discharge_pool = idx_eligible[split_pos:]

        if len(charge_pool) < n_qh or len(discharge_pool) < n_qh:
            continue

        charge_sorted = charge_pool[np.argsort(charge_prices_day[charge_pool])]
        selected_charge = np.sort(charge_sorted[:n_qh])

        discharge_sorted = discharge_pool[np.argsort(-discharge_prices_day[discharge_pool])]
        selected_discharge = np.sort(discharge_sorted[:n_qh])

        if len(selected_charge) < n_qh or len(selected_discharge) < n_qh:
            continue
        if selected_charge.max() >= selected_discharge.min():
            continue

        avg_charge_price = float(np.mean(charge_prices_day[selected_charge]))
        avg_discharge_price = float(np.mean(discharge_prices_day[selected_discharge]))

        effective_input_cost_per_mwh_out = avg_charge_price / max(eta_charge * eta_discharge, 1e-12)
        net_spread = avg_discharge_price - effective_input_cost_per_mwh_out - afrr_cycle_cost_eur_per_mwh

        if net_spread > best["expected_net_spread_eur_per_mwh"]:
            best = {
                "execute": net_spread >= afrr_min_spread_eur_per_mwh,
                "charge_indices": selected_charge.tolist(),
                "discharge_indices": selected_discharge.tolist(),
                "avg_charge_price": avg_charge_price,
                "avg_discharge_price": avg_discharge_price,
                "expected_net_spread_eur_per_mwh": float(net_spread),
                "expected_charge_input_mwh": float(total_charge_input_mwh),
                "expected_stored_mwh": float(total_stored_mwh),
                "expected_discharge_output_mwh": float(total_discharge_output_mwh),
                "reason": "OK" if net_spread >= afrr_min_spread_eur_per_mwh else "Spread insuffisant.",
            }

    if best["expected_net_spread_eur_per_mwh"] < afrr_min_spread_eur_per_mwh:
        best["execute"] = False

    return best


def _select_hourly_activation_by_pct(
    awarded_mask_h: np.ndarray,
    price_h: np.ndarray,
    activation_pct: float,
    prefer: str,
) -> np.ndarray:
    """Deterministically select awarded Capacity hours for aFRR Energy activation."""
    awarded_idx = np.where(np.asarray(awarded_mask_h, dtype=bool))[0]
    selected = np.zeros(QH_PER_YEAR, dtype=int)
    if len(awarded_idx) == 0:
        return selected

    pct = min(max(float(activation_pct), 0.0), 100.0)
    n_select = int(np.floor(len(awarded_idx) * pct / 100.0 + 0.5))
    n_select = min(max(n_select, 0), len(awarded_idx))
    if n_select == 0:
        return selected

    prices = np.asarray(price_h, dtype=float)[awarded_idx]
    if prefer == "low":
        order = np.lexsort((awarded_idx, prices))
    elif prefer == "high":
        order = np.lexsort((awarded_idx, -prices))
    else:
        raise ValueError("prefer must be 'low' or 'high'.")

    chosen = awarded_idx[order[:n_select]]
    selected[chosen] = 1
    return selected


def _select_best_daily_afrr_competing_blocks(
    charge_prices_day: np.ndarray,
    discharge_prices_day: np.ndarray,
    grid_buy_prices_day: np.ndarray,
    batt_sell_prices_day: np.ndarray,
    eligible_mask_day: np.ndarray,
    eta_charge: float,
    eta_discharge: float,
    afrr_cycle_cost_eur_per_mwh: float,
    afrr_min_spread_eur_per_mwh: float,
    n_qh: int,
) -> Dict[str, object]:
    """Mode 2 candidate selection: aFRR competes with wholesale routes on eligible QHs."""
    eligible = np.asarray(eligible_mask_day, dtype=bool)
    charge_candidate = np.where(eligible & (charge_prices_day < grid_buy_prices_day))[0]
    discharge_candidate = np.where(eligible & (discharge_prices_day > batt_sell_prices_day))[0]

    best = {
        "execute": False,
        "charge_indices": [],
        "discharge_indices": [],
        "avg_charge_price": np.nan,
        "avg_discharge_price": np.nan,
        "expected_net_spread_eur_per_mwh": np.nan,
        "reason": "Aucune combinaison aFRR meilleure que wholesale.",
    }

    if len(charge_candidate) < n_qh or len(discharge_candidate) < n_qh:
        return best

    eligible_idx = np.where(eligible)[0]
    for split_abs in eligible_idx:
        charge_pool = charge_candidate[charge_candidate < split_abs]
        discharge_pool = discharge_candidate[discharge_candidate > split_abs]
        if len(charge_pool) < n_qh or len(discharge_pool) < n_qh:
            continue

        selected_charge = np.sort(charge_pool[np.argsort(charge_prices_day[charge_pool])[:n_qh]])
        selected_discharge = np.sort(discharge_pool[np.argsort(-discharge_prices_day[discharge_pool])[:n_qh]])
        if selected_charge.max() >= selected_discharge.min():
            continue

        avg_charge_price = float(np.mean(charge_prices_day[selected_charge]))
        avg_discharge_price = float(np.mean(discharge_prices_day[selected_discharge]))
        net_spread = (
            avg_discharge_price
            - avg_charge_price / max(eta_charge * eta_discharge, 1e-12)
            - afrr_cycle_cost_eur_per_mwh
        )

        if (not np.isfinite(best["expected_net_spread_eur_per_mwh"])) or net_spread > best["expected_net_spread_eur_per_mwh"]:
            best = {
                "execute": net_spread >= afrr_min_spread_eur_per_mwh,
                "charge_indices": selected_charge.tolist(),
                "discharge_indices": selected_discharge.tolist(),
                "avg_charge_price": avg_charge_price,
                "avg_discharge_price": avg_discharge_price,
                "expected_net_spread_eur_per_mwh": float(net_spread),
                "reason": "OK" if net_spread >= afrr_min_spread_eur_per_mwh else "Spread insuffisant.",
            }

    if not best["execute"]:
        best["charge_indices"] = []
        best["discharge_indices"] = []
    return best



def enforce_afrr_capacity_deliverability_from_final_dispatch(
    afrr_capacity_result: Dict[str, np.ndarray],
    reconciliation: Dict[str, np.ndarray] | None,
    tolerance_mwh: float = 1e-6,
) -> tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Remove aFRR capacity awards that cannot be delivered in final dispatch.

    simulate_afrr_capacity() uses a forward-SOC approximation before the final
    wholesale/aFRR reconciliation is known. This post-pass validates awarded
    capacity against the actual final combined dispatch. Any UP award whose
    expected activation is not physically delivered in the final reconciliation
    is removed, then the caller should rerun final DP/aFRR dispatch using the
    filtered awards. This makes UP capacity awards require deliverability under
    the final combined SOC trajectory, not only the preliminary forward tracker.
    """
    if afrr_capacity_result is None or reconciliation is None:
        return afrr_capacity_result, {"removed_up": 0, "removed_down": 0}

    filtered: Dict[str, np.ndarray] = {}
    for key, value in afrr_capacity_result.items():
        if isinstance(value, np.ndarray):
            filtered[key] = value.copy()
        else:
            filtered[key] = value

    selected = np.asarray(filtered.get("afrr_capacity_selected_market_h", np.full(QH_PER_YEAR, "none", dtype=object)), dtype=object).copy()
    selected_market = np.asarray(filtered.get("selected_market", np.full(QH_PER_YEAR, "none", dtype=object)), dtype=object).copy()
    selected_direction = np.asarray(filtered.get("selected_capacity_direction", selected), dtype=object).copy()

    expected_up = _validate_array_length(filtered.get("expected_up_activated_mwh", np.zeros(QH_PER_YEAR)), "Expected aFRR UP activated MWh")
    expected_down = _validate_array_length(filtered.get("expected_down_activated_mwh", np.zeros(QH_PER_YEAR)), "Expected aFRR DOWN activated MWh")
    actual_up = _validate_array_length(reconciliation.get("afrr_discharge_qh_mwh", np.zeros(QH_PER_YEAR)), "Actual aFRR UP discharge MWh")
    actual_down = _validate_array_length(reconciliation.get("afrr_charge_qh_mwh", np.zeros(QH_PER_YEAR)), "Actual aFRR DOWN charge MWh")

    up_shortfall = np.maximum(expected_up - actual_up, 0.0)
    down_shortfall = np.maximum(expected_down - actual_down, 0.0)

    # UP deliverability is the critical issue observed in the simulations:
    # reject any UP award that final combined SOC/export constraints cannot activate.
    remove_up = (selected == "up") & (up_shortfall > tolerance_mwh)
    # Apply the same consistency rule to DOWN as a safety check; it usually removes few/no intervals.
    remove_down = (selected == "down") & (down_shortfall > tolerance_mwh)

    removed_up_count = int(np.sum(remove_up))
    removed_down_count = int(np.sum(remove_down))

    if removed_up_count == 0 and removed_down_count == 0:
        filtered["afrr_up_expected_vs_actual_shortfall_mwh"] = up_shortfall
        filtered["afrr_down_expected_vs_actual_shortfall_mwh"] = down_shortfall
        filtered["afrr_up_rejected_due_to_final_combined_soc"] = np.zeros(QH_PER_YEAR, dtype=int)
        filtered["afrr_down_rejected_due_to_final_combined_soc"] = np.zeros(QH_PER_YEAR, dtype=int)
        return filtered, {"removed_up": 0, "removed_down": 0}

    rejected_up_final = np.asarray(filtered.get("afrr_up_rejected_due_to_final_combined_soc", np.zeros(QH_PER_YEAR)), dtype=int).copy()
    rejected_down_final = np.asarray(filtered.get("afrr_down_rejected_due_to_final_combined_soc", np.zeros(QH_PER_YEAR)), dtype=int).copy()
    rejected_up_final[remove_up] = 1
    rejected_down_final[remove_down] = 1

    remove_any = remove_up | remove_down
    selected[remove_any] = "none"
    selected_market[remove_any] = "none"
    selected_direction[remove_any] = "none"

    filtered["afrr_capacity_selected_market_h"] = selected
    filtered["selected_market"] = selected_market
    filtered["selected_capacity_direction"] = selected_direction

    if "afrr_capacity_up_awarded_h" in filtered:
        arr = np.asarray(filtered["afrr_capacity_up_awarded_h"], dtype=int).copy()
        arr[remove_up] = 0
        filtered["afrr_capacity_up_awarded_h"] = arr
    if "afrr_capacity_down_awarded_h" in filtered:
        arr = np.asarray(filtered["afrr_capacity_down_awarded_h"], dtype=int).copy()
        arr[remove_down] = 0
        filtered["afrr_capacity_down_awarded_h"] = arr

    zero_when_removed_keys = [
        "afrr_capacity_up_revenue_h_eur",
        "expected_up_activated_mwh",
        "afrr_up_energy_expected_value_eur",
        "expected_degradation_cost_eur",
    ]
    for key in zero_when_removed_keys:
        if key in filtered:
            arr = np.asarray(filtered[key], dtype=float).copy()
            arr[remove_up] = 0.0
            filtered[key] = arr

    zero_down_keys = [
        "afrr_capacity_down_revenue_h_eur",
        "expected_down_activated_mwh",
        "afrr_down_energy_expected_value_eur",
    ]
    for key in zero_down_keys:
        if key in filtered:
            arr = np.asarray(filtered[key], dtype=float).copy()
            arr[remove_down] = 0.0
            filtered[key] = arr

    for key, mask in [
        ("afrr_up_total_expected_value_eur", remove_up),
        ("afrr_down_total_expected_value_eur", remove_down),
        ("expected_up_capacity_revenue_eur", remove_up),
        ("expected_down_capacity_revenue_eur", remove_down),
    ]:
        # Keep raw price/value columns for audit, but zero selected expected value columns on rejected awards.
        if key in filtered and key not in ("expected_up_capacity_revenue_eur", "expected_down_capacity_revenue_eur"):
            arr = np.asarray(filtered[key], dtype=float).copy()
            arr[mask] = 0.0
            filtered[key] = arr

    up_rev = np.asarray(filtered.get("afrr_capacity_up_revenue_h_eur", np.zeros(QH_PER_YEAR)), dtype=float)
    down_rev = np.asarray(filtered.get("afrr_capacity_down_revenue_h_eur", np.zeros(QH_PER_YEAR)), dtype=float)
    filtered["afrr_capacity_total_revenue_h_eur"] = up_rev + down_rev

    filtered["afrr_up_expected_vs_actual_shortfall_mwh"] = up_shortfall
    filtered["afrr_down_expected_vs_actual_shortfall_mwh"] = down_shortfall
    filtered["afrr_up_rejected_due_to_final_combined_soc"] = rejected_up_final
    filtered["afrr_down_rejected_due_to_final_combined_soc"] = rejected_down_final

    return filtered, {"removed_up": removed_up_count, "removed_down": removed_down_count}

def simulate_afrr_night_arbitrage(inputs: SimulationInputs, result_hourly: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not inputs.enable_afrr:
        return {
            "afrr_charge_qh_mwh": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_discharge_qh_mwh": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_soc_qh": np.asarray(result_hourly["soc"][:-1], dtype=float),
            "afrr_charge_cost_qh_eur": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_sale_revenue_qh_eur": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_cycle_cost_qh_eur": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_net_revenue_qh_eur": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_energy_down_activated_qh": np.zeros(QH_PER_YEAR, dtype=int),
            "afrr_energy_up_activated_qh": np.zeros(QH_PER_YEAR, dtype=int),
            "selected_charge_market_qh": np.full(QH_PER_YEAR, "none", dtype=object),
            "selected_discharge_market_qh": np.full(QH_PER_YEAR, "none", dtype=object),
            "afrr_daily_log": pd.DataFrame(),
        }

    if inputs.afrr_charge_price_qh is None or inputs.afrr_discharge_price_qh is None:
        raise ValueError("Les courbes de prix aFRR quart-horaires doivent être fournies si aFRR est activé.")

    charge_prices_qh = _validate_array_length(inputs.afrr_charge_price_qh, "Prix aFRR charge", QH_PER_YEAR)
    discharge_prices_qh = _validate_array_length(inputs.afrr_discharge_price_qh, "Prix aFRR décharge", QH_PER_YEAR)
    grid_buy_price_qh = _validate_array_length(inputs.grid_buy_price, "Prix achat réseau BESS", QH_PER_YEAR)
    batt_sell_price_qh = _validate_array_length(inputs.batt_sell_price, "Prix vente BESS", QH_PER_YEAR)

    idx_qh = build_quarter_hour_index(DEFAULT_YEAR)
    pv_qh = _validate_array_length(inputs.solar_profile, "Production PV nette quart-horaire", QH_PER_YEAR)

    # Phase 1: aFRR Energy is eligible at any 15-minute timestep.
    # Do not restrict to night hours and do not block DOWN activation during PV production.
    eligible_mask_qh = np.ones(QH_PER_YEAR, dtype=bool)

    afrr_charge_qh_mwh = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_discharge_qh_mwh = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_charge_cost_qh_eur = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_sale_revenue_qh_eur = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_cycle_cost_qh_eur = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_net_revenue_qh_eur = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_soc_qh = np.zeros(QH_PER_YEAR, dtype=float)
    down_activated_qh = np.zeros(QH_PER_YEAR, dtype=int)
    up_activated_qh = np.zeros(QH_PER_YEAR, dtype=int)
    selected_charge_market_qh = np.full(QH_PER_YEAR, "none", dtype=object)
    selected_discharge_market_qh = np.full(QH_PER_YEAR, "none", dtype=object)
    up_activation_shortfall_qh = np.zeros(QH_PER_YEAR, dtype=float)
    down_activation_shortfall_qh = np.zeros(QH_PER_YEAR, dtype=float)

    min_soc_mwh = inputs.batt_energy_mwh * inputs.min_soc_pct / 100.0
    max_soc_mwh = inputs.batt_energy_mwh * inputs.max_soc_pct / 100.0
    soc_current = min(max(float(result_hourly["soc"][0]), min_soc_mwh), max_soc_mwh)
    max_charge_input_qh = inputs.batt_power_mw * QH_DT_HOURS
    max_discharge_output_qh = inputs.batt_power_mw * QH_DT_HOURS
    max_export_qh = inputs.grid_export_limit_mw * QH_DT_HOURS

    daily_logs = []

    if inputs.enable_afrr_capacity:
        if inputs.afrr_capacity_selected_market_h is None:
            capacity_selected_h = np.full(QH_PER_YEAR, "none", dtype=object)
        else:
            capacity_selected_h = np.asarray(inputs.afrr_capacity_selected_market_h, dtype=object).reshape(-1)
            if len(capacity_selected_h) != QH_PER_YEAR:
                raise ValueError("La courbe de sélection aFRR Capacity doit contenir 35040 pas de 15 minutes.")

        # Capacity-linked aFRR energy is represented as expected activated MWh on every awarded interval.
        # Activation percentages scale MWh, not the number of selected intervals.
        # IMPORTANT FIX: the physical aFRR energy dispatch now follows the same
        # expected activated MWh arrays used in simulate_afrr_capacity() for the
        # market-value comparison. This keeps the selected aFRR capacity value and
        # the actual aFRR energy dispatch consistent, subject only to hard physical
        # SOC, power and export constraints.
        down_selected_qh = (capacity_selected_h == "down")
        up_selected_qh = (capacity_selected_h == "up")
        down_selected_h = down_selected_qh.astype(int)
        up_selected_h = up_selected_qh.astype(int)

        if inputs.afrr_expected_down_activated_mwh_qh is not None:
            expected_down_dispatch_qh = _validate_array_length(
                inputs.afrr_expected_down_activated_mwh_qh,
                "Expected aFRR Down activated MWh",
                QH_PER_YEAR,
            )
        else:
            activation_down_factor = min(max(float(inputs.afrr_energy_down_activation_pct) / 100.0, 0.0), 1.0)
            expected_down_dispatch_qh = down_selected_qh.astype(float) * max_charge_input_qh * activation_down_factor

        if inputs.afrr_expected_up_activated_mwh_qh is not None:
            expected_up_dispatch_qh = _validate_array_length(
                inputs.afrr_expected_up_activated_mwh_qh,
                "Expected aFRR Up activated MWh",
                QH_PER_YEAR,
            )
        else:
            activation_up_factor = min(max(float(inputs.afrr_energy_up_activation_pct) / 100.0, 0.0), 1.0)
            expected_up_dispatch_qh = up_selected_qh.astype(float) * max_discharge_output_qh * activation_up_factor

        # Ensure non-awarded directions cannot create activation.
        expected_down_dispatch_qh = np.where(down_selected_qh, np.maximum(expected_down_dispatch_qh, 0.0), 0.0)
        expected_up_dispatch_qh = np.where(up_selected_qh, np.maximum(expected_up_dispatch_qh, 0.0), 0.0)

        up_activation_shortfall_qh = np.zeros(QH_PER_YEAR, dtype=float)
        down_activation_shortfall_qh = np.zeros(QH_PER_YEAR, dtype=float)

        for t in range(QH_PER_YEAR):
            if down_selected_qh[t]:
                target_input_qh = min(float(expected_down_dispatch_qh[t]), max_charge_input_qh)
                feasible_input_qh = min(
                    target_input_qh,
                    max(max_soc_mwh - soc_current, 0.0) / max(inputs.eta_charge, 1e-12),
                )
                down_activation_shortfall_qh[t] = max(target_input_qh - feasible_input_qh, 0.0)
                if feasible_input_qh > 1e-12:
                    afrr_charge_qh_mwh[t] = feasible_input_qh
                    afrr_charge_cost_qh_eur[t] = feasible_input_qh * charge_prices_qh[t]
                    afrr_net_revenue_qh_eur[t] -= afrr_charge_cost_qh_eur[t]
                    soc_current += feasible_input_qh * inputs.eta_charge
                    down_activated_qh[t] = 1
                    selected_charge_market_qh[t] = "afrr"
            elif up_selected_qh[t]:
                pv_direct_t = float(np.asarray(result_hourly.get("pv_direct", np.zeros(QH_PER_YEAR)), dtype=float)[t])
                export_room_t = max(max_export_qh - pv_direct_t, 0.0)
                target_discharge_qh = min(float(expected_up_dispatch_qh[t]), max_discharge_output_qh)
                feasible_discharge_qh = min(
                    target_discharge_qh,
                    export_room_t,
                    max(soc_current - min_soc_mwh, 0.0) * inputs.eta_discharge,
                )
                up_activation_shortfall_qh[t] = max(target_discharge_qh - feasible_discharge_qh, 0.0)
                if feasible_discharge_qh > 1e-12:
                    soc_removed = feasible_discharge_qh / max(inputs.eta_discharge, 1e-12)
                    theoretical_cycle_cost = soc_removed * inputs.afrr_cycle_cost_eur_per_mwh
                    expected_sale_revenue = feasible_discharge_qh * discharge_prices_qh[t]
                    # Do not re-apply the cycle-cost hurdle here: the expected aFRR
                    # capacity selection already included degradation/cycle cost in
                    # the value comparison. Re-applying it would make actual dispatch
                    # diverge from the expected MWh used for market selection.
                    afrr_discharge_qh_mwh[t] = feasible_discharge_qh
                    afrr_sale_revenue_qh_eur[t] = expected_sale_revenue
                    afrr_cycle_cost_qh_eur[t] = theoretical_cycle_cost
                    afrr_net_revenue_qh_eur[t] += afrr_sale_revenue_qh_eur[t]
                    soc_current -= soc_removed
                    up_activated_qh[t] = 1
                    selected_discharge_market_qh[t] = "afrr"

            soc_current = min(max(soc_current, min_soc_mwh), max_soc_mwh)
            afrr_soc_qh[t] = soc_current

        daily_logs.append({
            "day": pd.NaT,
            "mode": "capacity_activated",
            "executed": bool(down_activated_qh.any() or up_activated_qh.any()),
            "down_activation_pct": inputs.afrr_energy_down_activation_pct,
            "up_activation_pct": inputs.afrr_energy_up_activation_pct,
            "down_awarded_hours": int(np.sum(capacity_selected_h == "down")),
            "up_awarded_hours": int(np.sum(capacity_selected_h == "up")),
            "down_activated_hours": int(np.sum(down_selected_h)),
            "up_activated_hours": int(np.sum(up_selected_h)),
            "charge_cost_eur": float(afrr_charge_cost_qh_eur.sum()),
            "sale_revenue_eur": float(afrr_sale_revenue_qh_eur.sum()),
            "cycle_cost_eur": float(afrr_cycle_cost_qh_eur.sum()),
            "net_revenue_eur": float(afrr_net_revenue_qh_eur.sum()),
            "reason": "Capacity directional activation; aFRR cycle cost used as reference hurdle on upward activation, not deducted from cash revenue.",
        })

    else:
        df = pd.DataFrame({
            "datetime": idx_qh,
            "charge_price": charge_prices_qh,
            "discharge_price": discharge_prices_qh,
            "grid_buy_price": grid_buy_price_qh,
            "batt_sell_price": batt_sell_price_qh,
            "eligible": eligible_mask_qh,
        })
        df["day"] = df["datetime"].dt.date

        for day, group in df.groupby("day", sort=True):
            group_idx = group.index.to_numpy()
            best_trade = _select_best_daily_afrr_competing_blocks(
                charge_prices_day=group["charge_price"].to_numpy(dtype=float),
                discharge_prices_day=group["discharge_price"].to_numpy(dtype=float),
                grid_buy_prices_day=group["grid_buy_price"].to_numpy(dtype=float),
                batt_sell_prices_day=group["batt_sell_price"].to_numpy(dtype=float),
                eligible_mask_day=group["eligible"].to_numpy(dtype=bool),
                eta_charge=inputs.eta_charge,
                eta_discharge=inputs.eta_discharge,
                afrr_cycle_cost_eur_per_mwh=inputs.afrr_cycle_cost_eur_per_mwh,
                afrr_min_spread_eur_per_mwh=inputs.afrr_min_spread_eur_per_mwh,
                n_qh=int(inputs.afrr_n_qh_per_side),
            )

            selected_charge_abs_idx = []
            selected_discharge_abs_idx = []
            charged_input_mwh_total = 0.0
            discharged_mwh_total = 0.0
            charge_cost_eur_total = 0.0
            sale_revenue_eur_total = 0.0
            cycle_cost_eur_total = 0.0

            if best_trade["execute"]:
                for rel_idx in [int(i) for i in best_trade["charge_indices"]]:
                    t = int(group_idx[rel_idx])
                    input_this_qh = min(max_charge_input_qh, max(max_soc_mwh - soc_current, 0.0) / max(inputs.eta_charge, 1e-12))
                    if input_this_qh <= 1e-12:
                        continue
                    afrr_charge_qh_mwh[t] = input_this_qh
                    afrr_charge_cost_qh_eur[t] = input_this_qh * charge_prices_qh[t]
                    afrr_net_revenue_qh_eur[t] -= afrr_charge_cost_qh_eur[t]
                    soc_current += input_this_qh * inputs.eta_charge
                    down_activated_qh[t] = 1
                    selected_charge_market_qh[t] = "afrr"
                    selected_charge_abs_idx.append(t)
                    charged_input_mwh_total += input_this_qh
                    charge_cost_eur_total += afrr_charge_cost_qh_eur[t]
                    afrr_soc_qh[t] = soc_current

                for rel_idx in [int(i) for i in best_trade["discharge_indices"]]:
                    t = int(group_idx[rel_idx])
                    pv_direct_t = float(np.asarray(result_hourly.get("pv_direct", np.zeros(QH_PER_YEAR)), dtype=float)[t])
                    export_room_t = max(max_export_qh - pv_direct_t, 0.0)
                    discharge_this_qh = min(
                        max_discharge_output_qh,
                        export_room_t,
                        max(soc_current - min_soc_mwh, 0.0) * inputs.eta_discharge,
                    )
                    if discharge_this_qh <= 1e-12:
                        continue
                    soc_removed = discharge_this_qh / max(inputs.eta_discharge, 1e-12)
                    theoretical_cycle_cost = soc_removed * inputs.afrr_cycle_cost_eur_per_mwh
                    expected_sale_revenue = discharge_this_qh * discharge_prices_qh[t]
                    # Reference-only aFRR cycle-cost hurdle: skip activation if the
                    # activation value does not cover the theoretical degradation cost.
                    # The cost is NOT deducted from reported cash revenue when accepted.
                    if expected_sale_revenue <= theoretical_cycle_cost + 1e-12:
                        continue
                    afrr_discharge_qh_mwh[t] = discharge_this_qh
                    afrr_sale_revenue_qh_eur[t] = expected_sale_revenue
                    afrr_cycle_cost_qh_eur[t] = theoretical_cycle_cost
                    afrr_net_revenue_qh_eur[t] += afrr_sale_revenue_qh_eur[t]  # aFRR cycle cost is a decision/reference metric only, not deducted from cash revenue
                    soc_current -= soc_removed
                    up_activated_qh[t] = 1
                    selected_discharge_market_qh[t] = "afrr"
                    selected_discharge_abs_idx.append(t)
                    discharged_mwh_total += discharge_this_qh
                    sale_revenue_eur_total += afrr_sale_revenue_qh_eur[t]
                    cycle_cost_eur_total += afrr_cycle_cost_qh_eur[t]
                    afrr_soc_qh[t] = soc_current

            group_soc_missing = afrr_soc_qh[group_idx] == 0.0
            afrr_soc_qh[group_idx[group_soc_missing]] = soc_current
            daily_logs.append({
                "day": pd.to_datetime(day),
                "mode": "merchant_competing_routes",
                "executed": bool(len(selected_charge_abs_idx) or len(selected_discharge_abs_idx)),
                "charge_qh_indices": selected_charge_abs_idx,
                "discharge_qh_indices": selected_discharge_abs_idx,
                "charge_times": [idx_qh[i] for i in selected_charge_abs_idx],
                "discharge_times": [idx_qh[i] for i in selected_discharge_abs_idx],
                "avg_charge_price_eur_per_mwh": best_trade.get("avg_charge_price", np.nan),
                "avg_discharge_price_eur_per_mwh": best_trade.get("avg_discharge_price", np.nan),
                "expected_net_spread_eur_per_mwh": best_trade.get("expected_net_spread_eur_per_mwh", np.nan),
                "charged_input_mwh": charged_input_mwh_total,
                "discharged_mwh": discharged_mwh_total,
                "charge_cost_eur": charge_cost_eur_total,
                "sale_revenue_eur": sale_revenue_eur_total,
                "cycle_cost_eur": cycle_cost_eur_total,
                "net_revenue_eur": sale_revenue_eur_total - charge_cost_eur_total,  # aFRR cycle cost reference-only, not deducted
                "reason": best_trade.get("reason", "OK"),
            })

    return {
        "afrr_charge_qh_mwh": afrr_charge_qh_mwh,
        "afrr_discharge_qh_mwh": afrr_discharge_qh_mwh,
        "afrr_soc_qh": afrr_soc_qh,
        "afrr_charge_cost_qh_eur": afrr_charge_cost_qh_eur,
        "afrr_sale_revenue_qh_eur": afrr_sale_revenue_qh_eur,
        "afrr_cycle_cost_qh_eur": afrr_cycle_cost_qh_eur,
        "afrr_net_revenue_qh_eur": afrr_net_revenue_qh_eur,
        "afrr_energy_down_activated_qh": down_activated_qh,
        "afrr_energy_up_activated_qh": up_activated_qh,
        "afrr_up_activation_shortfall_qh_mwh": up_activation_shortfall_qh,
        "afrr_down_activation_shortfall_qh_mwh": down_activation_shortfall_qh,
        "selected_charge_market_qh": selected_charge_market_qh,
        "selected_discharge_market_qh": selected_discharge_market_qh,
        "afrr_daily_log": pd.DataFrame(daily_logs),
    }


def reconcile_wholesale_afrr_dispatch_qh(
    result_hourly: Dict[str, np.ndarray],
    afrr_result: Dict[str, np.ndarray],
    inputs: SimulationInputs,
) -> Dict[str, np.ndarray]:
    idx_qh = build_quarter_hour_index(DEFAULT_YEAR)

    pv_direct_qh = np.asarray(result_hourly["pv_direct"], dtype=float)
    wholesale_pv_to_batt_qh = np.asarray(result_hourly["pv_to_batt"], dtype=float)
    wholesale_pv_curtailed_to_batt_qh = np.asarray(result_hourly.get("pv_curtailed_to_battery", np.zeros(QH_PER_YEAR)), dtype=float)
    wholesale_grid_charge_qh = np.asarray(result_hourly["grid_charge"], dtype=float)
    wholesale_discharge_qh = np.asarray(result_hourly["discharge"], dtype=float)

    batt_sell_price_qh = np.asarray(inputs.batt_sell_price, dtype=float)
    grid_buy_price_qh = np.asarray(inputs.grid_buy_price, dtype=float)
    afrr_charge_price_qh = np.asarray(inputs.afrr_charge_price_qh, dtype=float)
    afrr_discharge_price_qh = np.asarray(inputs.afrr_discharge_price_qh, dtype=float)

    afrr_charge_qh = np.asarray(afrr_result["afrr_charge_qh_mwh"], dtype=float).copy()
    afrr_discharge_qh = np.asarray(afrr_result["afrr_discharge_qh_mwh"], dtype=float).copy()
    down_activated_qh = np.asarray(afrr_result.get("afrr_energy_down_activated_qh", np.zeros(QH_PER_YEAR)), dtype=int)
    up_activated_qh = np.asarray(afrr_result.get("afrr_energy_up_activated_qh", np.zeros(QH_PER_YEAR)), dtype=int)
    up_activation_shortfall_qh = np.asarray(afrr_result.get("afrr_up_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)), dtype=float)
    down_activation_shortfall_qh = np.asarray(afrr_result.get("afrr_down_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)), dtype=float)

    if inputs.afrr_capacity_selected_market_h is None:
        afrr_capacity_selected_market_h = np.full(QH_PER_YEAR, "none", dtype=object)
    else:
        afrr_capacity_selected_market_h = np.asarray(inputs.afrr_capacity_selected_market_h, dtype=object).reshape(-1)
        if len(afrr_capacity_selected_market_h) != QH_PER_YEAR:
            raise ValueError("La courbe de sélection aFRR Capacity doit contenir 35040 pas de 15 minutes.")
    afrr_capacity_selected_market_qh = afrr_capacity_selected_market_h

    corrected_wholesale_pv_to_batt_qh = wholesale_pv_to_batt_qh.copy()
    corrected_wholesale_pv_curtailed_to_batt_qh = wholesale_pv_curtailed_to_batt_qh.copy()
    corrected_wholesale_grid_charge_qh = wholesale_grid_charge_qh.copy()
    corrected_wholesale_discharge_qh = wholesale_discharge_qh.copy()
    corrected_afrr_charge_qh = afrr_charge_qh.copy()
    corrected_afrr_discharge_qh = afrr_discharge_qh.copy()

    selected_charge_market_qh = np.full(QH_PER_YEAR, "none", dtype=object)
    selected_charge_price_qh = np.full(QH_PER_YEAR, np.nan, dtype=float)
    selected_discharge_market_qh = np.full(QH_PER_YEAR, "none", dtype=object)
    selected_discharge_price_qh = np.full(QH_PER_YEAR, np.nan, dtype=float)
    stored_energy_cost_qh = np.nan_to_num(np.asarray(result_hourly.get("avg_stored_charge_price", np.zeros(QH_PER_YEAR + 1)), dtype=float).reshape(-1)[:QH_PER_YEAR], nan=0.0, posinf=0.0, neginf=0.0)
    effective_discharge_value_qh = np.zeros(QH_PER_YEAR, dtype=float)
    spread_condition_respected_qh = np.zeros(QH_PER_YEAR, dtype=int)
    wholesale_discharge_spread_ok_qh = np.zeros(QH_PER_YEAR, dtype=int)
    afrr_up_discharge_spread_ok_qh = np.zeros(QH_PER_YEAR, dtype=int)

    export_limit_qh_mwh = inputs.grid_export_limit_mw * QH_DT_HOURS
    min_soc_mwh = inputs.batt_energy_mwh * inputs.min_soc_pct / 100.0
    max_soc_mwh = inputs.batt_energy_mwh * inputs.max_soc_pct / 100.0
    combined_soc_qh = np.zeros(QH_PER_YEAR + 1, dtype=float)
    combined_soc_qh[0] = min(max(float(inputs.initial_soc_mwh), min_soc_mwh), max_soc_mwh)

    for t in range(QH_PER_YEAR):
        capacity_market = afrr_capacity_selected_market_qh[t]

        if capacity_market in ("up", "down"):
            corrected_wholesale_pv_to_batt_qh[t] = 0.0
            corrected_wholesale_pv_curtailed_to_batt_qh[t] = 0.0
            corrected_wholesale_grid_charge_qh[t] = 0.0
            corrected_wholesale_discharge_qh[t] = 0.0
            if capacity_market == "down":
                corrected_afrr_discharge_qh[t] = 0.0
            elif capacity_market == "up":
                corrected_afrr_charge_qh[t] = 0.0
        elif inputs.enable_afrr_capacity and not inputs.allow_afrr_energy_without_capacity:
            # If aFRR Energy without awarded Capacity is not allowed, remove all aFRR Energy
            # in quarter-hours where the battery did not receive an aFRR Capacity award.
            corrected_afrr_charge_qh[t] = 0.0
            corrected_afrr_discharge_qh[t] = 0.0
        else:
            # Mode 2: aFRR and wholesale compete as routes for the same physical battery.
            # This branch also allows aFRR Energy without Capacity when the checkbox is ticked.
            if corrected_afrr_charge_qh[t] > 1e-12:
                if afrr_charge_price_qh[t] < grid_buy_price_qh[t]:
                    corrected_wholesale_grid_charge_qh[t] = 0.0
                else:
                    corrected_afrr_charge_qh[t] = 0.0
            if corrected_afrr_discharge_qh[t] > 1e-12:
                if afrr_discharge_price_qh[t] > batt_sell_price_qh[t]:
                    corrected_wholesale_discharge_qh[t] = 0.0
                else:
                    corrected_afrr_discharge_qh[t] = 0.0

        export_room_qh = max(export_limit_qh_mwh - pv_direct_qh[t], 0.0)
        total_discharge_qh = corrected_wholesale_discharge_qh[t] + corrected_afrr_discharge_qh[t]
        if total_discharge_qh > export_room_qh + 1e-12:
            scale = export_room_qh / max(total_discharge_qh, 1e-12)
            corrected_wholesale_discharge_qh[t] *= scale
            corrected_afrr_discharge_qh[t] *= scale

        total_charge_qh = (
            corrected_wholesale_pv_to_batt_qh[t]
            + corrected_wholesale_pv_curtailed_to_batt_qh[t]
            + corrected_wholesale_grid_charge_qh[t]
            + corrected_afrr_charge_qh[t]
        )
        total_discharge_qh = corrected_wholesale_discharge_qh[t] + corrected_afrr_discharge_qh[t]

        # Never charge and discharge simultaneously. Keep the economically stronger selected route.
        if total_charge_qh > 1e-12 and total_discharge_qh > 1e-12:
            charge_saving = max(grid_buy_price_qh[t] - afrr_charge_price_qh[t], 0.0) if corrected_afrr_charge_qh[t] > 1e-12 else 0.0
            discharge_uplift = max(afrr_discharge_price_qh[t] - batt_sell_price_qh[t], 0.0) if corrected_afrr_discharge_qh[t] > 1e-12 else 0.0
            if discharge_uplift >= charge_saving:
                corrected_wholesale_pv_to_batt_qh[t] = 0.0
                corrected_wholesale_pv_curtailed_to_batt_qh[t] = 0.0
                corrected_wholesale_grid_charge_qh[t] = 0.0
                corrected_afrr_charge_qh[t] = 0.0
            else:
                corrected_wholesale_discharge_qh[t] = 0.0
                corrected_afrr_discharge_qh[t] = 0.0

        soc_now = combined_soc_qh[t]
        total_charge_input = (
            corrected_wholesale_pv_to_batt_qh[t]
            + corrected_wholesale_pv_curtailed_to_batt_qh[t]
            + corrected_wholesale_grid_charge_qh[t]
            + corrected_afrr_charge_qh[t]
        )
        max_charge_input_by_headroom = max(max_soc_mwh - soc_now, 0.0) / max(inputs.eta_charge, 1e-12)
        if total_charge_input > max_charge_input_by_headroom + 1e-12:
            scale = max_charge_input_by_headroom / max(total_charge_input, 1e-12)
            corrected_wholesale_pv_to_batt_qh[t] *= scale
            corrected_wholesale_pv_curtailed_to_batt_qh[t] *= scale
            corrected_wholesale_grid_charge_qh[t] *= scale
            corrected_afrr_charge_qh[t] *= scale

        total_discharge_output = corrected_wholesale_discharge_qh[t] + corrected_afrr_discharge_qh[t]
        max_discharge_output_by_soc = max(soc_now - min_soc_mwh, 0.0) * inputs.eta_discharge
        if total_discharge_output > max_discharge_output_by_soc + 1e-12:
            scale = max_discharge_output_by_soc / max(total_discharge_output, 1e-12)
            corrected_wholesale_discharge_qh[t] *= scale
            corrected_afrr_discharge_qh[t] *= scale

        # Enforce minimum spread before final discharge into wholesale or aFRR UP.
        cost_per_output = stored_energy_cost_qh[t] / max(inputs.eta_discharge, 1e-12)
        wholesale_spread_t = batt_sell_price_qh[t] - cost_per_output - inputs.cycle_cost_eur_per_mwh / max(inputs.eta_discharge, 1e-12)
        afrr_up_spread_t = afrr_discharge_price_qh[t] - cost_per_output - inputs.afrr_cycle_cost_eur_per_mwh / max(inputs.eta_discharge, 1e-12)
        if corrected_wholesale_discharge_qh[t] > 1e-12:
            wholesale_discharge_spread_ok_qh[t] = int(wholesale_spread_t + 1e-12 >= inputs.min_spread_arbitrage_eur_per_mwh)
            if not wholesale_discharge_spread_ok_qh[t]:
                corrected_wholesale_discharge_qh[t] = 0.0
        if corrected_afrr_discharge_qh[t] > 1e-12:
            afrr_up_discharge_spread_ok_qh[t] = int(afrr_up_spread_t + 1e-12 >= inputs.afrr_min_spread_eur_per_mwh)
            if not afrr_up_discharge_spread_ok_qh[t]:
                corrected_afrr_discharge_qh[t] = 0.0
        if corrected_wholesale_discharge_qh[t] > 1e-12 or corrected_afrr_discharge_qh[t] > 1e-12:
            spread_condition_respected_qh[t] = 1
            effective_discharge_value_qh[t] = max(
                batt_sell_price_qh[t] if corrected_wholesale_discharge_qh[t] > 1e-12 else -1e30,
                afrr_discharge_price_qh[t] if corrected_afrr_discharge_qh[t] > 1e-12 else -1e30,
            )

        if corrected_afrr_charge_qh[t] > 1e-12:
            selected_charge_market_qh[t] = "afrr"
            selected_charge_price_qh[t] = afrr_charge_price_qh[t]
        elif corrected_wholesale_grid_charge_qh[t] > 1e-12:
            selected_charge_market_qh[t] = "wholesale_grid"
            selected_charge_price_qh[t] = grid_buy_price_qh[t]
        elif corrected_wholesale_pv_to_batt_qh[t] > 1e-12:
            selected_charge_market_qh[t] = "pv"
        elif corrected_wholesale_pv_curtailed_to_batt_qh[t] > 1e-12:
            selected_charge_market_qh[t] = "curtailed_pv"

        if corrected_afrr_discharge_qh[t] > 1e-12:
            selected_discharge_market_qh[t] = "afrr"
            selected_discharge_price_qh[t] = afrr_discharge_price_qh[t]
        elif corrected_wholesale_discharge_qh[t] > 1e-12:
            selected_discharge_market_qh[t] = "wholesale"
            selected_discharge_price_qh[t] = batt_sell_price_qh[t]

        charge_to_soc = (
            corrected_wholesale_pv_to_batt_qh[t]
            + corrected_wholesale_pv_curtailed_to_batt_qh[t]
            + corrected_wholesale_grid_charge_qh[t]
            + corrected_afrr_charge_qh[t]
        ) * inputs.eta_charge
        discharge_from_soc = (
            corrected_wholesale_discharge_qh[t]
            + corrected_afrr_discharge_qh[t]
        ) / max(inputs.eta_discharge, 1e-12)
        combined_soc_qh[t + 1] = min(max(soc_now + charge_to_soc - discharge_from_soc, min_soc_mwh), max_soc_mwh)

    corrected_wholesale_batt_sale_revenue_qh = corrected_wholesale_discharge_qh * batt_sell_price_qh
    corrected_wholesale_grid_charge_cost_qh = corrected_wholesale_grid_charge_qh * grid_buy_price_qh
    corrected_afrr_charge_cost_qh = corrected_afrr_charge_qh * afrr_charge_price_qh
    corrected_afrr_sale_revenue_qh = corrected_afrr_discharge_qh * afrr_discharge_price_qh
    corrected_afrr_cycle_cost_qh = (corrected_afrr_discharge_qh / max(inputs.eta_discharge, 1e-12)) * inputs.afrr_cycle_cost_eur_per_mwh
    corrected_afrr_net_revenue_qh = corrected_afrr_sale_revenue_qh - corrected_afrr_charge_cost_qh  # aFRR cycle cost reference-only, not deducted

    charge_to_soc_qh = (
        corrected_wholesale_pv_to_batt_qh
        + corrected_wholesale_pv_curtailed_to_batt_qh
        + corrected_wholesale_grid_charge_qh
        + corrected_afrr_charge_qh
    ) * inputs.eta_charge
    discharge_from_soc_qh = (corrected_wholesale_discharge_qh + corrected_afrr_discharge_qh) / max(inputs.eta_discharge, 1e-12)

    def reshape_sum(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=float).reshape(HOURS_PER_YEAR, QH_PER_HOUR).sum(axis=1)

    def reshape_last(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=float).reshape(HOURS_PER_YEAR, QH_PER_HOUR)[:, -1]

    return {
        "datetime_qh": idx_qh,
        "wholesale_pv_to_batt_qh_mwh": corrected_wholesale_pv_to_batt_qh,
        "wholesale_pv_curtailed_to_batt_qh_mwh": corrected_wholesale_pv_curtailed_to_batt_qh,
        "wholesale_pv_curtailed_to_batt_hourly_mwh": corrected_wholesale_pv_curtailed_to_batt_qh,
        "wholesale_grid_charge_qh_mwh": corrected_wholesale_grid_charge_qh,
        "wholesale_discharge_qh_mwh": corrected_wholesale_discharge_qh,
        "wholesale_batt_sale_revenue_qh_eur": corrected_wholesale_batt_sale_revenue_qh,
        "wholesale_grid_charge_cost_qh_eur": corrected_wholesale_grid_charge_cost_qh,
        "afrr_charge_qh_mwh": corrected_afrr_charge_qh,
        "afrr_discharge_qh_mwh": corrected_afrr_discharge_qh,
        "afrr_charge_cost_qh_eur": corrected_afrr_charge_cost_qh,
        "afrr_sale_revenue_qh_eur": corrected_afrr_sale_revenue_qh,
        "afrr_cycle_cost_qh_eur": corrected_afrr_cycle_cost_qh,
        "afrr_net_revenue_qh_eur": corrected_afrr_net_revenue_qh,
        "afrr_energy_down_activated_qh": down_activated_qh,
        "afrr_energy_up_activated_qh": up_activated_qh,
        "afrr_up_activation_shortfall_qh_mwh": up_activation_shortfall_qh,
        "afrr_down_activation_shortfall_qh_mwh": down_activation_shortfall_qh,
        "selected_charge_market_qh": selected_charge_market_qh,
        "selected_charge_price_qh": selected_charge_price_qh,
        "selected_discharge_market_qh": selected_discharge_market_qh,
        "selected_discharge_channel_qh": selected_discharge_market_qh,
        "selected_discharge_price_qh": selected_discharge_price_qh,
        "afrr_capacity_selected_market_qh": afrr_capacity_selected_market_qh,
        "combined_charge_to_soc_qh_mwh": charge_to_soc_qh,
        "combined_discharge_from_soc_qh_mwh": discharge_from_soc_qh,
        "combined_soc_qh": combined_soc_qh,
        "combined_soc_hourly_end_mwh": combined_soc_qh[1:],
        "wholesale_pv_to_batt_hourly_mwh": corrected_wholesale_pv_to_batt_qh,
        "wholesale_grid_charge_hourly_mwh": corrected_wholesale_grid_charge_qh,
        "wholesale_discharge_hourly_mwh": corrected_wholesale_discharge_qh,
        "wholesale_batt_sale_revenue_hourly_eur": corrected_wholesale_batt_sale_revenue_qh,
        "wholesale_grid_charge_cost_hourly_eur": corrected_wholesale_grid_charge_cost_qh,
        "afrr_charge_hourly_mwh": corrected_afrr_charge_qh,
        "afrr_discharge_hourly_mwh": corrected_afrr_discharge_qh,
        "afrr_charge_cost_hourly_eur": corrected_afrr_charge_cost_qh,
        "afrr_sale_revenue_hourly_eur": corrected_afrr_sale_revenue_qh,
        "afrr_cycle_cost_hourly_eur": corrected_afrr_cycle_cost_qh,
        "afrr_net_revenue_hourly_eur": corrected_afrr_net_revenue_qh,
        "afrr_energy_down_activated_hourly": down_activated_qh,
        "afrr_energy_up_activated_hourly": up_activated_qh,
        "stored_energy_cost_eur_per_mwh": stored_energy_cost_qh,
        "effective_discharge_value_eur_per_mwh": effective_discharge_value_qh,
        "spread_condition_respected": spread_condition_respected_qh,
        "wholesale_discharge_spread_ok": wholesale_discharge_spread_ok_qh,
        "afrr_up_discharge_spread_ok": afrr_up_discharge_spread_ok_qh,
    }



def enforce_hard_annual_cycle_cap_on_reconciliation(
    reconciliation: Dict[str, np.ndarray],
    inputs: SimulationInputs,
    afrr_capacity_result: Dict[str, np.ndarray] | None = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Enforce max_cycles_per_year as a hard annual discharge budget.

    The existing DP limits wholesale discharge, but final reconciliation can add
    aFRR UP discharge on top. This pass ranks all final discharge candidates
    (wholesale and aFRR UP) by net EUR/MWh value, keeps the best candidates
    within the annual discharge cap, clips the marginal interval if needed, and
    rejects the rest. It then recomputes revenues, SOC and audit columns.
    """
    if reconciliation is None:
        return reconciliation, {"wholesale_rejected": 0, "afrr_rejected": 0}

    out: Dict[str, np.ndarray] = {}
    for key, value in reconciliation.items():
        if isinstance(value, np.ndarray):
            out[key] = value.copy()
        else:
            out[key] = value

    wh = _validate_array_length(out.get("wholesale_discharge_qh_mwh", np.zeros(QH_PER_YEAR)), "Wholesale discharge for cycle cap")
    afrr = _validate_array_length(out.get("afrr_discharge_qh_mwh", np.zeros(QH_PER_YEAR)), "aFRR discharge for cycle cap")
    wh_before = wh.copy()
    afrr_before = afrr.copy()

    annual_cap = max(float(inputs.max_cycles_per_year), 0.0) * float(inputs.batt_energy_mwh)
    if annual_cap <= 1e-12:
        keep_wh = np.zeros(QH_PER_YEAR, dtype=float)
        keep_afrr = np.zeros(QH_PER_YEAR, dtype=float)
    else:
        batt_sell = _validate_array_length(inputs.batt_sell_price, "BESS sell price for cycle cap")
        afrr_sell = _validate_array_length(inputs.afrr_discharge_price_qh if inputs.afrr_discharge_price_qh is not None else np.zeros(QH_PER_YEAR), "aFRR UP price for cycle cap")
        stored_cost = np.nan_to_num(
            np.asarray(out.get("stored_energy_cost_eur_per_mwh", np.zeros(QH_PER_YEAR)), dtype=float).reshape(-1)[:QH_PER_YEAR],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if len(stored_cost) != QH_PER_YEAR:
            stored_cost = np.zeros(QH_PER_YEAR, dtype=float)

        expected_up = np.zeros(QH_PER_YEAR, dtype=float)
        expected_cap_rev = np.zeros(QH_PER_YEAR, dtype=float)
        if afrr_capacity_result is not None:
            expected_up = np.asarray(afrr_capacity_result.get("expected_up_activated_mwh", np.zeros(QH_PER_YEAR)), dtype=float).reshape(-1)
            expected_cap_rev = np.asarray(afrr_capacity_result.get("expected_up_capacity_revenue_eur", np.zeros(QH_PER_YEAR)), dtype=float).reshape(-1)
            if len(expected_up) != QH_PER_YEAR:
                expected_up = np.zeros(QH_PER_YEAR, dtype=float)
            if len(expected_cap_rev) != QH_PER_YEAR:
                expected_cap_rev = np.zeros(QH_PER_YEAR, dtype=float)
        cap_value_per_activated_mwh = np.divide(
            expected_cap_rev,
            np.maximum(expected_up, 1e-12),
            out=np.zeros(QH_PER_YEAR, dtype=float),
            where=expected_up > 1e-12,
        )

        cost_per_output = stored_cost / max(inputs.eta_discharge, 1e-12)
        wh_value = batt_sell - cost_per_output - (float(inputs.cycle_cost_eur_per_mwh) / max(inputs.eta_discharge, 1e-12))
        afrr_value = afrr_sell + cap_value_per_activated_mwh - cost_per_output - (float(inputs.afrr_cycle_cost_eur_per_mwh) / max(inputs.eta_discharge, 1e-12))

        candidates: list[tuple[float, int, str, float]] = []
        for t in range(QH_PER_YEAR):
            if wh[t] > 1e-12:
                candidates.append((float(wh_value[t]), t, "wholesale", float(wh[t])))
            if afrr[t] > 1e-12:
                candidates.append((float(afrr_value[t]), t, "afrr", float(afrr[t])))

        # Highest value first; stable tie-break by time for deterministic results.
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        keep_wh = np.zeros(QH_PER_YEAR, dtype=float)
        keep_afrr = np.zeros(QH_PER_YEAR, dtype=float)
        remaining = float(annual_cap)
        for value, t, market, qty in candidates:
            if remaining <= 1e-9:
                break
            kept = min(qty, remaining)
            if kept <= 1e-12:
                continue
            if market == "wholesale":
                keep_wh[t] += kept
            else:
                keep_afrr[t] += kept
            remaining -= kept

    wh_rejected = np.maximum(wh_before - keep_wh, 0.0)
    afrr_rejected = np.maximum(afrr_before - keep_afrr, 0.0)

    out["wholesale_discharge_qh_mwh"] = keep_wh
    out["wholesale_discharge_hourly_mwh"] = keep_wh
    out["afrr_discharge_qh_mwh"] = keep_afrr
    out["afrr_discharge_hourly_mwh"] = keep_afrr

    # Remove aFRR UP activation flags where no aFRR discharge remains.
    if "afrr_energy_up_activated_qh" in out:
        out["afrr_energy_up_activated_qh"] = (keep_afrr > 1e-12).astype(int)
    if "afrr_energy_up_activated_hourly" in out:
        out["afrr_energy_up_activated_hourly"] = (keep_afrr > 1e-12).astype(int)

    # Recompute revenues and SOC after the budget cut. Charges are clipped by
    # headroom during this SOC replay, so the final trajectory remains feasible.
    batt_sell = _validate_array_length(inputs.batt_sell_price, "BESS sell price for post cap")
    grid_buy = _validate_array_length(inputs.grid_buy_price, "Grid buy price for post cap")
    afrr_charge_price = _validate_array_length(inputs.afrr_charge_price_qh if inputs.afrr_charge_price_qh is not None else np.zeros(QH_PER_YEAR), "aFRR DOWN price for post cap")
    afrr_discharge_price = _validate_array_length(inputs.afrr_discharge_price_qh if inputs.afrr_discharge_price_qh is not None else np.zeros(QH_PER_YEAR), "aFRR UP price for post cap")

    pv_to_batt = _validate_array_length(out.get("wholesale_pv_to_batt_qh_mwh", np.zeros(QH_PER_YEAR)), "PV to battery post cap")
    pv_curt_to_batt = _validate_array_length(out.get("wholesale_pv_curtailed_to_batt_qh_mwh", np.zeros(QH_PER_YEAR)), "Curtailed PV to battery post cap")
    grid_charge = _validate_array_length(out.get("wholesale_grid_charge_qh_mwh", np.zeros(QH_PER_YEAR)), "Grid charge post cap")
    afrr_charge = _validate_array_length(out.get("afrr_charge_qh_mwh", np.zeros(QH_PER_YEAR)), "aFRR charge post cap")

    min_soc = float(inputs.batt_energy_mwh) * float(inputs.min_soc_pct) / 100.0
    max_soc = float(inputs.batt_energy_mwh) * float(inputs.max_soc_pct) / 100.0
    soc = np.zeros(QH_PER_YEAR + 1, dtype=float)
    soc[0] = min(max(float(inputs.initial_soc_mwh), min_soc), max_soc)

    for t in range(QH_PER_YEAR):
        total_charge_input = pv_to_batt[t] + pv_curt_to_batt[t] + grid_charge[t] + afrr_charge[t]
        max_charge_input = max(max_soc - soc[t], 0.0) / max(inputs.eta_charge, 1e-12)
        if total_charge_input > max_charge_input + 1e-12:
            scale = max_charge_input / max(total_charge_input, 1e-12)
            pv_to_batt[t] *= scale
            pv_curt_to_batt[t] *= scale
            grid_charge[t] *= scale
            afrr_charge[t] *= scale
            total_charge_input = max_charge_input
        total_discharge = keep_wh[t] + keep_afrr[t]
        max_discharge = max(soc[t] - min_soc, 0.0) * max(inputs.eta_discharge, 1e-12)
        if total_discharge > max_discharge + 1e-12:
            scale = max_discharge / max(total_discharge, 1e-12)
            keep_wh[t] *= scale
            keep_afrr[t] *= scale
            total_discharge = max_discharge
        soc[t + 1] = min(max(soc[t] + total_charge_input * inputs.eta_charge - total_discharge / max(inputs.eta_discharge, 1e-12), min_soc), max_soc)

    out["wholesale_pv_to_batt_qh_mwh"] = pv_to_batt
    out["wholesale_pv_to_batt_hourly_mwh"] = pv_to_batt
    out["wholesale_pv_curtailed_to_batt_qh_mwh"] = pv_curt_to_batt
    out["wholesale_pv_curtailed_to_batt_hourly_mwh"] = pv_curt_to_batt
    out["wholesale_grid_charge_qh_mwh"] = grid_charge
    out["wholesale_grid_charge_hourly_mwh"] = grid_charge
    out["afrr_charge_qh_mwh"] = afrr_charge
    out["afrr_charge_hourly_mwh"] = afrr_charge
    out["wholesale_discharge_qh_mwh"] = keep_wh
    out["wholesale_discharge_hourly_mwh"] = keep_wh
    out["afrr_discharge_qh_mwh"] = keep_afrr
    out["afrr_discharge_hourly_mwh"] = keep_afrr

    out["combined_soc_qh"] = soc
    out["combined_soc_hourly_end_mwh"] = soc[1:]
    out["combined_charge_to_soc_qh_mwh"] = (pv_to_batt + pv_curt_to_batt + grid_charge + afrr_charge) * inputs.eta_charge
    out["combined_discharge_from_soc_qh_mwh"] = (keep_wh + keep_afrr) / max(inputs.eta_discharge, 1e-12)

    out["wholesale_batt_sale_revenue_qh_eur"] = keep_wh * batt_sell
    out["wholesale_batt_sale_revenue_hourly_eur"] = out["wholesale_batt_sale_revenue_qh_eur"]
    out["wholesale_grid_charge_cost_qh_eur"] = grid_charge * grid_buy
    out["wholesale_grid_charge_cost_hourly_eur"] = out["wholesale_grid_charge_cost_qh_eur"]
    out["afrr_charge_cost_qh_eur"] = afrr_charge * afrr_charge_price
    out["afrr_charge_cost_hourly_eur"] = out["afrr_charge_cost_qh_eur"]
    out["afrr_sale_revenue_qh_eur"] = keep_afrr * afrr_discharge_price
    out["afrr_sale_revenue_hourly_eur"] = out["afrr_sale_revenue_qh_eur"]
    out["afrr_cycle_cost_qh_eur"] = keep_afrr / max(inputs.eta_discharge, 1e-12) * inputs.afrr_cycle_cost_eur_per_mwh
    out["afrr_cycle_cost_hourly_eur"] = out["afrr_cycle_cost_qh_eur"]
    out["afrr_net_revenue_qh_eur"] = out["afrr_sale_revenue_qh_eur"] - out["afrr_charge_cost_qh_eur"]
    out["afrr_net_revenue_hourly_eur"] = out["afrr_net_revenue_qh_eur"]

    combined_discharge = keep_wh + keep_afrr
    cumulative = np.cumsum(combined_discharge)
    remaining = np.maximum(annual_cap - cumulative, 0.0)
    out["annual_discharge_cap_mwh"] = np.full(QH_PER_YEAR, annual_cap, dtype=float)
    out["cumulative_battery_discharge_mwh"] = cumulative
    out["remaining_discharge_budget_mwh"] = remaining
    out["cycle_budget_used_pct"] = np.divide(cumulative, max(annual_cap, 1e-12), out=np.zeros(QH_PER_YEAR), where=annual_cap > 1e-12) * 100.0
    out["cycle_budget_available_flag"] = (remaining > 1e-9).astype(int)
    out["wholesale_discharge_rejected_due_to_cycle_budget"] = (wh_rejected > 1e-9).astype(int)
    out["afrr_up_discharge_rejected_due_to_cycle_budget"] = (afrr_rejected > 1e-9).astype(int)
    out["afrr_up_capacity_rejected_due_to_cycle_budget"] = (afrr_rejected > 1e-9).astype(int)
    out["discharge_rejected_due_to_cycle_budget"] = ((wh_rejected + afrr_rejected) > 1e-9).astype(int)

    # Ranking audit: 1 is highest net value among selected/rejected discharge candidates.
    net_value = np.zeros(QH_PER_YEAR, dtype=float)
    rank = np.zeros(QH_PER_YEAR, dtype=int)
    try:
        stored_cost = np.nan_to_num(np.asarray(out.get("stored_energy_cost_eur_per_mwh", np.zeros(QH_PER_YEAR)), dtype=float).reshape(-1)[:QH_PER_YEAR], nan=0.0)
        cost_per_output = stored_cost / max(inputs.eta_discharge, 1e-12)
        wh_value = batt_sell - cost_per_output - inputs.cycle_cost_eur_per_mwh / max(inputs.eta_discharge, 1e-12)
        afrr_value = afrr_discharge_price - cost_per_output - inputs.afrr_cycle_cost_eur_per_mwh / max(inputs.eta_discharge, 1e-12)
        net_value = np.maximum(np.where((keep_wh + wh_rejected) > 1e-12, wh_value, -1e30), np.where((keep_afrr + afrr_rejected) > 1e-12, afrr_value, -1e30))
        candidate_idx = np.where(net_value > -1e20)[0]
        order = candidate_idx[np.argsort(-net_value[candidate_idx])]
        rank[order] = np.arange(1, len(order) + 1)
        net_value[net_value <= -1e20] = 0.0
    except Exception:
        net_value = np.zeros(QH_PER_YEAR, dtype=float)
        rank = np.zeros(QH_PER_YEAR, dtype=int)
    out["net_dispatch_value_eur_per_mwh"] = net_value
    out["cycle_budget_rank"] = rank

    return out, {
        "wholesale_rejected": int(np.sum(wh_rejected > 1e-9)),
        "afrr_rejected": int(np.sum(afrr_rejected > 1e-9)),
    }

def build_final_result_after_market_arbitration(
    base_result: Dict[str, np.ndarray],
    reconciliation: Dict[str, np.ndarray],
    inputs: SimulationInputs,
) -> Dict[str, np.ndarray]:
    final = dict(base_result)

    final["pv_to_batt"] = reconciliation["wholesale_pv_to_batt_hourly_mwh"]
    final["grid_charge"] = reconciliation["wholesale_grid_charge_hourly_mwh"]
    final["pv_curtailed_to_battery"] = reconciliation[
        "wholesale_pv_curtailed_to_batt_hourly_mwh"
    ]
    final["discharge"] = reconciliation["wholesale_discharge_hourly_mwh"]
    final["batt_sale_revenue"] = reconciliation["wholesale_batt_sale_revenue_hourly_eur"]
    final["grid_charge_cost"] = reconciliation["wholesale_grid_charge_cost_hourly_eur"]
    # Option A: cycle cost is kept as a theoretical degradation metric only; it is not deducted from cash revenue.
    final["wholesale_cycle_cost_eur"] = final["discharge"] * inputs.cycle_cost_eur_per_mwh

    total_batt_sale_revenue = float(final["batt_sale_revenue"].sum())
    total_grid_charge_cost = float(final["grid_charge_cost"].sum())
    total_wholesale_cycle_cost = float(final["wholesale_cycle_cost_eur"].sum())
    total_direct_pv_revenue = float(final["pv_direct_revenue"].sum())
    nightly_revenue_total = float(final["nightly_revenue_total"][0])

    total_discharged_mwh = float(final["discharge"].sum() + reconciliation["afrr_discharge_hourly_mwh"].sum())
    annual_discharge_cap_mwh = float(inputs.max_cycles_per_year) * float(inputs.batt_energy_mwh)

    final["total_batt_sale_revenue"] = np.array([total_batt_sale_revenue])
    final["total_grid_charge_cost"] = np.array([total_grid_charge_cost])
    final["total_wholesale_cycle_cost_eur"] = np.array([total_wholesale_cycle_cost])
    final["gross_bess_revenue_before_cycle_cost_eur"] = np.array([total_batt_sale_revenue])
    final["net_bess_revenue_after_cycle_cost_eur"] = np.array([
        total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total
    ])
    final["bess_cash_revenue_eur"] = np.array([
        total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total
    ])
    final["energy_shifted_mwh"] = np.array([total_discharged_mwh])
    final["energy_sold_total_mwh"] = np.array([float(final["pv_direct"].sum() + total_discharged_mwh)])
    final["equivalent_cycles"] = np.array([float(total_discharged_mwh / max(inputs.batt_energy_mwh, 1e-12))])
    final["max_cycles_per_year"] = np.array([float(inputs.max_cycles_per_year)])
    final["annual_discharge_cap_mwh"] = np.array([annual_discharge_cap_mwh])
    final["remaining_cycle_budget_mwh"] = np.array([max(annual_discharge_cap_mwh - total_discharged_mwh, 0.0)])
    if float(final["equivalent_cycles"][0]) > float(inputs.max_cycles_per_year) + 1e-6:
        raise RuntimeError(
            "Annual cycle cap exceeded after final dispatch reconciliation: "
            f"{float(final['equivalent_cycles'][0]):.6f} cycles > {float(inputs.max_cycles_per_year):.6f} cycles."
        )

    final["total_revenue"] = np.array([
        total_direct_pv_revenue + total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total
    ])

    final["afrr_charge_hourly_mwh"] = reconciliation["afrr_charge_hourly_mwh"]
    final["afrr_discharge_hourly_mwh"] = reconciliation["afrr_discharge_hourly_mwh"]
    final["afrr_charge_cost_hourly_eur"] = reconciliation["afrr_charge_cost_hourly_eur"]
    final["afrr_sale_revenue_hourly_eur"] = reconciliation["afrr_sale_revenue_hourly_eur"]
    final["afrr_cycle_cost_hourly_eur"] = reconciliation["afrr_cycle_cost_hourly_eur"]
    final["afrr_net_revenue_hourly_eur"] = final["afrr_sale_revenue_hourly_eur"] - final["afrr_charge_cost_hourly_eur"]
    for _audit_key in [
        "stored_energy_cost_eur_per_mwh",
        "effective_discharge_value_eur_per_mwh",
        "spread_condition_respected",
        "wholesale_discharge_spread_ok",
        "afrr_up_discharge_spread_ok",
        "annual_discharge_cap_mwh",
        "cumulative_battery_discharge_mwh",
        "remaining_discharge_budget_mwh",
        "cycle_budget_used_pct",
        "cycle_budget_available_flag",
        "discharge_rejected_due_to_cycle_budget",
        "wholesale_discharge_rejected_due_to_cycle_budget",
        "afrr_up_discharge_rejected_due_to_cycle_budget",
        "afrr_up_capacity_rejected_due_to_cycle_budget",
        "net_dispatch_value_eur_per_mwh",
        "cycle_budget_rank",
    ]:
        if _audit_key in reconciliation:
            final[_audit_key] = reconciliation[_audit_key]

    final["total_afrr_charge_cost_eur"] = np.array([float(reconciliation["afrr_charge_cost_hourly_eur"].sum())])
    final["total_afrr_sale_revenue_eur"] = np.array([float(reconciliation["afrr_sale_revenue_hourly_eur"].sum())])
    final["total_afrr_cycle_cost_eur"] = np.array([float(reconciliation["afrr_cycle_cost_hourly_eur"].sum())])
    final["total_afrr_net_revenue_eur"] = np.array([float(final["afrr_net_revenue_hourly_eur"].sum())])

    final["total_battery_revenue_including_afrr_eur"] = np.array([
        total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total + float(final["afrr_net_revenue_hourly_eur"].sum())
    ])

    final["total_revenue_including_afrr_eur"] = np.array([
        total_direct_pv_revenue + total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total + float(final["afrr_net_revenue_hourly_eur"].sum())
    ])

    return final


def add_afrr_capacity_to_final_result(
    result: Dict[str, np.ndarray],
    afrr_capacity_result: Dict[str, np.ndarray] | None,
) -> Dict[str, np.ndarray]:
    """Attach aFRR Capacity hourly arrays and revenue totals to the result dict."""
    final = dict(result)

    if afrr_capacity_result is None:
        up_revenue = np.zeros(QH_PER_YEAR, dtype=float)
        down_revenue = np.zeros(QH_PER_YEAR, dtype=float)
        total_revenue_h = np.zeros(QH_PER_YEAR, dtype=float)
        up_awarded = np.zeros(QH_PER_YEAR, dtype=int)
        down_awarded = np.zeros(QH_PER_YEAR, dtype=int)
        selected_market = np.full(QH_PER_YEAR, "none", dtype=object)
        certified_up_h = np.zeros(QH_PER_YEAR, dtype=float)
        certified_down_h = np.zeros(QH_PER_YEAR, dtype=float)
        eligible_h = np.zeros(QH_PER_YEAR, dtype=int)
    else:
        up_revenue = np.asarray(afrr_capacity_result["afrr_capacity_up_revenue_h_eur"], dtype=float)
        down_revenue = np.asarray(afrr_capacity_result["afrr_capacity_down_revenue_h_eur"], dtype=float)
        total_revenue_h = np.asarray(afrr_capacity_result["afrr_capacity_total_revenue_h_eur"], dtype=float)
        up_awarded = np.asarray(afrr_capacity_result["afrr_capacity_up_awarded_h"], dtype=int)
        down_awarded = np.asarray(afrr_capacity_result["afrr_capacity_down_awarded_h"], dtype=int)
        selected_market = np.asarray(afrr_capacity_result["afrr_capacity_selected_market_h"], dtype=object)
        certified_up_h = np.asarray(afrr_capacity_result["afrr_certified_capacity_up_mw_h"], dtype=float)
        certified_down_h = np.asarray(afrr_capacity_result["afrr_certified_capacity_down_mw_h"], dtype=float)
        eligible_h = np.asarray(afrr_capacity_result["afrr_capacity_eligible_h"], dtype=int)

    cap_up_total = float(up_revenue.sum())
    cap_down_total = float(down_revenue.sum())
    cap_total = float(total_revenue_h.sum())

    afrr_energy_net = float(final["total_afrr_net_revenue_eur"][0]) if "total_afrr_net_revenue_eur" in final else 0.0
    base_battery_revenue = (
        float(final["total_batt_sale_revenue"][0])
        - float(final["total_grid_charge_cost"][0])
        + float(final["nightly_revenue_total"][0])
    )
    total_direct_pv_revenue = float(final["total_direct_pv_revenue"][0])

    final["afrr_capacity_up_revenue_h_eur"] = up_revenue
    final["afrr_capacity_down_revenue_h_eur"] = down_revenue
    final["afrr_capacity_total_revenue_h_eur"] = total_revenue_h
    final["afrr_capacity_up_awarded_h"] = up_awarded
    final["afrr_capacity_down_awarded_h"] = down_awarded
    final["afrr_capacity_selected_market_h"] = selected_market
    final["afrr_capacity_eligible_h"] = eligible_h
    final["afrr_certified_capacity_up_mw_h"] = certified_up_h
    final["afrr_certified_capacity_down_mw_h"] = certified_down_h

    if afrr_capacity_result is not None:
        for _k in [
            "wholesale_opportunity_value_eur",
            "wholesale_expected_value_after_capture_rate_eur",
            "raw_up_capacity_revenue_eur",
            "expected_up_capacity_revenue_eur",
            "raw_down_capacity_revenue_eur",
            "expected_down_capacity_revenue_eur",
            "expected_up_activated_mwh",
            "expected_down_activated_mwh",
            "afrr_up_energy_expected_value_eur",
            "afrr_down_energy_expected_value_eur",
            "afrr_up_total_expected_value_eur",
            "afrr_down_total_expected_value_eur",
            "selected_market",
            "selected_capacity_direction",
            "afrr_capacity_success_rate_pct",
            "bess_wholesale_capture_rate_pct",
            "afrr_up_activation_pct",
            "afrr_down_activation_pct",
            "available_export_headroom_mwh",
            "available_soc_headroom_mwh",
            "available_discharge_from_soc_mwh",
            "required_up_soc_reserve_mwh",
            "required_down_soc_headroom_mwh",
            "expected_degradation_cost_eur",
            "future_best_market_value_eur_per_mwh",
            "future_best_market_type",
            "cross_market_spread_eur_per_mwh",
            "required_min_spread_eur_per_mwh",
            "spread_condition_respected",
            "charge_reason",
            "discharge_reason",
            "stored_energy_cost_eur_per_mwh",
            "effective_discharge_value_eur_per_mwh",
            "future_expected_afrr_up_value_eur",
            "future_expected_wholesale_value_eur",
            "future_expected_best_discharge_market",
            "wholesale_charge_for_future_afrr_flag",
            "afrr_down_charge_for_future_wholesale_flag",
            "afrr_down_charge_for_future_afrr_up_flag",
            "wholesale_discharge_spread_ok",
            "afrr_up_discharge_spread_ok",
            "forward_horizon_hours",
            "future_opportunity_selected",
            "forward_soc_before_capacity_selection_mwh",
            "forward_soc_after_capacity_selection_mwh",
            "afrr_up_soc_feasible",
            "afrr_down_soc_feasible",
            "afrr_up_rejected_due_to_soc",
            "afrr_down_rejected_due_to_soc",
            "afrr_up_expected_vs_actual_shortfall_mwh",
            "afrr_down_expected_vs_actual_shortfall_mwh",
            "afrr_up_rejected_due_to_final_combined_soc",
            "afrr_down_rejected_due_to_final_combined_soc",
        ]:
            final[_k] = np.asarray(afrr_capacity_result.get(_k, np.full(QH_PER_YEAR, "none", dtype=object) if _k in ("selected_market", "selected_capacity_direction", "future_best_market_type", "charge_reason", "discharge_reason", "future_expected_best_discharge_market") else np.zeros(QH_PER_YEAR)), dtype=object if _k in ("selected_market", "selected_capacity_direction", "future_best_market_type", "charge_reason", "discharge_reason", "future_expected_best_discharge_market") else float)

    final["total_afrr_capacity_up_revenue_eur"] = np.array([cap_up_total])
    final["total_afrr_capacity_down_revenue_eur"] = np.array([cap_down_total])
    final["total_afrr_capacity_revenue_eur"] = np.array([cap_total])

    final["total_battery_revenue_including_afrr_capacity_eur"] = np.array([
        base_battery_revenue + afrr_energy_net + cap_total
    ])
    final["total_revenue_including_afrr_capacity_eur"] = np.array([
        total_direct_pv_revenue + base_battery_revenue + afrr_energy_net + cap_total
    ])

    return final


def build_summary_table(
    result: Dict[str, np.ndarray],
    pv_stats: Dict[str, float],
    pure_pv_benchmark: Dict[str, np.ndarray],
    pv_dc_mw: float,
    batt_power_mw: float,
    pv_capture_rate_pct: float,
    bess_capture_rate_pct: float,
    curtailment_outputs: Dict[str, np.ndarray],
) -> pd.DataFrame:
    pv_revenue = float(result["total_direct_pv_revenue"][0])

    wholesale_cycle_cost = float(result["total_wholesale_cycle_cost_eur"][0]) if "total_wholesale_cycle_cost_eur" in result else 0.0
    bess_revenue_base = (
        float(result["total_batt_sale_revenue"][0])
        - float(result["total_grid_charge_cost"][0])
        + float(result["nightly_revenue_total"][0])
    )

    afrr_net_revenue = float(result["total_afrr_net_revenue_eur"][0]) if "total_afrr_net_revenue_eur" in result else 0.0
    afrr_sale_revenue = float(result["total_afrr_sale_revenue_eur"][0]) if "total_afrr_sale_revenue_eur" in result else 0.0
    afrr_charge_cost = float(result["total_afrr_charge_cost_eur"][0]) if "total_afrr_charge_cost_eur" in result else 0.0
    afrr_cycle_cost = float(result["total_afrr_cycle_cost_eur"][0]) if "total_afrr_cycle_cost_eur" in result else 0.0
    total_cycle_cost = wholesale_cycle_cost + afrr_cycle_cost

    afrr_capacity_up_revenue = float(result["total_afrr_capacity_up_revenue_eur"][0]) if "total_afrr_capacity_up_revenue_eur" in result else 0.0
    afrr_capacity_down_revenue = float(result["total_afrr_capacity_down_revenue_eur"][0]) if "total_afrr_capacity_down_revenue_eur" in result else 0.0
    afrr_capacity_revenue = float(result["total_afrr_capacity_revenue_eur"][0]) if "total_afrr_capacity_revenue_eur" in result else 0.0
    certified_up_mw = float(np.nanmax(result["afrr_certified_capacity_up_mw_h"])) if "afrr_certified_capacity_up_mw_h" in result and len(result["afrr_certified_capacity_up_mw_h"]) else 0.0
    certified_down_mw = float(np.nanmax(result["afrr_certified_capacity_down_mw_h"])) if "afrr_certified_capacity_down_mw_h" in result and len(result["afrr_certified_capacity_down_mw_h"]) else 0.0
    capacity_up_hours = int(np.sum(result["afrr_capacity_up_awarded_h"])) if "afrr_capacity_up_awarded_h" in result else 0
    capacity_down_hours = int(np.sum(result["afrr_capacity_down_awarded_h"])) if "afrr_capacity_down_awarded_h" in result else 0

    bess_revenue_total = bess_revenue_base + afrr_net_revenue + afrr_capacity_revenue
    if "total_revenue_including_afrr_capacity_eur" in result:
        total_revenue = float(result["total_revenue_including_afrr_capacity_eur"][0])
    elif "total_revenue_including_afrr_eur" in result:
        total_revenue = float(result["total_revenue_including_afrr_eur"][0])
    else:
        total_revenue = float(result["total_revenue"][0])

    pure_pv_revenue = float(pure_pv_benchmark["total_pv_only_revenue_eur"][0])
    hybrid_added_value = total_revenue - pure_pv_revenue

    pv_rev_keur_per_mw = pv_revenue / max(pv_dc_mw, 1e-12) / 1000.0
    bess_rev_keur_per_mw = bess_revenue_total / max(batt_power_mw, 1e-12) / 1000.0

    pv_sold_mwh = float(result["pv_direct_sold_mwh"][0])
    bess_sold_mwh = float(result["energy_shifted_mwh"][0])

    afrr_discharged_mwh = float(np.sum(result["afrr_discharge_hourly_mwh"])) if "afrr_discharge_hourly_mwh" in result else 0.0
    bess_grid_charged_mwh = float(np.sum(result["grid_charge"])) if "grid_charge" in result else 0.0
    afrr_charged_mwh = float(np.sum(result["afrr_charge_hourly_mwh"])) if "afrr_charge_hourly_mwh" in result else 0.0
    bess_total_discharged_mwh = bess_sold_mwh + afrr_discharged_mwh
    bess_total_charged_mwh = bess_grid_charged_mwh + afrr_charged_mwh
    bess_total_throughput_mwh = bess_total_charged_mwh + bess_total_discharged_mwh

    pv_rev_eur_per_mwh = pv_revenue / max(pv_sold_mwh, 1e-12)
    bess_rev_eur_per_mwh = bess_revenue_total / max(bess_total_discharged_mwh, 1e-12)

    tso_dso_curtailed = float(np.sum(curtailment_outputs["tso_dso_curtailed_mwh"]))
    self_curtailed = float(np.sum(curtailment_outputs["self_curtailed_mwh"]))
    candidate_curtailed = float(np.sum(curtailment_outputs["pv_curtailment_candidate_mwh"]))
    recovered_to_battery = float(np.sum(curtailment_outputs["pv_curtailed_to_battery_mwh_actual"]))
    residual_lost = float(np.sum(curtailment_outputs["pv_curtailed_residual_lost_mwh"]))
    max_cycles_per_year = float(result["max_cycles_per_year"][0]) if "max_cycles_per_year" in result else np.nan
    annual_discharge_cap_mwh = float(result["annual_discharge_cap_mwh"][0]) if "annual_discharge_cap_mwh" in result else np.nan
    remaining_cycle_budget_mwh = float(result["remaining_cycle_budget_mwh"][0]) if "remaining_cycle_budget_mwh" in result else np.nan
    avg_raw_bess_sell_price = float(result["avg_raw_bess_sell_price_eur_per_mwh"][0]) if "avg_raw_bess_sell_price_eur_per_mwh" in result else np.nan
    avg_effective_bess_sell_price = float(result["avg_effective_bess_sell_price_eur_per_mwh"][0]) if "avg_effective_bess_sell_price_eur_per_mwh" in result else np.nan
    revenue_loss_capture_rate = float(result["bess_revenue_loss_due_to_capture_rate_eur"][0]) if "bess_revenue_loss_due_to_capture_rate_eur" in result else np.nan
    gross_bess_revenue_before_cycle_cost = float(result["gross_bess_revenue_before_cycle_cost_eur"][0]) if "gross_bess_revenue_before_cycle_cost_eur" in result else float(result["total_batt_sale_revenue"][0])
    net_bess_revenue_after_cycle_cost = float(result["net_bess_revenue_after_cycle_cost_eur"][0]) if "net_bess_revenue_after_cycle_cost_eur" in result else bess_revenue_base
    avg_cycle_cost_per_discharged_mwh = total_cycle_cost / max(float(result["energy_shifted_mwh"][0]), 1e-12)
    cycles_without_cycle_cost = float(result["equivalent_cycles_without_cycle_cost"][0]) if "equivalent_cycles_without_cycle_cost" in result else np.nan
    cycles_with_cycle_cost = float(result["equivalent_cycles"][0]) if "equivalent_cycles" in result else np.nan

    rows = [
        ("PV Capture Rate", pv_capture_rate_pct, "%"),
        ("BESS Capture Rate", bess_capture_rate_pct, "%"),
        ("Average Raw BESS Sell Price", avg_raw_bess_sell_price, "€/MWh"),
        ("Average Effective BESS Sell Price", avg_effective_bess_sell_price, "€/MWh"),
        ("Revenue loss due to BESS capture rate", revenue_loss_capture_rate, "€"),
        ("Theoretical Cycle Cost (not deducted from cash revenue)", total_cycle_cost, "€"),
        ("Theoretical Cycle Cost per Year (not deducted)", total_cycle_cost, "€/an"),
        ("Gross BESS Revenue Before Cycle Cost", gross_bess_revenue_before_cycle_cost, "€"),
        ("BESS Cash Revenue (cycle cost not deducted)", net_bess_revenue_after_cycle_cost, "€"),
        ("Average Theoretical Cycle Cost per Discharged MWh", avg_cycle_cost_per_discharged_mwh, "€/MWh"),
        ("Number of cycles without cycle cost", cycles_without_cycle_cost, "cycles/an"),
        ("Number of cycles with cycle cost", cycles_with_cycle_cost, "cycles/an"),
        ("Revenu total", total_revenue, "€"),
        ("Revenu PV-only Project", pure_pv_revenue, "€"),
        ("Valeur ajoutée de l'hybridation vs PV-only", hybrid_added_value, "€"),
        ("Revenu PV direct", pv_revenue, "€"),
        ("Revenu batterie wholesale", float(result["total_batt_sale_revenue"][0]), "€"),
        ("Coût charge réseau wholesale", float(result["total_grid_charge_cost"][0]), "€"),
        ("Coût cycle wholesale théorique (non déduit)", wholesale_cycle_cost, "€"),
        ("Revenu services système de nuit", float(result["nightly_revenue_total"][0]), "€"),
        ("Revenu brut aFRR", afrr_sale_revenue, "€"),
        ("Cashflow charge aFRR", afrr_charge_cost, "€"),
        ("Coût cycle aFRR théorique (non déduit)", afrr_cycle_cost, "€"),
        ("Revenu net aFRR", afrr_net_revenue, "€"),
        ("Revenu aFRR Capacity UP", afrr_capacity_up_revenue, "€"),
        ("Revenu aFRR Capacity Down", afrr_capacity_down_revenue, "€"),
        ("Revenu total aFRR Capacity", afrr_capacity_revenue, "€"),
        ("Certified Capacity UP MW", certified_up_mw, "MW"),
        ("Certified Capacity Down MW", certified_down_mw, "MW"),
        ("Number of hours awarded UP", capacity_up_hours, "h"),
        ("Number of hours awarded Down", capacity_down_hours, "h"),
        ("TSO/DSO curtailed energy", tso_dso_curtailed, "MWh"),
        ("Self-curtailed energy", self_curtailed, "MWh"),
        ("Total curtailed PV candidate energy", candidate_curtailed, "MWh"),
        ("Curtailed PV recovered by battery", recovered_to_battery, "MWh"),
        ("Residual curtailed PV energy lost", residual_lost, "MWh"),
        ("Revenu PV spécifique", pv_rev_keur_per_mw, "k€/MW"),
        ("Revenu BESS spécifique", bess_rev_keur_per_mw, "k€/MW"),
        ("Revenu PV spécifique énergie", pv_rev_eur_per_mwh, "€/MWh"),
        ("Revenu BESS spécifique énergie", bess_rev_eur_per_mwh, "€/MWh"),
        ("Énergie totale vendue", float(result["energy_sold_total_mwh"][0]) + afrr_discharged_mwh, "MWh"),
        ("Énergie shiftée wholesale", bess_sold_mwh, "MWh"),
        ("Énergie déchargée aFRR", afrr_discharged_mwh, "MWh"),
        ("Énergie chargée BESS depuis réseau", bess_grid_charged_mwh, "MWh"),
        ("Énergie chargée aFRR", afrr_charged_mwh, "MWh"),
        ("Total BESS throughput", bess_total_throughput_mwh, "MWh"),
        ("Énergie PV vendue directement", pv_sold_mwh, "MWh"),
        ("Cycles équivalents batterie", float(result["equivalent_cycles"][0]), "cycles/an"),
        ("Cycles max / an", max_cycles_per_year, "cycles/an"),
        ("Annual discharge cap MWh", annual_discharge_cap_mwh, "MWh"),
        ("Remaining cycle budget", remaining_cycle_budget_mwh, "MWh"),
        ("Production PV théorique brute", float(pv_stats["annual_dc_mwh"]), "MWh"),
        ("Production PV nette valorisable", float(pv_stats["annual_net_mwh"]), "MWh"),
        ("Énergie PV perdue (pertes + disponibilité)", float(pv_stats["annual_losses_mwh"]), "MWh"),
    ]
    return pd.DataFrame(rows, columns=["Indicateur", "Valeur", "Unité"])


def format_synthese_number(value):
    """Format Synthèse numeric values with French-style space thousands separators.

    Rules:
    - Values with absolute value >= 1 000 are rounded to the nearest whole number
      and displayed without decimal places.
    - Whole numbers below 1 000 are displayed without decimal places.
    - Non-whole decimal numbers below 1 000 are displayed with max 1 decimal place.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric_value = float(value)

        if abs(numeric_value) >= 1000:
            return f"{int(round(numeric_value)):,}".replace(",", " ")

        if abs(numeric_value - round(numeric_value)) < 1e-9:
            return f"{int(round(numeric_value))}"

        return f"{numeric_value:.1f}"

    return value


def format_synthese_table_for_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only copy of the Synthèse table with formatted numeric values."""
    display_df = summary_df.copy()
    if "Valeur" in display_df.columns:
        display_df["Valeur"] = display_df["Valeur"].apply(format_synthese_number)
    return display_df


def monthly_dataframe(
    result: Dict[str, np.ndarray],
    pure_pv_benchmark: Dict[str, np.ndarray],
    pv_dc_mw: float,
    batt_power_mw: float,
    curtailment_outputs: Dict[str, np.ndarray],
) -> pd.DataFrame:
    idx = build_quarter_hour_index(DEFAULT_YEAR)

    df = pd.DataFrame({
        "datetime": idx,
        "pv_direct_revenue": result["pv_direct_revenue"],
        "batt_sale_revenue": result["batt_sale_revenue"],
        "grid_charge_cost": result["grid_charge_cost"],
        "wholesale_cycle_cost": result["wholesale_cycle_cost_eur"] if "wholesale_cycle_cost_eur" in result else np.zeros(QH_PER_YEAR),
        "pv_direct_mwh": result["pv_direct"],
        "shifted_mwh": result["discharge"],
        "grid_charge_mwh": result["grid_charge"],
        "pv_to_batt_mwh": result["pv_to_batt"],
        "pv_curtailed_to_battery_mwh_actual": curtailment_outputs["pv_curtailed_to_battery_mwh_actual"],
        "pv_curtailment_candidate_mwh": curtailment_outputs["pv_curtailment_candidate_mwh"],
        "pv_curtailed_residual_lost_mwh": curtailment_outputs["pv_curtailed_residual_lost_mwh"],
        "pv_only_direct_mwh": pure_pv_benchmark["pv_only_direct_mwh"],
        "pv_only_revenue": pure_pv_benchmark["pv_only_revenue_eur"],
        "afrr_charge_mwh": result["afrr_charge_hourly_mwh"] if "afrr_charge_hourly_mwh" in result else np.zeros(QH_PER_YEAR),
        "afrr_discharge_mwh": result["afrr_discharge_hourly_mwh"] if "afrr_discharge_hourly_mwh" in result else np.zeros(QH_PER_YEAR),
        "afrr_charge_cost": result["afrr_charge_cost_hourly_eur"] if "afrr_charge_cost_hourly_eur" in result else np.zeros(QH_PER_YEAR),
        "afrr_sale_revenue": result["afrr_sale_revenue_hourly_eur"] if "afrr_sale_revenue_hourly_eur" in result else np.zeros(QH_PER_YEAR),
        "afrr_cycle_cost": result["afrr_cycle_cost_hourly_eur"] if "afrr_cycle_cost_hourly_eur" in result else np.zeros(QH_PER_YEAR),
        "afrr_net_revenue": result["afrr_net_revenue_hourly_eur"] if "afrr_net_revenue_hourly_eur" in result else np.zeros(QH_PER_YEAR),
        "afrr_capacity_up_revenue": result["afrr_capacity_up_revenue_h_eur"] if "afrr_capacity_up_revenue_h_eur" in result else np.zeros(QH_PER_YEAR),
        "afrr_capacity_down_revenue": result["afrr_capacity_down_revenue_h_eur"] if "afrr_capacity_down_revenue_h_eur" in result else np.zeros(QH_PER_YEAR),
        "afrr_capacity_total_revenue": result["afrr_capacity_total_revenue_h_eur"] if "afrr_capacity_total_revenue_h_eur" in result else np.zeros(QH_PER_YEAR),
        "afrr_capacity_up_awarded_hours": result["afrr_capacity_up_awarded_h"] if "afrr_capacity_up_awarded_h" in result else np.zeros(QH_PER_YEAR),
        "afrr_capacity_down_awarded_hours": result["afrr_capacity_down_awarded_h"] if "afrr_capacity_down_awarded_h" in result else np.zeros(QH_PER_YEAR),
        "bess_revenue_loss_due_to_capture_rate": result["bess_revenue_loss_due_to_capture_rate_hourly_eur"] if "bess_revenue_loss_due_to_capture_rate_hourly_eur" in result else np.zeros(QH_PER_YEAR),
        "bess_theoretical_revenue_without_capture": result["bess_theoretical_revenue_without_capture_hourly_eur"] if "bess_theoretical_revenue_without_capture_hourly_eur" in result else np.zeros(QH_PER_YEAR),
    })

    df["month"] = df["datetime"].dt.strftime("%Y-%m")
    monthly = df.groupby("month", as_index=False).sum(numeric_only=True)

    monthly["bess_net_revenue"] = (
        monthly["batt_sale_revenue"]
        - monthly["grid_charge_cost"]
        + monthly["afrr_net_revenue"]
        + monthly["afrr_capacity_total_revenue"]
    )
    monthly["net_revenue"] = monthly["pv_direct_revenue"] + monthly["bess_net_revenue"]

    monthly["pv_revenue_keur_per_mw"] = monthly["pv_direct_revenue"] / max(pv_dc_mw, 1e-12) / 1000.0
    monthly["bess_revenue_keur_per_mw"] = monthly["bess_net_revenue"] / max(batt_power_mw, 1e-12) / 1000.0

    monthly["pv_revenue_eur_per_mwh"] = monthly["pv_direct_revenue"] / monthly["pv_direct_mwh"].clip(lower=1e-12)
    monthly["bess_total_discharged_mwh"] = monthly["shifted_mwh"] + monthly["afrr_discharge_mwh"]
    monthly["bess_revenue_eur_per_mwh"] = monthly["bess_net_revenue"] / monthly["bess_total_discharged_mwh"].clip(lower=1e-12)

    return monthly


def build_inputs_dataframe(inputs: SimulationInputs) -> pd.DataFrame:
    rows = [
        ("batt_power_mw", inputs.batt_power_mw),
        ("nominal_batt_energy_mwh", inputs.nominal_batt_energy_mwh),
        ("bess_availability_pct", inputs.bess_availability_pct),
        ("effective_batt_energy_mwh", inputs.batt_energy_mwh),
        ("pv_dc_mw", inputs.pv_dc_mw),
        ("productible_kwh_per_kwp", inputs.productible_kwh_per_kwp),
        ("pv_losses_pct", inputs.pv_losses_pct),
        ("plant_availability_pct", inputs.plant_availability_pct),
        ("eta_charge", inputs.eta_charge),
        ("eta_discharge", inputs.eta_discharge),
        ("nightly_bess_revenue_eur", inputs.nightly_bess_revenue_eur),
        ("soc_steps", inputs.soc_steps),
        ("initial_soc_mwh", inputs.initial_soc_mwh),
        ("final_soc_mwh", inputs.final_soc_mwh),
        ("min_soc_pct", inputs.min_soc_pct),
        ("max_soc_pct", inputs.max_soc_pct),
        ("grid_export_limit_mw", inputs.grid_export_limit_mw),
        ("cycle_cost_eur_per_mwh", inputs.cycle_cost_eur_per_mwh),
        ("charge_quantile", inputs.charge_quantile),
        ("discharge_quantile", inputs.discharge_quantile),
        ("max_cycles_per_year", inputs.max_cycles_per_year),
        ("min_spread_arbitrage_eur_per_mwh", inputs.min_spread_arbitrage_eur_per_mwh),
        ("forward_optimization_horizon_hours", inputs.forward_optimization_horizon_hours),
        ("afrr_up_cross_market_min_spread_eur_per_mwh", inputs.afrr_up_cross_market_min_spread_eur_per_mwh),
        ("afrr_down_to_wholesale_min_spread_eur_per_mwh", inputs.afrr_down_to_wholesale_min_spread_eur_per_mwh),
        ("pv_capture_rate_pct", inputs.pv_capture_rate_pct),
        ("bess_capture_rate_pct", inputs.bess_capture_rate_pct),
        ("enable_afrr", inputs.enable_afrr),
        ("afrr_min_spread_eur_per_mwh", inputs.afrr_min_spread_eur_per_mwh),
        ("afrr_cycle_cost_eur_per_mwh", inputs.afrr_cycle_cost_eur_per_mwh),
        ("afrr_max_events_per_day", inputs.afrr_max_events_per_day),
        ("afrr_night_start_hour", "removed_phase_1_all_qh_eligible"),
        ("afrr_night_end_hour", "removed_phase_1_all_qh_eligible"),
        ("afrr_pv_zero_tolerance_mwh", inputs.afrr_pv_zero_tolerance_mwh),
        ("afrr_n_qh_per_side", inputs.afrr_n_qh_per_side),
        ("afrr_energy_down_activation_pct", inputs.afrr_energy_down_activation_pct),
        ("afrr_energy_up_activation_pct", inputs.afrr_energy_up_activation_pct),
        ("enable_afrr_capacity", inputs.enable_afrr_capacity),
        ("afrr_certified_capacity_pct", inputs.afrr_certified_capacity_pct),
        ("afrr_capacity_success_rate_pct", inputs.afrr_capacity_success_rate_pct),
        ("afrr_capacity_start_hour", "removed_phase_1_all_qh_eligible"),
        ("afrr_capacity_end_hour", "removed_phase_1_all_qh_eligible"),
        ("allow_afrr_energy_without_capacity", inputs.allow_afrr_energy_without_capacity),
        ("afrr_certified_capacity_up_mw", inputs.afrr_certified_capacity_up_mw),
        ("afrr_certified_capacity_down_mw", inputs.afrr_certified_capacity_down_mw),
        ("enable_tso_dso_curtailment", inputs.enable_tso_dso_curtailment),
        ("enable_self_curtailment", inputs.enable_self_curtailment),
        ("curtailment_threshold_eur_per_mwh", inputs.curtailment_threshold_eur_per_mwh),
        ("pv_commercial_structure", inputs.pv_commercial_structure),
        ("cfd_price_eur_per_mwh", inputs.cfd_price_eur_per_mwh),
        ("negative_price_rule", inputs.negative_price_rule),
        ("consecutive_negative_hours_limit", inputs.consecutive_negative_hours_limit),
        ("ppa_price_eur_per_mwh", inputs.ppa_price_eur_per_mwh),
        ("charge_battery_if_curtailment", inputs.charge_battery_if_curtailment),
        ("enable_cfd", inputs.enable_cfd),
        ("cfd_price_standalone_eur_per_mwh", inputs.cfd_price_standalone_eur_per_mwh),
        ("enable_ppa", inputs.enable_ppa),
        ("ppa_price_standalone_eur_per_mwh", inputs.ppa_price_standalone_eur_per_mwh),
        ("bess_degradation_curve_pct", "" if inputs.bess_degradation_curve_pct is None else list(inputs.bess_degradation_curve_pct)),
        ("degraded_bess_energy_by_year_mwh", "" if inputs.degraded_bess_energy_by_year_mwh is None else list(inputs.degraded_bess_energy_by_year_mwh)),
        ("project_lifetime_years", inputs.project_lifetime_years),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def to_excel_bytes(
    inputs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    afrr_qh_df: pd.DataFrame | None = None,
    afrr_daily_log_df: pd.DataFrame | None = None,
    afrr_capacity_df: pd.DataFrame | None = None,
    bess_degradation_df: pd.DataFrame | None = None,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        inputs_df.to_excel(writer, sheet_name="Inputs", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        monthly_df.to_excel(writer, sheet_name="Monthly", index=False)
        hourly_df.to_excel(writer, sheet_name="Hourly", index=False)

        if afrr_qh_df is not None:
            afrr_qh_df.to_excel(writer, sheet_name="aFRR_QH", index=False)

        if afrr_daily_log_df is not None:
            afrr_daily_log_df.to_excel(writer, sheet_name="aFRR_Daily_Log", index=False)

        if afrr_capacity_df is not None:
            afrr_capacity_df.to_excel(writer, sheet_name="aFRR_Capacity", index=False)

        if bess_degradation_df is not None:
            bess_degradation_df.to_excel(writer, sheet_name="BESS_Degradation", index=False)

    return output.getvalue()


def app():
    st.set_page_config(page_title="Évaluation revenus projet hybride PV + BESS", layout="wide")
    st.title("Évaluation des revenus d'un projet hybride PV + batterie")
    st.caption("Simulation 15 minutes (35040 pas) avec optimisation économique annuelle de la batterie + co-optimisation aFRR quart-horaire Phase 1.")

    with st.expander("Hypothèses structurantes", expanded=False):
        st.markdown(
            """
            - Simulation **quart-horaire sur 35040 pas** pour le cœur du dispatch PV + BESS.
            - La batterie peut **charger depuis le PV et/ou depuis le réseau**.
            - Le moteur choisit la meilleure valorisation économique entre vente immédiate du PV, stockage PV et charge réseau.
            - Les **revenus de services système la nuit** sont ajoutés comme un **revenu fixe par nuit**, sans contrainte de capacité ni de SOC.
            - L'optimisation principale utilise une **programmation dynamique discrétisée sur le SOC**.
            - Une couche **aFRR quart-horaire Phase 1** compare wholesale, aFRR UP, aFRR DOWN et no action à chaque pas de 15 minutes.
            - La curtailment PV peut être:
              1. imposée par TSO/DSO
              2. auto-courtailment selon structure commerciale
            - Option supplémentaire: **Charge Battery if Curtailment**
              pour récupérer une partie de l'énergie autrement curtailed dans la batterie.
            """
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("BESS Parameters")
        batt_power_mw = st.number_input("BESS Usable Power (MW)", min_value=0.0, value=50.0, step=1.0)
        batt_energy_mwh = st.number_input("BESS Usable Capacity (MWh)", min_value=0.0, value=200.0, step=1.0)
        bess_availability_pct = st.number_input("BESS Availability (%)", min_value=0.0, max_value=100.0, value=100.0, step=0.1)
        effective_batt_energy_mwh = batt_energy_mwh * bess_availability_pct / 100.0
        st.caption(f"Effective BESS usable capacity: {effective_batt_energy_mwh:.2f} MWh")
        eta_charge = st.number_input("BESS Charging Efficiency (%)", min_value=1.0, max_value=100.0, value=95.0, step=0.5) / 100.0
        eta_discharge = st.number_input("BESS Discharging Efficiency (%)", min_value=1.0, max_value=100.0, value=95.0, step=0.5) / 100.0
        min_soc_pct = st.slider("BESS Minimum SOC (%)", 0, 100, 0)
        max_soc_pct = st.slider("BESS Maximum SOC (%)", 0, 100, 100)
        initial_soc = st.number_input("BESS BoL SOC (MWh)", min_value=0.0, value=effective_batt_energy_mwh*max_soc_pct/100, step=1.0)
        final_soc = st.number_input("BESS EoL SOC (MWh)", min_value=0.0, value=effective_batt_energy_mwh*min_soc_pct/100, step=1.0)
        bess_capture_rate_pct = st.number_input("BESS Wholesale Capture Rate (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        max_cycles_per_year = st.number_input("Max Cycles / year", min_value=0.0, value=547.0, step=0.1)
        cycle_cost = st.number_input("BESS Cycle Cost (EUR/MWh)", value=0.0)
        charge_quantile = st.slider("Charge Percentile (%)", 0, 100, 100)
        discharge_quantile = st.slider("Discharge Percentile (%)", 0, 100, 0)
        min_spread_arbitrage = st.number_input("Minimum Spread for Arbitrage (EUR/MWh)", min_value=0.0, value=10.0, step=1.0)
        nightly_bess_revenue = st.number_input("Ancillary Services Revenues (EUR/nuit)", min_value=0.0, value=0.0, step=10.0)
        
    with col2:
        st.subheader("PV Parameters")
        pv_dc_mw = st.number_input("PV DC Power (MWc)", min_value=0.0, value=100.0, step=1.0)
        productible = st.number_input("PV Yield (kWh/kWc/an)", min_value=0.0, value=1500.0, step=10.0)
        availability_pct = st.number_input("PV Availability (%)", min_value=0.0, max_value=100.0, value=100.0, step=0.1)
        pv_losses_pct = st.number_input("PV System Losses (%)", min_value=0.0, max_value=100.0, value=8.0, step=0.5)
        pv_capture_rate_pct = st.number_input("PV Capture Rate (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
        
    with col3:
        st.subheader("General Parameters")
        project_lifetime_years = int(st.number_input("Project Lifetime (years)", min_value=1, value=1, step=1))
        grid_export_limit_mw = st.number_input("Grid Injection Limit (MW)", min_value=0.0, value=100.0, step=1.0)
        soc_steps = st.slider("SOC Steps for Optimization", min_value=21, max_value=201, value=21, step=10)

    bess_degradation_upload = st.file_uploader(
            "BESS Degradation Curve",
            type=["xlsx", "xls", "csv"],
            key="bess_degradation_curve",
        )

    st.subheader("PV Commercial Structure")

    contract_col1, contract_col2, contract_col3 = st.columns(3)

    with contract_col1:
        enable_cfd = st.radio("CfD", ["No", "Yes"], horizontal=True) == "Yes"
        cfd_price_standalone = 0.0
        if enable_cfd:
            cfd_price_standalone = st.number_input("CfD Price (€/MWh)", value=50.0, step=1.0)

    with contract_col2:
        enable_ppa = st.radio("PPA", ["No", "Yes"], horizontal=True) == "Yes"
        ppa_price_standalone = 0.0
        if enable_ppa:
            ppa_price_standalone = st.number_input("PPA Price (€/MWh)", value=50.0, step=1.0)
            
    st.subheader("Courbe solaire 15 min (35040 pas)")
    solar_mode = st.radio("Source du profil solaire", ["Courbe standard France", "Upload CSV 35040"], horizontal=True)

    solar_upload = None
    uploaded_solar_is_relative = True
    if solar_mode == "Upload CSV 35040":
        solar_upload = st.file_uploader("Upload du profil solaire CSV (35040 lignes, première colonne numérique)", type=["xlsx", "xls", "csv"], key="solar_csv")
        uploaded_solar_is_relative = st.checkbox(
            "Le CSV uploadé est un profil relatif à normaliser sur le productible annuel (sinon : MWh nets 15 minutes absolus)",
            value=True,
        )

    st.subheader("Sell Price - PV/Grid")
    pv_price_mode = st.radio(
        "Source du prix de vente du PV",
        ["Prix moyen annuel", "Upload CSV 35040"],
        index=1,
        horizontal=True,
    )
    pv_price_value = None
    pv_price_upload = None
    if pv_price_mode == "Prix moyen annuel":
        pv_price_value = st.number_input("Prix moyen PV (EUR/MWh)", value=55.0, step=1.0)
    elif pv_price_mode == "Upload CSV 35040":
        pv_price_upload = st.file_uploader("Upload prix PV CSV (35040 lignes)", type=["xlsx", "xls", "csv"], key="pv_price")

    st.subheader("Sell Price - BESS/Grid")
    batt_sell_mode = st.radio(
        "Source du prix de vente de l'énergie shiftée",
        ["Prix moyen annuel", "Upload CSV 35040"],
        index=1,
        horizontal=True,
    )
    batt_sell_value = None
    batt_sell_upload = None
    if batt_sell_mode == "Prix moyen annuel":
        batt_sell_value = st.number_input("Prix moyen vente batterie (EUR/MWh)", value=90.0, step=1.0)
    elif batt_sell_mode == "Upload CSV 35040":
        batt_sell_upload = st.file_uploader("Upload prix vente batterie CSV (35040 lignes)", type=["xlsx", "xls", "csv"], key="batt_sell")

    st.subheader("Buy Price - BESS/Grid")
    grid_mode = st.radio(
        "Source du prix d'achat réseau",
        ["Identique au prix vente batterie", "Prix moyen annuel", "Upload CSV 35040"],
        index=2,
        horizontal=True,
    )
    grid_buy_value = None
    grid_buy_upload = None
    if grid_mode == "Prix moyen annuel":
        grid_buy_value = st.number_input("Prix moyen achat réseau (EUR/MWh)", value=55.0, step=1.0)
    elif grid_mode == "Upload CSV 35040":
        grid_buy_upload = st.file_uploader("Upload prix achat réseau CSV (35040 lignes)", type=["xlsx", "xls", "csv"], key="grid_buy")

    st.subheader("Curtailment")
    cur1, cur2, cur3 = st.columns(3)

    with cur1:
        tso_dso_curtailment = st.radio("TSO/DSO Curtailment", ["No", "Yes"], index=1, horizontal=True)
        tso_dso_upload = None
        tso_dso_source = "Curtailment Curve"
        if tso_dso_curtailment == "Yes":
            tso_dso_source = st.radio(
                "Source de la courbe TSO/DSO",
                ["Curtailment Curve", "Upload Annual Curtailment Curve Excel (12 monthly %)"],
                horizontal=False,
            )
            if tso_dso_source == "Upload Annual Curtailment Curve Excel (12 monthly %)":
                tso_dso_upload = st.file_uploader("Upload Annual Curtailment Curve Excel (12 monthly %)", type=["xlsx", "xls", "csv"], key="tso_dso_curve")

    with cur2:
        self_curtailment = st.radio("Self Curtailment", ["No", "Yes"], horizontal=True)
        curtailment_threshold = -1.0
        pv_structure = "Fully merchant"
        cfd_price = 0.0
        negative_price_rule = False
        consecutive_negative_hours_limit = 6
        ppa_price = 0.0

        if self_curtailment == "Yes":
            curtailment_threshold = st.number_input("Curtailment Threshold (EUR/MWh)", value=-1.0, step=1.0)
            pv_structure = st.radio("PV Commercial Structure", ["Fully merchant", "With CfD", "With PPA"], horizontal=False)

            if pv_structure == "With CfD":
                cfd_price = st.number_input("CfD Price (EUR/MWh)", value=50.0, step=1.0)
                negative_price_rule_str = st.radio("Negative Price Rule", ["No", "Yes"], horizontal=True)
                negative_price_rule = negative_price_rule_str == "Yes"
                if negative_price_rule:
                    consecutive_negative_hours_limit = int(st.number_input("Consecutive Negative Hours Limit", min_value=1, value=6, step=1))

            if pv_structure == "With PPA":
                ppa_price = st.number_input("PPA Price (EUR/MWh)", value=50.0, step=1.0)

    with cur3:
        charge_battery_if_curtailment = st.radio("Charge Battery if Curtailment", ["Yes", "No"], horizontal=True) == "Yes"
        
    st.subheader("aFRR Capacity")
    enable_afrr_capacity = st.checkbox("Activer aFRR Capacity", value=False)

    afrr_capacity_up_upload = None
    afrr_capacity_down_upload = None
    afrr_certified_capacity_pct = 100.0
    afrr_capacity_start_hour = 0
    afrr_capacity_end_hour = 0
    afrr_capacity_success_rate_pct = 80.0
    afrr_capacity_up_source = "Upload afrr_up_capacity_price_15min_spain_2025 Excel"
    afrr_capacity_down_source = "Upload afrr_down_capacity_price_15min_spain_2025 Excel"

    if enable_afrr_capacity:
        cap_col1, cap_col2, cap_col3 = st.columns(3)

        with cap_col1:
            st.caption("aFRR Capacity datasets are no longer embedded. Upload 35040-step files.")
            afrr_capacity_up_upload = st.file_uploader(
                "Upload afrr_up_capacity_price_15min_spain_2025 Excel/CSV (35040 lignes)",
                type=["xlsx", "xls", "csv"],
                key="afrr_capacity_up",
            )
            afrr_capacity_down_upload = st.file_uploader(
                "Upload afrr_down_capacity_price_15min_spain_2025 Excel/CSV (35040 lignes)",
                type=["xlsx", "xls", "csv"],
                key="afrr_capacity_down",
            )

        with cap_col2:
            afrr_certified_capacity_pct = st.number_input(
                "% of Certified Capacity for aFRR",
                min_value=0.0,
                max_value=100.0,
                value=100.0,
                step=1.0,
            )
            afrr_capacity_success_rate_pct = st.slider(
                "aFRR Capacity Bid Success Rate (%)",
                min_value=0,
                max_value=100,
                value=80,
                step=1,
            )
            st.caption("Used only in expected-value optimization; it does not reduce physical MW/MWh dispatch.")

        with cap_col3:
            st.info("Phase 1: aFRR Capacity is eligible at any 15-minute timestep. Start/end hour filters were removed.")

    st.subheader("aFRR Energy")
    enable_afrr = st.checkbox("Activer aFRR Energy", value=False)
    allow_afrr_energy_without_capacity = st.checkbox(
        "Allow aFRR energy without aFRR capacity",
        value=True,
    )

    afrr_charge_upload = None
    afrr_discharge_upload = None
    afrr_min_spread = 0.0
    afrr_cycle_cost = cycle_cost
    afrr_night_start_hour = 20
    afrr_night_end_hour = 8
    afrr_max_events_per_day = 1
    afrr_energy_down_activation_pct = 100.0
    afrr_energy_up_activation_pct = 100.0
    forward_optimization_horizon_hours = 24.0
    afrr_up_cross_market_min_spread = 20.0
    afrr_down_to_wholesale_min_spread = 20.0
    afrr_charge_source = "Upload prix aFRR charge Excel/CSV (35040 lignes)"
    afrr_discharge_source = "Upload prix aFRR décharge Excel/CSV (35040 lignes)"

    if enable_afrr:
        c_afrr1, c_afrr2, c_afrr3 = st.columns(3)

        with c_afrr1:
            st.caption("aFRR Energy datasets are no longer embedded. Upload 35040-step files.")
            afrr_charge_upload = st.file_uploader(
                "Upload prix aFRR charge / down energy Excel/CSV (35040 lignes)",
                type=["xlsx", "xls", "csv"],
                key="afrr_charge",
            )
            afrr_discharge_upload = st.file_uploader(
                "Upload prix aFRR décharge / up energy Excel/CSV (35040 lignes)",
                type=["xlsx", "xls", "csv"],
                key="afrr_discharge",
            )

        with c_afrr2:
            afrr_min_spread = st.number_input("Spread minimum aFRR net (EUR/MWh)", min_value=0.0, value=min_spread_arbitrage, step=1.0)
            afrr_cycle_cost = st.number_input("Coût de cycle aFRR (EUR/MWh)", min_value=0.0, value=float(cycle_cost), step=1.0)
            afrr_energy_down_activation_pct = st.number_input("aFRR Energy Down Activation (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
            afrr_energy_up_activation_pct = st.number_input("aFRR Energy Up Activation (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

        with c_afrr3:
            st.info("Phase 1: aFRR Energy is eligible at any 15-minute timestep. Night filters were removed.")
            forward_optimization_horizon_hours = st.slider("Forward Optimization Horizon (hours)", min_value=1, max_value=72, value=24, step=1)
            afrr_up_cross_market_min_spread = st.number_input("Minimum Spread Wholesale Charge → aFRR UP Discharge (€/MWh)", min_value=0.0, value=20.0, step=1.0)
            afrr_down_to_wholesale_min_spread = st.number_input("Minimum Spread aFRR DOWN Charge → Wholesale Discharge (€/MWh)", min_value=0.0, value=20.0, step=1.0)
            afrr_max_events_per_day = st.number_input("Nombre max d'événements aFRR / jour (legacy, not used in Phase 1 capacity mode)", min_value=1, value=1, step=1)

    st.markdown("---")
    run = st.button("Lancer la simulation", type="primary")

    if not run:
        return

    start_time = time.time()

    try:
        if effective_batt_energy_mwh < batt_power_mw and effective_batt_energy_mwh > 0:
            st.warning("Attention : la capacité batterie est inférieure à 1h de puissance. C'est possible, mais atypique.")
        if initial_soc > effective_batt_energy_mwh:
            st.error("Le SOC initial ne peut pas dépasser la capacité batterie.")
            return
        if final_soc > effective_batt_energy_mwh:
            st.error("Le SOC final ne peut pas dépasser la capacité batterie.")
            return
        if not (0.0 <= min_soc_pct <= 100.0):
            st.error("Minimum SOC batterie (%) doit être compris entre 0 et 100 %.")
            return
        if not (0.0 <= max_soc_pct <= 100.0):
            st.error("Maximum SOC batterie (%) doit être compris entre 0 et 100 %.")
            return
        if min_soc_pct >= max_soc_pct:
            st.error("Minimum SOC batterie (%) doit être strictement inférieur au Maximum SOC batterie (%).")
            return

        min_soc_mwh = effective_batt_energy_mwh * min_soc_pct / 100.0
        max_soc_mwh = effective_batt_energy_mwh * max_soc_pct / 100.0

        if initial_soc < min_soc_mwh or initial_soc > max_soc_mwh:
            st.error(
                f"Le SOC initial doit être compris entre {min_soc_mwh:.2f} MWh "
                f"et {max_soc_mwh:.2f} MWh."
            )
            return
        if final_soc < min_soc_mwh or final_soc > max_soc_mwh:
            st.error(
                f"Le SOC final doit être compris entre {min_soc_mwh:.2f} MWh "
                f"et {max_soc_mwh:.2f} MWh."
            )
            return
        if enable_cfd and enable_ppa:
            st.error("CfD et PPA ne peuvent pas être activés en même temps.")
            return
        if enable_afrr_capacity and not enable_afrr:
            st.error("Veuillez activer aFRR Energy pour utiliser aFRR Capacity.")
            return
        if enable_afrr and (not enable_afrr_capacity) and (not allow_afrr_energy_without_capacity):
            st.error("La participation en aFRR Energy sans aFRR Capacity n’est pas autorisée.")
            return
        if enable_afrr_capacity:
            if afrr_capacity_up_upload is None:
                st.error("Merci d'uploader le fichier aFRR Capacity UP 15 minutes (35040 lignes).")
                return
            if afrr_capacity_down_upload is None:
                st.error("Merci d'uploader le fichier aFRR Capacity Down 15 minutes (35040 lignes).")
                return
        if not (0.0 <= afrr_certified_capacity_pct <= 100.0):
            st.error("% of Certified Capacity for aFRR doit être compris entre 0 et 100 %.")
            return
        if not (0.0 <= afrr_energy_down_activation_pct <= 100.0):
            st.error("aFRR Energy Down Activation (%) doit être compris entre 0 et 100 %.")
            return
        if not (0.0 <= afrr_energy_up_activation_pct <= 100.0):
            st.error("aFRR Energy Up Activation (%) doit être compris entre 0 et 100 %.")
            return

        try:
            bess_degradation_curve_pct, degraded_bess_energy_by_year_mwh, bess_degradation_df = read_bess_degradation_excel(
                bess_degradation_upload,
                project_lifetime_years,
                effective_batt_energy_mwh,
            )
        except Exception as e:
            st.error(f"Erreur courbe de dégradation BESS: {e}")
            return
            
        # Base PV
        if solar_mode == "Courbe standard France":
            solar_relative = build_standard_france_solar_profile()
            base_pv_hourly_mwh, pv_stats = build_pv_generation_mwh(
                solar_relative, pv_dc_mw, productible, pv_losses_pct, availability_pct
            )
        else:
            if solar_upload is None:
                st.error("Merci d'uploader un fichier solaire 35040 pas de 15 minutes.")
                return

            uploaded = _read_single_column_csv(solar_upload)
            if uploaded_solar_is_relative:
                base_pv_hourly_mwh, pv_stats = build_pv_generation_mwh(
                    uploaded, pv_dc_mw, productible, pv_losses_pct, availability_pct
                )
            else:
                base_pv_hourly_mwh = np.maximum(uploaded, 0.0) * pv_dc_mw
                annual_net = float(base_pv_hourly_mwh.sum())
                annual_dc = float(pv_dc_mw * productible)
                pv_stats = {
                    "annual_dc_mwh": annual_dc,
                    "annual_net_mwh": annual_net,
                    "annual_losses_mwh": float(max(annual_dc - annual_net, 0.0)),
                }

        # Market/aFRR datasets are not embedded anymore. Upload files when an upload source is selected.
        if (not enable_cfd) and (not enable_ppa) and pv_price_mode == "Upload CSV 35040" and pv_price_upload is None:
            st.error("Merci d'uploader le fichier de prix PV / spot 15 minutes (35040 lignes).")
            return
        if batt_sell_mode == "Upload CSV 35040" and batt_sell_upload is None:
            st.error("Merci d'uploader le fichier de prix de vente batterie 15 minutes (35040 lignes).")
            return
        if grid_mode == "Upload CSV 35040" and grid_buy_upload is None:
            st.error("Merci d'uploader le fichier de prix d'achat réseau 15 minutes (35040 lignes).")
            return

        # Raw price curves
        pv_price_curve_raw = None

        if enable_cfd:
            pv_price_curve_raw = _make_flat_curve(cfd_price_standalone)
        elif enable_ppa:
            pv_price_curve_raw = _make_flat_curve(ppa_price_standalone)
        else:
            if pv_price_mode == "Prix moyen annuel":
                pv_price_curve_raw = _make_flat_curve(pv_price_value)
            else:
                pv_price_curve_raw = _read_single_column_csv(pv_price_upload)

        if pv_price_curve_raw is None:
            raise ValueError("pv_price_curve_raw was not properly initialized.")

        if batt_sell_mode == "Prix moyen annuel":
            batt_sell_curve_raw = _make_flat_curve(batt_sell_value)
        else:
            batt_sell_curve_raw = _read_single_column_csv(batt_sell_upload)

        if grid_mode == "Identique au prix vente batterie":
            grid_buy_curve_raw = batt_sell_curve_raw.copy()
        elif grid_mode == "Prix moyen annuel":
            grid_buy_curve_raw = _make_flat_curve(grid_buy_value)
        else:
            grid_buy_curve_raw = _read_single_column_csv(grid_buy_upload)

        afrr_charge_curve_qh_raw = None
        afrr_discharge_curve_qh_raw = None
        if enable_afrr:
            if afrr_charge_upload is None:
                st.error("Merci d'uploader le fichier aFRR charge / down energy 15 minutes (35040 lignes).")
                return
            if afrr_discharge_upload is None:
                st.error("Merci d'uploader le fichier aFRR décharge / up energy 15 minutes (35040 lignes).")
                return
            afrr_charge_curve_qh_raw = _read_single_column_csv_qh(afrr_charge_upload)
            afrr_discharge_curve_qh_raw = _read_single_column_csv_qh(afrr_discharge_upload)

        # aFRR Capacity hourly prices and certified capacities
        afrr_capacity_up_price_h_raw = None
        afrr_capacity_down_price_h_raw = None
        afrr_certified_capacity_up_mw = 0.0
        afrr_certified_capacity_down_mw = 0.0

        def read_afrr_capacity_file(uploaded_file, year):
            # 2025 aFRR capacity datasets are already in 15-minute resolution: one numeric column, 35040 rows.
            return _read_single_column_csv_qh(uploaded_file)
        
        
        if enable_afrr_capacity:
            try:
                afrr_capacity_up_price_h_raw = read_afrr_capacity_file(afrr_capacity_up_upload, DEFAULT_YEAR)
                afrr_capacity_down_price_h_raw = read_afrr_capacity_file(afrr_capacity_down_upload, DEFAULT_YEAR)
            except Exception as e:
                st.error(f"Erreur fichier aFRR Capacity: {e}")
                return

            afrr_certified_capacity_up_mw = (
                batt_power_mw
                * afrr_certified_capacity_pct / 100.0
                * availability_pct / 100.0
                * eta_discharge
            )
            afrr_certified_capacity_down_mw = (
                batt_power_mw
                * afrr_certified_capacity_pct / 100.0
                * availability_pct / 100.0
                * eta_charge
            )

        # Capture rates
        pv_capture_factor = pv_capture_rate_pct / 100.0
        bess_capture_factor = bess_capture_rate_pct / 100.0

        pv_spot_price_effective = pv_price_curve_raw * pv_capture_factor

        # BESS Capture Rate represents imperfect monetization of discharge value only.
        # It reduces BESS sell prices used by the optimizer, but it must not reduce
        # grid charging prices, PV prices, charging energy, or physical capacity.
        batt_sell_curve_effective = batt_sell_curve_raw * bess_capture_factor
        grid_buy_curve_effective = grid_buy_curve_raw.copy()

        afrr_charge_curve_qh_effective = None
        afrr_discharge_curve_qh_effective = None
        if enable_afrr:
            afrr_charge_curve_qh_effective = afrr_charge_curve_qh_raw.copy()
            afrr_discharge_curve_qh_effective = afrr_discharge_curve_qh_raw * bess_capture_factor

        # 1) TSO/DSO curtailment
        if tso_dso_curtailment == "Yes":
            if tso_dso_source == "Curtailment Curve":
                with _open_builtin_file(BUILTIN_CURTAILMENT_CURVE, "Curtailment Curve") as f:
                    tso_dso_monthly_pct = read_monthly_curtailment_excel(f)
            else:
                if tso_dso_upload is None:
                    st.error("Merci d'uploader la courbe annuelle de curtailment TSO/DSO.")
                    return
                tso_dso_monthly_pct = read_monthly_curtailment_excel(tso_dso_upload)
            tso_out = apply_tso_dso_curtailment(base_pv_hourly_mwh, tso_dso_monthly_pct)
        else:
            tso_out = {
                "pv_after_tso_dso_mwh": base_pv_hourly_mwh.copy(),
                "tso_dso_curtailed_mwh": np.zeros(QH_PER_YEAR, dtype=float),
                "tso_dso_curtailment_flag": np.zeros(QH_PER_YEAR, dtype=int),
                "tso_dso_monthly_pct_hourly": np.zeros(QH_PER_YEAR, dtype=float),
            }
            tso_dso_monthly_pct = np.zeros(12, dtype=float)

        # 2) Self curtailment
        self_out = apply_self_curtailment(
            pv_hourly_mwh=tso_out["pv_after_tso_dso_mwh"],
            pv_spot_price_raw=pv_price_curve_raw,
            pv_spot_price_effective=pv_spot_price_effective,
            enable_self_curtailment=(self_curtailment == "Yes"),
            pv_commercial_structure=pv_structure,
            curtailment_threshold_eur_per_mwh=curtailment_threshold,
            cfd_price_eur_per_mwh=cfd_price,
            negative_price_rule=negative_price_rule,
            consecutive_negative_hours_limit=consecutive_negative_hours_limit,
            ppa_price_eur_per_mwh=ppa_price,
        )

        if enable_cfd or enable_ppa:
            self_out["pv_effective_price_eur_per_mwh"] = pv_spot_price_effective.copy()
            
        # Curtailment pipeline
        pv_after_tso_dso = tso_out["pv_after_tso_dso_mwh"]
        pv_after_self = self_out["pv_after_self_curtailment_mwh"]
        pv_curtailment_candidate_mwh = np.maximum(base_pv_hourly_mwh - pv_after_self, 0.0)

        if charge_battery_if_curtailment:
            curtailed_pv_recoverable_mwh = pv_curtailment_candidate_mwh.copy()
        else:
            curtailed_pv_recoverable_mwh = np.zeros(QH_PER_YEAR, dtype=float)

        pv_sellable_for_dispatch_mwh = pv_after_self.copy()
        pv_effective_price_for_revenue = self_out["pv_effective_price_eur_per_mwh"]

        # PV-only benchmark uses only sellable curtailed PV
        pure_pv_benchmark = build_pure_pv_benchmark(
            pv_generation_mwh=pv_sellable_for_dispatch_mwh,
            pv_price=pv_effective_price_for_revenue,
            grid_export_limit_mw=grid_export_limit_mw,
        )
        pure_pv_cfd_benchmark = None

        if enable_cfd:
            pv_cfd_price_curve = _make_flat_curve(cfd_price_standalone)
        
            pure_pv_cfd_benchmark = build_pure_pv_benchmark(
                pv_generation_mwh=pv_sellable_for_dispatch_mwh,
                pv_price=pv_cfd_price_curve,
                grid_export_limit_mw=grid_export_limit_mw,
            )
        sim_inputs = SimulationInputs(
            batt_power_mw=batt_power_mw,
            batt_energy_mwh=effective_batt_energy_mwh,
            nominal_batt_energy_mwh=batt_energy_mwh,
            bess_availability_pct=bess_availability_pct,
            pv_dc_mw=pv_dc_mw,
            productible_kwh_per_kwp=productible,
            pv_losses_pct=pv_losses_pct,
            plant_availability_pct=availability_pct,
            eta_charge=eta_charge,
            eta_discharge=eta_discharge,
            pv_price=pv_effective_price_for_revenue,
            batt_sell_price=batt_sell_curve_effective,
            grid_buy_price=grid_buy_curve_effective,
            solar_profile=pv_sellable_for_dispatch_mwh,
            curtailed_pv_recoverable_mwh=curtailed_pv_recoverable_mwh,
            nightly_bess_revenue_eur=nightly_bess_revenue,
            soc_steps=soc_steps,
            initial_soc_mwh=initial_soc,
            final_soc_mwh=final_soc,
            min_soc_pct=min_soc_pct,
            max_soc_pct=max_soc_pct,
            grid_export_limit_mw=grid_export_limit_mw,
            cycle_cost_eur_per_mwh=cycle_cost,
            charge_quantile=charge_quantile,
            discharge_quantile=discharge_quantile,
            max_cycles_per_year=max_cycles_per_year,
            min_spread_arbitrage_eur_per_mwh=min_spread_arbitrage,
            forward_optimization_horizon_hours=float(forward_optimization_horizon_hours),
            afrr_up_cross_market_min_spread_eur_per_mwh=float(afrr_up_cross_market_min_spread),
            afrr_down_to_wholesale_min_spread_eur_per_mwh=float(afrr_down_to_wholesale_min_spread),
            pv_capture_rate_pct=pv_capture_rate_pct,
            bess_capture_rate_pct=bess_capture_rate_pct,
            enable_afrr=enable_afrr,
            afrr_charge_price_qh=afrr_charge_curve_qh_effective,
            afrr_discharge_price_qh=afrr_discharge_curve_qh_effective,
            afrr_min_spread_eur_per_mwh=afrr_min_spread,
            afrr_cycle_cost_eur_per_mwh=afrr_cycle_cost,
            afrr_max_events_per_day=int(afrr_max_events_per_day),
            afrr_night_start_hour=int(afrr_night_start_hour),
            afrr_night_end_hour=int(afrr_night_end_hour),
            afrr_pv_zero_tolerance_mwh=PV_ZERO_TOLERANCE_MWH,
            afrr_n_qh_per_side=16,
            afrr_energy_down_activation_pct=afrr_energy_down_activation_pct,
            afrr_energy_up_activation_pct=afrr_energy_up_activation_pct,
            enable_afrr_capacity=enable_afrr_capacity,
            afrr_capacity_up_price_h=afrr_capacity_up_price_h_raw,
            afrr_capacity_down_price_h=afrr_capacity_down_price_h_raw,
            afrr_certified_capacity_pct=afrr_certified_capacity_pct,
            afrr_capacity_success_rate_pct=float(afrr_capacity_success_rate_pct),
            afrr_capacity_start_hour=0,
            afrr_capacity_end_hour=0,
            allow_afrr_energy_without_capacity=allow_afrr_energy_without_capacity,
            afrr_certified_capacity_up_mw=afrr_certified_capacity_up_mw,
            afrr_certified_capacity_down_mw=afrr_certified_capacity_down_mw,
            enable_tso_dso_curtailment=(tso_dso_curtailment == "Yes"),
            tso_dso_monthly_curtailment_pct=tso_dso_monthly_pct,
            enable_self_curtailment=(self_curtailment == "Yes"),
            curtailment_threshold_eur_per_mwh=curtailment_threshold,
            pv_commercial_structure=pv_structure,
            cfd_price_eur_per_mwh=cfd_price,
            negative_price_rule=negative_price_rule,
            consecutive_negative_hours_limit=consecutive_negative_hours_limit,
            ppa_price_eur_per_mwh=ppa_price,
            charge_battery_if_curtailment=charge_battery_if_curtailment,
            enable_cfd=enable_cfd,
            cfd_price_standalone_eur_per_mwh=cfd_price_standalone,
            enable_ppa=enable_ppa,
            ppa_price_standalone_eur_per_mwh=ppa_price_standalone,
            project_lifetime_years=project_lifetime_years,
            bess_degradation_curve_pct=bess_degradation_curve_pct,
            degraded_bess_energy_by_year_mwh=degraded_bess_energy_by_year_mwh,
        )

        # Phase 1 co-optimization flow:
        # 1) run a baseline wholesale DP without aFRR capacity blocking,
        # 2) select aFRR capacity by expected-value comparison against that wholesale reference,
        # 3) rerun the final DP with selected aFRR capacity intervals blocked from wholesale dispatch.
        inputs_df = build_inputs_dataframe(sim_inputs)

        with st.spinner("Optimisation wholesale de référence en cours..."):
            wholesale_reference_result = optimize_dispatch_dp(sim_inputs)

        afrr_capacity_result = simulate_afrr_capacity(sim_inputs, wholesale_reference_result=wholesale_reference_result)
        sim_inputs.afrr_capacity_selected_market_h = afrr_capacity_result["afrr_capacity_selected_market_h"]
        sim_inputs.afrr_expected_up_activated_mwh_qh = afrr_capacity_result.get("expected_up_activated_mwh", np.zeros(QH_PER_YEAR, dtype=float))
        sim_inputs.afrr_expected_down_activated_mwh_qh = afrr_capacity_result.get("expected_down_activated_mwh", np.zeros(QH_PER_YEAR, dtype=float))

        with st.spinner("Optimisation économique annuelle finale en cours..."):
            result = optimize_dispatch_dp(sim_inputs)

        afrr_result = None
        reconciliation = None
        final_result = result

        if sim_inputs.enable_afrr:
            with st.spinner("Simulation aFRR quart-horaire et validation de livrabilité SOC finale en cours..."):
                afrr_result = simulate_afrr_night_arbitrage(sim_inputs, result)
                reconciliation = reconcile_wholesale_afrr_dispatch_qh(result_hourly=result, afrr_result=afrr_result, inputs=sim_inputs)
                reconciliation, _cycle_budget_stats = enforce_hard_annual_cycle_cap_on_reconciliation(
                    reconciliation,
                    sim_inputs,
                    afrr_capacity_result=afrr_capacity_result,
                )
                final_result = build_final_result_after_market_arbitration(base_result=result, reconciliation=reconciliation, inputs=sim_inputs)

                # Final-combined-SOC deliverability enforcement.
                # The first aFRR capacity pass uses a forward approximation; this loop
                # removes UP/DOWN awards that the final physical combined SOC trajectory
                # could not deliver, then reruns the final DP/aFRR dispatch.
                for _deliverability_pass in range(3):
                    afrr_capacity_result, _deliverability_stats = enforce_afrr_capacity_deliverability_from_final_dispatch(
                        afrr_capacity_result,
                        reconciliation,
                        tolerance_mwh=1e-6,
                    )

                    if (
                        _deliverability_stats.get("removed_up", 0) == 0
                        and _deliverability_stats.get("removed_down", 0) == 0
                    ):
                        break

                    sim_inputs.afrr_capacity_selected_market_h = afrr_capacity_result["afrr_capacity_selected_market_h"]
                    sim_inputs.afrr_expected_up_activated_mwh_qh = afrr_capacity_result.get(
                        "expected_up_activated_mwh",
                        np.zeros(QH_PER_YEAR, dtype=float),
                    )
                    sim_inputs.afrr_expected_down_activated_mwh_qh = afrr_capacity_result.get(
                        "expected_down_activated_mwh",
                        np.zeros(QH_PER_YEAR, dtype=float),
                    )

                    result = optimize_dispatch_dp(sim_inputs)
                    afrr_result = simulate_afrr_night_arbitrage(sim_inputs, result)
                    reconciliation = reconcile_wholesale_afrr_dispatch_qh(result_hourly=result, afrr_result=afrr_result, inputs=sim_inputs)
                    reconciliation, _cycle_budget_stats = enforce_hard_annual_cycle_cap_on_reconciliation(
                        reconciliation,
                        sim_inputs,
                        afrr_capacity_result=afrr_capacity_result,
                    )
                    final_result = build_final_result_after_market_arbitration(base_result=result, reconciliation=reconciliation, inputs=sim_inputs)

                # Store final actual shortfalls in the capacity audit arrays.
                if reconciliation is not None:
                    afrr_capacity_result["afrr_up_expected_vs_actual_shortfall_mwh"] = reconciliation.get(
                        "afrr_up_activation_shortfall_qh_mwh",
                        np.zeros(QH_PER_YEAR, dtype=float),
                    )
                    afrr_capacity_result["afrr_down_expected_vs_actual_shortfall_mwh"] = reconciliation.get(
                        "afrr_down_activation_shortfall_qh_mwh",
                        np.zeros(QH_PER_YEAR, dtype=float),
                    )

        final_result = add_afrr_capacity_to_final_result(final_result, afrr_capacity_result)

        # Recompute actual curtailed PV recovered AFTER final dispatch/reconciliation
        pv_curtailed_to_battery_actual = final_result.get(
            "pv_curtailed_to_battery",
            result["pv_curtailed_to_battery"],
        )
        
        pv_curtailed_residual_lost_mwh = np.maximum(
            pv_curtailment_candidate_mwh - pv_curtailed_to_battery_actual,
            0.0,
        )
        
        curtailment_outputs = {
            "base_pv_generation_mwh": base_pv_hourly_mwh,
            "pv_after_tso_dso_curtailment_mwh": pv_after_tso_dso,
            "pv_after_self_curtailment_mwh": pv_after_self,
            "tso_dso_curtailed_mwh": tso_out["tso_dso_curtailed_mwh"],
            "self_curtailed_mwh": self_out["self_curtailed_mwh"],
            "pv_curtailment_candidate_mwh": pv_curtailment_candidate_mwh,
            "pv_curtailed_to_battery_mwh_actual": pv_curtailed_to_battery_actual,
            "pv_curtailed_residual_lost_mwh": pv_curtailed_residual_lost_mwh,
            "pv_effective_price_eur_per_mwh": pv_effective_price_for_revenue,
            "tso_dso_curtailment_flag": tso_out["tso_dso_curtailment_flag"],
            "self_curtailment_flag": self_out["self_curtailment_flag"],
            "self_curtailment_reason": self_out["self_curtailment_reason"],
            "pv_commercial_structure_hourly": self_out["pv_commercial_structure_hourly"],
        }

        # BESS Capture Rate reporting
        # Theoretical revenue uses the same dispatched discharge volumes but the raw, uncaptured sell prices.
        wholesale_theoretical_revenue_without_capture = final_result["discharge"] * batt_sell_curve_raw
        wholesale_actual_revenue_with_capture = final_result["batt_sale_revenue"]
        wholesale_revenue_loss_due_to_capture = (
            wholesale_theoretical_revenue_without_capture
            - wholesale_actual_revenue_with_capture
        )

        afrr_theoretical_revenue_without_capture = np.zeros(QH_PER_YEAR, dtype=float)
        afrr_actual_revenue_with_capture = final_result["afrr_sale_revenue_hourly_eur"] if "afrr_sale_revenue_hourly_eur" in final_result else np.zeros(QH_PER_YEAR, dtype=float)
        if reconciliation is not None and afrr_discharge_curve_qh_raw is not None:
            afrr_theoretical_revenue_without_capture = (
                reconciliation["afrr_discharge_qh_mwh"] * afrr_discharge_curve_qh_raw
            )

        afrr_revenue_loss_due_to_capture = (
            afrr_theoretical_revenue_without_capture
            - afrr_actual_revenue_with_capture
        )

        bess_theoretical_revenue_without_capture_hourly = (
            wholesale_theoretical_revenue_without_capture
            + afrr_theoretical_revenue_without_capture
        )
        bess_actual_revenue_with_capture_hourly = (
            wholesale_actual_revenue_with_capture
            + afrr_actual_revenue_with_capture
        )
        bess_revenue_loss_due_to_capture_hourly = (
            bess_theoretical_revenue_without_capture_hourly
            - bess_actual_revenue_with_capture_hourly
        )

        final_result["bess_theoretical_revenue_without_capture_hourly_eur"] = bess_theoretical_revenue_without_capture_hourly
        final_result["bess_actual_revenue_with_capture_hourly_eur"] = bess_actual_revenue_with_capture_hourly
        final_result["bess_revenue_loss_due_to_capture_rate_hourly_eur"] = bess_revenue_loss_due_to_capture_hourly
        final_result["bess_theoretical_revenue_without_capture_eur"] = np.array([float(np.sum(bess_theoretical_revenue_without_capture_hourly))])
        final_result["bess_actual_revenue_with_capture_eur"] = np.array([float(np.sum(bess_actual_revenue_with_capture_hourly))])
        final_result["bess_revenue_loss_due_to_capture_rate_eur"] = np.array([float(np.sum(bess_revenue_loss_due_to_capture_hourly))])
        final_result["avg_raw_bess_sell_price_eur_per_mwh"] = np.array([float(np.mean(batt_sell_curve_raw))])
        final_result["avg_effective_bess_sell_price_eur_per_mwh"] = np.array([float(np.mean(batt_sell_curve_effective))])

        # Cycle cost accounting and comparison vs. a zero-cycle-cost run.
        final_result["total_cycle_cost_eur"] = np.array([
            float(final_result.get("total_wholesale_cycle_cost_eur", np.array([0.0]))[0])
            + float(final_result.get("total_afrr_cycle_cost_eur", np.array([0.0]))[0])
        ])
        total_discharged_for_cycle_cost_mwh = float(final_result.get("energy_shifted_mwh", np.array([0.0]))[0])
        final_result["average_cycle_cost_per_discharged_mwh"] = np.array([
            float(final_result["total_cycle_cost_eur"][0]) / max(total_discharged_for_cycle_cost_mwh, 1e-12)
        ])
        final_result["equivalent_cycles_with_cycle_cost"] = final_result["equivalent_cycles"].copy()

        try:
            if cycle_cost > 1e-12 or afrr_cycle_cost > 1e-12:
                no_cycle_inputs = replace(
                    sim_inputs,
                    cycle_cost_eur_per_mwh=0.0,
                    afrr_cycle_cost_eur_per_mwh=0.0,
                )
                no_cycle_inputs.afrr_capacity_selected_market_h = sim_inputs.afrr_capacity_selected_market_h
                no_cycle_result = optimize_dispatch_dp(no_cycle_inputs)
                if no_cycle_inputs.enable_afrr:
                    no_cycle_afrr_result = simulate_afrr_night_arbitrage(no_cycle_inputs, no_cycle_result)
                    no_cycle_reconciliation = reconcile_wholesale_afrr_dispatch_qh(
                        result_hourly=no_cycle_result,
                        afrr_result=no_cycle_afrr_result,
                        inputs=no_cycle_inputs,
                    )
                    no_cycle_final = build_final_result_after_market_arbitration(
                        base_result=no_cycle_result,
                        reconciliation=no_cycle_reconciliation,
                        inputs=no_cycle_inputs,
                    )
                else:
                    no_cycle_final = no_cycle_result
                no_cycle_final = add_afrr_capacity_to_final_result(no_cycle_final, afrr_capacity_result)
                final_result["equivalent_cycles_without_cycle_cost"] = np.array([float(no_cycle_final["equivalent_cycles"][0])])
                final_result["energy_shifted_without_cycle_cost_mwh"] = np.array([float(no_cycle_final["energy_shifted_mwh"][0])])
            else:
                final_result["equivalent_cycles_without_cycle_cost"] = final_result["equivalent_cycles"].copy()
                final_result["energy_shifted_without_cycle_cost_mwh"] = final_result["energy_shifted_mwh"].copy()
        except Exception as e:
            final_result["equivalent_cycles_without_cycle_cost"] = np.array([np.nan])
            final_result["energy_shifted_without_cycle_cost_mwh"] = np.array([np.nan])
            st.warning(f"Impossible de calculer le scénario sans coût de cycle: {e}")

        if reconciliation is not None:
            combined_qh_df = pd.DataFrame({
                "datetime": reconciliation["datetime_qh"],
                "combined_charge_to_soc_qh_mwh": reconciliation["combined_charge_to_soc_qh_mwh"],
                "combined_discharge_from_soc_qh_mwh": reconciliation["combined_discharge_from_soc_qh_mwh"],
                "wholesale_charge_to_soc_qh_mwh": (
                    reconciliation["wholesale_pv_to_batt_qh_mwh"]
                    + reconciliation["wholesale_pv_curtailed_to_batt_qh_mwh"]
                    + reconciliation["wholesale_grid_charge_qh_mwh"]
                ) * sim_inputs.eta_charge,
                "wholesale_pv_curtailed_to_batt_qh_mwh": reconciliation["wholesale_pv_curtailed_to_batt_qh_mwh"],
                "wholesale_discharge_from_soc_qh_mwh": reconciliation["wholesale_discharge_qh_mwh"] / max(sim_inputs.eta_discharge, 1e-12),
                "afrr_charge_to_soc_qh_mwh": reconciliation["afrr_charge_qh_mwh"] * sim_inputs.eta_charge,
                "afrr_discharge_from_soc_qh_mwh": reconciliation["afrr_discharge_qh_mwh"] / max(sim_inputs.eta_discharge, 1e-12),
                "afrr_charge_market_qh_mwh": reconciliation["afrr_charge_qh_mwh"],
                "afrr_discharge_market_qh_mwh": reconciliation["afrr_discharge_qh_mwh"],
                "afrr_energy_down_activated": reconciliation["afrr_energy_down_activated_qh"],
                "afrr_energy_up_activated": reconciliation["afrr_energy_up_activated_qh"],
                "selected_charge_market_qh": reconciliation["selected_charge_market_qh"],
                "selected_charge_price_qh": reconciliation["selected_charge_price_qh"],
                "selected_discharge_channel_qh": reconciliation["selected_discharge_channel_qh"],
                "selected_discharge_market_qh": reconciliation["selected_discharge_market_qh"],
                "selected_discharge_price_qh": reconciliation["selected_discharge_price_qh"],
                "battery_soc_mwh_end_qh": reconciliation["combined_soc_qh"][1:],
            })
            combined_soc_hourly_end = reconciliation["combined_soc_hourly_end_mwh"]
        else:
            combined_soc_result = build_combined_soc_with_afrr(
                result_hourly=result,
                afrr_result=None,
                batt_energy_mwh=sim_inputs.batt_energy_mwh,
                initial_soc_mwh=sim_inputs.initial_soc_mwh,
                eta_charge=sim_inputs.eta_charge,
                eta_discharge=sim_inputs.eta_discharge,
                min_soc_pct=sim_inputs.min_soc_pct,
                max_soc_pct=sim_inputs.max_soc_pct,
            )

            combined_qh_df = pd.DataFrame({
                "datetime": build_quarter_hour_index(DEFAULT_YEAR),
                "combined_charge_to_soc_qh_mwh": combined_soc_result["combined_charge_to_soc_qh"],
                "combined_discharge_from_soc_qh_mwh": combined_soc_result["combined_discharge_from_soc_qh"],
                "wholesale_charge_to_soc_qh_mwh": combined_soc_result["wholesale_charge_to_soc_qh"],
                "wholesale_pv_curtailed_to_batt_qh_mwh": combined_soc_result["wholesale_pv_curtailed_to_batt_qh"],
                "wholesale_discharge_from_soc_qh_mwh": combined_soc_result["wholesale_discharge_from_soc_qh"],
                "afrr_charge_to_soc_qh_mwh": combined_soc_result["afrr_charge_to_soc_qh"],
                "afrr_discharge_from_soc_qh_mwh": combined_soc_result["afrr_discharge_from_soc_qh"],
                "afrr_charge_market_qh_mwh": combined_soc_result["afrr_charge_market_qh"],
                "afrr_discharge_market_qh_mwh": combined_soc_result["afrr_discharge_market_qh"],
                "afrr_energy_down_activated": np.zeros(QH_PER_YEAR, dtype=int),
                "afrr_energy_up_activated": np.zeros(QH_PER_YEAR, dtype=int),
                "selected_charge_market_qh": np.full(QH_PER_YEAR, "none", dtype=object),
                "selected_charge_price_qh": np.full(QH_PER_YEAR, np.nan, dtype=float),
                "selected_discharge_channel_qh": np.full(QH_PER_YEAR, "none", dtype=object),
                "selected_discharge_market_qh": np.full(QH_PER_YEAR, "none", dtype=object),
                "selected_discharge_price_qh": np.full(QH_PER_YEAR, np.nan, dtype=float),
                "battery_soc_mwh_end_qh": combined_soc_result["combined_soc_qh"][1:],
            })
            combined_soc_hourly_end = combined_soc_result["combined_soc_hourly_end"]

        summary_df = build_summary_table(
            final_result,
            pv_stats,
            pure_pv_benchmark,
            pv_dc_mw,
            batt_power_mw,
            pv_capture_rate_pct,
            bess_capture_rate_pct,
            curtailment_outputs,
        )

        monthly_df = monthly_dataframe(final_result, pure_pv_benchmark, pv_dc_mw, batt_power_mw, curtailment_outputs)
        
        if enable_cfd:
             monthly_df["pv_only_cfd_revenue"] = (
                 monthly_df["pv_only_direct_mwh"] * cfd_price_standalone
             )
        else:
            monthly_df["pv_only_cfd_revenue"] = np.nan

        idx = build_quarter_hour_index(DEFAULT_YEAR)
        
        # === FIX SOC to include curtailed PV ===
        min_soc_mwh = sim_inputs.batt_energy_mwh * sim_inputs.min_soc_pct / 100.0
        max_soc_mwh = sim_inputs.batt_energy_mwh * sim_inputs.max_soc_pct / 100.0

        if reconciliation is not None:
            combined_soc_hourly_end = reconciliation["combined_soc_hourly_end_mwh"]
        else:
            hourly_charge_to_soc = (
                final_result["pv_to_batt"]
                + pv_curtailed_to_battery_actual
                + final_result["grid_charge"]
            ) * sim_inputs.eta_charge
        
            hourly_discharge_from_soc = (
                final_result["discharge"]
            ) / max(sim_inputs.eta_discharge, 1e-12)
        
            soc_hourly = np.zeros(QH_PER_YEAR + 1)
            soc_hourly[0] = min(max(sim_inputs.initial_soc_mwh, min_soc_mwh), max_soc_mwh)
        
            for t in range(QH_PER_YEAR):
                soc_hourly[t + 1] = min(
                    max(
                        soc_hourly[t]
                        + hourly_charge_to_soc[t]
                        - hourly_discharge_from_soc[t],
                        min_soc_mwh,
                    ),
                    max_soc_mwh,
                )
        
            combined_soc_hourly_end = soc_hourly[1:]
            
        hourly_df = pd.DataFrame({
            "datetime": idx,
            "base_pv_generation_mwh": base_pv_hourly_mwh,
            "pv_after_tso_dso_curtailment_mwh": pv_after_tso_dso,
            "pv_after_self_curtailment_mwh": pv_after_self,
            "pv_curtailment_candidate_mwh": pv_curtailment_candidate_mwh,
            "pv_curtailed_to_battery_mwh": pv_curtailed_to_battery_actual,
            "pv_curtailed_residual_lost_mwh": pv_curtailed_residual_lost_mwh,
            "tso_dso_curtailment_flag": tso_out["tso_dso_curtailment_flag"],
            "self_curtailment_flag": self_out["self_curtailment_flag"],
            "self_curtailment_reason": self_out["self_curtailment_reason"],
            "pv_commercial_structure": self_out["pv_commercial_structure_hourly"],
            "pv_price_raw_eur_per_mwh": pv_price_curve_raw,
            "pv_price_effective_eur_per_mwh": pv_effective_price_for_revenue,
            "pv_only_direct_mwh": pure_pv_benchmark["pv_only_direct_mwh"],
            "pv_only_revenue_eur": pure_pv_benchmark["pv_only_revenue_eur"],
            "battery_sell_price_raw_eur_per_mwh": batt_sell_curve_raw,
            "battery_sell_price_effective_eur_per_mwh": batt_sell_curve_effective,
            "grid_buy_price_raw_eur_per_mwh": grid_buy_curve_raw,
            "grid_buy_price_effective_eur_per_mwh": grid_buy_curve_effective,
            "pv_direct_mwh": final_result["pv_direct"],
            "pv_to_battery_mwh": final_result["pv_to_batt"],
            "grid_charge_mwh": final_result["grid_charge"],
            "battery_discharge_mwh": final_result["discharge"],
            "battery_soc_mwh_end": combined_soc_hourly_end,
            "pv_direct_revenue_eur": final_result["pv_direct_revenue"],
            "battery_sale_revenue_eur": final_result["batt_sale_revenue"],
            "bess_theoretical_revenue_without_capture_eur": final_result["bess_theoretical_revenue_without_capture_hourly_eur"],
            "bess_revenue_loss_due_to_capture_rate_eur": final_result["bess_revenue_loss_due_to_capture_rate_hourly_eur"],
            "grid_charge_cost_eur": final_result["grid_charge_cost"],
            "wholesale_cycle_cost_eur": final_result["wholesale_cycle_cost_eur"] if "wholesale_cycle_cost_eur" in final_result else np.zeros(QH_PER_YEAR),
            "avg_stored_charge_price_eur_per_mwh": final_result["avg_stored_charge_price"][1:],
            "required_discharge_price_eur_per_mwh": final_result["required_discharge_price"],
            "required_discharge_price_gate_estimate_eur_per_mwh": final_result["required_discharge_price_gate_estimate"],
            "afrr_charge_mwh": final_result["afrr_charge_hourly_mwh"] if "afrr_charge_hourly_mwh" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_discharge_mwh": final_result["afrr_discharge_hourly_mwh"] if "afrr_discharge_hourly_mwh" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_energy_down_activated": reconciliation["afrr_energy_down_activated_hourly"] if reconciliation is not None and "afrr_energy_down_activated_hourly" in reconciliation else np.zeros(QH_PER_YEAR),
            "afrr_energy_up_activated": reconciliation["afrr_energy_up_activated_hourly"] if reconciliation is not None and "afrr_energy_up_activated_hourly" in reconciliation else np.zeros(QH_PER_YEAR),
            "selected_charge_market": (pd.Series(reconciliation["selected_charge_market_qh"]).to_numpy() if reconciliation is not None and "selected_charge_market_qh" in reconciliation else np.full(QH_PER_YEAR, "none", dtype=object)),
            "selected_discharge_market": (pd.Series(reconciliation["selected_discharge_market_qh"]).to_numpy() if reconciliation is not None and "selected_discharge_market_qh" in reconciliation else np.full(QH_PER_YEAR, "none", dtype=object)),
            "afrr_charge_cost_eur": final_result["afrr_charge_cost_hourly_eur"] if "afrr_charge_cost_hourly_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_sale_revenue_eur": final_result["afrr_sale_revenue_hourly_eur"] if "afrr_sale_revenue_hourly_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_cycle_cost_eur": final_result["afrr_cycle_cost_hourly_eur"] if "afrr_cycle_cost_hourly_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_net_revenue_eur": final_result["afrr_net_revenue_hourly_eur"] if "afrr_net_revenue_hourly_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_capacity_up_price_eur_per_mw_h": afrr_capacity_up_price_h_raw if afrr_capacity_up_price_h_raw is not None else np.zeros(QH_PER_YEAR),
            "afrr_capacity_down_price_eur_per_mw_h": afrr_capacity_down_price_h_raw if afrr_capacity_down_price_h_raw is not None else np.zeros(QH_PER_YEAR),
            "afrr_capacity_selected_market": final_result["afrr_capacity_selected_market_h"] if "afrr_capacity_selected_market_h" in final_result else np.full(QH_PER_YEAR, "none", dtype=object),
            "afrr_capacity_up_awarded": final_result["afrr_capacity_up_awarded_h"] if "afrr_capacity_up_awarded_h" in final_result else np.zeros(QH_PER_YEAR, dtype=int),
            "afrr_capacity_down_awarded": final_result["afrr_capacity_down_awarded_h"] if "afrr_capacity_down_awarded_h" in final_result else np.zeros(QH_PER_YEAR, dtype=int),
            "afrr_certified_capacity_up_mw": final_result["afrr_certified_capacity_up_mw_h"] if "afrr_certified_capacity_up_mw_h" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_certified_capacity_down_mw": final_result["afrr_certified_capacity_down_mw_h"] if "afrr_certified_capacity_down_mw_h" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_capacity_up_revenue_eur": final_result["afrr_capacity_up_revenue_h_eur"] if "afrr_capacity_up_revenue_h_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_capacity_down_revenue_eur": final_result["afrr_capacity_down_revenue_h_eur"] if "afrr_capacity_down_revenue_h_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_capacity_total_revenue_eur": final_result["afrr_capacity_total_revenue_h_eur"] if "afrr_capacity_total_revenue_h_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_energy_down_activated": reconciliation["afrr_energy_down_activated_hourly"] if reconciliation is not None and "afrr_energy_down_activated_hourly" in reconciliation else np.zeros(QH_PER_YEAR),
            "afrr_energy_up_activated": reconciliation["afrr_energy_up_activated_hourly"] if reconciliation is not None and "afrr_energy_up_activated_hourly" in reconciliation else np.zeros(QH_PER_YEAR),
            "battery_blocked_by_afrr_capacity": final_result["battery_blocked_by_afrr_capacity"] if "battery_blocked_by_afrr_capacity" in final_result else np.zeros(QH_PER_YEAR, dtype=int),
            "wholesale_opportunity_value_eur": final_result.get("wholesale_opportunity_value_eur", np.zeros(QH_PER_YEAR)),
            "wholesale_expected_value_after_capture_rate_eur": final_result.get("wholesale_expected_value_after_capture_rate_eur", np.zeros(QH_PER_YEAR)),
            "raw_up_capacity_revenue_eur": final_result.get("raw_up_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "expected_up_capacity_revenue_eur": final_result.get("expected_up_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "raw_down_capacity_revenue_eur": final_result.get("raw_down_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "expected_down_capacity_revenue_eur": final_result.get("expected_down_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "expected_up_activated_mwh": final_result.get("expected_up_activated_mwh", np.zeros(QH_PER_YEAR)),
            "expected_down_activated_mwh": final_result.get("expected_down_activated_mwh", np.zeros(QH_PER_YEAR)),
            "afrr_up_energy_expected_value_eur": final_result.get("afrr_up_energy_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_down_energy_expected_value_eur": final_result.get("afrr_down_energy_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_up_total_expected_value_eur": final_result.get("afrr_up_total_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_down_total_expected_value_eur": final_result.get("afrr_down_total_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "selected_market": final_result.get("selected_market", np.full(QH_PER_YEAR, "none", dtype=object)),
            "selected_capacity_direction": final_result.get("selected_capacity_direction", np.full(QH_PER_YEAR, "none", dtype=object)),
            "afrr_capacity_success_rate_pct": final_result.get("afrr_capacity_success_rate_pct", np.zeros(QH_PER_YEAR)),
            "afrr_up_activation_pct": final_result.get("afrr_up_activation_pct", np.zeros(QH_PER_YEAR)),
            "afrr_down_activation_pct": final_result.get("afrr_down_activation_pct", np.zeros(QH_PER_YEAR)),
            "available_export_headroom_mwh": final_result.get("available_export_headroom_mwh", np.zeros(QH_PER_YEAR)),
            "available_soc_headroom_mwh": final_result.get("available_soc_headroom_mwh", np.zeros(QH_PER_YEAR)),
            "available_discharge_from_soc_mwh": final_result.get("available_discharge_from_soc_mwh", np.zeros(QH_PER_YEAR)),
            "required_up_soc_reserve_mwh": final_result.get("required_up_soc_reserve_mwh", np.zeros(QH_PER_YEAR)),
            "required_down_soc_headroom_mwh": final_result.get("required_down_soc_headroom_mwh", np.zeros(QH_PER_YEAR)),
            "expected_degradation_cost_eur": final_result.get("expected_degradation_cost_eur", np.zeros(QH_PER_YEAR)),
            "future_best_market_value_eur_per_mwh": final_result.get("future_best_market_value_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "future_best_market_type": final_result.get("future_best_market_type", np.full(QH_PER_YEAR, "none", dtype=object)),
            "cross_market_spread_eur_per_mwh": final_result.get("cross_market_spread_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "required_min_spread_eur_per_mwh": final_result.get("required_min_spread_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "spread_condition_respected": final_result.get("spread_condition_respected", np.zeros(QH_PER_YEAR)),
            "charge_reason": final_result.get("charge_reason", np.full(QH_PER_YEAR, "none", dtype=object)),
            "discharge_reason": final_result.get("discharge_reason", np.full(QH_PER_YEAR, "none", dtype=object)),
            "stored_energy_cost_eur_per_mwh": final_result.get("stored_energy_cost_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "effective_discharge_value_eur_per_mwh": final_result.get("effective_discharge_value_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "future_expected_afrr_up_value_eur": final_result.get("future_expected_afrr_up_value_eur", np.zeros(QH_PER_YEAR)),
            "future_expected_wholesale_value_eur": final_result.get("future_expected_wholesale_value_eur", np.zeros(QH_PER_YEAR)),
            "future_expected_best_discharge_market": final_result.get("future_expected_best_discharge_market", np.full(QH_PER_YEAR, "none", dtype=object)),
            "wholesale_charge_for_future_afrr_flag": final_result.get("wholesale_charge_for_future_afrr_flag", np.zeros(QH_PER_YEAR)),
            "afrr_down_charge_for_future_wholesale_flag": final_result.get("afrr_down_charge_for_future_wholesale_flag", np.zeros(QH_PER_YEAR)),
            "afrr_down_charge_for_future_afrr_up_flag": final_result.get("afrr_down_charge_for_future_afrr_up_flag", np.zeros(QH_PER_YEAR)),
            "wholesale_discharge_spread_ok": final_result.get("wholesale_discharge_spread_ok", np.zeros(QH_PER_YEAR)),
            "afrr_up_discharge_spread_ok": final_result.get("afrr_up_discharge_spread_ok", np.zeros(QH_PER_YEAR)),
            "forward_horizon_hours": final_result.get("forward_horizon_hours", np.zeros(QH_PER_YEAR)),
            "future_opportunity_selected": final_result.get("future_opportunity_selected", np.zeros(QH_PER_YEAR)),
            "forward_soc_before_capacity_selection_mwh": final_result.get("forward_soc_before_capacity_selection_mwh", np.zeros(QH_PER_YEAR)),
            "forward_soc_after_capacity_selection_mwh": final_result.get("forward_soc_after_capacity_selection_mwh", np.zeros(QH_PER_YEAR)),
            "afrr_up_soc_feasible": final_result.get("afrr_up_soc_feasible", np.zeros(QH_PER_YEAR)),
            "afrr_down_soc_feasible": final_result.get("afrr_down_soc_feasible", np.zeros(QH_PER_YEAR)),
            "afrr_up_rejected_due_to_soc": final_result.get("afrr_up_rejected_due_to_soc", np.zeros(QH_PER_YEAR)),
            "afrr_down_rejected_due_to_soc": final_result.get("afrr_down_rejected_due_to_soc", np.zeros(QH_PER_YEAR)),
            "afrr_up_rejected_due_to_final_combined_soc": final_result.get("afrr_up_rejected_due_to_final_combined_soc", np.zeros(QH_PER_YEAR)),
            "afrr_down_rejected_due_to_final_combined_soc": final_result.get("afrr_down_rejected_due_to_final_combined_soc", np.zeros(QH_PER_YEAR)),
            "afrr_up_expected_vs_actual_shortfall_mwh": reconciliation.get("afrr_up_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)) if reconciliation is not None else np.zeros(QH_PER_YEAR),
            "afrr_down_expected_vs_actual_shortfall_mwh": reconciliation.get("afrr_down_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)) if reconciliation is not None else np.zeros(QH_PER_YEAR),
            "annual_discharge_cap_mwh": final_result.get("annual_discharge_cap_mwh", np.full(QH_PER_YEAR, sim_inputs.max_cycles_per_year * sim_inputs.batt_energy_mwh)),
            "cumulative_battery_discharge_mwh": final_result.get("cumulative_battery_discharge_mwh", np.zeros(QH_PER_YEAR)),
            "remaining_discharge_budget_mwh": final_result.get("remaining_discharge_budget_mwh", np.zeros(QH_PER_YEAR)),
            "cycle_budget_used_pct": final_result.get("cycle_budget_used_pct", np.zeros(QH_PER_YEAR)),
            "cycle_budget_available_flag": final_result.get("cycle_budget_available_flag", np.zeros(QH_PER_YEAR)),
            "discharge_rejected_due_to_cycle_budget": final_result.get("discharge_rejected_due_to_cycle_budget", np.zeros(QH_PER_YEAR)),
            "wholesale_discharge_rejected_due_to_cycle_budget": final_result.get("wholesale_discharge_rejected_due_to_cycle_budget", np.zeros(QH_PER_YEAR)),
            "afrr_up_discharge_rejected_due_to_cycle_budget": final_result.get("afrr_up_discharge_rejected_due_to_cycle_budget", np.zeros(QH_PER_YEAR)),
            "afrr_up_capacity_rejected_due_to_cycle_budget": final_result.get("afrr_up_capacity_rejected_due_to_cycle_budget", np.zeros(QH_PER_YEAR)),
            "net_dispatch_value_eur_per_mwh": final_result.get("net_dispatch_value_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "cycle_budget_rank": final_result.get("cycle_budget_rank", np.zeros(QH_PER_YEAR)),
            "pv_capture_rate_pct": np.full(QH_PER_YEAR, pv_capture_rate_pct),
            "bess_capture_rate_pct": np.full(QH_PER_YEAR, bess_capture_rate_pct),
        })
        hourly_df["soc_expected_from_flows"] = (
            hourly_df["battery_soc_mwh_end"].shift(1).fillna(sim_inputs.initial_soc_mwh)
            + (
                hourly_df["pv_to_battery_mwh"]
                + hourly_df["pv_curtailed_to_battery_mwh"]
                + hourly_df["grid_charge_mwh"]
                + hourly_df["afrr_charge_mwh"]
            ) * sim_inputs.eta_charge
            - (
                hourly_df["battery_discharge_mwh"]
                + hourly_df["afrr_discharge_mwh"]
            ) / sim_inputs.eta_discharge
        )

        afrr_qh_df = None
        if reconciliation is not None:
            afrr_qh_df = pd.DataFrame({
                "datetime": reconciliation["datetime_qh"],
                "afrr_charge_price_raw_eur_per_mwh": afrr_charge_curve_qh_raw if afrr_charge_curve_qh_raw is not None else np.zeros(QH_PER_YEAR),
                "afrr_charge_price_effective_eur_per_mwh": sim_inputs.afrr_charge_price_qh,
                "afrr_discharge_price_raw_eur_per_mwh": afrr_discharge_curve_qh_raw if afrr_discharge_curve_qh_raw is not None else np.zeros(QH_PER_YEAR),
                "afrr_discharge_price_effective_eur_per_mwh": sim_inputs.afrr_discharge_price_qh,
                "afrr_charge_mwh": reconciliation["afrr_charge_qh_mwh"],
                "afrr_discharge_mwh": reconciliation["afrr_discharge_qh_mwh"],
                "expected_down_activated_mwh_from_capacity_selection": sim_inputs.afrr_expected_down_activated_mwh_qh if sim_inputs.afrr_expected_down_activated_mwh_qh is not None else np.zeros(QH_PER_YEAR),
                "expected_up_activated_mwh_from_capacity_selection": sim_inputs.afrr_expected_up_activated_mwh_qh if sim_inputs.afrr_expected_up_activated_mwh_qh is not None else np.zeros(QH_PER_YEAR),
                "afrr_down_activation_shortfall_mwh": reconciliation.get("afrr_down_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)),
                "afrr_up_activation_shortfall_mwh": reconciliation.get("afrr_up_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)),
                "afrr_energy_down_activated": reconciliation["afrr_energy_down_activated_qh"],
                "afrr_energy_up_activated": reconciliation["afrr_energy_up_activated_qh"],
                "selected_charge_market": reconciliation["selected_charge_market_qh"],
                "selected_charge_price_eur_per_mwh": reconciliation["selected_charge_price_qh"],
                "wholesale_grid_charge_mwh": reconciliation["wholesale_grid_charge_qh_mwh"],
                "wholesale_discharge_mwh": reconciliation["wholesale_discharge_qh_mwh"],
                "selected_discharge_channel": reconciliation["selected_discharge_channel_qh"],
                "selected_discharge_market": reconciliation["selected_discharge_market_qh"],
                "selected_discharge_price_eur_per_mwh": reconciliation["selected_discharge_price_qh"],
                "stored_energy_cost_eur_per_mwh": reconciliation.get("stored_energy_cost_eur_per_mwh", np.zeros(QH_PER_YEAR)),
                "effective_discharge_value_eur_per_mwh": reconciliation.get("effective_discharge_value_eur_per_mwh", np.zeros(QH_PER_YEAR)),
                "spread_condition_respected": reconciliation.get("spread_condition_respected", np.zeros(QH_PER_YEAR)),
                "wholesale_discharge_spread_ok": reconciliation.get("wholesale_discharge_spread_ok", np.zeros(QH_PER_YEAR)),
                "afrr_up_discharge_spread_ok": reconciliation.get("afrr_up_discharge_spread_ok", np.zeros(QH_PER_YEAR)),
                "afrr_capacity_selected_market": reconciliation["afrr_capacity_selected_market_qh"],
                "combined_soc_mwh": reconciliation["combined_soc_qh"][1:],
                "afrr_charge_cost_eur": reconciliation["afrr_charge_cost_qh_eur"],
                "afrr_sale_revenue_eur": reconciliation["afrr_sale_revenue_qh_eur"],
                "afrr_cycle_cost_eur": reconciliation["afrr_cycle_cost_qh_eur"],
                "afrr_net_revenue_eur": reconciliation["afrr_net_revenue_qh_eur"],
                "bess_capture_rate_pct": np.full(QH_PER_YEAR, bess_capture_rate_pct),
            })

        afrr_capacity_df = pd.DataFrame({
            "datetime": idx,
            "afrr_capacity_up_price_eur_per_mw_h": afrr_capacity_up_price_h_raw if afrr_capacity_up_price_h_raw is not None else np.zeros(QH_PER_YEAR),
            "afrr_capacity_down_price_eur_per_mw_h": afrr_capacity_down_price_h_raw if afrr_capacity_down_price_h_raw is not None else np.zeros(QH_PER_YEAR),
            "afrr_capacity_selected_market": final_result["afrr_capacity_selected_market_h"] if "afrr_capacity_selected_market_h" in final_result else np.full(QH_PER_YEAR, "none", dtype=object),
            "afrr_capacity_up_awarded": final_result["afrr_capacity_up_awarded_h"] if "afrr_capacity_up_awarded_h" in final_result else np.zeros(QH_PER_YEAR, dtype=int),
            "afrr_capacity_down_awarded": final_result["afrr_capacity_down_awarded_h"] if "afrr_capacity_down_awarded_h" in final_result else np.zeros(QH_PER_YEAR, dtype=int),
            "afrr_certified_capacity_up_mw": final_result["afrr_certified_capacity_up_mw_h"] if "afrr_certified_capacity_up_mw_h" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_certified_capacity_down_mw": final_result["afrr_certified_capacity_down_mw_h"] if "afrr_certified_capacity_down_mw_h" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_capacity_up_revenue_eur": final_result["afrr_capacity_up_revenue_h_eur"] if "afrr_capacity_up_revenue_h_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_capacity_down_revenue_eur": final_result["afrr_capacity_down_revenue_h_eur"] if "afrr_capacity_down_revenue_h_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_capacity_total_revenue_eur": final_result["afrr_capacity_total_revenue_h_eur"] if "afrr_capacity_total_revenue_h_eur" in final_result else np.zeros(QH_PER_YEAR),
            "afrr_energy_down_activated": reconciliation["afrr_energy_down_activated_hourly"] if reconciliation is not None and "afrr_energy_down_activated_hourly" in reconciliation else np.zeros(QH_PER_YEAR),
            "afrr_energy_up_activated": reconciliation["afrr_energy_up_activated_hourly"] if reconciliation is not None and "afrr_energy_up_activated_hourly" in reconciliation else np.zeros(QH_PER_YEAR),
            "battery_blocked_by_afrr_capacity": final_result["battery_blocked_by_afrr_capacity"] if "battery_blocked_by_afrr_capacity" in final_result else np.zeros(QH_PER_YEAR, dtype=int),
            "wholesale_opportunity_value_eur": final_result.get("wholesale_opportunity_value_eur", np.zeros(QH_PER_YEAR)),
            "wholesale_expected_value_after_capture_rate_eur": final_result.get("wholesale_expected_value_after_capture_rate_eur", np.zeros(QH_PER_YEAR)),
            "raw_up_capacity_revenue_eur": final_result.get("raw_up_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "expected_up_capacity_revenue_eur": final_result.get("expected_up_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "raw_down_capacity_revenue_eur": final_result.get("raw_down_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "expected_down_capacity_revenue_eur": final_result.get("expected_down_capacity_revenue_eur", np.zeros(QH_PER_YEAR)),
            "expected_up_activated_mwh": final_result.get("expected_up_activated_mwh", np.zeros(QH_PER_YEAR)),
            "expected_down_activated_mwh": final_result.get("expected_down_activated_mwh", np.zeros(QH_PER_YEAR)),
            "afrr_up_energy_expected_value_eur": final_result.get("afrr_up_energy_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_down_energy_expected_value_eur": final_result.get("afrr_down_energy_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_up_total_expected_value_eur": final_result.get("afrr_up_total_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "afrr_down_total_expected_value_eur": final_result.get("afrr_down_total_expected_value_eur", np.zeros(QH_PER_YEAR)),
            "selected_market": final_result.get("selected_market", np.full(QH_PER_YEAR, "none", dtype=object)),
            "selected_capacity_direction": final_result.get("selected_capacity_direction", np.full(QH_PER_YEAR, "none", dtype=object)),
            "afrr_capacity_success_rate_pct": final_result.get("afrr_capacity_success_rate_pct", np.zeros(QH_PER_YEAR)),
            "afrr_up_activation_pct": final_result.get("afrr_up_activation_pct", np.zeros(QH_PER_YEAR)),
            "afrr_down_activation_pct": final_result.get("afrr_down_activation_pct", np.zeros(QH_PER_YEAR)),
            "available_export_headroom_mwh": final_result.get("available_export_headroom_mwh", np.zeros(QH_PER_YEAR)),
            "available_soc_headroom_mwh": final_result.get("available_soc_headroom_mwh", np.zeros(QH_PER_YEAR)),
            "available_discharge_from_soc_mwh": final_result.get("available_discharge_from_soc_mwh", np.zeros(QH_PER_YEAR)),
            "required_up_soc_reserve_mwh": final_result.get("required_up_soc_reserve_mwh", np.zeros(QH_PER_YEAR)),
            "required_down_soc_headroom_mwh": final_result.get("required_down_soc_headroom_mwh", np.zeros(QH_PER_YEAR)),
            "expected_degradation_cost_eur": final_result.get("expected_degradation_cost_eur", np.zeros(QH_PER_YEAR)),
            "future_best_market_value_eur_per_mwh": final_result.get("future_best_market_value_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "future_best_market_type": final_result.get("future_best_market_type", np.full(QH_PER_YEAR, "none", dtype=object)),
            "cross_market_spread_eur_per_mwh": final_result.get("cross_market_spread_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "required_min_spread_eur_per_mwh": final_result.get("required_min_spread_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "spread_condition_respected": final_result.get("spread_condition_respected", np.zeros(QH_PER_YEAR)),
            "charge_reason": final_result.get("charge_reason", np.full(QH_PER_YEAR, "none", dtype=object)),
            "discharge_reason": final_result.get("discharge_reason", np.full(QH_PER_YEAR, "none", dtype=object)),
            "stored_energy_cost_eur_per_mwh": final_result.get("stored_energy_cost_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "effective_discharge_value_eur_per_mwh": final_result.get("effective_discharge_value_eur_per_mwh", np.zeros(QH_PER_YEAR)),
            "future_expected_afrr_up_value_eur": final_result.get("future_expected_afrr_up_value_eur", np.zeros(QH_PER_YEAR)),
            "future_expected_wholesale_value_eur": final_result.get("future_expected_wholesale_value_eur", np.zeros(QH_PER_YEAR)),
            "future_expected_best_discharge_market": final_result.get("future_expected_best_discharge_market", np.full(QH_PER_YEAR, "none", dtype=object)),
            "wholesale_charge_for_future_afrr_flag": final_result.get("wholesale_charge_for_future_afrr_flag", np.zeros(QH_PER_YEAR)),
            "afrr_down_charge_for_future_wholesale_flag": final_result.get("afrr_down_charge_for_future_wholesale_flag", np.zeros(QH_PER_YEAR)),
            "afrr_down_charge_for_future_afrr_up_flag": final_result.get("afrr_down_charge_for_future_afrr_up_flag", np.zeros(QH_PER_YEAR)),
            "wholesale_discharge_spread_ok": final_result.get("wholesale_discharge_spread_ok", np.zeros(QH_PER_YEAR)),
            "afrr_up_discharge_spread_ok": final_result.get("afrr_up_discharge_spread_ok", np.zeros(QH_PER_YEAR)),
            "forward_horizon_hours": final_result.get("forward_horizon_hours", np.zeros(QH_PER_YEAR)),
            "future_opportunity_selected": final_result.get("future_opportunity_selected", np.zeros(QH_PER_YEAR)),
            "forward_soc_before_capacity_selection_mwh": final_result.get("forward_soc_before_capacity_selection_mwh", np.zeros(QH_PER_YEAR)),
            "forward_soc_after_capacity_selection_mwh": final_result.get("forward_soc_after_capacity_selection_mwh", np.zeros(QH_PER_YEAR)),
            "afrr_up_soc_feasible": final_result.get("afrr_up_soc_feasible", np.zeros(QH_PER_YEAR)),
            "afrr_down_soc_feasible": final_result.get("afrr_down_soc_feasible", np.zeros(QH_PER_YEAR)),
            "afrr_up_rejected_due_to_soc": final_result.get("afrr_up_rejected_due_to_soc", np.zeros(QH_PER_YEAR)),
            "afrr_down_rejected_due_to_soc": final_result.get("afrr_down_rejected_due_to_soc", np.zeros(QH_PER_YEAR)),
            "afrr_up_rejected_due_to_final_combined_soc": final_result.get("afrr_up_rejected_due_to_final_combined_soc", np.zeros(QH_PER_YEAR)),
            "afrr_down_rejected_due_to_final_combined_soc": final_result.get("afrr_down_rejected_due_to_final_combined_soc", np.zeros(QH_PER_YEAR)),
            "afrr_up_expected_vs_actual_shortfall_mwh": reconciliation.get("afrr_up_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)) if reconciliation is not None else np.zeros(QH_PER_YEAR),
            "afrr_down_expected_vs_actual_shortfall_mwh": reconciliation.get("afrr_down_activation_shortfall_qh_mwh", np.zeros(QH_PER_YEAR)) if reconciliation is not None else np.zeros(QH_PER_YEAR),
        })

        inputs_df = build_inputs_dataframe(sim_inputs)
        excel_bytes = to_excel_bytes(
            inputs_df=inputs_df,
            summary_df=summary_df,
            monthly_df=monthly_df,
            hourly_df=hourly_df,
            afrr_qh_df=afrr_qh_df,
            afrr_daily_log_df=afrr_result["afrr_daily_log"] if afrr_result is not None else None,
            afrr_capacity_df=afrr_capacity_df if enable_afrr_capacity else None,
            bess_degradation_df=bess_degradation_df,
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time < 60:
            optimization_time_str = f"{elapsed_time:.2f} seconds"
        else:
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            optimization_time_str = f"{minutes} min {seconds} sec"

        st.subheader("Optimization Time")
        st.write(optimization_time_str)

        st.success("Simulation terminée.")

        k1, k2, k3, k4 = st.columns(4)
        if "total_revenue_including_afrr_capacity_eur" in final_result:
            total_revenue_display = final_result["total_revenue_including_afrr_capacity_eur"][0]
        elif "total_revenue_including_afrr_eur" in final_result:
            total_revenue_display = final_result["total_revenue_including_afrr_eur"][0]
        else:
            total_revenue_display = final_result["total_revenue"][0]
        total_energy_display = final_result["energy_sold_total_mwh"][0] + (np.sum(final_result["afrr_discharge_hourly_mwh"]) if "afrr_discharge_hourly_mwh" in final_result else 0.0)

        k1.metric("Revenu total", f"{total_revenue_display:,.0f} EUR")
        k2.metric("Énergie totale vendue", f"{total_energy_display:,.0f} MWh")
        k3.metric("Énergie shiftée", f"{final_result['energy_shifted_mwh'][0]:,.0f} MWh")
        k4.metric("Cycles équivalents", f"{final_result['equivalent_cycles'][0]:,.1f}")

        st.subheader("BESS availability")
        b1, b2, b3 = st.columns(3)
        b1.metric("Nominal BESS Energy Capacity", f"{batt_energy_mwh:,.2f} MWh")
        b2.metric("BESS Availability", f"{bess_availability_pct:,.1f} %")
        b3.metric("Effective BESS Energy Capacity", f"{effective_batt_energy_mwh:,.2f} MWh")

        st.subheader("Synthèse")
        summary_display_df = format_synthese_table_for_display(summary_df)
        summary_display_styler = summary_display_df.style.set_properties(
            subset=["Valeur"],
            **{"text-align": "right"}
        )
        st.dataframe(summary_display_styler, use_container_width=True, hide_index=True)

        debug = hourly_df[
            (hourly_df["datetime"] >= pd.Timestamp(f"{DEFAULT_YEAR}-06-01 00:00:00")) &
            (hourly_df["datetime"] < pd.Timestamp(f"{DEFAULT_YEAR}-06-04 00:00:00"))
        ].copy()

        st.subheader("Debug curtailment (3 premiers jours de juin)")
        st.dataframe(
            debug[[
                "datetime",
                "base_pv_generation_mwh",
                "pv_after_tso_dso_curtailment_mwh",
                "pv_after_self_curtailment_mwh",
                "pv_curtailment_candidate_mwh",
                "pv_curtailed_to_battery_mwh",
                "pv_curtailed_residual_lost_mwh",
                "pv_price_raw_eur_per_mwh",
                "pv_price_effective_eur_per_mwh",
                "self_curtailment_flag",
                "self_curtailment_reason",
                "pv_commercial_structure",
            ]],
            use_container_width=True,
        )

        st.subheader("Debug batterie - 5 premiers jours de juin")

        battery_debug = hourly_df[
            (hourly_df["datetime"] >= pd.Timestamp(f"{DEFAULT_YEAR}-06-01 00:00:00")) &
            (hourly_df["datetime"] < pd.Timestamp(f"{DEFAULT_YEAR}-06-06 00:00:00"))
        ].copy()

        battery_debug["total_battery_charge_mwh"] = (
            battery_debug["pv_to_battery_mwh"]
            + battery_debug["grid_charge_mwh"]
            + battery_debug["afrr_charge_mwh"]
            + battery_debug["pv_curtailed_to_battery_mwh"]
        )

        battery_debug["spread_check"] = (
            battery_debug["battery_sell_price_effective_eur_per_mwh"]
            - battery_debug["grid_buy_price_effective_eur_per_mwh"]
        )

        battery_debug["total_battery_discharge_mwh"] = (
            battery_debug["battery_discharge_mwh"]
            + battery_debug["afrr_discharge_mwh"]
        )

        battery_debug["wholesale_charge_price_eur_per_mwh"] = np.where(
            battery_debug["grid_charge_mwh"] > 1e-9,
            battery_debug["grid_buy_price_effective_eur_per_mwh"],
            np.where(
                battery_debug["pv_to_battery_mwh"] > 1e-9,
                battery_debug["pv_price_effective_eur_per_mwh"],
                np.nan,
            )
        )

        battery_debug["wholesale_discharge_price_eur_per_mwh"] = np.where(
            battery_debug["battery_discharge_mwh"] > 1e-9,
            battery_debug["battery_sell_price_effective_eur_per_mwh"],
            np.nan,
        )

        battery_debug["battery_activity"] = np.select(
            [
                battery_debug["total_battery_charge_mwh"] > 1e-9,
                battery_debug["total_battery_discharge_mwh"] > 1e-9,
            ],
            [
                "Charging",
                "Discharging",
            ],
            default="Idle",
        )
        # aFRR Capacity awarded MW and winning prices
        battery_debug["afrr_capacity_up_won_mw"] = np.where(
            battery_debug["afrr_capacity_up_awarded"] == 1,
            battery_debug["afrr_certified_capacity_up_mw"],
            0.0,
        )
        
        battery_debug["afrr_capacity_down_won_mw"] = np.where(
            battery_debug["afrr_capacity_down_awarded"] == 1,
            battery_debug["afrr_certified_capacity_down_mw"],
            0.0,
        )
        
        battery_debug["afrr_capacity_winning_price_eur_per_mw_h"] = np.select(
            [
                battery_debug["afrr_capacity_up_awarded"] == 1,
                battery_debug["afrr_capacity_down_awarded"] == 1,
            ],
            [
                battery_debug["afrr_capacity_up_price_eur_per_mw_h"],
                battery_debug["afrr_capacity_down_price_eur_per_mw_h"],
            ],
            default=np.nan,
        )
        
        battery_debug["afrr_capacity_winning_direction"] = np.select(
            [
                battery_debug["afrr_capacity_up_awarded"] == 1,
                battery_debug["afrr_capacity_down_awarded"] == 1,
            ],
            [
                "UP",
                "DOWN",
            ],
            default="None",
        )
        st.dataframe(
            battery_debug[[
                "datetime",
                "battery_activity",
                "battery_soc_mwh_end",
                "total_battery_charge_mwh",
                "total_battery_discharge_mwh",
                "pv_to_battery_mwh",
                "pv_curtailed_to_battery_mwh",
                "grid_charge_mwh",
                "afrr_charge_mwh",
                "battery_discharge_mwh",
                "afrr_discharge_mwh",
                "afrr_capacity_winning_direction",
                "afrr_capacity_up_won_mw",
                "afrr_capacity_down_won_mw",
                "afrr_capacity_winning_price_eur_per_mw_h",
                "wholesale_charge_price_eur_per_mwh",
                "avg_stored_charge_price_eur_per_mwh",
                "required_discharge_price_eur_per_mwh",
                "wholesale_discharge_price_eur_per_mwh",
                "battery_sale_revenue_eur",
                "grid_charge_cost_eur",
                "wholesale_cycle_cost_eur",
                "afrr_charge_cost_eur",
                "afrr_sale_revenue_eur",
                "afrr_cycle_cost_eur",
                "afrr_net_revenue_eur",
            ]],
            use_container_width=True,
            hide_index=True,
        )
        
        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(8, 4.5))
            bars = [
                float(final_result["total_direct_pv_revenue"][0]) / 1e6,
                float(final_result["total_batt_sale_revenue"][0]) / 1e6,
                -float(final_result["total_grid_charge_cost"][0]) / 1e6,
                float(final_result["nightly_revenue_total"][0]) / 1e6,
                float(final_result["total_afrr_net_revenue_eur"][0]) / 1e6 if "total_afrr_net_revenue_eur" in final_result else 0.0,
                float(final_result["total_afrr_capacity_revenue_eur"][0]) / 1e6 if "total_afrr_capacity_revenue_eur" in final_result else 0.0,
                float(pure_pv_benchmark["total_pv_only_revenue_eur"][0]) / 1e6,
            ]
        
            labels = [
                "PV direct",
                "BESS Wholesale",
                "BESS Charging Cost",
                "SS nuit",
                "BESS aFRR Energy",
                "BESS aFRR Capacity",
                "PV-only",
            ]
        
            ax1.bar(labels, bars)
        
            # Thin horizontal zero line
            ax1.axhline(0, linewidth=0.8, color="black")
        
            # Remove scientific notation / 1e6 offset on y-axis
            ax1.ticklabel_format(axis="y", style="plain", useOffset=False)
        
            ax1.set_title("Revenue Breakdown")
            ax1.set_ylabel("million €")
            ax1.tick_params(axis="x", rotation=20)
        
            st.pyplot(fig1)
            plt.close(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(9, 4.8))

            x = np.arange(len(monthly_df))
            afrr_vals = monthly_df["afrr_net_revenue"].to_numpy(dtype=float) / max(batt_power_mw, 1e-12) / 1000.0
            afrr_capacity_vals = monthly_df["afrr_capacity_total_revenue"].to_numpy(dtype=float) / max(batt_power_mw, 1e-12) / 1000.0 if "afrr_capacity_total_revenue" in monthly_df.columns else np.zeros(len(monthly_df))
            bess_vals = monthly_df["bess_revenue_keur_per_mw"].to_numpy(dtype=float) - afrr_vals - afrr_capacity_vals

            ax2.bar(x, bess_vals, width=0.65, color="lightgreen", label="DA Arbitrage")
            ax2.bar(x, afrr_capacity_vals, width=0.65, bottom=bess_vals, label="aFRR Capacity")
            ax2.bar(x, afrr_vals, width=0.65, bottom=bess_vals + afrr_capacity_vals, color="blue", label="aFRR Energy")

            ax2.set_title("BESS Specific Monthly Revenues per MW")
            ax2.set_ylabel("k€/MW")
            ax2.set_xlabel("Month")
            ax2.set_xticks(x)
            ax2.set_xticklabels(monthly_df["month"], rotation=45)
            ax2.legend()

            st.pyplot(fig2)
            plt.close(fig2)

        c3, c4 = st.columns(2)

        with c3:
            fig3, ax3 = plt.subplots(figsize=(8, 4.5))

            ax3.plot(monthly_df["month"], monthly_df["pv_direct_mwh"], label="PV direct")
            ax3.plot(monthly_df["month"], monthly_df["shifted_mwh"], label="Énergie shiftée wholesale")
            ax3.plot(monthly_df["month"], monthly_df["pv_only_direct_mwh"], label="PV-only direct")

            if "afrr_discharge_mwh" in monthly_df.columns:
                ax3.plot(monthly_df["month"], monthly_df["afrr_discharge_mwh"], label="Décharge aFRR")

            if "pv_curtailment_candidate_mwh" in monthly_df.columns:
                ax3.plot(
                    monthly_df["month"],
                    monthly_df["pv_curtailment_candidate_mwh"],
                    linestyle="--",
                    marker="o",
                    label="PV curtailed"
                )

            if "pv_curtailed_to_battery_mwh_actual" in monthly_df.columns:
                ax3.plot(
                    monthly_df["month"],
                    monthly_df["pv_curtailed_to_battery_mwh_actual"],
                    linestyle="--",
                    marker="o",
                    label="PV curtailed → battery"
                )

            if "pv_curtailed_residual_lost_mwh" in monthly_df.columns:
                ax3.plot(
                    monthly_df["month"],
                    monthly_df["pv_curtailed_residual_lost_mwh"],
                    linestyle="--",
                    marker="o",
                    label="PV curtailed lost"
                )

            ax3.set_title("Énergies valorisées par mois")
            ax3.set_ylabel("MWh")
            ax3.set_xlabel("Mois")
            ax3.legend()
            ax3.tick_params(axis="x", rotation=45)
            st.pyplot(fig3)
            plt.close(fig3)

        with c4:
            start_date = pd.Timestamp(f"{DEFAULT_YEAR}-06-01 00:00:00")
            end_date = start_date + pd.Timedelta(hours=120)

            df_plot = hourly_df[
                (hourly_df["datetime"] >= start_date) &
                (hourly_df["datetime"] < end_date)
            ].copy()

            fig, ax1 = plt.subplots(figsize=(12, 5))
            bar_width = 0.03

            ax1.fill_between(
                df_plot["datetime"],
                df_plot["pv_direct_mwh"],
                color="orange",
                alpha=0.5,
                label="PV → Réseau"
            )
            ax1.plot(
                df_plot["datetime"],
                df_plot["pv_direct_mwh"],
                color="orange",
                linewidth=1.8
            )

            ax1.bar(
                df_plot["datetime"],
                df_plot["battery_discharge_mwh"],
                width=bar_width,
                label="Batterie → Réseau (wholesale)",
                alpha=0.8,
                color="green"
            )

            ax1.bar(
                df_plot["datetime"],
                -df_plot["pv_to_battery_mwh"],
                width=bar_width,
                label="PV → Batterie",
                alpha=0.6,
                color="red"
            )

            ax1.bar(
                df_plot["datetime"],
                -df_plot["grid_charge_mwh"],
                width=bar_width,
                bottom=-df_plot["pv_to_battery_mwh"],
                label="Réseau → Batterie",
                alpha=0.6
            )

            if "afrr_discharge_mwh" in df_plot.columns:
                ax1.bar(
                    df_plot["datetime"],
                    df_plot["afrr_discharge_mwh"],
                    width=bar_width,
                    label="aFRR → Décharge",
                    alpha=0.5,
                    color="purple"
                )

            if "afrr_charge_mwh" in df_plot.columns:
                ax1.bar(
                    df_plot["datetime"],
                    -df_plot["afrr_charge_mwh"],
                    width=bar_width,
                    label="aFRR → Charge",
                    alpha=0.5,
                    color="blue"
                )

            if "pv_curtailment_candidate_mwh" in df_plot.columns:
                ax1.plot(
                    df_plot["datetime"],
                    df_plot["pv_curtailment_candidate_mwh"],
                    linestyle="--",
                    linewidth=1.5,
                    label="PV curtailed"
                )

            if "pv_curtailed_to_battery_mwh" in df_plot.columns:
                ax1.bar(
                    df_plot["datetime"],
                    -df_plot["pv_curtailed_to_battery_mwh"],
                    width=bar_width,
                    label="PV curtailed → battery",
                    alpha=0.6
                )

            if "pv_curtailed_residual_lost_mwh" in df_plot.columns:
                ax1.plot(
                    df_plot["datetime"],
                    df_plot["pv_curtailed_residual_lost_mwh"],
                    linestyle=":",
                    linewidth=1.8,
                    label="PV curtailed lost"
                )

            ax1.axhline(0, linewidth=1)
            ax1.set_ylabel("Flux énergie (MWh)")
            ax1.set_xlabel("Heure")
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Hh"))
            ax1.tick_params(axis="x", rotation=0)

            ax2 = ax1.twinx()
            ax2.plot(
                df_plot["datetime"],
                df_plot["pv_price_effective_eur_per_mwh"],
                linestyle="--",
                alpha=0.7,
                label="Prix spot PV effectif"
            )
            ax2.set_ylabel("Prix (EUR/MWh)")

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=3,
                frameon=False,
            )
            ax1.set_title("Dispatch énergétique - 5 premiers jours de juin")
            fig.tight_layout(rect=[0, 0.12, 1, 1])

            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Dispatch énergétique - 5 derniers jours de mai + 5 premiers jours de juin")

            start_date = pd.Timestamp(f"{DEFAULT_YEAR}-05-27 00:00:00")
            end_date = pd.Timestamp(f"{DEFAULT_YEAR}-06-06 00:00:00")
            
            df_plot = hourly_df[
                (hourly_df["datetime"] >= start_date) &
                (hourly_df["datetime"] < end_date)
            ].copy()
            
            fig, ax1 = plt.subplots(figsize=(14, 5))
            bar_width = 0.03
            
            ax1.fill_between(
                df_plot["datetime"],
                df_plot["pv_direct_mwh"],
                color="orange",
                alpha=0.5,
                label="PV → Réseau"
            )
            ax1.plot(
                df_plot["datetime"],
                df_plot["pv_direct_mwh"],
                color="orange",
                linewidth=1.8
            )
            
            ax1.bar(
                df_plot["datetime"],
                df_plot["battery_discharge_mwh"],
                width=bar_width,
                label="Batterie → Réseau (wholesale)",
                alpha=0.8,
                color="green"
            )
            
            ax1.bar(
                df_plot["datetime"],
                -df_plot["pv_to_battery_mwh"],
                width=bar_width,
                label="PV → Batterie",
                alpha=0.6,
                color="red"
            )
            
            ax1.bar(
                df_plot["datetime"],
                -df_plot["grid_charge_mwh"],
                width=bar_width,
                bottom=-df_plot["pv_to_battery_mwh"],
                label="Réseau → Batterie",
                alpha=0.6
            )
            
            if "afrr_discharge_mwh" in df_plot.columns:
                ax1.bar(
                    df_plot["datetime"],
                    df_plot["afrr_discharge_mwh"],
                    width=bar_width,
                    label="aFRR → Décharge",
                    alpha=0.5,
                    color="purple"
                )
            
            if "afrr_charge_mwh" in df_plot.columns:
                ax1.bar(
                    df_plot["datetime"],
                    -df_plot["afrr_charge_mwh"],
                    width=bar_width,
                    label="aFRR → Charge",
                    alpha=0.5,
                    color="blue"
                )
            
            if "pv_curtailment_candidate_mwh" in df_plot.columns:
                ax1.plot(
                    df_plot["datetime"],
                    df_plot["pv_curtailment_candidate_mwh"],
                    linestyle="--",
                    linewidth=1.5,
                    label="PV curtailed"
                )
            
            if "pv_curtailed_to_battery_mwh" in df_plot.columns:
                ax1.bar(
                    df_plot["datetime"],
                    -df_plot["pv_curtailed_to_battery_mwh"],
                    width=bar_width,
                    label="PV curtailed → battery",
                    alpha=0.6
                )
            
            if "pv_curtailed_residual_lost_mwh" in df_plot.columns:
                ax1.plot(
                    df_plot["datetime"],
                    df_plot["pv_curtailed_residual_lost_mwh"],
                    linestyle=":",
                    linewidth=1.8,
                    label="PV curtailed lost"
                )
            
            ax1.axhline(0, linewidth=1)
            ax1.set_ylabel("Flux énergie (MWh)")
            ax1.set_xlabel("Heure")
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            ax1.tick_params(axis="x", rotation=45)
            
            ax2 = ax1.twinx()
            ax2.plot(
                df_plot["datetime"],
                df_plot["pv_price_effective_eur_per_mwh"],
                linestyle="--",
                alpha=0.7,
                label="Prix spot PV effectif"
            )
            ax2.set_ylabel("Prix (EUR/MWh)")
            
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=3,
                frameon=False,
            )
            
            ax1.set_title("Dispatch énergétique - 5 derniers jours de mai + 5 premiers jours de juin")
            fig.tight_layout(rect=[0, 0.12, 1, 1])
            
            st.pyplot(fig)
            plt.close(fig)

            # === Dispatch énergétique - mois complet de juin, par blocs de 10 jours ===
            st.subheader("Dispatch énergétique - mois complet de juin par blocs de 10 jours")
            
            june_blocks = [
                (
                    pd.Timestamp(f"{DEFAULT_YEAR}-06-01 00:00:00"),
                    pd.Timestamp(f"{DEFAULT_YEAR}-06-11 00:00:00"),
                    "1-10 juin",
                ),
                (
                    pd.Timestamp(f"{DEFAULT_YEAR}-06-11 00:00:00"),
                    pd.Timestamp(f"{DEFAULT_YEAR}-06-21 00:00:00"),
                    "11-20 juin",
                ),
                (
                    pd.Timestamp(f"{DEFAULT_YEAR}-06-21 00:00:00"),
                    pd.Timestamp(f"{DEFAULT_YEAR}-07-01 00:00:00"),
                    "21-30 juin",
                ),
            ]
            
            fig, axes = plt.subplots(
                nrows=3,
                ncols=1,
                figsize=(16, 12),
                sharey=True,
            )
            
            bar_width = 0.03
            legend_handles = []
            legend_labels = []
            
            for ax1, (start_date, end_date, block_title) in zip(axes, june_blocks):
                df_plot = hourly_df[
                    (hourly_df["datetime"] >= start_date) &
                    (hourly_df["datetime"] < end_date)
                ].copy()
            
                ax1.fill_between(
                    df_plot["datetime"],
                    df_plot["pv_direct_mwh"],
                    color="orange",
                    alpha=0.5,
                    label="PV → Réseau"
                )
                ax1.plot(
                    df_plot["datetime"],
                    df_plot["pv_direct_mwh"],
                    color="orange",
                    linewidth=1.8
                )
            
                ax1.bar(
                    df_plot["datetime"],
                    df_plot["battery_discharge_mwh"],
                    width=bar_width,
                    label="Batterie → Réseau (wholesale)",
                    alpha=0.8,
                    color="green"
                )
            
                ax1.bar(
                    df_plot["datetime"],
                    -df_plot["pv_to_battery_mwh"],
                    width=bar_width,
                    label="PV → Batterie",
                    alpha=0.6,
                    color="red"
                )
            
                ax1.bar(
                    df_plot["datetime"],
                    -df_plot["grid_charge_mwh"],
                    width=bar_width,
                    bottom=-df_plot["pv_to_battery_mwh"],
                    label="Réseau → Batterie",
                    alpha=0.6
                )
            
                if "afrr_discharge_mwh" in df_plot.columns:
                    ax1.bar(
                        df_plot["datetime"],
                        df_plot["afrr_discharge_mwh"],
                        width=bar_width,
                        label="aFRR → Décharge",
                        alpha=0.5,
                        color="purple"
                    )
            
                if "afrr_charge_mwh" in df_plot.columns:
                    ax1.bar(
                        df_plot["datetime"],
                        -df_plot["afrr_charge_mwh"],
                        width=bar_width,
                        label="aFRR → Charge",
                        alpha=0.5,
                        color="blue"
                    )
            
                if "pv_curtailment_candidate_mwh" in df_plot.columns:
                    ax1.plot(
                        df_plot["datetime"],
                        df_plot["pv_curtailment_candidate_mwh"],
                        linestyle="--",
                        linewidth=1.5,
                        label="PV curtailed"
                    )
            
                if "pv_curtailed_to_battery_mwh" in df_plot.columns:
                    ax1.bar(
                        df_plot["datetime"],
                        -df_plot["pv_curtailed_to_battery_mwh"],
                        width=bar_width,
                        label="PV curtailed → battery",
                        alpha=0.6
                    )
            
                if "pv_curtailed_residual_lost_mwh" in df_plot.columns:
                    ax1.plot(
                        df_plot["datetime"],
                        df_plot["pv_curtailed_residual_lost_mwh"],
                        linestyle=":",
                        linewidth=1.8,
                        label="PV curtailed lost"
                    )
            
                ax1.axhline(0, linewidth=1)
                ax1.set_ylabel("Flux énergie (MWh)")
                ax1.set_title(f"Dispatch énergétique - {block_title}")
                ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
                ax1.tick_params(axis="x", rotation=45)
            
                ax2 = ax1.twinx()
                ax2.plot(
                    df_plot["datetime"],
                    df_plot["pv_price_effective_eur_per_mwh"],
                    linestyle="--",
                    alpha=0.7,
                    label="Prix spot PV effectif"
                )
                ax2.set_ylabel("Prix (EUR/MWh)")
            
                if not legend_handles:
                    lines_1, labels_1 = ax1.get_legend_handles_labels()
                    lines_2, labels_2 = ax2.get_legend_handles_labels()
                    legend_handles = lines_1 + lines_2
                    legend_labels = labels_1 + labels_2
            
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=4,
                frameon=False,
            )
            
            fig.suptitle("Dispatch énergétique - mois complet de juin", fontsize=14)
            fig.tight_layout(rect=[0, 0.08, 1, 0.96])
            
            st.pyplot(fig)
            plt.close(fig)

        c5, c6 = st.columns(2)

        with c5:
            fig5, ax5 = plt.subplots(figsize=(9, 4.8))

            x = np.arange(len(monthly_df))
            width = 0.34

            pv_vals_mwh = monthly_df["pv_revenue_eur_per_mwh"].to_numpy(dtype=float)
            bess_vals_mwh = monthly_df["bess_revenue_eur_per_mwh"].to_numpy(dtype=float)
            pv_only_vals_mwh = (
                monthly_df["pv_only_revenue"].to_numpy(dtype=float)
                / monthly_df["pv_only_direct_mwh"].clip(lower=1e-12).to_numpy(dtype=float)
            )

            ax5.bar(
                x - width / 2,
                pv_vals_mwh,
                width=width,
                color="orange",
                label="PV hybride"
            )

            ax5.bar(
                x + width / 2,
                bess_vals_mwh,
                width=width,
                color="green",
                label="BESS"
            )

            ax5.plot(
                x,
                pv_only_vals_mwh,
                marker="o",
                linewidth=2.0,
                label="PV-only Project"
            )

            ax5.set_title("Specific Monthly Revenues per MWh")
            ax5.set_ylabel("€/MWh")
            ax5.set_xlabel("Month")
            ax5.set_xticks(x)
            ax5.set_xticklabels(monthly_df["month"], rotation=45)
            ax5.legend()

            st.pyplot(fig5)
            plt.close(fig5)

        with c6:
            if afrr_qh_df is not None:
                qh_debug_start = pd.Timestamp(f"{DEFAULT_YEAR}-06-01 00:00:00")
                qh_debug_end = qh_debug_start + pd.Timedelta(days=3)

                qh_plot = afrr_qh_df[
                    (afrr_qh_df["datetime"] >= qh_debug_start) &
                    (afrr_qh_df["datetime"] < qh_debug_end)
                ].copy()

                fig6, ax6 = plt.subplots(figsize=(12, 4.8))
                ax6.bar(qh_plot["datetime"], qh_plot["afrr_discharge_mwh"], width=0.008, label="Décharge aFRR", alpha=0.7)
                ax6.bar(qh_plot["datetime"], qh_plot["wholesale_discharge_mwh"], width=0.008, label="Décharge wholesale", alpha=0.7)
                ax6.bar(qh_plot["datetime"], -qh_plot["afrr_charge_mwh"], width=0.008, label="Charge aFRR", alpha=0.7)
                ax6.set_ylabel("MWh / 15 min")
                ax6.set_title("Arbitrage quart-horaire - 3 premiers jours de juin")
                ax6.xaxis.set_major_locator(mdates.HourLocator(interval=6))
                ax6.xaxis.set_major_formatter(mdates.DateFormatter("%d %Hh"))
                ax6.tick_params(axis="x", rotation=45)

                ax6b = ax6.twinx()
                ax6b.plot(qh_plot["datetime"], qh_plot["afrr_charge_price_effective_eur_per_mwh"], linestyle="--", alpha=0.7, label="Prix charge aFRR effectif")
                ax6b.plot(qh_plot["datetime"], qh_plot["afrr_discharge_price_effective_eur_per_mwh"], linestyle="-.", alpha=0.7, label="Prix décharge aFRR effectif")
                ax6b.set_ylabel("EUR/MWh")

                lines_a, labels_a = ax6.get_legend_handles_labels()
                lines_b, labels_b = ax6b.get_legend_handles_labels()
                ax6.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right")

                st.pyplot(fig6)
                plt.close(fig6)
            else:
                st.info("Activez l'aFRR et uploadez les deux fichiers quart-horaires pour afficher le graphique aFRR.")

        c7, c8 = st.columns(2)

        with c7:
            st.subheader("Comparaison Revenu PV-only vs Hybrid")

            fig_cmp, ax_cmp = plt.subplots(figsize=(9, 4.8))

            x = np.arange(len(monthly_df))

            pv_only_monthly_keur = monthly_df["pv_only_revenue"].to_numpy(dtype=float) / 1000.0
            hybrid_monthly_keur = monthly_df["net_revenue"].to_numpy(dtype=float) / 1000.0
            
            ax_cmp.plot(
                x,
                pv_only_monthly_keur,
                marker="o",
                linewidth=2.0,
                label="PV-only"
            )
            
            ax_cmp.plot(
                x,
                hybrid_monthly_keur,
                marker="o",
                linewidth=2.0,
                label="Hybrid (PV + BESS)"
            )
            
            if enable_cfd and "pv_only_cfd_revenue" in monthly_df.columns:
                pv_only_cfd_monthly_keur = (
                    monthly_df["pv_only_cfd_revenue"].to_numpy(dtype=float) / 1000.0
                )
            
                ax_cmp.plot(
                    x,
                    pv_only_cfd_monthly_keur,
                    marker="o",
                    linewidth=2.0,
                    label="PV-only-CfD"
                )

            ax_cmp.set_title("Comparaison Revenu PV-only vs Hybrid")
            ax_cmp.set_ylabel("kEUR")
            ax_cmp.set_xlabel("Mois")
            ax_cmp.set_xticks(x)
            ax_cmp.set_xticklabels(monthly_df["month"], rotation=45)
            ax_cmp.legend()

            st.pyplot(fig_cmp)
            plt.close(fig_cmp)

        with c8:
            fig8, ax8 = plt.subplots(figsize=(9, 4.8))

            x = np.arange(len(monthly_df))
            width = 0.26

            if "pv_curtailment_candidate_mwh" in monthly_df.columns:
                ax8.bar(
                    x - width,
                    monthly_df["pv_curtailment_candidate_mwh"].to_numpy(dtype=float),
                    width=width,
                    label="PV curtailed"
                )

            if "pv_curtailed_to_battery_mwh_actual" in monthly_df.columns:
                ax8.bar(
                    x,
                    monthly_df["pv_curtailed_to_battery_mwh_actual"].to_numpy(dtype=float),
                    width=width,
                    label="PV curtailed → battery"
                )

            if "pv_curtailed_residual_lost_mwh" in monthly_df.columns:
                ax8.bar(
                    x + width,
                    monthly_df["pv_curtailed_residual_lost_mwh"].to_numpy(dtype=float),
                    width=width,
                    label="PV curtailed lost"
                )

            ax8.set_title("Curtailment mensuel PV")
            ax8.set_ylabel("MWh")
            ax8.set_xlabel("Mois")
            ax8.set_xticks(x)
            ax8.set_xticklabels(monthly_df["month"], rotation=45)
            ax8.legend()

            st.pyplot(fig8)
            plt.close(fig8)

        st.subheader("Table mensuelle")
        st.dataframe(monthly_df, use_container_width=True, hide_index=True)

        st.subheader("Exports")
        st.download_button(
            "Télécharger cette simulation complète (Excel)",
            data=excel_bytes,
            file_name="simulation_complete_hybride_pv_bess.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            on_click="ignore",
        )
        st.download_button(
            "Télécharger l'horaire en CSV",
            data=hourly_df.to_csv(index=False).encode("utf-8"),
            file_name="dispatch_horaire_hybride.csv",
            mime="text/csv",
            on_click="ignore",
        )

        if afrr_qh_df is not None:
            st.download_button(
                "Télécharger l'aFRR quart-horaire en CSV",
                data=afrr_qh_df.to_csv(index=False).encode("utf-8"),
                file_name="dispatch_afrr_quart_horaire.csv",
                mime="text/csv",
                on_click="ignore",
            )

    except Exception as e:
        st.error(f"Erreur: {e}")


if __name__ == "__main__":
    app()
