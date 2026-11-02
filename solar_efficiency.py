"""
Solar Grid Efficiency AI — demo.

Simulates a small solar plant on synthetic weather, tracks where the energy
goes between the panels and the grid meter, and uses simple ML to do three
things operators actually care about:

  1. Forecast generation from weather features (RandomForest).
  2. Flag strings that underperform their physics-based expectation
     (soiling, degradation, a failing string) before a human notices.
  3. Rank the loss stages and estimate how much energy the top fixes
     would recover.

Everything is synthetic, the seed is fixed, and the numbers are small on
purpose: the point is the measurement chain, not the plant size.

Run:  python solar_efficiency.py
Writes a copy of the report to efficiency_report.txt next to this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

SEED = 42

# --- Plant configuration -------------------------------------------------
NUM_STRINGS = 6
STRING_KWP = 50.0                 # kWp per string -> 300 kWp plant
DAYS = 14
EXPORT_CAP_KW = 200.0             # grid export limit (curtailment above this)

# Loss-chain parameters (fractions)
BASE_SOILING = 0.01               # light dust on every string
SOILED_STRING = 2                 # index of the string nobody cleans
SOILING_GROWTH = 0.18             # ...which gets dirtier every day
BASE_MISMATCH = 0.02              # panel-to-panel mismatch
DEGRADED_STRING = 4               # index of the string that starts failing
DEGRADATION_START_DAY = 7
DEGRADATION_LOSS = 0.35           # loses 35% of its output once it degrades
DC_WIRING_LOSS = 0.015
INVERTER_LOSS = 0.025             # 97.5% inverter efficiency
AC_WIRING_LOSS = 0.01

ANOMALY_RATIO_THRESHOLD = 0.93    # string flagged below this avg efficiency


def simulate(rng):
    """Hour-by-hour simulation. Returns (hourly_df, stage_totals_kwh)."""
    rows = []
    stages = {
        "Soiling": 0.0,
        "Mismatch (base)": 0.0,
        "Degradation (faulted string)": 0.0,
        "Downtime": 0.0,
        "DC wiring": 0.0,
        "Inverter conversion": 0.0,
        "AC wiring & transformer": 0.0,
        "Curtailment (export cap)": 0.0,
    }

    # A few inverter downtime events during daylight hours.
    downtime_hours = set()
    for _ in range(3):
        d = int(rng.integers(1, DAYS - 1))
        h = int(rng.integers(9, 15))
        downtime_hours.add((d, h))
        downtime_hours.add((d, h + 1))

    for day in range(DAYS):
        day_weather = rng.uniform(0.75, 1.0)       # how good the day is
        day_cloud = rng.uniform(0.0, 0.3)
        for hour in range(24):
            if 6 <= hour <= 19:
                elevation = max(0.0, np.sin(np.pi * (hour - 6) / 13.0))
                cloud = np.clip(day_cloud + rng.normal(0, 0.06), 0, 0.7)
                irradiance = 1000.0 * elevation ** 1.2 * day_weather * (1 - cloud)
            else:
                cloud = day_cloud
                irradiance = 0.0
            ambient = 17 + 8 * np.sin(np.pi * (hour - 7) / 14.0) + rng.normal(0, 0.8)
            cell_temp = ambient + irradiance * 0.03
            temp_factor = max(0.8, 1 - 0.004 * (cell_temp - 25))

            plant_after_strings = 0.0
            for s in range(NUM_STRINGS):
                gross = STRING_KWP * (irradiance / 1000.0) * temp_factor
                expected = gross  # physics expectation, used for anomaly checks

                soiling = BASE_SOILING
                if s == SOILED_STRING:
                    soiling += SOILING_GROWTH * day / (DAYS - 1)
                e = gross * (1 - soiling)
                stages["Soiling"] += gross - e

                mismatch_loss = e * BASE_MISMATCH
                e -= mismatch_loss
                stages["Mismatch (base)"] += mismatch_loss

                if s == DEGRADED_STRING and day >= DEGRADATION_START_DAY:
                    deg_loss = e * DEGRADATION_LOSS
                    e -= deg_loss
                    stages["Degradation (faulted string)"] += deg_loss

                if (day, hour) in downtime_hours:
                    stages["Downtime"] += e
                    e = 0.0

                plant_after_strings += e
                rows.append({
                    "day": day, "hour": hour, "string": s,
                    "irradiance": irradiance, "ambient_temp": ambient,
                    "cloud_cover": cloud,
                    "expected_kwh": expected, "actual_kwh": e,
                })

            e = plant_after_strings
            dc_loss = e * DC_WIRING_LOSS
            e -= dc_loss
            stages["DC wiring"] += dc_loss

            inv_loss = e * INVERTER_LOSS
            e -= inv_loss
            stages["Inverter conversion"] += inv_loss

            ac_loss = e * AC_WIRING_LOSS
            e -= ac_loss
            stages["AC wiring & transformer"] += ac_loss

            exported = min(e, EXPORT_CAP_KW)
            stages["Curtailment (export cap)"] += e - exported

            # Plant-level features/target for the forecaster.
            rows.append({
                "day": day, "hour": hour, "string": -1,
                "irradiance": irradiance, "ambient_temp": ambient,
                "cloud_cover": cloud,
                "expected_kwh": np.nan, "actual_kwh": np.nan,
                "gross_kwh": STRING_KWP * NUM_STRINGS * (irradiance / 1000.0) * temp_factor,
                "exported_kwh": exported,
            })

    df = pd.DataFrame(rows)
    gross_total = df.loc[df["string"] == -1, "gross_kwh"].sum()
    exported_total = df.loc[df["string"] == -1, "exported_kwh"].sum()
    return df, stages, gross_total, exported_total, downtime_hours


def forecast(df):
    """Train on the first 70% of days, predict generation for the rest."""
    plant = df[df["string"] == -1].copy()
    features = ["hour", "irradiance", "ambient_temp", "cloud_cover"]
    split_day = int(DAYS * 0.7)
    train = plant[plant["day"] < split_day]
    test = plant[plant["day"] >= split_day]
    model = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
    model.fit(train[features], train["gross_kwh"])
    pred = model.predict(test[features])
    return mean_absolute_error(test["gross_kwh"], pred), r2_score(test["gross_kwh"], pred)


def detect_anomalies(df, downtime_hours):
    """Compare each string's actual output with its physics expectation.

    Plant-wide downtime hours are excluded: every string is off then, so
    counting them would flag the whole plant instead of the faulty string.
    """
    strings = df[df["string"] >= 0].copy()
    strings = strings[~strings.apply(
        lambda r: (r["day"], r["hour"]) in downtime_hours, axis=1)]
    daily = strings.groupby(["string", "day"]).sum(numeric_only=True).reset_index()
    daily["ratio"] = daily["actual_kwh"] / daily["expected_kwh"]
    summary = daily.groupby("string")["ratio"].agg(["mean", "min"]).reset_index()
    flagged = summary[summary["mean"] < ANOMALY_RATIO_THRESHOLD]
    return summary, flagged


def recommendations(stages, gross_total):
    """Rank loss stages; estimate recoverable energy for the actionable ones."""
    recs = []
    soil = stages["Soiling"]
    if soil > 0:
        recs.append(("Clean the flagged soiled string(s) and set a cleaning "
                     "schedule based on the soiling trend", soil * 0.85))
    deg = stages["Degradation (faulted string)"]
    if deg > 0:
        recs.append(("Inspect and repair/replace the flagged degraded string",
                     deg * 0.90))
    curt = stages["Curtailment (export cap)"]
    if curt > 0:
        recs.append(("Shift curtailed midday energy with on-site storage "
                     "(~70% recoverable after round-trip losses)", curt * 0.70))
    down = stages["Downtime"]
    if down > 0:
        recs.append(("Root-cause the inverter downtime events; preventive "
                     "maintenance typically avoids about half", down * 0.50))
    inv = stages["Inverter conversion"]
    if inv > 0:
        recs.append(("Inverter conversion is the largest fixed loss — a "
                     "higher-efficiency inverter at replacement time recovers "
                     "part of it", inv * 0.30))
    recs.sort(key=lambda r: r[1], reverse=True)
    return recs


def main():
    rng = np.random.default_rng(SEED)
    df, stages, gross_total, exported_total, downtime_hours = simulate(rng)
    mae, r2 = forecast(df)
    summary, flagged = detect_anomalies(df, downtime_hours)
    recs = recommendations(stages, gross_total)

    lost = gross_total - exported_total
    lines = []
    lines.append("=" * 68)
    lines.append("SOLAR GRID EFFICIENCY REPORT  (synthetic data, 14 days)")
    lines.append("=" * 68)
    lines.append(f"Plant: {NUM_STRINGS} strings x {STRING_KWP:.0f} kWp "
                 f"= {NUM_STRINGS * STRING_KWP:.0f} kWp, export cap {EXPORT_CAP_KW:.0f} kW")
    lines.append("")
    lines.append(f"Generated (gross DC):   {gross_total:>10,.1f} kWh")
    lines.append(f"Exported to grid (AC):  {exported_total:>10,.1f} kWh")
    lines.append(f"THE GAP:                {lost:>10,.1f} kWh  "
                 f"({lost / gross_total * 100:.1f}% of everything generated never reached the grid)")
    lines.append("")
    lines.append("Where it went — loss by stage:")
    lines.append(f"  {'Stage':<32}{'kWh':>10}{'% of gross':>12}")
    for name, kwh in sorted(stages.items(), key=lambda kv: kv[1], reverse=True):
        if kwh > 0:
            lines.append(f"  {name:<32}{kwh:>10,.1f}{kwh / gross_total * 100:>11.1f}%")
    lines.append("")
    lines.append("Generation forecast (RandomForest, weather features -> next-period kWh):")
    lines.append(f"  Test MAE: {mae:.2f} kWh per hour   R2: {r2:.3f}")
    lines.append("")
    lines.append("String health (actual vs physics-expected output):")
    for _, row in summary.iterrows():
        mark = "  <-- FLAGGED" if row["string"] in set(flagged["string"]) else ""
        lines.append(f"  String {int(row['string'])}: avg ratio {row['mean']:.3f}, "
                     f"worst day {row['min']:.3f}{mark}")
    if flagged.empty:
        lines.append("  No underperforming strings detected.")
    lines.append("")
    lines.append("Top recommendations (estimated recoverable energy over the same period):")
    for i, (text, kwh) in enumerate(recs[:3], 1):
        lines.append(f"  {i}. {text}")
        lines.append(f"     -> ~{kwh:,.1f} kWh ({kwh / gross_total * 100:.1f}% of gross generation)")
    lines.append("")
    lines.append("Note on storage/dispatch: curtailment happens at midday when the")
    lines.append("plant out-produces the export cap. Even a small battery charged in")
    lines.append("that window and discharged in the evening converts clipped energy")
    lines.append("into exported energy instead of losing it outright.")
    lines.append("=" * 68)

    report = "\n".join(lines)
    print(report)
    out = Path(__file__).with_name("efficiency_report.txt")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to {out.name}")


if __name__ == "__main__":
    main()
