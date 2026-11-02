# Solar Grid Efficiency AI (Demo)

A solar plant's nameplate is a promise; the grid meter tells the truth. Between the two sits a chain of losses — dirty panels, a quietly failing string, inverter inefficiency, downtime, curtailment — and in real projects they routinely add up to 10–20% of everything the panels generated. Most of that gap is invisible unless you measure stage by stage. This project simulates a small plant, makes the whole gap explicit, and uses simple ML to find the parts of it you can actually fix.

> This is a personal portfolio project exploring AI for power-system efficiency. It is built on 100% synthetic data with original code, and does not represent work done for any employer.

## The generation-to-grid gap

Energy leaves the system at every stage between the panel surface and the export meter:

| Stage | What happens |
|---|---|
| **Soiling** | Dust and dirt shade the cells. Worse, it rarely spreads evenly — one neglected string can lose far more than the plant average. |
| **Mismatch & degradation** | Panels age at different rates, and a damaged or failing string drags its output down long before it fails outright. |
| **Downtime** | Inverter or plant outages during daylight are pure loss: the sun doesn't reschedule. |
| **DC wiring** | Resistance losses between the arrays and the inverter. Small, constant, unavoidable — but worth measuring. |
| **Inverter conversion** | DC→AC conversion runs at ~97–98% efficiency, so roughly 2–3% of everything left is lost here. |
| **AC wiring & transformer** | Another fraction of a percent to a percent on the way to the meter. |
| **Curtailment** | When the plant out-produces the grid export limit at midday, the excess is simply clipped — generated, then thrown away. |

None of these stages is exotic. The gap exists because they're usually tracked in different systems (or not tracked at all), so nobody sees the total.

## What the demo does

`solar_efficiency.py` simulates a 300 kWp plant (6 strings × 50 kWp) hour by hour over 14 synthetic days: a clear-sky irradiance curve with weather noise, panel temperature effects, and three planted problems — one string that gets dirtier every day, one string that degrades partway through, and a few inverter downtime events. Every hour's energy is pushed through the full loss chain above, so the generated-vs-exported gap is measured, not assumed.

On top of the simulation sit three ML/analysis pieces:

1. **Generation forecasting** — a RandomForest trained on weather features (irradiance, ambient temperature, cloud cover, hour) predicts hourly generation, so operators can compare what's coming against what the plant should produce. Trained on the first 70% of days, tested on the rest.
2. **Underperformance detection** — each string's actual output is compared with a physics-based expectation (what that string *should* produce given the irradiance and temperature). Strings whose average ratio falls below a threshold get flagged. Plant-wide downtime hours are excluded, otherwise one inverter outage would "flag" the entire plant.
3. **Loss-reduction recommendations** — the loss stages are ranked, and for the actionable ones the demo estimates recoverable energy: cleaning the soiled string, repairing the degraded one, avoiding half the downtime, and shifting curtailed midday energy into storage (at ~70% after round-trip losses).

## Example output

Actual output from a run (fixed seed, so you'll get the same numbers):

```
Generated (gross DC):     23,515.3 kWh
Exported to grid (AC):    19,698.7 kWh
THE GAP:                   3,816.6 kWh  (16.2% of everything generated never reached the grid)

Where it went — loss by stage:
  Downtime                             958.4 kWh   4.1%
  Degradation (faulted string)         652.3 kWh   2.8%
  Soiling                              586.2 kWh   2.5%
  Inverter conversion                  513.7 kWh   2.2%
  Mismatch (base)                      458.6 kWh   2.0%
  DC wiring                            312.9 kWh   1.3%
  AC wiring & transformer              200.3 kWh   0.9%
  Curtailment (export cap)             134.3 kWh   0.6%

Generation forecast:  Test MAE 0.97 kWh/hour, R² 0.999
Strings flagged:      #2 (soiling, avg ratio 0.882), #4 (degraded, avg ratio 0.800)

Top recommendations:
  1. Repair the degraded string      -> ~587 kWh recoverable
  2. Clean the soiled string         -> ~498 kWh recoverable
  3. Root-cause the downtime events  -> ~479 kWh recoverable
```

The detector finds exactly the two planted problems and leaves the four healthy strings (ratio 0.970) alone.

## Run it

```bash
pip install -r requirements.txt
python solar_efficiency.py
```

Prints the efficiency report and saves a copy to `efficiency_report.txt`.

## Tech stack

Python · NumPy · pandas · scikit-learn (RandomForest, MAE/R² evaluation)

## Ideas for extension

- Feed the forecaster real weather-forecast data and schedule exports/storage against predicted peaks instead of the export cap doing it by force.
- Replace the fixed anomaly threshold with an IsolationForest over per-string daily features, and trend the soiling ratio to trigger cleaning only when the recovered energy pays for the truck roll.
- Add a battery model with charge/discharge scheduling optimized against the curtailment windows and time-of-use prices.
- Connect the same loss-chain accounting to real inverter/SCADA telemetry — the simulation's stage table is deliberately shaped like the report you'd want from live data.
