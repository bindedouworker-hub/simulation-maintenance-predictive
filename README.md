<div align="center">

<img src="docs/banner.png" alt="Banner — Hybrid Predictive Maintenance Simulation" width="100%"/>

**A simulated hybrid predictive-maintenance pipeline** — sensor data
generation, anomaly detection and RUL-prediction modeling.

[![CI](https://github.com/bindedouworker-hub/simulation-maintenance-predictive/actions/workflows/python-package.yml/badge.svg)](https://github.com/bindedouworker-hub/simulation-maintenance-predictive/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-31c48d)](LICENSE)
![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10%20%E2%80%93%203.12-3776ab)
![scikit-learn](https://img.shields.io/badge/Isolation%20Forest-scikit--learn-f89939)
![LSTM](https://img.shields.io/badge/LSTM-TensorFlow-ff8f00)

[English](README.md) · [Français](README.fr.md)

*Personal research project — open to everyone, contributions welcome!*

</div>

---

## 🎯 What it does

This project simulates the **full data pipeline of a predictive-maintenance
system** for IT / electronic equipment (CPU temperature, VCC voltage, fan
speed):

1. **Data generation** — synthetic sensor streams: healthy signal
   (daily cycle + Gaussian noise) with **injected faults** (drift, spikes)
2. **Anomaly detection (unsupervised)** — `IsolationForest` flags
   abnormal windows without any labeled data, upgraded with a
   **seasonal-residual stage** for persistent drifts (see Reliability)
3. **Preprocessing for deep learning** — Min-Max normalization +
   sliding-window sequences for time-series modeling
4. **LSTM model (RUL)** — stacked `LSTM → LSTM → Dense` network
   estimating the **Remaining Useful Life** of the component
5. **Evaluation** — confusion matrix, precision/recall/F1, with an eye
   on **false negatives** (missed failures = the costly mistakes)

## 🔬 Reliability & real-data validation

Signal + point-anomalies detected by the original Isolation Forest
(seeded run, see `results/simulation_plot.png` & recorded evaluation):

| | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.94 | 0.99 | 0.96 |
| **Anomaly** | **0.84** | **0.42** | **0.56** |

> 93% strict accuracy — but recall 0.42 on the anomaly: exactly the
> kind of honest finding this project surfaces and then fixes.

Because a detector whose results depend on the run is worthless, the
pipeline ships with a **Monte-Carlo reliability audit**
(`scripts/reliability_audit.py`, 200 independent runs):

| Detector | Point recall (16σ fault) | Fault episodes detected | Alarm latency (med) |
|---|---|---|---|
| Isolation Forest alone *(original)* | 0.44 ± 0.02 | 100 % | 0 pt |
| **Hybrid: IF + seasonal residual** *(built during this audit)* | **1.00 ± 0.00** | 100 % | 0 pt |

Two structural findings from this audit, both documented in
`results/reliability_report.txt`:

1. **Isolation Forest alone plateaus (~0.44) on *persistent* drifts** —
   a shifted plateau is not made of isolated points, and
   `contamination=0.05` can flag at most 25 points out of 500 while the
   fault covers 50.
2. The fix is a **hybrid detector**: IF keeps catching point anomalies,
   and a *seasonal residual* stage (`x[i] − x[i−P]`, period P estimated
   by bounded autocorrelation) catches the drift — recall 1.0 from
   ~4× the noise σ, near-zero latency. The latched effect on the next
   cycle is why strict point-precision reads 0.55; in industrial terms
   those flags are correct.

Reproducibility is guaranteed: all random draws are seeded
(`seed=42` by default everywhere) — two runs produce byte-identical
results.

### Benchmark on NASA C-MAPSS FD001 (real domain data)

`scripts/real_data_benchmark.py` runs the same "fit on healthy, score
the rest" idea on the public NASA Turbofan dataset (100 engines,
20 631 sensor cycles): fit Isolation Forest on healthy cycles of engines
1-80, score engines 81-100 (never seen), detect the 30-cycle pre-failure
window:

![Real-data benchmark](results/real_data_benchmark.png)

| Metric (hold-out engines 81-100) | Value |
|---|---|
| AUC-ROC (pre-failure vs healthy) | **0.984** |
| AUC-PR | **0.923** |
| Engines with a detected pre-failure phase | **20/20** |
| Lead time of the first alarm | **~29 cycles before failure** |

Run it yourself (dataset auto-downloaded if missing):

```bash
python scripts/real_data_benchmark.py     # real NASA data
python scripts/reliability_audit.py 200   # Monte-Carlo audit
```

## 🚀 Getting started

```bash
git clone https://github.com/bindedouworker-hub/simulation-maintenance-predictive.git
cd simulation-maintenance-predictive

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python scripts/generate_data.py          # 1. synthetic sensor stream
python scripts/maintenance_simulator.py  # 2. IsolationForest detection + plot
python scripts/data_preprocessing.py     # 3. normalization + sliding windows
python scripts/evaluation.py             # 4. confusion matrix + classification report
```

Outputs land in `results/` (CSV stream, text reports, `simulation_plot.png`).

> `scripts/lstm_model.py` defines the RUL network architecture
> (TensorFlow). The rest of the pipeline runs **without TensorFlow**, so
> you can explore detection/evaluation without a heavy install.

## 🗂️ Project structure

```
.
├── data/                       # generated sensor data (demo stream)
│   └── CMAPSSData/train_FD001.txt   # NASA Turbofan FD001 (real data)
├── scripts/
│   ├── generate_data.py        # synthetic sensors (temp CPU, VCC, fan RPM)
│   ├── data_preprocessing.py   # Min-Max + sliding windows for the LSTM
│   ├── lstm_model.py           # stacked LSTM → RUL regression head
│   ├── maintenance_simulator.py# IsolationForest + fault injection + plot
│   ├── evaluation.py           # confusion matrix, precision/recall/F1
│   ├── reliability_audit.py    # Monte-Carlo audit (seeds, mag sweep)
│   ├── real_data_benchmark.py  # NASA C-MAPSS hold-out benchmark
│   └── read_results.py         # re-read saved experiment outputs
├── results/                    # CSV / reports / plots from recorded runs
├── notebooks/read_results.ipynb# quick exploration notebook
└── .github/workflows/          # CI: lint + pipeline + audit + real benchmark
```

## 🗺️ Roadmap

- [x] ~~Raise anomaly recall~~ **done** — hybrid detector reaches 1.0
      point recall on persistent drifts (see Reliability)
- [x] ~~Validate on real data~~ **done** — NASA C-MAPSS FD001:
      AUC-ROC 0.98 on hold-out engines
- [ ] Real RUL training loop (`lstm_model.py` currently defines the architecture)
- [ ] ROC-AUC / PR-AUC curves for threshold-free detection on synthetic data
- [ ] Structured logging of runs (mlflow or CSV-only experiment tracker)
- [ ] Multi-sensor fusion (temperature + voltage + fan jointly)

## 🤝 Contributing

Everyone is welcome — this is an open study project! Please read
[CONTRIBUTING.md](CONTRIBUTING.md) (short). Open an issue or a PR.

## 📄 License

[MIT](LICENSE) — free to use, modify and share.

---

<div align="center">

**A project by Mr Jackson** 🇨🇮

</div>
