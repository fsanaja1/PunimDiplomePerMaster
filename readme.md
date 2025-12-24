
# ELF Exposure Modeling (NARDA vehicle measurements)

This project analyzes **ELF (Extremely Low Frequency) magnetic field exposure** measurements taken in/around vehicles (electric / hybrid / mild-hybrid) and produces **per-scenario (per Excel sheet) results**:

- a **feature-importance plot** (Random Forest) as a PNG
- a matching **CSV table** of top feature importances
- a consolidated **text summary**
- an **ICNIRP risk table** that ranks scenarios by measured exposure

All logic lives in: `src/rf_elf_model.py`.

## What it does

### Inputs
- Excel files located in `data/`:
	- `Narda_electricar_em.xlsx`
	- `Narda_hybrid_em.xlsx`
	- `Narda_mildhybrid_em.xlsx`

Each Excel file may contain **multiple sheets** (pages). The script reads **all sheets** from each file.

### Target
- Target variable: `B (µT)`

### Features used
The model uses these columns (if present in the sheet):

- `Location (front/rear seat/inside front shield)`
- `Area description (Traffic and road conditions)`
- `Electric Field (V/m)`
- `Speed` (or `Speed km/h` → normalized to `Speed`)
- `car_type` (added by the loader: electric/hybrid/mildhybrid)
- `sheet_name` (added by the loader)

**Battery is treated as constant (100%)** for all cases and is **not used as a model feature**.

### Modeling approach
- Baseline: `DummyRegressor(strategy="mean")`
- Model: `RandomForestRegressor` tuned with `GridSearchCV`
- Preprocessing: `ColumnTransformer` with:
	- categorical: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`
	- numeric: `SimpleImputer(median)`
- Evaluation: KFold cross-validation (3-fold by default).

Because some sheets can be small, you may see **negative or unstable R²**. With limited data per scenario, RMSE is usually the more stable metric.

## Outputs (generated in `results/`)

After running the script, you should find:

### 1) Per-sheet feature importance
For each `(car_type × sheet_name)` subset that has enough rows, the script generates:

- `feature_importance_<car_type>__<sheet_slug>.png`
- `feature_importance_<car_type>__<sheet_slug>.csv`

The PNG includes a standard annotation line with:

- Battery: 100% (constant)
- What-if prediction $\hat{B}$ for a fixed “scenario row”
- Measured max(B) in that sheet
- ICNIRP limit and whether the measured max exceeds it

### 2) Summary report
- `results/model_results.txt`

Contains overall CV metrics, best RF parameters, the list of generated PNG/CSV outputs, and per-sheet what-if predictions.

### 3) ICNIRP risk ranking table
- `results/icnirp_risk_by_sheet.csv`

Contains (per sheet):

- `measured_mean`, `measured_median`, `measured_p95`, `measured_max`
- `exceeds_icnirp` and `exceed_margin`
- model `best_params` and the sheet’s what-if prediction

## ICNIRP notes (important for thesis writing)

This project is for **ELF exposure** (typically **50 Hz** in Europe).

The script is configured with:

- `ICNIRP_FREQUENCY_HZ = 50`
- `ICNIRP_CATEGORY = "general_public"` (or `"occupational"`)

And by default uses these commonly cited reference levels at 50 Hz:

- General public: **200 µT**
- Occupational: **1000 µT**

These are implemented in `src/rf_elf_model.py` as `ICNIRP_LIMIT_B_UT_BY_CATEGORY`.

> Make sure the exact values you use match the **exact ICNIRP guideline/table you cite** in your thesis.

## Setup (Windows / PowerShell)

### 1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Run

From the repo root:

```powershell
python .\src\rf_elf_model.py
```

You should see console logs like:

- “Loading dataset…”
- “Dataset loaded successfully: … rows”
- baseline metrics and tuned RF metrics
- “Saved plot for …” lines

## Data expectations & troubleshooting

### Missing Excel files
If a file is missing you’ll see:

`Warning: missing file: ...`

If all are missing, the script raises `FileNotFoundError`.

### Speed values like “20-40”
The loader converts ranges to the midpoint (e.g., “20-40” → 30).

### Sheets with too few rows
Per-sheet modeling is skipped if a subset has fewer than 5 rows:

`Skipping car_type=..., sheet=...: not enough rows (...)`

### Headless plotting / Tkinter errors
The script forces Matplotlib `Agg` backend to safely generate many PNGs without GUI dependencies.

## Project structure

```
.
├── data/                      # Input Excel files (not generated)
├── results/                   # Generated outputs (PNGs/CSVs/tables)
├── src/
│   └── rf_elf_model.py        # Main script (end-to-end pipeline)
├── requirements.txt
└── readme.md
```

## How to interpret results (quick guidance)

- Use `icnirp_risk_by_sheet.csv` to identify scenarios with the largest measured exposure (e.g., highest `measured_max` or `measured_p95`).
- Use the per-sheet PNGs to explain *which conditions/features* correlate with higher exposure in that scenario.
- Treat model metrics carefully when per-sheet sample sizes are small; focus more on descriptive statistics (max/p95) for “risk” claims.

