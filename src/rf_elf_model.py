# =====================================================
# Random Forest Regressor
# Vlerësimi i ekspozimit ELF në automjet elektrik
# Dataset: Matje reale me NARDA
# =====================================================

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib

# Use a non-interactive backend to generate many PNGs safely (avoids Tkinter errors)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# -----------------------------------------------------
# 1. Leximi i dataset-it
# -----------------------------------------------------
print("Loading dataset...")

# Get the directory where this script is located
script_dir = Path(__file__).parent
data_dir = script_dir.parent / "data"

# Load multiple car datasets (electric / hybrid / mildhybrid)
data_files = [
    (data_dir / "Narda_electricar_em.xlsx", "electric"),
    (data_dir / "Narda_hybrid_em.xlsx", "hybrid"),
    (data_dir / "Narda_mildhybrid_em.xlsx", "mildhybrid"),
]

frames: list[pd.DataFrame] = []
for file_path, car_type in data_files:
    if not file_path.exists():
        print(f"Warning: missing file: {file_path}")
        continue

    # Read ALL sheets ("pages") from the Excel file
    # pandas returns a dict: {sheet_name: DataFrame}
    all_sheets = pd.read_excel(file_path, sheet_name=None)

    for sheet_name, df in all_sheets.items():
        df = df.copy()
        df["car_type"] = car_type
        df["sheet_name"] = sheet_name
        df["source_file"] = file_path.name

        # Normalize column names that differ between files
        if "Speed km/h" in df.columns and "Speed" not in df.columns:
            df = df.rename(columns={"Speed km/h": "Speed"})

        # Normalize 'Type of car' column name (optional)
        if "Type of car:" in df.columns and "Type of car" not in df.columns:
            df = df.rename(columns={"Type of car:": "Type of car"})

        frames.append(df)

if not frames:
    raise FileNotFoundError(
        f"No input Excel files found in {data_dir}. Expected: {[p.name for p, _ in data_files]}"
    )

data = pd.concat(frames, ignore_index=True)

print(f"Dataset loaded successfully: {len(data)} rows\n")

# -----------------------------------------------------
# 2. Përzgjedhja e kolonave relevante
# -----------------------------------------------------
features = [
    "Location (front/rear seat/inside front shield)",
    "Area description (Traffic and road conditions)",
    "Electric Field (V/m)",
    "Speed",
    "car_type",
    "sheet_name",
]

target = "B (µT)"

X = data[features].copy()
y = data[target].copy()

# Ensure categorical columns have consistent dtype across all input files
for _col in [
    "Location (front/rear seat/inside front shield)",
    "Area description (Traffic and road conditions)",
    "car_type",
    "sheet_name",
]:
    if _col in X.columns:
        X[_col] = X[_col].astype(str)

# Normalize Battery values so we don't end up with separate categories like
# Battery_0.5 and Battery_50 for the same concept.
def _normalize_battery(v: str) -> str:
    if v is None:
        return "unknown"
    s = str(v).strip().lower()
    if s in {"nan", "none", "", "-"}:
        return "unknown"
    # common textual forms
    if "half" in s and "fuel" in s:
        return "50%"
    if "half" == s:
        return "50%"

    # numeric forms
    num = pd.to_numeric(s.replace("%", ""), errors="coerce")
    if pd.notna(num):
        # if it looks like a fraction (0..1), convert to percent
        if 0 <= num <= 1:
            return f"{int(round(num * 100))}%"
        # otherwise treat it as already percent-like
        if 0 <= num <= 100:
            return f"{int(round(num))}%"
    return s

# Battery is treated as constant (100%) for all cases.
# We keep normalization helper in code history, but we do not use Battery as a feature.

# Coerce numeric columns to proper numeric dtype.
# Some files may contain ranges like "20-40"; we convert them to midpoint (30).
def _coerce_speed(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    if "-" in s:
        parts = [p.strip() for p in s.split("-") if p.strip()]
        if len(parts) == 2:
            a = pd.to_numeric(parts[0], errors="coerce")
            b = pd.to_numeric(parts[1], errors="coerce")
            if pd.notna(a) and pd.notna(b):
                return float(a + b) / 2.0
    return pd.to_numeric(s, errors="coerce")

if "Speed" in X.columns:
    X["Speed"] = X["Speed"].map(_coerce_speed)
if "Electric Field (V/m)" in X.columns:
    X["Electric Field (V/m)"] = pd.to_numeric(X["Electric Field (V/m)"], errors="coerce")

results_dir = script_dir.parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------
# ICNIRP (ELF) reference levels (B in µT)
# -----------------------------------------------------
# This project measures ELF exposure (typically ~50 Hz in Europe).
# The correct ICNIRP guideline for this is the 1 Hz – 100 kHz document (not RF 100 kHz – 300 GHz).
#
# NOTE: We keep the limits configurable here. If you want a different frequency/category,
# update these values to match the ICNIRP table you are using in the thesis.
ICNIRP_FREQUENCY_HZ = 50
ICNIRP_CATEGORY = "general_public"  # or: "occupational"

# Commonly cited ICNIRP 2010 reference levels at 50 Hz for magnetic flux density (B):
#   - General public: 200 µT
#   - Occupational:   1000 µT
# (Verify against the exact ICNIRP table you will cite.)
ICNIRP_LIMIT_B_UT_BY_CATEGORY = {
    "general_public": 200.0,
    "occupational": 1000.0,
}


def _icnirp_limit_b_ut() -> float:
    return float(ICNIRP_LIMIT_B_UT_BY_CATEGORY.get(ICNIRP_CATEGORY, 200.0))


ICNIRP_LIMIT_B_UT = _icnirp_limit_b_ut()

# -----------------------------------------------------
# 3. Përpunimi (Pipeline) + Vlerësim korrekt
# -----------------------------------------------------
# NOTE:
# Dataset-i ka vetëm 9 rreshta. Me kaq pak të dhëna, një train/test split
# mund të japë R² shumë negativ nëse 1-2 mostra "të vështira" bien në test.
# Përdorim cross-validation (CV) për vlerësim më stabil.

categorical_features = [
    "Location (front/rear seat/inside front shield)",
    "Area description (Traffic and road conditions)",
    "car_type",
    "sheet_name",
]
numeric_features = [
    "Electric Field (V/m)",
    "Speed",
]


def _drop_all_missing_numeric_features(df: pd.DataFrame, numeric_cols: list[str]) -> list[str]:
    """Return numeric cols that have at least one non-missing value in df."""
    kept: list[str] = []
    for c in numeric_cols:
        if c in df.columns and df[c].notna().any():
            kept.append(c)
    return kept

preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                ]
            ),
            numeric_features,
        ),
    ],
    remainder="drop",
)

# Baseline: gjithmonë parashikon mesataren
baseline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", DummyRegressor(strategy="mean")),
    ]
)

rf = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
            ),
        ),
    ]
)

# Dataset-i ka N=9. Për të shmangur fold-e me vetëm 1 mostër në test (ku R²
# nuk është i definueshëm), përdorim 3-fold CV.
cv = KFold(n_splits=3, shuffle=True, random_state=42)

print("Evaluating baseline (DummyRegressor) with 3-fold CV...")
y_pred_base = cross_val_predict(baseline, X, y, cv=cv)
rmse_base = np.sqrt(mean_squared_error(y, y_pred_base))
r2_base = r2_score(y, y_pred_base)
print("Baseline Performance (CV):")
print(f"RMSE: {rmse_base:.3f} µT")
print(f"R² Score: {r2_base:.3f}\n")

print("Training/tuning Random Forest with CV...")
param_grid = {
    "model__n_estimators": [200, 500],
    "model__max_depth": [None, 3, 5],
    "model__min_samples_leaf": [1, 2],
}

# R² është shumë i paqëndrueshëm me dataset kaq të vogël, prandaj për tuning
# përdorim RMSE (më stabil). Pastaj raportojmë edhe R² për interpretim.
search = GridSearchCV(
    rf,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
)
search.fit(X, y)
best_model = search.best_estimator_

print("Best params:")
print(search.best_params_, "\n")

# CV parashikime për modelin më të mirë
y_pred_cv = cross_val_predict(best_model, X, y, cv=cv)
rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cv))
r2_cv = r2_score(y, y_pred_cv)

print("Random Forest Performance (CV, best params):")
print(f"RMSE: {rmse_cv:.3f} µT")
print(f"R² Score: {r2_cv:.3f}\n")

# -----------------------------------------------------
# 4. Trajnim final (për përdorim) + rëndësi veçorish
# -----------------------------------------------------
best_model.fit(X, y)

# Nxjerrim feature importances për RF (pas OneHot)
rf_est = best_model.named_steps["model"]
ohe = best_model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
ohe_feature_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = list(ohe_feature_names) + numeric_features

importances = pd.Series(rf_est.feature_importances_, index=all_feature_names).sort_values(ascending=False)

print("Feature Importance (after OneHot):")
print(importances.head(20), "\n")

# NOTE: We no longer generate a single combined "top20" plot.

def _safe_slug(text: str) -> str:
    """Filesystem-friendly slug for sheet names."""
    s = str(text).strip().lower()
    # Keep letters/numbers; convert others to underscore
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")[:80] or "sheet"


PLOT_STYLE = {
    "top_n": 15,
    "figsize": (13, 7),
    "dpi": 250,
    "bar_color": "#2E86AB",
    "title_fontsize": 14,
    "base_fontsize": 11,
}


# -----------------------------------------------------
# 5. Modele & grafika veç e veç për çdo sheet (scenario)
#    (+ car_type), pra 1 PNG për 1 sheet.
# -----------------------------------------------------
print("\nTraining separate models per sheet_name (scenario) and car_type, saving one plot per sheet...")


def _train_and_plot_for_sheet(car: str, sheet: str) -> dict | None:
    mask = (X["car_type"] == car) & (X["sheet_name"] == sheet)
    Xs = X.loc[mask].copy()
    ys = y.loc[mask].copy()

    # Risk stats from *measured* values (not model prediction)
    y_clean = pd.to_numeric(ys, errors="coerce").dropna()
    if len(y_clean) == 0:
        print(f"Skipping car_type={car}, sheet={sheet}: no valid target values")
        return None
    y_mean = float(y_clean.mean())
    y_median = float(y_clean.median())
    y_p95 = float(y_clean.quantile(0.95))
    y_max = float(y_clean.max())
    exceeds_icnirp = bool(y_max > ICNIRP_LIMIT_B_UT)
    exceed_margin = float(y_max - ICNIRP_LIMIT_B_UT)

    if len(Xs) < 5:
        print(f"Skipping car_type={car}, sheet={sheet}: not enough rows ({len(Xs)})")
        return None

    n_splits = 3 if len(Xs) >= 6 else 2
    cv_s = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Avoid very noisy warnings: if a numeric column is all-missing inside this
    # sheet subset (common for Speed), drop it for this model.
    numeric_features_s = _drop_all_missing_numeric_features(Xs, numeric_features)
    preprocess_s = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features_s,
            ),
        ],
        remainder="drop",
    )
    rf_s = Pipeline(
        steps=[
            ("preprocess", preprocess_s),
            (
                "model",
                RandomForestRegressor(
                    random_state=42,
                ),
            ),
        ]
    )

    search_s = GridSearchCV(
        rf_s,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv_s,
        n_jobs=-1,
    )
    search_s.fit(Xs, ys)
    model_s = search_s.best_estimator_
    model_s.fit(Xs, ys)

    # What-if *for this sheet* (scenario) and this car_type
    what_if_row = {
        "Location (front/rear seat/inside front shield)": "front",
        "Area description (Traffic and road conditions)": "heavy traffic on the road",
        "Electric Field (V/m)": 1.2,
        "Speed": 80,
        "car_type": car,
        "sheet_name": sheet,
    }
    what_if_pred = float(model_s.predict(pd.DataFrame([what_if_row]))[0])

    rf_est_s = model_s.named_steps["model"]
    pre_s = model_s.named_steps["preprocess"]
    # IMPORTANT: In some sheet subsets, a numeric column can be entirely missing
    # (e.g., Speed is NaN for all rows). SimpleImputer will then drop that column,
    # changing the number of features. So we must read the *actual* output feature
    # names from the fitted preprocessor.
    feat_names_s = list(pre_s.get_feature_names_out())

    imp_s = pd.Series(rf_est_s.feature_importances_, index=feat_names_s)
    # car_type_ and sheet_name_ are constant within a sheet subset → drop from plot/table
    # ColumnTransformer prefixes features with transformer name by default:
    #   cat__<onehot name> , num__<original name>
    imp_s = imp_s[~imp_s.index.str.startswith("cat__car_type_")]
    imp_s = imp_s[~imp_s.index.str.startswith("cat__sheet_name_")]
    imp_s = imp_s.sort_values(ascending=False)

    top_n = PLOT_STYLE["top_n"]
    top = imp_s.head(top_n)

    sheet_slug = _safe_slug(sheet)
    csv_path = results_dir / f"feature_importance_{car}__{sheet_slug}.csv"
    top.to_frame(name="importance").to_csv(csv_path, index_label="feature")

    plt.rcParams.update(
        {
            "font.size": PLOT_STYLE["base_fontsize"],
            "axes.titlesize": PLOT_STYLE["title_fontsize"],
            "axes.labelsize": 12,
        }
    )

    fig = plt.figure(figsize=PLOT_STYLE["figsize"])
    ax = fig.gca()
    top.sort_values(ascending=True).plot(kind="barh", color=PLOT_STYLE["bar_color"], ax=ax)

    # Standard title format for thesis
    ax.set_title(f"Feature Importance – {car} – {sheet}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")

    # Standard grid and margins
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    # Put prediction as a consistent note box at the bottom
    fig.text(
        0.5,
        0.02,
        (
            f"Battery: 100%   |   What-if B̂: {what_if_pred:.3f} µT   |   "
            f"Measured max(B): {y_max:.3f} µT   |   ICNIRP({ICNIRP_FREQUENCY_HZ} Hz, {ICNIRP_CATEGORY}): {ICNIRP_LIMIT_B_UT:.0f} µT   |   "
            f"Exceeds: {'YES' if exceeds_icnirp else 'NO'}"
        ),
        ha="center",
        va="bottom",
        fontsize=PLOT_STYLE["base_fontsize"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F2F4F8", "edgecolor": "#C7CEDB"},
    )

    out_path = results_dir / f"feature_importance_{car}__{sheet_slug}.png"
    plt.savefig(out_path, dpi=PLOT_STYLE["dpi"])
    plt.close()

    print(f"Saved plot for {car} / {sheet}: {out_path}")
    print(f"Saved table for {car} / {sheet}: {csv_path}")

    return {
        "car_type": car,
        "sheet_name": sheet,
        "n_rows": int(len(Xs)),
        "measured_mean": y_mean,
        "measured_median": y_median,
        "measured_p95": y_p95,
        "measured_max": y_max,
        "icnirp_limit_b_ut": float(ICNIRP_LIMIT_B_UT),
        "exceeds_icnirp": exceeds_icnirp,
        "exceed_margin": exceed_margin,
        "cv_splits": int(n_splits),
        "best_params": dict(search_s.best_params_),
        "what_if_pred": float(what_if_pred),
        "png": str(out_path),
        "csv": str(csv_path),
    }


def _cleanup_legacy_per_car_outputs() -> None:
    """Remove old per-car combined files so results/ contains only per-sheet outputs."""
    for car in sorted(X["car_type"].unique()):
        for ext in (".png", ".csv"):
            p = results_dir / f"feature_importance_{car}{ext}"
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    # non-fatal (e.g., file opened)
                    pass


per_sheet_results: list[dict] = []
_cleanup_legacy_per_car_outputs()
for _car in sorted(X["car_type"].unique()):
    for _sheet in sorted(X.loc[X["car_type"] == _car, "sheet_name"].unique()):
        res = _train_and_plot_for_sheet(_car, _sheet)
        if res is not None:
            per_sheet_results.append(res)

# -----------------------------------------------------
# 5b. ICNIRP risk table export (measured values)
# -----------------------------------------------------
risk_csv_path = results_dir / "icnirp_risk_by_sheet.csv"
if per_sheet_results:
    risk_df = pd.DataFrame(per_sheet_results)
    # Rank by exceed first, then by measured max(B) (most conservative)
    risk_df = risk_df.sort_values(["exceeds_icnirp", "measured_max"], ascending=[False, False])
    risk_df.to_csv(risk_csv_path, index=False)
else:
    risk_df = pd.DataFrame()

# -----------------------------------------------------
# 6. Ruajtje e rezultateve (summary) në results/
# -----------------------------------------------------
summary_path = results_dir / "model_results.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("RF ELF Exposure Model – Results Summary\n")
    f.write("=====================================\n\n")
    f.write(f"Total rows used: {len(X)}\n")
    f.write(f"Target: {target}\n\n")

    f.write("Baseline (DummyRegressor, CV=3-fold)\n")
    f.write(f"  RMSE: {rmse_base:.3f} µT\n")
    f.write(f"  R²:   {r2_base:.3f}\n\n")

    f.write("Random Forest (best via GridSearchCV, CV=3-fold)\n")
    f.write(f"  Best params: {search.best_params_}\n")
    f.write(f"  RMSE: {rmse_cv:.3f} µT\n")
    f.write(f"  R²:   {r2_cv:.3f}\n\n")

    f.write("Per-sheet outputs (one PNG per sheet)\n")
    if not per_sheet_results:
        f.write("  (No per-sheet plots generated; likely not enough rows per sheet.)\n")
    else:
        for r in per_sheet_results:
            f.write(f"  - {Path(r['png']).name}\n")
            f.write(f"  - {Path(r['csv']).name}\n")

    f.write("\nWhat-if scenario used for each sheet\n")
    f.write("  Location: front\n")
    f.write("  Traffic:  heavy traffic on the road\n")
    f.write("  Battery:  100% (constant)\n")
    f.write("  E-field:  1.2 V/m\n")
    f.write("  Speed:    80 km/h\n")

    f.write("\nICNIRP comparison (ELF)\n")
    f.write(f"  Frequency: {ICNIRP_FREQUENCY_HZ} Hz\n")
    f.write(f"  Category:  {ICNIRP_CATEGORY}\n")
    f.write(f"  Limit B:   {ICNIRP_LIMIT_B_UT:.0f} µT\n")
    if risk_csv_path.exists():
        f.write(f"  Risk table: {risk_csv_path.name}\n")
    f.write("\nPer-sheet what-if predictions (B̂, µT)\n")
    for r in per_sheet_results:
        f.write(
            f"  - {r['car_type']} / {r['sheet_name']} (n={r['n_rows']}, CV={r['cv_splits']}): {r['what_if_pred']:.3f}\n"
        )

    if not risk_df.empty:
        f.write("\nPer-sheet measured risk (Top 10 by measured max(B))\n")
        top10 = risk_df.head(10)
        for _, row in top10.iterrows():
            f.write(
                f"  - {row['car_type']} / {row['sheet_name']}: "
                f"mean={row['measured_mean']:.3f} µT, p95={row['measured_p95']:.3f} µT, max={row['measured_max']:.3f} µT; "
                f"exceeds={bool(row['exceeds_icnirp'])}\n"
            )

print(f"Saved results summary: {summary_path}")

# -----------------------------------------------------
# 10. Test ekzekutimi (What-if scenario)
# -----------------------------------------------------
print("Running a test prediction (what-if scenario)...")

if not per_sheet_results:
    print("No per-sheet models were trained (not enough rows per sheet).")
else:
    # Print a compact sample (first few) to keep console readable
    for r in per_sheet_results[:10]:
        print(
            f"Predicted Magnetic Field B (µT) [{r['car_type']} / {r['sheet_name']}]: {r['what_if_pred']:.3f}"
        )

print("\nExecution finished successfully.")
