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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# TensorFlow/Keras për deep learning models
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
    # Set random seeds për reproducibility
    tf.random.set_seed(42)
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Skipping TensorFlow model evaluation.")

# -----------------------------------------------------
# 1. Leximi i dataset-it
# -----------------------------------------------------
print("Loading dataset...")

# Get the directory where this script is located
script_dir = Path(__file__).parent
data_dir = script_dir.parent / "data"

# Përdor vetëm file-in e kombinuar me të gjitha llojet e makinave
data_file = data_dir / "Narda_all_cars.xlsx"

if not data_file.exists():
    raise FileNotFoundError(f"Input file not found: {data_file}")

frames: list[pd.DataFrame] = []

# Read ALL sheets ("pages") from the combined Excel file
# pandas returns a dict: {sheet_name: DataFrame}
all_sheets = pd.read_excel(data_file, sheet_name=None)

# Mapping i kolonave nga file-i i ri në emrat e pritshëm nga kodi
COLUMN_MAPPING = {
    "speed_kmh": "Speed",
    "location": "Location (front/rear seat/inside front shield)",
    "traffic_condition": "Area description (Traffic and road conditions)",
    "driving_condition": "sheet_name",  # Përdor driving_condition si sheet_name/scenario
}

for sheet_name, df in all_sheets.items():
    df = df.copy()
    df["source_file"] = data_file.name

    # Apliko column mapping
    for old_col, new_col in COLUMN_MAPPING.items():
        if old_col in df.columns:
            # Nëse driving_condition ekziston, përdore atë si sheet_name
            if old_col == "driving_condition":
                df["sheet_name"] = df[old_col].astype(str).str.strip()
            else:
                df[new_col] = df[old_col]
    
    # Nëse nuk ka driving_condition, përdor sheet_name origjinal
    if "sheet_name" not in df.columns:
        df["sheet_name"] = sheet_name
    
    # Normalizo B (µT) column nëse ka encoding issues
    b_cols = [c for c in df.columns if c.startswith("B (") and "T)" in c]
    if b_cols and "B (µT)" not in df.columns:
        df = df.rename(columns={b_cols[0]: "B (µT)"})
    
    # Normalizo car_type nëse ekziston
    if "car_type" in df.columns:
        df["car_type"] = df["car_type"].astype(str).str.strip().str.lower()
        # Normalizoni vlerat e njohura
        df["car_type"] = df["car_type"].replace({
            "electric": "electric",
            "hybrid": "hybrid",
            "mild hybrid": "mildhybrid",
            "mildhybrid": "mildhybrid",
            "mild-hybrid": "mildhybrid",
            "mild_hybrid": "mildhybrid",
        })
    else:
        df["car_type"] = "unknown"

    frames.append(df)

if not frames:
    raise FileNotFoundError(f"No sheets found in {data_file}")

data = pd.concat(frames, ignore_index=True)

print(f"Dataset loaded successfully: {len(data)} rows")
print(f"Car types found: {data['car_type'].unique().tolist()}")
print(f"Scenarios (driving conditions) found: {data['sheet_name'].unique().tolist()}\n")

# -----------------------------------------------------
# 2. Përzgjedhja e kolonave relevante
# -----------------------------------------------------
features = [
    "Location (front/rear seat/inside front shield)",
    "Area description (Traffic and road conditions)",
    "Speed",
    "car_type",
    "sheet_name",
]

# Multi-output targets: B (magnetic field) dhe E (electric field)
targets = ["B (µT)", "Electric Field (V/m)"]

X = data[features].copy()
y = data[targets].copy()  # Tani y është DataFrame me 2 kolona

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
# IEEE C95.6-2002 reference levels (B in µT)
# -----------------------------------------------------
# IEEE Standard for Safety Levels with Respect to Human Exposure to Electromagnetic Fields,
# 0-3 kHz (IEEE C95.6-2002)
#
# NOTE: Verify these values against the exact IEEE C95.6-2002 document you cite in your thesis.
# The values below are commonly cited reference levels at 50 Hz for magnetic flux density (B).
IEEE_FREQUENCY_HZ = 50
IEEE_CATEGORY = "general_public"  # or: "occupational"

# IEEE C95.6-2002 reference levels at 50 Hz for magnetic flux density (B):
#   - General public: 904 µT (based on controlled environment limit)
#   - Occupational:   2710 µT (based on controlled environment limit)
# 
# NOTE: IEEE C95.6-2002 uses different terminology and may have different limits
# depending on the specific interpretation. These values should be verified against
# the actual standard document. Some sources cite:
#   - General public: 904 µT (controlled environment) or 2710 µT (uncontrolled)
#   - Occupational: 2710 µT
# 
# For 50 Hz specifically, verify the exact table values in IEEE C95.6-2002.
IEEE_LIMIT_B_UT_BY_CATEGORY = {
    "general_public": 904.0,   # µT - verify against IEEE C95.6-2002 document
    "occupational": 2710.0,     # µT - verify against IEEE C95.6-2002 document
}


def _ieee_limit_b_ut() -> float:
    return float(IEEE_LIMIT_B_UT_BY_CATEGORY.get(IEEE_CATEGORY, 904.0))


IEEE_LIMIT_B_UT = _ieee_limit_b_ut()

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
# Për multi-output, llogaritim metrikat për çdo target veç e veç
rmse_base_b = np.sqrt(mean_squared_error(y.iloc[:, 0], y_pred_base[:, 0]))
r2_base_b = r2_score(y.iloc[:, 0], y_pred_base[:, 0])
rmse_base_e = np.sqrt(mean_squared_error(y.iloc[:, 1], y_pred_base[:, 1]))
r2_base_e = r2_score(y.iloc[:, 1], y_pred_base[:, 1])
print("Baseline Performance (CV) - Multi-output:")
print(f"  B (µT):      RMSE: {rmse_base_b:.3f} µT,  R²: {r2_base_b:.3f}")
print(f"  E (V/m):     RMSE: {rmse_base_e:.3f} V/m, R²: {r2_base_e:.3f}\n")

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
# Për multi-output, llogaritim metrikat për çdo target veç e veç
rmse_cv_b = np.sqrt(mean_squared_error(y.iloc[:, 0], y_pred_cv[:, 0]))
r2_cv_b = r2_score(y.iloc[:, 0], y_pred_cv[:, 0])
rmse_cv_e = np.sqrt(mean_squared_error(y.iloc[:, 1], y_pred_cv[:, 1]))
r2_cv_e = r2_score(y.iloc[:, 1], y_pred_cv[:, 1])

print("Random Forest Performance (CV, best params) - Multi-output:")
print(f"  B (µT):      RMSE: {rmse_cv_b:.3f} µT,  R²: {r2_cv_b:.3f}")
print(f"  E (V/m):     RMSE: {rmse_cv_e:.3f} V/m, R²: {r2_cv_e:.3f}\n")

# -----------------------------------------------------
# 3b. TensorFlow/Keras Neural Network Model (për krahasim)
# -----------------------------------------------------
if TENSORFLOW_AVAILABLE:
    print("Training TensorFlow/Keras Neural Network model with CV...")
    
    # Përgatit të dhënat për TensorFlow (duhet të jenë numerike)
    X_preprocessed = preprocess.fit_transform(X)
    X_preprocessed_dense = X_preprocessed.toarray() if hasattr(X_preprocessed, 'toarray') else X_preprocessed
    
    # Funksion për të krijuar modelin Keras (multi-output)
    def create_keras_model(input_dim: int):
        """Krijon një model neural network për multi-output regresion (B dhe E)."""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(2)  # Output layer për 2 targets: B (µT) dhe E (V/m)
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    # Train në të gjithë dataset-in dhe vlerëso me train/test split të thjeshtë
    from sklearn.model_selection import train_test_split
    X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(
        X_preprocessed_dense, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Krijo dhe trajno modelin
    tf_model = create_keras_model(X_preprocessed_dense.shape[1])
    
    # Early stopping për të shmangur overfitting
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train modelin
    print("  Training TensorFlow model (this may take a moment)...")
    history = tf_model.fit(
        X_train_tf, y_train_tf,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        verbose=0,
        callbacks=[early_stopping]
    )
    
    # Vlerësimi (multi-output)
    y_pred_tf = tf_model.predict(X_test_tf, verbose=0)
    # Llogaritim metrikat për çdo target veç e veç
    rmse_tf_b = np.sqrt(mean_squared_error(y_test_tf.iloc[:, 0], y_pred_tf[:, 0]))
    r2_tf_b = r2_score(y_test_tf.iloc[:, 0], y_pred_tf[:, 0])
    rmse_tf_e = np.sqrt(mean_squared_error(y_test_tf.iloc[:, 1], y_pred_tf[:, 1]))
    r2_tf_e = r2_score(y_test_tf.iloc[:, 1], y_pred_tf[:, 1])
    
    print("TensorFlow/Keras Neural Network Performance (train/test split) - Multi-output:")
    print(f"  B (µT):      RMSE: {rmse_tf_b:.3f} µT,  R²: {r2_tf_b:.3f}")
    print(f"  E (V/m):     RMSE: {rmse_tf_e:.3f} V/m, R²: {r2_tf_e:.3f}")
    print(f"  Note: TensorFlow model uses train/test split (80/20) due to dataset size.\n")
    
    # Ruaj TensorFlow model metrics për summary
    tf_metrics = {
        "rmse_b": rmse_tf_b,
        "r2_b": r2_tf_b,
        "rmse_e": rmse_tf_e,
        "r2_e": r2_tf_e,
        "n_train": len(X_train_tf),
        "n_test": len(X_test_tf),
    }
else:
    tf_metrics = None
    print("TensorFlow not available. Skipping TensorFlow model evaluation.\n")

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
    ys = y.loc[mask].copy()  # Tani është DataFrame me 2 kolona: B dhe E

    # Risk stats from *measured* values (not model prediction)
    # Për B (magnetic field)
    y_b_clean = pd.to_numeric(ys.iloc[:, 0], errors="coerce").dropna()
    if len(y_b_clean) == 0:
        print(f"Skipping car_type={car}, sheet={sheet}: no valid B target values")
        return None
    
    y_b_mean = float(y_b_clean.mean())
    y_b_median = float(y_b_clean.median())
    y_b_p95 = float(y_b_clean.quantile(0.95))
    y_b_max = float(y_b_clean.max())
    
    # Për E (electric field)
    y_e_clean = pd.to_numeric(ys.iloc[:, 1], errors="coerce").dropna()
    y_e_mean = float(y_e_clean.mean()) if len(y_e_clean) > 0 else 0.0
    y_e_median = float(y_e_clean.median()) if len(y_e_clean) > 0 else 0.0
    y_e_p95 = float(y_e_clean.quantile(0.95)) if len(y_e_clean) > 0 else 0.0
    y_e_max = float(y_e_clean.max()) if len(y_e_clean) > 0 else 0.0
    
    # ICNIRP comparison (vetëm për B - magnetic field)
    exceeds_icnirp = bool(y_b_max > ICNIRP_LIMIT_B_UT)
    exceed_margin_icnirp = float(y_b_max - ICNIRP_LIMIT_B_UT)
    
    # IEEE C95.6-2002 comparison (vetëm për B - magnetic field)
    exceeds_ieee = bool(y_b_max > IEEE_LIMIT_B_UT)
    exceed_margin_ieee = float(y_b_max - IEEE_LIMIT_B_UT)

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
        "Speed": 80,
        "car_type": car,
        "sheet_name": sheet,
    }
    what_if_pred = model_s.predict(pd.DataFrame([what_if_row]))[0]  # Tani kthen array me 2 vlera: [B, E]
    what_if_pred_b = float(what_if_pred[0])  # B (µT)
    what_if_pred_e = float(what_if_pred[1])   # E (V/m)

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
    # Include both ICNIRP and IEEE comparisons, plus E field predictions
    icnirp_status = "YES" if exceeds_icnirp else "NO"
    ieee_status = "YES" if exceeds_ieee else "NO"
    fig.text(
        0.5,
        0.02,
        (
            f"What-if: B̂={what_if_pred_b:.3f} µT, Ê={what_if_pred_e:.3f} V/m   |   "
            f"Measured: max(B)={y_b_max:.3f} µT, max(E)={y_e_max:.3f} V/m   |   "
            f"ICNIRP({ICNIRP_FREQUENCY_HZ} Hz): {ICNIRP_LIMIT_B_UT:.0f} µT (Exceeds: {icnirp_status})   |   "
            f"IEEE C95.6-2002({IEEE_FREQUENCY_HZ} Hz): {IEEE_LIMIT_B_UT:.0f} µT (Exceeds: {ieee_status})"
        ),
        ha="center",
        va="bottom",
        fontsize=PLOT_STYLE["base_fontsize"] - 2,  # Even smaller to fit more text
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
        # B (magnetic field) statistics
        "measured_mean_b": y_b_mean,
        "measured_median_b": y_b_median,
        "measured_p95_b": y_b_p95,
        "measured_max_b": y_b_max,
        # E (electric field) statistics
        "measured_mean_e": y_e_mean,
        "measured_median_e": y_e_median,
        "measured_p95_e": y_e_p95,
        "measured_max_e": y_e_max,
        # ICNIRP statistics (vetëm për B)
        "icnirp_limit_b_ut": float(ICNIRP_LIMIT_B_UT),
        "exceeds_icnirp": exceeds_icnirp,
        "exceed_margin_icnirp": exceed_margin_icnirp,
        # IEEE C95.6-2002 statistics (vetëm për B)
        "ieee_limit_b_ut": float(IEEE_LIMIT_B_UT),
        "exceeds_ieee": exceeds_ieee,
        "exceed_margin_ieee": exceed_margin_ieee,
        # Model statistics
        "cv_splits": int(n_splits),
        "best_params": dict(search_s.best_params_),
        "what_if_pred_b": what_if_pred_b,
        "what_if_pred_e": what_if_pred_e,
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
# 5c. Combined plots per car type (all sheets merged)
# -----------------------------------------------------
print("\nGenerating combined plots per car type (all sheets merged)...")

def _create_combined_car_plot(car_type: str) -> None:
    """Krijon një plot të kombinuar për një car type, duke kombinuar të gjitha sheets."""
    # Filtro per-sheet results për këtë car type
    car_results = [r for r in per_sheet_results if r["car_type"] == car_type]
    
    if not car_results:
        print(f"Skipping combined plot for {car_type}: no data available")
        return
    
    # Lexo të gjitha CSV files për këtë car type dhe kombino ato
    all_feature_importances = []
    all_sheet_names = []
    
    for result in car_results:
        csv_path = Path(result["csv"])
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["sheet_name"] = result["sheet_name"]
            all_feature_importances.append(df)
            all_sheet_names.append(result["sheet_name"])
    
    if not all_feature_importances:
        print(f"Skipping combined plot for {car_type}: no CSV files found")
        return
    
    # Kombino të gjitha feature importances
    combined_df = pd.concat(all_feature_importances, ignore_index=True)
    
    # Agregato feature importances (mesatarja për të gjitha sheets)
    feature_agg = combined_df.groupby("feature")["importance"].agg(["mean", "count"]).reset_index()
    feature_agg = feature_agg.sort_values("mean", ascending=False)
    
    # Merr top N features
    top_n = PLOT_STYLE["top_n"]
    top_features = feature_agg.head(top_n)
    
    # Ruaj CSV për combined plot
    combined_csv_path = results_dir / f"feature_importance_{car_type}_combined.csv"
    top_features.to_csv(combined_csv_path, index=False)
    
    # Krijo plot
    plt.rcParams.update({
        "font.size": PLOT_STYLE["base_fontsize"],
        "axes.titlesize": PLOT_STYLE["title_fontsize"],
        "axes.labelsize": 12,
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Plot 1: Feature Importance (mesatarja për të gjitha sheets)
    ax1 = axes[0]
    top_features.sort_values("mean", ascending=True).plot(
        kind="barh", x="feature", y="mean", color=PLOT_STYLE["bar_color"], ax=ax1
    )
    ax1.set_title(f"Feature Importance (Average) – {car_type}", fontsize=PLOT_STYLE["title_fontsize"])
    ax1.set_xlabel("Importance")
    ax1.set_ylabel("")
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    
    # Plot 2: Distribution of B (µT) across all sheets
    ax2 = axes[1]
    b_values = []
    sheet_labels = []
    for result in car_results:
        # Lexo të dhënat origjinale për këtë sheet
        mask = (X["car_type"] == car_type) & (X["sheet_name"] == result["sheet_name"])
        ys_sheet = y.loc[mask].copy()
        y_b_clean = pd.to_numeric(ys_sheet.iloc[:, 0], errors="coerce").dropna()
        if len(y_b_clean) > 0:
            b_values.extend(y_b_clean.tolist())
            sheet_labels.extend([result["sheet_name"]] * len(y_b_clean))
    
    if b_values:
        ax2.hist(b_values, bins=20, color=PLOT_STYLE["bar_color"], alpha=0.7, edgecolor="black")
        ax2.axvline(ICNIRP_LIMIT_B_UT, color="red", linestyle="--", linewidth=2, label=f"ICNIRP: {ICNIRP_LIMIT_B_UT:.0f} µT")
        ax2.axvline(IEEE_LIMIT_B_UT, color="orange", linestyle="--", linewidth=2, label=f"IEEE: {IEEE_LIMIT_B_UT:.0f} µT")
        ax2.set_title(f"Distribution of B (µT) – {car_type}", fontsize=PLOT_STYLE["title_fontsize"])
        ax2.set_xlabel("B (µT)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(axis="y", linestyle="--", alpha=0.35)
    
    # Plot 3: Distribution of E (V/m) across all sheets
    ax3 = axes[2]
    e_values = []
    for result in car_results:
        mask = (X["car_type"] == car_type) & (X["sheet_name"] == result["sheet_name"])
        ys_sheet = y.loc[mask].copy()
        y_e_clean = pd.to_numeric(ys_sheet.iloc[:, 1], errors="coerce").dropna()
        if len(y_e_clean) > 0:
            e_values.extend(y_e_clean.tolist())
    
    if e_values:
        ax3.hist(e_values, bins=20, color="#4A90E2", alpha=0.7, edgecolor="black")
        ax3.set_title(f"Distribution of E (V/m) – {car_type}", fontsize=PLOT_STYLE["title_fontsize"])
        ax3.set_xlabel("E (V/m)")
        ax3.set_ylabel("Frequency")
        ax3.grid(axis="y", linestyle="--", alpha=0.35)
    
    # Statistikat e përgjithshme për këtë car type
    all_b_max = max([r["measured_max_b"] for r in car_results]) if car_results else 0
    all_e_max = max([r["measured_max_e"] for r in car_results]) if car_results else 0
    all_b_mean = np.mean([r["measured_mean_b"] for r in car_results]) if car_results else 0
    all_e_mean = np.mean([r["measured_mean_e"] for r in car_results]) if car_results else 0
    
    exceeds_icnirp_combined = bool(all_b_max > ICNIRP_LIMIT_B_UT)
    exceeds_ieee_combined = bool(all_b_max > IEEE_LIMIT_B_UT)
    
    # Shto tekst në fund të plot-it
    fig.text(
        0.5,
        0.02,
        (
            f"Combined stats: B: mean={all_b_mean:.3f} µT, max={all_b_max:.3f} µT | "
            f"E: mean={all_e_mean:.3f} V/m, max={all_e_max:.3f} V/m | "
            f"Sheets: {len(all_sheet_names)} | "
            f"ICNIRP exceeds: {'YES' if exceeds_icnirp_combined else 'NO'} | "
            f"IEEE exceeds: {'YES' if exceeds_ieee_combined else 'NO'}"
        ),
        ha="center",
        va="bottom",
        fontsize=PLOT_STYLE["base_fontsize"] - 1,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F2F4F8", "edgecolor": "#C7CEDB"},
    )
    
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    
    # Ruaj plot
    combined_png_path = results_dir / f"feature_importance_{car_type}_combined.png"
    plt.savefig(combined_png_path, dpi=PLOT_STYLE["dpi"])
    plt.close()
    
    print(f"Saved combined plot for {car_type}: {combined_png_path}")
    print(f"Saved combined CSV for {car_type}: {combined_csv_path}")

# Krijo combined plots për çdo car type
for car_type in sorted(X["car_type"].unique()):
    _create_combined_car_plot(car_type)

# -----------------------------------------------------
# 5b. Risk table export (ICNIRP and IEEE C95.6-2002 comparison)
# -----------------------------------------------------
risk_csv_path = results_dir / "exposure_risk_by_sheet.csv"
if per_sheet_results:
    risk_df = pd.DataFrame(per_sheet_results)
    # Rank by exceed first (ICNIRP is more conservative), then by measured max(B)
    risk_df = risk_df.sort_values(["exceeds_icnirp", "exceeds_ieee", "measured_max_b"], ascending=[False, False, False])
    risk_df.to_csv(risk_csv_path, index=False)
    print(f"Saved risk comparison table: {risk_csv_path}")
else:
    risk_df = pd.DataFrame()

# Also save a legacy ICNIRP-only file for backward compatibility
icnirp_only_csv_path = results_dir / "icnirp_risk_by_sheet.csv"
if not risk_df.empty:
    icnirp_cols = [col for col in risk_df.columns if col.startswith(("car_type", "sheet_name", "n_rows", "measured_", "icnirp_", "cv_splits", "best_params", "what_if_pred"))]
    risk_df[icnirp_cols].to_csv(icnirp_only_csv_path, index=False)

# -----------------------------------------------------
# 6. Ruajtje e rezultateve (summary) në results/
# -----------------------------------------------------
summary_path = results_dir / "model_results.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("RF ELF Exposure Model – Results Summary\n")
    f.write("=====================================\n\n")
    f.write(f"Total rows used: {len(X)}\n")
    f.write(f"Targets: {targets[0]} and {targets[1]} (Multi-output regression)\n\n")

    f.write("Baseline (DummyRegressor, CV=3-fold) - Multi-output\n")
    f.write(f"  B (µT):      RMSE: {rmse_base_b:.3f} µT,  R²: {r2_base_b:.3f}\n")
    f.write(f"  E (V/m):     RMSE: {rmse_base_e:.3f} V/m, R²: {r2_base_e:.3f}\n\n")

    f.write("Random Forest (best via GridSearchCV, CV=3-fold) - Multi-output\n")
    f.write(f"  Best params: {search.best_params_}\n")
    f.write(f"  B (µT):      RMSE: {rmse_cv_b:.3f} µT,  R²: {r2_cv_b:.3f}\n")
    f.write(f"  E (V/m):     RMSE: {rmse_cv_e:.3f} V/m, R²: {r2_cv_e:.3f}\n\n")
    
    if TENSORFLOW_AVAILABLE and tf_metrics:
        f.write("TensorFlow/Keras Neural Network (train/test split 80/20) - Multi-output\n")
        f.write(f"  Architecture: Dense(64) -> Dropout(0.2) -> Dense(32) -> Dropout(0.2) -> Dense(16) -> Dense(2)\n")
        f.write(f"  Optimizer: Adam (lr=0.001)\n")
        f.write(f"  Training samples: {tf_metrics['n_train']}\n")
        f.write(f"  Test samples: {tf_metrics['n_test']}\n")
        f.write(f"  B (µT):      RMSE: {tf_metrics['rmse_b']:.3f} µT,  R²: {tf_metrics['r2_b']:.3f}\n")
        f.write(f"  E (V/m):     RMSE: {tf_metrics['rmse_e']:.3f} V/m, R²: {tf_metrics['r2_e']:.3f}\n")
        f.write(f"  Note: TensorFlow model uses train/test split due to dataset size.\n\n")
    else:
        f.write("TensorFlow/Keras Neural Network\n")
        f.write("  Status: Not available (TensorFlow not installed or import failed)\n\n")

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
    f.write("  Speed:    80 km/h\n")
    f.write("  Note: Both B (µT) and E (V/m) are predicted for this scenario\n")

    f.write("\nICNIRP comparison (ELF)\n")
    f.write(f"  Frequency: {ICNIRP_FREQUENCY_HZ} Hz\n")
    f.write(f"  Category:  {ICNIRP_CATEGORY}\n")
    f.write(f"  Limit B:   {ICNIRP_LIMIT_B_UT:.0f} µT\n")
    
    f.write("\nIEEE C95.6-2002 comparison (ELF)\n")
    f.write(f"  Frequency: {IEEE_FREQUENCY_HZ} Hz\n")
    f.write(f"  Category:  {IEEE_CATEGORY}\n")
    f.write(f"  Limit B:   {IEEE_LIMIT_B_UT:.0f} µT\n")
    f.write("  NOTE: Verify these IEEE values against the exact IEEE C95.6-2002 document.\n")
    
    if risk_csv_path.exists():
        f.write(f"\nRisk comparison table (ICNIRP + IEEE): {risk_csv_path.name}\n")
    f.write("\nPer-sheet what-if predictions (Multi-output)\n")
    for r in per_sheet_results:
        f.write(
            f"  - {r['car_type']} / {r['sheet_name']} (n={r['n_rows']}, CV={r['cv_splits']}): "
            f"B̂={r['what_if_pred_b']:.3f} µT, Ê={r['what_if_pred_e']:.3f} V/m\n"
        )

    if not risk_df.empty:
        f.write("\nPer-sheet measured risk (Top 10 by measured max(B))\n")
        top10 = risk_df.head(10)
        for _, row in top10.iterrows():
            icnirp_exceed = "YES" if bool(row['exceeds_icnirp']) else "NO"
            ieee_exceed = "YES" if bool(row['exceeds_ieee']) else "NO"
            f.write(
                f"  - {row['car_type']} / {row['sheet_name']}: "
                f"B: mean={row['measured_mean_b']:.3f} µT, p95={row['measured_p95_b']:.3f} µT, max={row['measured_max_b']:.3f} µT; "
                f"E: mean={row['measured_mean_e']:.3f} V/m, max={row['measured_max_e']:.3f} V/m; "
                f"ICNIRP exceeds: {icnirp_exceed}, IEEE exceeds: {ieee_exceed}\n"
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
            f"Predicted [{r['car_type']} / {r['sheet_name']}]: "
            f"B={r['what_if_pred_b']:.3f} µT, E={r['what_if_pred_e']:.3f} V/m"
        )

print("\nExecution finished successfully.")
