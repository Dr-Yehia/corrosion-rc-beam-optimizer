# ============================================================
# src/data_preprocessing.py
# Corrosion RC Beam Optimizer
# Full pipeline: load → clean → engineer → encode → scale → split
#
# v4 changes:
#   + Target changed to Mmax,exp (kNm) — matching Zhang et al.
#   + R(%) kept as secondary column for comparison/reporting
#   + Encode 3 categorical features
#   + log1p(ηm) + derived features
#   + RobustScaler
# ============================================================

import re
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from loguru import logger

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    DATA_RAW, DATA_CLEAN, FEATURE_COLS, CAT_COLS, TARGET_COL,
    TARGET_COL_R, TEST_SIZE, RANDOM_STATE, SCALER_X_PATH, SCALER_Y_PATH
)

ETA_COL = 'Mass Loss (Tensile bars), \u03b7m (%)'


# ============================================================
# COLUMN NAME CANONICALIZER
# ============================================================
def _canonicalize(name: str) -> str:
    try:
        name = name.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    name = name.replace('\ufeff', '').replace('\u00ef\u00bb\u00bf', '')
    return re.sub(r'\s+', ' ', name).strip().lower()


def _fix_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        canon = _canonicalize(col)
        if 'mass loss' in canon and '(%)' in canon:
            rename[col] = ETA_COL
        elif 'mmax' in canon and 'exp' in canon:
            rename[col] = 'Mmax,exp (kNm)'
        elif canon in ('\uf8ffno.', 'no.', '\u00ef\u00bb\u00bfno.'):
            rename[col] = 'No.'
    if rename:
        logger.info(f"Column names normalised: {list(rename.values())}")
        df = df.rename(columns=rename)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    return df


# ============================================================
# 1. LOAD
# ============================================================
def load_raw_data(path: Path = DATA_RAW) -> pd.DataFrame:
    logger.info(f"Loading raw data from: {path}")
    for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info(f"Raw data loaded (encoding={enc}) — shape: {df.shape}")
            df = _fix_columns(df)
            logger.info(f"After column fix — shape: {df.shape}")
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode {path} with any known encoding.")


# ============================================================
# 2. INSPECT
# ============================================================
def inspect_data(df: pd.DataFrame) -> None:
    logger.info("=== Dataset Inspection ===")
    logger.info(f"  Rows    : {df.shape[0]}")
    logger.info(f"  Columns : {df.shape[1]}")
    missing = (df.isnull().mean() * 100).round(2)
    missing = missing[missing > 0]
    if len(missing) > 0:
        logger.info(f"  Missing %:\n{missing.to_string()}")
    else:
        logger.info("  No missing values.")


# ============================================================
# 3. CLEAN
# ============================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data cleaning ...")
    df = df.copy()

    # Select required columns: numeric + categorical + targets + ACI extras
    aci_extra     = [ETA_COL]
    targets       = [TARGET_COL, TARGET_COL_R]
    cat_available = [c for c in CAT_COLS if c in df.columns]
    required_cols = list(dict.fromkeys(
        FEATURE_COLS + cat_available + targets + aci_extra
    ))
    available    = [c for c in required_cols if c in df.columns]
    missing_cols = set(required_cols) - set(available)
    if missing_cols:
        logger.warning(f"Missing expected columns: {missing_cols}")
    df = df[available].copy()
    logger.info(f"Columns selected: {df.shape[1]} "
                f"({len(cat_available)} categorical + "
                f"{len([c for c in FEATURE_COLS if c in available])} numeric)")

    # Drop rows with missing PRIMARY target (Mmax,exp)
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    logger.info(f"Dropped {before - len(df)} rows with missing target ({TARGET_COL}).")

    # Impute missing numeric features with median
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in [TARGET_COL, TARGET_COL_R]:
            continue  # don't impute targets
        n_null = int(df[col].isnull().sum())
        if n_null > 0:
            med = float(df[col].median())
            df[col] = df[col].fillna(med)
            logger.info(f"  Imputed '{col}' ({n_null} nulls) with median = {med:.3f}")

    # Impute missing categorical features with mode
    for col in cat_available:
        n_null = int(df[col].isnull().sum())
        if n_null > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  Imputed '{col}' ({n_null} nulls) with mode = '{mode_val}'")

    # Physical filters
    if ETA_COL in df.columns:
        before = len(df)
        df = df[(df[ETA_COL] >= 0) & (df[ETA_COL] <= 64)]
        logger.info(f"Physical filter (\u03b7m 0-64%): removed {before - len(df)} rows.")

    # Mmax,exp must be positive
    before = len(df)
    df = df[df[TARGET_COL] > 0]
    logger.info(f"Physical filter (Mmax > 0): removed {before - len(df)} rows.")

    for col in ['Width (mm)', 'Depth (mm)']:
        if col in df.columns:
            df = df[df[col] > 0]

    df = df.reset_index(drop=True)
    logger.info(f"Clean data shape: {df.shape}")
    return df


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df  = df.copy()
    fy  = 'fy Longitudinal Bars (Tensile), (MPa) '
    fc  = "f'c (MPa)"
    d   = 'Depth (mm)'
    b   = 'Width (mm)'

    # Log1p-transform ηm (right-skewed)
    if ETA_COL in df.columns:
        df['eta_log'] = np.log1p(df[ETA_COL])
        logger.info("Applied log1p transform to \u03b7m \u2192 'eta_log'")

    if all(c in df.columns for c in [ETA_COL, fy, fc, d, b]):
        df['corr_severity_idx'] = df[ETA_COL] * (df[fy] / df[fc])
        df['d_b_ratio']         = df[d]        / df[b]
        df['eta_d_interaction'] = df['eta_log'] * df[d]
        # NEW: reinforcement index (As_proxy * fy / (fc * b * d))
        n_bars_col = '# Tensile Bars'
        db_col = 'Diameter Tensile Bars, db,t (mm)'
        if all(c in df.columns for c in [n_bars_col, db_col]):
            As_proxy = df[n_bars_col] * np.pi * (df[db_col] / 2.0) ** 2
            df['reinf_index'] = As_proxy * df[fy] / (df[fc] * df[b] * df[d])
        logger.info('Feature engineering: 5 derived features added.')
    else:
        miss = [c for c in [ETA_COL, fy, fc, d, b] if c not in df.columns]
        logger.warning(f'Feature engineering skipped — missing: {miss}')

    return df


# ============================================================
# 5. ENCODE CATEGORICALS
# ============================================================
def encode_categoricals(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    import json
    from config import MODELS_DIR

    df = df.copy()
    encoders = {}
    cat_available = [c for c in CAT_COLS if c in df.columns]

    for col in cat_available:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = list(le.classes_)
        logger.info(f"Encoded '{col}': {dict(enumerate(le.classes_))}")

    if save and encoders:
        enc_path = MODELS_DIR / "cat_encoders.json"
        with open(enc_path, 'w') as f:
            json.dump(encoders, f, indent=2)
        logger.info(f"Encoder mapping saved \u2192 {enc_path}")

    return df


# ============================================================
# 6. SCALE  (RobustScaler)
# ============================================================
def scale_features(X_train, X_test, y_train, y_test, save: bool = True):
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc  = scaler_X.transform(X_test)
    y_train_sc = scaler_y.fit_transform(
        y_train.values.reshape(-1, 1)).ravel()
    y_test_sc  = scaler_y.transform(
        y_test.values.reshape(-1, 1)).ravel()

    if save:
        joblib.dump(scaler_X, SCALER_X_PATH)
        joblib.dump(scaler_y, SCALER_Y_PATH)
        logger.info(f"Scalers saved \u2192 {SCALER_X_PATH}, {SCALER_Y_PATH}")

    return X_train_sc, X_test_sc, y_train_sc, y_test_sc, scaler_X, scaler_y


# ============================================================
# 7. SPLIT
# ============================================================
def split_data(df: pd.DataFrame):
    base_features = [c for c in FEATURE_COLS if c in df.columns]
    cat_available = [c for c in CAT_COLS     if c in df.columns]
    engineered    = ['eta_log', 'corr_severity_idx',
                     'd_b_ratio', 'eta_d_interaction', 'reinf_index']
    feature_cols  = (base_features + cat_available +
                     [c for c in engineered if c in df.columns])

    X      = df[feature_cols]
    y      = df[TARGET_COL]
    y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y_bins,
    )

    logger.info(f"Features total: {len(feature_cols)} "
                f"({len(base_features)} numeric + "
                f"{len(cat_available)} categorical + "
                f"{len([c for c in engineered if c in df.columns])} engineered)")
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"y_train — mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
    logger.info(f"y_test  — mean: {y_test.mean():.2f},  std: {y_test.std():.2f}")
    return X_train, X_test, y_train, y_test


# ============================================================
# 8. FULL PIPELINE
# ============================================================
def run_preprocessing(save_clean: bool = True) -> dict:
    logger.info('\u2550' * 50)
    logger.info(' Starting Preprocessing Pipeline')
    logger.info('\u2550' * 50)

    df_raw   = load_raw_data()
    inspect_data(df_raw)
    df_clean = clean_data(df_raw)
    df_feat  = engineer_features(df_clean)
    df_enc   = encode_categoricals(df_feat, save=True)

    if save_clean:
        df_enc.to_csv(DATA_CLEAN, index=False)
        logger.info(f"Clean data saved \u2192 {DATA_CLEAN}  ({len(df_enc)} rows)")

    X_train, X_test, y_train, y_test = split_data(df_enc)

    (X_train_sc, X_test_sc,
     y_train_sc, y_test_sc,
     scaler_X, scaler_y) = scale_features(X_train, X_test, y_train, y_test)

    logger.info('\u2550' * 50)
    logger.info(' Preprocessing complete \u2713')
    logger.info('\u2550' * 50)

    return {
        'X_train'      : X_train_sc,
        'X_test'       : X_test_sc,
        'y_train'      : y_train_sc,
        'y_test'       : y_test_sc,
        'X_train_raw'  : X_train,
        'X_test_raw'   : X_test,
        'y_train_raw'  : y_train,
        'y_test_raw'   : y_test,
        'scaler_X'     : scaler_X,
        'scaler_y'     : scaler_y,
        'feature_cols' : X_train.columns.tolist(),
        'df_clean'     : df_enc,
    }


if __name__ == '__main__':
    results = run_preprocessing(save_clean=True)
    print(f"\n\u2705 Preprocessing done.")
    print(f"   Target         : {TARGET_COL}")
    print(f"   Train samples  : {results['X_train'].shape[0]}")
    print(f"   Test  samples  : {results['X_test'].shape[0]}")
    print(f"   Features used  : {len(results['feature_cols'])}")
    print(f"   Feature list   : {results['feature_cols']}")
