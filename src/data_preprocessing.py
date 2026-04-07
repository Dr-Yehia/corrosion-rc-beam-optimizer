# ============================================================
# src/data_preprocessing.py
# Corrosion RC Beam Optimizer
# Full pipeline: load → clean → engineer → scale → split
# ============================================================

import re
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    DATA_RAW, DATA_CLEAN, FEATURE_COLS, TARGET_COL,
    TEST_SIZE, RANDOM_STATE, SCALER_X_PATH, SCALER_Y_PATH
)


# ============================================================
# COLUMN NAME CANONICALIZER
# ============================================================

def _canonicalize(name: str) -> str:
    """
    Strip encoding artefacts, lowercase, collapse whitespace.
    Used for fuzzy column matching.
    """
    # Re-encode latin-1 → utf-8 if possible
    try:
        name = name.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    # Remove BOM and non-ASCII noise
    name = name.replace('\ufeff', '').replace('\u00ef\u00bb\u00bf', '')
    # Lowercase + collapse whitespace
    return re.sub(r'\s+', ' ', name).strip().lower()


_CANONICAL_MAP = {
    # Mass loss / eta
    'mass loss (tensile bars), ηm (%)': 'Mass Loss (Tensile bars), ηm (%)',
    'mass loss (tensile bars), î·m (%)': 'Mass Loss (Tensile bars), ηm (%)',
    'mass loss (tensile bars), ïm (%)': 'Mass Loss (Tensile bars), ηm (%)',
    'mass loss (tensile bars), Îm (%)': 'Mass Loss (Tensile bars), ηm (%)',
    # Mmax experimental
    'mmax,exp (knm)': 'Mmax,exp (kNm)',
    'mmax,exp (kn·m)': 'Mmax,exp (kNm)',
    'mmax (knm)': 'Mmax,exp (kNm)',
    # No. column
    'ï»¿no.': 'No.',
    'no.': 'No.',
}


def _fix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename all columns to their canonical form."""
    rename = {}
    for col in df.columns:
        canon = _canonicalize(col)
        # Direct lookup
        if canon in _CANONICAL_MAP:
            rename[col] = _CANONICAL_MAP[canon]
            continue
        # Fuzzy: contains 'mass loss' and 'm (%)'  → eta column
        if 'mass loss' in canon and '(%)' in canon:
            rename[col] = 'Mass Loss (Tensile bars), ηm (%)'
            continue
        # Fuzzy: mmax and exp
        if 'mmax' in canon and 'exp' in canon:
            rename[col] = 'Mmax,exp (kNm)'
            continue
    if rename:
        logger.info(f"Column names normalised: {list(rename.values())}")
        df = df.rename(columns=rename)
    return df


# ============================================================
# 1. LOAD
# ============================================================
def load_raw_data(path: Path = DATA_RAW) -> pd.DataFrame:
    """Load CSV, trying multiple encodings, then fix column names."""
    logger.info(f"Loading raw data from: {path}")
    for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info(f"Raw data loaded (encoding={enc}) — shape: {df.shape}")
            df = _fix_columns(df)
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode {path} with any known encoding.")


# ============================================================
# 2. INSPECT
# ============================================================
def inspect_data(df: pd.DataFrame) -> None:
    logger.info("=== Dataset Inspection ===")
    logger.info(f"  Rows       : {df.shape[0]}")
    logger.info(f"  Columns    : {df.shape[1]}")
    missing = (df.isnull().mean() * 100).round(2)
    missing = missing[missing > 0]
    if len(missing) > 0:
        logger.info(f"  Missing %:\n{missing.to_string()}")
    else:
        logger.info("  No missing values detected.")


# ============================================================
# 3. CLEAN
# ============================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data cleaning ...")
    df = df.copy()

    # Always keep Mmax,exp for ACI benchmark even if not in FEATURE_COLS
    aci_extra = ['Mmax,exp (kNm)']
    required_cols = list(set(FEATURE_COLS + [TARGET_COL] + aci_extra))
    available     = [c for c in required_cols if c in df.columns]
    missing_cols  = set(required_cols) - set(available)
    if missing_cols:
        logger.warning(f"Missing expected columns: {missing_cols}")
    df = df[available].copy()
    logger.info(f"Columns selected: {df.shape[1]}")

    # Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    logger.info(f"Dropped {before - len(df)} rows with missing target.")

    # Impute missing numeric features with median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            logger.info(f"  Imputed '{col}' with median = {med:.3f}")

    # Physical filters
    eta_col = 'Mass Loss (Tensile bars), ηm (%)'
    if eta_col in df.columns:
        before = len(df)
        df = df[(df[eta_col] >= 0) & (df[eta_col] <= 64)]
        logger.info(f"Physical filter (ηm 0-64%): removed {before - len(df)} rows.")

    before = len(df)
    df = df[(df[TARGET_COL] > 0) & (df[TARGET_COL] <= 130.1)]
    logger.info(f"Physical filter (R 0-130%): removed {before - len(df)} rows.")

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
    df   = df.copy()
    eta  = 'Mass Loss (Tensile bars), ηm (%)'
    fy   = 'fy Longitudinal Bars (Tensile), (MPa) '
    fc   = "f'c (MPa)"
    d    = 'Depth (mm)'
    b    = 'Width (mm)'

    if all(c in df.columns for c in [eta, fy, fc, d, b]):
        df['corr_severity_idx'] = df[eta] * (df[fy] / df[fc])
        df['d_b_ratio']         = df[d]   / df[b]
        df['eta_d_interaction'] = df[eta] * df[d]
        logger.info('Feature engineering: 3 derived features added.')
    else:
        missing = [c for c in [eta, fy, fc, d, b] if c not in df.columns]
        logger.warning(f'Feature engineering skipped — missing: {missing}')

    return df


# ============================================================
# 5. SCALE
# ============================================================
def scale_features(X_train, X_test, y_train, y_test, save: bool = True):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc  = scaler_X.transform(X_test)
    y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_sc  = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    if save:
        joblib.dump(scaler_X, SCALER_X_PATH)
        joblib.dump(scaler_y, SCALER_Y_PATH)
        logger.info(f"Scalers saved → {SCALER_X_PATH}, {SCALER_Y_PATH}")

    return X_train_sc, X_test_sc, y_train_sc, y_test_sc, scaler_X, scaler_y


# ============================================================
# 6. SPLIT
# ============================================================
def split_data(df: pd.DataFrame):
    base_features = [c for c in FEATURE_COLS if c in df.columns]
    engineered    = ['corr_severity_idx', 'd_b_ratio', 'eta_d_interaction']
    feature_cols  = base_features + [c for c in engineered if c in df.columns]

    X      = df[feature_cols]
    y      = df[TARGET_COL]
    y_bins = pd.qcut(y, q=4, labels=False, duplicates='drop')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y_bins,
    )

    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"y_train — mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
    logger.info(f"y_test  — mean: {y_test.mean():.2f},  std: {y_test.std():.2f}")

    return X_train, X_test, y_train, y_test


# ============================================================
# 7. FULL PIPELINE
# ============================================================
def run_preprocessing(save_clean: bool = True) -> dict:
    logger.info('═' * 50)
    logger.info(' Starting Preprocessing Pipeline')
    logger.info('═' * 50)

    df_raw   = load_raw_data()
    inspect_data(df_raw)
    df_clean = clean_data(df_raw)
    df_feat  = engineer_features(df_clean)

    if save_clean:
        df_feat.to_csv(DATA_CLEAN, index=False)
        logger.info(f"Clean data saved → {DATA_CLEAN}  ({len(df_feat)} rows)")

    X_train, X_test, y_train, y_test = split_data(df_feat)

    (X_train_sc, X_test_sc,
     y_train_sc, y_test_sc,
     scaler_X, scaler_y) = scale_features(X_train, X_test, y_train, y_test)

    logger.info('═' * 50)
    logger.info(' Preprocessing complete ✓')
    logger.info('═' * 50)

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
        'df_clean'     : df_feat,
    }


if __name__ == '__main__':
    results = run_preprocessing(save_clean=True)
    print(f"\n✅ Preprocessing done.")
    print(f"   Train samples : {results['X_train'].shape[0]}")
    print(f"   Test  samples : {results['X_test'].shape[0]}")
    print(f"   Features used : {len(results['feature_cols'])}")
    print(f"   Feature list  : {results['feature_cols']}")
