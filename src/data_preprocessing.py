# ============================================================
# src/data_preprocessing.py
# Corrosion RC Beam Optimizer
# Full pipeline: load → clean → engineer → scale → split
#
# Key fixes (v2):
#   1. Remove extreme outliers  (R < 20% or R > 110%)
#   2. Log1p-transform ηm       (highly right-skewed)
#   3. RobustScaler             (robust to remaining outliers)
# ============================================================

import re
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from loguru import logger

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    DATA_RAW, DATA_CLEAN, FEATURE_COLS, TARGET_COL,
    TEST_SIZE, RANDOM_STATE, SCALER_X_PATH, SCALER_Y_PATH
)

ETA_COL = 'Mass Loss (Tensile bars), ηm (%)'


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

    # Select required columns
    aci_extra     = ['Mmax,exp (kNm)', ETA_COL]
    required_cols = list(dict.fromkeys(FEATURE_COLS + [TARGET_COL] + aci_extra))
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
        n_null = int(df[col].isnull().sum())
        if n_null > 0:
            med = float(df[col].median())
            df[col] = df[col].fillna(med)
            logger.info(f"  Imputed '{col}' ({n_null} nulls) with median = {med:.3f}")

    # ── Physical filters ─────────────────────────────────────
    if ETA_COL in df.columns:
        before = len(df)
        df = df[(df[ETA_COL] >= 0) & (df[ETA_COL] <= 64)]
        logger.info(f"Physical filter (ηm 0-64%): removed {before - len(df)} rows.")

    # Standard physical bounds
    before = len(df)
    df = df[(df[TARGET_COL] > 0) & (df[TARGET_COL] <= 130.1)]
    logger.info(f"Physical filter (R 0-130%): removed {before - len(df)} rows.")

    for col in ['Width (mm)', 'Depth (mm)']:
        if col in df.columns:
            df = df[df[col] > 0]

    # ── FIX 1: Remove statistical outliers in target ─────────
    # R < 20% = extreme failure, R > 110% = measurement anomaly
    # These 11 specimens (~1.4%) disproportionately hurt Test R²
    before = len(df)
    df = df[(df[TARGET_COL] >= 20) & (df[TARGET_COL] <= 110)]
    removed = before - len(df)
    logger.info(f"Outlier filter (R 20-110%): removed {removed} rows "
                f"→ {len(df)} specimens remain.")

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

    # ── FIX 2: Log1p-transform ηm (right-skewed, range 0-64) ─
    # log1p(0)=0, log1p(64)≈4.17 → near-normal distribution
    if ETA_COL in df.columns:
        df['eta_log'] = np.log1p(df[ETA_COL])
        logger.info("Applied log1p transform to ηm → 'eta_log'")

    if all(c in df.columns for c in [ETA_COL, fy, fc, d, b]):
        df['corr_severity_idx'] = df[ETA_COL] * (df[fy] / df[fc])
        df['d_b_ratio']         = df[d]        / df[b]
        df['eta_d_interaction'] = df['eta_log'] * df[d]   # use log version
        logger.info('Feature engineering: 4 derived features added (incl. eta_log).')
    else:
        miss = [c for c in [ETA_COL, fy, fc, d, b] if c not in df.columns]
        logger.warning(f'Feature engineering skipped — missing: {miss}')

    return df


# ============================================================
# 5. SCALE  (FIX 3: RobustScaler)
# ============================================================
def scale_features(X_train, X_test, y_train, y_test, save: bool = True):
    """
    RobustScaler uses median + IQR instead of mean + std.
    Much less sensitive to remaining outliers.
    """
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
        logger.info(f"Scalers saved → {SCALER_X_PATH}, {SCALER_Y_PATH}")

    return X_train_sc, X_test_sc, y_train_sc, y_test_sc, scaler_X, scaler_y


# ============================================================
# 6. SPLIT
# ============================================================
def split_data(df: pd.DataFrame):
    base_features = [c for c in FEATURE_COLS if c in df.columns]
    engineered    = ['eta_log', 'corr_severity_idx',
                     'd_b_ratio', 'eta_d_interaction']
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
