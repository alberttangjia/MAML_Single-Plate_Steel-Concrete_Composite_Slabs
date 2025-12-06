# -*- coding: utf-8 -*-
"""
Script Function: XGBoost prediction for punching shear capacity.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import os
import joblib

# ---Paths---
DATA_PATH = '../data'
OUTPUT_DIR = '../results'

# --- Model Settings ---
USE_K_FOLD = True
K_FOLDS = 5
RANDOM_SEED = 50
TRAIN_SET_SIZE = 36  # Only for non-K-Fold mode

# --- XGBoost Params ---
XGB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.005,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'n_jobs': 1,
    'random_state': RANDOM_SEED
}
EARLY_STOPPING_ROUNDS = 50

# Apply Seed
np.random.seed(RANDOM_SEED)


def load_data():
    print(">>> Loading data...")
    features = ['L1', 'h0', 'c/R', 'CKB', 'rho', 'fcu', 't_bot', 'fy_bot', 'fu_bot', 't_top', 'fy_top', 'fu_top',
                'stud_space', 'stud_D', 'stud_height']
    target = 'Vu'

    file_path = os.path.join(DATA_PATH, 'single_plate_slabs_fanlin.xlsx')

    df = pd.read_excel(file_path)
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df.dropna(subset=[target], inplace=True)

    X = df[features]
    y = df[target]
    return X, y, df


def train_xgb(X_train, y_train, X_val, y_val, scaler=None):
    """
    Trains the XGBoost model.
    If scaler is provided, it uses the scaled true values for the evaluation metric
    and provides the metrics for the scaled data (MSE_scaled, RMSE_scaled).
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = XGB_PARAMS.copy()
    num_boost_round = params.pop('n_estimators')
    params['eval_metric'] = 'rmse'
    if 'random_state' in params:
        params['seed'] = params.pop('random_state')

    watchlist = [(dtrain, 'train'), (dval, 'eval')]

    try:
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False
        )
    except TypeError:
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS)],
            verbose_eval=False
        )

    try:
        best_iteration = model.best_iteration
        preds = model.predict(dval, iteration_range=(0, best_iteration + 1))
    except (AttributeError, TypeError):
        try:
            preds = model.predict(dval, ntree_limit=model.best_ntree_limit)
        except:
            preds = model.predict(dval)

    mse = mean_squared_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    mse_scaled = None
    rmse_scaled = None
    if scaler is not None and scaler.mean_ is not None and scaler.scale_ is not None and scaler.mean_.shape[0] == 1:

        pass

    return preds, mse, r2, model, mse_scaled, rmse_scaled


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- Running XGBoost ---")
    X, y, df_full = load_data()
    X_np = X.values
    y_np = y.values

    if USE_K_FOLD:
        print(f"Mode: K-Fold CV (K={K_FOLDS})")
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        results = []
        all_preds = []
        all_mse_scaled = []
        all_rmse_scaled = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
            print(f"Processing Fold {fold + 1}...")

            X_tr, X_va = X_np[train_idx], X_np[val_idx]
            y_tr, y_va = y_np[train_idx], y_np[val_idx]

            # Scaler for features
            scaler_X = StandardScaler()
            X_tr_s = scaler_X.fit_transform(X_tr)
            X_va_s = scaler_X.transform(X_va)

            scaler_y = StandardScaler()
            y_tr_s = scaler_y.fit_transform(y_tr.reshape(-1, 1)).flatten()

            preds, mse, r2, _, _, _ = train_xgb(X_tr_s, y_tr, X_va_s, y_va)

            scale_y = scaler_y.scale_[0]
            mse_scaled = mse / (scale_y ** 2)
            rmse_scaled = np.sqrt(mse_scaled)

            print(f"  Fold {fold + 1} Result -> R2: {r2:.2f}, MSE: {mse:.2f}")
            print(f"  MSE (Standardized): {mse_scaled:.4f}, RMSE (Standardized): {rmse_scaled:.4f}")

            results.append({'Fold': fold + 1, 'R2': r2, 'MSE': mse, 'RMSE': np.sqrt(mse)})
            all_mse_scaled.append(mse_scaled)
            all_rmse_scaled.append(rmse_scaled)

            fold_df = pd.DataFrame({
                'Fold': fold + 1,
                'True_Value': y_va,
                'Predicted_Value': preds
            })
            all_preds.append(fold_df)

        df_res = pd.DataFrame(results)

        # Add the scaled metrics to the results DataFrame
        df_res['MSE_Scaled'] = all_mse_scaled
        df_res['RMSE_Scaled'] = all_rmse_scaled

        df_res.to_excel(os.path.join(OUTPUT_DIR, 'xgboost_kfold_metrics.xlsx'), index=False)

        df_preds = pd.concat(all_preds, ignore_index=True)
        df_preds.to_excel(os.path.join(OUTPUT_DIR, 'xgboost_kfold_predictions.xlsx'), index=False)

        print(f"\nAverage R2: {df_res['R2'].mean():.2f}")
        print(f"Average MSE (Standardized): {df_res['MSE_Scaled'].mean():.4f}")
        print(f"Average RMSE (Standardized): {df_res['RMSE_Scaled'].mean():.4f}")
        print(f"Results saved to {OUTPUT_DIR}")


    else:
        if len(X_np) <= TRAIN_SET_SIZE:
            raise ValueError(f"Total data ({len(X_np)}) is too small for requested train size ({TRAIN_SET_SIZE})")
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_np, y_np,
            train_size=TRAIN_SET_SIZE,
            random_state=RANDOM_SEED

        )
        print(f"Train samples: {len(X_tr)}, Validation samples: {len(X_va)}")

        # Scaler for features
        scaler_X = StandardScaler()
        X_tr_s = scaler_X.fit_transform(X_tr)
        X_va_s = scaler_X.transform(X_va)
        joblib.dump(scaler_X, os.path.join(OUTPUT_DIR, 'xgboost_std_scaler_X.joblib'))

        # Scaler for target variable (needed for metrics on standardized data)
        scaler_y = StandardScaler()
        y_tr_s = scaler_y.fit_transform(y_tr.reshape(-1, 1)).flatten()
        joblib.dump(scaler_y, os.path.join(OUTPUT_DIR, 'xgboost_std_scaler_y.joblib'))

        preds, mse, r2, model, _, _ = train_xgb(X_tr_s, y_tr, X_va_s, y_va)

        # Calculate MSE and RMSE on standardized data
        scale_y = scaler_y.scale_[0]
        mse_scaled = mse / (scale_y ** 2)
        rmse_scaled = np.sqrt(mse_scaled)

        print(f"Result -> R2: {r2:.4f}, MSE: {mse:.4f}")
        print(f"MSE (Standardized): {mse_scaled:.4f}, RMSE (Standardized): {rmse_scaled:.4f}")

        pd.DataFrame([{
            'R2': r2,
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MSE_Scaled': mse_scaled,
            'RMSE_Scaled': rmse_scaled
        }]).to_excel(os.path.join(OUTPUT_DIR, 'xgboost_std_metrics.xlsx'), index=False)

        pd.DataFrame({
            'True_Value': y_va,
            'Predicted_Value': preds
        }).to_excel(os.path.join(OUTPUT_DIR, 'xgboost_std_predictions.xlsx'), index=False)

    print("--- Done ---")