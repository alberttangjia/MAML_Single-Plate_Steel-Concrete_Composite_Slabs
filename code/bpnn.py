# -*- coding: utf-8 -*-
"""
Script Function: BPNN for Prediction.
"""

# -----------------------------------
# Import Libraries
# -----------------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os
import joblib
import glob
import matplotlib.pyplot as plt
import shap

# =======================================================================
# 1. Hyperparameters & Global Settings
# =======================================================================
# --- Cross Validation ---
USE_K_FOLD = False
K_FOLDS = 5

# --- Training ---
LEARNING_RATE = 0.001
EPOCHS = 1000
TRAIN_SET_SIZE = 36

# --- Paths ---
DATA_PATH = '../data'
INPUT_FILE = '../data/pure_concrete_slabs.xlsx'
OUTPUT_DIR = '../results'

# --- Reproducibility ---
RANDOM_SEED = 50
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- SHAP Settings ---
RUN_SHAP = True
SHAP_NSAMPLES = 100
SHAP_BACKGROUND_SIZE = 30
SHAP_DEPENDENCE_TOP_K = 3
JITTER_SCALE = 0.05  # Jitter magnitude for plotting


# =======================================================================
# 2. Data Loading
# =======================================================================
def load_clean_data():
    print(">>> Loading and cleaning dataset...")
    features = ['L1', 'h0', 'CKB', 'c/R', 'rho', 'fcu', 't_bot', 'fy_bot', 'fu_bot', 't_top', 'fy_top', 'fu_top',
                'stud_space', 'stud_D', 'stud_height']
    target = 'Vu'

    file_path = os.path.join(DATA_PATH, INPUT_FILE)
    if not os.path.exists(file_path):
        # Fallback for local testing
        if os.path.exists(INPUT_FILE):
            file_path = INPUT_FILE
        else:
            print(f"Error: Data file not found at {file_path}")
            return None, None, None, None

    df = pd.read_excel(file_path)
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df.dropna(subset=[target], inplace=True)

    X = df[features]
    y = df[[target]]
    return X, y, df, features


def load_and_preprocess_data_for_single_split():
    X, y, _, features = load_clean_data()

    if len(X) <= TRAIN_SET_SIZE:
        raise ValueError(f"Error: Not enough samples ({len(X)}) for requested train size ({TRAIN_SET_SIZE}).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SET_SIZE, random_state=RANDOM_SEED
    )

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    joblib.dump(scaler_x, os.path.join(OUTPUT_DIR, 'scaler_x.joblib'))
    joblib.dump(scaler_y, os.path.join(OUTPUT_DIR, 'scaler_y.joblib'))

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, len(features), scaler_y, features


# =======================================================================
# 3. Model Definition
# =======================================================================
class BPNN(nn.Module):
    def __init__(self, input_dim):
        super(BPNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


# =======================================================================
# 4. Training Function
# =======================================================================
def train_and_evaluate_model(X_train, y_train, X_val, y_val, input_dim, scaler_y):
    model = BPNN(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    train_loss_log, train_r2_log = [], []
    val_loss_log, val_r2_log = [], []

    for epoch in range(EPOCHS):
        model.train()
        train_preds = model(X_train)
        loss = loss_fn(train_preds, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss_log.append(loss.item())
            try:
                t_r2 = r2_score(y_train.numpy(), train_preds.numpy())
            except:
                t_r2 = 0
            train_r2_log.append(t_r2)

            val_preds = model(X_val)
            v_loss = loss_fn(val_preds, y_val)
            val_loss_log.append(v_loss.item())
            try:
                v_r2 = r2_score(y_val.numpy(), val_preds.numpy())
            except:
                v_r2 = 0
            val_r2_log.append(v_r2)

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch + 1}/{EPOCHS} | Val Loss: {v_loss.item():.4f} | Val R2: {v_r2:.4f}")

    # Final predictions (Inverse Transform)
    with torch.no_grad():
        final_preds = model(X_val).numpy()
        try:
            preds_unscaled = scaler_y.inverse_transform(final_preds)
            y_val_unscaled = scaler_y.inverse_transform(y_val.numpy())
        except:
            preds_unscaled = final_preds
            y_val_unscaled = y_val.numpy()

    return train_loss_log, train_r2_log, val_loss_log, val_r2_log, preds_unscaled, y_val_unscaled, model


# =======================================================================
# 5. SHAP Analysis (Single Fold/Run)
# =======================================================================
def run_shap_analysis(model, X_np, y_np, scaler_y, features_list, file_prefix=""):
    if shap is None: return None

    X_np = np.array(X_np)
    n_samples = X_np.shape[0]

    # 1. Prepare Explainer
    bg = X_np[:min(SHAP_BACKGROUND_SIZE, n_samples)]

    def predict_fn(x):
        xt = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return model(xt).numpy().reshape(-1)

    try:
        explainer = shap.KernelExplainer(predict_fn, bg)
        shap_values = explainer.shap_values(X_np, nsamples=SHAP_NSAMPLES)
    except Exception as e:
        print(f"    SHAP Error: {e}")
        return None

    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 3: shap_arr = shap_arr[0]

    # 2. Predictions for saving
    preds_scaled = predict_fn(X_np)
    y_true_scaled = np.array(y_np).reshape(-1)

    if scaler_y:
        try:
            preds_saved = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            trues_saved = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        except:
            preds_saved = preds_scaled
            trues_saved = y_true_scaled
    else:
        preds_saved = preds_scaled
        trues_saved = y_true_scaled

    # 3. Create Long-Format Data
    rows = []
    feat_mins = X_np.min(axis=0)
    feat_maxs = X_np.max(axis=0)
    denom = feat_maxs - feat_mins
    denom[denom == 0] = 1.0
    X_norm = (X_np - feat_mins) / denom

    for i in range(n_samples):
        for j, feat in enumerate(features_list):
            rows.append({
                'sample_idx': int(i),
                'feature': feat,
                'feature_idx': int(j + 1),
                'feature_value': float(X_np[i, j]),
                'feature_value_norm': float(X_norm[i, j]),
                'SHAP value': float(shap_arr[i, j]),
                'prediction': float(preds_saved[i]),
                'true': float(trues_saved[i])
            })

    df_long = pd.DataFrame(rows)

    # 4. Calculate Stats (Rank, Y_idx, Y_jitter)
    mean_abs = df_long.groupby('feature')['SHAP value'].apply(lambda x: np.mean(np.abs(x))).reset_index()
    mean_abs.columns = ['feature', 'mean_abs_shap']
    mean_abs = mean_abs.sort_values('mean_abs_shap', ascending=False)
    mean_abs['rank'] = range(1, len(mean_abs) + 1)

    # Merge stats back to long table
    df_long = df_long.merge(mean_abs, on='feature', how='left')

    # Calculate Y_idx and Y_jitter
    max_rank = int(df_long['rank'].max())
    df_long['Y_idx'] = (max_rank - df_long['rank']).astype(float)

    rng = np.random.default_rng(seed=RANDOM_SEED)
    df_long['Y_jitter'] = df_long['Y_idx'] + (rng.random(size=len(df_long)) - 0.5) * JITTER_SCALE

    # 5. Reorder Columns Strictly
    cols_order = ['sample_idx', 'feature', 'feature_idx', 'feature_value', 'feature_value_norm',
                  'SHAP value', 'prediction', 'true', 'mean_abs_shap', 'rank', 'Y_idx', 'Y_jitter']
    # Check if 'fold' exists (it might not in single run, but we add it if needed later)
    final_cols = [c for c in cols_order if c in df_long.columns]
    df_long = df_long[final_cols]

    # Save Files
    df_long.to_excel(os.path.join(OUTPUT_DIR, f'{file_prefix}SHAP_long_for_Origin.xlsx'), index=False)

    # Save Global Importance Table
    glob_imp = mean_abs.rename(columns={'feature': 'Feature', 'rank': 'Rank', 'mean_abs_shap': 'MeanAbsSHAP'})
    glob_imp.to_excel(os.path.join(OUTPUT_DIR, f'{file_prefix}SHAP_Global_importance.xlsx'), index=False)

    # Plot
    try:
        plt.figure()
        shap.summary_plot(shap_arr, X_np, feature_names=features_list, show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, f'{file_prefix}SHAP_summary.png'), bbox_inches='tight')
        plt.close()
    except:
        pass

    return mean_abs


# =======================================================================
# 6. Global Merge Function
# =======================================================================
def combine_shap_fold_outputs(output_dir):
    """
    Combines fold-level SHAP files and recalculates Rank/Jitter globally.
    """
    print(">>> Combining K-Fold SHAP results...")
    pattern = os.path.join(output_dir, '*fold_*SHAP_long_for_Origin.xlsx')
    files = glob.glob(pattern)

    if not files:
        print("-> No fold SHAP files found.")
        return

    # 1. Concatenate
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_excel(f))
        except:
            pass

    if not dfs: return
    df_all = pd.concat(dfs, ignore_index=True)

    # 2. Recalculate Global Rank based on ALL folds
    # Remove old stats
    cols_drop = ['rank', 'mean_abs_shap', 'Y_idx', 'Y_jitter']
    df_all = df_all.drop(columns=[c for c in cols_drop if c in df_all.columns])

    # Calc global mean |SHAP|
    global_stats = df_all.groupby('feature')['SHAP value'].apply(lambda x: np.mean(np.abs(x))).reset_index()
    global_stats.columns = ['feature', 'mean_abs_shap']
    global_stats = global_stats.sort_values('mean_abs_shap', ascending=False)
    global_stats['rank'] = range(1, len(global_stats) + 1)

    # Merge back
    df_all = df_all.merge(global_stats, on='feature', how='left')

    # 3. Calculate Y_idx and Jitter
    max_rank = int(df_all['rank'].max())
    df_all['Y_idx'] = (max_rank - df_all['rank']).astype(float)

    rng = np.random.default_rng(seed=RANDOM_SEED)
    df_all['Y_jitter'] = df_all['Y_idx'] + (rng.random(size=len(df_all)) - 0.5) * JITTER_SCALE

    # 4. Strict Column Ordering
    cols_order = ['sample_idx', 'feature', 'feature_idx', 'feature_value', 'feature_value_norm',
                  'SHAP value', 'prediction', 'true', 'mean_abs_shap', 'rank', 'Y_idx', 'Y_jitter']
    # keep 'Fold' if it exists in the merged df
    if 'Fold' in df_all.columns:
        cols_order.insert(0, 'Fold')

    final_cols = [c for c in cols_order if c in df_all.columns]
    df_all = df_all[final_cols]

    # Save
    path_long = os.path.join(output_dir, 'SHAP_long_for_Origin_all_folds.xlsx')
    df_all.to_excel(path_long, index=False)
    print(f"-> Combined Long-Format saved: {path_long}")

    path_rank = os.path.join(output_dir, 'SHAP_Global_importance_all_folds.xlsx')
    global_stats.rename(columns={'feature': 'Feature', 'rank': 'Rank', 'mean_abs_shap': 'MeanAbsSHAP'}).to_excel(
        path_rank, index=False)
    print(f"-> Global Importance saved: {path_rank}")


# =======================================================================
# 7. Main Execution
# =======================================================================
if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y, df_full, features_list = load_clean_data()
    if X is None: exit()

    if USE_K_FOLD:
        print(f"\n--- K-Fold CV (K={K_FOLDS}) ---")
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        splits_dir = os.path.join(OUTPUT_DIR, 'kfold_splits')
        os.makedirs(splits_dir, exist_ok=True)

        results = []
        all_preds = []

        X_np = X.values
        y_np = y.values

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
            print(f"\nProcessing Fold {fold + 1}...")

            # Save Splits
            df_full.iloc[train_idx].to_excel(os.path.join(splits_dir, f'fold_{fold + 1}_train.xlsx'), index=False)
            df_full.iloc[val_idx].to_excel(os.path.join(splits_dir, f'fold_{fold + 1}_val.xlsx'), index=False)

            # Prepare Data
            scaler_x = StandardScaler()
            X_tr_s = scaler_x.fit_transform(X_np[train_idx])
            X_va_s = scaler_x.transform(X_np[val_idx])

            scaler_y = StandardScaler()
            y_tr_s = scaler_y.fit_transform(y_np[train_idx])
            y_va_s = scaler_y.transform(y_np[val_idx])

            # Tensor
            X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
            y_tr_t = torch.tensor(y_tr_s, dtype=torch.float32)
            X_va_t = torch.tensor(X_va_s, dtype=torch.float32)
            y_va_t = torch.tensor(y_va_s, dtype=torch.float32)

            # Train
            t_loss, t_r2, v_loss, v_r2, preds, y_true, model = train_and_evaluate_model(
                X_tr_t, y_tr_t, X_va_t, y_va_t, len(features_list), scaler_y
            )

            results.append({'Fold': fold + 1, 'Val_Loss': v_loss[-1], 'Val_R2': v_r2[-1]})

            # Collect Predictions
            fold_df = pd.DataFrame({'Fold': fold + 1, 'True': y_true.flatten(), 'Pred': preds.flatten()})
            all_preds.append(fold_df)

            # SHAP Analysis per fold
            if RUN_SHAP:
                print(f"  Running SHAP for Fold {fold + 1}...")
                run_shap_analysis(model, X_va_s, y_va_s, scaler_y, features_list, file_prefix=f'fold_{fold + 1}_')

            # Save logs
            pd.DataFrame({'epoch': range(1, EPOCHS + 1), 'loss': v_loss, 'r2': v_r2}).to_excel(
                os.path.join(OUTPUT_DIR, f'fold_{fold + 1}_logs.xlsx'), index=False)

        # Final Summary
        pd.DataFrame(results).to_excel(os.path.join(OUTPUT_DIR, 'kfold_metrics.xlsx'), index=False)
        pd.concat(all_preds).to_excel(os.path.join(OUTPUT_DIR, 'kfold_predictions.xlsx'), index=False)

        # Combine SHAP results across folds
        if RUN_SHAP:
            combine_shap_fold_outputs(OUTPUT_DIR)

        print(f"\nAvg R2: {np.mean([r['Val_R2'] for r in results]):.4f}")

    else:
        # Standard Mode
        print("\n--- Standard Split Mode ---")
        X_tr, y_tr, X_te, y_te, dim, sy, feats = load_and_preprocess_data_for_single_split()

        t_loss, t_r2, v_loss, v_r2, preds, y_true, model = train_and_evaluate_model(
            X_tr, y_tr, X_te, y_te, dim, sy
        )

        # Save Results
        pd.DataFrame({'epoch': range(1, EPOCHS + 1), 'val_loss': v_loss, 'val_r2': v_r2}).to_excel(
            os.path.join(OUTPUT_DIR, 'training_logs.xlsx'), index=False)
        pd.DataFrame({'True': y_true.flatten(), 'Pred': preds.flatten()}).to_excel(
            os.path.join(OUTPUT_DIR, 'predictions.xlsx'), index=False)

        if RUN_SHAP:
            run_shap_analysis(model, X_te.numpy(), y_te.numpy(), sy, feats)

    print("\n--- Done ---")