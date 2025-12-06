# -*- coding: utf-8 -*-
"""
Script Function: MAML-based Few-Shot Prediction.
"""

# -----------------------------------
# Import Libraries
# -----------------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
import os
from copy import deepcopy
import joblib
import shap
import matplotlib.pyplot as plt

# =======================================================================
# 1. Hyperparameters & Global Settings
# =======================================================================
# --- Main Switches ---
INCLUDE_DOUBLE_PLATE_DATA = 1 # 1: Include double-plate data, 0: Exclude
FIRST_ORDER = False  # False: Second Order MAML
USE_SCHEDULER = True
# --- SHAP Settings ---
RUN_IMPORTANCE_ANALYSIS = 1
RUN_SHAP = 1
SHAP_NSAMPLES = 100
SHAP_BACKGROUND_SIZE = 42
SHAP_DEPENDENCE_TOP_K = 3
JITTER_SCALE = 0.1  # Jitter magnitude for plotting

# --- MAML Meta-Training Parameters ---
META_LR = 0.001
INNER_LR = 0.001
META_EPOCHS = 1000
NUM_TASKS_PER_EPOCH = 10
DOUBLE_PLATE_TASK_RATIO = 0.2
NUM_INNER_UPDATES = 4
GRAD_CLIP_NORM = 1.0

# --- Scheduler Parameters ---
SCHEDULER_STEP_SIZE = 100
SCHEDULER_GAMMA = 0.9

# --- Task Data Parameters (K-shot) ---
K_SHOT = 8
Q_QUERY = 2

# --- Fine-tuning Parameters ---
FINETUNE_EPOCHS = 800
N_SPLITS = 5

# --- File & Path Settings ---
DATA_PATH = '../data'
OUTPUT_DIR = '../results'
USE_DYNAMIC_FILENAME = True
RANDOM_SEED = 50

# Apply Random Seed
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =======================================================================
# 2. Data Loading and Preprocessing
# =======================================================================
def load_and_preprocess_data(file_prefix=""):
    print(">>> Loading and preprocessing data...")
    features = ['L1', 'h0', 'c/R', 'CKB', 'rho', 'fcu', 't_bot', 'fy_bot', 'fu_bot', 't_top', 'fy_top', 'fu_top',
                'stud_space', 'stud_D', 'stud_height']
    target = 'Vu'

    # --- Load Pure Concrete Data ---
    concrete_path = os.path.join(DATA_PATH, 'pure_concrete_slabs.xlsx')
    if not os.path.exists(concrete_path):
        print(f"Error: File not found at {concrete_path}")
        return None, None, None, None, None, None, None

    df_concrete = pd.read_excel(concrete_path)
    df_concrete[target] = pd.to_numeric(df_concrete[target], errors='coerce')
    df_concrete.dropna(subset=[target], inplace=True)

    # --- Load Single Plate Data (Target Task) ---
    single_plate_path = os.path.join(DATA_PATH, 'single_plate_slabs.xlsx')
    if not os.path.exists(single_plate_path):
        print(f"Error: File not found at {single_plate_path}")
        return None, None, None, None, None, None, None

    df_meta_test = pd.read_excel(single_plate_path)
    df_meta_test[target] = pd.to_numeric(df_meta_test[target], errors='coerce')
    df_meta_test.dropna(subset=[target], inplace=True)

    # --- Load Double Plate Data (Optional) ---
    df_double_plate = None
    all_dfs_for_scaling = [df_concrete, df_meta_test]

    double_plate_file = os.path.join(DATA_PATH, 'double_plate_slabs.xlsx')
    if INCLUDE_DOUBLE_PLATE_DATA == 1:
        if os.path.exists(double_plate_file):
            df_double_plate = pd.read_excel(double_plate_file)
            df_double_plate[target] = pd.to_numeric(df_double_plate[target], errors='coerce')
            df_double_plate.dropna(subset=[target], inplace=True)
            all_dfs_for_scaling.append(df_double_plate)
        else:
            print(f"Warning: Double plate data enabled but file not found.")

    df_all_for_scaling = pd.concat(all_dfs_for_scaling, ignore_index=True)

    # --- Standardization ---
    scaler_x = StandardScaler()
    scaler_x.fit(df_all_for_scaling[features])

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    joblib.dump(scaler_x, os.path.join(OUTPUT_DIR, f'{file_prefix}scaler_x.joblib'))

    scaler_y = StandardScaler()
    scaler_y.fit(df_all_for_scaling[[target]])
    joblib.dump(scaler_y, os.path.join(OUTPUT_DIR, f'{file_prefix}scaler_y.joblib'))

    # --- Transform Data to Tensors ---
    df_concrete[features] = scaler_x.transform(df_concrete[features])
    df_concrete[target] = scaler_y.transform(df_concrete[[target]])
    X_concrete = torch.tensor(df_concrete[features].values, dtype=torch.float32)
    y_concrete = torch.tensor(df_concrete[target].values, dtype=torch.float32).view(-1, 1)

    df_meta_test[features] = scaler_x.transform(df_meta_test[features])
    df_meta_test[target] = scaler_y.transform(df_meta_test[[target]])
    X_meta_test = torch.tensor(df_meta_test[features].values, dtype=torch.float32)
    y_meta_test = torch.tensor(df_meta_test[target].values, dtype=torch.float32).view(-1, 1)

    X_double, y_double = None, None
    if df_double_plate is not None:
        df_double_plate[features] = scaler_x.transform(df_double_plate[features])
        df_double_plate[target] = scaler_y.transform(df_double_plate[[target]])
        X_double = torch.tensor(df_double_plate[features].values, dtype=torch.float32)
        y_double = torch.tensor(df_double_plate[target].values, dtype=torch.float32).view(-1, 1)
        print(
            f">>> Data Ready. Meta-Train (Pure): {len(X_concrete)}, Meta-Train (Double): {len(X_double)}, Target: {len(X_meta_test)}")
    else:
        print(f">>> Data Ready. Meta-Train: {len(X_concrete)}, Target: {len(X_meta_test)}")

    return X_concrete, y_concrete, X_double, y_double, X_meta_test, y_meta_test, scaler_y


# =======================================================================
# 3. Model Definition (BPNN)
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
# 4. MAML Training
# =======================================================================
def sample_tasks(X, y, k_shot, q_query, num_tasks):
    if num_tasks == 0:
        return []
    num_total = X.shape[0]
    sample_size = k_shot + q_query
    if num_total < sample_size:
        return []

    tasks = []
    for _ in range(num_tasks):
        indices = np.random.choice(num_total, size=sample_size, replace=False)
        task_data = {
            'support_x': X[indices[:k_shot]], 'support_y': y[indices[:k_shot]],
            'query_x': X[indices[k_shot:]], 'query_y': y[indices[k_shot:]]
        }
        tasks.append(task_data)
    return tasks


def maml_training(model, X_concrete, y_concrete, X_double, y_double):
    print("\n--- Starting Meta-Training ---")
    meta_optimizer = optim.Adam(model.parameters(), lr=META_LR)
    if USE_SCHEDULER:
        scheduler = StepLR(meta_optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    loss_fn = nn.MSELoss()
    meta_losses = []
    meta_r2s = []
    create_graph = not FIRST_ORDER

    for epoch in range(META_EPOCHS):
        # Task Sampling
        if X_double is not None:
            n_double = int(NUM_TASKS_PER_EPOCH * DOUBLE_PLATE_TASK_RATIO)
            n_conc = NUM_TASKS_PER_EPOCH - n_double
            tasks = sample_tasks(X_concrete, y_concrete, K_SHOT, Q_QUERY, n_conc) + \
                    sample_tasks(X_double, y_double, K_SHOT, Q_QUERY, n_double)
            np.random.shuffle(tasks)
        else:
            tasks = sample_tasks(X_concrete, y_concrete, K_SHOT, Q_QUERY, NUM_TASKS_PER_EPOCH)

        if not tasks: continue

        total_meta_loss = 0.0
        total_meta_r2 = 0.0

        for task in tasks:
            fast_model = deepcopy(model)
            # Inner Loop
            for _ in range(NUM_INNER_UPDATES):
                loss = loss_fn(fast_model(task['support_x']), task['support_y'])
                grads = torch.autograd.grad(loss, fast_model.parameters(), create_graph=create_graph)
                with torch.no_grad():
                    for p, g in zip(fast_model.parameters(), grads):
                        p -= INNER_LR * g

            # Outer Loop Loss Calculation
            query_pred = fast_model(task['query_x'])
            total_meta_loss += loss_fn(query_pred, task['query_y'])
            with torch.no_grad():
                if len(task['query_y']) > 1:
                    total_meta_r2 += r2_score(task['query_y'].numpy(), query_pred.detach().numpy())

        # Meta Update
        meta_optimizer.zero_grad()
        avg_loss = total_meta_loss / len(tasks)
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        meta_optimizer.step()

        if USE_SCHEDULER: scheduler.step()

        meta_losses.append(avg_loss.item())
        meta_r2s.append(total_meta_r2 / len(tasks))

        if (epoch + 1) % 100 == 0:
            print(f"Meta-Training Epoch {epoch + 1}/{META_EPOCHS} | Loss: {avg_loss.item():.2f}")

    print("--- Meta-Training Completed ---")
    return model, meta_losses, meta_r2s


# =======================================================================
# 5. Fine-tuning and Evaluation (K-Fold)
# =======================================================================
def finetune_and_evaluate(meta_model, X_test, y_test, y_scaler):
    print(f"\n--- Starting Fine-tuning ({N_SPLITS}-Fold CV) ---")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    all_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_test)):
        print(f"--- Processing Fold {fold + 1}/{N_SPLITS} ---")
        X_supp, y_supp = X_test[train_idx], y_test[train_idx]
        X_query, y_query = X_test[test_idx], y_test[test_idx]

        model = deepcopy(meta_model)
        opt = optim.Adam(model.parameters(), lr=INNER_LR)
        loss_fn = nn.MSELoss()

        val_loss_log = []
        val_r2_log = []

        for epoch in range(FINETUNE_EPOCHS):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(X_supp), y_supp)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                preds = model(X_query)
                v_loss = loss_fn(preds, y_query).item()
                v_r2 = r2_score(y_query.numpy(), preds.numpy())
                val_loss_log.append(v_loss)
                val_r2_log.append(v_r2)

            if (epoch + 1) % 100 == 0:
                print(f"    Fold {fold + 1} Epoch {epoch + 1} | Val Loss: {v_loss:.2f} | R2: {v_r2:.2f}")

        # Final predictions
        with torch.no_grad():
            final_preds = model(X_query)

        preds_un = y_scaler.inverse_transform(final_preds.numpy())
        true_un = y_scaler.inverse_transform(y_query.numpy())

        all_results.append({
            'fold_num': fold + 1,
            'val_r2_log': val_r2_log,
            'val_loss_log': val_loss_log,
            'final_val_r2': val_r2_log[-1],
            'final_val_loss': val_loss_log[-1],
            'predictions_unscaled': preds_un,
            'true_values_unscaled': true_un
        })

    return all_results


# =======================================================================
# 6. SHAP / Feature Importance Analysis
# =======================================================================
def analyze_feature_importance(meta_trained_model, X_meta_test, y_meta_test, file_prefix="", y_scaler=None):
    print("\n" + "=" * 60)
    print("--- Running Feature Importance Analysis ---")
    print("=" * 60)
    features_list = ['L1', 'h0', 'c/R', 'CKB', 'rho', 'fcu', 't_bot', 'fy_bot', 'fu_bot', 't_top', 'fy_top', 'fu_top',
                     'stud_space', 'stud_D', 'stud_height']

    # 1. Fine-tune model on ALL target data for global explanation
    final_model = deepcopy(meta_trained_model)
    opt = optim.Adam(final_model.parameters(), lr=INNER_LR)
    loss_fn = nn.MSELoss()

    for _ in range(FINETUNE_EPOCHS):
        final_model.train()
        opt.zero_grad()
        loss_fn(final_model(X_meta_test), y_meta_test).backward()
        opt.step()
    final_model.eval()

    if not RUN_SHAP or shap is None:
        print(">>> SHAP skipped.")
        return

    print(">>> Calculating SHAP values...")
    try:
        # Prepare Data
        X_np = X_meta_test.numpy()
        bg = X_np[:min(SHAP_BACKGROUND_SIZE, len(X_np))]

        def predict_fn(x):
            xt = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                return final_model(xt).numpy().reshape(-1)

        explainer = shap.KernelExplainer(predict_fn, bg)
        shap_vals = explainer.shap_values(X_np, nsamples=SHAP_NSAMPLES)

        shap_arr = np.array(shap_vals)
        if shap_arr.ndim == 3: shap_arr = shap_arr[0]

        # Get predictions for table
        preds_sc = predict_fn(X_np)
        true_sc = y_meta_test.numpy().reshape(-1)

        if y_scaler:
            preds_un = y_scaler.inverse_transform(preds_sc.reshape(-1, 1)).flatten()
            true_un = y_scaler.inverse_transform(true_sc.reshape(-1, 1)).flatten()
        else:
            preds_un, true_un = preds_sc, true_sc

        # Build Long Format Table
        rows = []
        # Normalized features [0,1] for visualization
        f_min, f_max = X_np.min(0), X_np.max(0)
        denom = f_max - f_min
        denom[denom == 0] = 1.0
        X_norm = (X_np - f_min) / denom

        for i in range(len(X_np)):
            for j, feat in enumerate(features_list):
                rows.append({
                    'sample_idx': int(i),
                    'feature': feat,
                    'feature_idx': int(j + 1),
                    'feature_value': float(X_np[i, j]),
                    'feature_value_norm': float(X_norm[i, j]),
                    'SHAP value': float(shap_arr[i, j]),
                    'prediction': float(preds_un[i]),
                    'true': float(true_un[i])
                })

        df_long = pd.DataFrame(rows)

        # Calculate Rank and Stats
        mean_abs = df_long.groupby('feature')['SHAP value'].apply(lambda x: np.mean(np.abs(x))).reset_index()
        mean_abs.columns = ['feature', 'mean_abs_shap']
        mean_abs = mean_abs.sort_values('mean_abs_shap', ascending=False)
        mean_abs['rank'] = range(1, len(mean_abs) + 1)

        # Merge Stats
        df_long = df_long.merge(mean_abs, on='feature', how='left')

        # Calculate Y_idx and Jitter
        max_rank = int(df_long['rank'].max())
        df_long['Y_idx'] = (max_rank - df_long['rank']).astype(float)

        rng = np.random.default_rng(seed=RANDOM_SEED)
        df_long['Y_jitter'] = df_long['Y_idx'] + (rng.random(size=len(df_long)) - 0.5) * JITTER_SCALE

        # Strict Column Ordering
        cols = ['sample_idx', 'feature', 'feature_idx', 'feature_value', 'feature_value_norm',
                'SHAP value', 'prediction', 'true', 'mean_abs_shap', 'rank', 'Y_idx', 'Y_jitter']
        df_long = df_long[cols]

        # Save
        path_long = os.path.join(OUTPUT_DIR, f'{file_prefix}SHAP_long_for_Origin_all.xlsx')
        df_long.to_excel(path_long, index=False)

        path_rank = os.path.join(OUTPUT_DIR, f'{file_prefix}SHAP_Global—_importance.xlsx')
        mean_abs.rename(columns={'feature': 'Feature', 'rank': 'Rank', 'mean_abs_shap': 'MeanAbsSHAP'}).to_excel(
            path_rank, index=False)

        print(f"-> SHAP Data Saved: {path_long}")

        # Plot
        try:
            plt.figure()
            shap.summary_plot(shap_arr, X_np, feature_names=features_list, show=False)
            plt.savefig(os.path.join(OUTPUT_DIR, f'{file_prefix}SHAP_summary.png'), bbox_inches='tight')
            plt.close()
        except:
            pass

    except Exception as e:
        print(f"SHAP Error: {e}")


# =======================================================================
# 7. Main Execution
# =======================================================================
if __name__ == '__main__':
    # Construct filename prefix based on settings
    file_prefix = ""
    if USE_DYNAMIC_FILENAME:
        db_str = "WithDoublePlate" if INCLUDE_DOUBLE_PLATE_DATA == 1 else "NoDoublePlate"
        file_prefix = f"{db_str}_"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Data
    X_conc, y_conc, X_db, y_db, X_test, y_test, y_scaler = load_and_preprocess_data(file_prefix)

    if X_conc is not None:
        # Initialize
        base_model = BPNN(X_conc.shape[1])

        # Meta-Training
        trained_meta, m_loss, m_r2 = maml_training(base_model, X_conc, y_conc, X_db, y_db)

        # Save Meta-Train Logs
        pd.DataFrame({'epoch': range(1, len(m_loss) + 1), 'loss': m_loss}).to_excel(
            os.path.join(OUTPUT_DIR, f'{file_prefix}meta_train_loss.xlsx'), index=False)

        # Fine-Tuning
        ft_results = finetune_and_evaluate(trained_meta, X_test, y_test, y_scaler)

        # Save Fine-Tune Results
        if ft_results:
            # Summary
            final_r2s = [r['final_val_r2'] for r in ft_results]
            final_losses = [r['final_val_loss'] for r in ft_results]

            pd.DataFrame({
                'Fold': [r['fold_num'] for r in ft_results],
                'Final_Loss': final_losses,
                'Final_R2': final_r2s
            }).to_excel(os.path.join(OUTPUT_DIR, f'{file_prefix}kfold_summary.xlsx'), index=False)

            print(f"Mean R2: {np.mean(final_r2s):.2f}")

            # Logs per fold
            for r in ft_results:
                f_num = r['fold_num']
                pd.DataFrame({
                    'epoch': range(1, len(r['val_loss_log']) + 1),
                    'val_loss': r['val_loss_log'],
                    'val_r2': r['val_r2_log']
                }).to_excel(os.path.join(OUTPUT_DIR, f'{file_prefix}finetune_log_fold_{f_num}.xlsx'), index=False)

                pd.DataFrame({
                    'predicted': r['predictions_unscaled'].flatten(),
                    'true': r['true_values_unscaled'].flatten()
                }).to_excel(os.path.join(OUTPUT_DIR, f'{file_prefix}predictions_fold_{f_num}.xlsx'), index=False)

            # Combined predictions
            all_preds = np.concatenate([r['predictions_unscaled'] for r in ft_results])
            all_true = np.concatenate([r['true_values_unscaled'] for r in ft_results])
            pd.DataFrame({'predicted': all_preds.flatten(), 'true': all_true.flatten()}).to_excel(
                os.path.join(OUTPUT_DIR, f'{file_prefix}predictions_all_folds.xlsx'), index=False)

        # SHAP Analysis
        if RUN_IMPORTANCE_ANALYSIS:
            analyze_feature_importance(trained_meta, X_test, y_test, file_prefix=file_prefix, y_scaler=y_scaler)

        print("\n--- Process Completed Successfully ---")