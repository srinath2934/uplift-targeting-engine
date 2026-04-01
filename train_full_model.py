"""
AI Conversion Uplift Modeling - Production Training Pipeline
This script processes the full 13,979,592 row Criteo dataset and trains a T-Learner AI Engine.
"""
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import os

def main():
    print("🚀 STARTING PRODUCTION TRAINING PIPELINE (13,979,592 ROWS)")
    start_time = time.time()
    
    data_path = "data/criteo-uplift-v2.1.csv"
    
    # 1. Memory-Efficient Loading (Portfolio Flex: Memory Management)
    print(f"\n[1/5] Loading 3.2GB Dataset from {data_path} safely...")
    # We force Pandas to use tiny memory blocks (float32 and int8) instead of giant default blocks (float64).
    # This prevents the computer from crashing when loading 13.9 million rows.
    dtypes = {f'f{i}': 'float32' for i in range(12)}
    dtypes['treatment'] = 'int8'
    dtypes['conversion'] = 'int8'
    
    # Load ONLY the columns we need, ignoring useless tracking IDs
    cols_to_load = list(dtypes.keys())
    df = pd.read_csv(data_path, usecols=cols_to_load, dtype=dtypes)
    print(f"      Successfully loaded {len(df):,} rows into RAM perfectly!")
    
    # 2. Splitting the User Timelines (Treatment vs Control)
    print("\n[2/5] Sorting 13,979,592 users into Two-Brain Timelines...")
    mask_treat = df['treatment'] == 1
    
    # The Treatment World (Ads Shown)
    X_treat = df[mask_treat].drop(columns=['treatment', 'conversion'])
    y_treat = df[mask_treat]['conversion']
    
    # The Control World (No Ads)
    X_ctrl = df[~mask_treat].drop(columns=['treatment', 'conversion'])
    y_ctrl = df[~mask_treat]['conversion']
    
    # Free up system memory instantly! (Another Portfolio Flex)
    del df
    
    print(f"      Brain 1 (Ads): {len(X_treat):,} rows ready.")
    print(f"      Brain 2 (No Ads): {len(X_ctrl):,} rows ready.")
    
    # 3. Model Architecture Construction
    print("\n[3/5] Initializing XGBoost High-Performance Algorithms...")
    params = {
        'n_estimators': 50,      # Optimized via RandomizedSearchCV
        'learning_rate': 0.05,   # Optimized via RandomizedSearchCV
        'max_depth': 3,          # Optimized via RandomizedSearchCV
        'subsample': 0.7,        # Optimized via RandomizedSearchCV
        'colsample_bytree': 1.0, # Optimized via RandomizedSearchCV
        'tree_method': 'hist',   # Forces XGBoost to compress data perfectly for massive datasets
        'device': 'cuda',        # Attempts to use Nvidia GPU if available
        'n_jobs': -1             # Uses 100% of your computer's CPU cores
    }
    
    model_treat = xgb.XGBClassifier(**params)
    model_ctrl = xgb.XGBClassifier(**params)
    
    # 4. Training (The heavy mathematical lifting)
    print("\n[4/5] Training Brain 1 (Treatment). Please wait. This may take a few minutes...")
    model_treat.fit(X_treat, y_treat)
    
    print("      Training Brain 2 (Control). Please wait...")
    model_ctrl.fit(X_ctrl, y_ctrl)
    
    # 5. Saving to Disk
    print("\n[5/5] Pickling (Saving) AI Brains permanently to Disk...")
    os.makedirs('models', exist_ok=True)
    
    model_treat.save_model("models/treatment_model.xgb")
    model_ctrl.save_model("models/control_model.xgb")
    
    end_time = time.time()
    minutes = (end_time - start_time) / 60
    print(f"\n✅ SUCCESS! Models built and saved permanently. Total Time: {minutes:.2f} minutes.")

if __name__ == "__main__":
    main()
