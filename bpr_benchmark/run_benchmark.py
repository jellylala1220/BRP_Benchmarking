import pandas as pd
import numpy as np
import os
from pathlib import Path
from bpr_benchmark.data.loader import TrafficDataLoader
from bpr_benchmark.utils.estimation import calculate_metrics
from bpr_benchmark.models import (
    A0_Baseline, A1_Calibrated, A2_FD_VDF,
    B1_DP_BPR, B2_Rolling_DVDF, B3_Stochastic,
    C1_PCU_BPR, C2_Yun_Truck,
    D1_Weather_Capacity, D3_Reliability_ETT,
    E1_SVR, E2_RF
)

def main():
    # 1. Setup Output Directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 2. Load Data
    print("Loading Data...")
    loader = TrafficDataLoader()
    loader.load_data()
    loader.preprocess()
    train_df, test_df = loader.get_train_test_split()
    
    # 3. Define Models
    models = [
        A0_Baseline(), A1_Calibrated(), A2_FD_VDF(),
        B1_DP_BPR(), B2_Rolling_DVDF(), B3_Stochastic(),
        C1_PCU_BPR(), C2_Yun_Truck(),
        D1_Weather_Capacity(), D3_Reliability_ETT(),
        E1_SVR(), E2_RF()
    ]
    
    results = []
    
    # 4. Run Benchmark
    print("\nStarting Benchmark...")
    print(f"{'Model':<20} | {'Status':<10} | {'Train MAE':<10} | {'Test MAE':<10} | {'Test R2':<8}")
    print("-" * 80)
    
    # CRITICAL: Pre-train A1 for Cluster E residual models
    print("\n[Pre-training A1 baseline for Cluster E...]")
    A1 = A1_Calibrated()
    A1.fit(train_df)
    T_A1_train = A1.predict(train_df)
    T_A1_test = A1.predict(test_df)
    print(f"[A1 baseline ready: Train MAE = {np.mean(np.abs(train_df['T_obs'].values - T_A1_train)):.2f}]\n")
    
    for model in models:
        try:
            # Check if this is aCLUSTER E model (requires A1 predictions)
            is_cluster_e = model.name.startswith('E')
            
            # Train
            if is_cluster_e:
                model.fit(train_df, T_A1_train)
            else:
                model.fit(train_df)
            
            # Evaluate on Train
            if is_cluster_e:
                y_train_pred = model.predict(train_df, T_A1_train)
            else:
                y_train_pred = model.predict(train_df)
                
            y_train_true = train_df['T_obs'].values
            train_metrics = calculate_metrics(y_train_true, y_train_pred, f"{model.name} (Train)")
            
            # Evaluate on Test
            if is_cluster_e:
                y_pred = model.predict(test_df, T_A1_test)
            else:
                y_pred = model.predict(test_df)
                
            y_true = test_df['T_obs'].values
            test_metrics = calculate_metrics(y_true, y_pred, f"{model.name} (Test)")
            
            # Log
            print(f"{model.name:<20} | {'Done':<10} | {train_metrics['MAE']:.2f}       | {test_metrics['MAE']:.2f}       | {test_metrics['R2']:.2f}")
            
            # Store result
            res_entry = {
                "Cluster": model.name[0],
                "Model": model.name,
                "Params": str(model.get_params()),
                # Train Metrics
                "Train_RMSE": train_metrics['RMSE'],
                "Train_MAE": train_metrics['MAE'],
                "Train_MAPE": train_metrics['MAPE'],
                "Train_R2": train_metrics['R2'],
                "Train_P95": train_metrics['P95'],
                # Test Metrics
                "Test_RMSE": test_metrics['RMSE'],
                "Test_MAE": test_metrics['MAE'],
                "Test_MAPE": test_metrics['MAPE'],
                "Test_R2": test_metrics['R2'],
                "Test_P95": test_metrics['P95']
            }
            results.append(res_entry)
            
        except Exception as e:
            print(f"{model.name:<20} | {'Failed':<10} | Error: {e}")
            import traceback
            traceback.print_exc()
            
    # 5. Save Results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "benchmark_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nBenchmark complete. Results saved to {results_path}")
    
    # 6. Generate Summary Report
    summary_path = output_dir / "benchmark_report.md"
    with open(summary_path, "w") as f:
        f.write("# BPR Benchmark Report\n\n")
        f.write(f"**Date Range**: {train_df['timestamp'].min()} to {test_df['timestamp'].max()}\n")
        f.write(f"**Train Set**: {len(train_df)} records (Sept 1-23, 2024)\n")
        f.write(f"**Test Set**: {len(test_df)} records (Sept 24-30, 2024 - Last 7 days)\n\n")
        f.write("## Model Performance (Train vs Test)\n\n")
        f.write(results_df.to_string(index=False))
        
    print(f"Report generated at {summary_path}")

if __name__ == "__main__":
    main()
