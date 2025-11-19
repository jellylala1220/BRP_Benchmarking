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
    print(f"{'Model':<20} | {'Status':<10} | {'RMSE':<8} | {'MAE':<8} | {'R2':<8}")
    print("-" * 70)
    
    for model in models:
        try:
            # Train
            model.fit(train_df)
            
            # Predict
            y_pred = model.predict(test_df)
            y_true = test_df['T_obs'].values
            
            # Metrics
            metrics = calculate_metrics(y_true, y_pred, model.name)
            
            # Log
            print(f"{model.name:<20} | {'Done':<10} | {metrics['RMSE']:.2f}     | {metrics['MAE']:.2f}     | {metrics['R2']:.2f}")
            
            # Store result
            res_entry = {
                "Cluster": model.name[0], # First letter A, B, C...
                "Model": model.name,
                "Params": str(model.get_params()),
                **metrics
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
        f.write(f"**Train Size**: {len(train_df)}, **Test Size**: {len(test_df)}\n\n")
        f.write("## Model Performance\n\n")
        f.write(results_df.to_string(index=False))
        
    print(f"Report generated at {summary_path}")

if __name__ == "__main__":
    main()
