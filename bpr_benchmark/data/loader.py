import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class TrafficDataLoader:
    """
    Data Loader for BPR Benchmarking Project.
    Handles loading, cleaning, feature engineering, and splitting of traffic data.
    """
    
    def __init__(self, file_path: str = "Data/115030402 Data Merged.xlsx"):
        self.file_path = file_path
        self.df = None
        self.train_df = None
        self.test_df = None
        
        # Constants
        self.T0 = 60.0  # Free-flow travel time (seconds) - Placeholder, should be derived or config
        self.CAPACITY = 6649  # Theoretical capacity (veh/h)
        self.LINK_LENGTH_M = 2713.8  # Link length in meters
        
    def load_data(self) -> pd.DataFrame:
        """
        Loads data from Excel, filters by date, and performs initial cleaning.
        """
        print(f"Loading data from {self.file_path}...")
        # Load raw data
        df_raw = pd.read_excel(self.file_path)
        
        # Standardize column names (remove leading/trailing spaces)
        df_raw.columns = [c.strip() for c in df_raw.columns]
        
        # Parse Timestamp
        # Priority: MeasurementDateAdjusted + TimePeriod15MinGroup -> MeasurementStartUTC -> Other
        if 'MeasurementDateAdjusted' in df_raw.columns and 'TimePeriod15MinGroup' in df_raw.columns:
            df_raw['date'] = pd.to_datetime(df_raw['MeasurementDateAdjusted'])
            df_raw['minutes'] = df_raw['TimePeriod15MinGroup'] * 15
            df_raw['timestamp'] = df_raw['date'] + pd.to_timedelta(df_raw['minutes'], unit='m')
        elif 'MeasurementStartUTC' in df_raw.columns:
            df_raw['timestamp'] = pd.to_datetime(df_raw['MeasurementStartUTC'])
        else:
            raise ValueError("Could not find suitable timestamp columns.")
            
        # Filter Date Range: 2024-09-01 to 2024-09-30
        start_date = pd.Timestamp("2024-09-01")
        end_date = pd.Timestamp("2024-09-30 23:59:59")
        df_filtered = df_raw[(df_raw['timestamp'] >= start_date) & (df_raw['timestamp'] <= end_date)].copy()
        
        # Sort by timestamp
        df_filtered.sort_values('timestamp', inplace=True)
        
        print(f"Data loaded and filtered: {len(df_filtered)} records.")
        self.df = df_filtered
        return self.df

    def preprocess(self) -> pd.DataFrame:
        """
        Calculates basic variables and derived features.
        """
        if self.df is None:
            self.load_data()
            
        df = self.df.copy()
        
        # --- 0.1 Basic Variables ---
        
        # 1. Travel Time (T_obs)
        # Identify Fused Travel Time column
        tt_col = next((c for c in df.columns if 'Fused Travel Time' in c), None)
        if tt_col:
            df['T_obs'] = pd.to_numeric(df[tt_col], errors='coerce')
        else:
            # Fallback: Calculate from speed if available, else raise error
            # Assuming 'AverageSpeedLane1Value' etc exist or a global AverageSpeed
            # For now, let's look for 'AverageSpeed' or similar
            raise ValueError("Fused Travel Time column not found.")
            
        # 2. Flow (V_t) - Total Volume per 15min
        # Sum all FlowLaneXCategoryYValue columns
        flow_cols = [c for c in df.columns if 'FlowLane' in c and 'Category' in c and 'Value' in c]
        df['V_t'] = df[flow_cols].sum(axis=1)
        
        # 3. Hourly Flow Rate (q_t)
        df['q_t'] = df['V_t'] * 4
        
        # 4. Capacity (C)
        df['C'] = self.CAPACITY
        
        # 5. VOC (Volume-to-Capacity Ratio)
        df['VOC_t'] = df['q_t'] / df['C']
        
        # 6. Speed (s_t) - Optional, but good for validation
        # If not explicitly provided, derived from L / T_obs
        df['s_t'] = (self.LINK_LENGTH_M / 1000.0) / (df['T_obs'] / 3600.0)
        
        # 7. Free-flow Travel Time (t_0)
        # Use fixed constant or derive from low-flow conditions
        # Requirement says "Fixed constant", let's use 5th percentile of T_obs as a heuristic for now if not set
        self.T0 = df['T_obs'].quantile(0.05)
        df['t_0'] = self.T0
        
        # --- Vehicle Composition ---
        # Cat 3 + Cat 4 = HGV
        cat3_cols = [c for c in flow_cols if 'Category3' in c]
        cat4_cols = [c for c in flow_cols if 'Category4' in c]
        
        df['V_HGV_t'] = df[cat3_cols].sum(axis=1) + df[cat4_cols].sum(axis=1)
        df['HGV_share'] = df['V_HGV_t'] / df['V_t']
        df['HGV_share'] = df['HGV_share'].fillna(0) # Handle 0 flow
        
        # --- Time-of-Day ---
        # minute of day
        minutes = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
        df['tod_sin'] = np.sin(2 * np.pi * minutes / 1440)
        df['tod_cos'] = np.cos(2 * np.pi * minutes / 1440)
        
        # --- Daytype ---
        # 0-4: Mon-Fri, 5: Sat, 6: Sun
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['daytype'] = df['day_of_week']
        df['is_weekend'] = df['daytype'].isin([5, 6]).astype(int)
        
        # --- Weather ---
        # Assuming columns: 'precip', 'windspeed', 'visibility', 'conditions'
        # Map from provided data columns (need to verify exact names)
        
        if 'precip' in df.columns:
            df['R_t'] = pd.to_numeric(df['precip'], errors='coerce').fillna(0)
        else:
            df['R_t'] = 0
            
        if 'visibility' in df.columns:
            df['Vis_t'] = pd.to_numeric(df['visibility'], errors='coerce').fillna(999) # Default high vis
        else:
            df['Vis_t'] = 10 # Default km
            
        # Dummies
        df['Rain_t'] = (df['R_t'] > 0).astype(int)
        df['HeavyRain_t'] = (df['R_t'] > 2.5).astype(int) # Threshold example
        df['LowVis_t'] = (df['Vis_t'] < 1.0).astype(int) # Threshold example
        
        # Cleaning
        # Remove rows with missing T_obs or V_t
        df_clean = df.dropna(subset=['T_obs', 'V_t']).copy()
        
        # Remove bad data (T_obs <= 0 or V_t < 0)
        df_clean = df_clean[(df_clean['T_obs'] > 0) & (df_clean['V_t'] >= 0)]
        
        self.df = df_clean
        print(f"Preprocessing complete. Final shape: {self.df.shape}")
        return self.df

    def get_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data sequentially: First 3/4 Train, Last 1/4 Test.
        """
        if self.df is None:
            self.preprocess()
            
        n = len(self.df)
        split_idx = int(n * 0.75)
        
        self.train_df = self.df.iloc[:split_idx].copy()
        self.test_df = self.df.iloc[split_idx:].copy()
        
        print(f"Train set: {len(self.train_df)} records ({self.train_df['timestamp'].min()} to {self.train_df['timestamp'].max()})")
        print(f"Test set: {len(self.test_df)} records ({self.test_df['timestamp'].min()} to {self.test_df['timestamp'].max()})")
        
        return self.train_df, self.test_df

if __name__ == "__main__":
    loader = TrafficDataLoader()
    df = loader.load_data()
    loader.preprocess()
    train, test = loader.get_train_test_split()
    print(train.head())
