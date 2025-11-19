import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Callable, Tuple, Dict, Any, Optional

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """
    Calculates unified metrics for BPR benchmarking.
    
    Args:
        y_true: Observed travel times.
        y_pred: Predicted travel times.
        model_name: Name of the model for logging.
        
    Returns:
        Dictionary containing RMSE, MAE, MAPE, R2, P95.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter out NaNs if any (though data should be clean)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(mask):
        print(f"Warning: NaN values found in predictions for {model_name}. Filtering...")
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE: Mean Absolute Percentage Error
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
    r2 = r2_score(y_true, y_pred)
    
    # P95: 95th percentile of absolute error
    abs_error = np.abs(y_true - y_pred)
    p95 = np.percentile(abs_error, 95)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "P95": p95
    }

def fit_nls(
    func: Callable, 
    X: np.ndarray, 
    y: np.ndarray, 
    p0: list, 
    bounds: Tuple[list, list],
    maxfev: int = 10000
) -> Tuple[np.ndarray, Any]:
    """
    Unified NLS estimator using scipy.optimize.curve_fit.
    
    Args:
        func: The model function to fit (callable).
        X: Input features (can be 1D or 2D array).
        y: Target variable (observed travel time).
        p0: Initial guess for parameters.
        bounds: (lower_bounds, upper_bounds).
        maxfev: Maximum number of function evaluations.
        
    Returns:
        Tuple of (optimized_parameters, covariance_matrix).
    """
    try:
        popt, pcov = curve_fit(
            func, 
            X, 
            y, 
            p0=p0, 
            bounds=bounds, 
            maxfev=maxfev
        )
        return popt, pcov
    except Exception as e:
        print(f"NLS fitting failed: {e}")
        # Return initial guess if fitting fails
        return np.array(p0), None
