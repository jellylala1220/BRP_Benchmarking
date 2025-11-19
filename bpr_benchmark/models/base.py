from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any

class BaseVDF(ABC):
    """
    Abstract Base Class for all Volume Delay Functions (VDF).
    """
    
    def __init__(self, name: str):
        self.name = name
        self.params = {}
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, df_train: pd.DataFrame) -> 'BaseVDF':
        """
        Trains the model on the training dataset.
        
        Args:
            df_train: Training DataFrame containing features and target.
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        Predicts travel times on the test dataset.
        
        Args:
            df_test: Test DataFrame.
            
        Returns:
            Numpy array of predicted travel times.
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Returns the learned parameters."""
        return self.params
    
    def __repr__(self):
        return f"VDFModel(name='{self.name}', fitted={self.is_fitted})"
