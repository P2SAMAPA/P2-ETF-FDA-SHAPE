"""
Lightweight predictive model using Ridge regression.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import config


class ShapePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RidgeCV(alphas=config.RIDGE_ALPHAS, cv=5)
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict(X_scaled)

    def predict_returns(self, X: pd.DataFrame) -> dict:
        """Return dictionary of predicted returns per ETF."""
        preds = self.predict(X)
        return {col: preds[i] for i, col in enumerate(X.columns) if col in self.feature_names}
