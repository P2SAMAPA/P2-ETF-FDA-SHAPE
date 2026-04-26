""" Lightweight predictive model using HistGradientBoostingRegressor. """
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

class ShapePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = HistGradientBoostingRegressor(
            loss='squared_error',
            max_iter=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
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
