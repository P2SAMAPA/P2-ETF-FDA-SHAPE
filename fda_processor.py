"""
Functional Data Analysis: B‑spline smoothing, fPCA, and shape features.
"""
import numpy as np
import pandas as pd
from FDApy.representation import DenseArgvals
from FDApy.preprocessing import UFPCA
from FDApy.simulation import KarhunenLoeve
from scipy.interpolate import BSpline, make_lsq_spline


def smooth_univariate(data: np.ndarray, n_basis: int = 15, smoothing_parameter: float = 0.1):
    """
    Smooth univariate data of shape (n_samples, window_size) using B-splines.
    Returns a 2D array of smoothed curves.
    """
    n_samples, window_size = data.shape
    x = np.linspace(0, 1, window_size)
    
    # Create knot vector
    knots = np.linspace(0, 1, n_basis - 2)[1:-1]
    t = np.r_[np.zeros(3), knots, np.ones(3)]
    
    smoothed = np.zeros_like(data)
    for i in range(n_samples):
        # Fit smoothing spline with penalty on second derivative
        spline = make_lsq_spline(x, data[i], t, k=3)
        smoothed[i] = spline(x)
    
    return smoothed


def create_multivariate_fdata(data: np.ndarray, n_basis: int = None, smoothing_parameter: float = 0.1):
    """
    Smooth multivariate data of shape (n_samples, n_features, window_size).
    Returns a smoothed array and argvals object for FDApy.
    """
    if data.ndim == 2:
        data = data[:, np.newaxis, :]

    n_samples, n_features, window_size = data.shape

    if n_basis is None:
        n_basis = min(15, window_size // 4)

    smoothed_features = []
    for i in range(n_features):
        feature_data = data[:, i, :]  # (n_samples, window_size)
        smoothed = smooth_univariate(feature_data, n_basis, smoothing_parameter)
        smoothed_features.append(smoothed)

    smoothed_array = np.stack(smoothed_features, axis=1)  # (n_samples, n_features, window_size)
    
    # Create argvals object for FDApy
    argvals = DenseArgvals({'input_dim_0': np.linspace(0, 1, window_size)})
    
    return smoothed_array, argvals


def fit_fpca(smoothed_data: np.ndarray, argvals, n_components: int = 3):
    """Fit univariate FPCA using FDApy."""
    # FDApy's UFPCA expects data as a list of 2D arrays (one per feature)
    # For multivariate, we can fit separate UFPCA per feature
    n_samples, n_features, n_points = smoothed_data.shape
    
    fpca_models = []
    scores_list = []
    
    for i in range(n_features):
        feature_data = smoothed_data[:, i, :]  # (n_samples, n_points)
        ufpca = UFPCA(n_components=n_components)
        ufpca.fit(feature_data, argvals=argvals)
        fpca_models.append(ufpca)
        scores = ufpca.transform(feature_data, argvals=argvals)
        scores_list.append(scores)
    
    return fpca_models, scores_list


def extract_shape_features(smoothed_data: np.ndarray, argvals, fpca_models: list, scores_list: list, include_derivatives: bool = True):
    """
    Extract shape features: fPCA scores, and optionally first/second derivative values.
    Returns a DataFrame of features.
    """
    n_samples, n_features, n_points = smoothed_data.shape
    
    # Build feature dictionary from fPCA scores
    feature_dict = {}
    for feat_idx in range(n_features):
        scores = scores_list[feat_idx]  # (n_samples, n_components)
        for comp_idx in range(scores.shape[1]):
            feature_dict[f"fPC{comp_idx+1}_feat{feat_idx}"] = scores[:, comp_idx]
    
    if include_derivatives:
        x = np.linspace(0, 1, n_points)
        dx = x[1] - x[0]
        
        for feat_idx in range(n_features):
            feature_data = smoothed_data[:, feat_idx, :]
            deriv1 = np.gradient(feature_data, dx, axis=1)
            deriv2 = np.gradient(deriv1, dx, axis=1)
            
            mean_deriv1 = deriv1.mean(axis=1)
            mean_deriv2 = deriv2.mean(axis=1)
            
            feature_dict[f"deriv1_{feat_idx}"] = mean_deriv1
            feature_dict[f"deriv2_{feat_idx}"] = mean_deriv2
    
    return pd.DataFrame(feature_dict)
