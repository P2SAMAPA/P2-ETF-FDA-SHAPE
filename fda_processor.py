"""
Functional Data Analysis: B‑spline smoothing, fPCA, and shape features.
"""
import numpy as np
import pandas as pd
from skfda import FDataGrid, concatenate
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.dim_reduction import FPCA


def smooth_univariate(data: np.ndarray, n_basis: int = None, smoothing_parameter: float = 0.1):
    """
    Smooth univariate data of shape (n_samples, window_size).
    Returns an FDataGrid of smoothed curves.
    """
    n_samples, window_size = data.shape
    if n_basis is None:
        n_basis = min(15, window_size // 4)

    basis = BSplineBasis(n_basis=n_basis, domain_range=(0, window_size - 1))
    smoother = BasisSmoother(basis, smoothing_parameter=smoothing_parameter)
    fdata = FDataGrid(data, sample_points=np.arange(window_size))
    return smoother.fit_transform(fdata)


def create_multivariate_fdata(data: np.ndarray, n_basis: int = None, smoothing_parameter: float = 0.1):
    """
    Smooth multivariate data of shape (n_samples, n_features, window_size).
    Returns a multivariate FDataGrid.
    """
    # Ensure data is 3D: (n_samples, n_features, window_size)
    if data.ndim == 2:
        data = data[:, np.newaxis, :]

    n_samples, n_features, window_size = data.shape

    # Smooth each feature independently
    smoothed_fdatas = []
    for i in range(n_features):
        feature_data = data[:, i, :]  # shape: (n_samples, window_size)
        smoothed_fdata = smooth_univariate(feature_data, n_basis, smoothing_parameter)
        smoothed_fdatas.append(smoothed_fdata)

    # Combine into a single multivariate FDataGrid
    if n_features == 1:
        return smoothed_fdatas[0]
    else:
        return concatenate(smoothed_fdatas, as_coordinates=True)


def fit_fpca(fdata: FDataGrid, n_components: int = 3):
    """Fit functional PCA and return the fitted FPCA object."""
    fpca = FPCA(n_components=n_components)
    fpca.fit(fdata)
    return fpca


def extract_shape_features(fdata: FDataGrid, fpca: FPCA, include_derivatives: bool = True):
    """
    Extract shape features: fPCA scores, and optionally first/second derivative values.
    Returns a DataFrame of features.
    """
    scores = fpca.transform(fdata)  # (n_samples, n_components)

    feature_dict = {f"fPC{i+1}": scores[:, i] for i in range(scores.shape[1])}

    if include_derivatives:
        deriv1 = fdata.derivative()
        deriv2 = deriv1.derivative()

        # Derivative data matrices have shape (n_samples, n_features, n_points)
        mean_deriv1 = deriv1.data_matrix.mean(axis=2)  # (n_samples, n_features)
        mean_deriv2 = deriv2.data_matrix.mean(axis=2)

        # Handle univariate case where shape might be squeezed to (n_samples,)
        if mean_deriv1.ndim == 1:
            mean_deriv1 = mean_deriv1[:, np.newaxis]
            mean_deriv2 = mean_deriv2[:, np.newaxis]

        for i in range(mean_deriv1.shape[1]):
            feature_dict[f"deriv1_{i}"] = mean_deriv1[:, i]
            feature_dict[f"deriv2_{i}"] = mean_deriv2[:, i]

    return pd.DataFrame(feature_dict)


def smooth_single_curve(data: np.ndarray, n_basis: int = None, smoothing_parameter: float = 0.1):
    """Smooth a single curve (window_size,) and return the smoothed values."""
    window_size = len(data)
    if n_basis is None:
        n_basis = min(15, window_size // 4)

    basis = BSplineBasis(n_basis=n_basis, domain_range=(0, window_size - 1))
    smoother = BasisSmoother(basis, smoothing_parameter=smoothing_parameter)
    fdata = FDataGrid(data[np.newaxis, :], sample_points=np.arange(window_size))
    smoothed = smoother.fit_transform(fdata)
    return smoothed.data_matrix.flatten()
