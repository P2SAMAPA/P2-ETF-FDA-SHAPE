"""
Functional Data Analysis: B‑spline smoothing, fPCA, and shape features.
"""
import numpy as np
import pandas as pd
from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.dim_reduction import FPCA


def create_fdatagrid(data: np.ndarray, n_basis: int = None, smoothing_parameter: float = 0.1):
    """
    Smooth a multivariate time series into functional data.
    data shape: (n_samples, n_features, window_size) or (n_samples, window_size) for univariate.
    Returns: FDataGrid of smoothed curves.
    """
    # Ensure data is 3D: (n_samples, n_features, window_size)
    if data.ndim == 2:
        # Univariate case: (n_samples, window_size) -> (n_samples, 1, window_size)
        data = data[:, np.newaxis, :]

    n_samples, n_features, window_size = data.shape

    if n_basis is None:
        n_basis = min(15, window_size // 4)

    basis = BSplineBasis(n_basis=n_basis, domain_range=(0, window_size - 1))
    smoother = BasisSmoother(basis, smoothing_parameter=smoothing_parameter)

    # Smooth each feature separately and combine
    smoothed_curves = []
    for i in range(n_features):
        # Extract feature i: shape (n_samples, window_size)
        feature_data = data[:, i, :]
        fdata = FDataGrid(feature_data, sample_points=np.arange(window_size))
        smoothed = smoother.fit_transform(fdata)
        # smoothed.data_matrix is (n_samples, window_size)
        smoothed_curves.append(smoothed.data_matrix)

    # Stack along feature dimension: list of (n_samples, window_size) -> (n_samples, n_features, window_size)
    smoothed_array = np.stack(smoothed_curves, axis=1)

    return FDataGrid(smoothed_array, sample_points=np.arange(window_size))


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

        # Average derivative values over the window
        mean_deriv1 = deriv1.data_matrix.mean(axis=2).squeeze()
        mean_deriv2 = deriv2.data_matrix.mean(axis=2).squeeze()

        # Ensure we have 2D arrays: (n_samples, n_features)
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
