"""
Functional Data Analysis: B‑spline smoothing, fPCA, and shape features.
"""
import numpy as np
import pandas as pd
from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.dim_reduction import FPCA


def create_multivariate_fdata(data: np.ndarray, n_basis: int = None, smoothing_parameter: float = 0.1):
    """
    Smooth multivariate data of shape (n_samples, n_features, window_size).
    Returns a multivariate FDataGrid.
    """
    if data.ndim == 2:
        data = data[:, np.newaxis, :]  # Ensure 3D: (n_samples, n_features, window_size)

    n_samples, n_features, window_size = data.shape

    if n_basis is None:
        n_basis = min(15, window_size // 4)

    basis = BSplineBasis(n_basis=n_basis, domain_range=(0, window_size - 1))
    smoother = BasisSmoother(basis, smoothing_parameter=smoothing_parameter)

    smoothed_features = []
    for i in range(n_features):
        feature_data = data[:, i, :]  # (n_samples, window_size)
        fdata = FDataGrid(feature_data, sample_points=np.arange(window_size))
        smoothed = smoother.fit_transform(fdata)
        smoothed_features.append(smoothed.data_matrix)  # each is (n_samples, window_size)

    # Stack to (n_samples, n_features, window_size)
    smoothed_array = np.stack(smoothed_features, axis=1)

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

        # deriv.data_matrix has shape (n_samples, n_features, n_points)
        mean_deriv1 = deriv1.data_matrix.mean(axis=2)  # (n_samples, n_features)
        mean_deriv2 = deriv2.data_matrix.mean(axis=2)

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
