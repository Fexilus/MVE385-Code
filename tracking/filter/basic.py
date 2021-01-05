"""A basic Kalman filter."""
import numpy as np


def predict(x_current, cov_current, F, G, Q):
    """Predict new states of object."""
    x_prediction = np.matmul(F, x_current.T).T # + B*u
    cov_prediction = np.matmul(np.matmul(F, cov_current), F.transpose()) \
                     + np.matmul(np.matmul(G, Q), G.transpose())

    return (x_prediction, cov_prediction)


def update(x_prediction, cov_prediction, measurement, H, R):
    """Update prediction based on measurement."""
    # The innovation
    residual = measurement - np.matmul(H,x_prediction.T).T
    residual_cov = np.matmul(H,np.matmul(cov_prediction,H.T)) + R

    # Kalman gain
    W = np.matmul(cov_prediction, np.matmul(H.T, np.linalg.inv(residual_cov)))

    x_updated = x_prediction + np.matmul(W, residual.T).T
    cov_updated = cov_prediction - np.matmul(W, np.matmul(residual_cov,W.transpose()))

    return (x_updated, cov_updated)


def normalized_innovation(state_prediction, cov_prediction, measurement, H, R):
    """Find the normalized innovation for a measurement."""
    # The innovation
    residual = measurement - np.matmul(H, state_prediction.T).T
    innovation_variance = np.matmul(H, np.matmul(cov_prediction, H.T)) + R

    return np.matmul(residual, np.matmul(np.linalg.inv(innovation_variance),
                                         residual.T))
