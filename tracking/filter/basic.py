"""A basic Kalman filter.
The filter needs an associated motion model to run, characterized by the
variables F, G, Q, H and R. The recommended way to implement this is by
creating a module that "overloads" the functions defined here. See 
const_acceleration.py as a reference.
"""
import numpy as np


def predict(state_current, cov_current, F, G, Q):
    """Predict new states of object."""
    state_prediction = np.matmul(F, state_current.T).T # + B*u
    cov_prediction = np.matmul(np.matmul(F, cov_current), F.T) \
                     + np.matmul(np.matmul(G, Q), G.T)

    return (state_prediction, cov_prediction)


def update(state_prediction, cov_prediction, measurement, H, R):
    """Update prediction based on measurement."""
    # The innovation
    residual = measurement - np.matmul(H,state_prediction.T).T
    residual_cov = np.matmul(H,np.matmul(cov_prediction,H.T)) + R

    # Kalman gain
    W = np.matmul(cov_prediction, np.matmul(H.T, np.linalg.inv(residual_cov)))

    state_updated = state_prediction + np.matmul(W, residual.T).T
    cov_updated = cov_prediction - np.matmul(W, np.matmul(residual_cov, W.T))

    return (state_updated, cov_updated)


def normalized_innovation(state_prediction, cov_prediction, measurement, H, R):
    """Find the normalized innovation for a measurement."""
    # The innovation
    residual = measurement - np.matmul(H, state_prediction.T).T
    innovation_variance = np.matmul(H, np.matmul(cov_prediction, H.T)) + R

    return np.matmul(residual, np.matmul(np.linalg.inv(innovation_variance),
                                         residual.T))
