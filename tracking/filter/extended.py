"""An extended Kalman filter.
The filter needs an associated motion model to run, characterized by the
functions state_transition_model and observation_model, and those functions'
jacobians. The recommended way to implement this is by creating a module that
"overloads" the functions defined here. See bicycle.py as a reference.
"""
import numpy as np


def predict(state_current, cov_current, state_transition_model,
            state_transition_jacobian, v, dt):
    """Predict new states of object."""
    F = state_transition_jacobian(state_current, dt)
    G = dt * np.identity(9)
    Q = np.matmul(G, v)

    state_prediction = state_transition_model(state_current, dt)

    cov_prediction = np.matmul(np.matmul(F, cov_current), F.transpose()) + Q

    return (state_prediction, cov_prediction)


def update(state_prediction, cov_prediction, observation_model,
           observation_jacobian, measurement, R, dt):
    """Update prediction based on measurement."""
    H = observation_jacobian(state_prediction, dt)

    # The innovation
    state_residual = measurement - observation_model(state_prediction, dt)
    cov_residual = np.matmul(H, np.matmul(cov_prediction,H.T)) + R

    # Kalman gain
    K = np.matmul(cov_prediction, np.matmul(H.T, np.linalg.inv(cov_residual)))

    state_updated = state_prediction + np.matmul(K, state_residual.T).T
    cov_updated = cov_prediction - np.matmul(K, np.matmul(H, cov_prediction))

    return (state_updated, cov_updated)


def normalized_innovation(state_prediction, cov_prediction, observation_model,
                          observation_jacobian, measurement, R, dt):
    """Find the normalized innovation for a measurement."""
    H = observation_jacobian(state_prediction, dt)

    # The innovation
    residual = measurement - observation_model(state_prediction, dt)
    residual_cov = np.matmul(H, np.matmul(cov_prediction, H.T)) + R

    return np.matmul(residual, np.matmul(np.linalg.inv(residual_cov),
                                         residual.T))
