"""A Kalman filter for states of the kinematic bicycle model."""
import numpy as np


def predict(x_current, cov_current, state_transition_model, F, Q, dt):
    """Predict new states of object."""
    x_prediction = state_transition_model(x_current, dt)

    cov_prediction = np.matmul(np.matmul(F, cov_current), F.transpose()) + Q

    return (x_prediction, cov_prediction)


def update(x_prediction, cov_prediction, observation_model, measurement, H, R,
           dt):
    """Update prediction based on measurement."""
    # The innovation
    x_residual = measurement - observation_model(x_prediction, dt)
    cov_residual = np.matmul(H, np.matmul(cov_prediction,H.T)) + R

    # Kalman gain
    K = np.matmul(cov_prediction, np.matmul(H.T, np.linalg.inv(cov_residual)))

    x_updated = x_prediction + np.matmul(K, x_residual.T).T
    cov_updated = cov_prediction - np.matmul(K, np.matmul(H, cov_prediction))

    return (x_updated, cov_updated)


def track(single_obj_det, time_steps,
          state_transition_model, state_transition_jacobian,
          observation_model, observation_jacobian,
          default_state, default_cov):
    """Track a single object with a basic Kalman filter."""
    # v = model noise
    v = np.array((1, 1, 1, 1, 1, 1, 1, 1, 1))
    # R = measurement noise covariance matrix
    R = np.identity(4)

    x_current =  default_state
    cov_current = default_cov

    for measurement, dt in zip(single_obj_det, time_steps):
        F = state_transition_jacobian(x_current, dt)
        G = dt * np.identity(9) # To be multiplied with model noise v(x)  = [vx,vy,vz]
        H = observation_jacobian(x_current, dt)

        Q = np.matmul(G, v)

        (x_prediction, cov_prediction) = predict(x_current, cov_current,
                                                 state_transition_model, F, Q,
                                                 dt)

        (x_updated, cov_updated) = update(x_prediction, cov_prediction,
                                          observation_model, measurement, H, R,
                                          dt)

        # Set current to update
        x_current = x_updated
        cov_current = cov_updated

        yield (x_updated, x_prediction, measurement)
