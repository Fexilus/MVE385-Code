"""An implementation of the basic Kalman filter using the constant acceleration model."""
import numpy as np

from . import basic


Q = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0]]) # "Measurement model"
R = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])


def predict(x_current, cov_current, dt):
    """Predict state using constant acceleration model."""
    F = np.array([[1, dt, 0,  0, 0,  0],
                      [0,  1, 0,  0, 0,  0],
                      [0,  0, 1, dt, 0,  0],
                      [0,  0, 0,  1, 0,  0],
                      [0,  0, 0,  0, 1, dt],
                      [0,  0, 0,  0, 0,  1]]) # The dynamics model
    G = np.array([[dt**2/2,       0,       0],
                    [     dt,       0,       0],
                    [      0, dt**2/2,       0],
                    [      0,      dt,       0],
                    [      0,       0, dt**2/2],
                    [      0,       0,      dt]])

    return basic.predict(x_current, cov_current, F, G, Q)


def update(x_prediction, cov_prediction, measurement, dt):
    """Update state using constant acceleration model."""
    return basic.update(x_prediction, cov_prediction, measurement, H, R)


def normalized_innovation(state_pred, cov_pred, measurement, dt):
    """Normalized innovation using constant acceleration model."""

    return basic.normalized_innovation(state_pred, cov_pred, measurement, H, R)


def defaultStateVector(detection, vel=2.0):
    """Initialize a new state vector based on the first detection."""
    default_state = np.full((1, 6), vel)
    default_state[0][(0, 2, 4),] = detection[0]

    return default_state


def state_to_position(state):
    position_states = np.asarray((0, 2, 4))
    position = state[:, position_states]

    return position


def detection_to_position(detection):
    position_detections = np.asarray((0, 1, 2))
    position = detection[:, position_detections]

    return position


def track(single_obj_det, time_steps):
    """Track a single object with a basic Kalman filter."""
    # Initilize positions and velocities
    init_velocity = 2#*np.ones((3,1)) # dt*init_velocity is how far the object moved between two frames
    pos_init = np.asarray(single_obj_det[0,:])

    pos_t = pos_init[..., None]
    x_current = defaultStateVector(pos_t, init_velocity)
    cov_current = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])

    for measurement, dt in zip(single_obj_det, time_steps):
        (x_prediction, cov_prediction) = predict(x_current, cov_current, dt)

        (x_updated, cov_updated) = update(x_prediction, cov_prediction,
                                          measurement, dt)

        # Set current to update
        x_current = x_updated
        cov_current = cov_updated

        yield (x_updated, x_prediction, measurement)
