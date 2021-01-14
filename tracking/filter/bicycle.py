"""An implementation of a basic Kalman with the bicycle motion model.
This module implements an "interface" for modules that pairs a filter type with
a motion model, and can thus be directly used to track objects.
"""
import numpy as np

from . import extended


NUM_STATES = 9
NUM_MEASUREMENTS = 4

# Default model noise
v = np.array((1, 1, 1, 1, 1, 1, 1, 1, 1))

# Measurement noise covariance matrix
R = np.identity(4)

def state_transition_model(state, dt):
    """The transition model of the bicycle model."""
    new_state = state

    x       = state[0][0]
    y       = state[0][1]
    z       = state[0][2]
    v       = state[0][3]
    a       = state[0][4]
    theta   = state[0][5]
    delta   = state[0][6]
    phi     = state[0][7]
    L       = state[0][8]

    beta = np.arctan(np.tan(delta) / 2) # The slip angle

    x_dot       = v * np.cos(beta + theta)
    y_dot       = v * np.sin(beta + theta)
    z_dot       = 0
    v_dot       = a
    a_dot       = 0
    theta_dot   = v * np.tan(delta) * np.cos(beta) / L
    delta_dot   = phi
    phi_dot     = 0
    L_dot       = 0

    new_state[0][0] += dt * x_dot
    new_state[0][1] += dt * y_dot
    new_state[0][2] += dt * z_dot
    new_state[0][3] += dt * v_dot
    new_state[0][4] += dt * a_dot
    new_state[0][5] += dt * theta_dot
    new_state[0][6] += dt * delta_dot
    new_state[0][7] += dt * phi_dot
    new_state[0][8] += dt * L_dot

    return new_state


def state_transition_jacobian(state, dt):
    """The jacobian of the transition model of the bicycle model."""
    transition_jacobian = np.identity(NUM_STATES)

    x       = state[0][0]
    y       = state[0][1]
    z       = state[0][2]
    v       = state[0][3]
    a       = state[0][4]
    theta   = state[0][5]
    delta   = state[0][6]
    phi     = state[0][7]
    L       = state[0][8]

    beta = np.arctan(np.tan(delta) / 2) # The slip angle
    beta_d_delta = 4 / (5 + 3 * np.cos(2 * delta))

    x_d_v           = np.cos(beta + theta)
    x_d_theta       = v * (-np.sin(beta + theta))
    x_d_delta       = v * (-np.sin(beta + theta)) * beta_d_delta
    y_d_v           = np.sin(beta + theta)
    y_d_theta       = v * np.cos(beta + theta)
    y_d_delta       = v * np.cos(beta + theta) * beta_d_delta
    v_d_a           = 1
    theta_d_v       = np.tan(delta) * np.cos(beta) / L
    theta_d_delta   = v / L * (np.cos(beta) / (np.cos(delta) ** 2)
                               - np.tan(delta) * np.sin(beta) * beta_d_delta)
    theta_d_delta   = (-v) * np.tan(delta) * np.cos(beta) / (L ** 2)
    delta_d_phi     = 1

    transition_jacobian[0, 3] = dt * x_d_v
    transition_jacobian[0, 5] = dt * x_d_theta
    transition_jacobian[0, 6] = dt * x_d_delta
    transition_jacobian[1, 3] = dt * y_d_v
    transition_jacobian[1, 5] = dt * y_d_theta
    transition_jacobian[1, 6] = dt * y_d_delta
    transition_jacobian[3, 4] = dt * v_d_a
    transition_jacobian[5, 3] = dt * theta_d_v
    transition_jacobian[5, 6] = dt * theta_d_delta
    transition_jacobian[5, 8] = dt * theta_d_delta
    transition_jacobian[6, 7] = dt * delta_d_phi

    return transition_jacobian


def observation_model(state, dt):
    """The observation model of the bicycle model."""
    observed_states = np.asarray((0, 1, 2, 8))
    observation = state[:, observed_states]

    return observation


def observation_jacobian(state, dt):
    """The jacobian of the observation model of the bicycle model."""
    jacobian = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    return jacobian


def predict(state_current, cov_current, dt):
    """Predict state using constant acceleration model."""
    return extended.predict(state_current, cov_current, state_transition_model,
                            state_transition_jacobian, v, dt)


def update(state_prediction, cov_prediction, measurement, dt):
    """Update state using constant acceleration model."""
    return extended.update(state_prediction, cov_prediction, observation_model,
                           observation_jacobian, measurement, R, dt)


def normalized_innovation(state_prediction, cov_prediction, measurement, dt):
    """Normalized innovation using constant acceleration model."""
    return extended.normalized_innovation(state_prediction, cov_prediction,
                                          observation_model,
                                          observation_jacobian, measurement,
                                          R, dt)


def default_state(detection, default_direction=0, default_speed=0):
    """Initialize a new state vector based on the first detection."""
    state = np.ndarray((1, 9), buffer=np.asarray((detection[0][0],
                                                  detection[0][1],
                                                  detection[0][2],
                                                  default_speed, 0,
                                                  default_direction, 0, 0,
                                                  detection[0][3])))

    return state


def state2position(state):
    """Get a position given a state."""
    position_states = np.asarray((0, 1, 2))
    position = state[:, position_states]

    return position


def measurement2position(detection):
    """Get a position given a measurement."""
    position_detections = np.asarray((0, 1, 2))
    position = detection[:, position_detections]

    return position
