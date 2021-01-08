"""An implementation of a basic Kalman with the constant acceleration model.
This module implements an "interface" for modules that pairs a filter type with
a motion model, and can thus be directly used to track objects.
"""
import numpy as np

from . import basic


# Default model dynamics noise
Q = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

# The measurement model
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0]])

# Observation noise
R = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

NUM_STATES = 6
NUM_MEASUREMENTS = 3


def predict(state_current, cov_current, dt):
    """Predict state using constant acceleration model."""
    # The model dynamics
    F = np.array([[1, dt, 0,  0, 0,  0],
                  [0,  1, 0,  0, 0,  0],
                  [0,  0, 1, dt, 0,  0],
                  [0,  0, 0,  1, 0,  0],
                  [0,  0, 0,  0, 1, dt],
                  [0,  0, 0,  0, 0,  1]])
    # Model dynamics noise matrix
    G = np.array([[dt**2/2,       0,       0],
                  [     dt,       0,       0],
                  [      0, dt**2/2,       0],
                  [      0,      dt,       0],
                  [      0,       0, dt**2/2],
                  [      0,       0,      dt]])

    return basic.predict(state_current, cov_current, F, G, Q)


def update(state_prediction, cov_prediction, measurement, dt):
    """Update state using constant acceleration model."""

    return basic.update(state_prediction, cov_prediction, measurement, H, R)


def normalized_innovation(state_pred, cov_pred, measurement, dt):
    """Normalized innovation using constant acceleration model."""

    return basic.normalized_innovation(state_pred, cov_pred, measurement, H, R)


def default_state(measurement):
    """Initialize a new state vector based on an initial measurement."""
    state = np.zeros((1, NUM_STATES))
    state[0][(0, 2, 4),] = measurement[0]

    return state


def state2position(state):
    """Get a position given a state."""
    position_states = np.asarray((0, 2, 4))
    position = state[:, position_states]

    return position


def measurement2position(measurement):
    """Get a position given a measurement."""
    position_measurements = np.asarray((0, 1, 2))
    position = measurement[:, position_measurements]

    return position
