"""Extended Kalman filter for one object associated by hand."""
from math import pi

import h5py
import numpy as np

from tracking.filter import extended
from tracking.visualize.predictions import visualize_predictions


# Step one: make a Kalman filter for single sensor for single object
datafile = "data/data_109.h5"
camera = h5py.File(datafile, 'r')
single_obj_ind = [2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 1, 1, 2, 1]
single_obj_det = np.zeros((20,4))

# Extract single object detections
for i in range(20):
    detection = camera["Sequence"][str(i)]["Detections"][single_obj_ind[i]]
    pos = list(detection[0])
    #angle = [detection[4]]
    length = [max(detection[1], detection[2])]
    single_obj_det[i,:] = pos + length

timestamps = camera["Timestamp"][0:20]
time_steps = np.insert(np.diff(timestamps), 0, 0)


# Bicycle model input
def state_transition_model(state, dt):
    new_state = state

    x       = state[0]
    y       = state[1]
    z       = state[2]
    v       = state[3]
    a       = state[4]
    theta   = state[5]
    delta   = state[6]
    phi     = state[7]
    L       = state[8]

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

    new_state[0] += dt * x_dot
    new_state[1] += dt * y_dot
    new_state[2] += dt * z_dot
    new_state[3] += dt * v_dot
    new_state[4] += dt * a_dot
    new_state[5] += dt * theta_dot
    new_state[6] += dt * delta_dot
    new_state[7] += dt * phi_dot
    new_state[8] += dt * L_dot

    return new_state


def state_transition_jacobian(state, dt):
    transition_jacobian = np.identity(9)

    x       = state[0]
    y       = state[1]
    z       = state[2]
    v       = state[3]
    a       = state[4]
    theta   = state[5]
    delta   = state[6]
    phi     = state[7]
    L       = state[8]

    beta = np.arctan(np.tan(delta) / 2) # The slip angle
    beta_d_delta = 4 / (5 + 3 * np.cos(2 * delta))

    transition_jacobian[0, 3] = dt * np.cos(beta + theta)
    transition_jacobian[0, 5] = dt * v * (-np.sin(beta + theta))
    transition_jacobian[0, 6] = dt * v * (-np.sin(beta + theta)) * beta_d_delta
    transition_jacobian[1, 3] = dt * np.sin(beta + theta)
    transition_jacobian[1, 5] = dt * v * np.cos(beta + theta)
    transition_jacobian[1, 6] = dt * v * np.cos(beta + theta) * beta_d_delta
    transition_jacobian[3, 4] = dt
    transition_jacobian[5, 3] = dt * np.tan(delta) * np.cos(beta) / L
    transition_jacobian[5, 6] = dt * v / L * (np.cos(beta) / (np.cos(delta) ** 2)
                                              - np.tan(delta) * np.sin(beta) * beta_d_delta)
    transition_jacobian[5, 8] = dt * (-v) * np.tan(delta) * np.cos(beta) / (L ** 2)
    transition_jacobian[6, 7] = dt

    return transition_jacobian


def observation_model(state, dt):
    observed_states = np.asarray((0, 1, 2, 8))
    observation = state[observed_states]

    return observation


def observation_jacobian(state, dt):
    jacobian = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    return jacobian


# Initilize positions and velocities
first_det = single_obj_det[0]
default_state = np.array((first_det[0], first_det[1], first_det[2],
                            1, 0.1, -pi / 2, 0, 0, 4))
default_cov = np.identity(9)

obj_track = extended.track(single_obj_det, time_steps, state_transition_model,
                           state_transition_jacobian, observation_model,
                           observation_jacobian, default_state, default_cov)

visualization_gen = zip(obj_track, range(20))
for ((x_updated, x_prediction, measurement), frame) in visualization_gen:
    visualize_predictions(measurement[0:3], x_prediction[0:3], x_updated[0:3],
                          camera, frame)
