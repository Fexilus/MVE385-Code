"""Extended Kalman filter for one object associated by hand."""
from math import pi

import h5py
import numpy as np

from tracking.filter import bicycle
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

# Initilize positions and velocities
first_det = single_obj_det[0]
default_state = np.array((first_det[0], first_det[1], first_det[2],
                            1, 0.1, -pi / 2, 0, 0, 4))
default_cov = np.identity(9)

obj_track = bicycle.track(single_obj_det, time_steps, default_state,
                          default_cov)

visualization_gen = zip(obj_track, range(20))
for ((x_updated, x_prediction, measurement), frame) in visualization_gen:
    visualize_predictions(measurement[0:3], x_prediction[0:3], x_updated[0:3],
                          camera, frame)
