"""Extended Kalman filter for one object associated by hand."""
from math import pi

import h5py
import numpy as np

from tracking.filter import bicycle
from tracking.track import track
from tracking.visualize.predictions import visualize_predictions


# Step one: make a Kalman filter for single sensor for single object
datafile = "data/data_109.h5"
camera = h5py.File(datafile, 'r')
single_obj_ind = [2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 1, 1, 2, 1]

def single_obj_det():
    """Extract single object detections."""
    for i in range(20):
        detection = camera["Sequence"][str(i)]["Detections"][single_obj_ind[i]]
        pos = list(detection[0])
        #angle = [detection[4]]
        length = [max(detection[1], detection[2])]

        object_det_raw = np.asarray(pos + length)

        yield object_det_raw.reshape((1, bicycle.NUM_MEASUREMENTS))

timestamps = iter(np.asarray(camera["Timestamp"][0:20]))

# Initilize positions and velocities
default_state = lambda measurement: bicycle.default_state(measurement, -pi / 2)

obj_track = track(single_obj_det(), timestamps, bicycle,
                  specific_default_state=default_state)

visualization_gen = zip(obj_track, range(20))

for (object_track, state_prediction, measurement), frame in visualization_gen:
    visualize_predictions(bicycle.measurement2position(measurement),
                          bicycle.state2position(state_prediction),
                          object_track[-1][0],
                          camera, frame)
