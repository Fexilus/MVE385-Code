"""Simple Kalman filter implementation for one view"""
import h5py
import numpy as np

from tracking.filter import const_acceleration as const_acc_model
from tracking.track import track
from tracking.visualize.predictions import visualize_predictions


# Step one: make a Kalman filter for single sensor for single object
datafile = "data/data_109.h5"
camera = h5py.File(datafile, 'r')
single_obj_ind = [2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 1, 1, 2, 1]


def single_obj_det():
    """Extract single object detections."""
    for i in range(20):
        detections = camera["Sequence"][str(i)]["Detections"]
        detections = np.asarray([list(det[0]) for det in list(detections)])

        object_det_raw = detections[single_obj_ind[i]]

        yield object_det_raw.reshape((1, const_acc_model.NUM_MEASUREMENTS))

timestamps = iter(np.asarray(camera["Timestamp"][0:20]))

obj_track = track(single_obj_det(), timestamps, const_acc_model)

visualization_gen = zip(obj_track, range(20))

for (object_track, state_prediction, measurement), frame in visualization_gen:
    visualize_predictions(const_acc_model.measurement2position(measurement),
                          const_acc_model.state2position(state_prediction),
                          object_track[-1][0],
                          camera, frame)
