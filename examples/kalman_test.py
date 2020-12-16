"""Simple Kalman filter implementation for one view"""
import h5py
import numpy as np

from tracking.filter import const_acceleration
from tracking.visualize.predictions import visualize_predictions


# Step one: make a Kalman filter for single sensor for single object
datafile = "data/data_109.h5"
camera = h5py.File(datafile, 'r')
single_obj_ind = [2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 1, 1, 2, 1]
single_obj_det = np.zeros((20,3))

# Extract single object detections
for i in range(20):
    detections = camera["Sequence"][str(i)]["Detections"]
    detections = np.asarray([list(det[0]) for det in list(detections)])
    single_obj_det[i,:] = detections[single_obj_ind[i],:]

timestamps = camera["Timestamp"][0:20]
time_steps = np.insert(np.diff(timestamps), 0, np.median(np.diff(timestamps)))

print(time_steps)
obj_track = const_acceleration.track(single_obj_det, time_steps)

visualization_gen = zip(obj_track, range(20))
for ((x_updated, x_prediction, measurement), frame) in visualization_gen:
    visualize_predictions(measurement, x_prediction[:,0::2], x_updated[:,0::2],
                          camera, frame)
