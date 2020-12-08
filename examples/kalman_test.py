"""Simple Kalman filter implementation for one view"""
import h5py
import numpy as np
import matplotlib as plt

from tracking.filter import basic
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

visualization_gen = zip(basic.track(single_obj_det), range(20))
for ((x_updated, x_prediction, measurement), frame) in visualization_gen:
    visualize_predictions(measurement, x_prediction, x_updated, camera, frame)
