import numpy as np
import h5py

from tracking.filter.const_acceleration import predict, update, defaultStateVector, normalized_innovation
from tracking.association.association_tracking import associate_NN
from tracking.visualize.predictions import visualize_predictions


def track_with_association(pos_init, camera, nbr_of_frames):
    # Detections contain all measurements made at one timestep
    # Difference from track: must associate each prediction with a measurement,
    # can therefore only do one timestep at a tim
    """For nbr_of_frames time steps: make a prediction, associate a measurement, and return the update"""
    # Initilize positions and velocities
    init_velocity = 2#*np.ones((3,1)) # dt*init_velocity is how far the object moved between two frames 
    pos_init = np.asarray(pos_init)
    timestamps = camera["Timestamp"][0:nbr_of_frames]
    time_steps = np.insert(np.diff(timestamps), 0, np.median(timestamps))

    pos_t = pos_init[..., None]

    x_current = defaultStateVector(pos_t, init_velocity)
    cov_current = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])

    for i in range(time_steps.shape[0]-1): # -1 because at the first position in time_step vector ther is not a time differe
        next_detections = camera["Sequence"][str(i+1)]["Detections"]
        next_detections = np.asarray([list(det[0]) for det in list(next_detections)])
        # Assume vector x = [px, vx, py, vy, pz, vz].T
        dt = time_steps[i+1]

        (x_prediction, cov_prediction) = predict(x_current, cov_current, dt)

        # Reshape detections
        next_detections = np.matrix(next_detections)
        associated_detection = associate_NN(x_prediction, next_detections,
                                            cov_prediction,
                                            normalized_innovation, dt)

        if(np.count_nonzero(associated_detection) == 0):
            # Don't make an update; there was no associated detection
            x_updated = x_current
            cov_updated = cov_current
        else:
            # Make an update
            (x_updated, cov_updated) = update(x_prediction, cov_prediction,
                                              associated_detection, dt)

        # Set current to update
        x_current = x_updated # Only useful if we can loop through time steps
        cov_current = cov_updated
        # yield only the position
        x_prediction = x_prediction[0,0::2]
        x_updated = x_updated[0,0::2]
        x_prediction = x_prediction.flatten()
        x_updated = x_updated.flatten()

        yield (x_updated, x_prediction, associated_detection)

datafile = "data/data_109.h5"
camera = h5py.File(datafile, 'r')
nbr_of_frames = 50

# Get initial detection of single object
detections_0 = camera["Sequence"]["0"]["Detections"]
detections_0 = np.asarray([list(det[0]) for det in list(detections_0)])

single_obj_det_start = detections_0[2,:]


obj_track = track_with_association(single_obj_det_start,camera,nbr_of_frames)

visualization_gen = zip(obj_track, range(nbr_of_frames))
for ((x_updated, x_prediction, associated_detection), frame) in visualization_gen:
    visualize_predictions(associated_detection, x_prediction, x_updated, camera, frame)
