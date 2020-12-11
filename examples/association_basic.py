import numpy as np
import h5py
from tracking.filter.basic import predict,update,createStateVector
from tracking.visualize.predictions import visualize_predictions

def associate_NN(prediction,detections,H,cov_prediction,R):
    # Nearest neighbour association
    innovation_dist = np.zeros((detections.shape[0],1))

    for i in range(detections.shape[0]):
        m = detections[i,:]
        innovation = m.T - np.matmul(H,prediction)
        S = np.matmul(H,np.matmul(cov_prediction,H.T)) + R
        norm_innovation = np.matmul(innovation.T,np.matmul(np.linalg.inv(S),innovation))
        innovation_dist[i] = norm_innovation

    closest_neighbour_ind = np.argmin(innovation_dist)
    closest_neighbour = detections[closest_neighbour_ind]
    
    # TODO: set a cut-off for how far away the next detection can be
    if (innovation_dist[closest_neighbour_ind] > 10):
        # Send back that there is no associated neighbour
        closest_neighbour = np.matrix([0,0,0])
    # TODO how to handle new detections
    
    return(closest_neighbour)


def track_with_association(pos_init,camera):
    # Detections contain all measurements made at one timestep
    # Difference from track: must associate each prediction with a measurement,
    # can therefore only do one timestep at a tim
    """For one time step: make a prediction, associate a measurement, and return the update"""
    # Initilize positions and velocities
    init_velocity = 2#*np.ones((3,1)) # dt*init_velocity is how far the object moved between two frames 
    pos_init = np.asarray(pos_init)
    timestamps = camera["Timestamp"][0:20]
    time_steps = np.insert(np.diff(timestamps), 0, np.median(timestamps))

    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0]]) # "Measurement model"

    # TODO: fill with accurate values
    # Q = model noise covariance matrix
    Q = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    # R = measurement noise covariance matrix
    R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    pos_t = pos_init[..., None]

    x_current = createStateVector(pos_t, init_velocity)
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
                      [      0,       0,      dt]]) # To be multiplied with model noise v(x)  = [vx,vy,vz]

        (x_prediction, cov_prediction) = predict(x_current, cov_current,
                                                 F, G, Q)

        # Reshape detections
        next_detections = np.matrix(next_detections)
        associated_detection = associate_NN(x_prediction,next_detections,H,cov_prediction,R)

        if(np.count_nonzero(associated_detection) == 0):
            # Don't make an update; there was no associated detection
            x_updated = x_current
            cov_updated = cov_current
        else:
            # Make an update
            (x_updated, cov_updated) = update(x_prediction, cov_prediction,
                                          associated_detection, H, R)

        # Set current to update
        x_current = x_updated # Only useful if we can loop through time steps
        cov_current = cov_updated

        yield (x_updated, x_prediction, associated_detection)

datafile = "tracking/data/data_109.h5"
camera = h5py.File(datafile, 'r')

# Get initial detection of single object
detections_0 = camera["Sequence"]["0"]["Detections"]
detections_0 = np.asarray([list(det[0]) for det in list(detections_0)])

single_obj_det_start = detections_0[2,:]


obj_track = track_with_association(single_obj_det_start,camera)

visualization_gen = zip(obj_track, range(20))
for ((x_updated, x_prediction, associated_detection), frame) in visualization_gen:
    visualize_predictions(associated_detection, x_prediction, x_updated, camera, frame)
