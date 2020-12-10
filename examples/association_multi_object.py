import numpy as np
import h5py
from tracking.filter.basic import predict,update,createStateVector
from tracking.visualize.predictions import visualize_predictions
from examples.association_basic import associate_NN


# TODO
# 1. NN for multi-objects to create tracks T_i
# 2. Add termination critera according to article
# 3. Add initilizations for unassociated tracks

# WILL GENERATE: Tracks T_i with a set of tuples (x_k,y_k,z_k,t_k) indicating location ob object i at time t_k
# (In line with article implementation)

datafile = "tracking/data/data_109.h5"
camera = h5py.File(datafile, 'r')
nbr_of_frames = 50

# Get initial detection of single object
detections_0 = camera["Sequence"]["0"]["Detections"]
detections_0 = np.asarray([list(det[0]) for det in list(detections_0)])

# Initialize tracks at time 0
# current_tracks is a list of the detection matrices
current_tracks = []
for i in range(detections_0.shape[0]):
    track_i_pos = detections_0[i,:]
    track_i_pos = np.asmatrix(track_i_pos)
    track_i = np.zeros((1,4)) # Last value indicate in which frame this object had this position
    track_i[0,0:3] = track_i_pos
    current_tracks.append(track_i)

# Initialize values
init_velocity = 2#*np.ones((3,1)) # dt*init_velocity is how far the object moved between two frames 
timestamps = camera["Timestamp"][0:nbr_of_frames]
time_steps = np.insert(np.diff(timestamps), 0, np.median(timestamps))
H = np.array([[1, 0, 0, 0, 0, 0],

              [0, 0, 1, 0, 0, 0],

              [0, 0, 0, 0, 1, 0]])

# TODO: fill with accurate values

# Q = model noise covariance matri
Q = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    # R = measurement noise covariance
R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])


for f in range(nbr_of_frames):
    next_detections = camera["Sequence"][str(f+1)]["Detections"]
    next_detections = np.asarray([list(det[0]) for det in list(next_detections)])
    next_detections = np.matrix(next_detections)
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
    
    for o in range(len(current_tracks)):
        pos = current_tracks[o][-1,:] # Last posistion of object o
        pos_t = pos[..., None]
        x_current = createStateVector(pos_t, init_velocity) #x = [px, vx, py, vy, pz, vz].T
        cov_current = np.array([[1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])

        # Predict
        (x_prediction, cov_prediction) = predict(x_current, cov_current, F, G, Q)

        # Associate detection
        associated_detection = associate_NN(x_prediction,next_detections,H,cov_prediction,R)

        # Check if association was made
        if(np.count_nonzero(associated_detection) == 0):
            # Don't make an update; there was no associated detection
            x_updated = x_current
            cov_updated = cov_current
        else:
            # Make an update
            (x_updated, cov_updated) = update(x_prediction, cov_prediction,
                                          associated_detection, H, R)

        # 
        #temp = current_tracks[1]
        #a = np.matrix([1, 1, 1, 1])
        #temp2 = np.vstack((temp,a))
        
        # Extract positions 
        pos_updated = x_updated[0::2]
        updated_track_state = np.zeros((4,1))
        updated_track_state[0:3] = pos_updated
        updated_track_state[3] = f
        updated_track_state =  updated_track_state.T
        current_tracks[o] = np.vstack((current_tracks[o],updated_track_state))

print(current_tracks)