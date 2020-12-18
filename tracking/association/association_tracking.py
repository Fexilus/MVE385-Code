import itertools
from collections import defaultdict

import numpy as np
import h5py


def associate_NN(prediction, detections, cov_prediction, norm_innovation, dt):
    """Nearest neighbour association.
    norm_innovation should be a function measuring the normalized innovation
    for a specific filter and motion model.
    """
    innovation_dist = np.zeros((detections.shape[0],1))

    for i in range(detections.shape[0]):
        m = detections[i,:]
        innovation_dist[i] = norm_innovation(prediction, cov_prediction, m, dt) 

    closest_neighbour_ind = np.argmin(innovation_dist)
    closest_neighbour = detections[closest_neighbour_ind]
    
    # TODO: set a cut-off for how far away the next detection can be
    if (innovation_dist[closest_neighbour_ind] > 10):
        # Send back that there is no associated neighbour
        closest_neighbour = np.matrix([0,0,0])
    # TODO how to handle new detections
    
    return(closest_neighbour)


def count_times_unassociated(track):
    # Count how many of the last states that are equal, starting from the end
    count = 0
    pos = track[-1,0:3]
    previous_pos = track[-2,0:3]
    if((pos == previous_pos).all()):
        count = 2
    else:
        count = 1
    return count


def initialize_tracks(next_detections, associated_detections, cov_init, time,
                      id_generator):
    new_tracks = defaultdict(list)
    new_cov = defaultdict(lambda: cov_init)

    for d in next_detections:
        if(not any(np.array_equal(d, det) for _, det in associated_detections.items())):
            track_id = next(id_generator)

            track_d = np.zeros(4)
            track_d[0:3] = d
            track_d[-1] = time
            new_tracks[track_id] = track_d
            new_cov[track_id] = cov_init

    return(new_tracks,new_cov)


def track_all_objects(current_tracks, current_cov, next_detections,
                      terminated_tracks, time, dt, predict, update,
                      normalized_innovation, defaultStateVector):
    tracks_to_remove = [] # Saves id:s of tracks to terminate
    associated_detections = {} # Dict of detections associated to tracks
    
    for track_id in current_tracks:

        if len(current_tracks[track_id] <= 1):
            pos = current_tracks[track_id].flatten()
        else:
            pos = current_tracks[track_id][-1,:].flatten() # Last posistion of object o
        pos_t = pos[..., None]

        x_current = defaultStateVector(pos_t) #x = [px, vx, py, vy, pz, vz].T
        cov_o = current_cov[track_id]
        
        # Predict
        (x_prediction, cov_prediction) = predict(x_current, cov_o, dt)

        # Associate detection
        associated_detection = associate_NN(x_prediction, next_detections,
                                            cov_prediction,
                                            normalized_innovation, dt)
        
        # Check if association was made
        if(np.count_nonzero(associated_detection) == 0):
            # Don't make an update; there was no associated detection
            # TODO: add termination only after third frame without association
            terminated_tracks.append(current_tracks[track_id])
            tracks_to_remove.append(track_id)
            continue
        
        # Check if another track has already been associated to this detection
        if(any(np.array_equal(associated_detection, det) 
               for _, det in associated_detections.items())):
            # Terminate both tracks that were associated to this detection and continue to next track
            terminated_tracks.append(current_tracks[track_id])
            tracks_to_remove.append(track_id)
            
            for other_track_id, det in associated_detections.items():
                if np.array_equal(associated_detection, det):
                    other_track = current_tracks[other_track_id]
                    terminated_tracks.append(other_track)
                    tracks_to_remove.append(other_track_id)
            continue
        
        associated_detections[track_id] = associated_detection
        
        # Make an update
        (x_updated, cov_updated) = update(x_prediction, cov_prediction,
                                          associated_detection, dt)
        
        # Extract positions 
        pos_updated = x_updated[0,0::2]
        updated_track_state = np.zeros((1,4))
        updated_track_state[0,0:3] = pos_updated
        updated_track_state[0,3] = time
        current_tracks[track_id] = np.vstack((current_tracks[track_id],
                                              updated_track_state))
        current_cov[track_id] = cov_updated
        # Loop through objects
    
    # Remove all terminated tracks from current_tracks
    for index in sorted(tracks_to_remove, reverse = True):#i in range(len(tracks_to_remove)):
        if index in current_tracks:
            del current_tracks[index]
    
    return (current_tracks, terminated_tracks, associated_detections)


def add_initialized_to_current_tracks(initialized_tracks, current_tracks,
                                      initialized_cov, current_cov, time,
                                      id_generator):
    ind_remove = []
    for track_id, track in initialized_tracks.items():
        if(len(track)>=3): # It survived for three frames
            current_tracks[track_id] = track
            current_cov[track_id] = initialized_cov[track_id]
            # Save index for removal
            ind_remove.append(track_id)

    # Remove from list of initialized
    for ind in sorted(ind_remove, reverse=True):
        initialized_tracks.pop(ind)
        initialized_cov.pop(ind)

    return(initialized_tracks,current_tracks,initialized_cov,current_cov)


def track_multiple_objects(datafile, predict, update, normalized_innovation,
                           defaultStateVector):
    camera = h5py.File(datafile, 'r')

    # Get initial detections
    detections_0 = camera["Sequence"]["0"]["Detections"]
    detections_0 = np.asarray([list(det[0]) for det in list(detections_0)])

    # Initialize tracks at time 0
    # current_tracks is a list of the detection matrices
    cov_init = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])

    current_tracks = defaultdict(list)
    current_cov = defaultdict(lambda: cov_init)

    initialized_tracks = defaultdict(list)
    initialized_cov = defaultdict(lambda: cov_init)

    terminated_tracks = []

    id_generator = itertools.count(0)

    for i in range(detections_0.shape[0]):
        track_id = next(id_generator)

        track_i_pos = detections_0[i,:]
        track_i_pos = np.asmatrix(track_i_pos)
        track_i = np.zeros((1,4)) # Last value indicate in which frame this object had this position
        track_i[0,0:3] = track_i_pos

        current_tracks[track_id] = track_i

        # Initialize covariance matrices
        current_cov[track_id] = cov_init # FIXME: Unneccesary with defaultdict?

    # Initialize values (currently for the basic Kalman filter)
    #init_velocity = 2.0#*np.ones((3,1)) # dt*init_velocity is how far the object moved between two frames 
    timestamps = camera["Timestamp"][1:]
    last_timestamp = camera["Timestamp"][0]

    for frame, timestamp in zip(range(1, timestamps.shape[0] - 1), timestamps):

        next_detections = camera["Sequence"][str(frame)]["Detections"]
        next_detections = np.asarray([list(det[0]) for det
                                      in list(next_detections)])

        # Assume vector x = [px, vx, py, vy, pz, vz].T
        dt = timestamp - last_timestamp

        current_tracks, terminated_tracks, associated_detections = \
            track_all_objects(current_tracks, current_cov, next_detections,
                              terminated_tracks, timestamp, dt, predict,
                              update, normalized_innovation,
                              defaultStateVector)

        # Track the tracks under initialization and add to current tracks if they survive for
        # three consecutive frames.
        templist = []
        if len(initialized_tracks) > 0:
            initialized_tracks, _, associated_detections = \
                track_all_objects(initialized_tracks, initialized_cov,
                                  next_detections, templist, timestamp, dt,
                                  predict, update, normalized_innovation,
                                  defaultStateVector)

            initialized_tracks, current_tracks, initialized_cov, current_cov = \
                add_initialized_to_current_tracks(initialized_tracks,
                                                  current_tracks,
                                                  initialized_cov, current_cov,
                                                  timestamp, id_generator)


        # Check unassociated detections and add these to initializing
        new_tracks, new_cov = initialize_tracks(next_detections,
                                                associated_detections, cov_init,
                                                timestamp, id_generator)
        initialized_tracks.update(new_tracks)
        initialized_cov.update(new_cov)

        last_timestamp = timestamp

        yield current_tracks
