import numpy as np
from tracking.filter.basic import predict,update,createStateVector

def associate_NN(prediction,detections,H,cov_prediction,R):
    # Nearest neighbour association
    innovation_dist = np.zeros((detections.shape[0],1))

    for i in range(detections.shape[0]):
        m = detections[i,:]
        innovation = m.T - np.matmul(H,prediction.T)
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

def initialize_tracks(next_detections,associated_detections,cov_init,f):
    new_tracks = []
    new_cov = []
    for d in next_detections:
        if(not any(np.array_equal(d, ind) for ind in associated_detections)):
            track_d = np.zeros(4)
            track_d[0:3] = d
            track_d[-1] = f
            new_tracks.append(track_d)
            new_cov.append(cov_init)
    return(new_tracks,new_cov)

def track_all_objects(current_tracks,current_cov,init_velocity,F,G,Q,R,H,next_detections,terminated_tracks,frame):
    tracks_to_remove = [] # Saves indices of tracks to terminate
    associated_detections = [] # Saves list of all detections that has been associated to a track
    
    for o in range(len(current_tracks)):

        if len(current_tracks[o]<=1):
            pos = current_tracks[o].flatten()
        else:
            pos = current_tracks[o][-1,:].flatten() # Last posistion of object o
        pos_t = pos[..., None]

        x_current = createStateVector(pos_t, init_velocity) #x = [px, vx, py, vy, pz, vz].T
        cov_o = current_cov[o]
        
        # Predict
        (x_prediction, cov_prediction) = predict(x_current, cov_o, F, G, Q)

        # Associate detection
        associated_detection = associate_NN(x_prediction,next_detections,H,cov_prediction,R)
        
        # Check if association was made
        if(np.count_nonzero(associated_detection) == 0):
            # Don't make an update; there was no associated detection
            # TODO: add termination only after third frame without association
            o_track = current_tracks[o] 
            terminated_tracks.append(o_track)
            tracks_to_remove.append(o)
            continue
        
        # Check if another track has already been associated to this detection
        if(any(np.array_equal(associated_detection, ind) for ind in associated_detections)):
            # Terminate both tracks that were associated to this detection and continue to next track
            o_track = current_tracks[o] 
            terminated_tracks.append(o_track)
            tracks_to_remove.append(o)
            
            other_track_ind = [np.array_equal(associated_detection,i) for i in associated_detections].index(True)
            other_track = current_tracks[other_track_ind]
            terminated_tracks.append(other_track)
            tracks_to_remove.append(other_track_ind)
            continue
        
        associated_detections.append(associated_detection)
        
        # Make an update
        (x_updated, cov_updated) = update(x_prediction, cov_prediction,
                                          associated_detection, H, R)
        
        # Extract positions 
        pos_updated = x_updated[0,0::2]
        updated_track_state = np.zeros((1,4))
        updated_track_state[0,0:3] = pos_updated
        updated_track_state[0,3] = frame
        current_tracks[o] = np.vstack((current_tracks[o],updated_track_state))
        current_cov[o] = cov_updated
        # Loop through objects
    
    for index in sorted(tracks_to_remove,reverse = True):#i in range(len(tracks_to_remove)):
        current_tracks.pop(index)
    return(current_tracks,terminated_tracks,associated_detections)

def add_initialized_to_current_tracks(initialized_tracks,current_tracks,initialized_cov,current_cov,frame):
    ind_remove = []
    for t in range(len(initialized_tracks)):
        track = initialized_tracks[t]

        if(len(track)>=3): # It survived for three frames
            current_tracks.append(track)
            current_cov.append(initialized_cov[t])
            # Save index for removal
            ind_remove.append(t)

    # Remove from list of initialized
    for ind in sorted(ind_remove,reverse=True):
        initialized_tracks.pop(ind)
        initialized_cov.pop(ind)
    return(initialized_tracks,current_tracks,initialized_cov,current_cov)