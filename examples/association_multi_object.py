import numpy as np
import matplotlib.pyplot as plt
import h5py
from tracking.filter.basic import predict,update,createStateVector
from tracking.visualize.predictions import visualize_predictions,convert_to_image_space
from examples.association_basic import associate_NN

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
        updated_track_state[0,3] = f
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



# TODO
# 1. NN for multi-objects to create tracks T_i
# 2. Add termination critera according to article
# 3. Add initilizations for unassociated tracks

# WILL GENERATE: Tracks T_i with a set of tuples (x_k,y_k,z_k,t_k) indicating location ob object i at time t_k
# (In line with article implementation)

datafile = "tracking/data/data_109.h5"
camera = h5py.File(datafile, 'r')
nbr_of_frames = 500

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
current_tracks = []
current_cov = []
terminated_tracks = []
initialized_tracks = []
initialized_cov = []
for i in range(detections_0.shape[0]):
    track_i_pos = detections_0[i,:]
    track_i_pos = np.asmatrix(track_i_pos)
    track_i = np.zeros((1,4)) # Last value indicate in which frame this object had this position
    track_i[0,0:3] = track_i_pos
    current_tracks.append(track_i)

    # Initialize covariance matrices
    current_cov.append(cov_init)

# Initialize values
init_velocity = 2.0#*np.ones((3,1)) # dt*init_velocity is how far the object moved between two frames 
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
    if(len(current_tracks)==0):
        break

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
    
    current_tracks, terminated_tracks,associated_detections = track_all_objects(current_tracks,current_cov,init_velocity,F,G,Q,R,H,
                                                            next_detections,terminated_tracks,f)
    
    # Track the tracks under initialization and add to current tracks if they survival for
    # three consecutive frames. 
    templist = []
    if (len(initialized_tracks)>0):
        initialized_tracks,temp_terminated,associated_detections = track_all_objects(initialized_tracks,initialized_cov,init_velocity,F,G,Q,R,H,
                                                            next_detections,templist,f)
                        
        initialized_tracks,current_tracks,initialized_cov,current_cov = add_initialized_to_current_tracks(
                                            initialized_tracks,current_tracks,initialized_cov,current_cov,f)


    #TODO: Check unassociated tracks and mark these as initializing
    new_tracks,new_cov = initialize_tracks(next_detections,associated_detections,cov_init,f)
    initialized_tracks.extend(new_tracks)
    initialized_cov.extend(new_cov)
    
    # Remove all terminated tracks
    #for index in sorted(tracks_to_remove,reverse = True):#i in range(len(tracks_to_remove)):
    #    current_tracks.pop(index)
    #Loop through frames

# visalization
exit()
def visualize_track(track,camera):
    x = track[0:3]
    frame = int(track[-1])
    world2cam = np.asarray(camera['TMatrixWorldToCam'])
    cam2im = np.asarray(camera['ProjectionMatrix'])
    x_im_space = convert_to_image_space(x, world2cam, cam2im)
    
    plt.figure()
    image = camera['Sequence'][str(frame)]['Image']
    im_arr = np.zeros(image.shape)
    image.read_direct(im_arr)
    plt.imshow(im_arr, cmap='gray')
    # plot points
    plt.scatter(x_im_space[0], x_im_space[1],
                c = 'blue', label="Measurement")
    plt.show()

track = terminated_tracks[3]
for i in range(track.shape[0]):
    visualize_track(track[i,:],camera)