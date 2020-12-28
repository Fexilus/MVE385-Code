import numpy as np
from scipy import integrate
from sklearn.linear_model import LinearRegression
import h5py

from tracking.filter.const_acceleration import predict

def compute_temporal(track1,track2):
    # compute the temporal component
    end1 = track1[-1,-1]
    start2 = track2[0,-1]
    if end1 < start2:
        val = 1
    else:
        val = 0
    return(val)

def g(x,sigma_g):
    if x**2 < sigma_g**2:
        val = np.exp(-0.5*x**2)
    else:
        val = 0
    return(val)

def f(x,mu,Sigma,sigma_g):
    g_in = np.matmul((x-mu).T,np.matmul(np.inverse(Sigma),(x-mu)))
    return(g(g_in,sigma_g))

def P_gn(x,mu,Sigma,sigma_g):
    num = f(x,mu,Sigma,sigma_g)
    den = integrate.quad(f,-np.inf,np.inf,args = (mu,Sigma,sigma_g))

    return(num/den)  

def forward_estimate(track,n,t):
    if track.shape[0] > n:
        n_track = track[-n:-1,:]
    else:
        n_track = track
    predictor = n_track[:,-1] # The times
    predictor = predictor[...,None] 
    print(predictor)
    response = n_track[:,0:3] # The positions
    reg = LinearRegression().fit(predictor, response)
    pred = reg.predict(np.array([[t]]))
    return(pred)

def backwards_estimate(track,n,t):
    if track.shape[0] > n:
        n_track = track[0:n,:]
        print(n_track)
    else:
        n_track = track
    predictor = n_track[:,-1] # The times
    predictor = predictor[...,None] 
    print(predictor.shape)
    response = n_track[:,0:3] # The positions
    reg = LinearRegression().fit(predictor, response)
    pred = reg.predict(np.array([[t]]))
    return(pred)


def compute_kinematic(track_i,track_j):
    n = 15
    end_i = track_i[-1,-1]
    start_j = track_j[0,-1]
    sigma_g = 1
    
    xb_j_start = backwards_estimate(track_j,n,start_j)
    xf_i = forward_estimate(track_i,n,start_j)
    xb_i = backwards_estimate(track_i,n,end_i)
    xb_j_end = forward_estimate(track_j,n,end_i)

    # TODO compute the covariances
    term1 = P_gn(xb_j_start,xf_i,Sigma_f,sigma_g)
    term2 = P_gn(xb_i,xb_j_end,Sigma_b,sigma_g)

    prob = term1*term2

    return(prob)

def compute_appearence(track1,track2):
    # Compute the appearence component/probability
    return(1)

def compute_pairwise_probability(track1,track2):
    temporal = compute_temporal(track1,track2)
    kinematic = compute_kinematic(track1,track2)
    appearence = compute_appearence(track1,track2)

    pw_prob = temporal*kinematic*appearence
    return(pw_prob)

def create_cost_matrix(all_tracks): 
    # for i in all tracks
        # for j in all tracks
            # compute_pairwise_probability(track_i,track_j)
            # save in position (i,j) in cost matrix (negative log)
    
    return(1)



##### For testing #########
def extract_positions():
    datafile = "tracking/data/data_109.h5"
    camera = h5py.File(datafile, 'r')

    detections = camera["Sequence"]["0"]["Detections"]
    detections = np.asarray([list(det[0]) for det in list(detections)])
    detections_init = detections[2,:]
    #print(detections_init)

    # COllect detections over a range of frames
    single_obj_dets = np.zeros((20,4))
    single_obj_dets[0,0:3] = detections_init
    single_obj_ind = np.zeros((20,1))
    single_obj_ind[0] = 2

    for i in range(19):
        next_dets = camera["Sequence"][str(i+1)]["Detections"]
        next_dets = np.asarray([list(det[0]) for det in list(next_dets)])

        dist = np.zeros(len(next_dets))
        for d in range(len(next_dets)): 
            # Compute norm
            # Find smallest
            dist[d] = np.linalg.norm(next_dets[d,:]-single_obj_dets[i,0:3])

        # FInd minimum distance
        min_ind = np.where(dist == np.amin(dist))
        single_obj_dets[i+1,0:3] = next_dets[min_ind,:]
        single_obj_dets[i+1,-1] = i+1
        single_obj_ind[i+1] = min_ind

    #print(single_obj_dets)
    #print(single_obj_ind)
    return(single_obj_dets)


temp_pos = extract_positions()
be = backwards_estimate(temp_pos,15,0)
print(be)