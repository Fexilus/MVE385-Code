## Simple Kalman filter implementation for one view ## 
import itertools

import h5py
import numpy as np
import open3d as o3d
import seaborn as sns
from PIL import Image
import matplotlib as plt


# Not working for vector input, only matrix...
def visualization2D(detections,camera,world2cam,cam2im):
    # Converting to camera view for visualization
    for f in range(len(detections)):
        detections_4 = np.ones((detections[f,:].shape[0],detections.shape[1]+1))
        detections_4[:,0:3] = detections[f,:]
        cams_det = (np.matmul(world2cam, detections_4.T)).T
        cams_det_4 = np.ones((cams_det.shape[0],4))
        cams_det_4[:,0:3] = cams_det[:,0:3]
        ims_det = (np.matmul(cam2im,cams_det_4.T)).T
        # Divide by z coordinate for some reason
        ims_det[:,0] = ims_det[:,0]/ims_det[:,2]
        ims_det[:,1] = ims_det[:,1]/ims_det[:,2]

        # Show image
        plt.pyplot.figure()
        image = camera['Sequence'][str(f)]['Image']
        imArr = np.zeros(image.shape)
        image.read_direct(imArr)
        plt.pyplot.imshow(imArr, cmap='gray')
        # plot points
        plt.pyplot.scatter(ims_det[2,0],ims_det[2,1])
        plt.pyplot.show()
        #plt.pyplot.close()

# Step one: make a Kalman filter for single sensor for single object
datafile = "tracking/data/data_109.h5"
camera = h5py.File(datafile, 'r')
single_obj_ind = [2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 1, 1, 2, 1]
single_obj_det = np.zeros((20,3))
world2cam = np.asarray(camera['TMatrixWorldToCam'])
cam2im = np.asarray(camera['ProjectionMatrix'])

# Extract single object detections
for i in range(20):
    detections = camera["Sequence"][str(i)]["Detections"]
    detections = np.asarray([list(det[0]) for det in list(detections)])
    single_obj_det[i,:] = detections[single_obj_ind[i],:]

#for i in range(20):
#    visualization2D(single_obj_det,camera,world2cam,cam2im)#RUns too many times!!

# Initilize positions and velocities
# TODO
dt = 0.1
velocity = 2#*np.ones((3,1)) # dt*velocity is how far the object moved between two frames
pos_init = np.asarray(single_obj_det[0,:])

# Assume vector x = [px, vx, py, vy, pz, vz].T
F = np.array([[1, dt, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, dt, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, dt],
              [0, 0, 0, 0, 0, 1]]) # The dynamics model
G = np.array([[dt**2/2, 0, 0],
              [dt, 0, 0],
              [0, dt**2/2, 0],
              [0, dt, 0],
              [0, 0, dt**2/2],
              [0, 0, dt]]) # To be multiplied with model noise v(x)  = [vx,vy,vz]
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0]]) # "Measurement model"

# TODO fill with accurate values
# Q = model noise covariance matrix
Q = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
# R = measurement noise covariance matrix
R = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])


def createStateVector(pos,vel):
    a = np.full((6,1),vel)
    a[0] = pos[0]
    a[2] = pos[1]
    a[4] = pos[2]
    return(a)

pos_t = pos_init[...,None]
x_current = createStateVector(pos_t,velocity)
cov_current = np.array([[1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]])

# Prediction #
x_prediction = np.matmul(F,x_current) # + B*u
cov_prediction = np.matmul(np.matmul(F,cov_current),F.transpose()) \
                    + np.matmul(np.matmul(G,Q),G.transpose())

# Update #
measurement = single_obj_det[1,:] 
measurement = np.matrix(measurement)

# The innovation
residual = measurement.T - np.matmul(H,x_prediction) # innovation
residual_cov = R + np.matmul(H,np.matmul(cov_prediction,H.T)) # Should H be transposed?

# Kalman gain
W = np.matmul(cov_prediction, np.matmul(H.T, np.linalg.inv(residual_cov)))

x_updated = x_prediction + np.matmul(W, residual)
cov_updated = cov_prediction - np.matmul(W, np.matmul(residual_cov,W.transpose()))

# Input single set of coordinatetes
def convertToImageSpace(coordinates):
    coord_4 = np.ones(4)
    coord_4[0:3] = coordinates
    cams_coord = (np.matmul(world2cam, coord_4.T)).T
    cams_coord_4 = np.ones(4)
    
    cams_coord_4[0:3] = cams_coord[0:3]
    ims_coord = (np.matmul(cam2im,cams_coord_4.T)).T
    # Divide by z coordinate for some reason
    ims_coord[0] = ims_coord[0]/ims_coord[2]
    ims_coord[1] = ims_coord[1]/ims_coord[2]
    ims_coord = ims_coord[0:2]
    return(ims_coord)

# Make a visualization of predictions, updates and measurements
def visualizePredictions(measurements,x_prediction,x_updated,camera):
    x_prediction = x_prediction[0::2]
    x_updated = x_updated[0::2]
    x_prediction = x_prediction.flatten()
    x_updated = x_updated.flatten()

    measurement_imSpace = convertToImageSpace(measurements)
    xPred_imSpace = convertToImageSpace(x_prediction)
    xUpd_imSpace = convertToImageSpace(x_updated)

    plt.pyplot.figure()
    image = camera['Sequence'][str(1)]['Image']
    imArr = np.zeros(image.shape)
    image.read_direct(imArr)
    plt.pyplot.imshow(imArr, cmap='gray')
    # plot points
    plt.pyplot.scatter(measurement_imSpace[0],measurement_imSpace[1],c = 'blue')
    plt.pyplot.scatter(xPred_imSpace[0],xPred_imSpace[1],c='red')
    plt.pyplot.scatter(xUpd_imSpace[0],xUpd_imSpace[1],c = 'green')
    plt.pyplot.show()
    #plt.pyplot.close()

visualizePredictions(measurement,x_prediction,x_updated,camera)