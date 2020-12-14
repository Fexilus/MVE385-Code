"""Visualize predictions in camera space"""
import numpy as np
import matplotlib.pyplot as plt

def convert_to_image_space(coordinates, world2cam, cam2im):
    """Input single set of coordinatetes"""
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


def visualize_predictions(measurements, x_prediction, x_updated, camera, 
                          frame):
    """Make a visualization of predictions, updates and measurements"""
    world2cam = np.asarray(camera['TMatrixWorldToCam'])
    cam2im = np.asarray(camera['ProjectionMatrix'])

    measurement_im_space = convert_to_image_space(measurements, world2cam,
                                                  cam2im)
    x_pred_im_space = convert_to_image_space(x_prediction, world2cam, cam2im)
    x_upd_im_space = convert_to_image_space(x_updated, world2cam, cam2im)

    plt.figure()
    image = camera['Sequence'][str(frame)]['Image']
    im_arr = np.zeros(image.shape)
    image.read_direct(im_arr)
    plt.imshow(im_arr, cmap='gray')
    # plot points
    plt.scatter(measurement_im_space[0], measurement_im_space[1],
                c = 'blue', label="Measurement")
    plt.scatter(x_pred_im_space[0], x_pred_im_space[1],
                c='red', label="Prediction")
    plt.scatter(x_upd_im_space[0], x_upd_im_space[1],
                c = 'green', label="A posteriori estimate")

    plt.legend()
    plt.show()
    #plt.close()
