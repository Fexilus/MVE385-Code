"""A basic Kalman filter."""
import numpy as np


def predict(x_current, cov_current, F, G, Q):
    """Predict new states of object."""
    x_prediction = np.matmul(F, x_current.T).T # + B*u
    cov_prediction = np.matmul(np.matmul(F, cov_current), F.transpose()) \
                     + np.matmul(np.matmul(G, Q), G.transpose())

    return (x_prediction, cov_prediction)


def update(x_prediction, cov_prediction, measurement, H, R):
    """Update prediction based on measurement."""
    # The innovation
    residual = measurement - np.matmul(H,x_prediction.T).T # innovation
    residual_cov = R + np.matmul(H,np.matmul(cov_prediction,H.T)) # Should H be transposed?

    # Kalman gain
    W = np.matmul(cov_prediction, np.matmul(H.T, np.linalg.inv(residual_cov)))

    x_updated = x_prediction + np.matmul(W, residual.T).T
    cov_updated = cov_prediction - np.matmul(W, np.matmul(residual_cov,W.transpose()))

    return (x_updated, cov_updated)


def createStateVector(pos,vel):
    """Initialize a new state vector."""
    a = np.full((1,6),vel)
    a[0][0] = pos[0]
    a[0][2] = pos[1]
    a[0][4] = pos[2]

    return(a)


def track(single_obj_det, time_steps):
    """Track a single object with a basic Kalman filter."""
    # Initilize positions and velocities
    init_velocity = 2#*np.ones((3,1)) # dt*init_velocity is how far the object moved between two frames
    pos_init = np.asarray(single_obj_det[0,:])

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

    for measurement, dt in zip(single_obj_det, time_steps):
        # Assume vector x = [px, vx, py, vy, pz, vz].T
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

        (x_updated, cov_updated) = update(x_prediction, cov_prediction,
                                          measurement, H, R)

        # Set current to update
        x_current = x_updated
        cov_current = cov_updated

        yield (x_updated, x_prediction, measurement)
