"""Show frame time non-correspondence between datasets."""
import h5py
import numpy as np
import matplotlib.pyplot as plt


camera0 = h5py.File("data/data_109.h5", 'r')
camera1 = h5py.File("data/data_130.h5", 'r')

min_num_frames = min(camera0["Timestamp"].size, camera1["Timestamp"].size)
cropped_times0 = np.asarray(camera0["Timestamp"])[0:min_num_frames]
cropped_times1 = np.asarray(camera1["Timestamp"])[0:min_num_frames]

time_diff = np.asarray(cropped_times0 - cropped_times1)

plt.plot(time_diff)

plt.xlabel("Frame number in dataset")
plt.ylabel("Time difference datasets (s)")

plt.show()
