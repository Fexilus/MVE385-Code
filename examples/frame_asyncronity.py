"""Show frame time non-correspondence between datasets."""
import h5py
import numpy as np
import matplotlib.pyplot as plt


camera0 = h5py.File("data/data_109.h5", 'r')
camera1 = h5py.File("data/data_130.h5", 'r')

min_num_frames = min(camera0["Timestamp"].size, camera1["Timestamp"].size)

cropped_times0 = np.asarray(camera0["Timestamp"])[0:min_num_frames]
cropped_times1 = np.asarray(camera1["Timestamp"])[0:min_num_frames]

time_step_frames0 = np.diff(np.asarray(camera0["Timestamp"]))
time_step_frames1 = np.diff(np.asarray(camera1["Timestamp"]))

average_step_size = np.median(np.concatenate((time_step_frames0,
                                              time_step_frames1)))

earliest_time = min(camera0["Timestamp"][0], camera1["Timestamp"][0])

expected_end_time = average_step_size * min_num_frames
expected_times = np.arange(0, expected_end_time, average_step_size)

norm_frame_times0 = cropped_times0 - expected_times - earliest_time
norm_frame_times1 = cropped_times1 - expected_times - earliest_time

plt.subplot()
plt.plot(norm_frame_times0)
plt.plot(norm_frame_times1)

plt.xlabel("Frame number in dataset")
plt.ylabel("Time difference compared to expected behavior (s)")

plt.show()
