"""Show frame time non-correspondence between datasets."""
import h5py
import numpy as np
import matplotlib.pyplot as plt


data_files = ["data/data_109.h5",
              "data/data_130.h5",
              "data/data_142.h5",
              "data/data_143.h5"]

cameras = [h5py.File(data_file, 'r') for data_file in data_files]

frame_counts = [camera["Timestamp"].size for camera in cameras]
min_num_frames = min(frame_counts)

cropped_timestamps = [np.asarray(camera["Timestamp"])[0:min_num_frames]
                      for camera in cameras]

time_steps = [np.diff(np.asarray(camera["Timestamp"])) for camera in cameras]

median_step_size = np.median(np.concatenate(time_steps))

earliest_time = min([camera["Timestamp"][0] for camera in cameras])

expected_end_time = median_step_size * min_num_frames
expected_times = np.arange(0, expected_end_time, median_step_size)

norm_frame_times = [cropped_timestamp - expected_times - earliest_time
                    for cropped_timestamp in cropped_timestamps]

plt.subplot()
for norm_frame_time, file_name in zip(norm_frame_times, data_files):
    plt.plot(norm_frame_time, label=file_name)

plt.hlines(median_step_size, 0, min_num_frames,
           linestyles="dashed", label="Median step size")

plt.xlabel("Frame number in dataset")
plt.ylabel("Time difference compared to expected behavior (s)")
plt.legend()

plt.show()
