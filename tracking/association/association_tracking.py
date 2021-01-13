import itertools
from collections import defaultdict

import numpy as np


def associate_NN(prediction, measurements, cov_prediction, filter_impl, dt):
    """Nearest neighbour association.
    norm_innovation should be a function measuring the normalized innovation
    for a specific filter_impl and motion model.
    """
    innovation_dist = []

    for measurement in measurements:
        innovation_dist += [filter_impl.normalized_innovation(prediction,
                                                              cov_prediction,
                                                              measurement, dt)]

    closest_neighbour_ind = np.argmin(innovation_dist)
    closest_neighbour = measurements[closest_neighbour_ind]

    # TODO: This is just a quick fix, stringency should maybe be introduced
    mean_log_dist = np.mean(np.log(innovation_dist))
    min_log_dist = np.log(innovation_dist[closest_neighbour_ind])
    std_dev_log_dist = np.std(np.log(innovation_dist))

    #print(str((mean_log_dist - min_log_dist) / std_dev_log_dist))
    if mean_log_dist - min_log_dist < std_dev_log_dist:
        # Send back that there is no associated neighbour
        closest_neighbour = None
    # TODO how to handle new measurements

    return closest_neighbour


def count_times_unassociated(track):
    # Count how many of the last states that are equal, starting from the end
    count = 0
    pos = track[-1,0:3]
    previous_pos = track[-2,0:3]
    if (pos == previous_pos).all():
        count = 2
    else:
        count = 1
    return count


def initialize_tracks(next_measurements, associated_measurements, cov_init,
                      time, id_generator, filter_impl):
    new_tracks = defaultdict(list)
    new_states = {}
    new_cov = defaultdict(lambda: cov_init)

    # TODO: Should be able to save measurements more effectively
    for measurement in next_measurements:
        if not (any(np.array_equal(measurement, ass_measurement)
                for _, ass_measurement in associated_measurements.items())):
            track_id = next(id_generator)

            new_state = filter_impl.default_state(measurement)
            new_tracks[track_id] += [(filter_impl.state2position(new_state),
                                      time)]
            new_states[track_id] = new_state
            new_cov[track_id] = cov_init

    return (new_tracks, new_states, new_cov)


def track_all_objects(current_tracks, current_states, current_cov,
                      next_measurements, terminated_tracks, time, dt,
                      filter_impl):
    tracks_to_remove = [] # Saves id:s of tracks to terminate
    associated_measurements = {} # Dict of measurements associated to tracks

    for track_id in current_tracks:

        last_state = current_states[track_id]

        last_cov = current_cov[track_id]

        # Predict
        state_pred, cov_pred = filter_impl.predict(last_state, last_cov, dt)

        # Associate measurement
        associated_measurement = associate_NN(state_pred, next_measurements,
                                              cov_pred, filter_impl,
                                              dt)

        # Check if association was made
        if np.count_nonzero(associated_measurement) == 0:
            # Don't make an update; there was no associated measurement
            # TODO: add termination only after third frame without association
            terminated_tracks[track_id] = current_tracks[track_id]
            tracks_to_remove.append(track_id)

            continue

        # Check if other track has already been associated to this measurement
        for other_id, other_measurement in associated_measurements.items():
            if np.array_equal(associated_measurement, other_measurement):
                # Terminate both tracks that were associated to this
                # measurement and continue to next track
                terminated_tracks[track_id] = current_tracks[track_id]
                tracks_to_remove.append(track_id)

                terminated_tracks[track_id] = current_tracks[other_id]
                tracks_to_remove.append(other_id)

                continue

        associated_measurements[track_id] = associated_measurement

        # Make an update
        state_update, cov_update = filter_impl.update(state_pred, cov_pred,
                                                      associated_measurement,
                                                      dt)

        # Add new track state and update covariance
        current_tracks[track_id] += [(filter_impl.state2position(state_update),
                                      time)]
        current_states[track_id] = state_update
        current_cov[track_id] = cov_update

    # Remove all terminated tracks from current_tracks
    for index in sorted(tracks_to_remove, reverse = True):
        if index in current_tracks:
            del current_tracks[index]
            del current_states[index]
            del current_cov[index]

    return associated_measurements


def add_initialized_to_current_tracks(initialized_tracks, current_tracks,
                                      initialized_states, current_states,
                                      initialized_cov, current_cov):
    ind_remove = []

    for track_id, track in initialized_tracks.items():
        if len(track) >= 3: # It survived for three frames
            current_tracks[track_id] = track
            current_states[track_id] = initialized_states[track_id]
            current_cov[track_id] = initialized_cov[track_id]
            # Save index for removal
            ind_remove.append(track_id)

    # Remove from list of initialized
    for ind in sorted(ind_remove, reverse=True):
        initialized_tracks.pop(ind)
        initialized_states.pop(ind)
        initialized_cov.pop(ind)


def track_multiple_objects(measurements, timestamps, filter_impl):
    # Get initial measurements
    measurements_0_raw = next(measurements)

    # Format fist measurements and initial covariance matrix
    measurements_0 = [np.ndarray((1, filter_impl.NUM_MEASUREMENTS),
                                 buffer=np.asarray(measurement))
                    for measurement in measurements_0_raw]

    cov_init = np.eye(filter_impl.NUM_STATES)

    # Initialize tracks
    # Tracks are dictonaries with lists of two-tuples, where the first tuple
    # element is the state and the second is the timestamp.
    # The covariance is only saved for the latest time step.
    current_tracks = defaultdict(list)
    current_states = {}
    current_cov = defaultdict(lambda: cov_init)

    initialized_tracks = defaultdict(list)
    initialized_states = {}
    initialized_cov = defaultdict(lambda: cov_init)

    terminated_tracks = defaultdict(list)

    # Generator of unique ids
    id_generator = itertools.count(0)

    for measurement in measurements_0:
        track_id = next(id_generator)

        state = filter_impl.default_state(measurement)

        current_tracks[track_id] += [(filter_impl.state2position(state), 0)]
        current_states[track_id] = state

    last_timestamp = next(timestamps)

    for timestamp in timestamps:

        next_measurements_raw = next(measurements)
        next_measurements = [np.ndarray((1, filter_impl.NUM_MEASUREMENTS),
                                        buffer=measurement)
                             for measurement in next_measurements_raw]

        # Assume vector x = [px, vx, py, vy, pz, vz].T
        dt = timestamp - last_timestamp

        associated_measurements = track_all_objects(current_tracks,
                                                    current_states,
                                                    current_cov,
                                                    next_measurements,
                                                    terminated_tracks,
                                                    timestamp, dt, filter_impl)

        # Track the tracks under initialization and add to current tracks if
        # they survive for three consecutive frames.
        tempdict = defaultdict(list)
        if len(initialized_tracks) > 0:
            associated_measurements = track_all_objects(initialized_tracks,
                                                        initialized_states,
                                                        initialized_cov,
                                                        next_measurements,
                                                        tempdict, timestamp,
                                                        dt, filter_impl)

            add_initialized_to_current_tracks(initialized_tracks,
                                              current_tracks,
                                              initialized_states,
                                              current_states, initialized_cov,
                                              current_cov)


        # Check unassociated measurements and add these to initializing
        new_tracks, new_states, new_cov = initialize_tracks(next_measurements,
                                                            associated_measurements,
                                                            cov_init, timestamp,
                                                            id_generator,
                                                            filter_impl)
        initialized_tracks.update(new_tracks)
        initialized_states.update(new_states)
        initialized_cov.update(new_cov)

        last_timestamp = timestamp

        yield current_tracks, terminated_tracks
