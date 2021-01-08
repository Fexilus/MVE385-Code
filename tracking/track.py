import numpy as np


def track(single_obj_measurements, timestamps, filter_impl, specific_default_state=None,
          specific_default_cov=None):
    """Track a single object with a given filter-motion implementation."""

    measurement0 = next(single_obj_measurements)

    if specific_default_state:
        state_current = specific_default_state(measurement0)
    else:
        state_current = filter_impl.default_state(measurement0)

    if specific_default_cov:
        cov_current = specific_default_cov
    else:
        cov_current = np.eye(filter_impl.NUM_STATES)

    last_timestamp = next(timestamps)

    object_track = [(filter_impl.state2position(state_current),
                     last_timestamp)]

    for measurement, timestamp in zip(single_obj_measurements, timestamps):

        dt = timestamp - last_timestamp

        state_prediction, cov_prediction = filter_impl.predict(state_current,
                                                               cov_current, dt)

        state_updated, cov_updated = filter_impl.update(state_prediction,
                                                        cov_prediction,
                                                        measurement, dt)

        # Set current to update
        state_current = state_updated
        cov_current = cov_updated

        object_track += [(filter_impl.state2position(state_current),
                          timestamp)]

        last_timestamp = timestamp

        yield (object_track, state_prediction, measurement)
