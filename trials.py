import numpy as np


def align_traces(time, trace, trials, *, window_left, window_right):
    dt = time[1] - time[0]
    window_left_idx = int(round(window_left / dt))
    window_right_idx = int(round(window_right / dt))
    window_size = window_left_idx + window_right_idx
    ntrials = len(trials)
    output = np.empty((ntrials, window_size), dtype=trace.dtype)
    for trial_num, timepoint in enumerate(trials):
        timepoint_idx = np.argmin(np.abs(time - timepoint))
        idx_start = timepoint_idx - window_left_idx
        idx_end = timepoint_idx + window_right_idx
        output[trial_num] = trace[idx_start:idx_end]
    return output