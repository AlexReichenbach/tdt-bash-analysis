import numpy as np
import matplotlib.pyplot as plt


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


def plot_heatmap(traces, *, window_left, dt, aspect=2, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    top, bottom = -0.5, traces.shape[0] - 0.5
    left, right = -window_left, traces.shape[1] * dt - window_left
    ax.imshow(
        traces,
        extent=(left, right, bottom, top),
        aspect=aspect,
    )
    ax.set_yticks(list(range(traces.shape[0])))
    ax.set_ylabel('trial num')
    ax.set_xlabel('time (s)')
    ax.axvline(0, color='red')
    return ax
