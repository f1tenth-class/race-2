# speed profile generation

# imports
import numpy as np

# generate np array with speed col at default max_speed
def generate_segments(raceline_csv, max_speed, segment_list):
    # read in raceline
    raceline = np.loadtxt(raceline_csv, delimiter=',', skiprows=1)

    # add column for speed, fill with max_speed
    raceline = np.append(raceline, np.full((raceline.shape[0], 1), max_speed), axis=1)

    # set speed for each segment (x, y, speed)
    for segment in segment_list:
        # find waypoint indices
        start = np.argmin(np.linalg.norm(raceline[:, 0:2] - segment[0:2], axis=1))
        end = np.argmin(np.linalg.norm(raceline[:, 0:2] - segment[2:4], axis=1))
        raceline[start:end, 3] = segment[2]
    return raceline

# change segment
def change_segment(raceline, segment, speed):
    raceline[segment[0]:segment[1], 3] = speed
    return raceline

# get speed at a point
def get_speed(raceline, point):
    # find closest waypoint
    dist = np.linalg.norm(raceline[:, 0:2] - point, axis=1)
    closest = np.argmin(dist)
    return raceline[closest, 3]
