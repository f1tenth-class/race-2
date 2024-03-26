# hand-wavy velocity interpolation in lieu of proper MPC

# imports
import numpy as np

# constants
LOOKAHEAD_M = 2.0
VEL_MAX_MPS = 7.0
ACCEL_MAX_MPS2 = 2.0
DECCEL_MAX_MPS2 = -3.0

# helper to compute turn angle given a list of waypoints
def turn_angle(waypoints):
    # TODO
    return 0.0

# helper to compute minimum turn angle at which slip will occur
# at current velocity
def min_slip_angle(vel_curr_mps):
    # TODO
    return 0.25

# helper to extract waypoints within the lookahead distance
def lookahead_waypoints(waypoints):
    # TODO
    return []

# helper to see if car will slip within lookahead distance
# at current velocity
def needs_deccel(lookahead_waypoints, vel_curr_mps):
    # TODO
    return false, 0.0
