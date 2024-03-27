# hand-wavy speed interpolation in lieu of proper MPC

# imports
import numpy as np

# constants
PI = np.pi

# parameters
LOOKAHEAD_M = 2.0
SPEED_MAX_MPS = 7.0
SPEED_MIN_MPS = 2.0
ACCEL_MAX_MPS2 = 2.0
ACCEL_MIN_MPS2 = -3.0
DT_S = 0.1 # 1 / frequency (Hz)
TURN_MAX_RAD = PI / 4.0

# TODO: switch from lists to numpy arrays for performance
# TODO: switch to dynamic lookahead distance based on speed

# helper to compute turn angle given a list of waypoints
def turn_angle(waypoints):
    # TODO: use more waypoints, maybe smooth
    x1, y1 = waypoints[0]
    x2, y2 = waypoints[1]
    angle = np.arctan2(y2 - y1, x2 - x1)
    return angle

# helper to compute distance between two waypoints
def distance(waypoint1, waypoint2):
    x1, y1 = waypoint1
    x2, y2 = waypoint2
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def lookahead_waypoints(pos_curr_m, waypoints):
    """
    Return the upcoming waypoints within the lookahead distance.
    """
    # TODO: implement path reconciliation to avoid parsing all waypoints
    # and to catch prolonged and hairpin turns.
    # parametrize on spline?
    lookahead_waypoints = []
    for i in range(len(waypoints) - 1):
        dist = distance(pos_curr_m, waypoints[i])
        angle = turn_angle([pos_curr_m, waypoints[i]])
        if dist < LOOKAHEAD_M and angle < abs(PI / 2.0):
            lookahead_waypoints.append(waypoints[i])
    return lookahead_waypoints

# helper to compute acceleration needed to achieve a target speed at a given distance
def compute_accel(speed_curr_mps, speed_target_mps, dist_m):
    accel_mps2 = (speed_target_mps**2 - speed_curr_mps**2) / (2 * dist_m)
    if accel_mps2 >= 0:
        return min(ACCEL_MAX_MPS2, accel_mps2)
    else:
        return max(ACCEL_MIN_MPS2, accel_mps2)

def speed_map(turn_angle_rad):
    """
    Return the estimated maximum speed that can be maintained at the given turn angle
    before the car will slip. Extremely hand-wavy, needs tuning.
    """
    # TODO
    return SPEED_MAX_MPS

def check_slip(lookahead_waypoints, speed_curr_mps, pos_curr_m):
    """
    Check if the car will slip within the lookahead distance if the current
    speed is maintained. Return the boolean and the decceleration if needed.
    """
    for i in range(len(lookahead_waypoints) - 1):
        angle = turn_angle(lookahead_waypoints[i:i+2])
        if speed_curr_mps > speed_map(angle):
            dist_to_slip_m = distance(pos_curr_m, lookahead_waypoints[i])
            deccel_mps2 = compute_accel(speed_curr_mps, speed_map(angle), dist_to_slip_m)
            return True, deccel_mps2
    return False, 0.0

def accelerate(speed_curr_mps, accel_mps2):
    """
    Return the new speed after applying the given acceleration for the time step.
    """
    speed_new_mps = speed_curr_mps + accel_mps2 * DT_S
    if accel_mps2 > 0:
        return min(SPEED_MAX_MPS, speed_new_mps)
    else:
        return max(SPEED_MIN_MPS, speed_new_mps)

def compute_speed(pos_curr_m, vel_curr, next_waypoint, lookahead_waypoints):
    """
    Compute the new speed based on the current position, speed, and upcoming waypoints.
    """
    speed_curr_mps = np.sqrt(vel_curr[0]**2 + vel_curr[1]**2)
    future_slip, deccel_mps2 = check_slip(lookahead_waypoints, speed_curr_mps, pos_curr_m)
    if future_slip:
        return accelerate(speed_curr_mps, deccel_mps2)
    else:
        speed_target_mps = speed_map(turn_angle([pos_curr_m, next_waypoint]))
        dist_to_target_m = distance(pos_curr_m, lookahead_waypoints[-1])
        accel_mps2 = compute_accel(vel_curr, speed_target_mps, dist_to_target_m)
        return accelerate(speed_curr_mps, accel_mps2)
