# real time speed adjustment

# imports
import numpy as np

# constants
PI = np.pi

# parameters
ACCEL_MAX_MPS2 = 2.0
DECCEL_MAX_MPS2 = -2.0
DT_S = 0.1 # 1 / frequency (Hz)

# get max acceleration feasible for current speed
def accel_map(speed_mps):
    # TODO: interpolate from a lookup table
    return ACCEL_MAX_MPS2

# get max deceleration feasible for current speed
def deccel_map(speed_mps):
    # TODO: interpolate from a lookup table
    return DECCEL_MAX_MPS2

# apply acceleration to speed over time step
def accelerate(speed_curr_mps, accel_mps2):
    speed_new_mps = speed_curr_mps + accel_mps2 * DT_S
    return speed_new_mps

# if current speed is less than target speed, accelerate
def speed_control(speed_curr_mps, speed_target_mps):
    if speed_curr_mps < speed_target_mps:
        accel_mps2 = accel_map(speed_curr_mps)
        return accelerate(speed_curr_mps, accel_mps2)
    elif speed_curr_mps > speed_target_mps:
        deccel_mps2 = deccel_map(speed_curr_mps)
        return accelerate(speed_curr_mps, deccel_mps2)
    else:
        return speed_curr_mps
