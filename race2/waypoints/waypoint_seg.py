import numpy as np
import matplotlib.pyplot as plt

plot = True
original_waypoints = np.loadtxt('race2/waypoints/traj_raceline_0.5margin.csv', delimiter=',')
segment_points = [
    [-5.96, 0.515, 4.0, 1.5, 0.5],
    [0.654, -0.63, 1.5, 0.5, 0.1],
    [1.15, 3.25, 3.0, 1.5, 0.5],
    [-3.36, 2.59, 1.5, 0.5, 0.1]
    ]
segment_points = np.array(segment_points)

seg_start_idx = []
for i in range(segment_points.shape[0]):
    point = segment_points[i, :2]
    dist = np.linalg.norm(original_waypoints[:,:2] - point, axis=1)
    idx = np.argmin(dist)
    seg_start_idx.append(idx)
print(seg_start_idx)

seg_waypoints = np.zeros((original_waypoints.shape[0], 7))
seg_waypoints[:, :3] = original_waypoints[:, :3]
seg_start_idx.append(seg_start_idx[0])
for i in range(len(seg_start_idx)-1):
    if seg_start_idx[i] > seg_start_idx[i+1]:
        for j in range(seg_start_idx[i], original_waypoints.shape[0]):
            seg_waypoints[j, 3:6] = segment_points[i, 2:]
            seg_waypoints[j, 6] = i
        for j in range(seg_start_idx[i+1]):
            seg_waypoints[j, 3:6] = segment_points[0, 2:]
            seg_waypoints[j, 6] = i
    else:
        for j in range(seg_start_idx[i], seg_start_idx[i+1]):
            seg_waypoints[j, 3:6] = segment_points[i, 2:]
            seg_waypoints[j, 6] = i

np.savetxt('race2/waypoints/traj_raceline_0.5margin_seg.csv', seg_waypoints, delimiter=',', fmt='%.3f')

colors = ['ro', 'bo', 'go', 'yo']
for i in range(len(seg_start_idx)-1):
    plt.plot(seg_waypoints[np.where(seg_waypoints[:,6] == i),0], seg_waypoints[np.where(seg_waypoints[:,6] == i),1], colors[i])
    plt.plot(seg_waypoints[seg_start_idx[i], 0], seg_waypoints[seg_start_idx[i], 1], 'x')