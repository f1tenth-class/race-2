import numpy as np
import matplotlib.pyplot as plt
import cv2

plot = False
original_waypoints = np.loadtxt('race1_1_0.7.csv', delimiter=',')
lobby_map = cv2.imread('../map/race_1.pgm', cv2.IMREAD_GRAYSCALE)

segment_points = [
    # x, y, vel, lookahead, p, d
    [-5.96, 0.515, 3.0, 0.5, 0.3, 0.1],
    [-4.5, -0.3, 5.0, 1.0, 0.3, 0.1],
    [0.654, -0.63, 5.0, 0.8, 0.3, 0.1],
    [4.0, 2.0, 2.0, 0.5, 0.3, 0.1], # haripin 1
    [2.0, 4.0, 3.0, 0.5, 0.3, 0.1],
    [0.74, 2.84, 5.0, 1.5, 0.3, 0.1],
    [-3.36, 2.59, 2.0, 0.5, 0.3, 0.1] # hairpin 2
    ]
segment_points = np.array(segment_points)
seg_start_idx = []
for i in range(segment_points.shape[0]):
    point = segment_points[i, :2]
    dist = np.linalg.norm(original_waypoints[:,:2] - point, axis=1)
    idx = np.argmin(dist)
    seg_start_idx.append(idx)
print(seg_start_idx)

# print(original_waypoints)
seg_waypoints = np.zeros((original_waypoints.shape[0], 7))
seg_waypoints[:, :3] = original_waypoints[:, [0, 1, 4]]
seg_start_idx.append(seg_start_idx[0])
for i in range(len(seg_start_idx)-1):
    if seg_start_idx[i] > seg_start_idx[i+1]:
        for j in range(seg_start_idx[i], original_waypoints.shape[0]):
            seg_waypoints[j, 3:6] = segment_points[i, 3:]
            seg_waypoints[j, 6] = i
        for j in range(seg_start_idx[i+1]):
            seg_waypoints[j, 3:6] = segment_points[0, 3:]
            seg_waypoints[j, 6] = i
    else:
        for j in range(seg_start_idx[i], seg_start_idx[i+1]):
            seg_waypoints[j, 3:6] = segment_points[i, 3:]
            seg_waypoints[j, 6] = i


np.savetxt('race1_0.7_seg.csv', seg_waypoints, delimiter=',', fmt='%.3f')

if plot:
    # fig, ax1 = plt.subplots()
    # colors = ['ro', 'bo', 'go', 'yo', 'co', 'mo']
    # for i in range(len(seg_start_idx)-1):
    #     ax1.plot(seg_waypoints[np.where(seg_waypoints[:,6] == i),0], seg_waypoints[np.where(seg_waypoints[:,6] == i),1], colors[i%4])
    #     ax1.text(seg_waypoints[seg_start_idx[i], 0], seg_waypoints[seg_start_idx[i], 1], str(i), fontweight='bold')
    # ax1.axis('equal')
    resolution = 0.05
    origin = [-8.99, -1.15]

    blackpts = np.argwhere(lobby_map <= 40).astype(np.float32)
    blackpts[:, 0] = (lobby_map.shape[0] - blackpts[:, 0]) * resolution + origin[1]
    blackpts[:, 1] = blackpts[:, 1] * resolution + origin[0]
    colors = ['ro', 'bo', 'go', 'yo']
    plt.plot(blackpts[:,1], blackpts[:,0], 'k.')
    for i in range(len(seg_start_idx)-1):
        plt.plot(seg_waypoints[np.where(seg_waypoints[:,7] == i),0], seg_waypoints[np.where(seg_waypoints[:,7] == i),1], colors[i%4])
        plt.text(seg_waypoints[seg_start_idx[i], 0], seg_waypoints[seg_start_idx[i], 1], str(i), fontweight='bold')
    
    plt.show()

