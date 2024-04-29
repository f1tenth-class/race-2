import numpy as np
import matplotlib.pyplot as plt
import cv2


original_waypoints = np.loadtxt('./waypoints/race3_rl_3.csv', delimiter=',')
lobby_map = cv2.imread('./map/race3_2.pgm', cv2.IMREAD_GRAYSCALE)


print(original_waypoints[0])

resolution = 0.05
origin = [-10.3, -1.55]

# segment_points = [
#     # x, y, vel, lookahead, p, d
#     [-6.0, 0.7, 1.0, 1.5, 0.35, 0.01],
#     [-4.5, -0.3, 1.0, 2.0, 0.2, 0.01],
#     [0.654, -0.63, 0.9, 1.5, 0.35, 0.01],
#     [4.0, 2.0, 0.9, 1.0, 0.3, 0.01], # haripin 1
#     [2.0, 4.0, 0.7, 1.0, 0.2, 0.01],
#     [-0.5, 2.2, 0.9, 1.0, 0.1, 0.01],
#     [-2.7, 2.59, 0.7, 1.5, 0.1, 0.01], # hairpin 2
#     [-4.4, 2.5, 0.8, 1.5, 0.35, 0.01]
# ]
segment_points = [
    [-0.48, -0.07, 1.0, 2.0, 0.3, 0.01], # S1
    [3.56, -0.25, 0.9, 1.2, 0.4, 0.01], # T1cd 
    [7.1, 2.57, 0.9, 2.0, 0.3, 0.01], # S2 
    [7.12, 7.49, 0.8, 1.0, 0.3, 0.01], # T2-2
    [5.63, 9.21, 0.9, 1.5, 0.2, 0.01], # T2-2
    [4.27, 7.86, 1.0, 2.0, 0.3, 0.01], # S3 
    [-0.5, 3.63, 0.7,1.5, 0.3, 0.02], #T3-1
    [-1.69, 1.46, 0.8,1.5, 0.2, 0.00] #T3-2
]
segment_points = np.array(segment_points)
print(segment_points.shape)

blackpts = np.argwhere(lobby_map <= 40).astype(np.float32)
blackpts[:, 0] = (lobby_map.shape[0] - blackpts[:, 0]) * resolution + origin[1]
blackpts[:, 1] = blackpts[:, 1] * resolution + origin[0]
plt.plot(blackpts[:,1], blackpts[:,0], 'k.')
plt.plot(original_waypoints[:,0], original_waypoints[:,1], 'ro')
plt.plot(segment_points[:,0], segment_points[:,1], 'bo')
plt.axis('equal')
# plt.show()

plt.plot(original_waypoints[:,0], original_waypoints[:,1], 'ro')
plt.plot(segment_points[:,0], segment_points[:,1], 'bo')
plt.axis('equal')
# plt.show()


seg_start_idx = []
for i in range(segment_points.shape[0]):
    point = segment_points[i, :2]
    dist = np.linalg.norm(original_waypoints[:,:2] - point, axis=1)
    idx = np.argmin(dist)
    seg_start_idx.append(idx)
print(seg_start_idx)


# print(original_waypoints)
seg_waypoints = np.zeros((original_waypoints.shape[0], 9))
seg_waypoints[:, :3] = original_waypoints[:, [0, 1, 2]]
seg_waypoints[:, -1] = original_waypoints[:, -1]
seg_start_idx.append(seg_start_idx[0])
for i in range(len(seg_start_idx)-1):
    if seg_start_idx[i] > seg_start_idx[i+1]:
        for j in range(seg_start_idx[i], original_waypoints.shape[0]):
            seg_waypoints[j, 3:7] = segment_points[i, 2:]
            seg_waypoints[j, 7] = i
        for j in range(seg_start_idx[i+1]):
            seg_waypoints[j, 3:7] = segment_points[0, 2:]
            seg_waypoints[j, 7] = i
    else:
        for j in range(seg_start_idx[i], seg_start_idx[i+1]):
            seg_waypoints[j, 3:7] = segment_points[i, 2:]
            seg_waypoints[j, 7] = i
            
np.savetxt('race3_seg_2.csv', seg_waypoints, delimiter=',', fmt='%.3f')

blackpts = np.argwhere(lobby_map <= 40).astype(np.float32)
blackpts[:, 0] = (lobby_map.shape[0] - blackpts[:, 0]) * resolution + origin[1]
blackpts[:, 1] = blackpts[:, 1] * resolution + origin[0]
colors = ['ro', 'bo', 'go', 'yo']
plt.plot(blackpts[:,1], blackpts[:,0], 'k.')
for i in range(len(seg_start_idx)-1):
    plt.plot(seg_waypoints[np.where(seg_waypoints[:,-1] == i),0], seg_waypoints[np.where(seg_waypoints[:,-1] == i),1], colors[i%4])
    plt.text(seg_waypoints[seg_start_idx[i], 0], seg_waypoints[seg_start_idx[i], 1], str(i), fontweight='bold')

# plt.show()

blackpts = np.argwhere(lobby_map <= 40).astype(np.float32)
blackpts[:, 0] = (lobby_map.shape[0] - blackpts[:, 0]) * resolution + origin[1]
blackpts[:, 1] = blackpts[:, 1] * resolution + origin[0]
colors = ['ro', 'bo', 'go', 'yo']
plt.plot(blackpts[:,1], blackpts[:,0], 'k.')


velocities = seg_waypoints[:,2]
# gloabl_v_min = velocities.min()
# global_v_max = velocities.max()

# set_v_min = 2.0
# set_v_max = 5.0

# velocities = (velocities - gloabl_v_min) / (global_v_max - gloabl_v_min) * (set_v_max - set_v_min) + set_v_min

plt.scatter(seg_waypoints[:,0], seg_waypoints[:,1], c=velocities, cmap='RdYlGn')
for i in range(len(seg_waypoints)):
    if i % 10 == 0:
        plt.text(seg_waypoints[i, 0], seg_waypoints[i, 1], str(i), font='bold')
plt.colorbar()
plt.axis('equal')
# plt.show()