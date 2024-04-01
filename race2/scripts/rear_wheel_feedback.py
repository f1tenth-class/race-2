#!/usr/bin/env python3

# rear wheel position based feedback
# implemented in https://arxiv.org/pdf/1604.07446.pdf

import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as R
import csv
from time import sleep
from std_msgs.msg import Float32MultiArray

PI = np.pi


class RearWheelFeedback(Node):
    """ 
    Rear wheel position based feedback.
    """
    def __init__(self):
        super().__init__('rear_wheel_feedback_node')
        
        self.vel = 4.0
        self.lookahead = 0.2
        self.k_te = 0.8
        self.k_e = 0.4
        self.omega_1_weight = 0.5
        self.heading_prev = None
        self.time_prev = None

        self.angular_velocity_past = 0.0
        self.angular_velocity_past_past = 0.0

        self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        self.waypoints_publisher = self.create_publisher(MarkerArray, '/rear_wheel_feedback/waypoints', 50)
        self.goalpoint_publisher = self.create_publisher(Marker, '/rear_wheel_feedback/goalpoint', 5)
        self.testpoint_publisher = self.create_publisher(MarkerArray, '/rear_wheel_feedback/testpoints', 10)
        self.future_pos_publisher = self.create_publisher(Marker, '/rear_wheel_feedback/future_pos', 5)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        self.map_to_car_rotation = None
        self.map_to_car_translation = None

        waypoints = self.load_waypoints("/home/empaul/ESE_6150/race2_ws/race2/waypoints/lobby_raceline_kappa4.csv")
        self.waypoints = waypoints[:, 1:3]
        self.params = waypoints[:, 4]
        self.publish_waypoints()

        # debug for rear-wheel-feedback
        self.debug_publisher = self.create_publisher(Float32MultiArray, '/debug_values', 10)

        # self.timer_period = 0.1  # seconds (10Hz)
        # self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.last_publish_time = self.get_clock().now()


        
    def load_waypoints(self, path):
        waypoints = np.loadtxt(path, delimiter=';')
        return waypoints

    def publish_waypoints(self):
        if len(self.waypoints) == 0:
            return
        
        markerArray = MarkerArray()
        for i, wp in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.pose.position.x = float(wp[0])
            marker.pose.position.y = float(wp[1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            markerArray.markers.append(marker)
        self.waypoints_publisher.publish(markerArray)

    def find_current_waypoint(self, current_pos, current_heading):
        euler_angles = current_heading.as_euler('zyx')
        future_pos = current_pos + self.lookahead * np.array([np.cos(euler_angles[0]), np.sin(euler_angles[0])])
        closest_wp = None
        min_dist = float('inf')
        for idx, wp in enumerate(self.waypoints):
            dist = np.linalg.norm(np.array(wp) - future_pos)
            if dist < min_dist:
                min_dist = dist
                closest_wp = wp
                min_idx = idx
        dist_to_curr_pos = np.linalg.norm(closest_wp - current_pos)
        if dist_to_curr_pos <= self.lookahead:
            two_wps = [self.waypoints[min_idx]]
            if min_idx+1 < len(self.waypoints):
                two_wps.append(self.waypoints[min_idx+1])
            else:
                two_wps.append(self.waypoints[0])
        else:
            two_wps = []
            if min_idx-1 >= 0:
                two_wps.append(self.waypoints[min_idx-1])
            else:
                two_wps.append(self.waypoints[-1])
            two_wps.append(self.waypoints[min_idx])
        self.publish_future_pos(future_pos)
        self.publish_testpoints(two_wps)
        return self.interpolate_waypoints(two_wps, current_pos), self.params[min_idx], two_wps, min_idx
    
    def interpolate_waypoints(self, two_wps, curr_pos):
        self.publish_testpoints(two_wps)
        wp_vec = two_wps[0] - two_wps[1]
        pos_vec = two_wps[0] - curr_pos
        alpha = np.arccos(np.dot(wp_vec, pos_vec) / (np.linalg.norm(wp_vec) * np.linalg.norm(pos_vec)))
        beta = np.pi - alpha
        a = np.linalg.norm(pos_vec) * np.cos(beta)
        b = np.linalg.norm(pos_vec) * np.sin(beta)
        if self.lookahead**2 - b**2 >= 0:
            c = np.sqrt(self.lookahead**2 - b**2) - a
        else:
            c = -a
        return two_wps[0] - c * wp_vec / np.linalg.norm(wp_vec)

    def publish_future_pos(self, future_pos):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.x = future_pos[0]
        marker.pose.position.y = future_pos[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        self.future_pos_publisher.publish(marker)

    def publish_testpoints(self, testpoints):
        markerArray = MarkerArray()
        for i, tp in enumerate(testpoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.pose.position.x = tp[0]
            marker.pose.position.y = tp[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.21
            marker.scale.y = 0.21
            marker.scale.z = 0.21
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            markerArray.markers.append(marker)
        self.testpoint_publisher.publish(markerArray)

    def publish_goalpoint(self, goalpoint):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.x = goalpoint[0]
        marker.pose.position.y = goalpoint[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.goalpoint_publisher.publish(marker)
    
    
    def pose_callback(self, pose_msg):
        current_pos = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        current_heading = R.from_quat(np.array([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]))
        
        if self.heading_prev is None:
            self.heading_prev = current_heading.as_euler('zyx')[0]
        current_waypoint, current_params, two_wps, min_idx = self.find_current_waypoint(current_pos, current_heading)
        self.publish_goalpoint(current_waypoint)

        self.map_to_car_translation = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z])
        self.map_to_car_rotation = R.from_quat([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w])

        current_waypoint_cf = (np.array([current_waypoint[0], current_waypoint[1], 0]) - self.map_to_car_translation)
        current_waypoint_cf = current_waypoint_cf @ self.map_to_car_rotation.as_matrix()

        current_pos = (np.array([current_pos[0], current_pos[1], 0]) - self.map_to_car_translation)
        current_pos = current_pos @ self.map_to_car_rotation.as_matrix()
        current_pos = np.array([current_pos[0], current_pos[1]])
        e = (current_pos - np.array([current_waypoint_cf[0], current_waypoint_cf[1]]))[1]

        track_heading = np.arctan2(two_wps[1][1] - two_wps[0][1], two_wps[1][0] - two_wps[0][0])
        tangent_path = two_wps[0][1] - two_wps[1][1], two_wps[0][0] - two_wps[1][0]
        
        theta_e = track_heading - current_heading.as_euler('zyx')[0]
        if theta_e  < -np.pi:
            theta_e += 2 * PI
        elif theta_e  > np.pi:
            theta_e -= 2 * PI
        
        kappa_s = self.params[min_idx]
        v_r = pose_msg.twist.twist.linear.x
        
        omega_1 = v_r * kappa_s * abs(np.cos(theta_e)) / (1 - kappa_s * e) * self.omega_1_weight
        omega_2 = (self.k_te * abs(v_r) * theta_e)
        omega_3 = -self.k_e * v_r * (np.sin(theta_e)/theta_e) * e
        self.omega = omega_1 + omega_2 + omega_3
        
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.stamp = self.get_clock().now().to_msg()
        self.drive_msg.header.frame_id = "ego_racecar/base_link"
        time_curr = self.get_clock().now().nanoseconds * 1e-9
        if self.time_prev is None:
            self.time_prev = time_curr
        dt = time_curr - self.time_prev
        self.time_prev = time_curr
        self.angular_velocities = [self.angular_velocity_past_past, self.angular_velocity_past, self.omega]
        
        self.pf_speed = np.linalg.norm(np.array([pose_msg.twist.twist.linear.x, pose_msg.twist.twist.linear.y]))
        self.drive_msg.drive.speed = self.interpolate_vel(self.pf_speed, self.vel, 1.0) # TODO: pull from params after updating csv

        debug_msg = Float32MultiArray()
        # self.debug_publisher.publish(debug_msg)

        self.angular_velocity_past = self.omega
        self.angular_velocity_past_past = self.angular_velocity_past_past
        
        current_time = self.get_clock().now()
        if self.last_publish_time is None or current_time - self.last_publish_time >= rclpy.time.Duration(seconds=0.1):
            self.drive_msg.drive.steering_angle = float(self.omega * 0.1 * 2 + self.heading_prev)
            self.drive_publisher.publish(self.drive_msg)
            self.last_publish_time = current_time
            debug_msg.data = [track_heading, current_heading.as_euler('zyx')[0], kappa_s, omega_1, omega_2, omega_3, self.drive_msg.drive.steering_angle, theta_e]
            self.debug_publisher.publish(debug_msg)


    def integrate_angular_velocity(angular_velocities, delta_t):
        """
        Approximate angular displacement using the trapezoidal rule.
        
        :param angular_velocities: List of angular velocities in radians per second.
        :param delta_t: Time interval between measurements in seconds.
        :return: Approximated angular displacement in radians.
        """
        return np.trapz(angular_velocities, dx=delta_t)

    def interpolate_vel(self, current_vel, seg_vel, acc=1.0, timestep=0.5):
        """
        param:
            current_vel : current velocity given by the particle filter
            seg_vel : velocity in current segment
        returns:
            command_vel : interpolated velocity
        """
        if current_vel < seg_vel: # accelerating
            return min(current_vel + acc * timestep, seg_vel)
        else: # deccelerating
            return max(current_vel - acc * timestep, seg_vel)


def main(args=None):
    rclpy.init(args=args)
    print("RearWheelFeedback Initialized")
    rear_wheel_feedback_node = RearWheelFeedback()
    rclpy.spin(rear_wheel_feedback_node)

    rear_wheel_feedback_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

