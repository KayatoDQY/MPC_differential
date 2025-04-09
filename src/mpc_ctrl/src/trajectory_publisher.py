#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import csv
import os
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from scipy.interpolate import CubicSpline

from scipy.interpolate import PchipInterpolator

def generate_spline_path(path, sampling_distance):
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
    total_distance = cumulative_distances[-1]
    num_samples = int(total_distance / sampling_distance)
    sampled_distances = np.linspace(0, total_distance, num_samples)
    interp_x = PchipInterpolator(cumulative_distances, path[:, 0])
    interp_y = PchipInterpolator(cumulative_distances, path[:, 1])
    sampled_x = interp_x(sampled_distances)
    sampled_y = interp_y(sampled_distances)
    
    return np.vstack((sampled_x, sampled_y)).T, sampled_distances

def load_csv_path(csv_file):
    csv_file = os.path.expanduser(csv_file)
    path = []
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2: continue
                x = float(row[0])
                y = float(row[1])
                path.append([x, y])
        return np.array(path)
    except Exception as e:
        rospy.logerr(f"加载CSV文件失败: {str(e)}")
        return None

def calculate_orientation(path, distances):
    dx = np.gradient(path[:, 0], distances)
    dy = np.gradient(path[:, 1], distances)
    angles = np.arctan2(dy, dx)
    return angles

def publish_trajectory(path_type='circle', radius=1, sampling_distance=1, csv_file='~/odom_data.csv'):
    rospy.init_node('custom_trajectory_publisher')
    pub = rospy.Publisher('/trajectory', Path, queue_size=10)
    
    if path_type == 'circle':
        path = generate_circle_path(radius=radius)
    elif path_type == 'eight':
        path = generate_eight_path(radius=radius)
    elif path_type == 'custom':
        path = load_csv_path(csv_file)
        if path is None or len(path) <= 2:
            rospy.logerr("无效的CSV轨迹数据!")
            return
    else:
        rospy.logerr("无效路径类型，支持: circle/eight/custom")
        return

    smooth_path, distances = generate_spline_path(path, sampling_distance)
    orientations = calculate_orientation(smooth_path, distances)

    path_msg = Path()
    path_msg.header.frame_id = 'map'
    
    for i, point in enumerate(smooth_path):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = point[0]
        pose.pose.position.y = point[1]
        pose.pose.position.z = 0.0
        
        yaw = orientations[i]
        quat = tf.transformations.quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        
        path_msg.poses.append(pose)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        path_msg.header.stamp = rospy.Time.now()
        pub.publish(path_msg)
        rate.sleep()

def generate_circle_path(radius=10, center=(0, 1), num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.vstack((x, y)).T

def generate_eight_path(radius=10, num_points=100):
    t = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.sin(t)
    y = radius * np.sin(2 * t)
    return np.vstack((x, y)).T

if __name__ == '__main__':
    try:
        publish_trajectory(
            path_type='custom', 
            sampling_distance=0.1*0.8,
            csv_file='~/rounded_quadrilateral_contour.csv'
        )
    except rospy.ROSInterruptException:
        pass
