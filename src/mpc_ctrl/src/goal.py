#!/usr/bin/env python3
import rospy
import csv
import math
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class LoopTrajectoryTracker:
    def __init__(self):
        rospy.init_node('loop_trajectory_tracker')
        
        # 参数配置
        self.csv_file = rospy.get_param('~csv_file', '/home/nvidia/rounded_quadrilateral_contour.csv')
        self.goal_tolerance = rospy.get_param('~tolerance', 3)  # 目标容差
        self.lap_counter = 0                                      # 圈数计数器
        
        # 轨迹数据
        self.waypoints = self.load_waypoints()
        self.current_goal_index = -1  # 初始未设置
        
        # ROS通信
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.current_pose = None
        
        # 等待初始位置
        self.wait_for_initial_pose()
        self.find_initial_goal()
        self.control_loop()

    def load_waypoints(self):
        """加载环形轨迹点"""
        points = []
        try:
            with open(self.csv_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        points.append((float(row[0]), float(row[1])))
            rospy.loginfo(f"Loaded {len(points)} waypoints")
            return points
        except Exception as e:
            rospy.logerr(f"CSV加载失败: {str(e)}")
            return []

    def wait_for_initial_pose(self):
        """等待获取初始位置"""
        rate = rospy.Rate(1)
        while not rospy.is_shutdown() and self.current_pose is None:
            rospy.loginfo("等待初始位置数据...")
            rate.sleep()

    def find_initial_goal(self):
        """寻找距离最近的轨迹点作为起点"""
        if not self.waypoints:
            return
        
        min_dist = float('inf')
        start_idx = 0
        for idx, (x, y) in enumerate(self.waypoints):
            dx = x - self.current_pose['x']
            dy = y - self.current_pose['y']
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                start_idx = idx
        
        self.current_goal_index = start_idx
        rospy.loginfo(f"初始目标点设为第{start_idx}号点")

    def odom_callback(self, msg):
        """更新当前位置"""
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'yaw': euler_from_quaternion([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w])[2]
        }

    def publish_goal(self, x, y):
        """发布导航目标"""
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0  # 保持默认朝向
        self.goal_pub.publish(goal)
        rospy.loginfo(f"导航目标更新至: ({x:.2f}, {y:.2f})")

    def control_loop(self):
        """主控制循环"""
        rate = rospy.Rate(2)  # 2Hz控制频率
        
        # 发布初始目标
        x, y = self.waypoints[self.current_goal_index]
        self.publish_goal(x, y)
        
        while not rospy.is_shutdown():
            if not self.waypoints or self.current_pose is None:
                rate.sleep()
                continue
            
            # 计算当前位置到目标的距离
            target_x, target_y = self.waypoints[self.current_goal_index]
            dx = target_x - self.current_pose['x']
            dy = target_y - self.current_pose['y']
            distance = math.hypot(dx, dy)
            
            # 目标到达判定
            if distance <= self.goal_tolerance:
                # 更新到下一个目标点（环形索引）
                self.current_goal_index = (self.current_goal_index + 1) % len(self.waypoints)
                
                # 检测完成一圈
                if self.current_goal_index == 0:
                    self.lap_counter += 1
                    rospy.loginfo(f"完成第{self.lap_counter}圈!")
                
                # 发布新目标
                new_x, new_y = self.waypoints[self.current_goal_index]
                self.publish_goal(new_x, new_y)
            
            rate.sleep()

if __name__ == '__main__':
    try:
        LoopTrajectoryTracker()
    except rospy.ROSInterruptException:
        pass
