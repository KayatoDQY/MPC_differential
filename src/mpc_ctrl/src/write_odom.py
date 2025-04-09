#!/usr/bin/env python3  
import rospy
from nav_msgs.msg import Odometry
import csv
import os
import time
import atexit

class OdomRecorder:
    def __init__(self):
        rospy.init_node('odom_recorder', anonymous=True)
        self.file_path = os.path.expanduser("~/odom_data.csv")
        self.odom_data = []
        self.sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        atexit.register(self.save_to_csv)
        rospy.loginfo("Odom Recorder已启动，数据将保存到: %s", self.file_path)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        self.odom_data.append([x, y])

    def save_to_csv(self):
        try:
            with open(self.file_path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(self.odom_data)
            rospy.loginfo("成功保存 %d 条数据到 %s", len(self.odom_data), self.file_path)
        except Exception as e:
            rospy.logerr("保存文件失败: %s", str(e))

if __name__ == '__main__':
    try:
        recorder = OdomRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
