#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>
#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>
#include <mutex>
#include <queue>
#include <algorithm>
#include <climits>
#include <cmath>
#include <chrono>

#include <ros/ros.h>
#include <tf/tf.h>
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/PoseStamped.h>

#include "qp.hpp"

#define PI 3.14159265359
ros::Publisher odom_path_pub, pre_path_pub, cmd_vel_pub, transformed_path_pub;

ros::Subscriber odom_sub, refer_path_sub;
nav_msgs::Path odom_path;
std::vector<geometry_msgs::PoseStamped> original_path;

constexpr unsigned short STATE_NUM = 3;
constexpr unsigned short CTRL_NUM = 2;
constexpr unsigned short MPC_WINDOW = 50;

std::vector<Eigen::Matrix<double, STATE_NUM, 1>> xref;
Eigen::Matrix<double, STATE_NUM, 1> x0;
double pitch = 0, roll = 0;

bool grid_map[100][100] = {false};

const double resolution = 0.1;
std::mutex odom_mutex_, scan_mutex_, path_mutex_;
geometry_msgs::PoseStamped original_pose;
void scanCallback(const sensor_msgs::LaserScan::ConstPtr &scan_msg)
{
	scan_mutex_.lock();
	const int radius = 3;
	const int squared_radius = radius * radius;
    //重置网格地图
	for (int i = 0; i < 100; ++i)	
	{
		for (int j = 0; j < 100; ++j)
		{
			grid_map[i][j] = false;
		}
	}
	//将激光雷达数据转换为网格地图
	for (size_t i = 0; i < scan_msg->ranges.size(); ++i)
	{
		double angle = scan_msg->angle_min + i * scan_msg->angle_increment;
		double range = scan_msg->ranges[i];

		if (range < scan_msg->range_max)
		{
			double x = range * cos(angle);
			double y = range * sin(angle);

			int center_x = static_cast<int>(x / resolution) + 50;
			int center_y = static_cast<int>(y / resolution) + 50;

			for (int dx = -radius; dx <= radius; ++dx)
			{
				for (int dy = -radius; dy <= radius; ++dy)
				{
					if (dx * dx + dy * dy > squared_radius)
						continue;

					int new_x = center_x + dx;
					int new_y = center_y + dy;
					if (new_x >= 0 && new_x < 100 && new_y >= 0 && new_y < 100)
					{
						grid_map[new_x][new_y] = true;
					}
				}
			}
		}
	}
	scan_mutex_.unlock();
}
struct Node
{
	int x, y;
	int g, h;

	bool operator<(const Node &other) const
	{
		return (g + h) > (other.g + other.h); // 小顶堆
	}
};
std::pair<int, int> adjustTarget(int tx, int ty)
{
	const int start_x = 50, start_y = 50;
	int dx = abs(tx - start_x);
	int dy = abs(ty - start_y);
	int step_x = (tx > start_x) ? 1 : -1;
	int step_y = (ty > start_y) ? 1 : -1;
	int err = dx - dy;

	int x = start_x, y = start_y;
	std::pair<int, int> last_valid = {-1, -1};

	while (true)
	{
		if (x >= 0 && x < 100 && y >= 0 && y < 100)
		{
			if (!grid_map[x][y])
			{
				last_valid = {x, y};
			}
			else
			{
				break;
			}
		}
		else
		{
			break;
		}

		if (x == tx && y == ty)
			break;

		int e2 = 2 * err;
		if (e2 > -dy)
		{
			err -= dy;
			x += step_x;
		}
		if (e2 < dx)
		{
			err += dx;
			y += step_y;
		}
	}

	return last_valid;
}
std::vector<std::pair<int, int>> aStar(int target_x, int target_y)
{
	const int start_x = 50, start_y = 50;
	int adjusted_x = target_x, adjusted_y = target_y;
	if (target_x < 0 || target_x >= 100 || target_y < 0 || target_y >= 100)
	{
		auto adjusted_target = adjustTarget(target_x, target_y);
		if (adjusted_target.first == -1)
		{
			ROS_WARN("No valid target found!");
			return {};
		}
		adjusted_x = adjusted_target.first;
		adjusted_y = adjusted_target.second;
	}
	else if (grid_map[adjusted_x][adjusted_y])
	{	
		ROS_WARN("Target is blocked!");
		return {};
	}

	if (grid_map[start_x][start_y])
	{
		ROS_WARN("Start position is blocked!");
		return {};
	}
	const int dirs[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
	const int costs[8] = {10, 10, 10, 10, 14, 14, 14, 14};
	int gScore[100][100];
	std::fill(&gScore[0][0], &gScore[0][0] + 100 * 100, INT_MAX);
	gScore[start_x][start_y] = 0;

	std::priority_queue<Node> open;
	open.push({start_x, start_y, 0, 0});

	std::pair<int, int> cameFrom[100][100];
	bool closed[100][100] = {false};

	while (!open.empty())
	{
		Node current = open.top();
		open.pop();

		int x = current.x;
		int y = current.y;

		if (closed[x][y])
		{
			continue;
		}
		closed[x][y] = true;

		if (x == adjusted_x && y == adjusted_y)
		{
			std::vector<std::pair<int, int>> path;
			while (x != start_x || y != start_y)
			{
				path.emplace_back(x, y);
				auto parent = cameFrom[x][y];
				x = parent.first;
				y = parent.second;
			}
			path.emplace_back(start_x, start_y);
			reverse(path.begin(), path.end());
			return path;
		}

		for (int i = 0; i < 8; ++i)
		{
			int nx = x + dirs[i][0];
			int ny = y + dirs[i][1];

			if (nx < 0 || nx >= 100 || ny < 0 || ny >= 100)
				continue;
			if (grid_map[nx][ny])
				continue;

			int new_g = current.g + costs[i];
			if (new_g < gScore[nx][ny])
			{
				cameFrom[nx][ny] = {x, y};
				gScore[nx][ny] = new_g;
				int dx = abs(nx - adjusted_x);
				int dy = abs(ny - adjusted_y);
				int h = 10 * (dx + dy) - 6 * std::min(dx, dy);
				open.push({nx, ny, new_g, h});
			}
		}
	}
	ROS_WARN("No path found!");
	return {};
}

static int findClosestPathIndex(const geometry_msgs::Pose &odom_pose)
{
	double min_distance = std::numeric_limits<double>::max();
	int closest_index = 0;
	for (size_t i = 0; i < original_path.size(); ++i)
	{
		double dx = original_path[i].pose.position.x - odom_pose.position.x;
		double dy = original_path[i].pose.position.y - odom_pose.position.y;
		double distance = dx * dx + dy * dy;

		if (distance < min_distance)
		{
			min_distance = distance;
			closest_index = i;
		}
	}
	if (closest_index > original_path.size() - 20)
	{
		closest_index = 2;
	}
	return closest_index;
}
static void pathCallback(const nav_msgs::Path::ConstPtr &path_msg)
{
	original_path = path_msg->poses;
}
static void move_goal_pathCallback(const geometry_msgs::PoseStamped::ConstPtr &goal_msg)
{
	original_pose = *goal_msg;
}
void convertPathToXRef(const nav_msgs::Path &transformed_path)
{
	path_mutex_.lock();
	xref.clear();
	for (const auto &pose : transformed_path.poses)
	{
		Eigen::Matrix<double, STATE_NUM, 1> state;
		state(0) = pose.pose.position.x;
		state(1) = pose.pose.position.y;
		state(2) = tf::getYaw(pose.pose.orientation);
		xref.push_back(state);
	}
	if (xref.size() <= MPC_WINDOW)
	{

		const Eigen::Matrix<double, STATE_NUM, 1> &last_state = xref.back();
		const size_t needed = MPC_WINDOW - xref.size();
		for (size_t i = 0; i <= needed; ++i)
		{
			xref.push_back(last_state);
		}
	}
	path_mutex_.unlock();
}
nav_msgs::Path convertPath(const std::vector<std::pair<int, int>> &grid_path)
{
	nav_msgs::Path map_path;
	map_path.header.stamp = ros::Time::now();
	map_path.header.frame_id = "livox_frame";

	if (grid_path.empty())
		return map_path;
	map_path.poses.reserve(grid_path.size());
	double last_valid_yaw = 0.0;

	for (size_t i = 0; i < grid_path.size(); ++i)
	{
		geometry_msgs::PoseStamped pose;
		pose.header = map_path.header;

		// 坐标转换
		const auto &pt = grid_path[i];
		pose.pose.position.x = (pt.first - 50) * resolution;
		pose.pose.position.y = (pt.second - 50) * resolution;

		// 偏航角计算逻辑
		double yaw = 0.0;
		if (i < grid_path.size() - 1)
		{
			// 计算当前点到下个点的方向
			const auto &next_pt = grid_path[i + 1];
			const double dx = (next_pt.first - pt.first) * resolution;
			const double dy = (next_pt.second - pt.second) * resolution;

			if (dx != 0 || dy != 0)
			{
				yaw = atan2(dy, dx);
				last_valid_yaw = yaw;
			}
			else
			{
				yaw = last_valid_yaw; // 相同点保持先前角度
			}
		}
		else
		{
			// 最后一个点使用倒数第二个点的角度
			if (grid_path.size() >= 2)
			{
				const auto &prev_pt = grid_path[i - 1];
				const double dx = (pt.first - prev_pt.first) * resolution;
				const double dy = (pt.second - prev_pt.second) * resolution;
				if (dx != 0 || dy != 0)
				{
					yaw = atan2(dy, dx);
				}
			}
		}

		// 角度转四元数
		tf2::Quaternion q;
		q.setRPY(0, 0, yaw);
		pose.pose.orientation = tf2::toMsg(q);

		map_path.poses.push_back(pose);
	}

	// 处理单点路径的特殊情况
	if (grid_path.size() == 1)
	{
		tf2::Quaternion q;
		q.setRPY(0, 0, 0); // 默认朝东
		map_path.poses[0].pose.orientation = tf2::toMsg(q);
	}

	return map_path;
}
void odomCallback(const nav_msgs::Odometry::ConstPtr &odom_msg, tf2_ros::Buffer &tf_buffer)
{
	odom_mutex_.lock();
	try
	{

		tf2::Quaternion quat;
		quat.setX(odom_msg->pose.pose.orientation.x);
		quat.setY(odom_msg->pose.pose.orientation.y);
		quat.setZ(odom_msg->pose.pose.orientation.z);
		quat.setW(odom_msg->pose.pose.orientation.w);

		quat.normalize();
		tf2::Matrix3x3 mat(quat);
		double yaw;
		mat.getRPY(roll, pitch, yaw);

		auto transform = tf_buffer.lookupTransform(
			odom_msg->child_frame_id,
			odom_msg->header.frame_id,
			ros::Time(0),
			ros::Duration(0.1));

		geometry_msgs::PoseStamped transformed_pose;

		tf2::doTransform(original_pose, transformed_pose, transform);
		int target_x = static_cast<int>(transformed_pose.pose.position.x / resolution) + 50;
		int target_y = static_cast<int>(transformed_pose.pose.position.y / resolution) + 50;
		ROS_INFO("Target: (%d, %d)", target_x, target_y);
		auto aStar_path = aStar(target_x, target_y);
		if (aStar_path.empty())
		{
			ROS_WARN("A* failed to find path!");
			odom_mutex_.unlock();
			return;
		}
		nav_msgs::Path transformed_path = convertPath(aStar_path);
		convertPathToXRef(transformed_path);
		transformed_path_pub.publish(transformed_path);
	}
	catch (tf2::TransformException &ex)
	{
		ROS_WARN("Transform unavailable: %s", ex.what());
	}

	geometry_msgs::PoseStamped odom_pose;
	odom_pose.header = odom_msg->header;
	odom_pose.pose = odom_msg->pose.pose;
	odom_path.poses.push_back(odom_pose);
	odom_path.header.frame_id = "map";
	odom_path.header.stamp = ros::Time::now();
	odom_path_pub.publish(odom_path);
	odom_mutex_.unlock();
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "mpc_node");
	ros::NodeHandle n;
	tf2_ros::Buffer tf_buffer;
	tf2_ros::TransformListener tf_listener(tf_buffer);

	cmd_vel_pub = n.advertise<geometry_msgs::Twist>("/cmd_vel", 100);
	transformed_path_pub = n.advertise<nav_msgs::Path>("/transformed_path", 100);
	odom_path_pub = n.advertise<nav_msgs::Path>("/odom_path", 100);
	pre_path_pub = n.advertise<nav_msgs::Path>("/pre_path", 100);
	ros::Subscriber scan_sub = n.subscribe("/scan", 1, scanCallback);
	ros::Subscriber move_sub = n.subscribe("/move_base_simple/goal", 1, move_goal_pathCallback);
	refer_path_sub = n.subscribe<nav_msgs::Path>("/trajectory", 100, pathCallback);
	odom_sub = n.subscribe<nav_msgs::Odometry>(
		"/odom", 100, boost::bind(odomCallback, _1, boost::ref(tf_buffer)));

	Eigen::Matrix<double, STATE_NUM, 1> xMax;
	Eigen::Matrix<double, STATE_NUM, 1> xMin;
	Eigen::Matrix<double, CTRL_NUM, 1> uMax;
	Eigen::Matrix<double, CTRL_NUM, 1> uMin;
	Eigen::DiagonalMatrix<double, STATE_NUM> Q;
	Eigen::DiagonalMatrix<double, CTRL_NUM> R;

	const double MAX_ANGULAR_Z = PI / 2;
	uMax << 0.8, PI / 8;
	uMin << -0.8, -PI / 8;

	xMax << OsqpEigen::INFTY, OsqpEigen::INFTY, PI;
	xMin << -OsqpEigen::INFTY, -OsqpEigen::INFTY, -PI;

	Q.diagonal() << 1, 1, 1;
	R.diagonal() << 1, 1;
	x0 << 0, 0, 0;
	Eigen::VectorXd out;
	out.resize(2);
	out << 0, 0;
	ros::Rate loop_rate(10);

	const double Kp = 0.5;
	while (ros::ok())
	{
		if (roll <= PI / 6 && roll >= -PI / 6 && pitch <= PI / 6 && pitch >= -PI / 6)
		{
			ROS_INFO("Path size: %zu", xref.size());
			if (xref.size() > 0)
			{

				double yaw_error = xref.at(0)[2];
				if (yaw_error < PI / 3 && yaw_error > -PI / 3)
				{
					MPC_problem<STATE_NUM, CTRL_NUM, MPC_WINDOW> MPC_Solver(Q, R, xMax, xMin, uMax, uMin);
					MPC_Solver.set_x_xref(x0, out, xref); // out0 has no use
					out = MPC_Solver.Solver();
					geometry_msgs::Twist vel_msg;
					vel_msg.linear.x = out(STATE_NUM * (MPC_WINDOW + 1));
					vel_msg.angular.z = out(STATE_NUM * (MPC_WINDOW + 1) + 1);

					nav_msgs::Path pre_path;
					for (auto index = 0; index < MPC_WINDOW; index++)
					{
						geometry_msgs::PoseStamped pre_pose;
						pre_pose.pose.position.x = out(index * STATE_NUM);
						pre_pose.pose.position.y = out(index * STATE_NUM + 1);
						pre_path.poses.push_back(pre_pose);
					}
					pre_path.header.frame_id = "livox_frame";
					pre_path.header.stamp = ros::Time::now();
					pre_path_pub.publish(pre_path);
					cmd_vel_pub.publish(vel_msg);
				}
				else
				{
					double angular_z = Kp * yaw_error;
					angular_z = std::clamp(angular_z, -MAX_ANGULAR_Z, MAX_ANGULAR_Z);

					geometry_msgs::Twist vel_msg;
					vel_msg.linear.x = 0.0;
					vel_msg.angular.z = angular_z;
					cmd_vel_pub.publish(vel_msg);
				}
			}
		}
		else
		{
			geometry_msgs::Twist vel_msg;
			vel_msg.linear.x = 0.0;
			vel_msg.angular.z = 0.0;
			cmd_vel_pub.publish(vel_msg);
		}
		loop_rate.sleep();
		ros::spinOnce();
	}
}
