#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>
#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>
#include <mutex>
#include <queue>
#include <string.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <chrono>
#include <array>

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
geometry_msgs::PoseStamped original_pose;

constexpr unsigned short STATE_NUM = 3;
constexpr unsigned short CTRL_NUM = 2;
constexpr unsigned short MPC_WINDOW = 50;

std::vector<Eigen::Matrix<double, STATE_NUM, 1>> xref;
Eigen::Matrix<double, STATE_NUM, 1> x0;
double pitch = 0, roll = 0;

bool grid_map[100][100] = {false};
constexpr double resolution = 0.1;
constexpr unsigned short radius = 4;
constexpr unsigned short squared_radius = radius * radius;

std::mutex odom_mutex_, scan_mutex_, path_mutex_;
void scanCallback(const sensor_msgs::LaserScan::ConstPtr &scan_msg)
{
	scan_mutex_.lock();
	const int radius = 3;
	const int squared_radius = radius * radius;
	// 重置网格地图
	for (int i = 0; i < 100; ++i)
	{
		for (int j = 0; j < 100; ++j)
		{
			grid_map[i][j] = false;
		}
	}
	// 将激光雷达数据转换为网格地图
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
static void convertPathToXRef(const nav_msgs::Path &transformed_path)
{
	double V_MAX = 0.5;
	double CTRL_DT = 0.1;
	path_mutex_.lock();
	xref.clear();

	if (!transformed_path.poses.empty())
	{
		Eigen::Matrix<double, STATE_NUM, 1> last_state;
		last_state(0) = transformed_path.poses[0].pose.position.x;
		last_state(1) = transformed_path.poses[0].pose.position.y;
		last_state(2) = tf::getYaw(transformed_path.poses[0].pose.orientation);
		xref.push_back(last_state);

		for (size_t i = 1; i < transformed_path.poses.size(); ++i)
		{
			const auto &prev_pose = transformed_path.poses[i - 1];
			const auto &curr_pose = transformed_path.poses[i];

			const double x0 = prev_pose.pose.position.x;
			const double y0 = prev_pose.pose.position.y;
			const double x1 = curr_pose.pose.position.x;
			const double y1 = curr_pose.pose.position.y;
			const double yaw0 = tf::getYaw(prev_pose.pose.orientation);
			const double dx = x1 - x0;
			const double dy = y1 - y0;
			const double dist = hypot(dx, dy);
			const int interp_steps = std::max(1,
											  static_cast<int>(dist / (V_MAX * CTRL_DT)));

			for (int j = 1; j <= interp_steps; ++j)
			{
				const double ratio = static_cast<double>(j) / interp_steps;
				Eigen::Matrix<double, STATE_NUM, 1> state;

				state(0) = x0 + dx * ratio;
				state(1) = y0 + dy * ratio;

				state(2) = yaw0;
				xref.push_back(state);
				if (xref.size() >= MPC_WINDOW)
					break;
			}
			if (xref.size() >= MPC_WINDOW)
				break;
		}
	}

	const size_t current_size = xref.size();
	if (current_size <= MPC_WINDOW)
	{
		const auto &last = xref.back();
		xref.resize(MPC_WINDOW + 1, last);
	}
	else if (current_size > MPC_WINDOW)
	{
		xref.resize(MPC_WINDOW);
	}
	path_mutex_.unlock();
}
static bool isPathClear(const std::pair<int, int> &start, const std::pair<int, int> &end)
{
	int x0 = start.first, y0 = start.second;
	int x1 = end.first, y1 = end.second;

	int dx = abs(x1 - x0);
	int dy = abs(y1 - y0);
	int sx = x0 < x1 ? 1 : -1;
	int sy = y0 < y1 ? 1 : -1;
	int err = dx - dy;

	while (true)
	{
		if (grid_map[x0][y0])
			return false;

		if (x0 == x1 && y0 == y1)
			break;

		int e2 = 2 * err;
		if (e2 > -dy)
		{
			err -= dy;
			x0 += sx;
		}
		if (e2 < dx)
		{
			err += dx;
			y0 += sy;
		}
	}
	return true;
}
static std::vector<std::pair<int, int>> simplifyPath(const std::vector<std::pair<int, int>> &grid_path)
{
	std::vector<std::pair<int, int>> simplified;
	if (grid_path.empty())
		return simplified;

	size_t n = grid_path.size();
	size_t current = 0;
	simplified.push_back(grid_path[current]);

	while (current < n - 1)
	{
		size_t farthest = current + 1;
		for (size_t next = current + 1; next < n; ++next)
		{
			if (isPathClear(grid_path[current], grid_path[next]))
			{
				farthest = next;
			}
			else
			{
				break;
			}
		}
		if (farthest != current)
		{
			simplified.push_back(grid_path[farthest]);
			current = farthest;
		}
		else
		{
			simplified.push_back(grid_path[current + 1]);
			current++;
		}
	}

	return simplified;
}
static nav_msgs::Path convertPath(const std::vector<std::pair<int, int>> &grid_path)
{
	nav_msgs::Path map_path;
	map_path.header.stamp = ros::Time::now();
	map_path.header.frame_id = "livox_frame";

	if (grid_path.empty())
	{
		return map_path;
	}
	map_path.poses.reserve(grid_path.size());
	double last_valid_yaw = 0.0;
	for (size_t i = 0; i < grid_path.size(); ++i)
	{
		geometry_msgs::PoseStamped pose;
		pose.header = map_path.header;

		const auto &pt = grid_path[i];
		pose.pose.position.x = (pt.first - 50) * resolution;
		pose.pose.position.y = (pt.second - 50) * resolution;
		double yaw = 0.0;
		if (i < grid_path.size() - 1)
		{
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
				yaw = last_valid_yaw;
			}
		}
		else
		{
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
		tf2::Quaternion q;
		q.setRPY(0, 0, yaw);
		pose.pose.orientation = tf2::toMsg(q);

		map_path.poses.push_back(pose);
	}
	if (grid_path.size() == 1)
	{
		tf2::Quaternion q;
		q.setRPY(0, 0, 0);
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
		auto simplified_path = simplifyPath(aStar_path);
		simplified_path.push_back(std::make_pair(target_x, target_y));
		nav_msgs::Path transformed_path = convertPath(simplified_path);
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
		bool obs_mark = false;
		for (auto i = 49; i <= 51; i++)
		{
			for (auto j = 49; j <= 51; j++)
			{
				if (grid_map[i][j])
				{
					obs_mark = true;
					ROS_INFO("obstacle stop");
				}
			}
		}
		if (!obs_mark || roll <= PI / 6 && roll >= -PI / 6 && pitch <= PI / 6 && pitch >= -PI / 6)
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
