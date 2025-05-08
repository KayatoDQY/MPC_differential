#include <opencv2/opencv.hpp>

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

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <EllipseUtils.hpp>
#include <qp_new.hpp>
#include <deal_path.hpp>

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>

#include <mutex>
#include <EllipseUtils.hpp>
#include <qp_new.hpp>
#include <deal_path.hpp>
std::mutex map_mutex_;
std::mutex path_mutex_;

cv::Mat grid_map_image;
nav_msgs::PathConstPtr current_path;

constexpr unsigned short STATE_NUM = 3;
constexpr unsigned short CTRL_NUM = 2;
constexpr unsigned short MPC_WINDOW = 50;

#define PI 3.14159265359

struct MapMeta
{
    double resolution;
    double origin_x;
    double origin_y;
    int width;
    int height;
};
MapMeta map_meta_;

void occupancyGridCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg)
{
    std::lock_guard<std::mutex> map_lock(map_mutex_);

    int width = msg->info.width;
    int height = msg->info.height;
    map_meta_.resolution = msg->info.resolution;
    map_meta_.origin_x = msg->info.origin.position.x;
    map_meta_.origin_y = msg->info.origin.position.y;
    map_meta_.width = width;
    map_meta_.height = height;
    cv::Mat gray_image(height, width, CV_8UC1);
    for (size_t i = 0; i < msg->data.size(); ++i)
    {
        int8_t value = msg->data[i];
        if (value == -1)
        {
            gray_image.data[i] = 127;
        }
        else if (value == 0)
        {
            gray_image.data[i] = 255;
        }
        else if (value == 100)
        {
            gray_image.data[i] = 0;
        }
        else
        {
            gray_image.data[i] = 255 - static_cast<uchar>(value * 255 / 100);
        }
    }
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)); //(40/5-30/5)*2+1
    cv::dilate(gray_image, gray_image, kernel);

    gray_image.copyTo(grid_map_image);
}

void pathCallback(const nav_msgs::Path::ConstPtr &msg)
{
    std::lock_guard<std::mutex> path_lock(path_mutex_);
    current_path = msg;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "occupancy_grid_visualizer");
    ros::NodeHandle nh("~");

    ros::Subscriber grid_sub = nh.subscribe("/ipc/occupancy_grid", 1, occupancyGridCallback);
    ros::Subscriber path_sub = nh.subscribe("/Astar/Simplified_path", 1, pathCallback);
    ros::Publisher cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 100);
    ros::Publisher pre_path_pub = nh.advertise<nav_msgs::Path>("/pre_path", 100);

    cv::namedWindow("Occupancy Grid with Path", cv::WINDOW_AUTOSIZE);

    Eigen::Matrix<double, STATE_NUM, 1> xMax;
    Eigen::Matrix<double, STATE_NUM, 1> xMin;
    Eigen::Matrix<double, CTRL_NUM, 1> uMax;
    Eigen::Matrix<double, CTRL_NUM, 1> uMin;
    Eigen::DiagonalMatrix<double, STATE_NUM> Q;
    Eigen::DiagonalMatrix<double, CTRL_NUM> R;

    uMax << 0.3, PI / 3;
    uMin << -0.3, -PI / 3;
    xMax << OsqpEigen::INFTY, OsqpEigen::INFTY, PI;
    xMin << -OsqpEigen::INFTY, -OsqpEigen::INFTY, -PI;

    Q.diagonal() << 1, 1, 1;
    R.diagonal() << 1, 1;

    Eigen::Matrix<double, STATE_NUM, 1> x0;
    x0 << 0, 0, 0;
    Eigen::VectorXd out;
    out.resize(2);
    out << 0, 0;

    ros::Rate rate(10);
    while (ros::ok())
    {
        ros::spinOnce();
        cv::Mat color_image, map_image;
        {
            std::lock_guard<std::mutex> map_lock(map_mutex_);
            if (!grid_map_image.empty())
            {
                grid_map_image.copyTo(map_image);
                cv::cvtColor(grid_map_image, color_image, cv::COLOR_GRAY2BGR);
            }
        }
        nav_msgs::PathConstPtr local_path;
        {
            std::lock_guard<std::mutex> path_lock(path_mutex_);
            local_path = current_path;
        }
        if (local_path && !local_path->poses.empty())
        {
            cv::Scalar path_color(0, 0, 255);
            int thickness = 1;
            cv::Point prev_point;
            bool first = true;
            std::vector<Eigen::MatrixXd> A_all;
            std::vector<Eigen::VectorXd> b_all;
            for (const auto &pose : local_path->poses)
            {
                double x = pose.pose.position.x;
                double y = pose.pose.position.y;

                double px = (x - map_meta_.origin_x) / map_meta_.resolution;
                double py = (y - map_meta_.origin_y) / map_meta_.resolution;

                int img_x = static_cast<int>(px);
                int img_y = static_cast<int>(py);
                
                if (img_x >= 0 && img_x < map_meta_.width && img_y >= 0 && img_y < map_meta_.height)
                {
                    cv::Point current_point(img_x, img_y);
                    if (!first)
                    {
                        cv::line(color_image, prev_point, current_point, path_color, thickness);
                        EllipseParams ellipse = adjustEllipse(map_image, prev_point, current_point);
                        cv::ellipse(color_image, ellipse.center,
                                    cv::Size(ellipse.E(0, 0), ellipse.E(1, 1)),
                                    ellipse.angle, 0, 360, cv::Scalar(0, 255, 0), thickness);
                        Eigen::Matrix2d X0;
                        X0 << current_point.x, prev_point.x,
                            current_point.y, prev_point.y;
                        Eigen::Matrix2d X0_inv = X0.inverse();
                        Eigen::MatrixXd A;
                        Eigen::VectorXd b;
                        A = (Eigen::Vector2d(1, 1).transpose() * X0_inv).transpose();
                        double K = A.norm();
                        const double r = 10;
                        A.conservativeResize(Eigen::NoChange, A.cols() + 1);

                        A.col(A.cols() - 1) = (Eigen::Vector2d(-1, -1).transpose() * X0_inv).transpose();
                        b = Eigen::VectorXd::Constant(1, 1 + r * K);
                        b.conservativeResize(b.size() + 1);
                        b(b.size() - 1) = r * K - 1;
                        int iteration = 0;
                        const int max_iterations = 100;
                        while (iteration++ < max_iterations)
                        {
                            cv::Point2d closest_point = find_closest_obstacle_to_rotated_ellipse(
                                map_image, ellipse, (A.cols() > 0) ? &A : nullptr, (b.size() > 0) ? &b : nullptr);
                            if (closest_point.x < 0)
                                break;

                            ellipse = expand_ellipse_to_point(ellipse, closest_point);

                            LinearConstraint constraint = calculate_aj_bj(ellipse, closest_point);

                            if (A.cols() == 0)
                            {
                                A = constraint.a_j;
                                b = Eigen::VectorXd::Constant(1, constraint.b_j);
                            }
                            else
                            {
                                A.conservativeResize(Eigen::NoChange, A.cols() + 1);
                                A.col(A.cols() - 1) = constraint.a_j;

                                b.conservativeResize(b.size() + 1);
                                b(b.size() - 1) = constraint.b_j;
                            }
                            if (!has_obstacle_in_region(map_image, ellipse, A, b))
                                break;
                        }
                        int num_rows = A.rows();
                        int num_cols = A.cols();
                        A_all.push_back(A);
                        b_all.push_back(b);
                        cv::Scalar region_color(rand() % 256, rand() % 256, rand() % 256);
                        for (int y = 0; y < color_image.rows; ++y)
                        {
                            for (int x = 0; x < color_image.cols; ++x)
                            {
                                Eigen::Vector2d point(x, y);
                                cv::Vec3b pixel = color_image.at<cv::Vec3b>(y, x);
                                bool is_white = (pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255);

                                if (!is_white)
                                {
                                    continue;
                                }
                                bool satisfies_all = true;
                                for (int i = 0; i < A.cols(); ++i)
                                {
                                    double ax = A.col(i).dot(Eigen::Vector2d(x, y));
                                    if (ax >= b(i))
                                    {
                                        satisfies_all = false;
                                        break;
                                    }
                                }

                                // 如果原图像素是白色且满足约束，才覆盖颜色
                                if (satisfies_all)
                                {
                                    color_image.at<cv::Vec3b>(y, x) = cv::Vec3b(
                                        region_color[0], // B
                                        region_color[1], // G
                                        region_color[2]  // R
                                    );
                                }
                            }
                        }
                    }
                    else
                    {
                        first = false;
                    }

                    prev_point = current_point;
                }
            }
            
            DealPath deal_path(A_all, b_all, local_path);

            std::vector<Eigen::MatrixXd> A_mpc;
            std::vector<Eigen::VectorXd> b_mpc;
            std::vector<Eigen::Matrix<double, STATE_NUM, 1>> xref;
            double x_r, y_r, yaw_r;
// 初始化前一个点的坐标为车辆的当前位置或路径的初始前一个点
double prev_x = 0; // 假设current_x_是当前车辆x坐标
double prev_y = 0; // 假设current_y_是当前车辆y坐标
double prev_yaw = 0; // 假设current_yaw_是当前车辆偏航角

for (auto step = 0; step < MPC_WINDOW; step++)
{
    Eigen::MatrixXd A_r_0;
    Eigen::VectorXd b_r_0;
    Eigen::MatrixXd A_r;
    Eigen::VectorXd b_r;
    deal_path.find_step(step, A_r_0, b_r_0, x_r, y_r);

    // 计算当前点的yaw_r
    if (step == 0) {
        // 第一个点，使用初始prev_x和prev_y计算yaw_r
        yaw_r = atan2(y_r - prev_y, x_r - prev_x);
    } else if (step == MPC_WINDOW - 1) {
        // 最后一个点，使用前一个点的yaw_r
        yaw_r = prev_yaw;
    } else {
        // 中间点，使用前一个点坐标计算yaw_r
        yaw_r = atan2(y_r - prev_y, x_r - prev_x);
    }

    // 更新前一个点的坐标和yaw
    prev_x = x_r;
    prev_y = y_r;
    prev_yaw = yaw_r;

    // 以下为原有的矩阵处理逻辑
    A_r = (A_r_0 / map_meta_.resolution).transpose();                                                                 
    b_r = b_r_0 + A_r_0.transpose() * Eigen::Vector2d(map_meta_.origin_x, map_meta_.origin_y) / map_meta_.resolution;

    Eigen::MatrixXd A_new(A_r.rows(), 3);
    A_new.leftCols(2) = A_r;
    A_new.col(2).setZero();

    Eigen::MatrixXd yaw_constraints(2, 3);
    yaw_constraints << 0, 0, -1,
                       0, 0, 1;

    A_new.conservativeResize(A_new.rows() + 2, Eigen::NoChange);
    A_new.bottomRows(2) = yaw_constraints;

    b_r.conservativeResize(b_r.size() + 2);
    b_r.tail<2>() << M_PI, M_PI;

    Eigen::Matrix<double, STATE_NUM, 1> x_r_eigen;
    x_r_eigen << x_r, y_r, yaw_r;
    xref.push_back(x_r_eigen);
    A_mpc.push_back(A_new);
    b_mpc.push_back(b_r);
}
            Eigen::Matrix<double, STATE_NUM, 1> x_r_eigen;
            x_r_eigen << x_r, y_r, yaw_r;
            xref.push_back(x_r_eigen);
            MPC_problem<STATE_NUM, CTRL_NUM, MPC_WINDOW> MPC_Solver(Q, R, xMax, xMin, uMax, uMin, A_mpc, b_mpc);
            MPC_Solver.set_x_xref(x0, out, xref);
            try
            {
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
            catch (const std::exception &e)
            {
                // 处理求解失败的情况，发送停止cmd_vel
                geometry_msgs::Twist vel_msg;
                vel_msg.linear.x = 0.0;
                vel_msg.angular.z = 0.0;
                cmd_vel_pub.publish(vel_msg);
                std::cerr << "Error: " << e.what() << std::endl;
                continue;
            }
        }
        else
        {
            // 发送停止cmd_vel
            geometry_msgs::Twist vel_msg;
            vel_msg.linear.x = 0.0;
            vel_msg.angular.z = 0.0;
            cmd_vel_pub.publish(vel_msg);
        }
        if (!color_image.empty())
        {
            cv::resize(color_image, color_image, cv::Size(500, 500));
            cv::imshow("Occupancy Grid with Path", color_image);
            cv::waitKey(1);
        }
        rate.sleep();
    }

    cv::destroyAllWindows();
    return 0;
}
