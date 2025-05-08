#ifndef DEAL_PATH_HPP

#define DEAL_PATH_HPP

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <nav_msgs/Path.h>
#include <cmath>

class DealPath {
public:
    DealPath(const std::vector<Eigen::MatrixXd> &A_all,
             const std::vector<Eigen::VectorXd> &b_all,
             const nav_msgs::PathConstPtr &local_path)
        : A_all_(A_all), b_all_(b_all), local_path_(local_path) {
            bool first = true;
            std::pair<double, double> prev_point;
            for(const auto &pose : local_path->poses) {
                if (first) {
                    prev_point = std::make_pair(pose.pose.position.x, pose.pose.position.y);
                    first = false;
                }
                else{
                    std::pair<double, double> current_point = std::make_pair(pose.pose.position.x, pose.pose.position.y);
                    double distance = std::sqrt(std::pow(current_point.first - prev_point.first, 2) +
                                                 std::pow(current_point.second - prev_point.second, 2));
                    path_lengths_.push_back(distance);
                }
            }

        }

    ~DealPath() {}

    void find_step(int step,Eigen::MatrixXd &A, Eigen::VectorXd &b,double &step_x, double &step_y) {
        double length = step * velocity_ * dt_;
        double total_length = 0.0;
        int step_index = 0;
        for(size_t i = 0; i < path_lengths_.size(); ++i) {
            total_length += path_lengths_[i];
            if(total_length >= length) {
                step_index = i;
                break;
            }
        }
        std::pair<double, double> start_point = std::make_pair(local_path_->poses[step_index].pose.position.x,
                                  local_path_->poses[step_index].pose.position.y);
        std::pair<double, double> end_point = std::make_pair(local_path_->poses[step_index + 1].pose.position.x,
                                  local_path_->poses[step_index + 1].pose.position.y);
        A=A_all_[step_index];
        b=b_all_[step_index];
        double step_length=path_lengths_[step_index];
        step_x = start_point.first+velocity_*dt_*(end_point.first-start_point.first)/step_length;
        step_y = start_point.second+velocity_*dt_*(end_point.second-start_point.second)/step_length;
    }
    
private:
    const std::vector<Eigen::MatrixXd> &A_all_;
    const std::vector<Eigen::VectorXd> &b_all_;
    const nav_msgs::PathConstPtr &local_path_;
    std::vector<double> path_lengths_;
    const double velocity_ = 0.8; // 速度
    const double dt_ = 0.2; // 时间间隔
};

#endif
