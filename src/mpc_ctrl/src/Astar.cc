#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <queue>
#include <vector>
#include <cmath>
#include <algorithm>

class AStarPlanner
{
public:
    AStarPlanner() : nh_("~"), tf_listener_(tf_buffer_)
    {
        map_sub_ = nh_.subscribe("/ipc/occupancy_grid", 1, &AStarPlanner::mapCallback, this);
        goal_sub_ = nh_.subscribe("/move_base_simple/goal", 1, &AStarPlanner::goalCallback, this);
        path_pub_ = nh_.advertise<nav_msgs::Path>("path", 1);
        Simplified_path_pub_ = nh_.advertise<nav_msgs::Path>("Simplified_path", 1);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber map_sub_, goal_sub_;
    ros::Publisher path_pub_, Simplified_path_pub_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    nav_msgs::OccupancyGrid::ConstPtr current_map_;
    geometry_msgs::PoseStamped current_goal_;
    bool has_map_ = false;
    bool has_goal_ = false;
    std::mutex map_mutex_;
    
    struct Node
    {
        int x, y;
        double g, h;
        Node(int x_, int y_, double g_, double h_) : x(x_), y(y_), g(g_), h(h_) {}
        bool operator<(const Node &other) const { return (g + h) > (other.g + other.h); }
    };

    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        std::lock_guard<std::mutex> map_lock(map_mutex_);
        current_map_ = msg;
        has_map_ = true;
        tryPlan();
    }

    void goalCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        ROS_INFO("GET Goal");
        current_goal_ = *msg;
        has_goal_ = true;
    }

    void tryPlan()
    {

        if (!has_map_ || !has_goal_)
        {
            return;
        }
        geometry_msgs::PoseStamped transformed;

        geometry_msgs::TransformStamped transform =
            tf_buffer_.lookupTransform("livox_frame",
                                       current_goal_.header.frame_id,
                                       ros::Time(0),
                                       ros::Duration(0.1));

        tf2::doTransform(current_goal_, transformed, transform);
        int start_x, start_y;
        if (!worldToGrid(0, 0, start_x, start_y))
        {
            ROS_WARN("Start position out of map");
            return;
        }

        int goal_x, goal_y;
        if (!worldToGrid(transformed.pose.position.x,
                         transformed.pose.position.y, goal_x, goal_y))
        {
            ROS_WARN("Goal position out of map");
            return;
        }

        if (isObstacle(start_x, start_y) || isObstacle(goal_x, goal_y))
        {
            ROS_WARN("Start/Goal in obstacle");
            return;
        }
        auto path = aStarSearch(start_x, start_y, goal_x, goal_y);
        auto simplified_path = samplePath(path);
        publishPath(path);
        publishSimplifiedPath(simplified_path);
    }

    bool worldToGrid(double wx, double wy, int &gx, int &gy) const
    {
        const auto &info = current_map_->info;
        gx = static_cast<int>((wx - info.origin.position.x) / info.resolution);
        gy = static_cast<int>((wy - info.origin.position.y) / info.resolution);
        return (gx >= 0 && gx < info.width && gy >= 0 && gy < info.height);
    }

    void gridToWorld(int gx, int gy, double &wx, double &wy) const
    {
        const auto &info = current_map_->info;
        wx = info.origin.position.x + (gx + 0.5) * info.resolution;
        wy = info.origin.position.y + (gy + 0.5) * info.resolution;
    }

    bool isObstacle(int x, int y) const
    {
        return current_map_->data[y * current_map_->info.width + x] == 100;
    }

    std::vector<std::pair<int, int>> samplePath(const std::vector<std::pair<int, int>> &path)
    {
        std::vector<std::pair<int, int>> simplified;
        if (path.empty())
            return simplified;

        size_t n = path.size();
        simplified.push_back(path[0]);

        for (size_t i = 0; i < path.size();)
        {
            size_t furthest = i;
            for (size_t j = i+1; j <path.size();j++)
            {
                if (!isLineClear(path[i].first, path[i].second,
                                path[j].first, path[j].second))
                {
                    furthest = j-1;
                    break;
                }
            }

            if (furthest > i)
            {
                simplified.push_back(path[furthest]);
                i = furthest;
            }
            else
            {
                simplified.push_back(path[path.size()-1]);
                break;
            }
            
        }
        return simplified;
    }

    bool isLineClear(int x0, int y0, int x1, int y1) const
    {
        const int start_x = x0;
        const int start_y = y0;

        const int dx = abs(x1 - x0);
        const int dy = -abs(y1 - y0);
        const int sx = x0 < x1 ? 1 : -1;
        const int sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;

        while (true)
        {
            if (!(x0 == start_x && y0 == start_y))
            {
                if (x0 < 0 || x0 >= current_map_->info.width ||
                    y0 < 0 || y0 >= current_map_->info.height)
                {
                    return false;
                }
                if (isObstacle(x0, y0))
                {
                    return false;
                }
            }

            if (x0 == x1 && y0 == y1)
                break;

            int e2 = 2 * err;
            if (e2 >= dy)
            {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx)
            {
                err += dx;
                y0 += sy;
            }
        }
        return true;
    }

    std::vector<std::pair<int, int>> aStarSearch(int sx, int sy, int gx, int gy)
    {
        const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        const double cost[8] = {M_SQRT2, 1, M_SQRT2, 1, 1, M_SQRT2, 1, M_SQRT2};

        const int width = current_map_->info.width;
        const int height = current_map_->info.height;

        std::vector<std::vector<bool>> closed(height,
                                              std::vector<bool>(width, false));
        std::priority_queue<Node> open;
        std::vector<std::vector<double>> g_cost(height,
                                                std::vector<double>(width, INFINITY));
        std::vector<std::vector<std::pair<int, int>>> parent(
            height, std::vector<std::pair<int, int>>(width, {-1, -1}));

        auto heuristic = [&](int x, int y)
        {
            int dx = abs(x - gx), dy = abs(y - gy);
            return (dx + dy) + (M_SQRT2 - 2) * std::min(dx, dy);
        };

        open.emplace(sx, sy, 0, heuristic(sx, sy));
        g_cost[sy][sx] = 0;

        while (!open.empty())
        {
            Node current = open.top();
            open.pop();

            if (closed[current.y][current.x])
            {
                continue;
            }

            closed[current.y][current.x] = true;

            if (current.x == gx && current.y == gy)
            {
                std::vector<std::pair<int, int>> path;
                int cx = current.x, cy = current.y;
                while (cx != sx || cy != sy)
                {
                    path.emplace_back(cx, cy);
                    auto [px, py] = parent[cy][cx];
                    cx = px;
                    cy = py;
                }
                path.emplace_back(sx, sy);
                std::reverse(path.begin(), path.end());
                return path;
            }

            for (int i = 0; i < 8; ++i)
            {
                int nx = current.x + dx[i];
                int ny = current.y + dy[i];

                if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                {
                    continue;
                }
                if (isObstacle(nx, ny))
                {
                    continue;
                }

                double new_g = current.g + cost[i];
                if (new_g < g_cost[ny][nx])
                {
                    g_cost[ny][nx] = new_g;
                    parent[ny][nx] = {current.x, current.y};
                    open.emplace(nx, ny, new_g, heuristic(nx, ny));
                }
            }
        }
        ROS_WARN("NOT FOUND!");
        return {};
    }

    void publishPath(const std::vector<std::pair<int, int>> &path)
    {
        nav_msgs::Path path_msg;
        path_msg.header.frame_id = "livox_frame";
        path_msg.header.stamp = ros::Time::now();

        for (const auto &p : path)
        {
            geometry_msgs::PoseStamped pose;
            pose.header = path_msg.header;
            gridToWorld(p.first, p.second,
                        pose.pose.position.x, pose.pose.position.y);
            pose.pose.orientation.w = 1.0;
            path_msg.poses.push_back(pose);
        }
        path_pub_.publish(path_msg);
    }

    void publishSimplifiedPath(const std::vector<std::pair<int, int>> &path)
    {
        nav_msgs::Path path_msg;
        path_msg.header.frame_id = "livox_frame";
        path_msg.header.stamp = ros::Time::now();

        for (const auto &p : path)
        {
            geometry_msgs::PoseStamped pose;
            pose.header = path_msg.header;
            gridToWorld(p.first, p.second,
                        pose.pose.position.x, pose.pose.position.y);
            pose.pose.orientation.w = 1.0;
            path_msg.poses.push_back(pose);
        }
        Simplified_path_pub_.publish(path_msg);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "astar_planner");
    AStarPlanner planner;
    ros::spin();
    return 0;
}
