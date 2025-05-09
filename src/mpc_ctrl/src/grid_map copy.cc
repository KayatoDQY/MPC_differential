#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Point32.h>
#include <opencv2/opencv.hpp>
#include <deque>
#include <unordered_set>

class LaserToOccupancyGrid
{
public:
    LaserToOccupancyGrid() : nh_("~")
    {
        // 参数初始化
        nh_.param("resolution", resolution_, 0.05f);
        nh_.param("grid_width", grid_width_, 40.0f);
        nh_.param("grid_height", grid_height_, 40.0f);
        nh_.param("origin_x", origin_x_, -20.0f);
        nh_.param("origin_y", origin_y_, -20.0f);
        nh_.param("inflation_radius", inflation_radius_, 0.2f);
        nh_.param("crop_x_min", crop_x_min_, -5.0f);
        nh_.param("crop_y_min", crop_y_min_, -5.0f);
        nh_.param("crop_x_max", crop_x_max_, 5.0f);
        nh_.param("crop_y_max", crop_y_max_, 5.0f);
        nh_.param("radius", radius_, 0.2f);

        grid_cols_ = static_cast<int>(grid_width_ / resolution_);
        grid_rows_ = static_cast<int>(grid_height_ / resolution_);
        inflation_pixels_ = static_cast<int>(inflation_radius_ / resolution_);

        // 订阅者和发布者
        scan_sub_ = nh_.subscribe("/scan", 1, &LaserToOccupancyGrid::scanCallback, this);
        grid_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("occupancy_grid", 1);
    }

private:
    struct GridCell
    {
        int x;
        int y;
        bool operator==(const GridCell &other) const
        {
            return x == other.x && y == other.y;
        }
    };

    struct GridCellHash
    {
        size_t operator()(const GridCell &k) const
        {
            return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1);
        }
    };

    void scanCallback(const sensor_msgs::LaserScanConstPtr &scan_msg)
    {
        // 转换激光数据到笛卡尔坐标
        std::vector<geometry_msgs::Point32> current_points;
        float angle = scan_msg->angle_min;
        for (float range : scan_msg->ranges)
        {
            if (range >= scan_msg->range_min && range <= scan_msg->range_max)
            {
                geometry_msgs::Point32 point;
                point.x = range * cos(angle);
                point.y = range * sin(angle);
                point.z = 0.0f;

                // 区域裁剪
                if (point.x >= crop_x_min_ && point.x <= crop_x_max_ &&
                    point.y >= crop_y_min_ && point.y <= crop_y_max_)
                {
                    current_points.push_back(point);
                }
            }
            angle += scan_msg->angle_increment;
        }

        // 累积最近10帧数据
        point_history_.push_back(current_points);
        if (point_history_.size() > 10)
        {
            point_history_.pop_front();
        }

        // 合并历史数据
        std::vector<geometry_msgs::Point32> accumulated_points;
        for (const auto &frame : point_history_)
        {
            accumulated_points.insert(accumulated_points.end(), frame.begin(), frame.end());
        }

        // 体素滤波
        std::unordered_set<GridCell, GridCellHash> voxel_grid;
        const float voxel_size = resolution_;
        for (const auto &p : accumulated_points)
        {
            GridCell cell{
                static_cast<int>(floor(p.x / voxel_size)),
                static_cast<int>(floor(p.y / voxel_size))};
            voxel_grid.insert(cell);
        }

        // 创建占用栅格
        cv::Mat grid = cv::Mat::zeros(grid_rows_, grid_cols_, CV_8UC1);
        const float radius_sq = radius_ * radius_;

        for (const auto &cell : voxel_grid)
        {
            float world_x = (cell.x + 0.5f) * voxel_size;
            float world_y = (cell.y + 0.5f) * voxel_size;

            // 过滤中心区域
            if ((world_x * world_x + world_y * world_y) < radius_sq)
                continue;

            // 转换到栅格坐标系
            int col = static_cast<int>((world_x - origin_x_) / resolution_);
            int row = static_cast<int>((world_y - origin_y_) / resolution_);

            if (col >= 0 && col < grid_cols_ && row >= 0 && row < grid_rows_)
            {
                grid.at<uchar>(row, col) = 255;
            }
        }

        // 膨胀操作
        cv::Mat dilated_grid;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                   cv::Size(2 * inflation_pixels_ + 1, 2 * inflation_pixels_ + 1));
        cv::dilate(grid, dilated_grid, kernel);

        // 发布OccupancyGrid
        nav_msgs::OccupancyGrid grid_msg;
        grid_msg.header.stamp = ros::Time::now();
        grid_msg.header.frame_id = "laser_frame";
        grid_msg.info.resolution = resolution_;
        grid_msg.info.width = grid_cols_;
        grid_msg.info.height = grid_rows_;
        grid_msg.info.origin.position.x = origin_x_;
        grid_msg.info.origin.position.y = origin_y_;
        grid_msg.info.origin.orientation.w = 1.0;

        grid_msg.data.resize(grid_cols_ * grid_rows_);
        for (int row = 0; row < grid_rows_; ++row)
        {
            for (int col = 0; col < grid_cols_; ++col)
            {
                int idx = row * grid_cols_ + col;
                grid_msg.data[idx] = (dilated_grid.at<uchar>(row, col) > 0) ? 100 : 0;
            }
        }

        grid_pub_.publish(grid_msg);
    }

    // 成员变量
    ros::NodeHandle nh_;
    ros::Subscriber scan_sub_;
    ros::Publisher grid_pub_;

    // 参数
    float resolution_, grid_width_, grid_height_;
    float origin_x_, origin_y_;
    float inflation_radius_;
    float crop_x_min_, crop_x_max_, crop_y_min_, crop_y_max_;
    float radius_;
    int grid_cols_, grid_rows_, inflation_pixels_;

    std::deque<std::vector<geometry_msgs::Point32>> point_history_;
};

int main(int argc, char **​ argv)
{
    ros::init(argc, argv, "laser_occupancy_grid");
    LaserToOccupancyGrid converter;
    ros::spin();
    return 0;
}