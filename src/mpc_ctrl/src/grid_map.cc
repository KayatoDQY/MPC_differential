#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include <opencv2/opencv.hpp>
#include <deque>

class PointCloudToOccupancyGrid
{
public:
    PointCloudToOccupancyGrid() : nh_("~")
    {

        nh_.param("resolution", resolution_, 0.05f);
        nh_.param("grid_width", grid_width_, 40.0f);
        nh_.param("grid_height", grid_height_, 40.0f);
        nh_.param("origin_x", origin_x_, -20.0f);
        nh_.param("origin_y", origin_y_, -20.0f);
        nh_.param("inflation_radius", inflation_radius_, 0.2f);
        grid_cols_ = static_cast<int>(grid_width_ / resolution_);
        grid_rows_ = static_cast<int>(grid_height_ / resolution_);
        inflation_pixels_ = static_cast<int>(inflation_radius_ / resolution_);
        nh_.param("crop_x_min", crop_min_.x, -5.0f);
        nh_.param("crop_y_min", crop_min_.y, -5.0f);
        nh_.param("crop_z_min", crop_min_.z, -0.1f);
        nh_.param("crop_x_max", crop_max_.x, 5.0f);
        nh_.param("crop_y_max", crop_max_.y, 5.0f);
        nh_.param("crop_z_max", crop_max_.z, 0.3f);

        nh_.param("radius", radius_, 0.2f);

        cloud_sub_ = nh_.subscribe("/ipc/lidar", 1, &PointCloudToOccupancyGrid::cloudCallback, this);
        grid_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("occupancy_grid", 1);
    }

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::CropBox<pcl::PointXYZ> crop;
        crop.setMin(crop_min_.getVector4fMap());
        crop.setMax(crop_max_.getVector4fMap());
        crop.setInputCloud(cloud);
        crop.filter(*filtered_cloud);

        cloud_queue_.push_back(filtered_cloud);
        if (cloud_queue_.size() > 10)
        {
            cloud_queue_.pop_front();
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto &cloud : cloud_queue_)
        {
            *accumulated_cloud += *cloud;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(accumulated_cloud);
        voxel_filter.setLeafSize(resolution_, resolution_, resolution_);
        voxel_filter.filter(*voxel_cloud);

        cv::Mat grid = cv::Mat::zeros(grid_rows_, grid_cols_, CV_8UC1);

        for (const auto &point : *voxel_cloud)
        {
            if (point.x * point.x + point.y * point.y >= radius_ * radius_)
            {
                int col = static_cast<int>((point.x - origin_x_) / resolution_);
                int row = static_cast<int>((point.y - origin_y_) / resolution_);
                if (col >= 0 && col < grid_cols_ && row >= 0 && row < grid_rows_)
                {
                    grid.at<uchar>(row, col) = 255;
                }
            }
        }

        cv::Mat dilated_grid;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                   cv::Size(2 * inflation_pixels_ + 1, 2 * inflation_pixels_ + 1));
        cv::dilate(grid, dilated_grid, kernel);

        nav_msgs::OccupancyGrid grid_msg;
        grid_msg.header = msg->header;
        grid_msg.header.frame_id = "livox_frame";
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

private:
    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    ros::Publisher grid_pub_;

    float resolution_;
    float grid_width_, grid_height_;
    float origin_x_, origin_y_;
    int grid_cols_, grid_rows_;
    float inflation_radius_;
    int inflation_pixels_;
    pcl::PointXYZ crop_min_, crop_max_;
    float radius_;

    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_queue_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "occupancy_grid");
    PointCloudToOccupancyGrid converter;
    ros::spin();
    return 0;
}