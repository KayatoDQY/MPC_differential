#ifndef ELLIPSEUtILS_H

#define ELLIPSEUtILS_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <limits>

struct EllipseParams
{
    Eigen::Matrix2d E;
    cv::Point2d center;
    double angle;
};
struct LinearConstraint
{
    Eigen::Vector2d a_j;
    double b_j;
};

double calculateAngle(const cv::Point2d &p1, const cv::Point2d &p2)
{
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return std::atan2(dy, dx) * 180.0 / CV_PI;
}

cv::Point2d findClosestObstacleInEllipse(const cv::Mat &img,
                                         const Eigen::Matrix2d &E,
                                         const cv::Point2d &center,
                                         double angle)
{
    const int rows = img.rows;
    const int cols = img.cols;
    const double a = E(0, 0);
    const double b = E(1, 1);
    const double angleRad = angle * CV_PI / 180.0;

    const double cosPhi = std::cos(angleRad);
    const double sinPhi = std::sin(angleRad);

    const double xExtent = std::sqrt(a * a * cosPhi * cosPhi + b * b * sinPhi * sinPhi);
    int left = static_cast<int>(std::floor(center.x - xExtent));
    int right = static_cast<int>(std::ceil(center.x + xExtent));

    const double yExtent = std::sqrt(a * a * sinPhi * sinPhi + b * b * cosPhi * cosPhi);
    int top = static_cast<int>(std::floor(center.y - yExtent));
    int bottom = static_cast<int>(std::ceil(center.y + yExtent));

    left = std::max(left, 0);
    right = std::min(right, cols - 1);
    top = std::max(top, 0);
    bottom = std::min(bottom, rows - 1);

    cv::Point2d closestPoint(-1, -1);
    double minDistance = std::numeric_limits<double>::max();

    const double cosAngle = cosPhi;
    const double sinAngle = sinPhi;

    for (int y = top; y <= bottom; ++y)
    {
        for (int x = left; x <= right; ++x)
        {
            if (img.at<uchar>(y, x) != 0)
                continue;

            const double xTrans = x - center.x;
            const double yTrans = y - center.y;

            const double xRot = cosAngle * xTrans + sinAngle * yTrans;
            const double yRot = -sinAngle * xTrans + cosAngle * yTrans;

            if ((xRot * xRot) / (a * a) + (yRot * yRot) / (b * b) > 1.0)
                continue;

            const double distance = cv::norm(cv::Point2d(x, y) - center);
            if (distance < minDistance)
            {
                minDistance = distance;
                closestPoint = cv::Point2d(x, y);
            }
        }
    }
    return closestPoint;
}

EllipseParams adjustEllipse(const cv::Mat &img, const cv::Point2d &p1, const cv::Point2d &p2)
{
    const cv::Point2d center = (p1 + p2) * 0.5;
    const double radius = cv::norm(p1 - p2) / 2.0;
    double minorAxis = radius;
    const double angle = calculateAngle(p1, p2);

    Eigen::Matrix2d E;
    E << radius, 0,
        0, minorAxis;
    int iteration = 0;
    const int max_iterations = 100;
    while (iteration++ < max_iterations)
    {
        const cv::Point2d obstacle = findClosestObstacleInEllipse(img, E, center, angle);
        if (obstacle.x < 0 && obstacle.y < 0)
        {
            break;
        }

        if (E(0, 0) <= 1e-3 || E(1, 1) <= 1e-3)
        {
            E(1, 1) = 1.0;
            break;
        }
        const double xShifted = obstacle.x - center.x;
        const double yShifted = obstacle.y - center.y;
        const double invAngleRad = -angle * CV_PI / 180.0;

        const double cosInv = std::cos(invAngleRad);
        const double sinInv = std::sin(invAngleRad);

        const double xPrime = xShifted * cosInv - yShifted * sinInv;
        const double yPrime = xShifted * sinInv + yShifted * cosInv;

        const double a = E(0, 0);
        const double denominator = a * a - xPrime * xPrime;

        if (denominator <= 0)
        {
            minorAxis = 1.0;
        }
        else
        {
            minorAxis = std::sqrt((yPrime * yPrime * a * a) / denominator) - 0.1;
            minorAxis = std::max(minorAxis, 1.0);
        }

        E(1, 1) = minorAxis;
    }
    return {E, center, angle};
}

cv::Point2d find_closest_obstacle_to_rotated_ellipse(
    const cv::Mat &img,
    const EllipseParams &ellipse,
    const Eigen::MatrixXd *A = nullptr,
    const Eigen::VectorXd *b = nullptr)
{
    const int rows = img.rows;
    const int cols = img.cols;

    const Eigen::Matrix2d &E = ellipse.E; // 椭圆结构体
    const cv::Point2d &d = ellipse.center;
    const double angle = ellipse.angle;

    const Eigen::Vector2d e1 = E * Eigen::Vector2d(1, 0);
    const Eigen::Vector2d e2 = E * Eigen::Vector2d(0, 1);
    const double major_axis_length = std::max(e1.norm(), e2.norm());

    // const int start_x = std::max(0, static_cast<int>(d.x - major_axis_length));
    // const int end_x = std::min(cols - 1, static_cast<int>(d.x + major_axis_length));
    // const int start_y = std::max(0, static_cast<int>(d.y - major_axis_length));
    // const int end_y = std::min(rows - 1, static_cast<int>(d.y + major_axis_length));
    const int start_x = 0;
    const int end_x = cols - 1;
    const int start_y = 0;
    const int end_y = rows - 1;

    cv::Point2d closest_point(-1, -1);
    double min_dist = std::numeric_limits<double>::max();

    const double angle_rad = angle * CV_PI / 180.0;
    const double cos_angle = std::cos(angle_rad);
    const double sin_angle = std::sin(angle_rad);

    for (int y = start_y; y <= end_y; ++y)
    {
        for (int x = start_x; x <= end_x; ++x)
        {
            if (img.at<uchar>(y, x) != 0)
                continue;

            if (A && b)
            {
                Eigen::VectorXd p(2);
                p << x, y;
                const Eigen::VectorXd constraint = A->transpose() * p;

                bool satisfy = true;
                for (int i = 0; i < b->size(); ++i)
                {
                    if (constraint(i) >= (*b)(i))
                    {
                        satisfy = false;
                        break;
                    }
                }
                if (!satisfy)
                    continue;
            }

            const Eigen::Vector2d point_vec(x - d.x, y - d.y);

            const Eigen::Vector2d rotated_point = {
                cos_angle * point_vec[0] + sin_angle * point_vec[1],
                -sin_angle * point_vec[0] + cos_angle * point_vec[1]};

            const Eigen::Vector2d normalized_point = E.inverse() * rotated_point;

            const double dist = normalized_point.norm();
            if (dist < min_dist)
            {
                min_dist = dist;
                closest_point = cv::Point2d(x, y);
            }
        }
    }
    return closest_point;
}

EllipseParams expand_ellipse_to_point(const EllipseParams &original,
                                      const cv::Point2d &point)
{
    const Eigen::Matrix2d &E = original.E;
    const cv::Point2d &d = original.center;
    const double angle = original.angle;

    const double angle_rad = angle * CV_PI / 180.0;
    const double cos_angle = std::cos(angle_rad);
    const double sin_angle = std::sin(angle_rad);

    const Eigen::Vector2d point_vec(point.x - d.x, point.y - d.y);

    const Eigen::Vector2d rotated_point = {
        cos_angle * point_vec[0] + sin_angle * point_vec[1],
        -sin_angle * point_vec[0] + cos_angle * point_vec[1]};

    const Eigen::Vector2d normalized_point = E.inverse() * rotated_point;

    const double scale_factor = normalized_point.norm();
    EllipseParams expanded = original;
    expanded.E = E * scale_factor;
    if (expanded.E(0, 0) < 1e-3)
        expanded.E(0, 0) = 1e-3;
    if (expanded.E(1, 1) < 1e-3)
        expanded.E(1, 1) = 1e-3;

    return expanded;
}

LinearConstraint calculate_aj_bj(const EllipseParams &ellipse,
                                 const cv::Point2d &closest_obstacle)
{
    const Eigen::Matrix2d &E = ellipse.E;
    const cv::Point2d &d = ellipse.center;
    const double angle = ellipse.angle;

    const double angle_rad = angle * CV_PI / 180.0;
    const double cos_angle = std::cos(angle_rad);
    const double sin_angle = std::sin(angle_rad);

    const Eigen::Vector2d p_jc(closest_obstacle.x, closest_obstacle.y);
    const Eigen::Vector2d d_vec(d.x, d.y);

    Eigen::Matrix2d R;
    R << cos_angle, -sin_angle,
        sin_angle, cos_angle;
    const Eigen::Matrix2d E_rotated = R * E * R.transpose();
    const Eigen::Matrix2d E_inv = E_rotated.inverse();
    const Eigen::Matrix2d E_inv_T = E_inv.transpose();

    const Eigen::Vector2d delta_p = p_jc - d_vec;

    const Eigen::Vector2d a_j = 2 * E_inv * E_inv_T * delta_p;

    const double b_j = a_j.dot(p_jc);

    return {a_j, b_j};
}
bool has_obstacle_in_region(const cv::Mat &img,
                            const EllipseParams &ellipse,
                            const Eigen::MatrixXd &A,
                            const Eigen::VectorXd &b)
{
    const int rows = img.rows;
    const int cols = img.cols;
    const cv::Point2d &d = ellipse.center;
    const Eigen::Vector2d e1 = ellipse.E * Eigen::Vector2d(1, 0);
    const Eigen::Vector2d e2 = ellipse.E * Eigen::Vector2d(0, 1);
    const double major_axis_length = std::max(e1.norm(), e2.norm());

    const int start_x = 0;
    const int end_x = cols - 1;
    const int start_y = 0;
    const int end_y = rows - 1;

    const int num_constraints = A.cols();

    for (int y = start_y; y <= end_y; ++y)
    {
        for (int x = start_x; x <= end_x; ++x)
        {
            if (img.at<uchar>(y, x) != 0)
                continue;

            bool satisfy = true;
            const Eigen::Vector2d p(x, y);

            for (int i = 0; i < num_constraints; ++i)
            {
                const double constraint = A.col(i).dot(p);
                if (constraint >= b(i))
                {
                    satisfy = false;
                    break;
                }
            }

            if (satisfy)
                return true;
        }
    }
    return false;
}

#endif
