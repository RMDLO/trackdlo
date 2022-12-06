#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>

using Eigen::MatrixXd;

int main(int argc, char **argv) {
    ros::init (argc, argv, "test");
    ros::NodeHandle nh;

    MatrixXd m1 = MatrixXd::Zero(5, 3);
    for (int i = 0; i < m1.rows(); i ++) {
        for (int j = 0; j < m1.cols(); j ++) {
            m1(i, j) = (static_cast<float>(i)*m1.cols() + static_cast<float>(j))/100;
            // m1(i, j) *= m1(i, j);
        }
    }

    MatrixXd m2 = m1.replicate(1, 1);
    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;

    // MatrixXd m2 = MatrixXd::Zero(10, 3);
    for (int i = 0; i < m2.rows(); i ++) {
        for (int j = 0; j < m2.cols(); j ++) {
            m2(i, j) = (static_cast<float>(i)*m2.cols() + static_cast<float>(j))/200;
            // m1(i, j) *= m1(i, j);
        }
    }

    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;
}