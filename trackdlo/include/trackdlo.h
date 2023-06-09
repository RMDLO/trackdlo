#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/rgbd.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Float64.h>

#include <ctime>
#include <chrono>
#include <thread>
#include <algorithm> 
#include <string>

#include <unistd.h>
#include <cstdlib>
#include <signal.h>


#ifndef TRACKDLO_H
#define TRACKDLO_H

using Eigen::MatrixXd;
using cv::Mat;

class trackdlo
{
    public:
        // default constructor
        trackdlo();
        trackdlo(int num_of_nodes, MatrixXd proj_matrix);
        // fancy constructor
        trackdlo(int num_of_nodes,
                 double visibility_threshold,
                 double beta,
                 double lambda,
                 double alpha,
                 double gamma,
                 double k_vis,
                 double mu,
                 int max_iter,
                 const double tol,
                 bool include_lle,
                 bool use_geodesic,
                 bool use_prev_sigma2,
                 int kernel,
                 MatrixXd proj_matrix);

        double get_sigma2();
        MatrixXd get_tracking_result();
        MatrixXd get_guide_nodes();
        std::vector<MatrixXd> get_correspondence_pairs();
        void initialize_geodesic_coord (std::vector<double> geodesic_coord);
        void initialize_nodes (MatrixXd Y_init);
        void set_sigma2 (double sigma2);

        bool cpd_lle (MatrixXd X,
                      MatrixXd& Y,
                      double& sigma2,
                      double beta,
                      double lambda,
                      double gamma,
                      double mu,
                      int max_iter = 30,
                      double tol = 0.00001,
                      bool include_lle = true,
                      bool use_geodesic = false,
                      bool use_prev_sigma2 = false,
                      bool use_ecpd = false,
                      std::vector<MatrixXd> correspondence_priors = {},
                      double alpha = 0,
                      int kernel = 3,
                      std::vector<int> visible_nodes = {},
                      double k_vis = 0,
                      double visibility_threshold = 0.01);

        void tracking_step (MatrixXd X,
                            std::vector<int> visible_nodes,
                            std::vector<MatrixXd> visible_nodes_vec);

    private:
        MatrixXd Y_;
        MatrixXd proj_matrix_;
        MatrixXd guide_nodes_;
        double sigma2_;
        double beta_;
        double lambda_;
        double alpha_;
        double lle_weight_;
        double k_vis_;
        double mu_;
        int max_iter_;
        double tol_;
        bool include_lle_;
        bool use_geodesic_;
        bool use_prev_sigma2_;
        int kernel_;
        std::vector<double> geodesic_coord_;
        std::vector<MatrixXd> correspondence_priors_;
        double visibility_threshold_;

        std::vector<int> get_nearest_indices (int k, int M, int idx);
        MatrixXd calc_LLE_weights (int k, MatrixXd X);
        std::vector<MatrixXd> traverse_geodesic (std::vector<double> geodesic_coord, const MatrixXd guide_nodes, 
                                                 const std::vector<int> visible_nodes, int alignment);
        std::vector<MatrixXd> traverse_euclidean (std::vector<double> geodesic_coord, const MatrixXd guide_nodes, 
                                                  const std::vector<int> visible_nodes, int alignment, int alignment_node_idx = -1);

};

#endif