#pragma once

#include "trackdlo.h"

#ifndef EVALUATOR_H
#define EVALUATOR_H

using Eigen::MatrixXd;
using cv::Mat;

class evaluator
{
    public:
        evaluator ();
        evaluator (int length, int trial, int pct_occlusion, std::string alg, int bag_file, std::string save_location, 
                   double start_record_at, double exit_at, double wait_before_occlusion, double bag_rate, int num_of_nodes);
        MatrixXd get_ground_truth_nodes (Mat rgb_img, pcl::PointCloud<pcl::PointXYZRGB> cloud_xyz);
        MatrixXd sort_pts (MatrixXd Y_0, MatrixXd head);
        double calc_min_distance (MatrixXd A, MatrixXd B, MatrixXd E, MatrixXd& closest_pt_on_AB_to_E);
        double get_piecewise_error (MatrixXd Y_track, MatrixXd Y_true);
        double compute_and_save_error (MatrixXd Y_track, MatrixXd Y_true);
        void set_start_time (std::chrono::steady_clock::time_point cur_time);
        double pct_occlusion ();
        std::chrono::steady_clock::time_point start_time ();
        double recording_start_time ();
        double exit_time ();
        int length ();
        double wait_before_occlusion ();
        double rate ();
        double compute_error (MatrixXd Y_track, MatrixXd Y_true);
        void increment_image_counter ();
        int image_counter ();

    private:
        int length_;
        int trial_;
        int pct_occlusion_;
        std::string alg_;
        int bag_file_;
        std::vector<double> errors_;
        std::string save_location_;
        std::chrono::steady_clock::time_point start_time_;
        double start_record_at_;
        double exit_at_;
        double wait_before_occlusion_;
        bool cleared_file_;
        double bag_rate_;
        int image_counter_;
        int num_of_nodes_;
};

#endif