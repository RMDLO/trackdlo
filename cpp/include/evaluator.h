#pragma once

#include "trackdlo.h"

#ifndef EVALUATOR_H
#define EVALUATOR_H

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using cv::Mat;

class evaluator
{
    public:
        evaluator ();
        evaluator (int length, int trial, double pct_occlusion, std::string alg, int bag_file);
        MatrixXf get_ground_truth_nodes (Mat rgb_img, pcl::PointCloud<pcl::PointXYZRGB> cloud_xyz);
        MatrixXf sort_pts (MatrixXf Y_0, MatrixXf head);
        double calc_min_distance (MatrixXf A, MatrixXf B, MatrixXf E, MatrixXf& closest_pt_on_AB_to_E);
        double get_piecewise_error (MatrixXf Y_track, MatrixXf Y_true);
        double compute_and_save_error (MatrixXf Y_track, MatrixXf Y_true);
    private:
        int length_;
        int trial_;
        double pct_occlusion_;
        std::string alg_;
        int bag_file_;
        std::vector<double> errors_;
};

#endif