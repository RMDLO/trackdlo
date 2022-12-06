#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifndef TRACKDLO_H
#define TRACKDLO_H

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using cv::Mat;

template <typename T>
void print_1d_vector (std::vector<T> vec);

bool ecpd_lle (MatrixXf X_orig,
               MatrixXf& Y,
               double& sigma2,
               double beta,
               double alpha,
               double gamma,
               double mu,
               int max_iter = 30,
               double tol = 0.00001,
               bool include_lle = true,
               bool use_geodesic = false,
               bool use_prev_sigma2 = false,
               bool use_ecpd = false,
               std::vector<MatrixXf> correspondence_priors = {},
               double omega = 0,
               std::string kernel = "Gaussian",
               std::vector<int> occluded_nodes = {});

MatrixXf sort_pts (MatrixXf pts_orig);
double pt2pt_dis (MatrixXf pt1, MatrixXf pt2);

MatrixXf tracking_step (MatrixXf X_orig,
                        MatrixXf& Y,
                        double& sigma2,
                        std::vector<double> geodesic_coord,
                        double total_len,
                        Mat bmask,
                        Mat bmask_transformed_normalized,
                        double mask_dist_threshold);

#endif