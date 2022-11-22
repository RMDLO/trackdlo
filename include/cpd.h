#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>

#ifndef CPD_H
#define CPD_H

using Eigen::MatrixXd;
using Eigen::MatrixXf;

bool cpd (MatrixXf X_orig,
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
          MatrixXf correspondence_priors = MatrixXf::Zero(0, 0),
          double omega = 0,
          std::string kernel = "Gaussian",
          std::vector<int> occluded_nodes = {});

MatrixXf sort_pts (MatrixXf pts_orig);

#endif