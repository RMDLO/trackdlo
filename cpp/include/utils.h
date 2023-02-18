#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifndef UTILS_H
#define UTILS_H

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using cv::Mat;

void signal_callback_handler(int signum);

template <typename T>
void print_1d_vector (std::vector<T> vec);

double pt2pt_dis_sq (MatrixXf pt1, MatrixXf pt2);
double pt2pt_dis (MatrixXf pt1, MatrixXf pt2);

void reg (MatrixXf pts, MatrixXf& Y, double& sigma2, int M, double mu = 0, int max_iter = 50);
void remove_row(MatrixXf& matrix, unsigned int rowToRemove);
MatrixXf sort_pts (MatrixXf Y_0);

std::vector<MatrixXf> line_sphere_intersection (MatrixXf point_A, MatrixXf point_B, MatrixXf sphere_center, double radius);

#endif