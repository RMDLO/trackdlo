#pragma once

#include "trackdlo.h"

#ifndef UTILS_H
#define UTILS_H

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using cv::Mat;

void signal_callback_handler(int signum);

template <typename T>
void print_1d_vector (std::vector<T> vec);

void print_1d_vector_eigen (std::vector<MatrixXf> vec);

double pt2pt_dis_sq (MatrixXf pt1, MatrixXf pt2);
double pt2pt_dis (MatrixXf pt1, MatrixXf pt2);

void reg (MatrixXf pts, MatrixXf& Y, double& sigma2, int M, double mu = 0, int max_iter = 50);
void remove_row(MatrixXf& matrix, unsigned int rowToRemove);
MatrixXf sort_pts (MatrixXf Y_0);

std::vector<MatrixXf> line_sphere_intersection (MatrixXf point_A, MatrixXf point_B, MatrixXf sphere_center, double radius);

visualization_msgs::MarkerArray MatrixXf2MarkerArray (MatrixXf Y, std::string marker_frame, std::string marker_ns, std::vector<float> node_color, std::vector<float> line_color);
visualization_msgs::MarkerArray MatrixXf2MarkerArray (std::vector<MatrixXf> Y, std::string marker_frame, std::string marker_ns, std::vector<float> node_color, std::vector<float> line_color);

MatrixXf cross_product (MatrixXf vec1, MatrixXf vec2);
double dot_product (MatrixXf vec1, MatrixXf vec2);

#endif