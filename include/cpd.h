#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>

#ifndef CPD_H
#define CPD_H

using Eigen::MatrixXd;
using Eigen::MatrixXf;

MatrixXf cpd (MatrixXf, MatrixXf, double, double, double, double, int, double, bool, 
              bool, bool, double, bool, MatrixXf, double, std::string, std::vector<int>);

MatrixXf sort_pts (MatrixXf pts_orig);

#endif