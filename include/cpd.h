#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>

#ifndef CPD_H
#define CPD_H

using Eigen::MatrixXd;

MatrixXd cpd (MatrixXd, MatrixXd, double, double, double, double, int, double, bool, 
              bool, bool, double, bool, MatrixXd, double, std::string, std::vector<int>);

MatrixXd sort_pts (MatrixXd pts_orig);

#endif