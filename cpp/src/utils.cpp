#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <ctime>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

#include <ctime>
#include <chrono>
#include <thread>

#include <unistd.h>
#include <cstdlib>
#include <signal.h>

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::RowVectorXf;
using Eigen::RowVectorXd;
using cv::Mat;

void signal_callback_handler(int signum) {
   // Terminate program
   exit(signum);
}

template <typename T>
void print_1d_vector (std::vector<T> vec) {
    for (int i = 0; i < vec.size(); i ++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

double pt2pt_dis_sq (MatrixXf pt1, MatrixXf pt2) {
    return (pt1 - pt2).rowwise().squaredNorm().sum();
}

double pt2pt_dis (MatrixXf pt1, MatrixXf pt2) {
    return (pt1 - pt2).rowwise().norm().sum();
}

void reg (MatrixXf pts, MatrixXf& Y, double& sigma2, int M, double mu = 0, int max_iter = 50) {
    // initial guess
    MatrixXf X = pts.replicate(1, 1);
    Y = MatrixXf::Zero(M, 3);
    for (int i = 0; i < M; i ++) {
        Y(i, 1) = 0.1 / static_cast<double>(M) * static_cast<double>(i);
        Y(i, 0) = 0;
        Y(i, 2) = 0;
    }
    
    int N = X.rows();
    int D = 3;

    // diff_xy should be a (M * N) matrix
    MatrixXf diff_xy = MatrixXf::Zero(M, N);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            diff_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
        }
    }

    // initialize sigma2
    sigma2 = diff_xy.sum() / static_cast<double>(D * M * N);

    for (int it = 0; it < max_iter; it ++) {
        // update diff_xy
        for (int i = 0; i < M; i ++) {
            for (int j = 0; j < N; j ++) {
                diff_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
            }
        }

        MatrixXf P = (-0.5 * diff_xy / sigma2).array().exp();
        MatrixXf P_stored = P.replicate(1, 1);
        double c = pow((2 * M_PI * sigma2), static_cast<double>(D)/2) * mu / (1 - mu) * static_cast<double>(M)/N;
        P = P.array().rowwise() / (P.colwise().sum().array() + c);

        MatrixXf Pt1 = P.colwise().sum(); 
        MatrixXf P1 = P.rowwise().sum();
        double Np = P1.sum();
        MatrixXf PX = P * X;

        MatrixXf P1_expanded = MatrixXf::Zero(M, D);
        P1_expanded.col(0) = P1;
        P1_expanded.col(1) = P1;
        P1_expanded.col(2) = P1;

        Y = PX.cwiseQuotient(P1_expanded);

        double numerator = 0;
        double denominator = 0;

        for (int m = 0; m < M; m ++) {
            for (int n = 0; n < N; n ++) {
                numerator += P(m, n)*diff_xy(m, n);
                denominator += P(m, n)*D;
            }
        }

        sigma2 = numerator / denominator;
    }
}

// link to original code: https://stackoverflow.com/a/46303314
void remove_row(MatrixXf& matrix, unsigned int rowToRemove) {
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.bottomRows(numRows-rowToRemove);

    matrix.conservativeResize(numRows,numCols);
}

MatrixXf sort_pts (MatrixXf Y_0) {
    int N = Y_0.rows();
    MatrixXf Y_0_sorted = MatrixXf::Zero(N, 3);
    std::vector<MatrixXf> Y_0_sorted_vec = {};
    std::vector<bool> selected_node(N, false);
    selected_node[0] = true;
    int last_visited_b = 0;

    MatrixXf G = MatrixXf::Zero(N, N);
    for (int i = 0; i < N; i ++) {
        for (int j = 0; j < N; j ++) {
            G(i, j) = (Y_0.row(i) - Y_0.row(j)).squaredNorm();
        }
    }

    int reverse = 0;
    int counter = 0;
    int reverse_on = 0;
    int insertion_counter = 0;

    while (counter < N-1) {
        double minimum = INFINITY;
        int a = 0;
        int b = 0;

        for (int m = 0; m < N; m ++) {
            if (selected_node[m] == true) {
                for (int n = 0; n < N; n ++) {
                    if ((!selected_node[n]) && (G(m, n) != 0.0)) {
                        if (minimum > G(m, n)) {
                            minimum = G(m, n);
                            a = m;
                            b = n;
                        }
                    }
                }
            }
        }

        if (counter == 0) {
            Y_0_sorted_vec.push_back(Y_0.row(a));
            Y_0_sorted_vec.push_back(Y_0.row(b));
        }
        else {
            if (last_visited_b != a) {
                reverse += 1;
                reverse_on = a;
                insertion_counter = 1;
            }
            
            if (reverse % 2 == 1) {
                auto it = find(Y_0_sorted_vec.begin(), Y_0_sorted_vec.end(), Y_0.row(a));
                Y_0_sorted_vec.insert(it, Y_0.row(b));
            }
            else if (reverse != 0) {
                auto it = find(Y_0_sorted_vec.begin(), Y_0_sorted_vec.end(), Y_0.row(reverse_on));
                Y_0_sorted_vec.insert(it + insertion_counter, Y_0.row(b));
                insertion_counter += 1;
            }
            else {
                Y_0_sorted_vec.push_back(Y_0.row(b));
            }
        }

        last_visited_b = b;
        selected_node[b] = true;
        counter += 1;
    }

    // copy to Y_0_sorted
    for (int i = 0; i < N; i ++) {
        Y_0_sorted.row(i) = Y_0_sorted_vec[i];
    }

    return Y_0_sorted;
}

bool isBetween (MatrixXf x, MatrixXf a, MatrixXf b) {
    bool in_bound = true;

    for (int i = 0; i < 3; i ++) {
        if (!(a(0, i)-0.0001 <= x(0, i) && x(0, i) <= b(0, i)+0.0001) && 
            !(b(0, i)-0.0001 <= x(0, i) && x(0, i) <= a(0, i)+0.0001)) {
            in_bound = false;
        }
    }
    
    return in_bound;
}

std::vector<MatrixXf> line_sphere_intersection (MatrixXf point_A, MatrixXf point_B, MatrixXf sphere_center, double radius) {
    std::vector<MatrixXf> intersections = {};
    
    double a = pt2pt_dis_sq(point_A, point_B);
    double b = 2 * ((point_B(0, 0) - point_A(0, 0))*(point_A(0, 0) - sphere_center(0, 0)) + 
                    (point_B(0, 1) - point_A(0, 1))*(point_A(0, 1) - sphere_center(0, 1)) + 
                    (point_B(0, 2) - point_A(0, 2))*(point_A(0, 2) - sphere_center(0, 2)));
    double c = pt2pt_dis_sq(point_A, sphere_center) - pow(radius, 2);
    
    double delta = pow(b, 2) - 4*a*c;

    double d1 = (-b + sqrt(delta)) / (2*a);
    double d2 = (-b - sqrt(delta)) / (2*a);

    if (delta < 0) {
        // no solution
        return {};
    }
    else if (delta > 0) {
        // two solutions
        // the first one
        double x1 = point_A(0, 0) + d1*(point_B(0, 0) - point_A(0, 0));
        double y1 = point_A(0, 1) + d1*(point_B(0, 1) - point_A(0, 1));
        double z1 = point_A(0, 2) + d1*(point_B(0, 2) - point_A(0, 2));
        MatrixXf pt1(1, 3);
        pt1 << x1, y1, z1;

        // the second one
        double x2 = point_A(0, 0) + d2*(point_B(0, 0) - point_A(0, 0));
        double y2 = point_A(0, 1) + d2*(point_B(0, 1) - point_A(0, 1));
        double z2 = point_A(0, 2) + d2*(point_B(0, 2) - point_A(0, 2));
        MatrixXf pt2(1, 3);
        pt2 << x2, y2, z2;

        if (isBetween(pt1, point_A, point_B)) {
            intersections.push_back(pt1);
        }
        if (isBetween(pt2, point_A, point_B)) {
            intersections.push_back(pt2);
        }
    }
    else {
        // one solution
        d1 = -b / (2*a);
        double x1 = point_A(0, 0) + d1*(point_B(0, 0) - point_A(0, 0));
        double y1 = point_A(0, 1) + d1*(point_B(0, 1) - point_A(0, 1));
        double z1 = point_A(0, 2) + d1*(point_B(0, 2) - point_A(0, 2));
        MatrixXf pt1(1, 3);
        pt1 << x1, y1, z1;

        if (isBetween(pt1, point_A, point_B)) {
            intersections.push_back(pt1);
        }
    }
    
    return intersections;
}