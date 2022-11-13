#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<ctime>

// typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVector;
// typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ColumnVector;

// using Eigen::MatrixXd;
using Eigen::MatrixXd;
using Eigen::RowVectorXi;

std::vector<int> get_nearest_indices (int k, int M, int idx) {
    std::vector<int> indices_arr;
    if (idx - k < 0) {
        for (int i = 0; i <= idx + k; i ++) {
            if (i != idx) {
                indices_arr.push_back(i);
            }
        }
    }
    else if (idx + k >= M) {
        for (int i = idx - k; i <= M - 1; i ++) {
            if (i != idx) {
                indices_arr.push_back(i);
            }
        }
    }
    else {
        for (int i = idx - k; i <= idx + k; i ++) {
            if (i != idx) {
                indices_arr.push_back(i);
            }
        }
    }

    return indices_arr;
}

MatrixXd calc_LLE_weights (int k, MatrixXd X) {
    MatrixXd W = MatrixXd::Zero(X.rows(), X.rows());
    for (int i = 0; i < X.rows(); i ++) {
        std::vector<int> indices = get_nearest_indices(static_cast<int>(k/2), X.rows(), i);
        MatrixXd xi = X.row(i);
        // std::cout << "--- xi ---" << std::endl;
        // std::cout << xi << std::endl;

        MatrixXd Xi = MatrixXd(indices.size(), X.cols());

        // fill in Xi: Xi = X[indices, :]
        for (int r = 0; r < indices.size(); r ++) {
            Xi.row(r) = X.row(indices[r]);
        }

        // std::cout << "--- Xi ---" << std::endl;
        // std::cout << Xi << std::endl;

        // component = np.full((len(Xi), len(xi)), xi).T - Xi.T
        MatrixXd component = xi.replicate(Xi.rows(), 1).transpose() - Xi.transpose();
        
        MatrixXd Gi = component.transpose() * component;
        // std::cout << "--- Gi ---" << std::endl;
        // std::cout << Gi << std::endl;
        
        MatrixXd Gi_inv;

        if (Gi.determinant() != 0) {
            Gi_inv = Gi.inverse();
        }
        else {
            // std::cout << "Gi singular at entry " << i << std::endl;
            double epsilon = 0.000000001;
            Gi.diagonal().array() += epsilon;
            Gi_inv = Gi.inverse();
        }

        // std::cout << "--- Gi_inv ---" << std::endl;
        // std::cout << Gi_inv << std::endl;

        // wi = Gi_inv * 1 / (1^T * Gi_inv * 1)
        MatrixXd ones_row_vec = MatrixXd::Constant(1, Xi.rows(), 1.0);
        MatrixXd ones_col_vec = MatrixXd::Constant(Xi.rows(), 1, 1.0);

        MatrixXd wi = (Gi_inv * ones_col_vec) / (ones_row_vec * Gi_inv * ones_col_vec).value();
        // std::cout << "--- wi ---" << std::endl;
        // std::cout << wi << std::endl;
        
        MatrixXd wi_T = wi.transpose();
        // std::cout << "--- wi_T ---" << std::endl;
        // std::cout << wi_T << std::endl;

        for (int c = 0; c < indices.size(); c ++) {
            W(i, indices[c]) = wi_T(c);
        }
    }

    return W;
}

template <typename T>
void print_1d_vector (std::vector<T> vec) {
    for (int i = 0; i < vec.size(); i ++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

MatrixXd cpd (MatrixXd X_orig,
              MatrixXd Y_0,
              double beta,
              double alpha,
              double gamma,
              double mu,
              int max_iter = 30,
              double tol = 0.00001,
              bool use_geodesic = false,
              bool use_prev_sigma2 = false,
              double sigma2_0 = 0,
              bool use_ecpd = false,
              MatrixXd correspondence_priors = MatrixXd::Zero(0, 0),
              double omega = 0,
              std::string kernel = "Gaussian",
              std::vector<int> occluded_nodes = {}) {

    MatrixXd X = X_orig.replicate(1, 1);

    int M = Y_0.rows();
    int N = X.rows();
    int D = 3;

    std::cout << "---" << std::endl;

    MatrixXd diff = MatrixXd::Zero(M, M);
    MatrixXd diff_sqrt = MatrixXd::Zero(M, M);
    for (int i = 0; i < Y_0.rows(); i ++) {
        for (int j = 0; j < Y_0.rows(); j ++) {
            diff(i, j) = (Y_0.row(i) - Y_0.row(j)).squaredNorm();
            diff_sqrt(i, j) = (Y_0.row(i) - Y_0.row(j)).norm();
        }
    }

    MatrixXd converted_node_dis = MatrixXd::Zero(M, M); // this is a M*M matrix in place of diff_sqrt
    MatrixXd converted_node_dis_sq = MatrixXd::Zero(M, M);
    MatrixXd G = MatrixXd::Zero(M, M);
    if (!use_geodesic) {
        if (kernel == "Gaussian") {
            G = (-diff / (2 * beta * beta)).array().exp();
        }
        else if (kernel == "Laplacian") {
            G = (-diff_sqrt / (2 * beta * beta)).array().exp();
        }
        else if (kernel == "1st order") {
            G = 1/(2*beta * 2*beta) * (-sqrt(2)*diff_sqrt/beta).array().exp() * (sqrt(2)*diff_sqrt.array() + beta);
        }
        else if (kernel == "2nd order") {
            G = 27 * 1/(72 * pow(beta, 3)) * (-sqrt(3)*diff_sqrt/beta).array().exp() * (sqrt(3)*beta*beta + 3*beta*diff_sqrt.array() + sqrt(3)*diff.array());
        }
        else { // default to gaussian
            G = (-diff / (2 * beta * beta)).array().exp();
        }
    }
    else {
        std::vector<double> converted_node_coord = {0.0};
        double cur_sum = 0;
        for (int i = 0; i < M-1; i ++) {
            cur_sum += (Y_0.row(i+1) - Y_0.row(i)).norm();
            converted_node_coord.push_back(cur_sum);
        }

        for (int i = 0; i < converted_node_coord.size(); i ++) {
            for (int j = 0; j < converted_node_coord.size(); j ++) {
                converted_node_dis_sq(i, j) = pow(converted_node_coord[i] - converted_node_coord[j], 2);
                converted_node_dis(i, j) = abs(converted_node_coord[i] - converted_node_coord[j]);
            }
        }

        if (kernel == "Gaussian") {
            G = (-converted_node_dis_sq / (2 * beta * beta)).array().exp();
        }
        else if (kernel == "Laplacian") {
            G = (-converted_node_dis / (2 * beta * beta)).array().exp();
        }
        else if (kernel == "1st order") {
            G = 1/(2*beta * 2*beta) * (-sqrt(2)*converted_node_dis/beta).array().exp() * (sqrt(2)*converted_node_dis.array() + beta);
        }
        else if (kernel == "2nd order") {
            G = 27 * 1/(72 * pow(beta, 3)) * (-sqrt(3)*converted_node_dis/beta).array().exp() * (sqrt(3)*beta*beta + 3*beta*converted_node_dis.array() + sqrt(3)*diff.array());
        }
        else { // default to gaussian
            G = (-converted_node_dis_sq / (2 * beta * beta)).array().exp();
        }
    }
    
    return G;
}

int main(int argc, char **argv) {
    ros::init (argc, argv, "test");
    ros::NodeHandle nh;

    MatrixXd m1 = MatrixXd::Zero(5, 3);
    for (int i = 0; i < m1.rows(); i ++) {
        for (int j = 0; j < m1.cols(); j ++) {
            m1(i, j) = (static_cast<float>(i)*m1.cols() + static_cast<float>(j))/100;
            m1(i, j) *= m1(i, j);
        }
    }

    // // ----- test LLE weights -----
    // std::cout << m1 << std::endl;
    // clock_t cur_time = clock();
    // MatrixXd out = calc_LLE_weights(2, m1);
    // std::cout << static_cast<double>(clock() - cur_time) / static_cast<double>(CLOCKS_PER_SEC) << std::endl;
    // // std::cout << out << std::endl;

    // ----- test ecpd -----
    std::cout << cpd(MatrixXd::Zero(1, 3), m1, 0.3, 0, 0, 0, 30, 0.00001, false, false, 0, false, MatrixXd(0, 0), 0, "Gaussian") << std::endl;
}