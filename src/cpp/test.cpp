#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
              int max_iter,
              double tol,
              bool use_geodesic = false,
              bool use_prev_sigma2 = false,
              double sigma2_0 = 0,
              bool use_ecpd = false,
              MatrixXd correspondence_priors = MatrixXd::Zero(0, 0),
              double omega = 0,
              std::string kernel = "Gaussian",
              RowVectorXi occluded_nodes = RowVectorXi::Zero(0)) {
    
    return MatrixXd::Zero(0, 0);
}

int main(int argc, char **argv) {
    ros::init (argc, argv, "test");
    ros::NodeHandle nh;

    // test eigen matrix slicing
    MatrixXd m1 = MatrixXd::Zero(6, 3);
    for (int i = 0; i < m1.rows(); i ++) {
        for (int j = 0; j < m1.cols(); j ++) {
            m1(i, j) = (static_cast<float>(i)*m1.cols() + static_cast<float>(j))/100;
        }
    }

    std::cout << m1 << std::endl;

    MatrixXd out = calc_LLE_weights(2, m1);
    std::cout << "-----" << std::endl;
    std::cout << out << std::endl;
}