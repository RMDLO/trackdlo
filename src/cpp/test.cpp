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
        MatrixXd Xi = MatrixXd(indices.size(), X.cols());

        // fill in Xi: Xi = X[indices, :]
        for (int r = 0; r < indices.size(); r ++) {
            Xi.row(r) = X.row(indices[r]);
        }

        // component = np.full((len(Xi), len(xi)), xi).T - Xi.T
        MatrixXd component = xi.replicate(Xi.rows(), 1).transpose() - Xi.transpose();
        MatrixXd Gi = component.transpose() * component;
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

        // wi = Gi_inv * 1 / (1^T * Gi_inv * 1)
        MatrixXd ones_row_vec = MatrixXd::Constant(1, Xi.rows(), 1.0);
        MatrixXd ones_col_vec = MatrixXd::Constant(Xi.rows(), 1, 1.0);

        MatrixXd wi = (Gi_inv * ones_col_vec) / (ones_row_vec * Gi_inv * ones_col_vec).value();
        MatrixXd wi_T = wi.transpose();

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

double pt2pt_dis_sq (MatrixXd pt1, MatrixXd pt2) {
    return (pt1 - pt2).rowwise().squaredNorm().sum();
}

MatrixXd cpd (MatrixXd X_orig,
              MatrixXd Y_0,
              double beta,
              double alpha,
              double gamma,
              double mu,
              int max_iter = 30,
              double tol = 0.00001,
              bool include_lle = true,
              bool use_geodesic = false,
              bool use_prev_sigma2 = false,
              double sigma2_0 = 0,
              bool use_ecpd = false,
              MatrixXd correspondence_priors = MatrixXd::Zero(0, 0),
              double omega = 0,
              std::string kernel = "Gaussian",
              std::vector<int> occluded_nodes = {}) {

    // log time            
    clock_t cur_time = clock();

    MatrixXd X = X_orig.replicate(1, 1);

    int M = Y_0.rows();
    int N = X.rows();
    int D = 3;

    std::cout << "---" << std::endl;

    MatrixXd diff_yy = MatrixXd::Zero(M, M);
    MatrixXd diff_yy_sqrt = MatrixXd::Zero(M, M);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < M; j ++) {
            diff_yy(i, j) = (Y_0.row(i) - Y_0.row(j)).squaredNorm();
            diff_yy_sqrt(i, j) = (Y_0.row(i) - Y_0.row(j)).norm();
        }
    }

    // diff_xy should be a (M * N) matrix
    MatrixXd diff_xy = MatrixXd::Zero(M, N);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            diff_xy(i, j) = (Y_0.row(i) - X.row(j)).squaredNorm();
        }
    }

    MatrixXd converted_node_dis = MatrixXd::Zero(M, M); // this is a M*M matrix in place of diff_sqrt
    MatrixXd converted_node_dis_sq = MatrixXd::Zero(M, M);
    MatrixXd G = MatrixXd::Zero(M, M);
    if (!use_geodesic) {
        if (kernel == "Gaussian") {
            G = (-diff_yy / (2 * beta * beta)).array().exp();
        }
        else if (kernel == "Laplacian") {
            G = (-diff_yy_sqrt / (2 * beta * beta)).array().exp();
        }
        else if (kernel == "1st order") {
            G = 1/(2*beta * 2*beta) * (-sqrt(2)*diff_yy_sqrt/beta).array().exp() * (sqrt(2)*diff_yy_sqrt.array() + beta);
        }
        else if (kernel == "2nd order") {
            G = 27 * 1/(72 * pow(beta, 3)) * (-sqrt(3)*diff_yy_sqrt/beta).array().exp() * (sqrt(3)*beta*beta + 3*beta*diff_yy_sqrt.array() + sqrt(3)*diff_yy.array());
        }
        else { // default to gaussian
            G = (-diff_yy / (2 * beta * beta)).array().exp();
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
            G = 27 * 1/(72 * pow(beta, 3)) * (-sqrt(3)*converted_node_dis/beta).array().exp() * (sqrt(3)*beta*beta + 3*beta*converted_node_dis.array() + sqrt(3)*diff_yy.array());
        }
        else { // default to gaussian
            G = (-converted_node_dis_sq / (2 * beta * beta)).array().exp();
        }
    }

    MatrixXd Y = Y_0.replicate(1, 1);

    // initialize sigma2
    double sigma2 = 0;
    if (!use_prev_sigma2) {
        sigma2 = diff_xy.sum() / (static_cast<double>(D * M * N));
    }
    else {
        sigma2 = sigma2_0;
    }

    // get the LLE matrix
    MatrixXd L = calc_LLE_weights(2, Y_0);
    MatrixXd H = (MatrixXd::Identity(M, M) - L).transpose() * (MatrixXd::Identity(M, M) - L);

    // TODO: implement node deletion from the original point cloud

    for (int it = 0; it < max_iter; it ++) {

        // update diff_xy
        for (int i = 0; i < M; i ++) {
            for (int j = 0; j < N; j ++) {
                diff_xy(i, j) = (Y_0.row(i) - X.row(j)).squaredNorm();
            }
        }

        MatrixXd P = (-0.5 * diff_xy / sigma2).array().exp();
        double c = pow((2 * M_PI * sigma2), static_cast<double>(D)/2) * mu / (1 - mu) * static_cast<double>(M)/N;
        P = P.array().rowwise() / (P.colwise().sum().array() + c);

        std::vector<int> max_p_nodes(P.cols(), 0);
        for (int i = 0; i < N; i ++) {
            P.col(i).maxCoeff(&max_p_nodes[i]);
        }

        // TODO: implement geodesic dists
        // if (use_geodesic) {
        //     std::vector<int> potential_2nd_max_p_nodes_1(P.cols(), 0);   // -1
        //     std::vector<int> potential_2nd_max_p_nodes_2(P.cols(), 0);   // +1
        //     for (int i = 0; i < N; i ++) {
        //         // potential_2nd_max_p_nodes_1 = max_p_nodes - 1
        //         if (max_p_nodes[i] == 0) {
        //             potential_2nd_max_p_nodes_1[i] = 1;
        //         }
        //         else {
        //             potential_2nd_max_p_nodes_1[i] = max_p_nodes[i] - 1;
        //         }

        //         // potential_2nd_max_p_nodes_2 = max_p_nodes + 1
        //         if (max_p_nodes[i] == M - 1) {
        //             potential_2nd_max_p_nodes_2[i] = M - 2;
        //         }
        //         else {
        //             potential_2nd_max_p_nodes_2[i] = max_p_nodes[i] + 1;
        //         }
        //     }
        // }

        MatrixXd Pt1 = P.colwise().sum();
        MatrixXd P1 = P.rowwise().sum();
        double Np = P1.sum();
        MatrixXd PX = P * X;

        // M step
        MatrixXd A_matrix;
        MatrixXd B_matrix;
        if (include_lle) {
            A_matrix = P1.asDiagonal() * G + alpha * sigma2 * MatrixXd::Identity(M, M) + sigma2 * gamma * H * G;
            B_matrix = PX - P1.asDiagonal() * Y_0 - sigma2 * gamma * H * Y_0;
        }
        else {
            A_matrix = P1.asDiagonal() * G + alpha * sigma2 * MatrixXd::Identity(M, M);
            B_matrix = PX - P1.asDiagonal() * Y_0;
        }

        MatrixXd W = (A_matrix).householderQr().solve(B_matrix);

        MatrixXd T = Y_0 + G * W;
        double trXtdPt1X = (X.transpose() * Pt1.asDiagonal() * X).trace();
        double trPXtT = (PX.transpose() * T).trace();
        double trTtdP1T = (T.transpose() * P1.asDiagonal() * T).trace();

        sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D);

        if (pt2pt_dis_sq(Y, Y_0 + G*W) < tol) {
            Y = Y_0 + G*W;
            std::cout << "iterations until convergence: " << it << std::endl;
            break;
        }
        else {
            Y = Y_0 + G*W;
        }

        if (it == max_iter - 1) {
            std::cout << "optimization did not converge!" << std::endl;
            break;
        }
    }
    
    std::cout << "time taken:" << static_cast<double>(clock() - cur_time) / static_cast<double>(CLOCKS_PER_SEC) << std::endl;
    return Y;
}

int main(int argc, char **argv) {
    ros::init (argc, argv, "test");
    ros::NodeHandle nh;

    MatrixXd m1 = MatrixXd::Zero(5, 3);
    for (int i = 0; i < m1.rows(); i ++) {
        for (int j = 0; j < m1.cols(); j ++) {
            m1(i, j) = (static_cast<float>(i)*m1.cols() + static_cast<float>(j))/100;
            // m1(i, j) *= m1(i, j);
        }
    }

    MatrixXd m2 = MatrixXd::Zero(10, 3);
    for (int i = 0; i < m2.rows(); i ++) {
        for (int j = 0; j < m2.cols(); j ++) {
            m2(i, j) = (static_cast<float>(i)*m2.cols() + static_cast<float>(j))/200;
            // m1(i, j) *= m1(i, j);
        }
    }

    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;

    // // ----- test LLE weights -----
    // clock_t cur_time = clock();
    // MatrixXd out = calc_LLE_weights(2, m1);
    // std::cout << static_cast<double>(clock() - cur_time) / static_cast<double>(CLOCKS_PER_SEC) << std::endl;
    // // std::cout << out << std::endl;

    // // ----- test pt2pt_dis_sq -----
    // MatrixXd m3 = m2.array() + 0.5;
    // std::cout << m3 << std::endl;
    // std::cout << pt2pt_dis_sq(m2, m3) << std::endl;

    // ----- test ecpd -----
    std::cout << cpd(m2, m1, 0.3, 1, 1, 0.1, 30, 0.00001, true, false, false, 0, false, MatrixXd(0, 0), 0, "Gaussian") << std::endl;
    // print_1d_vector(cpd(m2, m1, 0.3, 1, 1, 0.05, 1, 0.00001, false, false, 0, false, MatrixXd(0, 0), 0, "Gaussian"));
}