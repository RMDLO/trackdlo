#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>

// typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVector;
// typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ColumnVector;

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::RowVectorXf;
using Eigen::RowVectorXd;

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

// link to original code: https://stackoverflow.com/a/46303314
void removeRow(MatrixXf& matrix, unsigned int rowToRemove) {
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.bottomRows(numRows-rowToRemove);

    matrix.conservativeResize(numRows,numCols);
}

void find_closest (MatrixXf pt, MatrixXf arr, MatrixXf& closest, int& idx) {
    closest = arr.row(0).replicate(1, 1);
    double min_dis = pt2pt_dis(pt, closest);
    idx = 0;

    for (int i = 0; i < arr.rows(); i ++) {
        double cur_dis = pt2pt_dis(pt, arr.row(i));
        if (cur_dis < min_dis) {
            min_dis = cur_dis;
            closest = arr.row(i).replicate(1, 1);
            idx = i;
        }
    }
}

void find_opposite_closest (MatrixXf pt, MatrixXf arr, MatrixXf direction_pt, MatrixXf& opposite_closest, bool& opposite_closest_found) {
    MatrixXf arr_copy = arr.replicate(1, 1);
    opposite_closest_found = false;
    opposite_closest = pt.replicate(1, 1);

    while (!opposite_closest_found && arr_copy.rows() != 0) {
        MatrixXf cur_closest;
        int cur_index;
        find_closest(pt, arr_copy, cur_closest, cur_index);
        removeRow(arr_copy, cur_index);

        RowVectorXf vec1 = cur_closest - pt;
        RowVectorXf vec2 = direction_pt - pt;

        if (vec1.dot(vec2) < 0 && pt2pt_dis(cur_closest, pt) < 0.07) {
            opposite_closest_found = true;
            opposite_closest = cur_closest.replicate(1, 1);
            break;
        }
    }
}

MatrixXf sort_pts (MatrixXf pts_orig) {

    int start_idx = 0;

    MatrixXf pts = pts_orig.replicate(1, 1);
    MatrixXf starting_pt = pts.row(start_idx).replicate(1, 1);
    removeRow(pts, start_idx);

    // starting point will be the current first point in the new list
    MatrixXf sorted_pts = MatrixXf::Zero(pts_orig.rows(), pts_orig.cols());
    std::vector<MatrixXf> sorted_pts_vec;
    sorted_pts_vec.push_back(starting_pt);

    // get the first closest point
    MatrixXf closest_1;
    int min_idx;
    find_closest(starting_pt, pts, closest_1, min_idx);
    sorted_pts_vec.push_back(closest_1);
    removeRow(pts, min_idx);

    // get the second closest point
    MatrixXf closest_2;
    bool found;
    find_opposite_closest(starting_pt, pts, closest_1, closest_2, found);
    bool true_start = false;
    if (!found) {
        true_start = true;
    }

    while (pts.rows() != 0) {
        MatrixXf cur_target = sorted_pts_vec[sorted_pts_vec.size() - 1];
        MatrixXf cur_direction = sorted_pts_vec[sorted_pts_vec.size() - 2];
        MatrixXf cur_closest;
        bool found;
        find_opposite_closest(cur_target, pts, cur_direction, cur_closest, found);

        if (!found) {
            std::cout << "not found!" << std::endl;
        }

        sorted_pts_vec.push_back(cur_closest);
        
        // really dumb method
        int row_num = 0;
        for (int i = 0; i < pts.rows(); i ++) {
            if (pt2pt_dis(pts.row(i), cur_closest) < 0.00001) {
                row_num = i;
                break;
            }
        }
        removeRow(pts, row_num);
    }

    if (!true_start) {
        sorted_pts_vec.insert(sorted_pts_vec.begin(), closest_2);

        int row_num = 0;
        for (int i = 0; i < pts.rows(); i ++) {
            if (pt2pt_dis(pts.row(i), closest_2) < 0.00001) {
                row_num = i;
                break;
            }
        }
        removeRow(pts, row_num);

        while (pts.rows() != 0) {
            MatrixXf cur_target = sorted_pts_vec[0];
            MatrixXf cur_direction = sorted_pts_vec[1];
            MatrixXf cur_closest;
            bool found;
            find_opposite_closest(cur_target, pts, cur_direction, cur_closest, found);
        
            if (!found) {
                std::cout << "not found!" << std::endl;
                break;
            }

            sorted_pts_vec.insert(sorted_pts_vec.begin(), cur_closest);

            int row_num = 0;
            for (int i = 0; i < pts.rows(); i ++) {
                if (pt2pt_dis(pts.row(i), cur_closest) < 0.00001) {
                    row_num = i;
                    break;
                }
            }
            removeRow(pts, row_num);
        }
    }

    // fill the eigen matrix
    for (int i = 0; i < sorted_pts.rows(); i ++) {
        sorted_pts.row(i) = sorted_pts_vec[i];
    }

    return sorted_pts;
}

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

MatrixXf calc_LLE_weights (int k, MatrixXf X) {
    MatrixXf W = MatrixXf::Zero(X.rows(), X.rows());
    for (int i = 0; i < X.rows(); i ++) {
        std::vector<int> indices = get_nearest_indices(static_cast<int>(k/2), X.rows(), i);
        MatrixXf xi = X.row(i);
        MatrixXf Xi = MatrixXf(indices.size(), X.cols());

        // fill in Xi: Xi = X[indices, :]
        for (int r = 0; r < indices.size(); r ++) {
            Xi.row(r) = X.row(indices[r]);
        }

        // component = np.full((len(Xi), len(xi)), xi).T - Xi.T
        MatrixXf component = xi.replicate(Xi.rows(), 1).transpose() - Xi.transpose();
        MatrixXf Gi = component.transpose() * component;
        MatrixXf Gi_inv;

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
        MatrixXf ones_row_vec = MatrixXf::Constant(1, Xi.rows(), 1.0);
        MatrixXf ones_col_vec = MatrixXf::Constant(Xi.rows(), 1, 1.0);

        MatrixXf wi = (Gi_inv * ones_col_vec) / (ones_row_vec * Gi_inv * ones_col_vec).value();
        MatrixXf wi_T = wi.transpose();

        for (int c = 0; c < indices.size(); c ++) {
            W(i, indices[c]) = wi_T(c);
        }
    }

    return W;
}

MatrixXf cpd (MatrixXf X_orig,
              MatrixXf& Y_0,
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
              MatrixXf correspondence_priors = MatrixXf::Zero(0, 0),
              double omega = 0,
              std::string kernel = "Gaussian",
              std::vector<int> occluded_nodes = {}) {

    // log time            
    clock_t cur_time = clock();

    MatrixXf X = X_orig.replicate(1, 1);

    int M = Y_0.rows();
    int N = X.rows();
    int D = 3;

    std::cout << "---" << std::endl;

    MatrixXf diff_yy = MatrixXf::Zero(M, M);
    MatrixXf diff_yy_sqrt = MatrixXf::Zero(M, M);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < M; j ++) {
            diff_yy(i, j) = (Y_0.row(i) - Y_0.row(j)).squaredNorm();
            diff_yy_sqrt(i, j) = (Y_0.row(i) - Y_0.row(j)).norm();
        }
    }

    // diff_xy should be a (M * N) matrix
    MatrixXf diff_xy = MatrixXf::Zero(M, N);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            diff_xy(i, j) = (Y_0.row(i) - X.row(j)).squaredNorm();
        }
    }

    MatrixXf converted_node_dis = MatrixXf::Zero(M, M); // this is a M*M matrix in place of diff_sqrt
    MatrixXf converted_node_dis_sq = MatrixXf::Zero(M, M);
    MatrixXf G = MatrixXf::Zero(M, M);
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

    MatrixXf Y = Y_0.replicate(1, 1);

    // initialize sigma2
    double sigma2 = 0;
    if (!use_prev_sigma2) {
        sigma2 = diff_xy.sum() / (static_cast<double>(D * M * N) / 1000);
    }
    else {
        sigma2 = sigma2_0;
    }

    // get the LLE matrix
    MatrixXf L = calc_LLE_weights(2, Y_0);
    MatrixXf H = (MatrixXf::Identity(M, M) - L).transpose() * (MatrixXf::Identity(M, M) - L);

    // TODO: implement node deletion from the original point cloud

    for (int it = 0; it < max_iter; it ++) {

        // update diff_xy
        for (int i = 0; i < M; i ++) {
            for (int j = 0; j < N; j ++) {
                diff_xy(i, j) = (Y_0.row(i) - X.row(j)).squaredNorm();
            }
        }

        MatrixXf P = (-0.5 * diff_xy / sigma2).array().exp();
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

        MatrixXf Pt1 = P.colwise().sum();
        MatrixXf P1 = P.rowwise().sum();
        double Np = P1.sum();
        MatrixXf PX = P * X;

        // M step
        MatrixXf A_matrix;
        MatrixXf B_matrix;
        if (include_lle) {
            A_matrix = P1.asDiagonal() * G + alpha * sigma2 * MatrixXf::Identity(M, M) + sigma2 * gamma * H * G;
            B_matrix = PX - P1.asDiagonal() * Y_0 - sigma2 * gamma * H * Y_0;
        }
        else {
            A_matrix = P1.asDiagonal() * G + alpha * sigma2 * MatrixXf::Identity(M, M);
            B_matrix = PX - P1.asDiagonal() * Y_0;
        }

        MatrixXf W = (A_matrix).householderQr().solve(B_matrix);

        MatrixXf T = Y_0 + G * W;
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

// int main(int argc, char **argv) {
//     ros::init (argc, argv, "test");
//     ros::NodeHandle nh;

//     MatrixXf m1 = MatrixXf::Zero(5, 3);
//     for (int i = 0; i < m1.rows(); i ++) {
//         for (int j = 0; j < m1.cols(); j ++) {
//             m1(i, j) = (static_cast<float>(i)*m1.cols() + static_cast<float>(j))/100;
//             // m1(i, j) *= m1(i, j);
//         }
//     }

//     MatrixXf m2 = MatrixXf::Zero(10, 3);
//     for (int i = 0; i < m2.rows(); i ++) {
//         for (int j = 0; j < m2.cols(); j ++) {
//             m2(i, j) = (static_cast<float>(i)*m2.cols() + static_cast<float>(j))/200;
//             // m1(i, j) *= m1(i, j);
//         }
//     }

//     std::cout << m1 << std::endl;
//     std::cout << m2 << std::endl;

//     // // ----- test LLE weights -----
//     // clock_t cur_time = clock();
//     // MatrixXf out = calc_LLE_weights(2, m1);
//     // std::cout << static_cast<double>(clock() - cur_time) / static_cast<double>(CLOCKS_PER_SEC) << std::endl;
//     // // std::cout << out << std::endl;

//     // // ----- test pt2pt_dis_sq -----
//     // MatrixXf m3 = m2.array() + 0.5;
//     // std::cout << m3 << std::endl;
//     // std::cout << pt2pt_dis_sq(m2, m3) << std::endl;

//     // ----- test ecpd -----
//     std::cout << cpd(m2, m1, 0.3, 1, 1, 0.1, 30, 0.00001, true, false, false, 0, false, MatrixXf(0, 0), 0, "Gaussian") << std::endl;
//     // print_1d_vector(cpd(m2, m1, 0.3, 1, 1, 0.05, 1, 0.00001, false, false, 0, false, MatrixXf(0, 0), 0, "Gaussian"));
// }