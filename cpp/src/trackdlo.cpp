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
#include <pcl/filters/conditional_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>

// typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVector;
// typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ColumnVector;

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::RowVectorXf;
using Eigen::RowVectorXd;
using cv::Mat;

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
void remove_row(MatrixXf& matrix, unsigned int rowToRemove) {
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
        remove_row(arr_copy, cur_index);

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
    remove_row(pts, start_idx);

    // starting point will be the current first point in the new list
    MatrixXf sorted_pts = MatrixXf::Zero(pts_orig.rows(), pts_orig.cols());
    std::vector<MatrixXf> sorted_pts_vec;
    sorted_pts_vec.push_back(starting_pt);

    // get the first closest point
    MatrixXf closest_1;
    int min_idx;
    find_closest(starting_pt, pts, closest_1, min_idx);
    sorted_pts_vec.push_back(closest_1);
    remove_row(pts, min_idx);

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
            // std::cout << "not found!" << std::endl;
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
        remove_row(pts, row_num);
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
        remove_row(pts, row_num);

        while (pts.rows() != 0) {
            MatrixXf cur_target = sorted_pts_vec[0];
            MatrixXf cur_direction = sorted_pts_vec[1];
            MatrixXf cur_closest;
            bool found;
            find_opposite_closest(cur_target, pts, cur_direction, cur_closest, found);
        
            if (!found) {
                // std::cout << "not found!" << std::endl;
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
            remove_row(pts, row_num);
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
            double epsilon = 0.00001;
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

bool ecpd_lle (MatrixXf X_orig,
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
               std::vector<MatrixXf> correspondence_priors = {},
               double omega = 0,
               std::string kernel = "Gaussian",
               std::vector<int> occluded_nodes = {}) {

    // log time            
    clock_t cur_time = clock();
    bool converged = true;

    MatrixXf X = X_orig.replicate(1, 1);

    int M = Y.rows();
    int N = X.rows();
    int D = 3;

    MatrixXf Y_0 = Y.replicate(1, 1);

    MatrixXf diff_yy = MatrixXf::Zero(M, M);
    MatrixXf diff_yy_sqrt = MatrixXf::Zero(M, M);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < M; j ++) {
            diff_yy(i, j) = (Y_0.row(i) - Y_0.row(j)).squaredNorm();
            diff_yy_sqrt(i, j) = (Y_0.row(i) - Y_0.row(j)).norm();
        }
    }

    MatrixXf converted_node_dis = MatrixXf::Zero(M, M); // this is a M*M matrix in place of diff_sqrt
    MatrixXf converted_node_dis_sq = MatrixXf::Zero(M, M);
    std::vector<double> converted_node_coord = {0.0};   // this is not squared

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

    // get the LLE matrix
    MatrixXf L = calc_LLE_weights(6, Y_0);
    MatrixXf H = (MatrixXf::Identity(M, M) - L).transpose() * (MatrixXf::Identity(M, M) - L);

    // point deletion from the original point cloud
    MatrixXf X_temp = MatrixXf::Zero(N, 3);
    if (occluded_nodes.size() != 0) {
        std::vector<int> max_p_nodes(N, 0);
        MatrixXf diff_xy = MatrixXf::Zero(M, N);

        // update diff_xy
        for (int i = 0; i < M; i ++) {
            for (int j = 0; j < N; j ++) {
                diff_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
            }
        }

        MatrixXf P = (-0.5 * diff_xy / sigma2).array().exp();
        double c = pow((2 * M_PI * sigma2), static_cast<double>(D)/2) * mu / (1 - mu) * static_cast<double>(M)/N;
        P = P.array().rowwise() / (P.colwise().sum().array() + c);

        int M_head = occluded_nodes[0];
        int M_tail = M - 1 - occluded_nodes[occluded_nodes.size()-1];

        int X_temp_counter = 0;

        for (int i = 0; i < N; i ++) {
            P.col(i).maxCoeff(&max_p_nodes[i]);
            int max_p_node = max_p_nodes[i];

            // critical nodes: M_head and M-M_tail-1
            if (max_p_node != M_head && max_p_node != (M-M_tail-1)) {
                X_temp.row(X_temp_counter) = X.row(i);
                X_temp_counter += 1;
            }
        }

        // std::cout << "X original len: " << X.rows() << std::endl;
        // X = X_temp.topRows(X_temp_counter);
        // std::cout << "X after deletion len: " << X.rows() << std::endl;
    }

    int N_orig = X.rows();

    // add correspondence priors to the set
    // this is different from the Python implementation; here the additional points are being appended at the end
    if (correspondence_priors.size() != 0) {
        int num_of_correspondence_priors = correspondence_priors.size();

        for (int i = 0; i < num_of_correspondence_priors; i ++) {
            MatrixXf temp = MatrixXf::Zero(1, 3);
            temp(0, 0) = correspondence_priors[i](0, 1);
            temp(0, 1) = correspondence_priors[i](0, 2);
            temp(0, 2) = correspondence_priors[i](0, 3);

            X.conservativeResize(X.rows() + 1, Eigen::NoChange);
            X.row(X.rows()-1) = temp;
        }
    }

    // update N
    N = X.rows();

    // diff_xy should be a (M * N) matrix
    MatrixXf diff_xy = MatrixXf::Zero(M, N);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            diff_xy(i, j) = (Y_0.row(i) - X.row(j)).squaredNorm();
        }
    }

    // initialize sigma2
    if (!use_prev_sigma2 || sigma2 == 0) {
        sigma2 = diff_xy.sum() / (static_cast<double>(D * M * N) / 1000);
    }

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

        std::vector<int> max_p_nodes(P.cols(), 0);

        // temp test
        for (int i = 0; i < N; i ++) {
            P.col(i).maxCoeff(&max_p_nodes[i]);
        }

        // if (use_geodesic) {
        //     MatrixXf pts_dis_sq_geodesic = MatrixXf::Zero(M, N);

        //     // loop through all points
        //     for (int i = 0; i < N; i ++) {
                
        //         P.col(i).maxCoeff(&max_p_nodes[i]);
        //         int max_p_node = max_p_nodes[i];

        //         int potential_2nd_max_p_node_1 = max_p_node - 1;
        //         if (potential_2nd_max_p_node_1 == -1) {
        //             potential_2nd_max_p_node_1 = 2;
        //         }

        //         int potential_2nd_max_p_node_2 = max_p_node + 1;
        //         if (potential_2nd_max_p_node_2 == M) {
        //             potential_2nd_max_p_node_2 = M - 3;
        //         }

        //         int next_max_p_node;
        //         if (pt2pt_dis(Y.row(potential_2nd_max_p_node_1), X.row(i)) < pt2pt_dis(Y.row(potential_2nd_max_p_node_2), X.row(i))) {
        //             next_max_p_node = potential_2nd_max_p_node_1;
        //         } 
        //         else {
        //             next_max_p_node = potential_2nd_max_p_node_2;
        //         }

        //         // fill the current column of pts_dis_sq_geodesic
        //         pts_dis_sq_geodesic(max_p_node, i) = pt2pt_dis_sq(Y.row(max_p_node), X.row(i));
        //         pts_dis_sq_geodesic(next_max_p_node, i) = pt2pt_dis_sq(Y.row(next_max_p_node), X.row(i));

        //         if (max_p_node < next_max_p_node) {
        //             for (int j = 0; j < max_p_node; j ++) {
        //                 pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[max_p_node]) + pt2pt_dis(Y.row(max_p_node), X.row(i)), 2);
        //             }
        //             for (int j = next_max_p_node; j < M; j ++) {
        //                 pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[next_max_p_node]) + pt2pt_dis(Y.row(next_max_p_node), X.row(i)), 2);
        //             }
        //         }
        //         else {
        //             for (int j = 0; j < next_max_p_node; j ++) {
        //                 pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[next_max_p_node]) + pt2pt_dis(Y.row(next_max_p_node), X.row(i)), 2);
        //             }
        //             for (int j = max_p_node; j < M; j ++) {
        //                 pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[max_p_node]) + pt2pt_dis(Y.row(max_p_node), X.row(i)), 2);
        //             }
        //         }
        //     }

        //     // update P
        //     P = (-0.5 * pts_dis_sq_geodesic / sigma2).array().exp();
        //     // P = P.array().rowwise() / (P.colwise().sum().array() + c);
        // }
        // else {
        //     P = P_stored.replicate(1, 1);
        // }

        // // temp test
        // P = P_stored.replicate(1, 1);

        // if (occluded_nodes.size() != 0) {

        //     ROS_INFO("modified membership probability");

        //     MatrixXf P_vis = MatrixXf::Zero(M, N);

        //     int M_head = occluded_nodes[0];
        //     int M_tail = M - 1 - occluded_nodes[occluded_nodes.size()-1];

        //     MatrixXf P_vis_fill_head = MatrixXf::Zero(M, 1);
        //     MatrixXf P_vis_fill_tail = MatrixXf::Zero(M, 1);
        //     MatrixXf P_vis_fill_floating = MatrixXf::Zero(M, 1);

        //     for (int i = 0; i < M; i ++) {
        //         if (i < M_head) {
        //             P_vis_fill_head(i, 0) = 1.0 / static_cast<double>(M_head);
        //         }
        //         else if (M_head <= i && i < (M - M_tail)) {
        //             P_vis_fill_floating(i, 0) = 1.0 / static_cast<double>(M - M_head - M_tail);
        //         }
        //         else {
        //             P_vis_fill_tail(i, 0) = 1.0 / static_cast<double>(M_tail);
        //         }
        //     }

        //     // fill in P_vis
        //     for (int i = 0; i < N; i ++) {
        //         int cur_max_p_node = max_p_nodes[i];

        //         if (cur_max_p_node >= 0 && cur_max_p_node < M_head) {
        //             P_vis.col(i) = P_vis_fill_head;
        //         }
        //         else if (cur_max_p_node >= M_head && cur_max_p_node < (M-M_tail)) {
        //             P_vis.col(i) = P_vis_fill_floating;
        //         }
        //         else {
        //             P_vis.col(i) = P_vis_fill_tail;
        //         }
        //     }

        //     // modify P
        //     P = P.cwiseProduct(P_vis);

        //     // modify c
        //     c = pow((2 * M_PI * sigma2), static_cast<double>(D)/2) * mu / (1 - mu) / N;
        //     P = P.array().rowwise() / (P.colwise().sum().array() + c);
        // }
        // else {
        //     P = P.array().rowwise() / (P.colwise().sum().array() + c);
        // }

        // // old
        // if (occluded_nodes.size() != 0) {
        //     for (int i = 0; i < occluded_nodes.size(); i ++) {
        //         P.row(occluded_nodes[i]) = MatrixXf::Zero(1, N);
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
            if (use_ecpd) {
                MatrixXf P_tilde = MatrixXf::Zero(M, N);
                // correspondence priors: index, x, y, z
                for (int i = 0; i < correspondence_priors.size(); i ++) {
                    int index = static_cast<int>(correspondence_priors[i](0, 0));
                    P_tilde(index, i + N_orig) = 1;
                }

                MatrixXf P_tilde_1 = P_tilde.rowwise().sum();
                MatrixXf P_tilde_X = P_tilde * X;

                A_matrix = P1.asDiagonal()*G + alpha*sigma2 * MatrixXf::Identity(M, M) + sigma2*gamma * H*G + sigma2/omega * P_tilde_1.asDiagonal()*G;
                B_matrix = PX - P1.asDiagonal()*Y_0 - sigma2*gamma * H*Y_0 + sigma2/omega * (P_tilde_X - (P_tilde_1.asDiagonal() * Y_0 + sigma2*gamma*H * Y_0));
            }
            else {
                A_matrix = P1.asDiagonal()*G + alpha*sigma2 * MatrixXf::Identity(M, M) + sigma2*gamma * H*G;
                B_matrix = PX - P1.asDiagonal()*Y_0 - sigma2*gamma * H*Y_0;
            }
        }
        else {
            if (use_ecpd) {
                MatrixXf P_tilde = MatrixXf::Zero(M, N);
                // correspondence priors: index, x, y, z
                for (int i = 0; i < correspondence_priors.size(); i ++) {
                    int index = static_cast<int>(correspondence_priors[i](0, 0));
                    P_tilde(index, i + N_orig) = 1;
                }

                MatrixXf P_tilde_1 = P_tilde.rowwise().sum();
                MatrixXf P_tilde_X = P_tilde * X;

                A_matrix = P1.asDiagonal() * G + alpha * sigma2 * MatrixXf::Identity(M, M) + sigma2/omega * P_tilde_1.asDiagonal()*G;
                B_matrix = PX - P1.asDiagonal() * Y_0 + sigma2/omega * (P_tilde_X - P_tilde_1.asDiagonal()*Y_0);
            }
            else {
                A_matrix = P1.asDiagonal() * G + alpha * sigma2 * MatrixXf::Identity(M, M);
                B_matrix = PX - P1.asDiagonal() * Y_0;
            }
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
            // ROS_INFO("Iteration until convergence: " + std::to_string(it));
            break;
        }
        else {
            Y = Y_0 + G*W;
        }

        if (it == max_iter - 1) {
            ROS_ERROR("optimization did not converge!");
            converged = false;
            break;
        }
    }
    
    return converged;
}

std::vector<MatrixXf> tracking_step (MatrixXf X_orig,
                                    MatrixXf& Y,
                                    double& sigma2,
                                    std::vector<double> geodesic_coord,
                                    double total_len,
                                    Mat bmask,
                                    Mat bmask_transformed_normalized,
                                    double mask_dist_threshold) {

    MatrixXf guide_nodes = Y.replicate(1, 1);
    double sigma2_pre_proc = 0;
    ecpd_lle (X_orig, guide_nodes, sigma2_pre_proc, 0.3, 1, 2, 0.05, 50, 0.00001, true, true);

    bool head_visible = false;
    bool tail_visible = false;

    if (pt2pt_dis(guide_nodes.row(0), Y.row(0)) < 0.01) {
        head_visible = true;
    }
    if (pt2pt_dis(guide_nodes.row(guide_nodes.rows()-1), Y.row(Y.rows()-1)) < 0.01) {
        tail_visible = true;
    }

    MatrixXf priors;
    std::vector<MatrixXf> priors_vec = {};
    std::vector<int> occluded_nodes = {};

    int state = 0;

    MatrixXf nodes_h = guide_nodes.replicate(1, 1);
    nodes_h.conservativeResize(nodes_h.rows(), nodes_h.cols()+1);
    nodes_h.col(nodes_h.cols()-1) = MatrixXf::Ones(nodes_h.rows(), 1);
    MatrixXf proj_matrix(3, 4);
    proj_matrix << 918.359130859375, 0.0, 645.8908081054688, 0.0,
                    0.0, 916.265869140625, 354.02392578125, 0.0,
                    0.0, 0.0, 1.0, 0.0;
    MatrixXf image_coords = (proj_matrix * nodes_h.transpose()).transpose();

    std::vector<int> valid_guide_nodes_indices;
    for (int i = 0; i < image_coords.rows(); i ++) {
        int x = static_cast<int>(image_coords(i, 0)/image_coords(i, 2));
        int y = static_cast<int>(image_coords(i, 1)/image_coords(i, 2));

        // not currently using the distance transform because I can't figure it out
        if (static_cast<int>(bmask_transformed_normalized.at<uchar>(y, x)) < mask_dist_threshold) {
            valid_guide_nodes_indices.push_back(i);
        }
        // else {
        //     if (i == 0) {head_visible = false;}
        //     if (i == image_coords.rows()-1) {tail_visible = false;}
        // }
    }

    if (!head_visible && !tail_visible) {
        // if head node is within the mask threshold
        if (valid_guide_nodes_indices[0] == 0) {
            if (pt2pt_dis(guide_nodes.row(0), Y.row(0)) < pt2pt_dis(guide_nodes.row(Y.rows()-1), Y.row(Y.rows()-1))) {
                head_visible = true;
            }
        }
        // if tail node is within the mask threshold
        if (valid_guide_nodes_indices[valid_guide_nodes_indices.size()-1] == Y.rows()-1) {
            if (pt2pt_dis(guide_nodes.row(0), Y.row(0)) > pt2pt_dis(guide_nodes.row(Y.rows()-1), Y.row(Y.rows()-1))) {
                tail_visible = true;
            }
        }
    }

    // print_1d_vector(valid_guide_nodes_indices);

    if (head_visible && tail_visible) {

        state = 2;

        std::vector<MatrixXf> valid_head_nodes;
        std::vector<int> valid_head_node_indices;
        for (int i = 0; i < guide_nodes.rows(); i ++) {
            if (i == valid_guide_nodes_indices[i]) {
                valid_head_nodes.push_back(guide_nodes.row(i));
                valid_head_node_indices.push_back(i);
            }
            else {
                break;
            }

            if (i == valid_guide_nodes_indices.size()-1) {
                break;
            }
        }

        std::vector<MatrixXf> valid_tail_nodes;
        std::vector<int> valid_tail_node_indices;
        for (int i = 0; i < guide_nodes.rows(); i ++) {
            if (guide_nodes.rows()-1-i == valid_guide_nodes_indices[valid_guide_nodes_indices.size()-1-i]) {
                valid_tail_nodes.insert(valid_tail_nodes.begin(), guide_nodes.row(guide_nodes.rows()-1-i));
                valid_tail_node_indices.insert(valid_tail_node_indices.begin(), guide_nodes.rows()-1-i);
            }
            else {
                break;
            }

            if (valid_guide_nodes_indices.size()-1-i == 0) {
                break;
            }
        }

        // ---- head visible part -----
        double total_dist_Y = 0;
        double total_dist_guide_nodes = 0;
        int it_gn = 0;
        int last_visible_index_head = -1;

        if (valid_head_nodes.size() != 0) {
            MatrixXf temp(1, 4);
            temp << 0, valid_head_nodes[0](0, 0), valid_head_nodes[0](0, 1), valid_head_nodes[0](0, 2);
            priors_vec.push_back(temp);
        }

        for (int it_y0 = 0; it_y0 < valid_head_nodes.size()-1; it_y0 ++) {
            total_dist_Y += abs(geodesic_coord[it_y0] - geodesic_coord[it_y0+1]);

            while (total_dist_guide_nodes < total_dist_Y) {
                total_dist_guide_nodes += pt2pt_dis(valid_head_nodes[it_gn], valid_head_nodes[it_gn+1]);
                if (total_dist_guide_nodes >= total_dist_Y) {
                    total_dist_guide_nodes -= pt2pt_dis(valid_head_nodes[it_gn], valid_head_nodes[it_gn+1]);
                    MatrixXf new_y0_coord = valid_head_nodes[it_gn] + (total_dist_Y - total_dist_guide_nodes)/pt2pt_dis(valid_head_nodes[it_gn], valid_head_nodes[it_gn + 1])
                                            * (valid_head_nodes[it_gn + 1] - valid_head_nodes[it_gn]);
                    MatrixXf temp(1, 4);
                    temp << it_y0+1, new_y0_coord(0, 0), new_y0_coord(0, 1), new_y0_coord(0, 2);
                    priors_vec.push_back(temp);
                    break;
                }
                // if at the end of guide nodes
                if (it_gn == valid_head_nodes.size() - 2) {
                    last_visible_index_head = it_y0;
                    break;
                }
                it_gn += 1;
            }

            if (last_visible_index_head != -1) {
                break;
            }
        }
        if (last_visible_index_head == -1) {
            last_visible_index_head = valid_head_nodes.size() - 1;
        }

        // ----- tail visible part -----
        total_dist_Y = 0;
        total_dist_guide_nodes = 0;
        it_gn = valid_tail_nodes.size() - 1;
        int last_visible_index_tail = -1;

        if (valid_tail_nodes.size() != 0) {
            MatrixXf temp(1, 4);
            temp << valid_tail_nodes.size()-1 + valid_tail_node_indices[0], valid_tail_nodes[valid_tail_nodes.size()-1](0, 0), valid_tail_nodes[valid_tail_nodes.size()-1](0, 1), valid_tail_nodes[valid_tail_nodes.size()-1](0, 2);
            priors_vec.push_back(temp);
        }

        for (int it_y0 = valid_tail_nodes.size()-1; it_y0 > 0; it_y0 --) {
            total_dist_Y += abs(geodesic_coord[it_y0] - geodesic_coord[it_y0-1]);
            while (total_dist_guide_nodes < total_dist_Y) {
                total_dist_guide_nodes += pt2pt_dis(valid_tail_nodes[it_gn], valid_tail_nodes[it_gn-1]);
                if (total_dist_guide_nodes >= total_dist_Y) {
                    total_dist_guide_nodes -= pt2pt_dis(valid_tail_nodes[it_gn], valid_tail_nodes[it_gn-1]);
                    MatrixXf new_y0_coord = valid_tail_nodes[it_gn] + (total_dist_Y - total_dist_guide_nodes)/pt2pt_dis(valid_tail_nodes[it_gn], valid_tail_nodes[it_gn - 1])
                                            * (valid_tail_nodes[it_gn - 1] - valid_tail_nodes[it_gn]);
                    MatrixXf temp(1, 4);
                    temp << it_y0-1 + valid_tail_node_indices[0], new_y0_coord(0, 0), new_y0_coord(0, 1), new_y0_coord(0, 2);
                    priors_vec.push_back(temp);
                    break;
                }
                // if at the end of guide nodes
                if (it_gn == 1) {
                    last_visible_index_tail = it_y0 + valid_tail_node_indices[0];
                    break;
                }
                it_gn -= 1;
            }
            if (last_visible_index_tail != -1) {
                break;
            }
        }
        if (last_visible_index_tail == -1) {
            last_visible_index_tail = 0;
        }

        for (int i = last_visible_index_head+1; i < last_visible_index_tail; i ++) {
            occluded_nodes.push_back(i);
        }

        // std::cout << last_visible_index_head << ", " << last_visible_index_tail << std::endl;
    }
    
    else if (head_visible && (!tail_visible)) {

        state = 1;

        std::vector<MatrixXf> valid_head_nodes;
        std::vector<int> valid_head_node_indices;
        for (int i = 0; i < guide_nodes.rows(); i ++) {
            if (i == valid_guide_nodes_indices[i]) {
                valid_head_nodes.push_back(guide_nodes.row(i));
                valid_head_node_indices.push_back(i);
            }
            else {
                break;
            }

            if (i == valid_guide_nodes_indices.size()-1) {
                break;
            }
        }

        double total_dist_Y = 0;
        double total_dist_guide_nodes = 0;
        int it_gn = 0;
        int last_visible_index_head = -1;

        if (valid_head_nodes.size() != 0) {
            MatrixXf temp(1, 4);
            temp << 0, valid_head_nodes[0](0, 0), valid_head_nodes[0](0, 1), valid_head_nodes[0](0, 2);
            priors_vec.push_back(temp);
        }

        for (int it_y0 = 0; it_y0 < valid_head_nodes.size()-1; it_y0 ++) {
            total_dist_Y += abs(geodesic_coord[it_y0] - geodesic_coord[it_y0+1]);

            while (total_dist_guide_nodes < total_dist_Y) {
                total_dist_guide_nodes += pt2pt_dis(valid_head_nodes[it_gn], valid_head_nodes[it_gn+1]);
                if (total_dist_guide_nodes >= total_dist_Y) {
                    total_dist_guide_nodes -= pt2pt_dis(valid_head_nodes[it_gn], valid_head_nodes[it_gn+1]);
                    MatrixXf new_y0_coord = valid_head_nodes[it_gn] + (total_dist_Y - total_dist_guide_nodes)/pt2pt_dis(valid_head_nodes[it_gn], valid_head_nodes[it_gn + 1])
                                            * (valid_head_nodes[it_gn + 1] - valid_head_nodes[it_gn]);
                    MatrixXf temp(1, 4);
                    temp << it_y0+1, new_y0_coord(0, 0), new_y0_coord(0, 1), new_y0_coord(0, 2);
                    priors_vec.push_back(temp);
                    break;
                }
                // if at the end of guide nodes
                if (it_gn == valid_head_nodes.size() - 2) {
                    last_visible_index_head = it_y0;
                    break;
                }
                it_gn += 1;
            }

            if (last_visible_index_head != -1) {
                break;
            }
        }
        if (last_visible_index_head == -1) {
            last_visible_index_head = valid_head_nodes.size() - 1;
        }

        for (int i = last_visible_index_head+1; i < guide_nodes.rows(); i ++) {
            occluded_nodes.push_back(i);
        }
    }

    else if ((!head_visible) && tail_visible) {

        state = 1;

        std::vector<MatrixXf> valid_tail_nodes;
        std::vector<int> valid_tail_node_indices;
        for (int i = 0; i < guide_nodes.rows(); i ++) {
            if (guide_nodes.rows()-1-i == valid_guide_nodes_indices[valid_guide_nodes_indices.size()-1-i]) {
                valid_tail_nodes.insert(valid_tail_nodes.begin(), guide_nodes.row(guide_nodes.rows()-1-i));
                valid_tail_node_indices.insert(valid_tail_node_indices.begin(), guide_nodes.rows()-1-i);
            }
            else {
                break;
            }

            if (valid_guide_nodes_indices.size()-1-i == 0) {
                break;
            }
        }

        double total_dist_Y = 0;
        double total_dist_guide_nodes = 0;
        int it_gn = valid_tail_nodes.size() - 1;
        int last_visible_index_tail = -1;

        if (valid_tail_nodes.size() != 0) {
            MatrixXf temp(1, 4);
            temp << valid_tail_nodes.size()-1 + valid_tail_node_indices[0], valid_tail_nodes[valid_tail_nodes.size()-1](0, 0), valid_tail_nodes[valid_tail_nodes.size()-1](0, 1), valid_tail_nodes[valid_tail_nodes.size()-1](0, 2);
            priors_vec.push_back(temp);
        }

        for (int it_y0 = valid_tail_nodes.size()-1; it_y0 > 0; it_y0 --) {
            total_dist_Y += abs(geodesic_coord[it_y0] - geodesic_coord[it_y0-1]);
            while (total_dist_guide_nodes < total_dist_Y) {
                total_dist_guide_nodes += pt2pt_dis(valid_tail_nodes[it_gn], valid_tail_nodes[it_gn-1]);
                if (total_dist_guide_nodes >= total_dist_Y) {
                    total_dist_guide_nodes -= pt2pt_dis(valid_tail_nodes[it_gn], valid_tail_nodes[it_gn-1]);
                    MatrixXf new_y0_coord = valid_tail_nodes[it_gn] + (total_dist_Y - total_dist_guide_nodes)/pt2pt_dis(valid_tail_nodes[it_gn], valid_tail_nodes[it_gn - 1])
                                            * (valid_tail_nodes[it_gn - 1] - valid_tail_nodes[it_gn]);
                    MatrixXf temp(1, 4);
                    temp << it_y0-1 + valid_tail_node_indices[0], new_y0_coord(0, 0), new_y0_coord(0, 1), new_y0_coord(0, 2);
                    priors_vec.push_back(temp);
                    break;
                }
                // if at the end of guide nodes
                if (it_gn == 1) {
                    last_visible_index_tail = it_y0 + valid_tail_node_indices[0];
                    break;
                }
                it_gn -= 1;
            }
            if (last_visible_index_tail != -1) {
                break;
            }
        }
        if (last_visible_index_tail == -1) {
            last_visible_index_tail = 0;
        }

        for (int i = 0; i < last_visible_index_tail; i ++) {
            occluded_nodes.push_back(i);
        }
    }

    else {
        ROS_ERROR("Neither tip visible!");
    }

    // // visualization for debug
    // Mat mask_rgb;
    // cv::cvtColor(bmask, mask_rgb, cv::COLOR_GRAY2BGR);
    // std::cout << "before draw image" << std::endl;
    // for (MatrixXf prior : priors_vec) {
    //     MatrixXf proj_matrix(3, 4);
    //     proj_matrix << 918.359130859375, 0.0, 645.8908081054688, 0.0,
    //                     0.0, 916.265869140625, 354.02392578125, 0.0,
    //                     0.0, 0.0, 1.0, 0.0;
    //     MatrixXf prior_h(1, 4);
    //     prior_h << prior(0, 1), prior(0, 2), prior(0, 3), 1.0;
    //     std::cout << prior_h << std::endl;

    //     MatrixXf image_coord = (proj_matrix * prior_h.transpose()).transpose();
    //     std::cout << image_coord << std::endl;

    //     int x = static_cast<int>(image_coord(0, 0)/image_coord(0, 2));
    //     int y = static_cast<int>(image_coord(0, 1)/image_coord(0, 2));
    //     std::cout << x << ", " << y << std::endl;

    //     cv::circle(mask_rgb, cv::Point(x, y), 5, cv::Scalar(0, 200, 0), -1);
    // }
    // std::cout << "after draw image" << std::endl;

    // cv::imshow("frame", mask_rgb);
    // cv::waitKey(3);

    if (state == 2) {
        ecpd_lle (X_orig, Y, sigma2, 1, 1, 2, 0.05, 50, 0.00001, false, true, true, true, priors_vec, 0.0001, "Gaussian", occluded_nodes);
    }
    else if (state == 1) {
        ecpd_lle (X_orig, Y, sigma2, 5, 1, 2, 0.05, 50, 0.00001, false, true, true, true, priors_vec, 0.00001, "1st order", occluded_nodes);
    }  
    else {
        ROS_ERROR("Not a valid state!");
    }

    return priors_vec;
}