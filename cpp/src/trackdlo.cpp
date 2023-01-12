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
               std::vector<int> occluded_nodes = {},
               double k_vis = 0,
               Mat bmask_transformed_normalized = Mat::zeros(cv::Size(0, 0), CV_64F),
               double mat_max = 0) {

    // log time            
    clock_t cur_time = clock();
    bool converged = true;

    if (correspondence_priors.size() == 0) {
        use_ecpd = false;
    }

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
            cur_sum += pt2pt_dis(Y_0.row(i+1), Y_0.row(i));
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
            G = 27 * 1/(72 * pow(beta, 3)) * (-sqrt(3)*converted_node_dis/beta).array().exp() * (sqrt(3)*beta*beta + 3*beta*converted_node_dis.array() + sqrt(3)*converted_node_dis_sq.array());
        }
        else { // default to gaussian
            G = (-converted_node_dis_sq / (2 * beta * beta)).array().exp();
        }
    }

    // get the LLE matrix
    MatrixXf L = calc_LLE_weights(6, Y_0);
    MatrixXf H = (MatrixXf::Identity(M, M) - L).transpose() * (MatrixXf::Identity(M, M) - L);

    int N_orig = X.rows();

    // add correspondence priors to the set
    // this is different from the Python implementation; here the additional points are being appended at the end
    MatrixXf priors = MatrixXf::Zero(correspondence_priors.size(), 3);
    if (correspondence_priors.size() != 0) {
        int num_of_correspondence_priors = correspondence_priors.size();

        for (int i = 0; i < num_of_correspondence_priors; i ++) {
            MatrixXf temp = MatrixXf::Zero(1, 3);
            temp(0, 0) = correspondence_priors[i](0, 1);
            temp(0, 1) = correspondence_priors[i](0, 2);
            temp(0, 2) = correspondence_priors[i](0, 3);

            X.conservativeResize(X.rows() + 1, Eigen::NoChange);
            X.row(X.rows()-1) = temp;

            priors.row(i) = temp;
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
        sigma2 = diff_xy.sum() / static_cast<double>(D * M * N);
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

        if (use_geodesic) {
            std::vector<int> max_p_nodes(P.cols(), 0);

            // temp test
            for (int i = 0; i < N; i ++) {
                P.col(i).maxCoeff(&max_p_nodes[i]);
            }

            MatrixXf pts_dis_sq_geodesic = MatrixXf::Zero(M, N);

            // loop through all points
            for (int i = 0; i < N; i ++) {
                
                P.col(i).maxCoeff(&max_p_nodes[i]);
                int max_p_node = max_p_nodes[i];

                int potential_2nd_max_p_node_1 = max_p_node - 1;
                if (potential_2nd_max_p_node_1 == -1) {
                    potential_2nd_max_p_node_1 = 2;
                }

                int potential_2nd_max_p_node_2 = max_p_node + 1;
                if (potential_2nd_max_p_node_2 == M) {
                    potential_2nd_max_p_node_2 = M - 3;
                }

                int next_max_p_node;
                if (pt2pt_dis(Y.row(potential_2nd_max_p_node_1), X.row(i)) < pt2pt_dis(Y.row(potential_2nd_max_p_node_2), X.row(i))) {
                    next_max_p_node = potential_2nd_max_p_node_1;
                } 
                else {
                    next_max_p_node = potential_2nd_max_p_node_2;
                }

                // fill the current column of pts_dis_sq_geodesic
                pts_dis_sq_geodesic(max_p_node, i) = pt2pt_dis_sq(Y.row(max_p_node), X.row(i));
                pts_dis_sq_geodesic(next_max_p_node, i) = pt2pt_dis_sq(Y.row(next_max_p_node), X.row(i));

                if (max_p_node < next_max_p_node) {
                    for (int j = 0; j < max_p_node; j ++) {
                        pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[max_p_node]) + pt2pt_dis(Y.row(max_p_node), X.row(i)), 2);
                    }
                    for (int j = next_max_p_node; j < M; j ++) {
                        pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[next_max_p_node]) + pt2pt_dis(Y.row(next_max_p_node), X.row(i)), 2);
                    }
                }
                else {
                    for (int j = 0; j < next_max_p_node; j ++) {
                        pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[next_max_p_node]) + pt2pt_dis(Y.row(next_max_p_node), X.row(i)), 2);
                    }
                    for (int j = max_p_node; j < M; j ++) {
                        pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[max_p_node]) + pt2pt_dis(Y.row(max_p_node), X.row(i)), 2);
                    }
                }
            }

            // update P
            P = (-0.5 * pts_dis_sq_geodesic / sigma2).array().exp();
            // P = P.array().rowwise() / (P.colwise().sum().array() + c);
        }
        else {
            P = P_stored.replicate(1, 1);
        }

        
        // use cdcpd's pvis
        if (occluded_nodes.size() != 0 && mat_max != 0) {
            // if has corresponding guide node, use that instead of the original position
            MatrixXf nodes_h = Y.replicate(1, 1);
            for (auto entry : correspondence_priors) {
                nodes_h.row(entry(0, 0)) = entry.rightCols(3);
            }

            // project onto the bmask to find distance to closest none zero pixel
            nodes_h.conservativeResize(nodes_h.rows(), nodes_h.cols()+1);
            nodes_h.col(nodes_h.cols()-1) = MatrixXf::Ones(nodes_h.rows(), 1);
            MatrixXf proj_matrix(3, 4);
            proj_matrix << 918.359130859375, 0.0, 645.8908081054688, 0.0,
                            0.0, 916.265869140625, 354.02392578125, 0.0,
                            0.0, 0.0, 1.0, 0.0;
            MatrixXf image_coords = (proj_matrix * nodes_h.transpose()).transpose();

            MatrixXf P_vis = MatrixXf::Ones(P.rows(), P.cols());
            double total_P_vis = 0;
            for (int i = 0; i < image_coords.rows(); i ++) {
                int x = static_cast<int>(image_coords(i, 0)/image_coords(i, 2));
                int y = static_cast<int>(image_coords(i, 1)/image_coords(i, 2));

                double pixel_dist = static_cast<double>(bmask_transformed_normalized.at<uchar>(y, x)) * mat_max / 255;
                double P_vis_i = exp(-k_vis*pixel_dist);
                total_P_vis += P_vis_i;

                // // test
                // if (P_vis_i < 1e-10) {
                //     P_vis_i = 0;
                // }

                P_vis.row(i) = P_vis_i * P_vis.row(i);
            }

            // normalize P_vis
            P_vis = P_vis / total_P_vis;

            // modify P
            P = P.cwiseProduct(P_vis);

            // modify c
            c = pow((2 * M_PI * sigma2), static_cast<double>(D)/2) * mu / (1 - mu) / N;
            P = P.array().rowwise() / (P.colwise().sum().array() + c);
        }
        else {
            P = P.array().rowwise() / (P.colwise().sum().array() + c);
        }


        // old
        // P = P.array().rowwise() / (P.colwise().sum().array() + c);
        // // quick test
        // if (occluded_nodes.size() != 0) {
        //     for (int i = 0; i < occluded_nodes.size(); i ++) {
        //         P.row(occluded_nodes[i]) = MatrixXf::Zero(1, N);
        //     }
        // }

        MatrixXf Pt1 = P.colwise().sum();  // this should have shape (N,) or (1, N)
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

        // MatrixXf W = A_matrix.householderQr().solve(B_matrix);
        MatrixXf W = A_matrix.completeOrthogonalDecomposition().solve(B_matrix);

        MatrixXf T = Y_0 + G * W;
        double trXtdPt1X = (X.transpose() * Pt1.asDiagonal() * X).trace();
        double trPXtT = (PX.transpose() * T).trace();
        double trTtdP1T = (T.transpose() * P1.asDiagonal() * T).trace();

        sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D);

        if (pt2pt_dis_sq(Y, Y_0 + G*W) < tol) {
            Y = Y_0 + G*W;
            ROS_INFO_STREAM("Iteration until convergence: " + std::to_string(it+1));
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

void tracking_step (MatrixXf X_orig,
                    MatrixXf& Y,
                    double& sigma2,
                    MatrixXf& guide_nodes,
                    std::vector<MatrixXf>& priors_vec,
                    std::vector<double> geodesic_coord,
                    Mat bmask_transformed_normalized,
                    double mask_dist_threshold,
                    double mat_max) {
    
    // variable initialization
    std::vector<int> occluded_nodes = {};
    std::vector<int> visible_nodes = {};
    std::vector<MatrixXf> valid_nodes_vec = {};
    int state = 0;

    // project Y onto the original image to determine occluded nodes
    MatrixXf nodes_h = Y.replicate(1, 1);
    nodes_h.conservativeResize(nodes_h.rows(), nodes_h.cols()+1);
    nodes_h.col(nodes_h.cols()-1) = MatrixXf::Ones(nodes_h.rows(), 1);
    MatrixXf proj_matrix(3, 4);
    proj_matrix << 918.359130859375, 0.0, 645.8908081054688, 0.0,
                    0.0, 916.265869140625, 354.02392578125, 0.0,
                    0.0, 0.0, 1.0, 0.0;
    MatrixXf image_coords = (proj_matrix * nodes_h.transpose()).transpose();
    for (int i = 0; i < image_coords.rows(); i ++) {
        int x = static_cast<int>(image_coords(i, 0)/image_coords(i, 2));
        int y = static_cast<int>(image_coords(i, 1)/image_coords(i, 2));

        // not currently using the original distance transform because I can't figure it out
        if (static_cast<int>(bmask_transformed_normalized.at<uchar>(y, x)) < mask_dist_threshold / mat_max * 255) {
            valid_nodes_vec.push_back(Y.row(i));
            visible_nodes.push_back(i);
        }
        else {
            occluded_nodes.push_back(i);
        }
    }

    if (valid_nodes_vec.size() == 0) {
        ROS_ERROR("The current state is too different from the last state!");
        ecpd_lle (X_orig, Y, sigma2, 6, 1, 10, 0.05, 50, 0.00001, true, true, true, false, {}, 0.0, "1st order", occluded_nodes, 0.02, bmask_transformed_normalized, mat_max);
        return;
    }

    // copy valid guide nodes vec to guide nodes
    // not using topRows() because it caused weird bugs
    guide_nodes = MatrixXf::Zero(valid_nodes_vec.size(), 3);
    for (int i = 0; i < valid_nodes_vec.size(); i ++) {
        guide_nodes.row(i) = valid_nodes_vec[i];
    }

    // run rigid registration on guide nodes and X
    double sigma2_pre_proc = sigma2;
    ecpd_lle (X_orig, guide_nodes, sigma2_pre_proc, 10000, 1, 2, 0.05, 50, 0.000000001, false, true, true, false);

    // copy guide nodes to priors_vec (this could be combined into one step in the future)
    priors_vec = {};
    for (int i = 0; i < guide_nodes.rows(); i ++) {
        MatrixXf temp = MatrixXf::Zero(1, 4);
        temp(0, 0) = visible_nodes[i];
        temp(0, 1) = guide_nodes(i, 0);
        temp(0, 2) = guide_nodes(i, 1);
        temp(0, 3) = guide_nodes(i, 2);
        priors_vec.push_back(temp);
    }

    // ----- for quick test -----

    // params for eval rope (short)
    ecpd_lle (X_orig, Y, sigma2, 10, 1, 1, 0.05, 50, 0.00001, true, true, true, true, priors_vec, 0.000006, "1st order", occluded_nodes, 0.01, bmask_transformed_normalized, mat_max);

    // params for long rope
    // ecpd_lle (X_orig, Y, sigma2, 6, 1, 2, 0.05, 50, 0.00001, true, true, true, true, priors_vec, 0.00001, "1st order", occluded_nodes, 0.02, bmask_transformed_normalized, mat_max);

    // test Gaussian
    // ecpd_lle (X_orig, Y, sigma2, 2, 1, 10, 0.05, 50, 0.00001, true, true, true, true, priors_vec, 0.000006, "Gaussian", occluded_nodes, 0.02, bmask_transformed_normalized, mat_max);
}