#include "../include/utils.h"
#include "../include/trackdlo.h"

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::RowVectorXf;
using Eigen::RowVectorXd;
using cv::Mat;

trackdlo::trackdlo () {}

trackdlo::trackdlo(int num_of_nodes) {
    // default initialize
    Y_ = MatrixXf::Zero(num_of_nodes, 3);
    guide_nodes_ = Y_.replicate(1, 1);
    sigma2_ = 0.0;
    beta_ = 1.0;
    lambda_ = 1.0;
    alpha_ = 0.0;
    lle_weight_ = 1.0;
    k_vis_ = 0.0;
    mu_ = 0.05;
    max_iter_ = 50;
    tol_ = 0.00001;
    include_lle_ = true;
    use_geodesic_ = true;
    use_prev_sigma2_ = true;
    kernel_ = 1;
    geodesic_coord_ = {};
    correspondence_priors_ = {};
}

trackdlo::trackdlo(int num_of_nodes,
                    double beta,
                    double lambda,
                    double alpha,
                    double lle_weight,
                    double k_vis,
                    double mu,
                    int max_iter,
                    double tol,
                    bool include_lle,
                    bool use_geodesic,
                    bool use_prev_sigma2,
                    int kernel) 
{
    Y_ = MatrixXf::Zero(num_of_nodes, 3);
    guide_nodes_ = Y_.replicate(1, 1);
    sigma2_ = 0.0;
    beta_ = beta;
    lambda_ = lambda;
    alpha_ = alpha;
    lle_weight_ = lle_weight;
    k_vis_ = k_vis;
    mu_ = mu;
    max_iter_ = max_iter;
    tol_ = tol;
    include_lle_ = include_lle;
    use_geodesic_ = use_geodesic;
    use_prev_sigma2_ = use_prev_sigma2;
    kernel_ = kernel;
    geodesic_coord_ = {};
    correspondence_priors_ = {};
}

double trackdlo::get_sigma2 () {
    return sigma2_;
}

MatrixXf trackdlo::get_tracking_result () {
    return Y_;
}

MatrixXf trackdlo::get_guide_nodes () {
    return guide_nodes_;
}

std::vector<MatrixXf> trackdlo::get_correspondence_pairs () {
    return correspondence_priors_;
}

void trackdlo::initialize_geodesic_coord (std::vector<double> geodesic_coord) {
    for (int i = 0; i < geodesic_coord.size(); i ++) {
        geodesic_coord_.push_back(geodesic_coord[i]);
    }
}

void trackdlo::initialize_nodes (MatrixXf Y_init) {
    Y_ = Y_init.replicate(1, 1);
    guide_nodes_ = Y_init.replicate(1, 1);
}

void trackdlo::set_sigma2 (double sigma2) {
    sigma2_ = sigma2;
}

std::vector<int> trackdlo::get_nearest_indices (int k, int M, int idx) {
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

MatrixXf trackdlo::calc_LLE_weights (int k, MatrixXf X) {
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

bool trackdlo::ecpd_lle (MatrixXf X_orig,
                        MatrixXf& Y,
                        double& sigma2,
                        double beta,
                        double lambda,
                        double lle_weight,
                        double mu,
                        int max_iter,
                        double tol,
                        bool include_lle,
                        bool use_geodesic,
                        bool use_prev_sigma2,
                        bool use_ecpd,
                        std::vector<MatrixXf> correspondence_priors,
                        double alpha,
                        int kernel,
                        std::vector<int> occluded_nodes,
                        double k_vis,
                        Mat bmask_transformed_normalized,
                        double mat_max) 
{

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
        if (kernel == 3) {
            G = (-diff_yy / (2 * beta * beta)).array().exp();
        }
        else if (kernel == 0) {
            G = (-diff_yy_sqrt / (2 * beta * beta)).array().exp();
        }
        else if (kernel == 1) {
            G = 1/(2*beta * 2*beta) * (-sqrt(2)*diff_yy_sqrt/beta).array().exp() * (sqrt(2)*diff_yy_sqrt.array() + beta);
        }
        else if (kernel == 2) {
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

        if (kernel == 3) {
            G = (-converted_node_dis_sq / (2 * beta * beta)).array().exp();
        }
        else if (kernel == 0) {
            G = (-converted_node_dis / (2 * beta * beta)).array().exp();
        }
        else if (kernel == 1) {
            G = 1/(2*beta * 2*beta) * (-sqrt(2)*converted_node_dis/beta).array().exp() * (sqrt(2)*converted_node_dis.array() + beta);
        }
        else if (kernel == 2) {
            G = 27 * 1/(72 * pow(beta, 3)) * (-sqrt(3)*converted_node_dis/beta).array().exp() * (sqrt(3)*beta*beta + 3*beta*converted_node_dis.array() + sqrt(3)*converted_node_dis_sq.array());
        }
        else { // default to gaussian
            G = (-converted_node_dis_sq / (2 * beta * beta)).array().exp();
        }
    }

    // get the LLE matrix
    MatrixXf L = calc_LLE_weights(6, Y_0);
    MatrixXf H = (MatrixXf::Identity(M, M) - L).transpose() * (MatrixXf::Identity(M, M) - L);

    // construct R and J
    MatrixXf priors = MatrixXf::Zero(correspondence_priors.size(), 3);
    MatrixXf J = MatrixXf::Zero(M, M);
    MatrixXf Y_extended = Y_0.replicate(1, 1);
    MatrixXf G_masked = MatrixXf::Zero(M, M);
    if (correspondence_priors.size() != 0) {
        int num_of_correspondence_priors = correspondence_priors.size();

        for (int i = 0; i < num_of_correspondence_priors; i ++) {
            MatrixXf temp = MatrixXf::Zero(1, 3);
            int index = correspondence_priors[i](0, 0);
            temp(0, 0) = correspondence_priors[i](0, 1);
            temp(0, 1) = correspondence_priors[i](0, 2);
            temp(0, 2) = correspondence_priors[i](0, 3);

            priors.row(i) = temp;
            J.row(index) = MatrixXf::Identity(M, M).row(index);
            Y_extended.row(index) = temp;
            G_masked.row(index) = G.row(index);

            // // enforce boundaries
            // if (i == 0 || i == num_of_correspondence_priors-1) {
            //     J.row(index) *= 5;
            // }
        }
    }

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
            MatrixXf nodes_h = Y.replicate(1, 1);
            // if has corresponding guide node, use that instead of the original position
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

        MatrixXf Pt1 = P.colwise().sum();  // this should have shape (N,) or (1, N)
        MatrixXf P1 = P.rowwise().sum();
        double Np = P1.sum();
        MatrixXf PX = P * X;

        // M step
        MatrixXf A_matrix;
        MatrixXf B_matrix;
        if (include_lle) {
            if (use_ecpd) {
                A_matrix = P1.asDiagonal()*G + lambda*sigma2 * MatrixXf::Identity(M, M) + sigma2*lle_weight * H*G + alpha*J*G;
                B_matrix = PX - P1.asDiagonal()*Y_0 - sigma2*lle_weight * H*Y_0 + alpha*(Y_extended - Y_0);
            }
            else {
                A_matrix = P1.asDiagonal()*G + lambda*sigma2 * MatrixXf::Identity(M, M) + sigma2*lle_weight * H*G;
                B_matrix = PX - P1.asDiagonal()*Y_0 - sigma2*lle_weight * H*Y_0;
            }
        }
        else {
            if (use_ecpd) {
                A_matrix = P1.asDiagonal() * G + lambda * sigma2 * MatrixXf::Identity(M, M) + alpha*J*G;
                B_matrix = PX - P1.asDiagonal() * Y_0 + alpha*(Y_extended - Y_0);
            }
            else {
                A_matrix = P1.asDiagonal() * G + lambda * sigma2 * MatrixXf::Identity(M, M);
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

// alignment: 0 --> align with head; 1 --> align with tail
std::vector<MatrixXf> trackdlo::traverse_geodesic (std::vector<double> geodesic_coord, const MatrixXf guide_nodes, const std::vector<int> visible_nodes, int alignment) {
    std::vector<MatrixXf> node_pairs = {};

    // extreme cases: only one guide node available
    // since this function will only be called when at least one of head or tail is visible, 
    // the only node will be head or tail
    if (guide_nodes.rows() == 1) {
        MatrixXf node_pair(1, 4);
        node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
        node_pairs.push_back(node_pair);
        return node_pairs;
    }

    double guide_nodes_total_dist = 0;
    double total_seg_dist = 0;
    
    if (alignment == 0) {
        // push back the first pair
        MatrixXf node_pair(1, 4);
        node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
        node_pairs.push_back(node_pair);

        // initialize iterators
        int guide_nodes_it = 0;
        int seg_dist_it = 0;
        int last_seg_dist_it = seg_dist_it;

        // ultimate terminating condition: run out of guide nodes to use. two conditions that can trigger this:
        //   1. next visible node index - current visible node index > 1
        //   2. currenting using the last two guide nodes
        while (visible_nodes[guide_nodes_it+1] - visible_nodes[guide_nodes_it] == 1 && guide_nodes_it+1 <= guide_nodes.rows()-1 && seg_dist_it+1 <= geodesic_coord.size()-1) {
            guide_nodes_total_dist += pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it+1));
            // now keep adding segment dists until the total seg dists exceed the current total guide node dists
            while (guide_nodes_total_dist > total_seg_dist) {
                // break condition
                if (seg_dist_it == geodesic_coord.size()-1) {
                    break;
                }

                total_seg_dist += fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it+1]);
                if (total_seg_dist <= guide_nodes_total_dist) {
                    seg_dist_it += 1;
                }
                else {
                    total_seg_dist -= fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it+1]);
                    break;
                }
            }
            // additional break condition
            if (seg_dist_it == geodesic_coord.size()-1) {
                break;
            }
            // upon exit, seg_dist_it will be at the locaiton where the total seg dist is barely smaller than guide nodes total dist
            // the node desired should be in between guide_nodes[guide_nodes_it] and guide_node[guide_nodes_it + 1]
            // seg_dist_it will also be within guide_nodes_it and guide_nodes_it + 1
            if (guide_nodes_it == 0 && seg_dist_it == 0) {
                continue;
            }
            // if one guide nodes segment is not long enough
            if (last_seg_dist_it == seg_dist_it) {
                guide_nodes_it += 1;
                continue;
            }
            double remaining_dist = total_seg_dist - (guide_nodes_total_dist - pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it+1)));
            MatrixXf temp = (guide_nodes.row(guide_nodes_it + 1) - guide_nodes.row(guide_nodes_it)) * remaining_dist / pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it+1));
            node_pair(0, 0) = seg_dist_it;
            node_pair(0, 1) = temp(0, 0) + guide_nodes(guide_nodes_it, 0);
            node_pair(0, 2) = temp(0, 1) + guide_nodes(guide_nodes_it, 1);
            node_pair(0, 3) = temp(0, 2) + guide_nodes(guide_nodes_it, 2);
            node_pairs.push_back(node_pair);

            // update guide_nodes_it at the very end
            guide_nodes_it += 1;
            last_seg_dist_it = seg_dist_it;
        }
    }
    else {
        // push back the first pair
        MatrixXf node_pair(1, 4);
        node_pair << visible_nodes.back(), guide_nodes(guide_nodes.rows()-1, 0), guide_nodes(guide_nodes.rows()-1, 1), guide_nodes(guide_nodes.rows()-1, 2);
        node_pairs.push_back(node_pair);

        // initialize iterators
        int guide_nodes_it = guide_nodes.rows()-1;
        int seg_dist_it = geodesic_coord.size()-1;
        int last_seg_dist_it = seg_dist_it;

        // ultimate terminating condition: run out of guide nodes to use. two conditions that can trigger this:
        //   1. next visible node index - current visible node index > 1
        //   2. currenting using the last two guide nodes
        while (visible_nodes[guide_nodes_it] - visible_nodes[guide_nodes_it-1] == 1 && guide_nodes_it-1 >= 0 && seg_dist_it-1 >= 0) {
            guide_nodes_total_dist += pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it-1));
            // now keep adding segment dists until the total seg dists exceed the current total guide node dists
            while (guide_nodes_total_dist > total_seg_dist) {
                // break condition
                if (seg_dist_it == 0) {
                    break;
                }

                total_seg_dist += fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1]);
                if (total_seg_dist <= guide_nodes_total_dist) {
                    seg_dist_it -= 1;
                }
                else {
                    total_seg_dist -= fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1]);
                    break;
                }
            }
            // additional break condition
            if (seg_dist_it == 0) {
                break;
            }
            // upon exit, seg_dist_it will be at the locaiton where the total seg dist is barely smaller than guide nodes total dist
            // the node desired should be in between guide_nodes[guide_nodes_it] and guide_node[guide_nodes_it + 1]
            // seg_dist_it will also be within guide_nodes_it and guide_nodes_it + 1
            if (guide_nodes_it == 0 && seg_dist_it == 0) {
                continue;
            }
            // if one guide nodes segment is not long enough
            if (last_seg_dist_it == seg_dist_it) {
                guide_nodes_it -= 1;
                continue;
            }
            double remaining_dist = total_seg_dist - (guide_nodes_total_dist - pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it-1)));
            MatrixXf temp = (guide_nodes.row(guide_nodes_it - 1) - guide_nodes.row(guide_nodes_it)) * remaining_dist / pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it-1));
            node_pair(0, 0) = seg_dist_it;
            node_pair(0, 1) = temp(0, 0) + guide_nodes(guide_nodes_it, 0);
            node_pair(0, 2) = temp(0, 1) + guide_nodes(guide_nodes_it, 1);
            node_pair(0, 3) = temp(0, 2) + guide_nodes(guide_nodes_it, 2);
            node_pairs.insert(node_pairs.begin(), node_pair);

            // update guide_nodes_it at the very end
            guide_nodes_it -= 1;
            last_seg_dist_it = seg_dist_it;
        }
    }

    return node_pairs;
}

// overload
std::vector<MatrixXf> trackdlo::traverse_euclidean (std::vector<double> geodesic_coord, const MatrixXf guide_nodes, const std::vector<int> visible_nodes, int alignment, int alignment_node_idx) {
    std::vector<MatrixXf> node_pairs = {};

    // extreme cases: only one guide node available
    // since this function will only be called when at least one of head or tail is visible, 
    // the only node will be head or tail
    if (guide_nodes.rows() == 1) {
        MatrixXf node_pair(1, 4);
        node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
        node_pairs.push_back(node_pair);
        return node_pairs;
    }

    if (alignment == 0) {
        // push back the first pair
        MatrixXf node_pair(1, 4);
        node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
        node_pairs.push_back(node_pair);

        std::vector<int> consecutive_visible_nodes = {};
        for (int i = 0; i < visible_nodes.size(); i ++) {
            if (i == visible_nodes[i]) {
                consecutive_visible_nodes.push_back(i);
            }
            else {
                break;
            }
        }

        int last_found_index = 0;
        int seg_dist_it = 0;
        MatrixXf cur_center = guide_nodes.row(0);

        // basically pure pursuit lol
        while (last_found_index+1 <= consecutive_visible_nodes.size()-1 && seg_dist_it+1 <= geodesic_coord.size()-1) {
            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it+1] - geodesic_coord[seg_dist_it]);
            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i+1 <= consecutive_visible_nodes.size()-1; i ++) {
                std::vector<MatrixXf> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i+1), cur_center, look_ahead_dist);

                // if no intersection found
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i+1)) > pt2pt_dis(cur_center, guide_nodes.row(i+1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;

                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i+1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i+1))) {
                            // the first solution is closer
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            // the second one is closer
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) {
                break;
            }
            else {
                MatrixXf temp = MatrixXf::Zero(1, 4);
                temp(0, 0) = seg_dist_it + 1;
                temp(0, 1) = intersection[0];
                temp(0, 2) = intersection[1];
                temp(0, 3) = intersection[2];
                node_pairs.push_back(temp);

                seg_dist_it += 1;
            }
        }
    }
    else if (alignment == 1){
        // push back the first pair
        MatrixXf node_pair(1, 4);
        node_pair << visible_nodes.back(), guide_nodes(guide_nodes.rows()-1, 0), guide_nodes(guide_nodes.rows()-1, 1), guide_nodes(guide_nodes.rows()-1, 2);
        node_pairs.push_back(node_pair);

        std::vector<int> consecutive_visible_nodes = {};
        for (int i = 1; i <= visible_nodes.size(); i ++) {
            if (visible_nodes[visible_nodes.size()-i] == geodesic_coord.size()-i) {
                consecutive_visible_nodes.push_back(geodesic_coord.size()-i);
            }
            else {
                break;
            }
        }

        int last_found_index = guide_nodes.rows()-1;
        int seg_dist_it = geodesic_coord.size()-1;
        MatrixXf cur_center = guide_nodes.row(guide_nodes.rows()-1);

        // basically pure pursuit lol
        while (last_found_index-1 >= (guide_nodes.rows() - consecutive_visible_nodes.size()) && seg_dist_it-1 >= 0) {

            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1]);

            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i >= (guide_nodes.rows() - consecutive_visible_nodes.size() + 1); i --) {
                std::vector<MatrixXf> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i-1), cur_center, look_ahead_dist);

                // if no intersection found
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i-1)) > pt2pt_dis(cur_center, guide_nodes.row(i-1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;

                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i-1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i-1))) {
                            // the first solution is closer
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            // the second one is closer
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) {
                break;
            }
            else {
                MatrixXf temp = MatrixXf::Zero(1, 4);
                temp(0, 0) = seg_dist_it - 1;
                temp(0, 1) = intersection[0];
                temp(0, 2) = intersection[1];
                temp(0, 3) = intersection[2];
                node_pairs.push_back(temp);

                seg_dist_it -= 1;
            }
        }
    }
    else {
        // push back the first pair
        MatrixXf node_pair(1, 4);
        node_pair << visible_nodes[alignment_node_idx], guide_nodes(alignment_node_idx, 0), guide_nodes(alignment_node_idx, 1), guide_nodes(alignment_node_idx, 2);
        node_pairs.push_back(node_pair);

        std::vector<int> consecutive_visible_nodes_2 = {visible_nodes[alignment_node_idx]};
        for (int i = alignment_node_idx+1; i < visible_nodes.size(); i ++) {
            if (visible_nodes[i] - visible_nodes[i-1] == 1) {
                consecutive_visible_nodes_2.push_back(visible_nodes[i]);
            }
            else {
                break;
            }
        }

        // ----- traverse from the alignment node to the tail node -----
        int last_found_index = alignment_node_idx;
        int seg_dist_it = visible_nodes[alignment_node_idx];
        MatrixXf cur_center = guide_nodes.row(alignment_node_idx);

        // basically pure pursuit lol
        while (last_found_index+1 <= alignment_node_idx+consecutive_visible_nodes_2.size()-1 && seg_dist_it+1 <= geodesic_coord.size()-1) {
            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it+1] - geodesic_coord[seg_dist_it]);
            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i+1 <= alignment_node_idx+consecutive_visible_nodes_2.size()-1; i ++) {
                std::vector<MatrixXf> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i+1), cur_center, look_ahead_dist);

                // if no intersection found
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i+1)) > pt2pt_dis(cur_center, guide_nodes.row(i+1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;

                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i+1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i+1))) {
                            // the first solution is closer
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            // the second one is closer
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) {
                break;
            }
            else {
                MatrixXf temp = MatrixXf::Zero(1, 4);
                temp(0, 0) = seg_dist_it + 1;
                temp(0, 1) = intersection[0];
                temp(0, 2) = intersection[1];
                temp(0, 3) = intersection[2];
                node_pairs.push_back(temp);

                seg_dist_it += 1;
            }
        }


        // ----- traverse from alignment node to head node -----
        std::vector<int> consecutive_visible_nodes_1 = {visible_nodes[alignment_node_idx]};
        for (int i = alignment_node_idx-1; i >= 0; i ++) {
            if (visible_nodes[i+1] - visible_nodes[i] == 1) {
                consecutive_visible_nodes_1.push_back(visible_nodes[i]);
            }
            else {
                break;
            }
        }

        last_found_index = alignment_node_idx;
        seg_dist_it = visible_nodes[alignment_node_idx];
        cur_center = guide_nodes.row(alignment_node_idx);

        // basically pure pursuit lol
        while (last_found_index-1 >= alignment_node_idx-consecutive_visible_nodes_1.size() && seg_dist_it-1 >= 0) {
            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1]);
            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i-1 >= 0; i --) {
                std::vector<MatrixXf> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i-1), cur_center, look_ahead_dist);

                // if no intersection found
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i-1)) > pt2pt_dis(cur_center, guide_nodes.row(i-1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;

                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i-1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i-1))) {
                            // the first solution is closer
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            // the second one is closer
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) {
                break;
            }
            else {
                MatrixXf temp = MatrixXf::Zero(1, 4);
                temp(0, 0) = seg_dist_it - 1;
                temp(0, 1) = intersection[0];
                temp(0, 2) = intersection[1];
                temp(0, 3) = intersection[2];
                node_pairs.push_back(temp);

                seg_dist_it -= 1;
            }
        }
    }

    return node_pairs;
}

void trackdlo::tracking_step (MatrixXf X_orig,
                                Mat bmask_transformed_normalized,
                                double mask_dist_threshold,
                                double mat_max) {
    
    // variable initialization
    std::vector<int> occluded_nodes = {};
    std::vector<int> visible_nodes = {};
    std::vector<MatrixXf> valid_nodes_vec = {};
    correspondence_priors_ = {};
    int state = 0;

    // project Y onto the original image to determine occluded nodes
    MatrixXf nodes_h = Y_.replicate(1, 1);
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
            valid_nodes_vec.push_back(Y_.row(i));
            visible_nodes.push_back(i);
        }
        else {
            occluded_nodes.push_back(i);
        }
    }

    // copy valid guide nodes vec to guide nodes
    // not using topRows() because it caused weird bugs
    guide_nodes_ = MatrixXf::Zero(valid_nodes_vec.size(), 3);
    if (occluded_nodes.size() != 0) {
        for (int i = 0; i < valid_nodes_vec.size(); i ++) {
            guide_nodes_.row(i) = valid_nodes_vec[i];
        }
    }
    else {
        guide_nodes_ = Y_.replicate(1, 1);
    }

    // determine DLO state: heading visible, tail visible, both visible, or both occluded
    // priors_vec should be the final output; priors_vec[i] = {index, x, y, z}
    double sigma2_pre_proc = sigma2_;
    ecpd_lle(X_orig, guide_nodes_, sigma2_pre_proc, 4, 1, 1, 0.1, 50, 0.00001, true, true, true, false, {}, 0.0, 1);

    if (occluded_nodes.size() == 0) {
        ROS_INFO("All nodes visible");

        // get priors vec
        std::vector<MatrixXf> priors_vec_1 = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 0);
        std::vector<MatrixXf> priors_vec_2 = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 1);
        // std::vector<MatrixXf> priors_vec_1 = traverse_geodesic(geodesic_coord, guide_nodes, visible_nodes, 0);
        // std::vector<MatrixXf> priors_vec_2 = traverse_geodesic(geodesic_coord, guide_nodes, visible_nodes, 1);

        // take average
        correspondence_priors_ = {};
        for (int i = 0; i < Y_.rows(); i ++) {
            if (i < priors_vec_2[0](0, 0) && i < priors_vec_1.size()) {
                correspondence_priors_.push_back(priors_vec_1[i]);
            }
            else if (i > priors_vec_1[priors_vec_1.size()-1](0, 0) && (i-(Y_.rows()-priors_vec_2.size())) < priors_vec_2.size()) {
                correspondence_priors_.push_back(priors_vec_2[i-(Y_.rows()-priors_vec_2.size())]);
            }
            else {
                correspondence_priors_.push_back((priors_vec_1[i] + priors_vec_2[i-(Y_.rows()-priors_vec_2.size())]) / 2.0);
            }
        }
    }
    else if (visible_nodes[0] == 0 && visible_nodes[visible_nodes.size()-1] == Y_.rows()-1) {
        ROS_INFO("Mid-section occluded");

        correspondence_priors_ = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 0);
        std::vector<MatrixXf> priors_vec_2 = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 1);
        // priors_vec = traverse_geodesic(geodesic_coord, guide_nodes, visible_nodes, 0);
        // std::vector<MatrixXf> priors_vec_2 = traverse_geodesic(geodesic_coord, guide_nodes, visible_nodes, 1);

        correspondence_priors_.insert(correspondence_priors_.end(), priors_vec_2.begin(), priors_vec_2.end());
    }
    else if (visible_nodes[0] == 0) {
        ROS_INFO("Tail occluded");

        correspondence_priors_ = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 0);
        // priors_vec = traverse_geodesic(geodesic_coord, guide_nodes, visible_nodes, 0);
    }
    else if (visible_nodes[visible_nodes.size()-1] == Y_.rows()-1) {
        ROS_INFO("Head occluded");

        correspondence_priors_ = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 1);
        // priors_vec = traverse_geodesic(geodesic_coord, guide_nodes, visible_nodes, 1);
    }
    else {
        ROS_INFO("Both ends occluded");

        // determine which node moved the least
        int alignment_node_idx = -1;
        double moved_dist = 999999;
        for (int i = 0; i < visible_nodes.size(); i ++) {
            if (pt2pt_dis(Y_.row(visible_nodes[i]), guide_nodes_.row(i)) < moved_dist) {
                moved_dist = pt2pt_dis(Y_.row(visible_nodes[i]), guide_nodes_.row(i));
                alignment_node_idx = i;
            }
        }

        // std::cout << "alignment node index: " << alignment_node_idx << std::endl;
        correspondence_priors_ = traverse_euclidean(geodesic_coord_, guide_nodes_, visible_nodes, 2, alignment_node_idx);
    }

    // include_lle == false because we have no space to discuss it in the paper
    ecpd_lle (X_orig, Y_, sigma2_, beta_, lambda_, lle_weight_, mu_, max_iter_, tol_, include_lle_, use_geodesic_, use_prev_sigma2_, true, correspondence_priors_, alpha_, kernel_, occluded_nodes, k_vis_, bmask_transformed_normalized, mat_max);
}