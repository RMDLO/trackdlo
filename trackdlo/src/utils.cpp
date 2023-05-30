#include "../include/trackdlo.h"
#include "../include/utils.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using cv::Mat;

void signal_callback_handler(int signum) {
   // Terminate program
   exit(signum);
}

double pt2pt_dis_sq (MatrixXd pt1, MatrixXd pt2) {
    return (pt1 - pt2).rowwise().squaredNorm().sum();
}

double pt2pt_dis (MatrixXd pt1, MatrixXd pt2) {
    return (pt1 - pt2).rowwise().norm().sum();
}

void reg (MatrixXd pts, MatrixXd& Y, double& sigma2, int M, double mu, int max_iter) {
    // initial guess
    MatrixXd X = pts.replicate(1, 1);
    Y = MatrixXd::Zero(M, 3);
    for (int i = 0; i < M; i ++) {
        Y(i, 1) = 0.1 / static_cast<double>(M) * static_cast<double>(i);
        Y(i, 0) = 0;
        Y(i, 2) = 0;
    }
    
    int N = X.rows();
    int D = 3;

    // diff_xy should be a (M * N) matrix
    MatrixXd diff_xy = MatrixXd::Zero(M, N);
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

        MatrixXd P = (-0.5 * diff_xy / sigma2).array().exp();
        MatrixXd P_stored = P.replicate(1, 1);
        double c = pow((2 * M_PI * sigma2), static_cast<double>(D)/2) * mu / (1 - mu) * static_cast<double>(M)/N;
        P = P.array().rowwise() / (P.colwise().sum().array() + c);

        MatrixXd Pt1 = P.colwise().sum(); 
        MatrixXd P1 = P.rowwise().sum();
        double Np = P1.sum();
        MatrixXd PX = P * X;

        MatrixXd P1_expanded = MatrixXd::Zero(M, D);
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
void remove_row(MatrixXd& matrix, unsigned int rowToRemove) {
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.bottomRows(numRows-rowToRemove);

    matrix.conservativeResize(numRows,numCols);
}

MatrixXd sort_pts (MatrixXd Y_0) {
    int N = Y_0.rows();
    MatrixXd Y_0_sorted = MatrixXd::Zero(N, 3);
    std::vector<MatrixXd> Y_0_sorted_vec = {};
    std::vector<bool> selected_node(N, false);
    selected_node[0] = true;
    int last_visited_b = 0;

    MatrixXd G = MatrixXd::Zero(N, N);
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

bool isBetween (MatrixXd x, MatrixXd a, MatrixXd b) {
    bool in_bound = true;

    for (int i = 0; i < 3; i ++) {
        if (!(a(0, i)-0.0001 <= x(0, i) && x(0, i) <= b(0, i)+0.0001) && 
            !(b(0, i)-0.0001 <= x(0, i) && x(0, i) <= a(0, i)+0.0001)) {
            in_bound = false;
        }
    }
    
    return in_bound;
}

std::vector<MatrixXd> line_sphere_intersection (MatrixXd point_A, MatrixXd point_B, MatrixXd sphere_center, double radius) {
    std::vector<MatrixXd> intersections = {};
    
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
        MatrixXd pt1(1, 3);
        pt1 << x1, y1, z1;

        // the second one
        double x2 = point_A(0, 0) + d2*(point_B(0, 0) - point_A(0, 0));
        double y2 = point_A(0, 1) + d2*(point_B(0, 1) - point_A(0, 1));
        double z2 = point_A(0, 2) + d2*(point_B(0, 2) - point_A(0, 2));
        MatrixXd pt2(1, 3);
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
        MatrixXd pt1(1, 3);
        pt1 << x1, y1, z1;

        if (isBetween(pt1, point_A, point_B)) {
            intersections.push_back(pt1);
        }
    }
    
    return intersections;
}

// node color and object color are in rgba format and range from 0-1
visualization_msgs::MarkerArray MatrixXd2MarkerArray (MatrixXd Y, std::string marker_frame, std::string marker_ns, std::vector<float> node_color, std::vector<float> line_color) {
    // publish the results as a marker array
    visualization_msgs::MarkerArray results = visualization_msgs::MarkerArray();
    for (int i = 0; i < Y.rows(); i ++) {
        visualization_msgs::Marker cur_node_result = visualization_msgs::Marker();
    
        // add header
        cur_node_result.header.frame_id = marker_frame;
        // cur_node_result.header.stamp = ros::Time::now();
        cur_node_result.type = visualization_msgs::Marker::SPHERE;
        cur_node_result.action = visualization_msgs::Marker::ADD;
        cur_node_result.ns = marker_ns + std::to_string(i);
        cur_node_result.id = i;

        // add position
        cur_node_result.pose.position.x = Y(i, 0);
        cur_node_result.pose.position.y = Y(i, 1);
        cur_node_result.pose.position.z = Y(i, 2);

        // add orientation
        cur_node_result.pose.orientation.w = 1.0;
        cur_node_result.pose.orientation.x = 0.0;
        cur_node_result.pose.orientation.y = 0.0;
        cur_node_result.pose.orientation.z = 0.0;

        // set scale
        cur_node_result.scale.x = 0.01;
        cur_node_result.scale.y = 0.01;
        cur_node_result.scale.z = 0.01;

        // set color
        cur_node_result.color.r = node_color[0];
        cur_node_result.color.g = node_color[1];
        cur_node_result.color.b = node_color[2];
        cur_node_result.color.a = node_color[3];

        results.markers.push_back(cur_node_result);

        // don't add line if at the last node
        if (i == Y.rows()-1) {
            break;
        }

        visualization_msgs::Marker cur_line_result = visualization_msgs::Marker();

        // add header
        cur_line_result.header.frame_id = "camera_color_optical_frame";
        cur_line_result.type = visualization_msgs::Marker::CYLINDER;
        cur_line_result.action = visualization_msgs::Marker::ADD;
        cur_line_result.ns = "line_results" + std::to_string(i);
        cur_line_result.id = i;

        // add position
        cur_line_result.pose.position.x = (Y(i, 0) + Y(i+1, 0)) / 2.0;
        cur_line_result.pose.position.y = (Y(i, 1) + Y(i+1, 1)) / 2.0;
        cur_line_result.pose.position.z = (Y(i, 2) + Y(i+1, 2)) / 2.0;

        // add orientation
        Eigen::Quaternionf q;
        Eigen::Vector3f vec1(0.0, 0.0, 1.0);
        Eigen::Vector3f vec2(Y(i+1, 0) - Y(i, 0), Y(i+1, 1) - Y(i, 1), Y(i+1, 2) - Y(i, 2));
        q.setFromTwoVectors(vec1, vec2);

        cur_line_result.pose.orientation.w = q.w();
        cur_line_result.pose.orientation.x = q.x();
        cur_line_result.pose.orientation.y = q.y();
        cur_line_result.pose.orientation.z = q.z();

        // set scale
        cur_line_result.scale.x = 0.005;
        cur_line_result.scale.y = 0.005;
        cur_line_result.scale.z = pt2pt_dis(Y.row(i), Y.row(i+1));

        // set color
        cur_line_result.color.r = line_color[0];
        cur_line_result.color.g = line_color[1];
        cur_line_result.color.b = line_color[2];
        cur_line_result.color.a = line_color[3];

        results.markers.push_back(cur_line_result);
    }

    return results;
}

// overload function
visualization_msgs::MarkerArray MatrixXd2MarkerArray (std::vector<MatrixXd> Y, std::string marker_frame, std::string marker_ns, std::vector<float> node_color, std::vector<float> line_color) {
    // publish the results as a marker array
    visualization_msgs::MarkerArray results = visualization_msgs::MarkerArray();
    for (int i = 0; i < Y.size(); i ++) {
        visualization_msgs::Marker cur_node_result = visualization_msgs::Marker();

        int dim = Y[0].cols();
    
        // add header
        cur_node_result.header.frame_id = marker_frame;
        // cur_node_result.header.stamp = ros::Time::now();
        cur_node_result.type = visualization_msgs::Marker::SPHERE;
        cur_node_result.action = visualization_msgs::Marker::ADD;
        cur_node_result.ns = marker_ns + std::to_string(i);
        cur_node_result.id = i;

        // add position
        cur_node_result.pose.position.x = Y[i](0, dim-3);
        cur_node_result.pose.position.y = Y[i](0, dim-2);
        cur_node_result.pose.position.z = Y[i](0, dim-1);

        // add orientation
        cur_node_result.pose.orientation.w = 1.0;
        cur_node_result.pose.orientation.x = 0.0;
        cur_node_result.pose.orientation.y = 0.0;
        cur_node_result.pose.orientation.z = 0.0;

        // set scale
        cur_node_result.scale.x = 0.01;
        cur_node_result.scale.y = 0.01;
        cur_node_result.scale.z = 0.01;

        // set color
        cur_node_result.color.r = node_color[0];
        cur_node_result.color.g = node_color[1];
        cur_node_result.color.b = node_color[2];
        cur_node_result.color.a = node_color[3];

        results.markers.push_back(cur_node_result);

        // don't add line if at the last node
        if (i == Y.size()-1) {
            break;
        }

        visualization_msgs::Marker cur_line_result = visualization_msgs::Marker();

        // add header
        cur_line_result.header.frame_id = "camera_color_optical_frame";
        cur_line_result.type = visualization_msgs::Marker::CYLINDER;
        cur_line_result.action = visualization_msgs::Marker::ADD;
        cur_line_result.ns = "line_results" + std::to_string(i);
        cur_line_result.id = i;

        // add position
        cur_line_result.pose.position.x = (Y[i](0, dim-3) + Y[i+1](0, dim-3)) / 2.0;
        cur_line_result.pose.position.y = (Y[i](0, dim-2) + Y[i+1](0, dim-2)) / 2.0;
        cur_line_result.pose.position.z = (Y[i](0, dim-1) + Y[i+1](0, dim-1)) / 2.0;

        // add orientation
        Eigen::Quaternionf q;
        Eigen::Vector3f vec1(0.0, 0.0, 1.0);
        Eigen::Vector3f vec2(Y[i+1](0, dim-3) - Y[i](0, dim-3), Y[i+1](0, dim-2) - Y[i](0, dim-2), Y[i+1](0, dim-1) - Y[i](0, dim-1));
        q.setFromTwoVectors(vec1, vec2);

        cur_line_result.pose.orientation.w = q.w();
        cur_line_result.pose.orientation.x = q.x();
        cur_line_result.pose.orientation.y = q.y();
        cur_line_result.pose.orientation.z = q.z();

        // set scale
        cur_line_result.scale.x = 0.005;
        cur_line_result.scale.y = 0.005;
        cur_line_result.scale.z = sqrt(pow(Y[i+1](0, dim-3) - Y[i](0, dim-3), 2) + pow(Y[i+1](0, dim-2) - Y[i](0, dim-2), 2) + pow(Y[i+1](0, dim-1) - Y[i](0, dim-1), 2));

        // set color
        cur_line_result.color.r = line_color[0];
        cur_line_result.color.g = line_color[1];
        cur_line_result.color.b = line_color[2];
        cur_line_result.color.a = line_color[3];

        results.markers.push_back(cur_line_result);
    }

    return results;
}

MatrixXd cross_product (MatrixXd vec1, MatrixXd vec2) {
    MatrixXd ret = MatrixXd::Zero(1, 3);
    
    ret(0, 0) = vec1(0, 1)*vec2(0, 2) - vec1(0, 2)*vec2(0, 1);
    ret(0, 1) = -(vec1(0, 0)*vec2(0, 2) - vec1(0, 2)*vec2(0, 0));
    ret(0, 2) = vec1(0, 0)*vec2(0, 1) - vec1(0, 1)*vec2(0, 0);

    return ret;
}

double dot_product (MatrixXd vec1, MatrixXd vec2) {
    return vec1(0, 0)*vec2(0, 0) + vec1(0, 1)*vec2(0, 1) + vec1(0, 2)*vec2(0, 2);
}