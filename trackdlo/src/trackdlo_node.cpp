#include "../include/trackdlo.h"
#include "../include/utils.h"

using cv::Mat;

ros::Publisher pc_pub;
ros::Publisher results_pub;
ros::Publisher guide_nodes_pub;
ros::Publisher corr_priors_pub;
ros::Publisher result_pc_pub;
ros::Subscriber init_nodes_sub;
ros::Subscriber camera_info_sub;

using Eigen::MatrixXd;
using Eigen::RowVectorXd;

MatrixXd Y;
double sigma2;
bool initialized = false;
bool received_init_nodes = false;
bool received_proj_matrix = false;
MatrixXd init_nodes;
std::vector<double> converted_node_coord = {0.0};
Mat occlusion_mask;
bool updated_opencv_mask = false;
MatrixXd proj_matrix(3, 4);

double total_len = 0;

bool use_eval_rope;
double beta;
double lambda;
double alpha;
double lle_weight;
double mu;
int max_iter;
double tol;
double k_vis;
bool include_lle;
bool use_geodesic;
bool use_prev_sigma2;
int kernel;
double downsample_leaf_size;

std::string camera_info_topic;
std::string rgb_topic;
std::string depth_topic;
std::string hsv_threshold_upper_limit;
std::string hsv_threshold_lower_limit;
std::vector<int> upper;
std::vector<int> lower;

trackdlo tracker;

void update_opencv_mask (const sensor_msgs::ImageConstPtr& opencv_mask_msg) {
    occlusion_mask = cv_bridge::toCvShare(opencv_mask_msg, "bgr8")->image;
    if (!occlusion_mask.empty()) {
        updated_opencv_mask = true;
    }
}

void update_init_nodes (const sensor_msgs::PointCloud2ConstPtr& pc_msg) {
    pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;
    pcl_conversions::toPCL(*pc_msg, *cloud);
    pcl::PointCloud<pcl::PointXYZRGB> cloud_xyz;
    pcl::fromPCLPointCloud2(*cloud, cloud_xyz);

    init_nodes = cloud_xyz.getMatrixXfMap().topRows(3).transpose().cast<double>();
    received_init_nodes = true;
    init_nodes_sub.shutdown();
}

void update_camera_info (const sensor_msgs::CameraInfoConstPtr& cam_msg) {
    auto P = cam_msg->P;
    for (int i = 0; i < P.size(); i ++) {
        proj_matrix(i/4, i%4) = P[i];
    }
    received_proj_matrix = true;
    camera_info_sub.shutdown();
}

double pre_proc_total = 0;
double algo_total = 0;
double pub_data_total = 0;
int frames = 0;

Mat color_thresholding (Mat cur_image_hsv) {
    std::vector<int> lower_blue = {90, 90, 60};
    std::vector<int> upper_blue = {130, 255, 255};

    std::vector<int> lower_red_1 = {130, 60, 50};
    std::vector<int> upper_red_1 = {255, 255, 255};

    std::vector<int> lower_red_2 = {0, 60, 50};
    std::vector<int> upper_red_2 = {10, 255, 255};

    std::vector<int> lower_yellow = {15, 100, 80};
    std::vector<int> upper_yellow = {40, 255, 255};

    Mat mask_blue, mask_red_1, mask_red_2, mask_red, mask_yellow, mask;
    // filter blue
    cv::inRange(cur_image_hsv, cv::Scalar(lower_blue[0], lower_blue[1], lower_blue[2]), cv::Scalar(upper_blue[0], upper_blue[1], upper_blue[2]), mask_blue);

    // filter red
    cv::inRange(cur_image_hsv, cv::Scalar(lower_red_1[0], lower_red_1[1], lower_red_1[2]), cv::Scalar(upper_red_1[0], upper_red_1[1], upper_red_1[2]), mask_red_1);
    cv::inRange(cur_image_hsv, cv::Scalar(lower_red_2[0], lower_red_2[1], lower_red_2[2]), cv::Scalar(upper_red_2[0], upper_red_2[1], upper_red_2[2]), mask_red_2);

    // filter yellow
    cv::inRange(cur_image_hsv, cv::Scalar(lower_yellow[0], lower_yellow[1], lower_yellow[2]), cv::Scalar(upper_yellow[0], upper_yellow[1], upper_yellow[2]), mask_yellow);

    // combine red mask
    cv::bitwise_or(mask_red_1, mask_red_2, mask_red);
    // combine overall mask
    cv::bitwise_or(mask_red, mask_blue, mask);
    cv::bitwise_or(mask_yellow, mask, mask);

    return mask;
}

sensor_msgs::ImagePtr Callback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::ImageConstPtr& depth_msg) {

    Mat cur_image_orig = cv_bridge::toCvShare(image_msg, "bgr8")->image;
    Mat cur_depth = cv_bridge::toCvShare(depth_msg, depth_msg->encoding)->image;

    // will get overwritten later if intialized
    sensor_msgs::ImagePtr tracking_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cur_image_orig).toImageMsg();
    
    if (!initialized) {
        if (received_init_nodes) {
            tracker = trackdlo(init_nodes.rows(), beta, lambda, alpha, lle_weight, k_vis, mu, max_iter, tol, include_lle, use_geodesic, use_prev_sigma2, kernel);

            sigma2 = 0.001;

            // record geodesic coord
            double cur_sum = 0;
            for (int i = 0; i < init_nodes.rows()-1; i ++) {
                cur_sum += (init_nodes.row(i+1) - init_nodes.row(i)).norm();
                converted_node_coord.push_back(cur_sum);
            }

            tracker.initialize_nodes(init_nodes);
            tracker.initialize_geodesic_coord(converted_node_coord);
            Y = init_nodes.replicate(1, 1);
        }

        if (received_init_nodes && received_proj_matrix) {
            initialized = true;
        }
    }
    else {
        // log time
        std::chrono::high_resolution_clock::time_point cur_time_cb = std::chrono::high_resolution_clock::now();
        double time_diff;
        std::chrono::high_resolution_clock::time_point cur_time;

        Mat mask, mask_rgb, mask_without_occlusion_block;
        Mat cur_image_hsv;

        // convert color
        cv::cvtColor(cur_image_orig, cur_image_hsv, cv::COLOR_BGR2HSV);

        if (!use_eval_rope) {
            // color_thresholding
            cv::inRange(cur_image_hsv, cv::Scalar(lower[0], lower[1], lower[2]), cv::Scalar(upper[0], upper[1], upper[2]), mask_without_occlusion_block);
        }
        else {
            mask_without_occlusion_block = color_thresholding(cur_image_hsv);
        }

        // update cur image for visualization
        Mat cur_image;
        Mat occlusion_mask_gray;
        if (updated_opencv_mask) {
            cv::cvtColor(occlusion_mask, occlusion_mask_gray, cv::COLOR_BGR2GRAY);
            cv::bitwise_and(mask_without_occlusion_block, occlusion_mask_gray, mask);
            cv::bitwise_and(cur_image_orig, occlusion_mask, cur_image);
        }
        else {
            mask_without_occlusion_block.copyTo(mask);
            cur_image_orig.copyTo(cur_image);
        }

        cv::cvtColor(mask, mask_rgb, cv::COLOR_GRAY2BGR);

        // distance transform
        Mat bmask_transformed (mask.rows, mask.cols, CV_32F);
        cv::distanceTransform((255-mask), bmask_transformed, cv::noArray(), cv::DIST_L2, 5);

        double mat_min, mat_max;
        cv::minMaxLoc(bmask_transformed, &mat_min, &mat_max);
        Mat bmask_transformed_normalized = bmask_transformed/mat_max * 255;
        bmask_transformed_normalized.convertTo(bmask_transformed_normalized, CV_8U);
        double mask_dist_threshold = 10;

        sensor_msgs::PointCloud2 output;
        sensor_msgs::PointCloud2 result_pc;

        bool simulated_occlusion = false;
        int occlusion_corner_i = -1;
        int occlusion_corner_j = -1;
        int occlusion_corner_i_2 = -1;
        int occlusion_corner_j_2 = -1;

        // filter point cloud
        pcl::PointCloud<pcl::PointXYZRGB> cur_pc;
        pcl::PointCloud<pcl::PointXYZRGB> cur_pc_downsampled;

        // filter point cloud from mask
        for (int i = 0; i < mask.rows; i ++) {
            for (int j = 0; j < mask.cols; j ++) {
                // for text label (visualization)
                if (updated_opencv_mask && !simulated_occlusion && occlusion_mask_gray.at<uchar>(i, j) == 0) {
                    occlusion_corner_i = i;
                    occlusion_corner_j = j;
                    simulated_occlusion = true;
                }

                // update the other corner of occlusion mask (visualization)
                if (updated_opencv_mask && occlusion_mask_gray.at<uchar>(i, j) == 0) {
                    occlusion_corner_i_2 = i;
                    occlusion_corner_j_2 = j;
                }

                double depth_threshold = 0.4 * 1000;  // millimeters
                if (mask.at<uchar>(i, j) != 0 && cur_depth.at<uint16_t>(i, j) > depth_threshold) {
                    // point cloud from image pixel coordinates and depth value
                    pcl::PointXYZRGB point;
                    double pixel_x = static_cast<double>(j);
                    double pixel_y = static_cast<double>(i);
                    double cx = proj_matrix(0, 2);
                    double cy = proj_matrix(1, 2);
                    double fx = proj_matrix(0, 0);
                    double fy = proj_matrix(1, 1);
                    double pc_z = cur_depth.at<uint16_t>(i, j) / 1000.0;

                    point.x = (pixel_x - cx) * pc_z / fx;
                    point.y = (pixel_y - cy) * pc_z / fy;
                    point.z = pc_z;

                    point.r = cur_image_orig.at<cv::Vec3b>(i, j)[0];
                    point.g = cur_image_orig.at<cv::Vec3b>(i, j)[1];
                    point.b = cur_image_orig.at<cv::Vec3b>(i, j)[2];

                    cur_pc.push_back(point);
                }
            }
        }

        // Perform downsampling
        pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloudPtr(cur_pc.makeShared());
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud (cloudPtr);
        sor.setLeafSize (downsample_leaf_size, downsample_leaf_size, downsample_leaf_size);
        sor.filter(cur_pc_downsampled);

        MatrixXd X = cur_pc_downsampled.getMatrixXfMap().topRows(3).transpose().cast<double>();
        ROS_INFO_STREAM("Number of points in downsampled point cloud: " + std::to_string(X.rows()));

        MatrixXd guide_nodes;
        std::vector<MatrixXd> priors;

        // log time
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - cur_time_cb).count() / 1000.0;
        ROS_INFO_STREAM("Before tracking step: " + std::to_string(time_diff) + " ms");
        pre_proc_total += time_diff;
        cur_time = std::chrono::high_resolution_clock::now();
        
        // ecpd_lle (X, Y, sigma2, 0.5, 1, 1, 0.05, 50, 0.00001, false, true, false, false, {}, 0, "Gaussian");
        tracker.tracking_step(X, bmask_transformed_normalized, mask_dist_threshold, mat_max);
        // tracker.ecpd_lle(X, Y, sigma2, 3, 1, 1, 0.1, 50, 0.00001, true, true, true, false, {}, 0, 1);
    
        Y = tracker.get_tracking_result();
        guide_nodes = tracker.get_guide_nodes();
        priors = tracker.get_correspondence_pairs();

        // log time
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - cur_time).count() / 1000.0;
        ROS_INFO_STREAM("Tracking step: " + std::to_string(time_diff) + " ms");
        algo_total += time_diff;
        cur_time = std::chrono::high_resolution_clock::now();

        // projection and pub image
        MatrixXd nodes_h = Y.replicate(1, 1);
        // MatrixXd nodes_h = guide_nodes.replicate(1, 1);

        nodes_h.conservativeResize(nodes_h.rows(), nodes_h.cols()+1);
        nodes_h.col(nodes_h.cols()-1) = MatrixXd::Ones(nodes_h.rows(), 1);

        // project and pub image
        MatrixXd image_coords = (proj_matrix * nodes_h.transpose()).transpose();

        Mat tracking_img;
        tracking_img = 0.5*cur_image_orig + 0.5*cur_image;

        // cur_image.copyTo(tracking_img);

        // draw points
        for (int i = 0; i < image_coords.rows(); i ++) {

            int row = static_cast<int>(image_coords(i, 0)/image_coords(i, 2));
            int col = static_cast<int>(image_coords(i, 1)/image_coords(i, 2));

            cv::Scalar point_color;
            cv::Scalar line_color;

            if (static_cast<int>(bmask_transformed_normalized.at<uchar>(col, row)) < mask_dist_threshold / mat_max * 255) {
                point_color = cv::Scalar(0, 150, 255);
                line_color = cv::Scalar(0, 255, 0);
            }
            else {
                point_color = cv::Scalar(0, 0, 255);
                line_color = cv::Scalar(0, 0, 255);
            }

            if (i != image_coords.rows()-1) {
                cv::line(tracking_img, cv::Point(row, col),
                                    cv::Point(static_cast<int>(image_coords(i+1, 0)/image_coords(i+1, 2)), 
                                                static_cast<int>(image_coords(i+1, 1)/image_coords(i+1, 2))),
                                                line_color, 2);
            }

            cv::circle(tracking_img, cv::Point(row, col), 5, point_color, -1);
        }

        // add text
        if (updated_opencv_mask && simulated_occlusion) {
            cv::putText(tracking_img, "occlusion", cv::Point(occlusion_corner_j, occlusion_corner_i-10), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 0, 240), 2);
        }

        // publish image
        tracking_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", tracking_img).toImageMsg();

        // publish filtered point cloud
        pcl::PCLPointCloud2 cur_pc_pointcloud2;
        pcl::toPCLPointCloud2(cur_pc_downsampled, cur_pc_pointcloud2);
        cur_pc_pointcloud2.header.frame_id = "camera_color_optical_frame";

        // Convert to ROS data type
        pcl_conversions::moveFromPCL(cur_pc_pointcloud2, output);

        // publish the results as a marker array
        visualization_msgs::MarkerArray results = MatrixXd2MarkerArray(Y, "camera_color_optical_frame", "node_results", {1.0, 150.0/255.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 1.0});
        visualization_msgs::MarkerArray guide_nodes_results = MatrixXd2MarkerArray(guide_nodes, "camera_color_optical_frame", "guide_node_results", {0.0, 0.0, 0.0, 0.5}, {0.0, 0.0, 1.0, 0.5});
        visualization_msgs::MarkerArray corr_priors_results = MatrixXd2MarkerArray(priors, "camera_color_optical_frame", "corr_prior_results", {0.0, 0.0, 0.0, 0.5}, {1.0, 0.0, 0.0, 0.5});

        // convert to pointcloud2 for eval
        pcl::PointCloud<pcl::PointXYZ> trackdlo_pc;
        for (int i = 0; i < Y.rows(); i++) {
            pcl::PointXYZ temp;
            temp.x = Y(i, 0);
            temp.y = Y(i, 1);
            temp.z = Y(i, 2);
            trackdlo_pc.points.push_back(temp);
        }

        pcl::PCLPointCloud2 result_pc_pclpoincloud2;
        
        pcl::toPCLPointCloud2(trackdlo_pc, result_pc_pclpoincloud2);
        pcl_conversions::moveFromPCL(result_pc_pclpoincloud2, result_pc);

        result_pc.header.frame_id = "camera_color_optical_frame";
        result_pc.header.stamp = image_msg->header.stamp;

        results_pub.publish(results);
        guide_nodes_pub.publish(guide_nodes_results);
        corr_priors_pub.publish(corr_priors_results);
        pc_pub.publish(output);
        result_pc_pub.publish(result_pc);

        // reset all guide nodes
        for (int i = 0; i < guide_nodes_results.markers.size(); i ++) {
            guide_nodes_results.markers[i].action = visualization_msgs::Marker::DELETEALL;
        }
        for (int i = 0; i < corr_priors_results.markers.size(); i ++) {
            corr_priors_results.markers[i].action = visualization_msgs::Marker::DELETEALL;
        }

        // log time
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - cur_time).count() / 1000.0;
        ROS_INFO_STREAM("Pub data: " + std::to_string(time_diff) + " ms");
        pub_data_total += time_diff;

        frames += 1;

        ROS_INFO_STREAM("Avg before tracking step: " + std::to_string(pre_proc_total / frames) + " ms");
        ROS_INFO_STREAM("Avg tracking step: " + std::to_string(algo_total / frames) + " ms");
        ROS_INFO_STREAM("Avg pub data: " + std::to_string(pub_data_total / frames) + " ms");
        ROS_INFO_STREAM("Avg total: " + std::to_string((pre_proc_total + algo_total + pub_data_total) / frames) + " ms");

        // pc_pub.publish(output);
    }
        
    return tracking_img_msg;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "tracker_node");
    ros::NodeHandle nh;

    // load parameters
    nh.getParam("/trackdlo/beta", beta); 
    nh.getParam("/trackdlo/lambda", lambda); 
    nh.getParam("/trackdlo/alpha", alpha); 
    nh.getParam("/trackdlo/lle_weight", lle_weight); 
    nh.getParam("/trackdlo/mu", mu); 
    nh.getParam("/trackdlo/max_iter", max_iter); 
    nh.getParam("/trackdlo/tol", tol);
    nh.getParam("/trackdlo/k_vis", k_vis);
    nh.getParam("/trackdlo/include_lle", include_lle); 
    nh.getParam("/trackdlo/use_geodesic", use_geodesic); 
    nh.getParam("/trackdlo/use_prev_sigma2", use_prev_sigma2); 
    nh.getParam("/trackdlo/kernel", kernel); 

    nh.getParam("/trackdlo/use_eval_rope", use_eval_rope);
    nh.getParam("/trackdlo/downsample_leaf_size", downsample_leaf_size);

    nh.getParam("/trackdlo/camera_info_topic", camera_info_topic);
    nh.getParam("/trackdlo/rgb_topic", rgb_topic);
    nh.getParam("/trackdlo/depth_topic", depth_topic);

    nh.getParam("/trackdlo/hsv_threshold_upper_limit", hsv_threshold_upper_limit);
    nh.getParam("/trackdlo/hsv_threshold_lower_limit", hsv_threshold_lower_limit);

    // update color thresholding upper bound
    std::string rgb_val = "";
    for (int i = 0; i < hsv_threshold_upper_limit.length(); i ++) {
        if (hsv_threshold_upper_limit.substr(i, 1) != " ") {
            rgb_val += hsv_threshold_upper_limit.substr(i, 1);
        }
        else {
            upper.push_back(std::stoi(rgb_val));
            rgb_val = "";
        }
        
        if (i == hsv_threshold_upper_limit.length()-1) {
            upper.push_back(std::stoi(rgb_val));
        }
    }

    // update color thresholding lower bound
    rgb_val = "";
    for (int i = 0; i < hsv_threshold_lower_limit.length(); i ++) {
        if (hsv_threshold_lower_limit.substr(i, 1) != " ") {
            rgb_val += hsv_threshold_lower_limit.substr(i, 1);
        }
        else {
            lower.push_back(std::stoi(rgb_val));
            rgb_val = "";
        }
        
        if (i == hsv_threshold_lower_limit.length()-1) {
            upper.push_back(std::stoi(rgb_val));
        }
    }

    int pub_queue_size = 30;

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber opencv_mask_sub = it.subscribe("/mask_with_occlusion", 10, update_opencv_mask);
    init_nodes_sub = nh.subscribe("/trackdlo/init_nodes", 1, update_init_nodes);
    camera_info_sub = nh.subscribe(camera_info_topic, 1, update_camera_info);

    image_transport::Publisher mask_pub = it.advertise("/mask", pub_queue_size);
    image_transport::Publisher tracking_img_pub = it.advertise("/tracking_img", pub_queue_size);
    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/pts", pub_queue_size);
    results_pub = nh.advertise<visualization_msgs::MarkerArray>("/results_marker", pub_queue_size);
    guide_nodes_pub = nh.advertise<visualization_msgs::MarkerArray>("/guide_nodes", pub_queue_size);
    corr_priors_pub = nh.advertise<visualization_msgs::MarkerArray>("/corr_priors", pub_queue_size);

    // trackdlo point cloud topic
    result_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/trackdlo_results_pc", pub_queue_size);

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, rgb_topic, 10);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, depth_topic, 10);
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync(image_sub, depth_sub, 10);

    sync.registerCallback<std::function<void(const sensor_msgs::ImageConstPtr&, 
                                             const sensor_msgs::ImageConstPtr&,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>)>>
    (
        [&](const sensor_msgs::ImageConstPtr& img_msg, 
            const sensor_msgs::ImageConstPtr& depth_msg,
            const boost::shared_ptr<const message_filters::NullType> var1,
            const boost::shared_ptr<const message_filters::NullType> var2,
            const boost::shared_ptr<const message_filters::NullType> var3,
            const boost::shared_ptr<const message_filters::NullType> var4,
            const boost::shared_ptr<const message_filters::NullType> var5,
            const boost::shared_ptr<const message_filters::NullType> var6,
            const boost::shared_ptr<const message_filters::NullType> var7)
        {
            sensor_msgs::ImagePtr tracking_img = Callback(img_msg, depth_msg);
            tracking_img_pub.publish(tracking_img);
        }
    );
    
    ros::spin();
}