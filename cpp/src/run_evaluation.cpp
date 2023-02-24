#include "../include/trackdlo.h"
#include "../include/utils.h"
#include "../include/evaluator.h"

#include <ros/package.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::RowVectorXf;
using Eigen::RowVectorXd;
using cv::Mat;

int bag_file;
std::string alg;
std::string bag_dir;
std::string save_location;
double pct_occlusion;

int callback_count = 0;
evaluator tracking_evaluator;
MatrixXf head_node = MatrixXf::Zero(1, 3);

MatrixXf proj_matrix(3, 4);
image_transport::Publisher occlusion_mask_pub;

void Callback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::PointCloud2ConstPtr& pc_msg, const sensor_msgs::PointCloud2ConstPtr& result_msg) {
    callback_count += 1;
    std::cout << "callback: " << callback_count << std::endl;

    Mat cur_image_orig = cv_bridge::toCvShare(image_msg, "bgr8")->image;

    // process original pc
    pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;
    pcl_conversions::toPCL(*pc_msg, *cloud);
    pcl::PointCloud<pcl::PointXYZRGB> cloud_xyz;
    pcl::fromPCLPointCloud2(*cloud, cloud_xyz);

    // process result pc
    pcl::PCLPointCloud2* result_cloud = new pcl::PCLPointCloud2;
    pcl_conversions::toPCL(*result_msg, *result_cloud);
    pcl::PointCloud<pcl::PointXYZ> result_cloud_xyz;
    pcl::fromPCLPointCloud2(*result_cloud, result_cloud_xyz);
    MatrixXf Y_track = result_cloud_xyz.getMatrixXfMap().topRows(3).transpose();

    MatrixXf gt_nodes = tracking_evaluator.get_ground_truth_nodes(cur_image_orig, cloud_xyz);
    MatrixXf Y_true = gt_nodes.replicate(1, 1);
    // if not initialized
    if (head_node(0, 0) == 0.0 && head_node(0, 1) == 0.0 && head_node(0, 2) == 0.0) {
        head_node = Y_track.row(0).replicate(1, 1);
        tracking_evaluator.set_start_time (std::chrono::steady_clock::now());
    }
    Y_true = tracking_evaluator.sort_pts(gt_nodes, head_node);

    // update head node
    head_node = Y_true.row(0).replicate(1, 1);

    std::cout << "Y_true size: " << Y_true.rows() << "; Y_track size: " << Y_track.rows() << std::endl;

    // simulate occlusion: occlude the first n nodes
    // strategy: first calculate the 3D boundary box based on point cloud, then project the four corners back to the image
    double time_from_start;
    time_from_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tracking_evaluator.start_time()).count();
    time_from_start = time_from_start * 0.9 / 1000;  // bag files played at 0.9x speed

    int num_of_occluded_nodes = static_cast<int>(Y_track.rows() * tracking_evaluator.pct_occlusion());

    if (num_of_occluded_nodes != 0 && time_from_start > 1.0) {

        double min_x = Y_true(0, 0);
        double min_y = Y_true(0, 1);
        double min_z = Y_true(0, 2);

        double max_x = Y_true(0, 0);
        double max_y = Y_true(0, 1);
        double max_z = Y_true(0, 2);

        for (int i = 0; i < num_of_occluded_nodes; i ++) {
            if (Y_true(i, 0) < min_x) {
                min_x = Y_true(i, 0);
            }
            if (Y_true(i, 1) < min_y) {
                min_y = Y_true(i, 1);
            }
            if (Y_true(i, 2) < min_z) {
                min_z = Y_true(i, 2);
            }

            if (Y_true(i, 0) > max_x) {
                max_x = Y_true(i, 0);
            }
            if (Y_true(i, 1) > max_y) {
                max_y = Y_true(i, 1);
            }
            if (Y_true(i, 2) > max_z) {
                max_z = Y_true(i, 2);
            }
        }

        MatrixXf min_corner(1, 3);
        min_corner << min_x, min_y, min_z;
        MatrixXf max_corner(1, 3);
        max_corner << max_x, max_y, max_z;

        MatrixXf corners = MatrixXf::Zero(2, 3);
        corners.row(0) = min_corner.replicate(1, 1);
        corners.row(1) = max_corner.replicate(1, 1);

        // project to find occlusion block coorindate
        MatrixXf nodes_h = corners.replicate(1, 1);
        nodes_h.conservativeResize(nodes_h.rows(), nodes_h.cols()+1);
        nodes_h.col(nodes_h.cols()-1) = MatrixXf::Ones(nodes_h.rows(), 1);
        MatrixXf image_coords = (proj_matrix * nodes_h.transpose()).transpose();

        int pix_coord_1_x = static_cast<int>(image_coords(0, 0)/image_coords(0, 2));
        int pix_coord_1_y = static_cast<int>(image_coords(0, 1)/image_coords(0, 2));
        int pix_coord_2_x = static_cast<int>(image_coords(1, 0)/image_coords(1, 2));
        int pix_coord_2_y = static_cast<int>(image_coords(1, 1)/image_coords(1, 2));
    
        // Mat occlusion_mask (cur_image_orig.rows, cur_image_orig.cols, CV_8UC3, cv::Scalar(255, 255, 255));
        Mat occlusion_mask = Mat::ones(cur_image_orig.rows, cur_image_orig.cols, CV_8UC1);
        int extra_border = 0;

        // best scenarios: min_corner and max_corner are the top left and bottom right corners
        if (pix_coord_1_x <= pix_coord_2_x && pix_coord_1_y <= pix_coord_2_y) {
            // cv::Point p1(pix_coord_1_x - extra_border, pix_coord_1_y - extra_border);
            // cv::Point p2(pix_coord_2_x + extra_border, pix_coord_2_y + extra_border);
            // cv::rectangle(occlusion_mask, p1, p2, cv::Scalar(0, 0, 0), -1);
            int top_left_x = pix_coord_1_x - extra_border;
            if (top_left_x < 0) {top_left_x = 0;}
            int top_left_y = pix_coord_1_y - extra_border;
            if (top_left_y < 0) {top_left_y = 0;}
            int bottom_right_x = pix_coord_2_x + extra_border;
            if (bottom_right_x >= cur_image_orig.cols) {bottom_right_x = cur_image_orig.cols-1;}
            int bottom_right_y = pix_coord_2_y + extra_border;
            if (bottom_right_y >= cur_image_orig.rows) {bottom_right_y = cur_image_orig.rows-1;}

            std::cout << "case 1" << std::endl;
            std::cout << top_left_x << ", " << top_left_y << "; " << bottom_right_x << ", " << bottom_right_y << std::endl;

            Mat mask_roi = occlusion_mask(cv::Rect(top_left_x, top_left_y, bottom_right_x-top_left_x, bottom_right_y-top_left_y));
            mask_roi.setTo(0);
        }
        // best scenarios: min_corner and max_corner are the top left and bottom right corners
        else if (pix_coord_2_x <= pix_coord_1_x && pix_coord_2_y <= pix_coord_1_y) {
            // cv::Point p1(pix_coord_2_x - extra_border, pix_coord_2_y - extra_border);
            // cv::Point p2(pix_coord_1_x + extra_border, pix_coord_1_y + extra_border);
            // cv::rectangle(occlusion_mask, p1, p2, cv::Scalar(0, 0, 0), -1);
            int top_left_x = pix_coord_2_x - extra_border;
            if (top_left_x < 0) {top_left_x = 0;}
            int top_left_y = pix_coord_2_y - extra_border;
            if (top_left_y < 0) {top_left_y = 0;}
            int bottom_right_x = pix_coord_1_x + extra_border;
            if (bottom_right_x >= cur_image_orig.cols) {bottom_right_x = cur_image_orig.cols-1;}
            int bottom_right_y = pix_coord_1_y + extra_border;
            if (bottom_right_y >= cur_image_orig.rows) {bottom_right_y = cur_image_orig.rows-1;}
            
            std::cout << "case 2" << std::endl;
            std::cout << top_left_x << ", " << top_left_y << "; " << bottom_right_x << ", " << bottom_right_y << std::endl;

            Mat mask_roi = occlusion_mask(cv::Rect(top_left_x, top_left_y, bottom_right_x, bottom_right_y));
            mask_roi.setTo(cv::Scalar(0, 0, 0));
        }
        // min_corner is top right, max_corner is bottom left
        else if (pix_coord_2_x <= pix_coord_1_x && pix_coord_1_y <= pix_coord_2_y) {
            // cv::Point p1(pix_coord_2_x - extra_border, pix_coord_1_y - extra_border);
            // cv::Point p2(pix_coord_1_x + extra_border, pix_coord_2_y + extra_border);
            // cv::rectangle(occlusion_mask, p1, p2, cv::Scalar(0, 0, 0), -1);
            int top_left_x = pix_coord_2_x - extra_border;
            if (top_left_x < 0) {top_left_x = 0;}
            int top_left_y = pix_coord_1_y - extra_border;
            if (top_left_y < 0) {top_left_y = 0;}
            int bottom_right_x = pix_coord_1_x + extra_border;
            if (bottom_right_x >= cur_image_orig.cols) {bottom_right_x = cur_image_orig.cols-1;}
            int bottom_right_y = pix_coord_2_y + extra_border;
            if (bottom_right_y >= cur_image_orig.rows) {bottom_right_y = cur_image_orig.rows-1;}

            std::cout << "case 3" << std::endl;
            std::cout << top_left_x << ", " << top_left_y << "; " << bottom_right_x << ", " << bottom_right_y << std::endl;

            Mat mask_roi = occlusion_mask(cv::Rect(top_left_x, top_left_y, bottom_right_x, bottom_right_y));
            mask_roi.setTo(cv::Scalar(0, 0, 0));
        }
        // max_corner is top right, min_corner is bottom left
        else {
            // cv::Point p1(pix_coord_1_x - extra_border, pix_coord_2_y - extra_border);
            // cv::Point p2(pix_coord_2_x + extra_border, pix_coord_1_y + extra_border);
            // cv::rectangle(occlusion_mask, p1, p2, cv::Scalar(0, 0, 0), -1);
            int top_left_x = pix_coord_1_x - extra_border;
            if (top_left_x < 0) {top_left_x = 0;}
            int top_left_y = pix_coord_2_y - extra_border;
            if (top_left_y < 0) {top_left_y = 0;}
            int bottom_right_x = pix_coord_2_x + extra_border;
            if (bottom_right_x >= cur_image_orig.cols) {bottom_right_x = cur_image_orig.cols-1;}
            int bottom_right_y = pix_coord_1_y + extra_border;
            if (bottom_right_y >= cur_image_orig.rows) {bottom_right_y = cur_image_orig.rows-1;}

            std::cout << "case 4" << std::endl;
            std::cout << top_left_x << ", " << top_left_y << "; " << bottom_right_x << ", " << bottom_right_y << std::endl;

            Mat mask_roi = occlusion_mask(cv::Rect(top_left_x, top_left_y, bottom_right_x, bottom_right_y));
            mask_roi.setTo(cv::Scalar(0, 0, 0));
        }

        // publish image
        sensor_msgs::ImagePtr occlusion_mask_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", occlusion_mask).toImageMsg();
        occlusion_mask_pub.publish(occlusion_mask_msg);
    }

    // compute error
    double cur_error = tracking_evaluator.compute_and_save_error(Y_track, Y_true);
    std::cout << "error = " << cur_error << std::endl;
}   

int main(int argc, char **argv) {
    ros::init(argc, argv, "evaluation");
    ros::NodeHandle nh;

    proj_matrix << 918.359130859375, 0.0, 645.8908081054688, 0.0,
                   0.0, 916.265869140625, 354.02392578125, 0.0,
                   0.0, 0.0, 1.0, 0.0;

    // load params
    nh.getParam("/evaluation/bag_file", bag_file);
    nh.getParam("/evaluation/alg", alg);
    nh.getParam("/evaluation/bag_dir", bag_dir);
    nh.getParam("/evaluation/save_location", save_location);
    nh.getParam("/evaluation/pct_occlusion", pct_occlusion);

    // get bag file length
    std::vector<std::string> topics;
    topics.push_back("/camera/color/image_raw");
    topics.push_back("/camera/aligned_depth_to_color/image_raw");
    topics.push_back("/camera/depth/color/points");
    topics.push_back("/tf");
    topics.push_back("/tf_static");

    rosbag::Bag bag(bag_dir, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(topics));
    
    int rgb_count = 0;
    int depth_count = 0;
    int pc_count = 0;

    for (rosbag::MessageInstance const& msg: view) {
        if (msg.getTopic() == topics[0]) {
            rgb_count += 1;
        }
        if (msg.getTopic() == topics[1]) {
            depth_count += 1;
        }
        if (msg.getTopic() == topics[2]) {
            pc_count += 1;
        }
    }

    std::cout << "num of rgb images: " << rgb_count << std::endl;
    std::cout << "num of depth images: " << depth_count << std::endl;
    std::cout << "num of point cloud messages: " << pc_count << std::endl;

    // initialize evaluator
    tracking_evaluator = evaluator(0, 0, pct_occlusion, alg, bag_file, save_location);

    image_transport::ImageTransport it(nh);
    occlusion_mask_pub = it.advertise("/mask_with_occlusion", 10);

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/camera/color/image_raw", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub(nh, "/camera/depth/color/points", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> result_sub(nh, "/" + alg + "_results_pc", 10);
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> sync(image_sub, pc_sub, result_sub, 10);

    sync.registerCallback<std::function<void(const sensor_msgs::ImageConstPtr&, 
                                             const sensor_msgs::PointCloud2ConstPtr&,
                                             const sensor_msgs::PointCloud2ConstPtr&,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>)>>
    (
        [&](const sensor_msgs::ImageConstPtr& img_msg, 
            const sensor_msgs::PointCloud2ConstPtr& pc_msg,
            const sensor_msgs::PointCloud2ConstPtr& result_msg,
            const boost::shared_ptr<const message_filters::NullType> var1,
            const boost::shared_ptr<const message_filters::NullType> var2,
            const boost::shared_ptr<const message_filters::NullType> var3,
            const boost::shared_ptr<const message_filters::NullType> var4,
            const boost::shared_ptr<const message_filters::NullType> var5,
            const boost::shared_ptr<const message_filters::NullType> var6)
        {
            Callback(img_msg, pc_msg, result_msg);
        }
    );
    
    ros::spin();
}