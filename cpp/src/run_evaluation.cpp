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

int callback_count = 0;
evaluator tracking_evaluator;
MatrixXf head_node = MatrixXf::Zero(1, 3);

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
    }
    Y_true = tracking_evaluator.sort_pts(gt_nodes, head_node);

    // update head node
    head_node = Y_true.row(0).replicate(1, 1);

    std::cout << "Y_true size: " << Y_true.rows() << "; Y_track size: " << Y_track.rows() << std::endl;

    // compute error
    double E1 = tracking_evaluator.get_piecewise_error(Y_track, Y_true);
    double E2 = tracking_evaluator.get_piecewise_error(Y_true, Y_track);

    std::cout << "error = " << (E1 + E2)/2 << std::endl;
}   

int main(int argc, char **argv) {
    ros::init(argc, argv, "evaluation");
    ros::NodeHandle nh;

    // load params
    nh.getParam("/evaluation/bag_file", bag_file);
    nh.getParam("/evaluation/alg", alg);
    nh.getParam("/evaluation/bag_dir", bag_dir);

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
    tracking_evaluator = evaluator(0, 0, 0, alg, bag_file);

    image_transport::ImageTransport it(nh);

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