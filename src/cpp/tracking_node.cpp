#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/features2d.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

#include <ctime>
#include <chrono>
#include <thread>

#include "../../include/cpd.h"

using cv::Mat;

ros::Publisher pc_pub;

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::RowVectorXf;
using Eigen::RowVectorXd;

MatrixXf Y;
bool initialized = false;

sensor_msgs::ImagePtr Callback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::PointCloud2ConstPtr& pc_msg) {
    
    std::chrono::steady_clock::time_point cur_time = std::chrono::steady_clock::now();
    
    std::vector<int> lower_blue = {90, 60, 40};
    std::vector<int> upper_blue = {130, 255, 255};

    std::vector<int> lower_red_1 = {130, 60, 40};
    std::vector<int> upper_red_1 = {255, 255, 255};

    std::vector<int> lower_red_2 = {0, 60, 40};
    std::vector<int> upper_red_2 = {10, 255, 255};

    sensor_msgs::ImagePtr tracking_img_msg = nullptr;

    Mat mask_blue, mask_red_1, mask_red_2, mask_red, mask, mask_rgb;
    Mat cur_image = cv_bridge::toCvShare(image_msg, "bgr8")->image;
    Mat cur_image_hsv;

    // convert color
    cv::cvtColor(cur_image, cur_image_hsv, cv::COLOR_BGR2HSV);

    // filter blue
    cv::inRange(cur_image_hsv, cv::Scalar(lower_blue[0], lower_blue[1], lower_blue[2]), cv::Scalar(upper_blue[0], upper_blue[1], upper_blue[2]), mask_blue);

    // filter red
    cv::inRange(cur_image_hsv, cv::Scalar(lower_red_1[0], lower_red_1[1], lower_red_1[2]), cv::Scalar(upper_red_1[0], upper_red_1[1], upper_red_1[2]), mask_red_1);
    cv::inRange(cur_image_hsv, cv::Scalar(lower_red_2[0], lower_red_2[1], lower_red_2[2]), cv::Scalar(upper_red_2[0], upper_red_2[1], upper_red_2[2]), mask_red_2);

    // combine red mask
    cv::bitwise_or(mask_red_1, mask_red_2, mask_red);
    // combine overall mask
    cv::bitwise_or(mask_red, mask_blue, mask);

    // simple blob detector
    cv::SimpleBlobDetector::Params blob_params;
    blob_params.filterByColor = false;
    blob_params.filterByArea = true;
    blob_params.filterByCircularity = false;
    blob_params.filterByInertia = true;
    blob_params.filterByConvexity = false;
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(blob_params);
    // detect
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(mask_red, keypoints);

    cv::cvtColor(mask, mask_rgb, cv::COLOR_GRAY2BGR);

    // Mat tracking_img;
    // cur_image.copyTo(tracking_img);
    // for (cv::KeyPoint key_point : keypoints) {
    //     cv::circle(tracking_img, key_point.pt, 5, cv::Scalar(0, 150, 255), -1);
    // }

    // // publish image
    // tracking_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", tracking_img).toImageMsg();

    // pcl test
    sensor_msgs::PointCloud2 output;
    pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;

    // Convert to PCL data type
    pcl_conversions::toPCL(*pc_msg, *cloud);   // cloud is 720*1280 (height*width) now, however is a ros pointcloud2 message. 
                                               // see message definition here: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html

    if (cloud->width != 0 && cloud->height != 0) {
        // convert to xyz point
        pcl::PointCloud<pcl::PointXYZRGB> cloud_xyz;
        pcl::fromPCLPointCloud2(*cloud, cloud_xyz);
        // now create objects for cur_pc
        pcl::PCLPointCloud2* cur_pc = new pcl::PCLPointCloud2;
        pcl::PointCloud<pcl::PointXYZRGB> cur_pc_xyz;
        pcl::PointCloud<pcl::PointXYZRGB> cur_nodes_xyz;

        // MatrixXf Y_0 = MatrixXf::Zero(keypoints.size(), 3);
        // for (int i = 0; i < keypoints.size(); i ++) {
        //     Y_0(i, 0) = 
        // }

        // filter point cloud from mask
        for (int i = 0; i < cloud->height; i ++) {
            for (int j = 0; j < cloud->width; j ++) {
                if (mask.at<uchar>(i, j) != 0) {
                    cur_pc_xyz.push_back(cloud_xyz(j, i));   // note: this is (j, i) not (i, j)
                }
            }
        }
        MatrixXf X = cur_pc_xyz.getMatrixXfMap().topRows(3).transpose();

        for (cv::KeyPoint key_point : keypoints) {
            cur_nodes_xyz.push_back(cloud_xyz(static_cast<int>(key_point.pt.x), static_cast<int>(key_point.pt.y)));
        }

        MatrixXf Y_0 = cur_nodes_xyz.getMatrixXfMap().topRows(3).transpose();
        MatrixXf Y_0_sorted = sort_pts(Y_0);
        // std::cout << Y_0_sorted.rows() << ", " << Y_0_sorted.cols() << std::endl;

        if (!initialized) {
            Y = Y_0_sorted.replicate(1, 1);
            initialized = true;
        } 
        else {
            Y = cpd(X, Y, 1, 1, 1, 0.05, 100);
        }

        std::cout << Y.rows() << ", " << Y.cols() << std::endl;
        std::cout << Y << std::endl;

        MatrixXf nodes_h = Y.replicate(1, 1);
        nodes_h.conservativeResize(nodes_h.rows(), nodes_h.cols()+1);
        nodes_h.col(nodes_h.cols()-1) = MatrixXf::Ones(nodes_h.rows(), 1);
        // std::cout << Y_0 << std::endl;
        // std::cout << Y_0_sorted << std::endl;

        // project and pub image
        MatrixXf proj_matrix(3, 4);
        proj_matrix << 918.359130859375, 0.0, 645.8908081054688, 0.0,
                       0.0, 916.265869140625, 354.02392578125, 0.0,
                       0.0, 0.0, 1.0, 0.0;
        // std::cout << proj_matrix.rows() << ", " << proj_matrix.cols() << std::endl;
        // std::cout << Y_0_sorted.rows() << ", " << Y_0_sorted.cols() << std::endl;
        MatrixXf image_coords = (proj_matrix * nodes_h.transpose()).transpose();
        // draw points
        Mat tracking_img;
        cur_image.copyTo(tracking_img);
        for (int i = 0; i < image_coords.rows(); i ++) { // image_coords.rows()
            cv::circle(tracking_img, cv::Point(static_cast<int>(image_coords(i, 0)/image_coords(i, 2)), 
                                               static_cast<int>(image_coords(i, 1)/image_coords(i, 2))), 
                                     5, cv::Scalar(0, 150, 255), -1);
            if (i != image_coords.rows()-1) {
                cv::line(tracking_img, cv::Point(static_cast<int>(image_coords(i, 0)/image_coords(i, 2)), 
                                                 static_cast<int>(image_coords(i, 1)/image_coords(i, 2))),
                                       cv::Point(static_cast<int>(image_coords(i+1, 0)/image_coords(i+1, 2)), 
                                                 static_cast<int>(image_coords(i+1, 1)/image_coords(i+1, 2))),
                                       cv::Scalar(0, 255, 0), 2);
            }
        }

        // publish image
        tracking_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", tracking_img).toImageMsg();

        // convert back to pointcloud2 message
        pcl::toPCLPointCloud2(cur_nodes_xyz, *cur_pc);

        // fill in header
        cur_pc->header.frame_id = "camera_color_optical_frame";
        cur_pc->header.seq = cloud->header.seq;
        cur_pc->fields = cloud->fields;

        // // Perform downsampling
        // pcl::PCLPointCloud2ConstPtr cloudPtr(cur_pc);
        // pcl::PCLPointCloud2 cur_pc_downsampled;
        // pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
        // sor.setInputCloud (cloudPtr);
        // sor.setLeafSize (0.004, 0.004, 0.004);
        // sor.filter (cur_pc_downsampled);

        // Convert to ROS data type
        pcl_conversions::moveFromPCL(*cur_pc, output);
        // pcl_conversions::moveFromPCL(cur_pc_downsample1d, output);
    }
    else {
        ROS_ERROR("empty pointcloud!");
    }

    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    pc_pub.publish(output);

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - cur_time).count() << "[ms]" << std::endl;
    
    return tracking_img_msg;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;
    cv::namedWindow("view");

    image_transport::ImageTransport it(nh);
    image_transport::Publisher mask_pub = it.advertise("/mask", 1);
    image_transport::Publisher tracking_img_pub = it.advertise("/tracking_img", 1);
    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/pts", 1);

    // image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 1, [&](const sensor_msgs::ImageConstPtr& msg){
    //     sensor_msgs::ImagePtr test_image = imageCallback(msg);
    //     mask_pub.publish(test_image);
    // });

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/camera/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub(nh, "/camera/depth/color/points", 1);
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2> sync(image_sub, pc_sub, 1);

    sync.registerCallback<std::function<void(const sensor_msgs::ImageConstPtr&, 
                                             const sensor_msgs::PointCloud2ConstPtr&,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>)>>
    (
        [&](const sensor_msgs::ImageConstPtr& img_msg, 
            const sensor_msgs::PointCloud2ConstPtr& pc_msg,
            const boost::shared_ptr<const message_filters::NullType> var1,
            const boost::shared_ptr<const message_filters::NullType> var2,
            const boost::shared_ptr<const message_filters::NullType> var3,
            const boost::shared_ptr<const message_filters::NullType> var4,
            const boost::shared_ptr<const message_filters::NullType> var5,
            const boost::shared_ptr<const message_filters::NullType> var6,
            const boost::shared_ptr<const message_filters::NullType> var7)
        {
            // sensor_msgs::ImagePtr test_image = imageCallback(msg, _);
            // mask_pub.publish(test_image);
            sensor_msgs::ImagePtr tracking_img = Callback(img_msg, pc_msg);
            tracking_img_pub.publish(tracking_img);
        }
    );
    
    ros::spin();
    cv::destroyWindow("view");
}