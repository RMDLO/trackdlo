#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/features2d.hpp>

using cv::Mat;

sensor_msgs::ImagePtr imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    std::vector<int> lower_blue = {90, 60, 40};
    std::vector<int> upper_blue = {130, 255, 255};

    std::vector<int> lower_red_1 = {130, 60, 40};
    std::vector<int> upper_red_1 = {255, 255, 255};

    std::vector<int> lower_red_2 = {0, 60, 40};
    std::vector<int> upper_red_2 = {10, 255, 255};

    sensor_msgs::ImagePtr mask_msg = nullptr;

    try {
        Mat cur_image = cv_bridge::toCvShare(msg, "bgr8")->image;
        Mat cur_image_hsv;
        Mat mask_blue, mask_red_1, mask_red_2, mask_red, mask, mask_rgb;

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

        // draw points
        for (cv::KeyPoint key_point : keypoints) {
            cv::circle(mask_rgb, key_point.pt, 5, cv::Scalar(0, 150, 255), -1);
        }

        // cv::imshow("view", mask_rgb);
        // cv::waitKey(30);

        // publish image
        mask_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", mask_rgb).toImageMsg();
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }

    return mask_msg;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;
    cv::namedWindow("view");

    image_transport::ImageTransport it(nh);
    image_transport::Publisher mask_pub = it.advertise("/mask", 1);
    image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 1, [=](const sensor_msgs::ImageConstPtr& msg){
        sensor_msgs::ImagePtr test_image = imageCallback(msg);
        mask_pub.publish(test_image);
    });
    
    ros::spin();
    cv::destroyWindow("view");
}