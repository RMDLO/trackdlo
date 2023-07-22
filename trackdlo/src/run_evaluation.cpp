#include "../include/trackdlo.h"
#include "../include/utils.h"
#include "../include/evaluator.h"

#include <ros/package.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/Int32MultiArray.h>

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using cv::Mat;

int bag_file;
int trial;
std::string alg;
std::string bag_dir;
std::string save_location;
int pct_occlusion;
double start_record_at;
double exit_at;
double wait_before_occlusion;
double bag_rate;
int num_of_nodes;
bool save_images;
bool save_errors;

int callback_count = 0;
evaluator tracking_evaluator;
MatrixXd head_node = MatrixXd::Zero(1, 3);

MatrixXd proj_matrix(3, 4);
ros::Publisher corners_arr_pub;
bool initialized = false;

// sensor_msgs::ImagePtr Callback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::PointCloud2ConstPtr& pc_msg, const sensor_msgs::PointCloud2ConstPtr& result_msg) {
sensor_msgs::ImagePtr Callback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::PointCloud2ConstPtr& pc_msg, const sensor_msgs::PointCloud2ConstPtr& result_msg) {
    sensor_msgs::ImagePtr eval_img_msg = nullptr;
    
    if (!initialized) {
        tracking_evaluator.set_start_time (std::chrono::steady_clock::now());
        initialized = true;
    }
    
    double time_from_start;
    time_from_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tracking_evaluator.start_time()).count();
    time_from_start = (time_from_start / 1000.0 + 3.0) * tracking_evaluator.rate();  // initialization usually takes 1.5 seconds
    std::cout << time_from_start << "; " << tracking_evaluator.exit_time() << std::endl;
    
    if (tracking_evaluator.exit_time() == -1) {
        if (callback_count >= tracking_evaluator.length() - 3) {
            std::cout << "Shutting down evaluator..." << std::endl;
            ros::shutdown();
        }
    }
    else {
        if (time_from_start > tracking_evaluator.exit_time() || callback_count >= tracking_evaluator.length() - 3) {
            std::cout << "Shutting down evaluator..." << std::endl;
            ros::shutdown();
        }
    }
    
    callback_count += 1;
    std::cout << "callback: " << callback_count << std::endl;

    Mat cur_image_orig = cv_bridge::toCvShare(image_msg, "bgr8")->image;

    // for visualizing results
    Mat eval_img;
    cur_image_orig.copyTo(eval_img);

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
    MatrixXd Y_track = result_cloud_xyz.getMatrixXfMap().topRows(3).transpose().cast<double>();

    int top_left_x = -1;
    int top_left_y = -1;
    int bottom_right_x = -1;
    int bottom_right_y = -1;

    double cur_error = -1;
    if (time_from_start > tracking_evaluator.recording_start_time()) {
        
        if (bag_file != 3) {
            MatrixXd gt_nodes = tracking_evaluator.get_ground_truth_nodes(cur_image_orig, cloud_xyz);
            MatrixXd Y_true = gt_nodes.replicate(1, 1);
            // if not initialized
            if (head_node(0, 0) == 0.0 && head_node(0, 1) == 0.0 && head_node(0, 2) == 0.0) {
                // the one with greater x. this holds true for all 3 bag files
                if (Y_track(0, 0) > Y_track(Y_track.rows()-1, 0)) {
                    head_node = Y_track.row(Y_track.rows()-1).replicate(1, 1);
                }
                else {
                    head_node = Y_track.row(0).replicate(1, 1);
                }
            }
            Y_true = tracking_evaluator.sort_pts(gt_nodes, head_node);

            // update head node
            head_node = Y_true.row(0).replicate(1, 1);
            std::cout << "Y_true size: " << Y_true.rows() << "; Y_track size: " << Y_track.rows() << std::endl;

            if (time_from_start > tracking_evaluator.recording_start_time() + tracking_evaluator.wait_before_occlusion()) {
                if (bag_file == 0) {
                    // simulate occlusion: occlude the first n nodes
                    // strategy: first calculate the 3D boundary box based on point cloud, then project the four corners back to the image
                    int num_of_occluded_nodes = static_cast<int>(Y_track.rows() * tracking_evaluator.pct_occlusion() / 100.0);

                    if (num_of_occluded_nodes != 0) {

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

                        MatrixXd min_corner(1, 3);
                        min_corner << min_x, min_y, min_z;
                        MatrixXd max_corner(1, 3);
                        max_corner << max_x, max_y, max_z;

                        MatrixXd corners = MatrixXd::Zero(2, 3);
                        corners.row(0) = min_corner.replicate(1, 1);
                        corners.row(1) = max_corner.replicate(1, 1);

                        // project to find occlusion block coorindate
                        MatrixXd nodes_h = corners.replicate(1, 1);
                        nodes_h.conservativeResize(nodes_h.rows(), nodes_h.cols()+1);
                        nodes_h.col(nodes_h.cols()-1) = MatrixXd::Ones(nodes_h.rows(), 1);
                        MatrixXd image_coords = (proj_matrix * nodes_h.transpose()).transpose();

                        int pix_coord_1_x = static_cast<int>(image_coords(0, 0)/image_coords(0, 2));
                        int pix_coord_1_y = static_cast<int>(image_coords(0, 1)/image_coords(0, 2));
                        int pix_coord_2_x = static_cast<int>(image_coords(1, 0)/image_coords(1, 2));
                        int pix_coord_2_y = static_cast<int>(image_coords(1, 1)/image_coords(1, 2));
                    
                        int extra_border = 30;

                        // best scenarios: min_corner and max_corner are the top left and bottom right corners
                        if (pix_coord_1_x <= pix_coord_2_x && pix_coord_1_y <= pix_coord_2_y) {
                            // cv::Point p1(pix_coord_1_x - extra_border, pix_coord_1_y - extra_border);
                            // cv::Point p2(pix_coord_2_x + extra_border, pix_coord_2_y + extra_border);
                            // cv::rectangle(occlusion_mask, p1, p2, cv::Scalar(0, 0, 0), -1);
                            top_left_x = pix_coord_1_x - extra_border;
                            if (top_left_x < 0) {top_left_x = 0;}
                            top_left_y = pix_coord_1_y - extra_border;
                            if (top_left_y < 0) {top_left_y = 0;}
                            bottom_right_x = pix_coord_2_x + extra_border;
                            if (bottom_right_x >= cur_image_orig.cols) {bottom_right_x = cur_image_orig.cols-1;}
                            bottom_right_y = pix_coord_2_y + extra_border;
                            if (bottom_right_y >= cur_image_orig.rows) {bottom_right_y = cur_image_orig.rows-1;}
                        }
                        // best scenarios: min_corner and max_corner are the top left and bottom right corners
                        else if (pix_coord_2_x <= pix_coord_1_x && pix_coord_2_y <= pix_coord_1_y) {
                            // cv::Point p1(pix_coord_2_x - extra_border, pix_coord_2_y - extra_border);
                            // cv::Point p2(pix_coord_1_x + extra_border, pix_coord_1_y + extra_border);
                            // cv::rectangle(occlusion_mask, p1, p2, cv::Scalar(0, 0, 0), -1);
                            top_left_x = pix_coord_2_x - extra_border;
                            if (top_left_x < 0) {top_left_x = 0;}
                            top_left_y = pix_coord_2_y - extra_border;
                            if (top_left_y < 0) {top_left_y = 0;}
                            bottom_right_x = pix_coord_1_x + extra_border;
                            if (bottom_right_x >= cur_image_orig.cols) {bottom_right_x = cur_image_orig.cols-1;}
                            bottom_right_y = pix_coord_1_y + extra_border;
                            if (bottom_right_y >= cur_image_orig.rows) {bottom_right_y = cur_image_orig.rows-1;}
                        }
                        // min_corner is top right, max_corner is bottom left
                        else if (pix_coord_2_x <= pix_coord_1_x && pix_coord_1_y <= pix_coord_2_y) {
                            // cv::Point p1(pix_coord_2_x - extra_border, pix_coord_1_y - extra_border);
                            // cv::Point p2(pix_coord_1_x + extra_border, pix_coord_2_y + extra_border);
                            // cv::rectangle(occlusion_mask, p1, p2, cv::Scalar(0, 0, 0), -1);
                            top_left_x = pix_coord_2_x - extra_border;
                            if (top_left_x < 0) {top_left_x = 0;}
                            top_left_y = pix_coord_1_y - extra_border;
                            if (top_left_y < 0) {top_left_y = 0;}
                            bottom_right_x = pix_coord_1_x + extra_border;
                            if (bottom_right_x >= cur_image_orig.cols) {bottom_right_x = cur_image_orig.cols-1;}
                            bottom_right_y = pix_coord_2_y + extra_border;
                            if (bottom_right_y >= cur_image_orig.rows) {bottom_right_y = cur_image_orig.rows-1;}
                        }
                        // max_corner is top right, min_corner is bottom left
                        else {
                            // cv::Point p1(pix_coord_1_x - extra_border, pix_coord_2_y - extra_border);
                            // cv::Point p2(pix_coord_2_x + extra_border, pix_coord_1_y + extra_border);
                            // cv::rectangle(occlusion_mask, p1, p2, cv::Scalar(0, 0, 0), -1);
                            top_left_x = pix_coord_1_x - extra_border;
                            if (top_left_x < 0) {top_left_x = 0;}
                            top_left_y = pix_coord_2_y - extra_border;
                            if (top_left_y < 0) {top_left_y = 0;}
                            bottom_right_x = pix_coord_2_x + extra_border;
                            if (bottom_right_x >= cur_image_orig.cols) {bottom_right_x = cur_image_orig.cols-1;}
                            bottom_right_y = pix_coord_1_y + extra_border;
                            if (bottom_right_y >= cur_image_orig.rows) {bottom_right_y = cur_image_orig.rows-1;}
                        }

                        std_msgs::Int32MultiArray corners_arr;
                        corners_arr.data = {top_left_x, top_left_y, bottom_right_x, bottom_right_y};
                        corners_arr_pub.publish(corners_arr);
                    }
                }

                else if (bag_file == 1) {
                    top_left_x = 840;
                    top_left_y = 408;
                    bottom_right_x = 1191;
                    bottom_right_y = 678;

                    std_msgs::Int32MultiArray corners_arr;
                    corners_arr.data = {top_left_x, top_left_y, bottom_right_x, bottom_right_y};
                    corners_arr_pub.publish(corners_arr);
                }

                else if (bag_file == 2) {
                    top_left_x = 780;
                    top_left_y = 120;
                    bottom_right_x = 1050;
                    bottom_right_y = 290;

                    std_msgs::Int32MultiArray corners_arr;
                    corners_arr.data = {top_left_x, top_left_y, bottom_right_x, bottom_right_y};
                    corners_arr_pub.publish(corners_arr);
                }

                else if (bag_file == 4) {
                    top_left_x = 543;
                    top_left_y = 276;
                    bottom_right_x = 738;
                    bottom_right_y = 383;

                    std_msgs::Int32MultiArray corners_arr;
                    corners_arr.data = {top_left_x, top_left_y, bottom_right_x, bottom_right_y};
                    corners_arr_pub.publish(corners_arr);
                }

                else if (bag_file == 5) {
                    top_left_x = 300;
                    top_left_y = 317;
                    bottom_right_x = 698;
                    bottom_right_y = 440;

                    std_msgs::Int32MultiArray corners_arr;
                    corners_arr.data = {top_left_x, top_left_y, bottom_right_x, bottom_right_y};
                    corners_arr_pub.publish(corners_arr);
                }

                else {
                    throw std::invalid_argument("Invalid bag file ID!");
                }
            }

            // compute error
            if (save_errors) {
                cur_error = tracking_evaluator.compute_and_save_error(Y_track, Y_true);
            }
            else {
                cur_error = tracking_evaluator.compute_error(Y_track, Y_true);
            }
            std::cout << "error = " << cur_error << std::endl;

            // optional pub and save result image
            if (time_from_start > tracking_evaluator.recording_start_time() + tracking_evaluator.wait_before_occlusion()) {
                cv::Point p1(top_left_x, top_left_y);
                cv::Point p2(bottom_right_x, bottom_right_y);
                cv::rectangle(eval_img, p1, p2, cv::Scalar(0, 0, 0), -1);
                eval_img = 0.5*eval_img + 0.5*cur_image_orig;

                if (bag_file == 4) {
                    // cv::putText(tracking_img, "occlusion", cv::Point(occlusion_corner_j-190, occlusion_corner_i_2-5), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 0, 240), 2);
                    cv::putText(eval_img, "occlusion", cv::Point(top_left_x-190, bottom_right_y-5), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 0, 240), 2);
                }
                else {
                    cv::putText(eval_img, "occlusion", cv::Point(top_left_x, top_left_y-10), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 0, 240), 2);
                }
            }
            else {
                cur_error = tracking_evaluator.compute_error(Y_track, Y_true);
            }
        }
    }

    // project tracking results onto the image
    MatrixXd Y_track_h = Y_track.replicate(1, 1);
    Y_track_h.conservativeResize(Y_track_h.rows(), Y_track_h.cols()+1);
    Y_track_h.col(Y_track_h.cols()-1) = MatrixXd::Ones(Y_track_h.rows(), 1);

    // project and pub image
    MatrixXd image_coords_Y = (proj_matrix * Y_track_h.transpose()).transpose();
    // draw points
    for (int i = 0; i < image_coords_Y.rows(); i ++) {

        int row = static_cast<int>(image_coords_Y(i, 0)/image_coords_Y(i, 2));
        int col = static_cast<int>(image_coords_Y(i, 1)/image_coords_Y(i, 2));

        cv::Scalar point_color;
        cv::Scalar line_color;
        int line_width = 3;
        int point_radius = 7;
        // was 2 and 5

        if (row <= bottom_right_x && row >= top_left_x && col <= bottom_right_y && col >= top_left_y) {
            point_color = cv::Scalar(0, 0, 255);
            line_color = cv::Scalar(0, 0, 255);
        }
        else {
            point_color = cv::Scalar(0, 150, 255);
            line_color = cv::Scalar(0, 255, 0);
        }

        if (i != image_coords_Y.rows()-1) {
            cv::line(eval_img, cv::Point(row, col),
                cv::Point(static_cast<int>(image_coords_Y(i+1, 0)/image_coords_Y(i+1, 2)), 
                static_cast<int>(image_coords_Y(i+1, 1)/image_coords_Y(i+1, 2))),
                line_color, line_width);
        }

        cv::circle(eval_img, cv::Point(row, col), point_radius, point_color, -1);
    }

    // if (cur_error != -1) {
    //     std::string err = "Avg error per node: " + std::to_string(cur_error * 1000);
    //     cv::putText(eval_img, err.substr(0, err.find(".")+3) + "mm", cv::Point(20, eval_img.rows - 20), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 0, 0), 2);
    // }

    // save image
    if (save_images) {
        double diff = time_from_start - tracking_evaluator.recording_start_time();
        double time_step = 0.5;
        if (bag_file == 0) {
            time_step = 1.0;
        }
        if ((int)(diff/time_step) == tracking_evaluator.image_counter() && fabs(diff-(tracking_evaluator.image_counter()*time_step)) <= 0.1) {
            std::string dir;
            // 0 -> statinary.bag; 1 -> with_gripper_perpendicular.bag; 2 -> with_gripper_parallel.bag
            if (bag_file == 0) {
                dir = save_location + "images/" + alg + "_" + std::to_string(trial) + "_" + std::to_string(pct_occlusion) + "_stationary_frame_" + std::to_string(tracking_evaluator.image_counter()) + ".png";
            }
            else if (bag_file == 1) {
                dir = save_location + "images/" + alg + "_" + std::to_string(trial) + "_" + std::to_string(pct_occlusion) + "_perpendicular_motion_frame_" + std::to_string(tracking_evaluator.image_counter()) + ".png";
            }
            else if (bag_file == 2) {
                dir = save_location + "images/" + alg + "_" + std::to_string(trial) + "_" + std::to_string(pct_occlusion) + "_parallel_motion_frame_" + std::to_string(tracking_evaluator.image_counter()) + ".png";
            }
            else if (bag_file == 3) {
                dir = save_location + "images/" + alg + "_" + std::to_string(trial) + "_" + std::to_string(pct_occlusion) + "_self_occlusion_frame_" + std::to_string(tracking_evaluator.image_counter()) + ".png";
            }
            else if (bag_file == 4) {
                dir = save_location + "images/" + alg + "_" + std::to_string(trial) + "_" + std::to_string(pct_occlusion) + "_short_rope_folding_frame_" + std::to_string(tracking_evaluator.image_counter()) + ".png";
            }
            else if (bag_file == 5) {
                dir = save_location + "images/" + alg + "_" + std::to_string(trial) + "_" + std::to_string(pct_occlusion) + "_short_rope_stationary_frame_" + std::to_string(tracking_evaluator.image_counter()) + ".png";
            }
            cv::imwrite(dir, eval_img);
            tracking_evaluator.increment_image_counter();
        }
    }

    eval_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", eval_img).toImageMsg();
    return eval_img_msg;
}   

int main(int argc, char **argv) {
    ros::init(argc, argv, "evaluation");
    ros::NodeHandle nh;

    proj_matrix << 918.359130859375, 0.0, 645.8908081054688, 0.0,
                   0.0, 916.265869140625, 354.02392578125, 0.0,
                   0.0, 0.0, 1.0, 0.0;

    // load params
    nh.getParam("/evaluation/bag_file", bag_file);
    nh.getParam("/evaluation/trial", trial);
    nh.getParam("/evaluation/alg", alg);
    nh.getParam("/evaluation/bag_dir", bag_dir);
    nh.getParam("/evaluation/save_location", save_location);
    nh.getParam("/evaluation/pct_occlusion", pct_occlusion);
    nh.getParam("/evaluation/start_record_at", start_record_at);
    nh.getParam("/evaluation/exit_at", exit_at);
    nh.getParam("/evaluation/wait_before_occlusion", wait_before_occlusion);
    nh.getParam("/evaluation/bag_rate", bag_rate);
    nh.getParam("/evaluation/num_of_nodes", num_of_nodes);
    nh.getParam("/evaluation/save_images", save_images);
    nh.getParam("/evaluation/save_errors", save_errors);

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
    tracking_evaluator = evaluator(rgb_count, trial, pct_occlusion, alg, bag_file, save_location, start_record_at, exit_at, wait_before_occlusion, bag_rate, num_of_nodes);

    image_transport::ImageTransport it(nh);
    image_transport::Publisher eval_img_pub = it.advertise("/eval_img", 10);
    corners_arr_pub = nh.advertise<std_msgs::Int32MultiArray>("/corners", 10);

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/camera/color/image_raw", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub(nh, "/camera/depth/color/points", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> result_sub(nh, "/" + alg + "/results_pc", 10);
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
            sensor_msgs::ImagePtr eval_img = Callback(img_msg, pc_msg, result_msg);
            eval_img_pub.publish(eval_img);
        }
    );
    
    ros::spin();
}