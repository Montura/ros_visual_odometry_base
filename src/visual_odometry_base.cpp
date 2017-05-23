/*
 * Visual Odometry algorithm prototype (or template) for MIT dataset
 * http://projects.csail.mit.edu/stata/downloads.php
 * Works with *.bag files. Returns *.txt files.
 * File format:
 * timestamp t_x t_y t_z q_x q_y q_z q_w
 * t(t_x, t_y, t_z) - translation std::vector
 * q(q_x, q_y, q_z, q_w) - quaternion
 *
 * Requierments:
 * ROS Jade+
 * OpenCV 3.0+
 * Eigen 3.3
 * */

#include <message_filters/time_synchronizer.h>
#include "message_filters/sync_policies/approximate_time.h"
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>

#include "feature_detection.h"
#include "transformation_pose.h"
#include "motion_estimation.h"

using namespace sensor_msgs;
using namespace nav_msgs;
using namespace message_filters;
using namespace geometry_msgs;


bool first_image_pair = true;
int const min_feature_num = 1000;
cv::Mat prevImage, currImage;
std::vector<cv::Point2f> prevFeatures, currFeatures;

// Main visual odometry process. You can change feature tracking to feature matching and "re-define" them
// with SURF|SIFT|ORB and etc.
// Also you can change calculating motion (R,t) from E.
void visual_odometry(cv::Mat const & img_1, cv::Mat const & img_2, size_t const timestamp, std::vector<tfPose> & motions) {
    if (first_image_pair) {
        prevImage = img_1;
        currImage = img_2;
        featureDetection(img_1, prevFeatures); // detect features at first time
    } else {
        currImage = img_1; // work with previous image and current.
    }
    std::vector<uchar> status; // some std::vector for tracking
    // featurecv::Matching(prevImage, currImage, prevFeatures, currFeatures);
    featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
    // find essential matrix
    cv::Mat E = findEssentialMat(prevFeatures, currFeatures, focal_length, principal_point, cv::RANSAC, 0.999, 1.0, cv::noArray());
    if (first_image_pair) { // get initial pose at first time
        get_motion_from_essential_matrix(E, prevFeatures, currFeatures, R_f, t_f, motions, timestamp);
        first_image_pair = false;
    } else { // update pose
        update_motion(E, prevFeatures, currFeatures, R_f, t_f, motions, timestamp);
        // if a few features tracked from PREV to CURR image, detect again
        if (prevFeatures.size() < min_feature_num) {
            // featurecv::Matching(prevImage, currImage, prevFeatures, currFeatures);
            featureDetection(prevImage, prevFeatures);
            featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
        }
    }
    // prepare data for next iteration
    prevImage = currImage.clone();
    prevFeatures = currFeatures;
}

// create cv::Mat image from data from ImageConstPtr & image_msg
void create_image(std::vector<uint8_t> const & v, cv::Mat & img) {
    for (int i = 0; i < 480; ++i) {
        for (int j = 0; j < 640; ++j) {
            img.at<uint8_t>(i, j) = v[i * 640 + j];
        }
    }
}

// global values for callback steps
tfPose prev_odom_pose;  // pose = (Vector(0,0,0), Quaternion(0,0,0,0))
tfPose prev_robot_pose;  // pose = (Vector(0,0,0), Quaternion(0,0,0,0))
cv::Mat prev_img(480, 640, CV_8UC1);  // black image
cv::Mat curr_img(480, 640, CV_8UC1);  // black image
bool first_iter = true;

// process synchronized odemtery, geometry and image messages
void callback(const OdometryConstPtr & odometry,
              const geometry_msgs::PoseWithCovarianceStampedConstPtr & robot_pose,
              const ImageConstPtr & image_msg)
{
    // Tick that messages have come
    ROS_INFO_STREAM("Tick");

    // get odometry and robot poses
    tfPose curr_odom_pose(odometry);
    tfPose curr_robot_pose(robot_pose);

    // calc delta between i - 1 and i steps.
    tfPose new_odom_pose = curr_odom_pose - prev_odom_pose;
    tfPose new_robot_pose = curr_robot_pose - prev_robot_pose;

    // get image data, its timestamp and create mat image for processing
    std::vector<uint8_t> v = image_msg->data;
    size_t image_timestamp = image_msg->header.stamp.sec;
    create_image(v, curr_img);

    // check for 1st step.
    if (first_iter) {
        first_iter = false;
    } else {
        // create std::vector for several poses from visual odometry
        std::vector<tfPose> visualOdometryPoses(2); // 2 - because of two method for get (R,t)
        // process (i-1) and i images and get visual odometry poses
        visual_odometry(prev_img, curr_img, image_timestamp, visualOdometryPoses);
        // write updated poses
        write_motion_to_file(new_odom_pose, new_robot_pose, visualOdometryPoses);
    }

    // prepare data for next iteration
    prev_img = curr_img.clone();
    prev_odom_pose = curr_odom_pose;
    prev_robot_pose = curr_robot_pose;
}

int main(int argc, char **argv) {
    // init ROS node
    ros::init(argc, argv, "publish_scene");
    ros::NodeHandle nodeHandle;

    ROS_INFO_STREAM("ROS node alive!");
    // in another terminal type "rosbag play *.bag --pause". Then press "Enter".

    // synchronize topics: odometry, odom_combined(kalman filter) and Images
    Subscriber<Odometry> baseOdomometryOdom(nodeHandle, "/base_odometry/odom", 1);
    Subscriber<geometry_msgs::PoseWithCovarianceStamped> robotPoseEkf(nodeHandle, "/robot_pose_ekf/odom_combined", 1);
    Subscriber<Image> stereoLeftImage(nodeHandle, "/wide_stereo/left/image_rect_throttle", 1);

    // approximate time because topics have different message sending frequency
    typedef sync_policies::ApproximateTime<Odometry, geometry_msgs::PoseWithCovarianceStamped, Image> MySyncPolicy;

    // process incoming messages
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), baseOdomometryOdom, robotPoseEkf, stereoLeftImage);
    sync.registerCallback(boost::bind(&callback, _1, _2, _3));

    ros::spin();
}
