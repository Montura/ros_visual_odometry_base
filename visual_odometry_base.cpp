/*
 * Visual Odometry algorithm prototype (or template) for MIT dataset
 * http://projects.csail.mit.edu/stata/downloads.php
 * Works with *.bag files. Returns *.txt files.
 * File format:
 * timestamp t_x t_y t_z q_x q_y q_z q_w
 * t(t_x, t_y, t_z) - translation vector
 * q(q_x, q_y, q_z, q_w) - quaternion
 *
 * Requierments:
 * ROS Jade+
 * OpenCV 3.0+
 * Eigen 3.3
 * */

#include <utility>

#include <ros/ros.h>

#include <nav_msgs/Odometry.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
#include "message_filters/sync_policies/approximate_time.h"
#include <fstream>
#include "tf/transform_listener.h"

#include "eigen3/Eigen/Dense"
#include "opencv2/core/eigen.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace sensor_msgs;
using namespace nav_msgs;
using namespace message_filters;

using Vector3f = tf::Vector3;
using Quaternion = tf::Quaternion;

bool isRotationMatrix(Mat const & R) {
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3,3, shouldBeIdentity.type());
    return  norm(I, shouldBeIdentity) < 1e-6;
}

// Class Pose(Vector3f, Quaternion, timestamp) for saving transfrormation between two ROS-messages
class Pose {
public:
    Pose() : timestamp(0) {}

    // Consturctor from odometry message: topic "/base_odometry/odom"
    Pose(Odometry::ConstPtr const & msg)
            :
            timestamp(msg->header.stamp.sec),
            t(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z),
            q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w)
    {}

    // Consturctor from geometry message: topic "/robot_pose_ekf/odom_combined"
    Pose(geometry_msgs::PoseWithCovarianceStamped::ConstPtr const & msg)
            :
            timestamp(msg->header.stamp.sec),
            t(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z),
            q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w)
    {}

    // Consturctor from rotation matrix, translation vector and timestamp
    Pose(Mat const & R, Mat const & tr, size_t time_stamp) : timestamp(time_stamp)
    {
        assert(isRotationMatrix(R));
        Eigen::Matrix3d rot;
        cv2eigen(R, rot);
        Eigen::Quaterniond qvat(rot);
        qvat = Eigen::Quaterniond(rot);
        q = Quaternion(qvat.coeffs()[0], qvat.coeffs()[1], qvat.coeffs()[2], qvat.coeffs()[3]);
        t = Vector3f(tr.at<double>(0), tr.at<double>(1), tr.at<double>(2));
    }

    Pose operator-(Pose const & p) {
        Pose new_pose;
        new_pose.t = t - p.t;
        new_pose.q = q - p.q;
        new_pose.timestamp = timestamp;
        return new_pose;
    }

    Pose & operator=(Pose const & p) {
        if (*this != p) {
            timestamp = p.timestamp;
            t = p.t;
            q = p.q;
        }
        return *this;
    }

    friend ostream & operator<<(ostream & out, Pose const & p) {
        ROS_INFO("%i, %f, %f, %f, %f, %f, %f, %f",
                 p.timestamp,
                 p.t.x(),
                 p.t.y(),
                 p.t.z(),
                 p.q.x(),
                 p.q.y(),
                 p.q.z(),
                 p.q.w());
        return out;
    }

    friend ofstream & operator<<(ofstream & out, Pose const & p) {
        out << p.timestamp << ' '
            << p.t.x() << ' '
            << p.t.y() << ' '
            << p.t.z() << ' '
            << p.q.x() << ' '
            << p.q.y() << ' '
            << p.q.z() << ' '
            << p.q.w() << '\n';
        return out;
    }

    bool operator!=(Pose const & p) {
        return (t != p.t) || (q != p.q);
    }
private:
    size_t timestamp;
    Vector3f t;
    Quaternion q;
};

//// image processing with feature matching

// feature matching with SURF
void runSURF (Mat const & img_1,
              Mat const & img_2,
              vector<KeyPoint> & keypoints_1,
              vector<KeyPoint> & keypoints_2,
              Mat & descriptors_1,
              Mat & descriptors_2,
              vector< DMatch > & matches)
{
    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 10;
    Ptr<SURF> surf = SURF::create(minHessian);
    surf->detect(img_1, keypoints_1);
    surf->detect(img_2, keypoints_2);

    //-- Step 2: Calculate descriptors (feature vectors)
    surf->compute(img_1, keypoints_1, descriptors_1);
    surf->compute(img_2, keypoints_2, descriptors_2);

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    matcher.match(descriptors_1, descriptors_2, matches);

}

void featureMatching(Mat const & img_1,
                     Mat const & img_2,
                     vector<Point2f> & imgpts1,
                     vector<Point2f> & imgpts2)
{
    // Feature detection and matching
    Mat descriptors_1, descriptors_2;
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector< DMatch > matches;
    runSURF(img_1, img_2, keypoints_1, keypoints_2, descriptors_1, descriptors_2, matches);

    //  Get the keypoints from the matches
    for (auto i = 0; i < matches.size(); ++i) {
        // queryIdx is the "left" image
        imgpts1.push_back(keypoints_1[matches[i].queryIdx].pt);
        // trainIdx is the "right" image
        imgpts2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }
}

//// image processing with feature tracking like in https://github.com/avisingh599/mono-vo

// track features from img_1 to img_2
void featureTracking(Mat const & img_1,
                     Mat const & img_2,
                     vector<Point2f> & points1,
                     vector<Point2f> & points2,
                     vector<uchar> & status)
{
//this function automatically gets rid of points for which tracking fails
    vector<float> err;
    Size winSize = Size(21, 21);
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++) {
        Point2f pt = points2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0))  {
            if ((pt.x < 0) || (pt.y < 0))  {
                status.at(i) = 0;
            }
            points1.erase (points1.begin() + (i - indexCorrection));
            points2.erase (points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

// Feature detection on image. Here you can change feature detector (SITF, SURF, etc.)
void featureDetection(Mat const & img_1, vector<Point2f> & points1)  {
    vector<KeyPoint> keypoints_1;

    // FAST
    int fast_threshold = 15;
    bool nonmaxSuppression = true;
    FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);

    // SIFT (or SURF)
//    Ptr<SIFT> sift = SIFT::create();
//    sift->detect(img_1, keypoints_1);

    KeyPoint::convert(keypoints_1, points1, vector<int>());
}

// Camera intrinsic params for 2011-01-25-06-29-26.bag and 2011-01-24-06-18-27.bag
double const focal_length = 421.60728;
cv::Point2d const principal_point(316.5664, 253.0677);
//Camera Matrix
double IntrinsicParameters[] = {  421.60728, 0.0, 316.5664,
                                  0.0, 421.11498, 253.0677,
                                  0.0,        0.0,        1.0};
Mat const K(3, 3, CV_64F, IntrinsicParameters);

// global values for updating motion
vector<Mat> R_f(2), t_f(2);
Mat R, t;

void save_motion(vector<Pose> &motions, vector<Mat> const & R_f, vector<Mat> const & t_f, size_t const timestamp) {
    for (auto i = 0; i < motions.size(); ++i) {
        motions[i] = std::move(Pose(R_f[i], t_f[i], timestamp));
    }
}

// get initial R,t transformation between first two frames
void get_motion_from_essential_matrix(Mat const & E,
                                      vector<Point2f> const &  points1,
                                      vector<Point2f> const &  points2,
                                      vector<Mat> & R_f, vector<Mat> & t_f,
                                      vector<Pose> & motions,
                                      size_t const timestamp)
{
    recoverPose(E, points1, points2, R, t, focal_length, principal_point, noArray()); // method 1
    assert(isRotationMatrix(R));
    R_f[0] = R.clone();
    t_f[0] = t.clone();

    recoverPose(E, points1, points2, K, R, t, noArray()); // method 2
    assert(isRotationMatrix(R));
    R_f[1] = R.clone();
    t_f[1] = t.clone();

    save_motion(motions, R_f, t_f, timestamp);
}

// update R,t transformation between (i-1)-th and i-th frames
void update_motion(Mat const & E,
                   vector<Point2f> const &  points1,
                   vector<Point2f> const &  points2,
                   vector<Mat> & R_f, vector<Mat> & t_f,
                   vector<Pose> & motions,
                   size_t const timestamp)
{
    recoverPose(E, points1, points2, R, t, focal_length, principal_point, noArray()); // method 1
    assert(isRotationMatrix(R));
    t_f[0] += R_f[0] * t;
    R_f[0] = R * R_f[0];

    recoverPose(E, points1, points2, K, R, t, noArray()); // method 2
    assert(isRotationMatrix(R));
    t_f[1] += R_f[1] * t;
    R_f[1] = R * R_f[1];

    save_motion(motions, R_f, t_f, timestamp);
}

bool first_image_pair = true;
int const min_feature_num = 1000;
Mat prevImage, currImage;
vector<Point2f> prevFeatures, currFeatures;

// Main visual odometry process. You can change feature tracking to feature matching and "re-define" them
// with SURF|SIFT|ORB and etc.
// Also you can change calculating motion (R,t) from E.

void visual_odometry(Mat const & img_1, Mat const & img_2, size_t const timestamp, vector<Pose> & motions) {
    if (first_image_pair) {
        prevImage = img_1;
        currImage = img_2;
        featureDetection(img_1, prevFeatures); // detect features at first time
    } else {
        currImage = img_1; // work with previous image and current.
    }
    vector<uchar> status; // some vector for tracking
    // featureMatching(prevImage, currImage, prevFeatures, currFeatures);
    featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
    // find essential matrix
    Mat E = findEssentialMat(prevFeatures, currFeatures, focal_length, principal_point, RANSAC, 0.999, 1.0, noArray());
    if (first_image_pair) { // get initial pose at first time
        get_motion_from_essential_matrix(E, prevFeatures, currFeatures, R_f, t_f, motions, timestamp);
        first_image_pair = false;
    } else { // update pose
        update_motion(E, prevFeatures, currFeatures, R_f, t_f, motions, timestamp);
        // if a few features tracked from PREV to CURR image, detect again
        if (prevFeatures.size() < min_feature_num) {
            // featureMatching(prevImage, currImage, prevFeatures, currFeatures);
            featureDetection(prevImage, prevFeatures);
            featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
        }
    }
    // prepare data for next iteration
    prevImage = currImage.clone();
    prevFeatures = currFeatures;
}

// create Mat image from data from ImageConstPtr & image_msg
void create_image(vector<uint8_t> const & v, Mat & img) {
    for (int i = 0; i < 480; ++i) {
        for (int j = 0; j < 640; ++j) {
            img.at<uint8_t>(i, j) = v[i * 640 + j];
        }
    }
}

void write_pose_to_file(Pose const & some_pose, string const & filename) {
    string ouputfile = "/home/montura/ROS_Workaspace/catkin_ws/src/hello_world/src/" + filename + ".txt";
    ofstream file(ouputfile, ios::app);
    file << some_pose;
}

void write_motion_to_file(Pose const & odom_pose, Pose const & robot_pose, vector<Pose> const & visualOdometryPoses) {
    write_pose_to_file(std::move(odom_pose), "odom_pose");
    write_pose_to_file(std::move(robot_pose), "robot_pose");
    for (auto i = 0; i < visualOdometryPoses.size(); ++i) {
        string pose_number = "visual_pose_" + to_string(i);
        write_pose_to_file(std::move(visualOdometryPoses[i]), pose_number);
    }
}

// global values for callback steps
Pose prev_odom_pose;  // pose = (Vector(0,0,0), Quaternion(0,0,0,0))
Pose prev_robot_pose;  // pose = (Vector(0,0,0), Quaternion(0,0,0,0))
Mat prev_img(480, 640, CV_8UC1);  // black image
Mat curr_img(480, 640, CV_8UC1);  // black image
bool first_iter = true;

// process synchronized odemtery, geometry and image messages
void callback(const OdometryConstPtr& odometry,
              const geometry_msgs::PoseWithCovarianceStampedConstPtr& robot_pose,
              const ImageConstPtr & image_msg)
{
    // Tick that messages have come
    ROS_INFO_STREAM("Tick");

    // get odometry and robot poses
    Pose curr_odom_pose(odometry);
    Pose curr_robot_pose(robot_pose);

    // calc delta between i - 1 and i steps.
    Pose new_odom_pose = curr_odom_pose - prev_odom_pose;
    Pose new_robot_pose = curr_robot_pose - prev_robot_pose;

    // get image data, its timestamp and create mat image for processing
    vector<uint8_t> v = image_msg->data;
    size_t image_timestamp = image_msg->header.stamp.sec;
    create_image(v, curr_img);

    // check for 1st step.
    if (first_iter) {
        first_iter = false;
    } else {
        // create vector for several poses from visual odometry
        vector<Pose> visualOdometryPoses(2); // 2 - because of two method for get (R,t)
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
    message_filters::Subscriber<Odometry> baseOdomometryOdom(nodeHandle, "/base_odometry/odom", 1);
    message_filters::Subscriber<geometry_msgs::PoseWithCovarianceStamped> robotPoseEkf(nodeHandle, "/robot_pose_ekf/odom_combined", 1);
    message_filters::Subscriber<Image> stereoLeftImage(nodeHandle, "/wide_stereo/left/image_rect_throttle", 1);

    // approximate time because topics have different message sending frequency
    typedef sync_policies::ApproximateTime<Odometry, geometry_msgs::PoseWithCovarianceStamped, Image> MySyncPolicy;

    // process incoming messages
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), baseOdomometryOdom, robotPoseEkf, stereoLeftImage);
    sync.registerCallback(boost::bind(&callback, _1, _2, _3));

    ros::spin();
}
