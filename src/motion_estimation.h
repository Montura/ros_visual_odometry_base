//
// Created by montura on 23.05.17.
//

#ifndef MOTION_ESTIMATION_H
#define MOTION_ESTIMATION_H

#include "opencv2/opencv.hpp"
#include "transformation_pose.h"

// Camera intrinsic params for 2011-01-25-06-29-26.bag and 2011-01-24-06-18-27.bag
double const focal_length = 421.60728;
cv::Point2d const principal_point(316.5664, 253.0677);
double IntrinsicParameters[] = {  421.60728, 0.0, 316.5664,
                                  0.0, 421.11498, 253.0677,
                                  0.0,        0.0,        1.0};
cv::Mat const K(3, 3, CV_64F, IntrinsicParameters);


// global values for updating motion
std::vector<cv::Mat> R_f(2), t_f(2);
cv::Mat R, t;

void save_motion(std::vector<tfPose> &motions,
                 std::vector<cv::Mat> const & R_f,
                 std::vector<cv::Mat> const & t_f,
                 size_t const timestamp)
{
    for (auto i = 0; i < motions.size(); ++i) {
        motions[i] = std::move(tfPose(R_f[i], t_f[i], timestamp));
    }
}

// get initial R,t transformation between first two frames
void get_motion_from_essential_matrix(cv::Mat const & E,
                                      std::vector<cv::Point2f> const &  points1,
                                      std::vector<cv::Point2f> const &  points2,
                                      std::vector<cv::Mat> & R_f, std::vector<cv::Mat> & t_f,
                                      std::vector<tfPose> & motions,
                                      size_t const timestamp)
{
    recoverPose(E, points1, points2, R, t, focal_length, principal_point, cv::noArray()); // method 1
    assert(isRotationMatrix(R));
    R_f[0] = R.clone();
    t_f[0] = t.clone();

    recoverPose(E, points1, points2, K, R, t, cv::noArray()); // method 2
    assert(isRotationMatrix(R));
    R_f[1] = R.clone();
    t_f[1] = t.clone();

    save_motion(motions, R_f, t_f, timestamp);
}

// update R,t transformation between (i-1)-th and i-th frames
void update_motion(cv::Mat const & E,
                   std::vector<cv::Point2f> const &  points1,
                   std::vector<cv::Point2f> const &  points2,
                   std::vector<cv::Mat> & R_f, std::vector<cv::Mat> & t_f,
                   std::vector<tfPose> & motions,
                   size_t const timestamp)
{
    recoverPose(E, points1, points2, R, t, focal_length, principal_point, cv::noArray()); // method 1
    assert(isRotationMatrix(R));
    t_f[0] += R_f[0] * t;
    R_f[0] = R * R_f[0];

    recoverPose(E, points1, points2, K, R, t, cv::noArray()); // method 2
    assert(isRotationMatrix(R));
    t_f[1] += R_f[1] * t;
    R_f[1] = R * R_f[1];

    save_motion(motions, R_f, t_f, timestamp);
}


void write_motion_to_file(tfPose const & odom_pose,
                          tfPose const & robot_pose,
                          std::vector<tfPose> const & visualOdometryPoses)
{
    write_pose_to_file(std::move(odom_pose), "odom_pose");
    write_pose_to_file(std::move(robot_pose), "robot_pose");
    for (auto i = 0; i < visualOdometryPoses.size(); ++i) {
        std::string pose_number = "visual_pose_" + std::to_string(i);
        write_pose_to_file(std::move(visualOdometryPoses[i]), pose_number);
    }
}

#endif //MOTION_ESTIMATION_H
