//
// Created by montura on 23.05.17.
//

#ifndef TRANSFORMATION_POSE_H
#define TRANSFORMATION_POSE_H


#include <fstream>

#include "tf/transform_listener.h"
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <ros/ros.h>
#include <vector>

#include "opencv2/opencv.hpp"
#include "eigen3/Eigen/Dense"
#include "opencv2/core/eigen.hpp"

bool isRotationMatrix(cv::Mat const & R) {
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
    return  norm(I, shouldBeIdentity) < 1e-6;
}

using Vector3f = tf::Vector3;
using Quaternion = tf::Quaternion;

// Class tfPose(Vector3f, Quaternion, timestamp) for saving transfrormation between two ROS-messages
class tfPose {
public:
    tfPose() : timestamp(0) {}

    // Consturctor from odometry message: topic "/base_odometry/odom"
    tfPose(nav_msgs::Odometry::ConstPtr const & msg)
            :
            timestamp(msg->header.stamp.sec),
            t(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z),
            q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w)
    {}

    // Consturctor from geometry message: topic "/robot_pose_ekf/odom_combined"
    tfPose(geometry_msgs::PoseWithCovarianceStamped::ConstPtr const & msg)
            :
            timestamp(msg->header.stamp.sec),
            t(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z),
            q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w)
    {}

    // Consturctor from rotation matrix, translation vector and timestamp
    tfPose(cv::Mat const & R, cv::Mat const & tr, size_t time_stamp) : timestamp(time_stamp)
    {
        assert(isRotationMatrix(R));
        Eigen::Matrix3d rot;
        cv::cv2eigen(R, rot);
        Eigen::Quaterniond qvat(rot);
        qvat = Eigen::Quaterniond(rot);
        q = Quaternion(qvat.coeffs()[0], qvat.coeffs()[1], qvat.coeffs()[2], qvat.coeffs()[3]);
        t = Vector3f(tr.at<double>(0), tr.at<double>(1), tr.at<double>(2));
    }

    tfPose operator-(tfPose const & p) {
        tfPose new_pose;
        new_pose.t = t - p.t;
        new_pose.q = q - p.q;
        new_pose.timestamp = timestamp;
        return new_pose;
    }

    tfPose & operator=(tfPose const & p) {
        if (*this != p) {
            timestamp = p.timestamp;
            t = p.t;
            q = p.q;
        }
        return *this;
    }

    friend std::ostream & operator<<(std::ostream & out, tfPose const & p) {
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

    friend std::ofstream & operator<<(std::ofstream & out, tfPose const & p) {
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

    bool operator!=(tfPose const & p) {
        return (t != p.t) || (q != p.q);
    }

private:
    size_t timestamp;
    Vector3f t;
    Quaternion q;
};

void write_pose_to_file(tfPose const & some_pose, std::string const & filename) {
    std::string ouputfile = filename;
    std::ofstream file(ouputfile, std::ios::app);
    file << some_pose;
}

#endif //TRANSFORMATION_POSE_H
