//
// Created by montura on 23.05.17.
//

#ifndef FEATURE_DETECTION_H
#define FEATURE_DETECTION_H




//#include "opencv2/video/tracking.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"




//// image processing with feature matching

// feature matching with SURF
void runSURF (cv::Mat const & img_1,
              cv::Mat const & img_2,
              std::vector<cv::KeyPoint> & keypoints_1,
              std::vector<cv::KeyPoint> & keypoints_2,
              cv::Mat & descriptors_1,
              cv::Mat & descriptors_2,
              std::vector<cv::DMatch> & matches)
{
    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 10;
    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(minHessian);
    surf->detect(img_1, keypoints_1);
    surf->detect(img_2, keypoints_2);

    //-- Step 2: Calculate descriptors (feature vectors)
    surf->compute(img_1, keypoints_1, descriptors_1);
    surf->compute(img_2, keypoints_2, descriptors_2);

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    cv::FlannBasedMatcher matcher;
    matcher.match(descriptors_1, descriptors_2, matches);

}

void featureMatching(cv::Mat const & img_1,
                     cv::Mat const & img_2,
                     std::vector<cv::Point2f> & imgpts1,
                     std::vector<cv::Point2f> & imgpts2)
{
    // Feature detection and matching
    cv::Mat descriptors_1, descriptors_2;
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
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
void featureTracking(cv::Mat const & img_1,
                     cv::Mat const & img_2,
                     std::vector<cv::Point2f> & points1,
                     std::vector<cv::Point2f> & points2,
                     std::vector<uchar> & status)
{
//this function automatically gets rid of points for which tracking fails
    std::vector<float> err;
    cv::Size winSize = cv::Size(21, 21);
    cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++) {
        cv::Point2f pt = points2.at(i - indexCorrection);
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
void featureDetection(cv::Mat const & img_1, std::vector<cv::Point2f> & points1)  {
    std::vector<cv::KeyPoint> keypoints_1;

    // FAST
    int fast_threshold = 15;
    bool nonmaxSuppression = true;
    FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);

    // SIFT (or SURF)
//    Ptr<SIFT> sift = SIFT::create();
//    sift->detect(img_1, keypoints_1);

    cv::KeyPoint::convert(keypoints_1, points1, std::vector<int>());
}

#endif //FEATURE_DETECTION_H
