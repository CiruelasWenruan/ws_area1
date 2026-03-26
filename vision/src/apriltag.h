//
// Created by Administrator on 2026/3/24.
//

#ifndef WS_AREA1_APRILTAG_H
#define WS_AREA1_APRILTAG_H
#include <string>
#include <opencv2/core/core.hpp>   // 必须加！FileStorage 在这个头文件里
#include <opencv2/opencv.hpp>      // 通用 OpenCV 头文件

#endif //WS_AREA1_APRILTAG_H

cv::Mat imageOpt_for_Apriltag(cv::Mat& BGR)
{
    cv:Mat gray_image;
    cv::cvtColor(BGR, gray_image, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray_image, gray_image);
    return gray_image;
};


struct Apriltag{
    apriltag_family_t* family;
    apriltag_detector_t* detector;
    double size;
    int ID {};
    bool is_plan_tag {};
    Eigen::Vector3d position_corner_1 {};
    Eigen::Vector3d position_corner_2 {};
    Eigen::Vector3d position_corner_3 {};
    Eigen::Vector3d position_corner_4 {};
    Eigen::Vector3d R2_to_tag vector {};


void initialize_detector()
{
    detector = apriltag_detector_create();
    apriltag_detector_add_family(detector, family);
    detector->decimate;
};


void is_plan_tag_get ()
{
    if
    }

}
void location_tag_get ( cv::Mat image, ,)


}