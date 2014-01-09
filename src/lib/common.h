#ifndef __COMMON_H__
#define __COMMON_H__

#include <opencv2/opencv.hpp>

namespace floordetection {
  cv::Vec3f hsv_diff(const cv::Vec3f& v1, const cv::Vec3f& v2);
  void rgb2hsv(const cv::Vec3f& rgb, cv::Vec3f& hsv);
}


#endif
