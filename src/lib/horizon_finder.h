#ifndef __HORIZON_FINDER_H__
#define __HORIZON_FINDER_H__

#include <opencv2/opencv.hpp>
#include <ros/ros.h>

namespace floordetection {
  class HorizonFinder {
    public:
      HorizonFinder(ros::NodeHandle& ros_node);

      int find(const cv::Mat& input);

    private:
      int subimages;
      int last_horizon;
  
  };
}

#endif
