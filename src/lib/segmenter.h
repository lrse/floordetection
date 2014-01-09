#ifndef __SEGMENTER_H__
#define __SEGMENTER_H__

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include "segmenter.h"

namespace floordetection {
  class Segmenter {
    public:
      Segmenter(ros::NodeHandle& ros_node);

      void save_parameters(void);
      
      void segment(const cv::Mat& intput, cv::Mat& output);

      double c_threshold;
      int min_segment;

    private:
      ros::NodeHandle& ros_node;
  };
}


#endif
