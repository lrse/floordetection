#ifndef __PATH_FOLLOWER_H__
#define __PATH_FOLLOWER_H__

#include <ros/ros.h>
#include <vector>
#include "detector.h"
#include "robot.h"

namespace floordetection {
  class PathFollower {
    public:
      PathFollower(ros::NodeHandle& ros_node);

      void on_image(const sensor_msgs::Image::ConstPtr& msg);

      void follow(const cv::Mat& input);
      void compute_control(std::vector<cv::Point>& frontier, const cv::Mat& input, int horizon_level, const cv::Rect& roi);

      ros::Subscriber image_sub;

      Detector detector;
      Robot robot;
      float accum_xspeed, accum_aspeed;
  };
}

#endif
