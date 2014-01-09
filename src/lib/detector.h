#ifndef __DETECTOR_H__
#define __DETECTOR_H__

#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include "classifier.h"
#include "segmenter.h"
#include "horizon_finder.h"

namespace floordetection
{
  class Detector {
    public:
      Detector(ros::NodeHandle& ros_node, bool show_windows = false);

      void configure(void);
      void detect(const cv::Mat& image, std::vector<cv::Point>& frontier, int& horizon_level, cv::Rect& roi);

      void save_parameters(void);
      
      /*cv::Mat input, input_hsv, avg_rgb, avg_hsv, blured, blured_hsv, labeled, colored_labels, classified, eroded, final;
      cv::Mat input_original;
      cv::Mat hue, saturation, value, hue_segments, saturation_segments, value_segments;*/

      Classifier classifier;
      Segmenter segmenter;
      HorizonFinder horizon_finder;

      int median_blur;
      int erode_size;
      
    private:
      ros::NodeHandle& ros_node;
      bool show_windows;
      
      void extract_segments(const cv::Mat& input, const cv::Mat& input_hsv, cv::Mat& labeled, std::vector<Segment>& label2segment);      
      void extract_contour(const cv::Mat& classified, std::vector<cv::Point>& frontier, const cv::Rect& roi);
  };
}

#endif
