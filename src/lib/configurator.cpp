#include <cv_bridge/cv_bridge.h>
#include "configurator.h"
namespace fd = floordetection;

fd::Configurator::Configurator(ros::NodeHandle& _ros_node)
  : detector(_ros_node, true), ros_node(_ros_node)
{
  cv::namedWindow("options");

  create_variables();
  create_trackbars();

  image_sub = ros_node.subscribe("/camera/image_raw", 1, &Configurator::on_image, this);
}

void fd::Configurator::on_image(const sensor_msgs::Image::ConstPtr& msg)
{
  cv_bridge::CvImageConstPtr bridge_ptr = cv_bridge::toCvShare(msg, "rgb8");
  std::vector<cv::Point> frontier;
  int horizon_level;
  cv::Rect roi;
  detector.detect(bridge_ptr->image, frontier, horizon_level, roi);

  cv::imshow("options", cv::Mat::zeros(1, 256, CV_8UC3));
  read_variables();
}


void fd::Configurator::create_variables(void)
{
  vars["median_blur"].value = (detector.median_blur - 1) / 2;
  vars["median_blur"].max = 11;

  vars["erode_size"].value = detector.erode_size;
  vars["erode_size"].max = 20;

  vars["c_threshold"].value = detector.segmenter.c_threshold * 1000;
  vars["c_threshold"].max = 1000;

  vars["min_segment"].value = detector.segmenter.min_segment;
  vars["min_segment"].max = 1000;

  vars["rectangle_height"].value = detector.classifier.rectangle_height * 1000;
  vars["rectangle_height"].max = 1000;

  vars["rectangle_width"].value = detector.classifier.rectangle_width * 1000;
  vars["rectangle_width"].max = 1000;

  vars["models"].value = detector.classifier.models;
  vars["models"].max = 10;

  vars["threshold_h"].value = detector.classifier.threshold(0) * 1000;
  vars["threshold_s"].value = detector.classifier.threshold(1) * 1000;
  vars["threshold_v"].value = detector.classifier.threshold(2) * 1000;

  vars["threshold_h_merge"].value = detector.classifier.threshold_merge(0) * 1000;
  vars["threshold_s_merge"].value = detector.classifier.threshold_merge(1) * 1000;
  vars["threshold_v_merge"].value = detector.classifier.threshold_merge(2) * 1000;
  
  vars["threshold_h"].max = vars["threshold_s"].max = vars["threshold_v"].max = 1000;
  vars["threshold_h_merge"].max = vars["threshold_s_merge"].max = vars["threshold_v_merge"].max = 1000;
}

void fd::Configurator::read_variables(void)
{
  detector.median_blur = vars["median_blur"].value * 2 + 1;
  detector.erode_size = vars["erode_size"].value;
  //detector.save_parameters();

  detector.segmenter.c_threshold = vars["c_threshold"].value / 1000.0;
  detector.segmenter.min_segment = vars["min_segment"].value;
  //detector.segmenter.save_parameters();

  detector.classifier.rectangle_height = vars["rectangle_height"].value / 1000.0;
  detector.classifier.rectangle_width = vars["rectangle_width"].value / 1000.0;
  detector.classifier.models = vars["models"].value;
  detector.classifier.threshold(0) = vars["threshold_h"].value / 1000.0;
  detector.classifier.threshold(1) = vars["threshold_s"].value / 1000.0;
  detector.classifier.threshold(2) = vars["threshold_v"].value / 1000.0;
  detector.classifier.threshold_merge(0) = vars["threshold_h_merge"].value / 1000.0;
  detector.classifier.threshold_merge(1) = vars["threshold_s_merge"].value / 1000.0;
  detector.classifier.threshold_merge(2) = vars["threshold_v_merge"].value / 1000.0;
  //detector.classifier.save_parameters();
}

void fd::Configurator::create_trackbars(void)
{
  for (auto& kv : vars) cv::createTrackbar(kv.first, "options", &kv.second.value, kv.second.max);
}
