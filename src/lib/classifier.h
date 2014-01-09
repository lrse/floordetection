#ifndef __CLASSIFIER_H__
#define __CLASSIFIER_H__

#include <opencv2/opencv.hpp>
#include <list>
#include <ros/ros.h>

namespace floordetection {
  class Segment {
    public:
      Segment(void);
      
      float distance(const Segment& m, const cv::Vec3f& minimums = cv::Vec3f(FLT_MAX, FLT_MAX, FLT_MAX)) const;
      float distance(const cv::Vec3f& elem, const cv::Vec3f& minimums = cv::Vec3f(FLT_MAX, FLT_MAX, FLT_MAX)) const;
      void mix(const Segment& other, float alpha = -1);
      
      cv::Vec3f mu;
      cv::Matx33f sigma;
      
      cv::Vec3f mu_hsv;
      cv::Matx33f sigma_hsv;
      
      int floor_pixels;
      int mass;
  
      float floor_certainty;
  };
  
  class Classifier {
    public:
      Classifier(ros::NodeHandle& ros_node);

      void save_parameters(void);
      
      void classify(const cv::Mat& input, const cv::Mat& input_hsv, const cv::Mat& labeled,
                    std::vector<Segment>& label2segment, cv::Mat& output, cv::Rect& roi);
                    
      void extract_training_models(const cv::Rect& roi, const cv::Mat& labeled,
                             const std::vector<Segment>& label2segment, std::list<Segment>& training_models);
      void update_learnt_models(const std::list<Segment>& training_models);
      void classify_segments(std::vector<Segment>& label2segment, int floor_mass);
      void classify_pixels(const cv::Mat& input_hsv, int floor_mass);
      
      static void set_minimum_cov(cv::Matx33f& cov, const cv::Vec3f& minimums);
      
      //Segment floor_segment, classified_segment;
      std::list<Segment> learnt_models, training_models;
      
      cv::Mat training_model_colors;
      cv::Mat debug;

      double rectangle_width, rectangle_height;
      cv::Vec3f threshold, threshold_merge;
      int models;

    private:
      ros::NodeHandle& ros_node;
  };
}

std::ostream& operator<<(std::ostream& out, const floordetection::Segment& s);

#endif
