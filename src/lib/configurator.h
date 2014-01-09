#ifndef __CONFIGURATOR_H__
#define __CONFIGURATOR_H__

#include <ros/ros.h>
#include <map>
#include <string>
#include "detector.h"

namespace floordetection {
  class Configurator {
    public:
      Configurator(ros::NodeHandle& n);

      void on_image(const sensor_msgs::Image::ConstPtr& msg);

      void configure(void);
      void create_variables(void);
      void create_trackbars(void);
      void read_variables(void);

      struct Variable {
        int value, max;
      };

    private:
      Detector detector;
      std::map<std::string, Variable> vars;

      ros::Subscriber image_sub;
      ros::NodeHandle& ros_node;
  };
}

#endif
