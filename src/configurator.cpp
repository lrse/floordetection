#include <ros/ros.h>
#include "lib/configurator.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "configurator");
  ros::NodeHandle n("~");

  floordetection::Configurator c(n);
  ros::spin();
}
