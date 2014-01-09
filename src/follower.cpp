#include <ros/ros.h>
#include "lib/path_follower.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "follower");
  ros::NodeHandle n("~");

  floordetection::PathFollower f(n);
  ros::spin();
}
