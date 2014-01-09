#ifndef __ROBOT_H__
#define __ROBOT_H__

#include <ros/ros.h>

namespace floordetection {
  class Robot {
    public:
      Robot(ros::NodeHandle& ros_node);
      ~Robot(void);
      
      void set_speeds(float xspeed, float aspeed);
      void stop(void);
      
    private:
      ros::Publisher vel_pub;
      ros::Subscriber odo_sub;
  };
}

#endif
