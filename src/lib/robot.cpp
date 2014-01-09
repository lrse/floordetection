#include <geometry_msgs/Twist.h>
#include "robot.h"
namespace fd = floordetection;

fd::Robot::Robot(ros::NodeHandle& ros_node)
{
  vel_pub = ros_node.advertise<geometry_msgs::Twist>("/robot/cmd_vel", 1);
}

fd::Robot::~Robot(void)
{
  stop();
}

void fd::Robot::stop(void)
{
  set_speeds(0,0);
}
 
void fd::Robot::set_speeds(float xspeed, float aspeed)
{
  geometry_msgs::Twist t;
  t.linear.x = xspeed;
  t.angular.z = aspeed;
  vel_pub.publish(t);
}

