#ifndef __PATH_FOLLOWER_H__
#define __PATH_FOLLOWER_H__

#include <ros/ros.h>
#include <vector>
#include <std_srvs/Empty.h>
#include "detector.h"
#include "robot.h"
#include <universal_teleop/Event.h>
#include <actionlib/server/simple_action_server.h>
#include <floordetection/FollowPathAction.h>
#include <nav_msgs/Odometry.h>

namespace floordetection {
  class PathFollower {
    public:
      PathFollower(ros::NodeHandle& ros_node);

      void on_image(const sensor_msgs::Image::ConstPtr& msg);

      void follow(const cv::Mat& input);
      void compute_control(std::vector<cv::Point>& frontier, const cv::Mat& input, int horizon_level, const cv::Rect& roi);

      ros::Subscriber image_sub;
      ros::Subscriber event_sub;
      ros::Subscriber odo_sub;

      void on_event(const universal_teleop::Event::ConstPtr& msg);
      void on_odometry(const nav_msgs::Odometry::ConstPtr& msg);

      ros::ServiceServer start_service, stop_service;
      bool start_request(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);
      bool stop_request(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);

      typedef actionlib::SimpleActionServer<floordetection::FollowPathAction> ActionServer;
      boost::shared_ptr<ActionServer> action_server;
      floordetection::FollowPathFeedback action_feedback;
      floordetection::FollowPathGoal current_goal;
      void new_goal(void);
      bool has_last_odometry;
      nav_msgs::Odometry last_odometry;
      

      Detector detector;
      Robot robot;
      float accum_xspeed, accum_aspeed;
      double xspeed_scale, aspeed_scale;
      bool started;
  };
}

#endif
