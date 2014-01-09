#include <cv_bridge/cv_bridge.h>
#include "path_follower.h"
namespace fd = floordetection;

fd::PathFollower::PathFollower(ros::NodeHandle& ros_node) : detector(ros_node, true), robot(ros_node)
{
  image_sub = ros_node.subscribe("/camera/image_raw", 1, &PathFollower::on_image, this);
  accum_xspeed = accum_aspeed = 0;
  cv::namedWindow("output");
}

void fd::PathFollower::on_image(const sensor_msgs::Image::ConstPtr& msg)
{
  cv_bridge::CvImageConstPtr bridge_ptr = cv_bridge::toCvShare(msg, "rgb8");
  follow(bridge_ptr->image);
}

void fd::PathFollower::follow(const cv::Mat& input)
{
  std::vector<cv::Point> frontier;
  int horizon_level;
  cv::Rect roi;
  detector.detect(input, frontier, horizon_level, roi);
  compute_control(frontier, input, horizon_level, roi);
}

void fd::PathFollower::compute_control(std::vector<cv::Point>& frontier, const cv::Mat& input, int horizon_level, const cv::Rect& roi)
{
  cv::Mat output;
  cv::cvtColor(input, output, CV_RGB2BGR);
  
  /* rasterize contour */
  cv::Size cropped_size(input.size().width, input.size().height - horizon_level);
  cv::Mat raster_contour(cropped_size, CV_8UC1);
  raster_contour = cv::Scalar(0);
  cv::drawContours(raster_contour, std::vector< std::vector<cv::Point> >(1, frontier), 0, cv::Scalar(255));
  
  std::vector<cv::Point> points_by_row(cropped_size.height, cv::Point(-1,-1));
  uchar* raster_ptr = raster_contour.ptr<uchar>(0);
  for (int i = 0; i < cropped_size.height; i++) {
    bool was_set = false;
    
    for (int j = 0; j < cropped_size.width; j++, raster_ptr++) {
      if (*raster_ptr == 255) {
        if (!was_set) { points_by_row[i] = cv::Point(j, j); was_set = true; }
        else {
          if (j > points_by_row[i].y) points_by_row[i].y = j;
        }
      }
    }
  }
  
  float xspeed = 0, aspeed = 0;
  int n_points = 0;
  for (uint i = 0; i < cropped_size.height; i++) {
    if (points_by_row[i].x == -1) continue;
    cv::Point minmax = points_by_row[i];    
    uint row = i;
    float x = (minmax.y + minmax.x) / 2;
    aspeed += x - cropped_size.width * 0.5;
    n_points++;
    
    cv::circle(output, cv::Point(x, row + horizon_level) , 1, cv::Scalar(0, 255, 255), 1, CV_AA);
  }
  
  float alpha = 0.00005, beta = 0.005;
  aspeed *= alpha;
  xspeed = beta * n_points - fabs(aspeed);
  if (xspeed < 0) xspeed = 0;
  if (xspeed > 0.3) xspeed = 0.3;
  
  float delta = 0.3;
  accum_aspeed = delta * accum_aspeed + (1 - delta) * aspeed;
  accum_xspeed = delta * accum_xspeed + (1 - delta) * xspeed;
  ROS_INFO_STREAM("xspeed: " << accum_xspeed << " aspeed: " << accum_aspeed);
  robot.set_speeds(accum_xspeed, accum_aspeed);
    
  size_t xorigin = 32;
  size_t xsize = 20;
  cv::putText(output, "^", cv::Point(1, 7), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(0,0,0), 1, CV_AA);
  cv::rectangle(output, cv::Point(xorigin - xsize, 7), cv::Point(xorigin + xsize, 1), cv::Scalar(255,255,255), -1, 4);
  cv::rectangle(output, cv::Point(xorigin - xsize, 7), cv::Point(xorigin + xsize, 1), cv::Scalar(0,0,0), 1, 4);
  cv::rectangle(output, cv::Point(xorigin - xsize, 7), cv::Point((xorigin - xsize) + xspeed * 2 * xsize, 1), cv::Scalar(0, 255, 0), -1, 4);
  cv::rectangle(output, cv::Point(xorigin - xsize, 7), cv::Point((xorigin - xsize) + xspeed * 2 * xsize, 1), cv::Scalar(0, 0, 0), 1, 4);
  
  cv::putText(output, (aspeed < 0 ? "< " : "> "), cv::Point(1, 16), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(0,0,0), 1, CV_AA);
  cv::rectangle(output, cv::Point(xorigin - xsize, 16), cv::Point(xorigin + xsize, 10), cv::Scalar(255,255,255), -1, 4);
  cv::rectangle(output, cv::Point(xorigin - xsize, 16), cv::Point(xorigin + xsize, 10), cv::Scalar(0,0,0), 1, 4);
  cv::rectangle(output, cv::Point(xorigin, 16), cv::Point(xorigin + aspeed * xsize, 10), cv::Scalar(0, 255, 0), -1, 4);
  cv::rectangle(output, cv::Point(xorigin, 16), cv::Point(xorigin + aspeed * xsize, 10), cv::Scalar(0, 0, 0), 1, 4);

  cv::Mat output_lower = output(cv::Range(horizon_level, output.rows), cv::Range::all());
  cv::drawContours(output_lower, std::vector<std::vector<cv::Point> >(1, frontier), 0, cv::Scalar(255, 0, 0), 3, CV_AA);
  cv::rectangle(output_lower, roi, cv::Scalar(0, 255, 0), 2, CV_AA);
  cv::circle(output_lower, cv::Point(roi.x + roi.width/2, roi.y + roi.height / 2), 1, cv::Scalar(0, 255, 0), 2, CV_AA);
  cv::line(output_lower, cv::Point(0, horizon_level), cv::Point(output.cols, horizon_level), cv::Scalar(0,255,255), 1);
  
  cv::imshow("output", output);
}
