#include <cv_bridge/cv_bridge.h>
#include <map>
#include "common.h"
#include "detector.h"
namespace fd = floordetection;

fd::Detector::Detector(ros::NodeHandle& _ros_node, bool _show_windows) :
  ros_node(_ros_node), horizon_finder(_ros_node), segmenter(_ros_node), classifier(_ros_node)
{
  show_windows = _show_windows;
  if (show_windows) {
    cvStartWindowThread();
    cv::namedWindow("labeled");
    cv::namedWindow("avg_hsv");
    cv::namedWindow("avg_rgb");
    cv::namedWindow("contour");
  }

  ros_node.param("median_blur", median_blur, 7);
  ros_node.param("erode_size", erode_size, 10);
}

void fd::Detector::save_parameters(void)
{
  ros_node.setParam("median_blur", median_blur);
  ros_node.setParam("erode_size", erode_size);
}

void fd::Detector::detect(const cv::Mat& input, std::vector<cv::Point>& frontier, int& horizon_level, cv::Rect& roi)
{
  cv::Mat input_original;
  input.copyTo(input_original);
  
  /* blur image */
  cv::Mat blured;
  cv::medianBlur(input, blured, median_blur);
  
  /* detect horizon */
  horizon_level = horizon_finder.find(blured);
  ROS_INFO_STREAM("horizon: " << horizon_level << "/" << input.rows);
  cv::Mat input_cropped = input(cv::Range(horizon_level, input_original.rows - 1), cv::Range::all());
  blured = blured(cv::Range(horizon_level, input_original.rows - 1), cv::Range::all());

  /* convert input to HSV */
  cv::Mat tmp, input_hsv;
  input_cropped.convertTo(tmp, CV_32FC3);
  tmp *= (1 / 255.0f);
  cv::cvtColor(tmp, input_hsv, CV_RGB2HSV);
  
  /* convert blured to HSV also */
  cv::Mat blured_hsv;
  blured.convertTo(tmp, CV_32FC3);
  tmp *= (1 / 255.0f);
  cv::cvtColor(tmp, blured_hsv, CV_RGB2HSV);

  /* perform segmentation */
  cv::Mat labeled;
  segmenter.segment(blured, labeled);

  /* build segment list */
  std::vector<Segment> label2segment;
  extract_segments(input_cropped, input_hsv, labeled, label2segment);

  /* classify segments */ 
  cv::Mat classified;
  classifier.classify(blured/*input*/, blured_hsv/*input_hsv*/, labeled, label2segment, classified, roi); // TODO: use blured to classify? (change also detect_segments)

  /* compute road contour */
  extract_contour(classified, frontier, roi);
  ROS_INFO_STREAM("frontier size: " << frontier.size());

  if (show_windows) {
    cv::Mat contour;
    cv::cvtColor(input_cropped, contour, CV_RGB2BGR);
    cv::drawContours(contour, std::vector< std::vector<cv::Point> >(1, frontier), 0, cv::Scalar(255, 255, 0));
    cv::imshow("contour", contour);
  }
    
  /* step 5: robot control law */
  //cv::Mat final_lower = final(cv::Range(horizon_level, final.rows), cv::Range::all());
  //robot_controller.apply_control(frontier, final_lower, final);
}

void fd::Detector::extract_segments(const cv::Mat& input, const cv::Mat& input_hsv, cv::Mat& labeled, std::vector<fd::Segment>& label2segment)
{
  int* labeled_ptr;
  uint size = input.rows * input.cols;
  int iterations = 0;
  
  /* rewrite labels for conitnuos numbering */
  std::map<int, int> label_equiv;
  labeled_ptr = labeled.ptr<int>(0);
  int last_label = 0;
  for (uint i = 0; i < size; i++, labeled_ptr++) {
    int& l = *labeled_ptr;
    std::map<int, int>::iterator it = label_equiv.find(l);
    if (it == label_equiv.end()) { label_equiv.insert(std::pair<int,int>(l, last_label)); l = last_label; last_label++; }
    else { l = it->second; }
  }
  label2segment.resize(last_label);
  
  /* generate random color for each segment */
  cv::RNG rng = cv::theRNG();
  std::map<int, cv::Vec3b> colors;
  labeled_ptr = labeled.ptr<int>(0);
  const cv::Vec3b* input_ptr = input.ptr<cv::Vec3b>(0);
  for (uint i = 0; i < size; i++, labeled_ptr++, input_ptr++) {
    int label = *labeled_ptr;    
    label2segment[label].mu += cv::Vec3f((*input_ptr)(0) / 255.0f, (*input_ptr)(1) / 255.0f, (*input_ptr)(2) / 255.0f);    
    label2segment[label].mass++;
    
    if (show_windows) {
      if (colors.find(label) == colors.end()) {
        colors[label] = cv::Vec3b(((int)rng % 16) * 16, ((int)rng % 16) * 16, ((int)rng % 16)*16);
      }
    }
  }
  //cout << "detected segments: " << label2segment.size() << " (in " << iterations << " iterations)" << endl;
  
  /* finish avg calculation */
  for (int i = 0; i < label2segment.size(); i++) {
    Segment& s = label2segment[i];
    int n = s.mass;
    s.mu *= 1 / (float)n;
    rgb2hsv(s.mu, s.mu_hsv);
  }
  
  /* compute covariance */
  input_ptr = input.ptr<cv::Vec3b>(0);
  const cv::Vec3f* input_hsv_ptr = input_hsv.ptr<cv::Vec3f>(0);
  labeled_ptr = labeled.ptr<int>(0);
  for (uint i = 0; i < size; i++, labeled_ptr++, input_ptr++, input_hsv_ptr++) {
    int label = *labeled_ptr;
    Segment& segment = label2segment[label];
    
    const cv::Vec3b& input3b = *input_ptr;
    cv::Vec3f sample(input3b(0) / 255.0f, input3b(1) / 255.0f, input3b(2) / 255.0f);
    cv::Vec3f diff = (sample - segment.mu);
    segment.sigma += diff * diff.t();     
    
    cv::Vec3f diff_hsv = hsv_diff(*input_hsv_ptr, segment.mu_hsv);
    segment.sigma_hsv += diff_hsv * diff_hsv.t();
  }
  
  /* finish cov calculation */
  for (int i = 0; i < label2segment.size(); i++) {
    Segment& s = label2segment[i];
    int n = s.mass;
    //cout << (1 / (float)(n - 1)) << endl;
    s.sigma *= 1 / (float)(n - 1);
    s.sigma_hsv *= 1 / (float)(n - 1);
  }

  // convert to hsv
  /*cv::cvtColor(cv::Mat(mu), cv::Mat(mu), CV_RGB2HSV);
  mu(0) /= 360.0f;*/

  if (show_windows) {
    /* create colored images */
    cv::Mat colored_labels(labeled.size(), CV_8UC3);
    cv::Mat avg_rgb(labeled.size(), CV_8UC3);
    cv::Mat avg_hsv(labeled.size(), CV_8UC3);
    cv::Vec3b* colored_labels_ptr = colored_labels.ptr<cv::Vec3b>(0);
    cv::Vec3b* avg_rgb_ptr = avg_rgb.ptr<cv::Vec3b>(0);
    cv::Vec3b* avg_hsv_ptr = avg_hsv.ptr<cv::Vec3b>(0);
    labeled_ptr = labeled.ptr<int>(0);
    for (uint i = 0; i < size; i++, colored_labels_ptr++, avg_rgb_ptr++, avg_hsv_ptr++, labeled_ptr++) {
      int label = *labeled_ptr;
      *colored_labels_ptr = colors[label];
      cv::Vec3f avg = label2segment[label].mu * 255.0f;
      *avg_rgb_ptr = cv::Vec3b((uchar)roundf(avg(0)), (uchar)roundf(avg(1)), (uchar)roundf(avg(2)));
      avg = label2segment[label].mu_hsv; avg(0) /= 360.0; avg *= 180.0f;
      *avg_hsv_ptr = cv::Vec3b((uchar)roundf(avg(0)), (uchar)roundf(avg(1)), (uchar)roundf(avg(2)));
    }
    cv::Mat debug1,debug2, debug3;
    cv::cvtColor(colored_labels, debug1, CV_RGB2BGR);
    cv::imshow("labeled", debug1);
    cv::cvtColor(avg_rgb, debug2, CV_RGB2BGR);
    cv::imshow("avg_rgb", debug2);
    cv::cvtColor(avg_hsv, debug3, CV_RGB2BGR);
    cv::imshow("avg_hsv", debug3);
  }
}

void fd::Detector::extract_contour(const cv::Mat& classified, std::vector<cv::Point>& frontier, const cv::Rect& roi)
{  
  /* find best contour */
  cv::Mat tmp;
  classified.copyTo(tmp); // need to copy, findContour overwrites input
  std::vector< std::vector<cv::Point> > contours;
  cv::findContours(tmp, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
  
  // find contour intersecting square's middle point
  size_t winner = contours.size();
  size_t max_size = 0;
  cv::Point center_point(roi.x + roi.width/2, roi.y + roi.height / 2);
  for (uint i = 0; i < contours.size(); i++) {
    size_t this_size = contours[i].size();
    if (this_size > max_size) { max_size = this_size; winner = i; }
  }
  //cout << "winner: " << winner << endl;

  cv::Mat eroded(classified.size(), CV_8UC1);
  if (winner != contours.size()) {
    if (erode_size > 0) {
      // re-create winner contour by drawing a mask
      cv::Mat classified_best(classified.size(), CV_8UC1); classified_best = cv::Scalar(0);
      cv::drawContours(classified_best, contours, winner, cv::Scalar(255), -1);

      // erode (grow black area) winner contour
      cv::Mat kernel(erode_size, erode_size, CV_8UC1);
      cv::circle(kernel, cv::Point((float)erode_size/2, (float)erode_size/2), (float)erode_size/2, cv::Scalar(255), -1);
      cv::morphologyEx(classified_best, eroded, cv::MORPH_OPEN, kernel);
      eroded.copyTo(tmp);
      
      // obtain new contours after eroding
      cv::findContours(tmp, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
      
      max_size = 0;
      winner = contours.size();
      
      for (uint i = 0; i < contours.size(); i++) {
        size_t this_size = contours[i].size();
        if (this_size > max_size) { max_size = this_size; winner = i; }
      }
      //cout << "winner: " << winner << endl;
    }
    if (winner != contours.size()) frontier = contours[winner];
  }
  
  //final.create(input_original.size(), CV_8UC3);
}

cv::Vec3f fd::hsv_diff(const cv::Vec3f& v1, const cv::Vec3f& v2) {
  cv::Vec3f res = (v1 - v2);
  if (res(0) > 180.0f) res(0) -= 360.0f;
  else if (res(0) < -180.0f) res(0) += 360.0f;
  return res;
}

void fd::rgb2hsv(const cv::Vec3f& rgb, cv::Vec3f& hsv) {
  cv::Vec3f tmp = rgb;
  cv::Mat_<cv::Vec3f> rgb_mat(1, 1, &tmp);
  cv::Mat_<cv::Vec3f> hsv_mat(1, 1, &hsv);  
  cv::cvtColor(rgb_mat, hsv_mat, CV_RGB2HSV);
}
