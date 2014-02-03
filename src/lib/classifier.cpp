#include <list>
#include <map>
#include <iostream>
#include "classifier.h"
#include "common.h"
namespace fd = floordetection;

fd::Classifier::Classifier(ros::NodeHandle& _ros_node) :
  ros_node(_ros_node)
{
  ros_node.param("rectangle_width", rectangle_width, 0.5);
  ros_node.param("rectangle_height", rectangle_height, 0.2);
  ros_node.param("classifier_models", models, 1);

  std::vector<double> t;
  ros_node.param("classifier_threshold_merge", t, std::vector<double>(3, 0.015));
  for (int i = 0; i < 3; i++) threshold_merge(i) = t[i];
  ros_node.param("classifier_threshold", t, std::vector<double>(3, 0.015));
  for (int i = 0; i < 3; i++) threshold(i) = t[i];
}

void fd::Classifier::save_parameters(void)
{
  ros_node.setParam("rectangle_width", rectangle_width);
  ros_node.setParam("rectangle_height", rectangle_height);
  ros_node.setParam("classifier_models", models);

  std::vector<double> t(3);
  for (int i = 0; i < 3; i++) t[i] = threshold_merge(i);
  ros_node.setParam("classifier_threshold_merge", t);
  for (int i = 0; i < 3; i++) t[i] = threshold(i);
  ros_node.setParam("classifier_threshold", t);
}

void fd::Classifier::classify(const cv::Mat& input, const cv::Mat& input_hsv, const cv::Mat& labeled,
    std::vector<Segment>& label2segment, cv::Mat& output, cv::Rect& roi)
{
  /* determine size of rectangle */
  int rectangle_width_px  = input.size().width * rectangle_width;
  int rectangle_height_px  = input.size().height * rectangle_height;
  int rectangle_x = (input.size().width / 2) - (rectangle_width_px / 2);
  int rectangle_y = input.size().height - rectangle_height_px;
  //cout << "w/h: " << rectangle_width_px << " " << rectangle_height_px << endl;
  roi = cv::Rect(rectangle_x, rectangle_y, rectangle_width_px, rectangle_height_px);
  cv::Mat input_subregion = input(roi);
  //cout << "window size: " << input_subregion.rows * input_subregion.cols << endl;
  
  /* extract floor segments */
  std::list<Segment> training_models;
  extract_training_models(roi, labeled, label2segment, training_models);
  
  /* update learnt models */
  update_learnt_models(training_models);
      
  /* classify all segments */
  classify_segments(label2segment, input_subregion.rows * input_subregion.cols);
  
  /* create binary mask */
  const int* labeled_ptr;    
  output.create(input.size(), CV_8UC1);
  uchar* output_ptr = output.ptr<uchar>(0);
  labeled_ptr = labeled.ptr<int>(0);
  for (int k = 0; k < output.rows * output.cols; k++, labeled_ptr++, output_ptr++) {
    Segment& s = label2segment[*labeled_ptr];
    if (s.floor_certainty == 1) *output_ptr = 255;
    else *output_ptr = 0;
  }
}

void fd::Classifier::extract_training_models(const cv::Rect& roi, const cv::Mat& labeled,
  const std::vector<Segment>& label2segment, std::list<Segment>& training_models)
{
  training_models.clear();
  
  // 1. compute ammount of overlapping of each segment over the floor region
  std::map<int, int> overlaps;
  cv::Mat floor_labeled = labeled(roi);
  int floor_size = floor_labeled.rows * floor_labeled.cols;
  for (int i = 0; i < floor_labeled.rows; i++) {
    int* label_ptr = floor_labeled.ptr<int>(i);
    for (int j = 0; j < floor_labeled.cols; j++, label_ptr++) {
      int l = *label_ptr;
      std::map<int,int>::iterator it = overlaps.find(l);
      if (it == overlaps.end()) overlaps.insert(std::pair<int,int>(l, 1));
      else it->second++;
    }
  }
   
  // 2. load overlapping segments into trainig_models set
  int mass_sum = 0;
  for (std::map<int,int>::iterator it = overlaps.begin(); it != overlaps.end();  ++it) {
    training_models.push_back(label2segment[it->first]);
    training_models.back().mass = it->second;
    mass_sum += it->second;
    //cout << "over: " << training_models.back().mass << endl;
  }
  //cout << "#floor segments:" << overlaps.size() << endl;
  //cout << "mass sum: " << mass_sum << " " << floor_size << endl;
  
  // 3. combine similar segments to form minimal group  
  bool changed;
  do {
    changed = false;
    
    // 3.1. compute nearest neighbors
    size_t model_count = training_models.size();
    std::vector<int> nearest(model_count, -1);
    std::vector<std::list<Segment>::iterator> nearest_it(model_count, training_models.end());
    int i = 0;
    for (std::list<Segment>::iterator it = training_models.begin(); it != training_models.end(); ++it, ++i) {
      float min_dist = FLT_MAX;
      int j = 0;
      for (std::list<Segment>::iterator it2 = training_models.begin(); it2 != training_models.end(); ++it2, ++j) {
        if (i == j) continue;
        float d = it->distance(*it2, threshold_merge);
        if (d <= 1 && d < min_dist) { nearest[i] = j; min_dist = d; nearest_it[i] = it2; }
      }
    }  
  
    // 3.2. join corresponding pairs
    std::list<Segment> new_list;
    i = 0;
    for (std::list<Segment>::iterator it = training_models.begin(); it != training_models.end(); ++it, ++i) {
      int nearest_i = nearest[i];
      if (nearest_i == -1) new_list.push_back(*it);
      else if (i == nearest[nearest_i]) {
        if (nearest_i > i) {
          it->mix(*nearest_it[i]);
          new_list.push_back(*it);
          changed = true;
        }
      }
      else new_list.push_back(*it);
    }
    training_models = new_list;
  } while (changed);
  
  //#if SAVE_GRAPHICS
  int rect_size = 32, i = 0;
  training_model_colors.create(50, 32 * training_models.size(), CV_32FC3);
  training_model_colors = cv::Scalar(1,1,1);
  for (std::list<Segment>::iterator it = training_models.begin(); it != training_models.end(); ++it, ++i) {
    int coverage = (int)((it->mass / (float)floor_size) * 100);
    std::ostringstream ostr; ostr << coverage;
    cv::putText(training_model_colors, ostr.str(), cv::Point(i * rect_size + 5, 47), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(1,0,0), 1, CV_AA);
    cv::rectangle(training_model_colors, cv::Rect(i * rect_size, 0, rect_size, rect_size), cv::Scalar(it->mu), -1);
    
  }
  cv::cvtColor(training_model_colors, training_model_colors, CV_RGB2BGR);
  //#endif
}

void fd::Classifier::update_learnt_models(const std::list<Segment>& training_models)
{
  #if 0
  list<Segment> new_models;
  for (std::list<Segment>::const_iterator it = training_models.begin(); it != training_models.end(); ++it) {
    bool matched = false;
    for (std::list<Segment>::iterator it2 = learnt_models.begin(); it2 != learnt_models.end(); ++it2) {
      if (it->distance(*it2, threshold_merge) <= 1) { matched = true; it2->mix(*it); break; }
    }
    if (!matched) new_models.push_back(*it);      
  }
  
  learnt_models.insert(learnt_models.end(), new_models.begin(), new_models.end());
  if (learnt_models.size() > models) learnt_models.resize(models); // TODO: size() is O(n) on some implementations*/
  #else
  learnt_models = training_models;
  #endif
}

void fd::Classifier::classify_pixels(const cv::Mat& input_hsv, int floor_mass)
{
  cv::Mat classified_pixels(input_hsv.size(), CV_8UC1);
  uchar* mask_ptr = classified_pixels.ptr<uchar>(0);
  
  size_t pixels = input_hsv.rows * input_hsv.cols;
  const cv::Vec3f* pixel_ptr = input_hsv.ptr<cv::Vec3f>(0);
  for (int i = 0; i < pixels; i++, pixel_ptr++, mask_ptr++) {
    *mask_ptr = 0;
    for (std::list<Segment>::iterator it = learnt_models.begin(); it != learnt_models.end(); ++it) {
      float coverage = it->mass / (float)floor_mass;
      //cout << "model coverage: " << coverage << endl;
      if (coverage < 0.3) continue; // small clusters are ignored
      float dist = it->distance(*pixel_ptr, threshold);
      //cout << "dist: " << dist << endl;
      if (dist <= 1) { *mask_ptr = 255; break; }
    }
  }
  //classified_pixels.copyTo(debug);
  int erode_size = erode_size;
  cv::Mat kernel(erode_size, erode_size, CV_8UC1);
  cv::circle(kernel, cv::Point((float)erode_size/2, (float)erode_size/2), (float)erode_size/2, cv::Scalar(255), -1);
  cv::morphologyEx(classified_pixels, debug, cv::MORPH_OPEN, kernel);
}

void fd::Classifier::classify_segments(std::vector<Segment>& label2segment, int floor_mass)
{
  for (int i = 0; i < label2segment.size(); i++) {
    Segment& s = label2segment[i];
    for (std::list<Segment>::const_iterator it = learnt_models.begin(); it != learnt_models.end(); ++it) {
      float coverage = it->mass / (float)floor_mass;
      //cout << "model coverage: " << coverage << endl;
      if (coverage < 0.3) continue; // small clusters are ignored
      float dist = s.distance(*it, threshold);
      //cout << "dist: " << dist << endl;
      if (dist <= 1) { s.floor_certainty = 1; break; }
    }
  }
}



void fd::Classifier::set_minimum_cov(cv::Matx33f& cov, const cv::Vec3f& minimums)
{
  cv::Vec3f eigenvalues;
  cv::Matx33f eigenvectors;
  cv::eigen(cov, eigenvalues, eigenvectors);
  for (int i = 0; i < 3; i++) {
    if (eigenvalues(i) < (minimums(i) * (i == 0 ? 360 * 50 : 0.25))) eigenvalues(i) = minimums(i) * (i == 0 ? 360.0 * 50 : 0.25);
  }
  
  cov = eigenvectors * cv::Matx33f::diag(eigenvalues) * eigenvectors.t();
}


/********** Segment **************/
fd::Segment::Segment(void) {
  mass = 0;
  mu = mu_hsv = cv::Vec3f(0,0,0);
  sigma = sigma_hsv = cv::Matx33f::zeros();
  floor_certainty = 0; 
  floor_pixels = 0;
}

float fd::Segment::distance(const cv::Vec3f& elem, const cv::Vec3f& minimums) const
{
  cv::Matx33f new_sigma = sigma_hsv;
  Classifier::set_minimum_cov(new_sigma, minimums);
  
  cv::Vec3f diff = hsv_diff(mu_hsv, elem);
  cv::Vec<float,1> result = (diff.t() * new_sigma.inv() * diff); 
  return result(0);  
}

float fd::Segment::distance(const Segment& other, const cv::Vec3f& minimums) const
{ 
  cv::Matx33f sigma_sum = (sigma_hsv + other.sigma_hsv);
  Classifier::set_minimum_cov(sigma_sum, minimums);
  
  cv::Vec3f diff = hsv_diff(mu_hsv, other.mu_hsv);
  cv::Vec<float,1> result = (diff.t() * sigma_sum.inv() * diff); 
  
  /*cout << "diff: " << diff(0) << " " << diff(1) << " " << diff(2) << endl;
  cout << "sigma_sum2" << endl;
  cout << sigma_sum2(0,0) << "," << sigma_sum2(0,1) << "," << sigma_sum2(0,2) << endl;
  cout << sigma_sum2(1,0) << "," << sigma_sum2(1,1) << "," << sigma_sum2(1,2) << endl;
  cout << sigma_sum2(2,0) << "," << sigma_sum2(2,1) << "," << sigma_sum2(2,2) << endl;*/
  /*cout << "floor mu/sample mu: " << mu_hsv(0) << "," << mu_hsv(1) << "," << mu_hsv(2) << " " << other.mu_hsv(0) << "," << other.mu_hsv(1) << "," << other.mu_hsv(2) << endl;
  cout << "floor sigma: " << sigma_hsv(0,0) << "," << sigma_hsv(0,1) << "," << sigma_hsv(0,2) << endl
                    << sigma_hsv(1,0) << "," << sigma_hsv(1,1) << "," << sigma_hsv(1,2) << endl
                    << sigma_hsv(2,0) << "," << sigma_hsv(2,1) << "," << sigma_hsv(2,2) << endl;   
  cout << "sample sigma: " << other.sigma_hsv(0,0) << "," << other.sigma_hsv(0,1) << "," << other.sigma_hsv(0,2) << endl
                    << other.sigma_hsv(1,0) << "," << other.sigma_hsv(1,1) << "," << other.sigma_hsv(1,2) << endl
                    << other.sigma_hsv(2,0) << "," << other.sigma_hsv(2,1) << "," << other.sigma_hsv(2,2) << endl;
  cout << "inv sigma: " << sigma_sum(0,0) << "," << sigma_sum(0,1) << "," << sigma_sum(0,2) << endl
                  << sigma_sum(1,0) << "," << sigma_sum(1,1) << "," << sigma_sum(1,2) << endl
                  << sigma_sum(2,0) << "," << sigma_sum(2,1) << "," << sigma_sum(2,2) << endl;
  sigma_sum = sigma_sum.inv();
  cout << "sigma: " << sigma_sum(0,0) << "," << sigma_sum(0,1) << "," << sigma_sum(0,2) << endl
                  << sigma_sum(1,0) << "," << sigma_sum(1,1) << "," << sigma_sum(1,2) << endl
                  << sigma_sum(2,0) << "," << sigma_sum(2,1) << "," << sigma_sum(2,2) << endl;
  cout << "similarity: " << result(0) << endl;*/
  
  return result(0);
  
  /**** TODO: los umbrales / las varianzas son interdependientes (mas que nada: hue depende del resto). pensar tambien en thresholds con signo (porque puedo aceptar cosas mas claras pero no mas oscuras) ****/
}

void fd::Segment::mix(const Segment& other, float alpha)
{
  float alpha1, alpha2;
  if (alpha == -1) {
    alpha1 = (float)mass / (float)(mass + other.mass);
    alpha2 = (float)other.mass / (float)(mass + other.mass);
    mass += other.mass;
  }
  else {
    alpha1 = alpha;
    alpha2 = 1 - alpha;
    // mass not updated, this version is used for long-tearm learning
  }
  
  mu = (mu * alpha1 + other.mu * alpha2);
  rgb2hsv(mu, mu_hsv);
  sigma = (sigma * alpha1 + other.sigma * alpha2);
  sigma_hsv = (sigma_hsv * alpha1 + other.sigma_hsv * alpha2);
}

std::ostream& operator<<(std::ostream& out, const fd::Segment& s) {
  out << "mu " << s.mu(0) << " " << s.mu(1) << " " << s.mu(2) << std::endl;
  out << "mu hsv " << s.mu_hsv(0) << " " << s.mu_hsv(1) << " " << s.mu_hsv(2) << std::endl;
  out << "mass " << s.mass << std::endl;
  cv::Vec3f eigenvalues;
  cv::Matx33f eigenvectors;
  cv::eigen(s.sigma_hsv, eigenvalues, eigenvectors);
  out << "std2 " << eigenvalues(0) << " " << eigenvalues(1) << " " << eigenvalues(2) << std::endl;
  out << s.sigma_hsv(0,0) << " " << s.sigma_hsv(1,1) << " " << s.sigma_hsv(2,2);
  return out;
}

