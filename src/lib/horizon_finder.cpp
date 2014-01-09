#include <vector>
#include "horizon_finder.h"
namespace fd = floordetection;

fd::HorizonFinder::HorizonFinder(ros::NodeHandle& ros_node)
{
  last_horizon = -1;
  ros_node.param("horizon_finder_subimages", subimages, 20);
}

int fd::HorizonFinder::find(const cv::Mat& blured)
{
  // subimg height, for the upper half of the image
  uint height = (int)(blured.rows / (float)(2 * subimages));

  // convert half of image to gray scale
  cv::Mat blured_gray;
  cv::Mat blured_half(blured, cv::Range(0, (int)(blured.rows * 0.5) + (height / 2) + 1), cv::Range::all());
  cv::cvtColor(blured_half, blured_gray, CV_RGB2GRAY);

  // Third apply a Sobel filter to obtain the Y derived image  
  cv::Mat sobel;
  cv::Sobel(blured_gray, sobel, -1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
  
  // Then apply the OTSU threshold to obtain a binary image
  cv::Mat otsu;
  cv::threshold(sobel, otsu, 0, 255, cv::THRESH_OTSU);
    
  // Then erode the binary image to reduce noise
  int erosion_size = 1;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size));
  cv::Mat dst;
  cv::erode(otsu, dst, kernel); 
    
  // Compute the amount of foregrounds pixels for each window.
  int maxforeground1 = 0;
  int index1 = 0;
  std::vector<int> foregrounds1(subimages);
  for(uint i = 0; i < subimages; i++){
    foregrounds1[i] = 0;
    cv::Mat subimg = dst(cv::Range(i * height, (i + 1) * height), cv::Range::all());
    uchar* ptrData = subimg.ptr<uchar>(0);
    for(int j = 0; j < subimg.rows * subimg.cols; j++, ++ptrData) {
      if (*ptrData == 255) foregrounds1[i]++;
    }
    
    if (foregrounds1[i] > maxforeground1) {
       maxforeground1 = foregrounds1[i];
       index1 = i;
    }
  }

  // The same as before, but for each window slip half a height to overlap them
  int maxforeground2 = 0;
  int index2 = 0;
  std::vector<int> foregrounds2(subimages);
  for(unsigned i = 0; i < subimages; i++){
    foregrounds2[i] = 0;
    cv::Mat subimg = dst(cv::Range(i * height + (height / 2), (i + 1) * height + (height / 2)), cv::Range::all());
    uchar* ptrData = subimg.ptr<uchar>(0);
    for(int j = 0; j < subimg.rows * subimg.cols; j++, ++ptrData) {
      if (*ptrData == 255) foregrounds2[i]++;
    }
    if (foregrounds2[i] > maxforeground2) {
       maxforeground2 = foregrounds2[i];
       index2 = i;
    }
  }

  // The max horizon between windows and overlap windows is the winner
  int horizon;
  if (maxforeground1 > maxforeground2) horizon = index1 * height;
  else horizon = index2 * height + (height / 2);

  float alpha = 0.5;
  if (last_horizon == -1) last_horizon = horizon;
  else last_horizon = (int)roundf(alpha * horizon + (1 - alpha) * last_horizon);

  return last_horizon;
}

