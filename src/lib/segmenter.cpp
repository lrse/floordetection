#include <opencv2/opencv.hpp>
#include <vector>
#include "segmenter.h"
#include "disjoint_set.h"
namespace fd = floordetection;

/* edge */
typedef struct {
  float w;
  int a, b;
} edge;

bool operator<(const edge &a, const edge &b) {
  return a.w < b.w;
}

/* segmenter */

fd::Segmenter::Segmenter(ros::NodeHandle& _ros_node) :
  ros_node(_ros_node)
{
  ros_node.param("segmenter_threshold", c_threshold, 0.5);
  ros_node.param("min_segment", min_segment, 500);
}

void fd::Segmenter::save_parameters(void)
{
  ros_node.setParam("segmenter_threshold", c_threshold);
  ros_node.setParam("min_segment", min_segment);
}

void fd::Segmenter::segment(const cv::Mat& input, cv::Mat& output)
{
  int width = input.size().width;
  int height = input.size().height;
  
  /* build graph */
  std::vector<edge> edges(width*height*4);
  int num_edges = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x < width-1) {
        edges[num_edges].a = y * width + x;
        edges[num_edges].b = y * width + (x+1);
        edges[num_edges].w = cv::norm(input.at<cv::Vec3b>(y,x), input.at<cv::Vec3b>(y,x+1));
        num_edges++;
      }
      if (y < height-1) {
        edges[num_edges].a = y * width + x;
        edges[num_edges].b = (y+1) * width + x;
        edges[num_edges].w = cv::norm(input.at<cv::Vec3b>(y,x), input.at<cv::Vec3b>(y+1,x));
        num_edges++;
      }
      if ((x < width-1) && (y < height-1)) {
        edges[num_edges].a = y * width + x;
        edges[num_edges].b = (y+1) * width + (x+1);
        edges[num_edges].w = cv::norm(input.at<cv::Vec3b>(y,x), input.at<cv::Vec3b>(y+1,x+1));
        num_edges++;
      }
      if ((x < width-1) && (y > 0)) {
        edges[num_edges].a = y * width + x;
        edges[num_edges].b = (y-1) * width + (x+1);
        edges[num_edges].w = cv::norm(input.at<cv::Vec3b>(y,x), input.at<cv::Vec3b>(y-1,x+1));
        num_edges++;
      }
    }
  }
  
  /* segment graph */
  std::sort(edges.begin(), edges.end()); // sort edges by weight

  int num_vertices = width * height;  
  universe u(num_vertices); // make a disjoint-set forest
  std::vector<float> threshold(num_vertices, c_threshold); // // init thresholds = c / 1

  // for each edge, in non-decreasing weight order...
  for (int i = 0; i < num_edges; i++) {
    edge* pedge = &edges[i];
    
    // components conected by this edge
    int a = u.find(pedge->a);
    int b = u.find(pedge->b);
    if (a != b) {
      if ((pedge->w <= threshold[a]) && (pedge->w <= threshold[b])) {
        u.join(a, b);
        a = u.find(a);
        threshold[a] = pedge->w + (c_threshold / u.size(a)); // w + threshold
      }
    }
  }

  // post process small components
  for (int i = 0; i < num_edges; i++) {
    int a = u.find(edges[i].a);
    int b = u.find(edges[i].b);
    if ((a != b) && ((u.size(a) < min_segment) || (u.size(b) < min_segment)))
      u.join(a, b);
  }

  // create labels
  output.create(input.rows, input.cols, CV_32S);
  int* output_ptr = output.ptr<int>(0);  
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++, output_ptr++) {
      int comp = u.find((y * width) + x);     
      u.get_elt(comp)->total_pixels++;
      *output_ptr = comp;
    }
  }    
}
