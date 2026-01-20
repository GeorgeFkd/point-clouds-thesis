#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input.pcd|input.ply>" << std::endl;
    return 1;
  }

  std::string filename = argv[1];
  std::cout << "Rendering pointcloud file: " << filename << "\n";
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  // Load point cloud based on file extension
  std::string ext = filename.substr(filename.find_last_of(".") + 1);

  if (ext == "pcd") {
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(filename, *cloud) == -1) {
      std::cerr << "Error: Could not load PCD file: " << filename << std::endl;
      return 1;
    }
  } else if (ext == "ply") {
    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(filename, *cloud) == -1) {
      std::cerr << "Error: Could not load PLY file: " << filename << std::endl;
      return 1;
    }
  } else {
    std::cerr << "Error: Unsupported file format. Use .pcd or .ply"
              << std::endl;
    return 1;
  }

  std::cout << "Loaded " << cloud->width * cloud->height << " points"
            << std::endl;
  std::cout << "Press 'q' to close the viewer" << std::endl;

  // Create viewer
  pcl::visualization::CloudViewer viewer("Point Cloud Viewer");
  viewer.showCloud(cloud);

  // Keep viewer open until user closes it
  while (!viewer.wasStopped()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  return 0;
}
