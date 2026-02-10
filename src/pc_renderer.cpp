#include <chrono>
#include <iostream>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <thread>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input.pcd|input.ply>" << std::endl;
    return 1;
  }

  std::string filename = argv[1];
  std::cout << "Loading: " << filename << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  std::string ext = filename.substr(filename.find_last_of(".") + 1);
  if (ext == "pcd") {
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
      std::cerr << "Error: Could not load PCD file" << std::endl;
      return 1;
    }
  } else if (ext == "ply") {
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(filename, *cloud) == -1) {
      std::cerr << "Error: Could not load PLY file" << std::endl;
      return 1;
    }
  } else {
    std::cerr << "Error: Unsupported format. Use .pcd or .ply" << std::endl;
    return 1;
  }

  std::cout << "Loaded " << cloud->points.size() << " points" << std::endl;

  if (cloud->empty()) {
    std::cerr << "Error: Point cloud is empty!" << std::endl;
    return 1;
  }

  if (cloud->isOrganized()) {
    std::cout << "Organized: " << cloud->width << "x" << cloud->height
              << std::endl;
  } else {
    std::cout << "Unorganized: " << cloud->points.size() << " points"
              << std::endl;
  }

  std::cout << "Origin X: " << cloud->sensor_origin_.x() << "\n";
  std::cout << "Origin Y: " << cloud->sensor_origin_.y() << "\n";
  std::cout << "Origin Z: " << cloud->sensor_origin_.z() << "\n";
  std::cout << "Orientation: " << cloud->sensor_orientation_ << "\n";
  //
  // pcl::visualization::PCLVisualizer::Ptr viewer(
  //     new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
  // viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");
  // viewer->setPointCloudRenderingProperties(
  //     pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
  //
  // // Get sensor pose
  // Eigen::Vector4f sensor_origin = cloud->sensor_origin_;
  // Eigen::Quaternionf sensor_orientation = cloud->sensor_orientation_;
  //
  // std::cout << "Sensor origin: " << sensor_origin.transpose() << std::endl;
  // std::cout << "Sensor orientation: " << sensor_orientation.w() << ", "
  //           << sensor_orientation.x() << ", " << sensor_orientation.y() << ",
  //           "
  //           << sensor_orientation.z() << std::endl;
  //
  // // Check if sensor pose is set
  // if (sensor_origin.norm() == 0) {
  //   std::cout << "Warning: No sensor origin set, using default view"
  //             << std::endl;
  // } else {
  //   // Set camera to sensor viewpoint
  //   Eigen::Matrix3f rotation = sensor_orientation.toRotationMatrix();
  //
  //   // Camera position
  //   float cam_x = sensor_origin[0];
  //   float cam_y = sensor_origin[1];
  //   float cam_z = sensor_origin[2];
  //
  //   // View direction (forward = -Z axis in camera frame)
  //   Eigen::Vector3f view_dir = rotation * Eigen::Vector3f(0, 0, -1);
  //
  //   // Up direction (Y axis in camera frame)
  //   Eigen::Vector3f up_dir = rotation * Eigen::Vector3f(0, -1, 0);
  //
  //   viewer->setCameraPosition(cam_x, cam_y, cam_z, // camera position
  //                             cam_x + view_dir[0], // look at point
  //                             cam_y + view_dir[1], cam_z + view_dir[2],
  //                             up_dir[0], up_dir[1], up_dir[2] // up direction
  //   );
  //
  //   std::cout << "Camera set to sensor viewpoint" << std::endl;
  // }
  // // viewer->resetCamera();
  //
  // std::cout << "Viewer controls: q=quit, r=reset camera, +/-=change point
  // size"
  //           << std::endl;
  //
  // while (!viewer->wasStopped()) {
  //   viewer->spinOnce(100);
  // }

  return 0;
}
