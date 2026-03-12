from abc import ABCMeta, abstractmethod
import numpy as np
import pyrealsense2 as rs
from typing import Tuple
import open3d as o3d

class Intrinsics:
    def __init__(self, width, height, fx, fy, ppx, ppy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy


class CameraReader(metaclass=ABCMeta):
    @abstractmethod
    def read_frame(self)->Tuple[bool,np.ndarray,np.ndarray]:
        pass
    @abstractmethod
    def get_intrinsics(self)->Intrinsics:
        pass
    @abstractmethod
    def get_depth_scale(self)->float:
        pass
    @abstractmethod
    def stop(self)->None:
        pass
    @abstractmethod
    def get_width_height(self)->Tuple[int,int]:
        pass

class StreamingCamera(CameraReader):
    pass


class FakeCamera(CameraReader):
    """Fake camera that reads from a PLY file"""
    
    def __init__(self, ply_path: str, width: int = 640, height: int = 480):
        self.ply_path = ply_path
        self.width = width
        self.height = height
        
        # Load point cloud
        self.pcd = o3d.io.read_point_cloud(ply_path)
        
        if len(self.pcd.points) == 0:
            raise ValueError(f"Empty point cloud: {ply_path}")
        
        # intrinsics from previous runs of the camera
        self._read_intrinsics_from_file("intrinsics.json")
        # Convert point cloud to depth and color images once
        self._generate_depth_color_images()
        
        print(f"✓ FakeCamera loaded from {ply_path}")
        print(f"  Points: {len(self.pcd.points)}")
        print(f"  Resolution: {width}x{height}")
    
    def _read_intrinsics_from_file(self,path):
        import json

        with open(path, 'r') as f:
            data = json.load(f)
        
        self.width = data["width"]
        self.height = data["height"]
        self.fx = float(data['fx'])
        self.fy = float(data['fy'])
        self.ppx = float(data['ppx'])
        self.ppy = float(data['ppy'])
    def _generate_depth_color_images(self):
        """Generate depth and color images from point cloud"""
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors) if self.pcd.has_colors() else np.ones((len(points), 3))
        
        # Initialize images
        self.depth_image = np.zeros((self.height, self.width), dtype=np.uint16)
        self.color_image = (colors * 255).astype(np.uint8)
        
        # Project 3D points to 2D
        print(f"The points read: {len(points)}")
        print(f"Width is:{self.width},Height is: {self.height}")
        for i in range(len(points)):
            x, y, z = points[i]
            # Project to image coordinates
            u = int(self.fx * x / z + self.ppx)
            v = int(self.fy * y / z + self.ppy)
            # Check bounds
            if 0 <= u < self.width and 0 <= v < self.height:
                # Convert depth to uint16 (millimeters)
                depth_mm = int(z * 1000)
                self.depth_image[v, u] = depth_mm
                # self.color_image[v, u] = (colors[i] * 255).astype(np.uint8)

    
    def read_frame(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Return the same depth and color images each time"""
        # Return copies to avoid external modifications
        return True, self.depth_image.copy(), self.color_image.copy()
    
    def get_intrinsics(self):
        """Return fake intrinsics object"""
        
        return Intrinsics(self.width, self.height, self.fx, self.fy, self.ppx, self.ppy)
        
    
    def get_depth_scale(self) -> float:
        """Return depth scale (meters per unit)"""
        # Depth is in millimeters, so scale is 0.001
        return 0.001
    
    def stop(self) -> None:
        """Nothing to stop for fake camera"""
        print("✓ FakeCamera stopped")
    
    def get_width_height(self) -> Tuple[int, int]:
        """Return image dimensions"""
        return self.width, self.height

class RealsenseCamera(CameraReader):

    @classmethod
    def check_if_present(cls):
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("No RealSense camera connected")
            return False
        return True
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)

        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        self.intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

    def stop(self):
        self.pipeline.stop()

    def get_depth_scale(self):
        return self.depth_scale

    def read_frame(self):
        """Capture depth and color frames"""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            return False,np.empty((0,3)),np.empty((0,3)) 

        return True,np.asanyarray(depth_frame.get_data()), np.asanyarray(
            color_frame.get_data()
        )

    def get_intrinsics(self):
        import json
        data = {
            'width': self.intrinsics.width,
            'height': self.intrinsics.height,
            'fx': self.intrinsics.fx,
            'fy': self.intrinsics.fy,
            'ppx': self.intrinsics.ppx,
            'ppy': self.intrinsics.ppy
        }
        
        with open('intrinsics.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved intrinsics to intrinsics.json")
        return self.intrinsics
    def get_width_height(self):
        return self.width,self.height


