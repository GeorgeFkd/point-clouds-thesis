from abc import ABCMeta, abstractmethod
import numpy as np
import pyrealsense2 as rs
from typing import Tuple

class CameraReader(metaclass=ABCMeta):
    @abstractmethod
    def read_frame(self)->Tuple[bool,np.ndarray,np.ndarray]:
        pass
    @abstractmethod
    def get_intrinsics(self):
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

class RealsenseCamera(CameraReader):
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
        return self.intrinsics
    def get_width_height(self):
        return self.width,self.height


