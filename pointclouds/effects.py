from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.typing as npt
import open3d as o3d
from typing import Union, Tuple, List, Callable
from dataclasses import dataclass
from models import ObjectDetector2D
import cv2
import os


@dataclass
class ColorChange:
    """Change color of points in a region"""

    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    color: Tuple[int, int, int]  # (r, g, b)


@dataclass
class Text3DAdd:
    """Add 3D text to the scene"""

    text: str
    position: Tuple[float, float, float]  # (x, y, z)
    size: float
    color: Tuple[int, int, int]  # (r, g, b)

@dataclass
class Box3DAdd:
    """3D bounding box for object detection"""
    center: Tuple[float, float, float]  # (x, y, z)
    extent: Tuple[float, float, float]  # (width, height, depth)
    yaw: float                          # Rotation around Z-axis in radians


@dataclass
class CubeAdd:
    """Add a cube to the scene"""

    center: Tuple[float, float, float]  # (x, y, z)
    size: float
    color: Tuple[int, int, int]  # (r, g, b)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (rx, ry, rz) in radians


@dataclass
class SideEffect:
    """Does nothing on the pointcloud itself"""

    callback: Callable[[np.ndarray, np.ndarray, o3d.geometry.PointCloud], None]


@dataclass
class Text2DAdd:
    text: str
    position: Tuple[int, int]


@dataclass
class Rect2DAdd:
    p1: Tuple[int, int]
    p2: Tuple[int, int]


Effect = Union[ColorChange, Text3DAdd, CubeAdd, SideEffect, Text2DAdd, Rect2DAdd,Box3DAdd]


class EffectProducer(metaclass=ABCMeta):
    @abstractmethod
    def apply(
        self, depth, color, pointcloud: o3d.geometry.PointCloud = None
    ) -> List[Effect]:
        pass


class Processor2D:
    @classmethod
    def gaussian_blur(cls, image, x1, x2, y1, y2, blur_amount=51):
        img = image.copy()

        region = img[y1:y2, x1:x2]
        if region.size > 0:
            region_blurred = cv2.GaussianBlur(region, (blur_amount, blur_amount), 0)
            img[y1:y2, x1:x2] = region_blurred
            return img
        return img


class Detections3DEffect(EffectProducer):
    def apply(self,depth,color,pointcloud:o3d.geometry.PointCloud=None) -> List[Effect]:
        effects = []
        effect1 = Box3DAdd(
        center=(0.5, 0.0, 1.0),
        extent=(0.3, 0.3, 0.5),
        yaw=0.0
        )

        effect2 = Box3DAdd(
            center=(-0.5, 0.3, 1.0),
            extent=(0.4, 0.4, 0.6),
            yaw=0.785  # 45 degrees
        )

        effects.append(effect1)
        effects.append(effect2)

        return effects

class BlurDetections2DEffect(EffectProducer):
    """Blur detected objects in color image"""

    def __init__(self, target_classes=None):
        self.detector = ObjectDetector2D(["face"])
        self.target_classes = target_classes if target_classes else ["face"]

    def apply(
        self, depth, color, pointcloud: o3d.geometry.PointCloud = None
    ) -> List[Effect]:
        effects = []

        detections = self.detector.detect(depth, color, self.target_classes)

        if len(detections) > 0:

            def blur_callback(depth, color, pointcloud):
                for text_label, x1, x2, y1, y2 in detections:
                    print(f"Blurring {text_label}")
                    color[:] = Processor2D.gaussian_blur(color, x1, x2, y1, y2)

            effects.append(SideEffect(callback=blur_callback))

        return effects


class Detection2DEffect(EffectProducer):
    """Draw 2D bounding boxes on detected objects"""

    def __init__(self, target_classes=None):
        self.detector = ObjectDetector2D(["face"])
        self.target_classes = target_classes if target_classes else ["face"]

    def apply(
        self, depth, color, pointcloud: o3d.geometry.PointCloud = None
    ) -> List[Effect]:
        effects = []

        detections = self.detector.detect(depth, color, self.target_classes)

        for text_label, x1, x2, y1, y2 in detections:
            effects.append(Rect2DAdd(p1=(x1, y1), p2=(x2, y2)))

            effects.append(Text2DAdd(text=text_label, position=(x1, y1 - 5)))

        return effects


class FnCall(EffectProducer):
    """Effect producer that wraps a function call as a side effect"""

    def __init__(
        self, fn: Callable[[np.ndarray, np.ndarray, o3d.geometry.PointCloud], None]
    ):
        self.fn = fn

    def apply(
        self, depth, color, pointcloud: o3d.geometry.PointCloud = None
    ) -> List[Effect]:
        return [SideEffect(callback=self.fn)]


class EffectsManager:
    effects_producers: List[EffectProducer]

    def __init__(self):
        self.effects_producers = []

    def add_effect_producer(self, e: EffectProducer):
        self.effects_producers.append(e)

    def add_effect_producers(self, e: List[EffectProducer]):
        self.effects_producers.extend(e)

    def create_effects(
        self, depth, color, pointcloud: o3d.geometry.PointCloud = None
    ) -> List[Effect]:
        effects_on_pointcloud = []
        for e in self.effects_producers:
            result = e.apply(depth, color, pointcloud)
            effects_on_pointcloud.extend(result)
        return effects_on_pointcloud


# TODO: implement it properly
# 
# class PointCloudObjectDetectionEffect(EffectProducer):
#     """Effect producer using Learning3D for classification"""
#
#     def __init__(self):
#         self.classifier = Learning3dDetector()
#
#     def apply(
#         self, depth, color, pointcloud: o3d.geometry.PointCloud = None
#     ) -> List[Effect]:
#         effects = []
#
#         # Classify the whole point cloud
#         class_name, confidence = self.classifier.classify(pointcloud)
#
#         if class_name and confidence > 0.5:
#             # Get center of point cloud
#             points = np.asarray(pointcloud.points)
#             center = points.mean(axis=0)
#
#             # Add text label
#             effects.append(
#                 Text3DAdd(
#                     text=f"{class_name} ({confidence:.2f})",
#                     position=(center[0], center[1] + 0.2, center[2]),
#                     size=0.05,
#                     color=(0, 255, 0),
#                 )
#             )
#
#             # Add bounding cube
#             effects.append(
#                 CubeAdd(
#                     center=(center[0], center[1], center[2]),
#                     size=0.15,
#                     color=(0, 255, 0),
#                 )
#             )
#
#         return effects
