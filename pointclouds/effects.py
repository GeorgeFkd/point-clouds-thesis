from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.typing as npt
import open3d as o3d
from typing import Union, Tuple, List
from dataclasses import dataclass
from models import Learning3dDetector
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
class CubeAdd:
    """Add a cube to the scene"""

    center: Tuple[float, float, float]  # (x, y, z)
    size: float
    color: Tuple[int, int, int]  # (r, g, b)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (rx, ry, rz) in radians


@dataclass
class SideEffect:
    """Does nothing on the pointcloud itself"""


Effect = Union[ColorChange, Text3DAdd, CubeAdd,SideEffect]


class EffectProducer(metaclass=ABCMeta):
    @abstractmethod
    def apply(self, pointcloud: o3d.geometry.PointCloud) -> List[Effect]:
        pass


class FaceAnnotator(EffectProducer):
    def apply(self, pointcloud: o3d.geometry.PointCloud) -> List[Effect]:
        effects = []

        # Add one text
        effects.append(
            Text3DAdd(
                text="Face", position=(0.5, 0.5, 1.0), size=0.55, color=(255, 0, 0)
            )
        )

        # Add one cube
        effects.append(CubeAdd(center=(1.5, 0.5, 1.5), size=0.55, color=(0, 255, 0)))

        return effects


class EffectsManager:
    effects_producers: List[EffectProducer]

    def __init__(self):
        self.effects_producers = []

    def add_effect_producer(self, e: EffectProducer):
        self.effects_producers.append(e)

    def create_effects(self, pointcloud: o3d.geometry.PointCloud) -> List[Effect]:
        effects_on_pointcloud = []
        for e in self.effects_producers:
            result = e.apply(pointcloud)
            effects_on_pointcloud.extend(result)

        return effects_on_pointcloud





class MMDet3DObjectDetectionEffect(EffectProducer):
    """Effect producer using MMDetection3D for 3D object detection"""
    
    def __init__(self):
        pass
        # self.detector = IMVoxelNetDetector()
    
    def apply(self, pointcloud: o3d.geometry.PointCloud) -> List[Effect]:
        effects = []
        
        # Detect objects in point cloud
        result = self.detector.classify(pointcloud)
        
        if result is None:
            return effects
        
        # Extract predictions
        pred_instances = result.pred_instances_3d
        boxes_3d = pred_instances.bboxes_3d.tensor.cpu().numpy()
        scores_3d = pred_instances.scores_3d.cpu().numpy()
        labels_3d = pred_instances.labels_3d.cpu().numpy()
        
        # Class names
        class_names = [
            'bed', 'table', 'sofa', 'chair', 'toilet',
            'desk', 'dresser', 'night_stand', 'bookshelf', 'bathtub'
        ]
        
        # Create effects for each detection
        for i in range(len(boxes_3d)):
            x, y, z, w, h, d, yaw = boxes_3d[i]
            confidence = scores_3d[i]
            label_id = int(labels_3d[i])
            
            if confidence < 0.3:  # Confidence threshold
                continue
            
            class_name = class_names[label_id] if label_id < len(class_names) else 'unknown'
            
            # Add bounding cube
            effects.append(CubeAdd(
                center=(float(x), float(y), float(z)),
                size=float(max(w, h, d)),  # Use max dimension for cube size
                color=(0, 255, 0),
                rotation=(0.0, 0.0, float(yaw))
            ))
            
            # Add text label
            effects.append(Text3DAdd(
                text=f"{class_name} {confidence:.2f}",
                position=(float(x), float(y) + float(h)/2 + 0.1, float(z)),
                size=0.05,
                color=(255, 255, 0)
            ))
        
        return effects


class PointCloudObjectDetectionEffect(EffectProducer):
    """Effect producer using Learning3D for classification"""
    
    def __init__(self):
        self.classifier = Learning3dDetector()
    
    def apply(self, pointcloud: o3d.geometry.PointCloud) -> List[Effect]:
        effects = []
        
        # Classify the whole point cloud
        class_name, confidence = self.classifier.classify(pointcloud)
        
        if class_name and confidence > 0.5:
            # Get center of point cloud
            points = np.asarray(pointcloud.points)
            center = points.mean(axis=0)
            
            # Add text label
            effects.append(Text3DAdd(
                text=f"{class_name} ({confidence:.2f})",
                position=(center[0], center[1] + 0.2, center[2]),
                size=0.05,
                color=(0, 255, 0)
            ))
            
            # Add bounding cube
            effects.append(CubeAdd(
                center=(center[0], center[1], center[2]),
                size=0.15,
                color=(0, 255, 0)
            ))
        
        return effects
