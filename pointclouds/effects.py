from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.typing as npt
import open3d as o3d
from typing import Union, Tuple, List
from dataclasses import dataclass
import torch


from learning3d.learning3d.models import PointNet
from learning3d.learning3d.models import Classifier
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




class PointCloudClassifier:
    """Classify objects in point clouds using Learning3D PointNet"""
    
    def __init__(self):
        device = "cpu"
        model_path = "/home/georgefkd/programming/learning3d/learning3d/pretrained/exp_classifier/models/best_model.t7"
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Device is: {self.device}")
        ptnet = PointNet(emb_dims=1024, use_bn=True,input_shape="bcn")
        self.model = Classifier(feature_model=ptnet, num_classes=40)  # ModelNet40 classes
        
        # Load pretrained weights
        if model_path and os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ Loaded model from {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # ModelNet40 class names
        self.class_names = [
            'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl',
            'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser',
            'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop',
            'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant',
            'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table',
            'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
        ]
    
    def preprocess_pointcloud(self, pcd: o3d.geometry.PointCloud, num_points=2048):
        """Convert Open3D point cloud to PointNet input"""
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            return None
        print("Points are: ",len(pcd.points))
        # Randomly sample or downsample to fixed number of points
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        elif len(points) < num_points:
            # Upsample by repeating points
            indices = np.random.choice(len(points), num_points, replace=True)
            points = points[indices]
        
        # Normalize to unit sphere
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
        
        # Convert to torch tensor (shape: 1, 3, N)
        points_tensor = torch.from_numpy(points.T).float().unsqueeze(0)
        
        return points_tensor
    
    def classify(self, pcd: o3d.geometry.PointCloud):
        """
        Classify a point cloud
        
        Returns:
            (class_name, confidence)
        """
        
        points_tensor = self.preprocess_pointcloud(pcd)
        print("Tensor shape is: ",points_tensor.shape)
        if points_tensor is None:
            return None, 0.0
        
        points_tensor = points_tensor.to(self.device)
        
        with torch.no_grad():
            #throws error [1,3,1024]
            print("Point Shape is:" ,points_tensor.shape)
            logits = self.model(points_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
        
        class_idx = pred_class.item()
        confidence_val = confidence.item()
        class_name = self.class_names[class_idx]
        
        return class_name, confidence_val
    
    def classify_segments(self, pcd: o3d.geometry.PointCloud, bboxes):
        """
        Classify multiple segmented regions in point cloud
        
        Args:
            pcd: Full point cloud
            bboxes: List of bounding boxes [(x, y, w, h), ...]
        
        Returns:
            List of (class_name, confidence) for each bbox
        """
        results = []
        points = np.asarray(pcd.points)
        
        for (x, y, w, h) in bboxes:
            # Extract points within bbox (rough 2D projection)
            mask = (points[:, 0] >= x) & (points[:, 0] <= x + w) & \
                   (points[:, 1] >= y) & (points[:, 1] <= y + h)
            
            if np.sum(mask) < 10:  # Too few points
                results.append(("unknown", 0.0))
                continue
            
            # Create sub-point cloud
            sub_pcd = o3d.geometry.PointCloud()
            sub_pcd.points = o3d.utility.Vector3dVector(points[mask])
            
            # Classify
            class_name, confidence = self.classify(sub_pcd)
            results.append((class_name, confidence))
        
        return results


class PointCloudObjectDetectionEffect(EffectProducer):
    """Effect producer using Learning3D for classification"""
    
    def __init__(self):
        self.classifier = PointCloudClassifier()
    
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
