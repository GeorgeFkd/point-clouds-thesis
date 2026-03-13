from abc import ABCMeta, abstractmethod
import open3d as o3d
from typing import Tuple, List

# from learning3d.learning3d.models import PointNet,create_pointconv
# from learning3d.learning3d.models import Classifier
import os
import numpy as np
import cv2
import open3d.ml.torch as ml3d
import open3d.ml.utils


class PointCloudClassifier(metaclass=ABCMeta):
    @abstractmethod
    def classify(self, pointcloud: o3d.geometry.PointCloud):
        pass

    @abstractmethod
    def __init__(self):
        pass


class PointRCNN(PointCloudClassifier):
    pass


class ObjectDetector2D:
    def __init__(self, objects_to_detect):
        assert "face" in objects_to_detect
        if "face" in objects_to_detect:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        self.class_names = {
            0: "background",
            1: "aeroplane",
            2: "bicycle",
            3: "bird",
            4: "boat",
            5: "bottle",
            6: "bus",
            7: "car",
            8: "cat",
            9: "chair",
            10: "cow",
            11: "diningtable",
            12: "dog",
            13: "horse",
            14: "motorbike",
            15: "person",
            16: "pottedplant",
            17: "sheep",
            18: "sofa",
            19: "train",
            20: "tvmonitor",
        }
        self.net = cv2.dnn.readNetFromCaffe(
            "MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel"
        )

    def convert_detections(self, detection):
        pass

    def detect(self, depth, color, objects):
        results = []
        img = color.copy()
        if "face" in objects:
            text_label = "Face"
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for x, y, w, h in faces:
                margin = int(w * 0.2)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(img.shape[1], x + w + margin)
                y2 = min(img.shape[0], y + h + margin)
                results.append((text_label, x1, x2, y1, y2))

        # frame_resized = cv2.resize(img, (300, 300))
        # blob = cv2.dnn.blobFromImage(
        #     frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False
        # )
        # self.net.setInput(blob)
        # detections = self.net.forward()
        # cols = frame_resized.shape[1]
        # rows = frame_resized.shape[0]
        # for i in range(detections.shape[2]):
        #     confidence = detections[0, 0, i, 2]  # Confidence of prediction
        #     if confidence > 0.65:  # Filter prediction
        #         class_id = int(detections[0, 0, i, 1])  # Class label
        #
        #         # Object location
        #         x1 = int(detections[0, 0, i, 3] * cols)
        #         y1 = int(detections[0, 0, i, 4] * rows)
        #         x2 = int(detections[0, 0, i, 5] * cols)
        #         y2 = int(detections[0, 0, i, 6] * rows)
        #
        #         # Factor for scale to original size of frame
        #         heightFactor = color.shape[0] / 300.0
        #         widthFactor = color.shape[1] / 300.0
        #         # Scale object detection to frame
        #         x1 = int(widthFactor * x1)
        #         y1 = int(heightFactor * y1)
        #         x2 = int(widthFactor * x2)
        #         y2 = int(heightFactor * y2)
        #         if class_id in self.class_names:
        #             label = self.class_names[class_id] + ": " + str(confidence)
        #             print(label)
        #             results.append((label, x1, x2, y1, y2))
        #             print(label)  # print class and confidence
        return results


# class IMVoxelNetDetector(PointCloudClassifier):
#     """3D Object Detection using IMVoxelNet from MMDetection3D"""
#
#     def __init__(self):
#
#         # Hardcoded config and checkpoint paths
#         config_file = '/home/georgefkd/programming/mmdetection3d/configs/imvoxelnet/imvoxelnet_2xb4_sunrgbd-3d-10class.py'
#         checkpoint_file = './imvoxelnet_4x2_sunrgbd-3d-10class_20220809_184416-29ca7d2e.pth'
#
#         # Determine device
#         if torch.cuda.is_available():
#             device = 'cuda:0'
#         else:
#             device = 'cpu'
#
#         # Initialize model
#         self.model = init_model(config_file, checkpoint_file, device=device)
#
#         print(f"✓ IMVoxelNet initialized on {device}")
#
#     def classify(self, pointcloud: o3d.geometry.PointCloud):
#         """
#         Detect objects in point cloud
#
#         Returns:
#             MMDetection3D result object (raw format)
#         """
#         # Convert Open3D to MMDetection3D format
#         points = np.asarray(pointcloud.points)
#
#         if len(points) == 0:
#             return None
#
#         # Add intensity channel (use ones if no color)
#         if not pointcloud.has_colors():
#             intensity = np.ones((len(points), 1))
#         else:
#             colors = np.asarray(pointcloud.colors)
#             intensity = np.mean(colors, axis=1, keepdims=True)
#
#         points_with_intensity = np.hstack([points, intensity]).astype(np.float32)
#
#         # Create input dict
#         data = {'points': points_with_intensity}
#
#         # Run inference - returns MMDetection3D result
#         result = inference_detector(self.model, data)
#
#         return result


class PointPillarsDetector:
    """3D Object Detection using PointPillars (KITTI dataset)"""

    def __init__(self):
        print("Initializing PointPillars detector...")

        # KITTI class names
        self.class_names = ["Car", "Pedestrian", "Cyclist"]

        # Load PointPillars model
        # Check Open3D-ML for actual config path
        # try:
        #     # Option 1: Load from config
        #     cfg_file = "path/to/pointpillars_kitti.yml"  # Need to find actual path
        #     cfg = ml3d.utils.Config.load_from_file(cfg_file)
        #     self.model = ml3d.models.PointPillars(**cfg.model)
        #
        # except:
        # Option 2: Create with default parameters
        config_path = "./pointpillars.yml"
        cfg = open3d.ml.utils.Config.load_from_file(config_path)
        self.model = ml3d.models.PointPillars(**cfg.model)

        # Create pipeline
        self.pipeline = ml3d.pipelines.ObjectDetection(self.model)

        # Load checkpoint
        checkpoint_path = "./pointpillars_kitti_202012221652utc.pth"
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.pipeline.load_ckpt(checkpoint_path)
            print("✓ PointPillars loaded")
        else:
            print("⚠️  No pretrained weights - model not loaded!")

    def detect(self, pointcloud: o3d.geometry.PointCloud):
        """
        Detect objects in point cloud

        Returns:
            Dict with bounding boxes, scores, labels
        """
        points = np.asarray(pointcloud.points).astype(np.float32)

        # PointPillars expects [x, y, z, intensity]
        if not pointcloud.has_colors():
            intensity = np.ones((len(points), 1), dtype=np.float32)
        else:
            colors = np.asarray(pointcloud.colors)
            intensity = np.mean(colors, axis=1, keepdims=True).astype(np.float32)

        points_with_intensity = np.hstack([points, intensity])

        # Prepare data
        data = {
            "point": points_with_intensity,
            "feat": None,
            "calib": None,
            "bounding_boxes": np.array([]),  # Dummy for inference
        }

        # Run inference
        print("Running PointPillars detection...")
        results = self.pipeline.run_inference(data)

        return results


supported_models = ["pointnet", "pointconv"]


class Learning3dDetector:
    """Classify objects in point clouds using Learning3D PointNet"""

    def __init__(self):
        import argparse

        device = "cpu"

        parser = argparse.ArgumentParser(
            description="Point Cloud Application", exit_on_error=False
        )
        parser.add_argument(
            "--model",
            choices=supported_models,
            help="PointCloud Model for the classification",
            default="pointnet",
        )
        args, unknown = parser.parse_known_args()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Device is: {self.device}")
        if args.model == "pointnet":
            architecture = PointNet(emb_dims=1024, use_bn=True, input_shape="bcn")
            model_path = (
                "./learning3d/learning3d/pretrained/exp_classifier/models/best_model.t7"
            )
        elif args.model == "pointconv":
            assert False
            model_path = "./learning3d/learning3d/pretrained/exp_classifier/models/"
            PointConv = create_pointconv(classifier=True, pretrained="")
            architecture = None
        else:
            model_path = "/home/georgefkd/programming/learning3d/learning3d/pretrained/exp_classifier/models/best_model.t7"
            architecture = PointNet(emb_dims=1024, use_bn=True, input_shape="bcn")
        self.model = Classifier(
            feature_model=architecture, num_classes=40
        )  # ModelNet40 classes

        if model_path and os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ Loaded model from {model_path}")

        self.model.to(self.device)
        self.model.eval()

        # ModelNet40 class names
        self.class_names = [
            "airplane",
            "bathtub",
            "bed",
            "bench",
            "bookshelf",
            "bottle",
            "bowl",
            "car",
            "chair",
            "cone",
            "cup",
            "curtain",
            "desk",
            "door",
            "dresser",
            "flower_pot",
            "glass_box",
            "guitar",
            "keyboard",
            "lamp",
            "laptop",
            "mantel",
            "monitor",
            "night_stand",
            "person",
            "piano",
            "plant",
            "radio",
            "range_hood",
            "sink",
            "sofa",
            "stairs",
            "stool",
            "table",
            "tent",
            "toilet",
            "tv_stand",
            "vase",
            "wardrobe",
            "xbox",
        ]

    def preprocess_pointcloud(self, pcd: o3d.geometry.PointCloud, num_points=2048):
        """Convert Open3D point cloud to PointNet input"""
        points = np.asarray(pcd.points)

        if len(points) == 0:
            return None
        print("Points are: ", len(pcd.points))
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
        print("Tensor shape is: ", points_tensor.shape)
        if points_tensor is None:
            return None, 0.0

        points_tensor = points_tensor.to(self.device)

        with torch.no_grad():
            # throws error [1,3,1024]
            print("Point Shape is:", points_tensor.shape)
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

        for x, y, w, h in bboxes:
            # Extract points within bbox (rough 2D projection)
            mask = (
                (points[:, 0] >= x)
                & (points[:, 0] <= x + w)
                & (points[:, 1] >= y)
                & (points[:, 1] <= y + h)
            )

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
