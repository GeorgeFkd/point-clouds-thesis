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


class PointCloudModel(metaclass=ABCMeta):
    @abstractmethod
    def detect(self, pointcloud: o3d.geometry.PointCloud):
        pass

    @abstractmethod
    def __init__(self):
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

        # NOTE: This code can be commented out to enable general object detection
        #
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




class PointPillarsDetector(PointCloudModel):
    """3D Object Detection using PointPillars (KITTI dataset)"""

    def __init__(self):
        print("Initializing PointPillars detector...")

        # KITTI class names
        self.class_names = ["Car", "Pedestrian", "Cyclist"]
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
