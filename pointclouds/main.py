#!/usr/bin/env python3
"""
Face Detection 2D → Point Cloud 3D with PLY Export (PyVista)
IMPORTANT: Exports point clouds with BLURRED FACES for privacy
"""

from effects import (
    EffectsManager,
    FaceAnnotator,
    Effect,
    ColorChange,
    CubeAdd,
    MMDet3DObjectDetectionEffect,
    PointCloudObjectDetectionEffect,
    Text3DAdd,
    SideEffect,
)
import subprocess
from models import Learning3dDetector
import pyrealsense2 as rs
import numpy as np
import cv2
import random
from pathlib import Path
from datetime import datetime
import os
import open3d as o3d
from camera import RealsenseCamera,FakeCamera

use_realsense_camera = RealsenseCamera.check_if_present()
display_session_type = os.getenv("XDG_SESSION_TYPE")
enable_pointclouds_rendering =  display_session_type != "wayland"

# if display_session_type == "wayland":
#     print("Wayland session is currently not supported")
#     print("Use sudo startx and try to run again")
#     exit(1)
# elif display_session_type == "x11":
#     print("x11 session is supported(open3d point cloud rendering)")
# else:
#     print("unrecognised session type")



class ObjectDetector:
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
        if "face" in objects:
            text_label = "Face"
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for x, y, w, h in faces:
                margin = int(w * 0.2)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(color.shape[1], x + w + margin)
                y2 = min(color.shape[0], y + h + margin)
                results.append((text_label, x1, x2, y1, y2))

        frame_resized = cv2.resize(color, (300, 300))
        blob = cv2.dnn.blobFromImage(
            frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        cols = frame_resized.shape[1]
        rows = frame_resized.shape[0]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Confidence of prediction
            if confidence > 0.65:  # Filter prediction
                class_id = int(detections[0, 0, i, 1])  # Class label

                # Object location
                x1 = int(detections[0, 0, i, 3] * cols)
                y1 = int(detections[0, 0, i, 4] * rows)
                x2 = int(detections[0, 0, i, 5] * cols)
                y2 = int(detections[0, 0, i, 6] * rows)

                # Factor for scale to original size of frame
                heightFactor = color.shape[0] / 300.0
                widthFactor = color.shape[1] / 300.0
                # Scale object detection to frame
                x1 = int(widthFactor * x1)
                y1 = int(heightFactor * y1)
                x2 = int(widthFactor * x2)
                y2 = int(heightFactor * y2)
                if class_id in self.class_names:
                    label = self.class_names[class_id] + ": " + str(confidence)
                    print(label)
                    results.append((label, x1, x2, y1, y2))
                    print(label)  # print class and confidence
        return results


class DataExporter:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_jpeg(self, image, filename):
        """Save 2D image as JPEG"""
        filepath = self.output_dir / filename
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filepath

    def save_ply(
        self, cloud: o3d.geometry.PointCloud, filename: str, faces_count=0, binary=False
    ):
        """Save point cloud to PLY file with proper RGB colors"""
        filepath = self.output_dir / filename

        points = cloud.points
        colors = cloud.colors

        num_points = len(points)

        if binary:
            # Binary PLY (smaller file size)
            import struct

            with open(filepath, "wb") as f:
                # ASCII header
                header = f"ply\n"
                header += f"format binary_little_endian 1.0\n"
                header += f"comment Organized point cloud with blurred faces\n"
                header += f"comment Faces blurred: {faces_count}\n"
                header += f"element vertex {num_points}\n"
                header += f"property float x\n"
                header += f"property float y\n"
                header += f"property float z\n"
                header += f"property uchar red\n"
                header += f"property uchar green\n"
                header += f"property uchar blue\n"
                header += f"end_header\n"

                f.write(header.encode("ascii"))

                # Binary data
                for i in range(num_points):
                    x, y, z = points[i]
                    r, g, b = (colors[i] * 255).astype(np.uint8)
                    # Pack: 3 floats + 3 unsigned chars
                    data = struct.pack("fffBBB", x, y, z, int(r), int(g), int(b))
                    f.write(data)
        else:
            # ASCII PLY (human-readable, larger file)
            with open(filepath, "w") as f:
                # Header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"comment Organized point cloud with blurred faces\n")
                f.write(f"comment Faces blurred: {faces_count}\n")
                f.write(f"comment Created with RealSense face detection\n")
                f.write(f"element vertex {num_points}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")

                # Data - write each point with its RGB color
                for i in range(num_points):
                    x, y, z = points[i]
                    r, g, b = (colors[i] * 255).astype(np.uint8)
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

        return filepath


class Processor2D:
    @classmethod
    def gaussian_blur(cls, image, x1, x2, y1, y2, blur_amount=51):
        img = image.copy()

        region = img[y1:y2, x1:x2]
        if region.size > 0:
            region_blurred = cv2.GaussianBlur(region, (blur_amount, blur_amount), 0)
            img[y1:y2, x1:x2] = region_blurred
            return img

        assert False



class EventBus:
    def __init__(self):
        self._key_handlers = {}
        self._frame_handlers = []

    def on_key(self, key, callback):
        if key not in self._key_handlers:
            self._key_handlers[key] = []
        self._key_handlers[key].append(callback)

    def on_frame(self, callback):
        self._frame_handlers.append(callback)

    def key_pressed(self, key, *args, **kwargs):
        if key in self._key_handlers:
            for callback in self._key_handlers[key]:
                callback(*args, **kwargs)

    def new_frame(self, *args, **kwargs):
        for callback in self._frame_handlers:
            callback(*args, **kwargs)


class Application:
    def __init__(self):
        self.camera = RealsenseCamera() if use_realsense_camera else FakeCamera("./example-faces.ply")
        self.detector = ObjectDetector(["face"])
        self.data_exporter = DataExporter()
        self.effects_mgr = EffectsManager()
        self.bus = EventBus()
        intr = self.camera.get_intrinsics()

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
            )
        if enable_pointclouds_rendering:
            self.o3dvis = o3d.visualization.Visualizer()
            self.o3dvis.create_window(
                window_name="PC Visualiser", width=640, height=480
            )
            self.o3dvis.get_render_option().point_size = 2
            
            self.init_3d_viewer()
    

    def test_classification(self):
        print("Testing classification")
        pointcloud = o3d.io.read_point_cloud("example-faces.ply")
        self.effects_mgr.add_effect_producer(PointCloudObjectDetectionEffect())
        effects = self.effects_mgr.create_effects(pointcloud)
        self.print_effects(effects)
    def print_effects(self,effects):
        for effect in effects:
            if isinstance(effect, ColorChange):
                x, y, w, h = effect.bbox
                r, g, b = effect.color
                print(f"ColorChange: bbox=({x}, {y}, {w}, {h}), color=({r}, {g}, {b})")
            elif isinstance(effect, Text3DAdd):
                print(
                    f"Text3DAdd: text='{effect.text}', position={effect.position}, size={effect.size}, color={effect.color}"
                )
            elif isinstance(effect, CubeAdd):
                print(
                    f"CubeAdd: center={effect.center}, size={effect.size}, color={effect.color}, rotation={effect.rotation}"
                )
            elif isinstance(effect, SideEffect):
                print(f"SideEffect: (does nothing)")
    def test_effects(self):
        print("testing effects")
        self.effects_mgr.add_effect_producer(FaceAnnotator())
        pointcloud = o3d.io.read_point_cloud(
            "../output_blurred/blurred_faces_20260218_165613_binary.ply"
        )
        effects = self.effects_mgr.create_effects(pointcloud)
        self.print_effects(effects)
        print(f"{len(effects)} effects created.")
        self.data_exporter.save_ply(pointcloud, "testing_effects.ply")
        self.quit()

    def display_with_detections(self, *args):
        depth = args[0]
        color = args[1]
        detections = self.detector.detect(depth, color, ["face"])

        preview = color.copy()
        print(f"Detections to display: {detections}")
        for text_label, x1, x2, y1, y2 in detections:
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                preview,
                text_label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                preview,
                "Press B to export BLURRED point cloud",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # Display
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Detection - Press B to Export", preview_bgr)

    def init_3d_viewer(self):
        """Initialize the 3D point cloud viewer"""
        # Create empty point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector([[0, 0, 1]])
        self.pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0]])
        # Add to visualizer
        self.o3dvis.add_geometry(self.pcd)
        # Setup view control
        view_control = self.o3dvis.get_view_control()
        view_control.set_zoom(0.8)
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 1])
        view_control.set_up([0, 1, 0])

        # Track first frame for reset
        self.first_frame = True
    

    def convert_frame_to_o3d_pointcloud(self,depth,color)->o3d.geometry.PointCloud:
        o3d_depth = o3d.geometry.Image(depth)
        o3d_color = o3d.geometry.Image(color)

        depth_scale = 1.0 / self.camera.get_depth_scale()
        # depth_scale = 1000.0
        # print(f"Depth scale: {depth_scale}")  # Should be ~0.001
        # print(f"Depth min/max: {depth.min()}/{depth.max()}")
        # print(f"Color shape: {color.shape}, dtype: {color.dtype}")
        # print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=depth_scale,
            convert_rgb_to_intensity=False,
        )

        pcd_new = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self.pinhole_camera_intrinsic
        )
        return pcd_new
        
    def display_pointcloud(self, *args):
        depth = args[0]
        color = args[1]
        pcd_new = self.convert_frame_to_o3d_pointcloud(depth,color)

        # for some reason it needs to be rotated bcs the PC gets displayed upside-down
        pcd_new.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        print(f"Points created: {len(pcd_new.points)}")
        # don't create new one
        self.pcd.points = pcd_new.points
        self.pcd.colors = pcd_new.colors

        # Reset view on first frame
        if self.first_frame:
            self.o3dvis.reset_view_point(True)
            self.first_frame = False

        # should include poll_events to render
        self.o3dvis.update_geometry(self.pcd)
        self.o3dvis.poll_events()
        self.o3dvis.update_renderer()

    def export_blurred(self, *args):
        print("\n" + "=" * 60)
        print("exporting pointcloud with blurred objects")
        print("=" * 60)
        depth = args[0]
        color = args[1]
        detections = self.detector.detect(depth, color, ["face"])
        if len(detections) > 0:
            print(f"Blurring {len(detections)} detected objects in 2D image...")
            color_blurred = color
            for label, x1, x2, y1, y2 in detections:
                print(f"Blurring {label} in point cloud.")
                color_blurred = Processor2D.gaussian_blur(color_blurred, x1, x2, y1, y2)
        else:
            print("No specified objects detected - using original image")
            color_blurred = color

        print("Creating 3D point cloud with blurred colors...")
        cloud = self.convert_frame_to_o3d_pointcloud(
            depth,
            color_blurred,
        )
        filename = "blurred"
        ply_filename = f"{filename}.ply"
        ply_path = self.data_exporter.save_ply(cloud, ply_filename, len(detections))
        print(f"Blurred 3D: {ply_path}")

        blurred_filename = f"{filename}.jpg"
        self.data_exporter.save_jpeg(color_blurred, blurred_filename)
        print(f"Blurred 2D: {blurred_filename}")

        pcd_filename = f"{filename}.pcd"
        pcd_path = self.data_exporter.output_dir / pcd_filename
        subprocess.run(["pcl_ply2pcd", str(ply_path), str(pcd_path)], check=True)
        # Launch viewer, the OS might say that this process has hanged, just kill the pcl_viewer window
        print(f"Launching PCL viewer...")
        subprocess.run(["pcl_viewer", str(pcd_path)])

        print("EXPORT COMPLETE!")
        print(f"   Points: {len(cloud.points)}")
        print(f"   Objects blurred: {len(detections)}")
        print("=" * 60 + "\n")
        pass

    def quit(self):
        self.should_continue = False
        self.camera.stop()
        if enable_pointclouds_rendering:
            self.o3dvis.destroy_window()

    def interactive_mode(self):
        """Interactive mode with live preview and export"""
        print("\n=== Face Detection 2D → Blurred Point Cloud Export ===")
        print("Controls:")
        print("  B - Export point cloud with BLURRED FACES (ASCII PLY)")
        print("  M - Export with BLURRED FACES (Binary PLY - smaller file)")
        print("  X - Export with blurred faces + random rect (ASCII)")
        print("  N - Export without blur (original)")
        print("  V - View current frame as 3D (no export)")
        print("  R - New random rectangle")
        print("  Q - Quit")
        print()

        self.bus.on_key("b", lambda *data: print(f"Hello from event bus, {len(data)}"))
        self.bus.on_key("b", self.export_blurred)
        self.bus.on_key("q", lambda *data: self.quit())
        self.bus.on_frame(self.display_with_detections)
        if enable_pointclouds_rendering:
            self.bus.on_frame(self.display_pointcloud)
        self.should_continue = True
        try:
            while self.should_continue:
                success,depth, color = self.camera.read_frame()
                if not success:
                    continue
                self.bus.new_frame(depth, color)
                key = cv2.waitKey(1) & 0xFF
                self.bus.key_pressed(chr(key), depth, color)

        finally:
            print("Should clear up resources")

    def stop(self):
        # should stop the camera.pipeline.stop()
        print("Pipeline stopped running operations for shutdown")


MODES = {
    "interactive": lambda app: app.interactive_mode(),
    "test_effects": lambda app: app.test_effects(),
    "test_classification": lambda app: app.test_classification()
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Point Cloud Application",exit_on_error=False)
    parser.add_argument(
        "--app",
        choices=MODES.keys(),
        default="interactive",
        help="Application mode to run (default: interactive)",
    )
    args,unknown = parser.parse_known_args()

    app = Application()

    try:
        MODES[args.app](app)
    finally:
        app.stop()


if __name__ == "__main__":
    main()
