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
    Text2DAdd,
    Rect2DAdd,
    FnCall,
    Detection2DEffect,
    BlurDetections2DEffect,
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
from camera import RealsenseCamera, FakeCamera

use_realsense_camera = RealsenseCamera.check_if_present()
display_session_type = os.getenv("XDG_SESSION_TYPE")
enable_pointclouds_rendering = display_session_type != "wayland"

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
        print("Saving PLY")
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
        self.camera = (
            RealsenseCamera()
            if use_realsense_camera
            else FakeCamera("./example-faces.ply")
        )
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
        effects = self.effects_mgr.create_effects([], [], pointcloud)
        self.apply_effects(effects, [], [], [])

    def apply_effects(self, effects, depth, color, pointcloud: o3d.geometry.PointCloud):
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
            elif isinstance(effect, Rect2DAdd):
                cv2.rectangle(color, effect.p1, effect.p2, (0, 255, 0), 2)
            elif isinstance(effect, Text2DAdd):
                cv2.putText(
                    color,
                    effect.text,
                    effect.position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            elif isinstance(effect, SideEffect):
                effect.callback(depth, color, pointcloud)

    def render_2d(self, depth, color, pointcloud):
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Detection - Press B to Export", color)

    def render_3d(self, depth, color, pointcloud):
        if not enable_pointclouds_rendering:
            filename = "rendered"
            ply_filename = f"{filename}.ply"
            ply_path = self.data_exporter.save_ply(pointcloud, ply_filename)
            pcd_filename = f"{filename}.pcd"
            pcd_path = self.data_exporter.output_dir / pcd_filename
            subprocess.run(["pcl_ply2pcd", str(ply_path), str(pcd_path)], check=True)
            # Launch viewer, the OS might say that this process has hanged, just kill the pcl_viewer window
            print(f"Launching PCL viewer...")
            subprocess.run(["pcl_viewer", str(pcd_path)])
        else:
            self.pcd.points = pointcloud.points
            self.pcd.colors = pointcloud.colors

            if self.first_frame:
                self.o3dvis.reset_view_point(True)
                self.first_frame = False

            self.o3dvis.update_geometry(self.pcd)
            self.o3dvis.poll_events()
            self.o3dvis.update_renderer()

    def collect_changes_to_pointcloud(self, depth, color, pointcloud):
        pcd_new = self.convert_frame_to_o3d_pointcloud(depth, color)
        pointcloud.points = pcd_new.points
        pointcloud.colors = pcd_new.colors

    def interactive_mode(self):
        self.effects_mgr.add_effect_producers(
            [
                FnCall(self.render_2d),
                BlurDetections2DEffect(),
                FnCall(self.collect_changes_to_pointcloud),
                FnCall(self.render_3d),
                FnCall(
                    lambda d, c, p: self.data_exporter.save_ply(
                        p, "testing_effects.ply"
                    )
                ),
            ]
        )
        print("\n=== Face Detection 2D → Blurred Point Cloud Export ===")
        print("Controls:")
        print("  Q - Quit")
        print()

        self.bus.on_key("q", lambda *data: self.quit())
        self.bus.on_frame(self.execute_effects)
        self.should_continue = True
        try:
            while self.should_continue:
                success, depth, color = self.camera.read_frame()
                if not success:
                    continue
                self.bus.new_frame(depth, color)
                key = cv2.waitKey(1) & 0xFF
                self.bus.key_pressed(chr(key), depth, color)
        finally:
            print("should gather resources")

    def execute_effects(self, *args):
        depth = args[0]
        color = args[1]
        pointcloud = self.convert_frame_to_o3d_pointcloud(depth, color)
        effects = self.effects_mgr.create_effects(depth, color, pointcloud)
        print(f"{len(effects)} effects created.")
        self.apply_effects(effects, depth, color, pointcloud)

    def init_3d_viewer(self):
        """Initialize the 3D point cloud viewer"""
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector([[0, 0, 1]])
        self.pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0]])
        self.o3dvis.add_geometry(self.pcd)
        view_control = self.o3dvis.get_view_control()
        view_control.set_zoom(0.8)
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 1])
        view_control.set_up([0, 1, 0])

        # Track first frame for reset
        self.first_frame = True

    def convert_frame_to_o3d_pointcloud(self, depth, color) -> o3d.geometry.PointCloud:
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
    def quit(self):
        self.should_continue = False
        self.camera.stop()
        if enable_pointclouds_rendering:
            self.o3dvis.destroy_window()

    def stop(self):
        # should stop the camera.pipeline.stop()
        print("Pipeline stopped running operations for shutdown")


MODES = {
    "interactive": lambda app: app.interactive_mode(),
    "test_classification": lambda app: app.test_classification(),
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Point Cloud Application", exit_on_error=False
    )
    parser.add_argument(
        "--app",
        choices=MODES.keys(),
        default="interactive",
        help="Application mode to run (default: interactive)",
    )
    args, unknown = parser.parse_known_args()

    app = Application()

    try:
        MODES[args.app](app)
    finally:
        app.stop()


if __name__ == "__main__":
    main()
