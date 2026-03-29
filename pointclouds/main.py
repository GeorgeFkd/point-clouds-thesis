from effects import *
    
import subprocess
import pyrealsense2 as rs
import numpy as np
import cv2
import random
from pathlib import Path
from datetime import datetime
import os
import open3d as o3d
import open3d.ml.torch as ml3d
from camera import RealsenseCamera, FakeCamera
from models import PointPillarsDetector

use_realsense_camera = RealsenseCamera.check_if_present()
display_session_type = os.getenv("XDG_SESSION_TYPE")
enable_pointclouds_rendering = display_session_type != "wayland"
enable_semantic_segmentation = False


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

    def get_torch_ckpts(self):
        """Download and return checkpoint paths from open3d-ml releases"""
        kpconv_url = "https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_semantickitti_202009090354utc.pth"
        randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth"

        weights_dir = "./weights"
        os.makedirs(weights_dir, exist_ok=True)

        ckpt_path_r = os.path.join(weights_dir, "vis_weights_RandLANet.pth")
        if not os.path.exists(ckpt_path_r):
            print(f"  Downloading RandLANet weights...")
            os.system(f"wget {randlanet_url} -O {ckpt_path_r}")

        ckpt_path_k = os.path.join(weights_dir, "vis_weights_KPFCNN.pth")
        if not os.path.exists(ckpt_path_k):
            print(f"  Downloading KPFCNN weights...")
            os.system(f"wget {kpconv_url} -O {ckpt_path_k}")

        return ckpt_path_r, ckpt_path_k

    def run_semantic_segmentation(
        self, pointcloud: o3d.geometry.PointCloud, model="kpfcnn"
    ):
        """Run semantic segmentation on point cloud"""
        points = np.asarray(pointcloud.points).astype(np.float32)

        # Prepare data for ML3D
        data = {
            "point": points,
            "feat": None,
            "label": np.zeros(len(points), dtype=np.int32),
        }

        if model == "randlanet":
            results = self.pipeline_randlanet.run_inference(data)
        else:  
            results = self.pipeline_kpfcnn.run_inference(data)

        pred_labels = (results["predict_labels"] + 1).astype(np.int32)
        pred_labels[0] = 0  # Fill "unlabeled" value

        return pred_labels

    def visualize_semantic_segmentation(self, pointcloud: o3d.geometry.PointCloud):
        """Visualize semantic segmentation results"""
        points = np.asarray(pointcloud.points).astype(np.float32)

        data = {
            "point": points,
            "feat": None,
            "label": np.zeros(len(points), dtype=np.int32),
        }

        print("Running semantic segmentation...")
        results_r = self.pipeline_randlanet.run_inference(data)
        pred_label_r = (results_r["predict_labels"] + 1).astype(np.int32)
        pred_label_r[0] = 0

        results_k = self.pipeline_kpfcnn.run_inference(data)
        pred_label_k = (results_k["predict_labels"] + 1).astype(np.int32)
        pred_label_k[0] = 0

        vis_points = []

        vis_points.append(
            {
                "name": "camera_frame",
                "points": points,
                "labels": data["label"],
                "pred": pred_label_k,
            }
        )

        vis_points.append(
            {
                "name": "randlanet",
                "points": points,
                "labels": pred_label_r,
            }
        )

        vis_points.append(
            {
                "name": "kpfcnn",
                "points": points,
                "labels": pred_label_k,
            }
        )

        print(f"Opening ML3D visualizer for {len(vis_points)}")
        try:
            self.ml3d_vis.visualize(vis_points)
        except Exception as e:
            print("Open3D-ML Error is:", e)

    def __init__(self):
        self.camera = (
            RealsenseCamera()
            if use_realsense_camera
            else FakeCamera("./example-faces.ply")
        )
        self.pointpillars_detector = PointPillarsDetector()
        self.data_exporter = DataExporter()
        self.effects_mgr = EffectsManager()
        self.bus = EventBus()
        intr = self.camera.get_intrinsics()

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
        )
        if enable_pointclouds_rendering:
            self.bboxes = []
            self.o3dvis = o3d.visualization.Visualizer()
            self.o3dvis.create_window(
                window_name="PC Visualiser", width=640, height=480
            )
            self.o3dvis.get_render_option().point_size = 2

            self.init_3d_viewer()

        if enable_semantic_segmentation:
            print("Loading semantic segmentation models...")

            ckpt_path_r, ckpt_path_k = self.get_torch_ckpts()

            print("  Loading RandLANet...")
            model_r = ml3d.models.RandLANet(ckpt_path=ckpt_path_r)
            self.pipeline_randlanet = ml3d.pipelines.SemanticSegmentation(model_r)
            self.pipeline_randlanet.load_ckpt(model_r.cfg.ckpt_path)

            print("  Loading KPFCNN...")
            model_k = ml3d.models.KPFCNN(ckpt_path=ckpt_path_k)
            self.pipeline_kpfcnn = ml3d.pipelines.SemanticSegmentation(model_k)
            self.pipeline_kpfcnn.load_ckpt(model_k.cfg.ckpt_path)

            kitti_labels = ml3d.datasets.SemanticKITTI.get_label_to_names()
            self.label_names = kitti_labels

            self.ml3d_vis = ml3d.vis.Visualizer()
            lut = ml3d.vis.LabelLUT()
            for val in sorted(kitti_labels.keys()):
                lut.add_label(kitti_labels[val], val)
            self.ml3d_vis.set_lut("labels", lut)
            self.ml3d_vis.set_lut("pred", lut)

            print("✓ Semantic segmentation models loaded")

    def visualize_detections_3d(self, pointcloud, results):
        """Visualize 3D bounding boxes on point cloud"""
    
        print(f"Results are: {results}")
        boxes = results.get("predict_bboxes", np.array([]))
        scores = results.get("predict_scores", np.array([]))
        labels = results.get("predict_labels", np.array([]))

        if len(boxes) == 0:
            print("No objects detected!")
            o3d.visualization.draw_geometries([pointcloud])
            return

        print(f"\nDetected {len(boxes)} objects:")

        geometries = [pointcloud]

        for i in range(len(boxes)):
            # PointPillars boxes: [x, y, z, w, h, d, yaw]
            x, y, z, w, h, d, yaw = boxes[i]
            score = scores[i]
            label = int(labels[i])

            if score < 0.3:  # Confidence threshold
                continue

            class_name = f"class_{label}"
            print(f"  {class_name}: {score:.2f} at ({x:.2f}, {y:.2f}, {z:.2f})")

            R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
            bbox = o3d.geometry.OrientedBoundingBox(
                center=[x, y, z], R=R, extent=[w, h, d]
            )

            # Color by class
            colors = {
                0: (1, 0, 0),  # Car - Red
                1: (0, 1, 0),  # Pedestrian - Green
                2: (0, 0, 1),  # Cyclist - Blue
            }
            bbox.color = colors.get(label, (1, 1, 1))

            geometries.append(bbox)

        # Visualize
        o3d.visualization.draw_geometries(
            geometries, window_name="PointPillars 3D Detection"
        )

    def test_detection_3d(self):
        model_used = self.pointpillars_detector
        while True:
            success, depth, color = self.camera.read_frame()

            if not success:
                return

            # Convert to point cloud
            pointcloud = self.convert_frame_to_o3d_pointcloud(depth, color)

            # Run detection
            results = model_used.detect(pointcloud)

            # Visualize
            self.visualize_detections_3d(pointcloud, results)
        pass

    def test_semantic_segmentation(self):
        """Test semantic segmentation on a captured frame"""
        if display_session_type == "wayland":
            print("Might not work correctly under wayland")
        while True:
            success, depth, color = self.camera.read_frame()
            if not success:
                print("Failed to capture frame!")
                return
            # Convert to point cloud
            pointcloud = self.convert_frame_to_o3d_pointcloud(depth, color)
            # Run and visualize semantic segmentation
            # self.render_2d(depth,color,pointcloud)
            self.visualize_semantic_segmentation(pointcloud)

    def apply_effects(self, effects, depth, color, pointcloud: o3d.geometry.PointCloud):
        for effect in effects:
            if isinstance(effect, ColorChange):
                x, y, w, h = effect.bbox
                r, g, b = effect.color
                print(f"ColorChange: bbox=({x}, {y}, {w}, {h}), color=({r}, {g}, {b})")
            elif isinstance(effect,Box3DAdd):
                if not enable_pointclouds_rendering:
                    return
                R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, effect.yaw))
                bbox = o3d.geometry.OrientedBoundingBox(
                    center=effect.center,
                    R=R,
                    extent=effect.extent
                )
                # Set color
                bbox.color = np.asarray([0.5,0.5,0.5],dtype=np.float64)
                self.bboxes.append(bbox)
                
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
            # Launch viewer, the OS might say that this process has hanged, just kill the pcl_viewer window to continue
            print(f"Launching PCL viewer...")
            subprocess.run(["pcl_viewer", str(pcd_path)])
        else:
            self.pcd.points = pointcloud.points
            self.pcd.colors = pointcloud.colors

            if self.first_frame:
                self.o3dvis.reset_view_point(True)
                self.first_frame = False

            self.o3dvis.update_geometry(self.pcd)
            for bbox in self.bboxes:
                self.o3dvis.add_geometry(bbox,reset_bounding_box=False)

            self.o3dvis.poll_events()
            self.o3dvis.update_renderer()
            self.bboxes = []

    def collect_changes_to_pointcloud(self, depth, color, pointcloud):
        pcd_new = self.convert_frame_to_o3d_pointcloud(depth, color)
        pointcloud.points = pcd_new.points
        pointcloud.colors = pcd_new.colors

    def interactive_mode(self):
        self.effects_mgr.add_effect_producers(
            [
                # FnCall(self.render_2d),
                BlurDetections2DEffect(),
                Detections3DEffect(),
                FnCall(self.collect_changes_to_pointcloud),
                FnCall(self.render_3d),
                # FnCall(
                #     lambda d, c, p: self.data_exporter.save_ply(
                #         p, "testing_effects.ply"
                #     )
                # ),
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
    "test_segmentation": lambda app: app.test_semantic_segmentation(),
    "test_detection_3d": lambda app: app.test_detection_3d(),
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

    if args.app == "test_segmentation":
        print("enabling semantic segmentation flag")
        global enable_semantic_segmentation
        enable_semantic_segmentation = True
    app = Application()

    try:
        MODES[args.app](app)
    finally:
        app.stop()


if __name__ == "__main__":
    main()
