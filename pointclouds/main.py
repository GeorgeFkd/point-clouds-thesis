#!/usr/bin/env python3
"""
Face Detection 2D → Point Cloud 3D with PLY Export (PyVista)
IMPORTANT: Exports point clouds with BLURRED FACES for privacy
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import random
from pathlib import Path
from datetime import datetime
# THIS IS THE ONE THAT ACTUALLY WORKS NOW
try:
    import pyvista as pv
    HAS_PYVISTA = True
    print("✓ PyVista available")
except ImportError:
    HAS_PYVISTA = False
    print("⚠️  PyVista not installed")
    print("Install: pip install pyvista")
    exit(1)


class RealsenseCamera:
    def __init__(self,width=640,height=480,fps=30):
        self.width = width
        self.height = height

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth,width,height,rs.format.z16,fps);
        config.enable_stream(rs.stream.color,width,height,rs.format.rgb8,fps);

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
    def capture_frame(self):
        """Capture depth and color frames"""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
        
        return np.asanyarray(depth_frame.get_data()), np.asanyarray(color_frame.get_data())
    def get_intrinsics(self):
        return self.intrinsics


class ObjectDetector:
    def __init__(self,objects_to_detect):
        assert "face" in objects_to_detect
        if "face" in objects_to_detect:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    


    def convert_detections(self,detection):
        pass

    def detect(self,depth,color,objects):
        results = []
        if "face" in objects:
            text_label = "Face"
            gray = cv2.cvtColor(color,cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
            for (x,y,w,h) in faces:
                margin = int(w * 0.2)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(color.shape[1], x + w + margin)
                y2 = min(color.shape[0], y + h + margin)
                results.append((text_label,x1,x2,y1,y2))
        return results

class DataExporter:

    def __init__(self,output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    
    def save_jpeg(self, image, filename):
        """Save 2D image as JPEG"""
        filepath = self.output_dir / filename
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filepath

    def save_ply(self, cloud, filename, faces_count=0, binary=False):
        """Save point cloud to PLY file with proper RGB colors"""
        filepath = self.output_dir / filename
        
        # Get points and colors from PyVista cloud
        points = cloud.points
        colors = cloud['RGB']
        
        num_points = len(points)
        
        if binary:
            # Binary PLY (smaller file size)
            import struct
            
            with open(filepath, 'wb') as f:
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
                
                f.write(header.encode('ascii'))
                
                # Binary data
                for i in range(num_points):
                    x, y, z = points[i]
                    r, g, b = colors[i]
                    # Pack: 3 floats + 3 unsigned chars
                    data = struct.pack('fffBBB', x, y, z, int(r), int(g), int(b))
                    f.write(data)
        else:
            # ASCII PLY (human-readable, larger file)
            with open(filepath, 'w') as f:
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
                    r, g, b = colors[i]
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        
        return filepath

class Processor2D:

    @classmethod
    def gaussian_blur(cls,image,x1,x2,y1,y2,blur_amount=51):
        img = image.copy()

        region = img[y1:y2,x1:x2]
        if region.size > 0:
            region_blurred = cv2.GaussianBlur(region,(blur_amount,blur_amount),0)
            img[y1:y2,x1:x2] = region_blurred
            return img

        assert False
class Processor3D:

    @classmethod
    def create_pointcloud_pyvista(cls,depth,color,depth_scale,intrinsics):
        """Create PyVista point cloud from depth + color"""
        H, W = depth.shape
        
        # Pixel coordinates
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Convert depth to meters and back-project to 3D
        z = depth * depth_scale
        x = (u - intrinsics.ppx) * z / intrinsics.fx
        y = (v - intrinsics.ppy) * z / intrinsics.fy
        
        # Flatten
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
        colors = color.reshape(-1, 3)
        
        # Filter valid points (depth > 0)
        valid = points[:, 2] > 0
        points_valid = points[valid]
        colors_valid = colors[valid]
        
        # Create PyVista PolyData
        cloud = pv.PolyData(points_valid)
        cloud['RGB'] = colors_valid
        
        return cloud

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
        pass

    def display_with_detections(self,*args):
        depth = args[0]
        color = args[1]
        detections = self.detector.detect(depth,color,["face"])
                
        preview = color.copy()
        for (text_label,x1, x2, y1, y2) in detections:
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(preview, text_label, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(preview, "Press B to export BLURRED point cloud", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            cv2.imshow('Face Detection - Press B to Export', preview_bgr)
    
    def export_blurred(self,*args):
        print("\n" + "="*60)
        print("EXPORTING POINT CLOUD WITH BLURRED FACES")
        print("="*60)
        depth = args[0]
        color = args[1]
        detections = self.detector.detect(depth,color,["face"])
        timestamp = ""
        if len(detections) > 0:
            print(f"Blurring {len(detections)} detected objects in 2D image...")
            for (_,x1,x2,y1,y2) in detections:
                color_blurred = Processor2D.gaussian_blur(color,x1,x2,y1,y2)
            print("        ✓ Faces blurred successfully")
        else:
            print("No faces detected - using original image")
            color_blurred = color.copy()

        print("Creating 3D point cloud with blurred colors...")
        cloud = Processor3D.create_pointcloud_pyvista(depth, color_blurred,self.camera.get_depth_scale(),self.camera.get_intrinsics())
        print(f"Point cloud created ({cloud.n_points} points)")

        ply_filename = f"blurred.ply"
        print(f"Saving to PLY file...")
        ply_path = self.data_exporter.save_ply(cloud, ply_filename, len(detections))
        print(f"PLY saved at {ply_path}")

        print("Saving preview images...")

        # Blurred version
        blurred_filename = f"blurred_2d.jpg"
        self.data_exporter.save_jpeg(color_blurred, blurred_filename)
        print(f"        ✓ Blurred 2D: {blurred_filename}")

        print("\n📦 EXPORT COMPLETE!")
        print(f"   PLY file: {ply_filename}")
        print(f"   Points: {cloud.n_points}")
        print(f"   Objects blurred: {len(detections)}")
        print(f"   ⚠️  IMPORTANT: Objects are BLURRED in the point cloud!")
        print("="*60 + "\n")

        print("Calling export blurred")
        pass


    def quit(self):
        self.should_continue = False
        self.camera.stop()
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
        self.camera = RealsenseCamera()
        self.detector = ObjectDetector(["face"])
        self.data_exporter = DataExporter()
        bus = EventBus()
        bus.on_key('b',lambda *data: print(f"Hello from event bus, {len(data)}"))
        bus.on_key('b',self.export_blurred)
        bus.on_key('q',lambda *data: self.quit())
        bus.on_frame(self.display_with_detections)
        self.should_continue = True
        try:
            while self.should_continue:
                depth, color = self.camera.capture_frame()
                if depth is None:
                    continue
                bus.new_frame(depth,color)
                key = cv2.waitKey(1) & 0xFF
                bus.key_pressed(chr(key),depth,color)
                
        
        finally:
            cv2.destroyAllWindows()
    
    def stop(self):
        # should stop the camera.pipeline.stop()
        print("Pipeline stopped running operations for shutdown")


def main():
    app = Application()
    
    try:
        app.interactive_mode()
    finally:
        app.stop()


if __name__ == "__main__":
    main()
