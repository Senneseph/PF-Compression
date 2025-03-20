import cv2
import numpy as np
from tkinter import Tk, Label, Button, OptionMenu, StringVar, Frame
from PIL import Image, ImageTk
import time
import threading
import queue
from queue import Queue, Empty  # Ensure Queue and Empty are imported

def get_available_cameras():
    cameras = []
    for index in range(10):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cameras.append((index, f"Camera {index}"))
            cap.release()
        else:
            break
    return cameras

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Shader Effects")
        self.cap = None
        self.running = False
        self.last_frame = None
        self.frame_count = 0
        self.frame_times = []
        self.start_time = time.time()
        self.last_update_time = 0
        self.last_metrics_time = 0
        self.last_fps = 0
        self.last_data_rate = 0
        self.last_display_frame = None
        self.last_frame_time = 0

        # Frame queue for threaded camera reading
        self.frame_queue = Queue(maxsize=5)  # Use Queue class
        self.camera_thread = None
        self.camera_running = False

        # Transformer map
        self.transformer_map = {
            "Dummy": self.transformer_dummy,
            "Vectorwave": self.transformer_vectorwave,
            "Fibonacci Compression": self.transformer_fibonacci,
            "Retro Compression": self.transformer_retro,
            "Intermediate": self.transformer_intermediate,
            "Retro Flashy": self.transformer_retro_flashy,
            "Interlaced": self.transformer_interlace,
        }
        self.current_transformer = self.transformer_dummy

        # GUI Setup
        self.frame = Frame(root)
        self.frame.pack()

        # Video Display
        self.display_width = 640
        self.display_height = 480
        self.label = Label(self.frame)
        self.label.pack()

        # Controls
        self.control_frame = Frame(self.frame)
        self.control_frame.pack()

        # On/Off Button
        self.toggle_btn = Button(self.control_frame, text="Start Video", command=self.toggle_video)
        self.toggle_btn.pack(side="left", padx=5, pady=5)

        # Effect Drop-down
        self.effect_var = StringVar(root)
        self.effect_var.set("Dummy")
        effects = list(self.transformer_map.keys())
        self.effect_menu = OptionMenu(self.control_frame, self.effect_var, *effects, command=self.update_transformer)
        self.effect_menu.pack(side="left", padx=5, pady=5)

        # Camera Drop-down
        self.camera_list = get_available_cameras()
        print(f"Available cameras: {self.camera_list}")
        camera_names = [name for _, name in self.camera_list] if self.camera_list else ["No Camera"]
        print(f"Camera names: {camera_names}")
        self.camera_var = StringVar(root)
        self.camera_var.set(camera_names[0])
        self.camera_menu = OptionMenu(self.control_frame, self.camera_var, *camera_names, command=self.update_camera)
        self.camera_menu.pack(side="left", padx=5, pady=5)

        # Camera Setup
        self.camera_index = self.camera_list[0][0] if self.camera_list else 0
        self.init_camera()

        # Start Update Loop
        self.update()

    def init_camera(self):
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera at index {self.camera_index}.")
            self.cap = None
            return

        # Query default settings
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera {self.camera_index} Info:")
        print(f"  Default Resolution: {width}x{height}")
        print(f"  Default Frame Rate: {fps}")

    def camera_reader(self):
        while self.camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    time.sleep(0.01)
                    continue
            time.sleep(0.01)

    def update_camera(self, value):
        for index, name in self.camera_list:
            if name == value:
                self.camera_index = index
                break
        self.init_camera()
        if self.running:
            self.start_time = time.time()
            self.frame_times = []
            self.frame_count = 0

    def toggle_video(self):
        self.running = not self.running
        self.toggle_btn.config(text="Stop Video" if self.running else "Start Video")
        if self.running:
            if not self.cap:
                self.init_camera()
            if self.cap:
                self.camera_running = True
                self.camera_thread = threading.Thread(target=self.camera_reader, daemon=True)
                self.camera_thread.start()
            self.start_time = time.time()
            self.frame_times = []
            self.frame_count = 0
        else:
            self.camera_running = False
            if self.camera_thread:
                self.camera_thread.join()

    def update_transformer(self, value):
        self.last_frame = None
        self.frame_count = 0
        self.current_transformer = self.transformer_map.get(value, self.transformer_dummy)

    def transformer_dummy(self, frame):
        return frame
    
    def transformer_vectorwave(self, frame, max_shapes=256):
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours (vectorized shapes)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first) and limit to max_shapes
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_shapes]
        
        # Create a blank canvas matching the original frame size
        output = np.zeros_like(frame)
        
        # Retro neon colors (e.g., cyan, magenta, green)
        colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0),
                    (0, 255, 0), (255, 0, 0), (0, 0, 255),
                    (128, 128, 128), (255, 255, 155)
                  ]
        
        # Draw contours as wireframes
        for i, contour in enumerate(contours):
            color = colors[i % len(colors)]  # Cycle through neon colors
            # Approximate the contour to reduce points (simplify the shape)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(output, [approx], -1, color, 1)  # Thin lines for wireframe
        
        return output

    def transformer_fibonacci(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = 256 // 8
        quantized = (gray // scale) * scale
        if self.last_frame is not None:
            delta = quantized.astype(np.int16) - self.last_frame.astype(np.int16)
            delta = np.where(np.abs(delta) > 10, delta, 0)
            quantized = (self.last_frame + delta).clip(0, 255).astype(np.uint8)
        self.last_frame = quantized.copy()
        return cv2.cvtColor(quantized, cv2.COLOR_GRAY2BGR)

    def transformer_retro(self, frame, target_bits=2048):
        frame = cv2.resize(frame, (320, 240))
        frame_rgb = (frame // 64) * 64
        self.frame_count += 1
        if self.last_frame is not None and self.frame_count % 30 != 0:
            delta = frame_rgb.astype(np.int16) - self.last_frame.astype(np.int16)
            delta = np.where(np.abs(delta) > 32, delta // 32, 0)
            frame_rgb = (self.last_frame + delta * 32).clip(0, 255).astype(np.uint8)
        self.last_frame = frame_rgb.copy()
        return cv2.resize(frame_rgb, (self.display_width, self.display_height), interpolation=cv2.INTER_NEAREST)

    # def transformer_intermediate(self, frame):
    def transformer_intermediate(self, frame, bitrate=61440):
        # Resolution Slice
        target_bits = bitrate // 30
        if bitrate <= 61440:
            frame = cv2.resize(frame, (320, 240))
            pixels = 76800
        else:  # 4K
            frame = cv2.resize(frame, (640, 360))
            pixels = 230400

        # Quantize to 16 colors (4-bit LUT)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quantized = (gray // 16) * 16  # ~4 bits/pixel

        # Delta Frame (every 30th as keyframe)
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
        self.frame_count += 1
        if self.last_frame is not None and self.frame_count % 30 != 0:
            delta = quantized.astype(np.int16) - self.last_frame.astype(np.int16)
            delta = np.where(np.abs(delta) > 8, delta // 8, 0)  # ~1 bit/pixel
            quantized = (self.last_frame + delta * 8).clip(0, 255).astype(np.uint8)
        self.last_frame = quantized.copy()

        # Upscale for display
        if bitrate <= 61440:
            frame = cv2.resize(quantized, (self.display_width, self.display_height), interpolation=cv2.INTER_NEAREST)
        else:
            frame = cv2.resize(quantized, (3840, 2160), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # if len(frame.shape) == 2:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # return frame

    def transformer_retro_flashy(self, frame):
        # Resize
        frame = cv2.resize(frame, (320, 240))

        # Quantize to 8 colors (3-bit LUT)
        frame_rgb = (frame // 32) * 32

        # Delta Frame (every 30th as keyframe)
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
        
        self.frame_count += 1

        if self.last_frame is not None and self.frame_count % 30 != 0:
            delta = frame_rgb.astype(np.int16) - self.last_frame.astype(np.int16)
            delta = np.where(np.abs(delta) > 16, delta // 16, 0)  # ~1 bit/pixel
            frame_rgb = (self.last_frame + delta * 16).clip(0, 255).astype(np.uint8)

        self.last_frame = frame_rgb.copy()

        # Upscale for display
        return cv2.resize(frame_rgb, (self.display_width, self.display_height), interpolation=cv2.INTER_NEAREST)

    def transformer_interlace(self, frame):
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if self.last_frame is None:
            self.last_frame = frame.copy()
        self.frame_count += 1
        is_even_frame = (self.frame_count % 2 == 0)
        output_frame = self.last_frame.copy()
        height = frame.shape[0]
        if is_even_frame:
            output_frame[0:height:2, :, :] = frame[0:height:2, :, :]
        else:
            output_frame[1:height:2, :, :] = frame[1:height:2, :, :]
        self.last_frame = output_frame.copy()
        return output_frame

    def calculate_metrics(self):
        current_time = time.time()
        self.frame_times.append(current_time)
        if len(self.frame_times) > 10:
            self.frame_times.pop(0)
        if len(self.frame_times) > 1:
            fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
        else:
            fps = 0

        effect = self.effect_var.get()
        if effect == "Retro Compression":
            data_rate = 61.44  # 2,048 bits/frame × 30 FPS
        elif effect == "Retro 4K":
            data_rate = 122.88  # 4,096 bits/frame × 30 FPS
        elif effect in ["Interlaced", "Fibonacci Compression", "Intermediate", "Retro Flashy"]:
            data_rate = 640 * 480 * 24 * 30 / 1000  # Raw 640x480 at 30 FPS
        else:
            data_rate = 0

        return fps, data_rate

    def update(self):
        current_time = time.time()
        if current_time - self.last_update_time < 0.030:
            self.root.after(10, self.update)
            return

        self.last_update_time = current_time

        if self.running:
            try:
                frame = self.frame_queue.get_nowait()
                self.last_frame_time = current_time
                try:
                    frame = self.current_transformer(frame)
                    self.last_display_frame = frame.copy()
                except Exception as e:
                    print(f"Transformer error: {e}")
                    if self.last_display_frame is not None:
                        frame = self.last_display_frame.copy()
                    else:
                        frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
                        cv2.putText(frame, "Transformer Error", (self.display_width//4, self.display_height//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            except queue.Empty:
                if self.last_display_frame is not None and (current_time - self.last_frame_time) < 1.0:
                    frame = self.last_display_frame.copy()
                else:
                    frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
                    cv2.putText(frame, "No Camera Feed", (self.display_width//4, self.display_height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if current_time - self.last_metrics_time >= 0.5:
                self.last_fps, self.last_data_rate = self.calculate_metrics()
                self.last_metrics_time = current_time
            # cv2.putText(frame, f"FPS: {self.last_fps:.1f}", (10, 30), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame, f"Data Rate: {self.last_data_rate:.1f} kbps", (10, 60), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.root.after(33, self.update)

    def __del__(self):
        self.camera_running = False
        if hasattr(self, 'camera_thread') and self.camera_thread:  # Check if attribute exists
            self.camera_thread.join()
        if self.cap:
            self.cap.release()

def main():
    root = Tk()
    app = VideoApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()