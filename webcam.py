import cv2
import numpy as np
from tkinter import Tk, Label, Button, OptionMenu, StringVar, Frame
from PIL import Image, ImageTk
import time
import threading
from queue import Queue, Empty

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
        self.last_display_frame = None  # Store the last frame for display
        self.last_frame_time = 0  # Track when we last got a frame

        # Frame queue for threaded camera reading
        self.frame_queue = Queue(maxsize=5)  # Increased to buffer more frames
        self.camera_thread = None
        self.camera_running = False

        # Transformer map
        self.transformer_map = {
            "Dummy": self.transformer_dummy,
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

        # Video Display (match camera's default 640x480)
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
                except Queue.Full:
                    pass
            # Removed time.sleep to let the thread run as fast as the camera can provide frames

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
        self.current_transformer = self.transformer_map.get(value, self.transformer_dummy)

    def transformer_dummy(self, frame):
        return frame  # No resizing since display matches camera resolution

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
        # Downsample to half resolution (320x240, keeping 4:3 aspect ratio)
        frame = cv2.resize(frame, (320, 240))
        frame_rgb = (frame // 64) * 64
        self.frame_count += 1
        if self.last_frame is not None and self.frame_count % 30 != 0:
            delta = frame_rgb.astype(np.int16) - self.last_frame.astype(np.int16)
            delta = np.where(np.abs(delta) > 32, delta // 32, 0)
            frame_rgb = (self.last_frame + delta * 32).clip(0, 255).astype(np.uint8)
        self.last_frame = frame_rgb.copy()
        return cv2.resize(frame_rgb, (self.display_width, self.display_height), interpolation=cv2.INTER_NEAREST)

    def transformer_intermediate(self, frame):
        return frame

    def transformer_retro_flashy(self, frame):
        return frame

    def transformer_interlace(self, frame):
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
        if current_time - self.last_update_time < 0.030:  # Slightly less than 33ms to account for jitter
            self.root.after(10, self.update)
            return

        self.last_update_time = current_time

        if self.running:
            try:
                frame = self.frame_queue.get_nowait()
                self.last_frame_time = current_time  # Update last frame time
                frame = self.current_transformer(frame)
                self.last_display_frame = frame.copy()  # Store for fallback
            except Empty:
                if self.last_display_frame is not None and (current_time - self.last_frame_time) < 1.0:
                    frame = self.last_display_frame.copy()  # Use last frame if recent
                else:
                    frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
                    cv2.putText(frame, "No Camera Feed", (self.display_width//4, self.display_height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if current_time - self.last_metrics_time >= 0.5:
                self.last_fps, self.last_data_rate = self.calculate_metrics()
                self.last_metrics_time = current_time
            cv2.putText(frame, f"FPS: {self.last_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Data Rate: {self.last_data_rate:.1f} kbps", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.root.after(33, self.update)

    def __del__(self):
        self.camera_running = False
        if self.camera_thread:
            self.camera_thread.join()
        if self.cap:
            self.cap.release()

def main():
    root = Tk()
    app = VideoApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()