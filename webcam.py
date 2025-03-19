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
        # Try MSMF first
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if cap.isOpened():
            cameras.append((index, f"Camera {index} (MSMF)"))
            cap.release()
        else:
            # Fallback to DSHOW
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                cameras.append((index, f"Camera {index} (DSHOW)"))
                cap.release()
            else:
                break  # Stop at the first failure
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

        # Frame queue for threaded camera reading
        self.frame_queue = Queue(maxsize=1)
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

        # Video Display
        self.display_width = 960
        self.display_height = 540
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
        print(f"Available cameras: {self.camera_list}")  # Debug
        camera_names = [name for _, name in self.camera_list] if self.camera_list else ["No Camera"]
        print(f"Camera names: {camera_names}")  # Debug
        self.camera_var = StringVar(root)
        self.camera_var.set(camera_names[0])  # Set initial value to first option
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

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera at index {self.camera_index} with MSMF. Trying DSHOW...")
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera at index {self.camera_index} with DSHOW.")
                self.cap = None
                return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        retries = 10
        for i in range(retries):
            ret, frame = self.cap.read()
            if ret:
                break
            print(f"Camera not ready, retrying ({i+1}/{retries})...")
            time.sleep(0.5)
        else:
            print("Error: Camera failed to provide frames after retries.")
            self.cap.release()
            self.cap = None
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera {self.camera_index} Info:")
        print(f"  Requested Resolution: 1920x1080")
        print(f"  Actual Resolution: {width}x{height}")
        print(f"  Requested Frame Rate: 30 FPS")
        print(f"  Actual Frame Rate: {fps}")

        if width != 1920 or height != 1080:
            print("Warning: Failed to set 1920x1080. Trying 1280x720...")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"  Fallback Resolution: {width}x{height}")

        self.cap.set(cv2.CAP_PROP_FPS, 30)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"  Final Frame Rate: {fps}")

    def camera_reader(self):
        while self.camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                try:
                    self.frame_queue.put(frame, block=False)
                except Queue.Full:
                    pass
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
        self.current_transformer = self.transformer_map.get(value, self.transformer_dummy)

    def transformer_dummy(self, frame):
        return frame

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
        if target_bits <= 2048:
            frame = cv2.resize(frame, (320, 180))
        else:
            frame = cv2.resize(frame, (640, 360))
        frame_rgb = (frame // 64) * 64
        self.frame_count += 1
        if self.last_frame is not None and self.frame_count % 30 != 0:
            delta = frame_rgb.astype(np.int16) - self.last_frame.astype(np.int16)
            delta = np.where(np.abs(delta) > 32, delta // 32, 0)
            frame_rgb = (self.last_frame + delta * 32).clip(0, 255).astype(np.uint8)
        self.last_frame = frame_rgb.copy()
        if target_bits <= 2048:
            frame = cv2.resize(frame_rgb, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        else:
            frame = cv2.resize(frame_rgb, (3840, 2160), interpolation=cv2.INTER_NEAREST)
        return frame

    def transformer_intermediate(self, frame):
        return self.transformer_dummy(frame)

    def transformer_retro_flashy(self, frame):
        return self.transformer_dummy(frame)

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
            data_rate = 49.152
        elif effect == "Retro 4K":
            data_rate = 98.304
        elif effect in ["Interlaced", "Fibonacci Compression", "Intermediate", "Retro Flashy"]:
            data_rate = 1920 * 1080 * 30 * 30 / 1000
        else:
            data_rate = 0

        return fps, data_rate

    def update(self):
        current_time = time.time()
        if current_time - self.last_update_time < 0.041:
            self.root.after(10, self.update)
            return

        self.last_update_time = current_time

        if self.running:
            try:
                frame = self.frame_queue.get_nowait()
            except Empty:
                frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
                cv2.putText(frame, "No Camera Feed", (self.display_width//4, self.display_height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                frame = self.current_transformer(frame)
                frame = cv2.resize(frame, (self.display_width, self.display_height))

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

        self.root.after(41, self.update)

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