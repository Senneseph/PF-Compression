import cv2
import numpy as np
from tkinter import Tk, Label, Button, OptionMenu, StringVar, Frame
from PIL import Image, ImageTk

# Fibonacci LUT (simplified to 8 entries for 256 colors)
FIB_LUT = [0, 1, 2, 3, 5, 8, 13, 21]  # Map 0-255 to these (quantized)
FIB_SCALE = 256 // len(FIB_LUT)  # ~32 levels

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Shader Effects")
        self.cap = None
        self.running = False
        self.last_frame = None
        self.frame_count = 0

        # GUI Setup
        self.frame = Frame(root)
        self.frame.pack()

        # Video Display
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
        self.effect_var.set("Dummy")  # Default
        effects = ["Dummy", "Fibonacci Compression", "Retro Compression", "Intermediate", "Retro Flashy", "Interlaced"]
        self.effect_menu = OptionMenu(self.control_frame, self.effect_var, *effects)
        self.effect_menu.pack(side="left", padx=5, pady=5)

        # Camera Setup (default to first available)
        self.camera_index = 0  # Change to 1, 2, etc., for OBS/Logitech if needed
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

        # Retrieve and print camera information
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
        contrast = self.cap.get(cv2.CAP_PROP_CONTRAST)
        
        print(f"Camera {self.camera_index} Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  Frame Rate: {fps}")
        print(f"  Brightness: {brightness}")
        print(f"  Contrast: {contrast}")

        # Force 1080p, 16:9, 24 FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 24)

    def toggle_video(self):
        self.running = not self.running
        self.toggle_btn.config(text="Stop Video" if self.running else "Start Video")
        if self.running and not self.cap:
            self.init_camera()

    def transformer_dummy(self, frame):
        """Pass-through transformer."""
        return frame
    
    def get_bitrate(self):
        return 61440

    def transformer_fibonacci(self, frame, user_profile="low"):
        # Dynamic Resolution
        bitrate = self.get_bitrate()
        if bitrate < 5_000_000:  # < 5 Mbps
            frame = cv2.resize(frame, (640, 480))
        elif bitrate < 50_000_000:  # < 50 Mbps
            frame = cv2.resize(frame, (1024, 768))

        # User-Specific LUT
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lut_size = 8 if user_profile == "low" else 16
        scale = 256 // lut_size
        quantized = (gray // scale) * scale  # Quantize to LUT levels

        # Delta Frame
        if self.last_frame is not None:
            delta = quantized.astype(np.int16) - self.last_frame.astype(np.int16)
            delta = np.where(np.abs(delta) > (10 if user_profile == "low" else 5), delta, 0)
            quantized = (self.last_frame + delta).clip(0, 255).astype(np.uint8)

        self.last_frame = quantized.copy()
        return cv2.cvtColor(quantized, cv2.COLOR_GRAY2BGR)
    
    def transformer_intermediate(self, frame, bitrate=61440):
        # Resolution Slice
        target_bits = bitrate // 30
        if bitrate <= 61440:  # 1080p
            frame = cv2.resize(frame, (320, 180))
            pixels = 57600
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
            frame = cv2.resize(quantized, (1600, 1200), interpolation=cv2.INTER_NEAREST)
        else:
            frame = cv2.resize(quantized, (3840, 2160), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    def transformer_retro(self, frame, bitrate=6556144036):
        # Resolution Slice
        target_bits = bitrate // 30
        if bitrate <= 61440:  # 1080p
            frame = cv2.resize(frame, (320, 180))
            pixels = 57600
        else:  # 4K
            frame = cv2.resize(frame, (640, 360))
            pixels = 230400

        # Quantize to 8 colors (3-bit LUT)
        frame_rgb = frame // 32 * 32  # ~3 bits/channel

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
        if bitrate <= 61440:
            frame = cv2.resize(frame_rgb, (1600, 1080), interpolation=cv2.INTER_NEAREST)
        else:
            frame = cv2.resize(frame_rgb, (3840, 2160), interpolation=cv2.INTER_NEAREST)
        return frame
    
    def transformer_retro_flashy(self, frame, target_bits=2048):
        # Resolution Slice (16:9 maintained)
        if target_bits <= 2048:  # 1080p
            frame = cv2.resize(frame, (320, 180))
            pixels = 57600
        else:  # 4K
            frame = cv2.resize(frame, (640, 360))
            pixels = 230400

        # Quantize to 4 colors (2-bit LUT total)
        frame_rgb = (frame // 64) * 64  # ~2 bits/channel, ~6 bits/pixel

        # Delta Frame (every 30th as keyframe)
        self.frame_count += 1
        if self.last_frame is not None and self.frame_count % 30 != 0:
            delta = frame_rgb.astype(np.int16) - self.last_frame.astype(np.int16)
            delta = np.where(np.abs(delta) > 32, delta // 32, 0)  # ~1 bit/pixel
            frame_rgb = (self.last_frame + delta * 32).clip(0, 255).astype(np.uint8)
        self.last_frame = frame_rgb.copy()

        # Upscale for display
        if target_bits <= 2048:
            frame = cv2.resize(frame_rgb, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        else:
            frame = cv2.resize(frame_rgb, (3840, 2160), interpolation=cv2.INTER_NEAREST)
        return frame
    
    def transformer_interlace(self, frame):
        """
        Simulates interlacing by alternating odd/even line updates each frame.
        Args:
            frame: Input frame (BGR, 1920x1080 or 3840x2160).
        Returns:
            Frame with interlaced effect.
        """
        # Ensure we have a last frame to work with
        if self.last_frame is None:
            self.last_frame = frame.copy()

        # Increment frame count
        self.frame_count += 1

        # Determine if this frame updates odd or even lines
        is_even_frame = (self.frame_count % 2 == 0)

        # Create output frame by copying last frame
        output_frame = self.last_frame.copy()

        # Update either odd or even lines
        height = frame.shape[0]
        if is_even_frame:
            # Update even lines (0, 2, 4, ...)
            output_frame[0:height:2, :, :] = frame[0:height:2, :, :]
        else:
            # Update odd lines (1, 3, 5, ...)
            output_frame[1:height:2, :, :] = frame[1:height:2, :, :]

        # Update last frame for next iteration
        self.last_frame = output_frame.copy()

        return output_frame

    def update(self):
        if self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Apply selected transformer
                effect = self.effect_var.get()
                if effect == "Fibonacci Compression":
                    frame = self.transformer_fibonacci(frame)
                if effect == "Retro Compression":
                    frame = self.transformer_retro(frame)
                if effect == "Intermediate":
                    frame = self.transformer_intermediate(frame)
                if effect == "Retro Flashy":
                    frame = self.transformer_retro_flashy(frame)
                if effect == "Interlaced":
                    frame = self.transformer_interlace(frame)
                else:  # Dummy
                    frame = self.transformer_dummy(frame)

                # Convert for Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.imgtk = imgtk  # Keep reference
                self.label.configure(image=imgtk)

        # Loop every 41ms (~24 FPS)
        self.root.after(41, self.update)

    def __del__(self):
        if self.cap:
            self.cap.release()

def main():
    root = Tk()
    app = VideoApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()