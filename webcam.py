import cv2
import numpy as np
from tkinter import Tk, Label, Button, OptionMenu, StringVar, Frame
from PIL import Image, ImageTk
import time
import threading
import queue
from queue import Queue, Empty  # Ensure Queue and Empty are imported
import ffmpeg
import random
import subprocess

# Mondrian palette definition
MONDRIAN_PALETTE_BASE = np.array([
    [227, 66, 52],    # Mondrian Red
    [238, 210, 20],   # Mondrian Yellow
    [39, 89, 180],    # Mondrian Blue
    [255, 255, 255],  # White
    [0, 0, 0],        # Black
    [192, 192, 192],  # Light Gray
    [128, 128, 128],  # Medium Gray
    [74, 74, 74],     # Dark Gray
    [181, 181, 181],  # Light Gray
])

MONDRIAN_PALETTE = np.array([
    [245, 66, 52],    # Mondrian Red 1
    [234, 66, 52],    # Mondrian Red 2
    [227, 66, 52],    # Mondrian Red 3 (orig)
    [207, 66, 52],    # Mondrian Red 4
    [193, 66, 52],    # Mondrian Red 5
    [254, 226, 20],   # Mondrian Yellow 1
    [245, 217, 20],   # Mondrian Yellow 2
    [238, 210, 20],   # Mondrian Yellow 3 (orig)
    [231, 203, 20],   # Mondrian Yellow 2
    [222, 197, 20],   # Mondrian Yellow 2
    [39, 89, 196],    # Mondrian Blue
    [39, 89, 187],    # Mondrian Blue
    [39, 89, 180],    # Mondrian Blue (orig)
    [39, 89, 173],    # Mondrian Blue
    [39, 89, 164],    # Mondrian Blue
    [250, 250, 250],  # White 1
    [241, 241, 241],  # White 2
    [234, 234, 234],  # White 3 (orig = 255, 255, 255)
    [227, 227, 227],  # White 4
    [218, 218, 218],  # White 5
    [5, 5, 5],        # Mondrian Black 1
    [14, 14, 14],     # Mondrian Black 2
    [21, 21, 21],     # Mondrian Black 3 (orig = 33, 33, 33)
    [28, 28, 28],     # Mondrian Black 4
    [37, 37, 37],     # Mondrian Black 5
    [58, 65, 65],  # Dark Gray 1
    [67, 70, 70],  # Dark Gray 2
    [74, 74, 74],  # Dark Gray 3 (orig)
    [81, 78, 78],  # Dark Gray 4
    [90, 83, 83],  # Dark Gray 5
    [195, 190, 190],  # Light Gray 1
    [188, 185, 185],  # Light Gray 2
    [181, 181, 181],  # Light Gray 3 (orig)
    [174, 177, 177],  # Light Gray 4
    [165, 172, 172],  # Light Gray 5
    [144, 144, 144],  # Medium Gray 1
    [135, 135, 135],  # Medium Gray 2
    [128, 128, 128],  # Medium Gray 3 (orig)
    [121, 121, 121],  # Medium Gray 4
    [112, 112, 112],  # Medium Gray 5
])

# Finds closest Mondrian color for each pixel
def mondrian_color(pixel):
    distances = np.sqrt(np.sum((MONDRIAN_PALETTE - pixel) ** 2, axis=1))
    return MONDRIAN_PALETTE[np.argmin(distances)]

class MatrixColumn:
    def __init__(self, x, height, speed):
        self.x = x  # X-coordinate of the column
        self.height = height  # Frame height
        self.speed = speed  # Speed of falling (pixels per frame)
        self.head_pos = random.randint(0, height)  # Starting position of the head
        self.characters = []  # List of (y_position, character) tuples
        # Katakana characters (subset for simplicity) + numbers
        self.char_set = [chr(c) for c in range(0x30A2, 0x30FF, 2)] + [str(i) for i in range(10)]
    
    def update(self):
        # Move the head downward
        self.head_pos += self.speed
        if self.head_pos > self.height:
            self.head_pos = 0  # Reset to top when it reaches the bottom
        
        # Add a new character at the head position
        new_char = random.choice(self.char_set)
        self.characters.append((self.head_pos, new_char))
        
        # Remove characters that are too far from the head (more than 20 positions behind)
        self.characters = [(y, char) for y, char in self.characters if self.head_pos - y <= 400]
    
    def draw(self, frame, bright_green, dark_green, font_scale):
        for i, (y, char) in enumerate(self.characters):
            # Calculate distance from head to determine brightness
            distance = self.head_pos - y
            if distance < 0:
                continue  # Skip characters below the head after wrapping
            # Interpolate color from bright green to dark green
            fade_factor = min(distance / 200, 1.0)  # Fade over 200 pixels
            color = tuple(int(bright * (1 - fade_factor) + dark * fade_factor) for bright, dark in zip(bright_green, dark_green))
            # Draw the character
            y_pos = int(y) % self.height
            cv2.putText(frame, char, (self.x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

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

def generate_semi_magic_area(m, n, seed=0):
    """
    Generate a semi-magic m x n grid with numbers 1 to m*n.
    """
    np.random.seed(seed)
    numbers = np.arange(1, m * n + 1)
    np.random.shuffle(numbers)
    grid = numbers.reshape(m, n)
    
    # Attempt to balance row and column sums by sorting
    for i in range(m):
        row = grid[i, :]
        grid[i, :] = np.sort(row)
    for j in range(n):
        col = grid[:, j]
        grid[:, j] = np.sort(col)
    
    return grid

def nearest_prime_vectorized(values, primes):
    """
    Vectorized version to find the nearest prime for an array of values.
    
    Args:
        values (np.ndarray): Array of values to round.
        primes (np.ndarray): Sorted array of prime numbers.
    
    Returns:
        np.ndarray: Array of nearest primes.
    """
    # Reshape values to 1D for processing
    values_flat = values.flatten()
    result = np.zeros_like(values_flat, dtype=np.uint8)
    
    # For each value, find the nearest prime
    for i, value in enumerate(values_flat):
        diffs = np.abs(primes - value)
        idx = np.argmin(diffs)
        result[i] = primes[idx]
    
    return result.reshape(values.shape)

def sieve_of_eratosthenes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return np.array([i for i in range(n + 1) if sieve[i]], dtype=np.uint8)

def precompute_nearest_primes(max_value=255):
    """
    Precompute a lookup table mapping each value from 0 to max_value to the nearest prime.
    
    Args:
        max_value (int): Maximum value (e.g., 255 for RGB).
    
    Returns:
        np.ndarray: Lookup table mapping each value to the nearest prime.
    """
    primes = sieve_of_eratosthenes(max_value)
    lookup = np.zeros(max_value + 1, dtype=np.uint8)
    
    for value in range(max_value + 1):
        min_diff = float('inf')
        nearest = primes[0]
        for prime in primes:
            diff = abs(value - prime)
            if diff < min_diff:
                min_diff = diff
                nearest = prime
            elif diff == min_diff:
                nearest = min(nearest, prime)
            else:
                break
        lookup[value] = nearest
    
    return lookup

def fibonacci_numbers(max_value=255):
    """
    Generate Fibonacci numbers up to max_value.
    
    Args:
        max_value (int): Upper bound (inclusive).
    
    Returns:
        np.ndarray: Array of Fibonacci numbers.
    """
    fib = [0, 1]
    while True:
        next_fib = fib[-1] + fib[-2]
        if next_fib > max_value:
            break
        fib.append(next_fib)
    return np.array(fib, dtype=np.uint8)

def precompute_nearest_fibonacci(max_value=255):
    """
    Precompute a lookup table mapping each value from 0 to max_value to the nearest Fibonacci number.
    
    Args:
        max_value (int): Maximum value (e.g., 255 for RGB).
    
    Returns:
        np.ndarray: Lookup table mapping each value to the nearest Fibonacci number.
    """
    fib_numbers = fibonacci_numbers(max_value)
    lookup = np.zeros(max_value + 1, dtype=np.uint8)
    
    for value in range(max_value + 1):
        min_diff = float('inf')
        nearest = fib_numbers[0]
        for fib in fib_numbers:
            diff = abs(value - fib)
            if diff < min_diff:
                min_diff = diff
                nearest = fib
            elif diff == min_diff:
                nearest = min(nearest, fib)
            else:
                break
        lookup[value] = nearest
    
    return lookup

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

        # Constraints:
        self.block_size = 16  # Customize this based on desired granularity
        self.encoded_constraints = []
        self.decoding_buffer = None
        self.frame_width = 640  # Your frame width
        self.frame_height = 480  # Your frame height
        self.decoding_buffer = None  # State buffer for decoding

        # Frame queue for threaded camera reading
        self.frame_queue = Queue(maxsize=5)  # Use Queue class
        self.camera_thread = None
        self.camera_running = False

        # H.265 encoding attributes
        self.h265_process = None
        self.h265_frame_count = 0

        # Magic Area Generator
        self.magic_areas = []
        for seed in range(16):  # num_magic_areas
            magic_area = generate_semi_magic_area(15, 20, seed)  # 480/32, 640/32
            self.magic_areas.append(magic_area)

        # Precompute nearest prime lookup table
        # self.prime_lookup = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251]
        self.prime_lookup = precompute_nearest_primes(255)

        # Real Fibonacci numbers
        self.fib_lookup = precompute_nearest_fibonacci(255)

        # Matrix Digital Rain
        self.transformer_matrix_columns = None
        self.transformer_matrix_font_scale = None

        # Mondrian 2 Settings
        self.mondrian_regions = None

        # Transformer map
        self.transformer_map = {
            "Dummy": self.transformer_dummy,
            "Matrix Digital Rain": self.transformer_matrix,
            "Incremental Encode": self.transformer_incremental,
            "Vectorwave": self.transformer_vectorwave,
            "Fibonacci Compression": self.transformer_fibonacci,
            "Retro Compression": self.transformer_retro,
            "Intermediate": self.transformer_intermediate,
            "Retro Flashy": self.transformer_retro_flashy,
            "Interlaced": self.transformer_interlace,
            "Cybergrid": self.transformer_cybergrid,
            "MacroBlast": self.transformer_macroblast,
            "Tile-Cycle": self.transformer_tilecycle,
            "H.265 Low Bitrate": self.transformer_h265_lowbitrate,
            "Halftone 3D": self.transformer_halftone_3d,
            "Magic Area": self.transformer_magic_area,
            "Prime RGB": self.transformer_prime_rgb,
            "Fibonacci RGB": self.transformer_fibonacci_rgb,
            "Mondrian": self.transformer_mondrian,
            "Mondrian 2": self.transformer_mondrian_2,
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
        # Close FFmpeg process if switching away from H.265 transformer
        if self.effect_var.get() == "H.265 Low Bitrate" and value != "H.265 Low Bitrate":
            if self.h265_process:
                try:
                    self.h265_process.stdin.close()
                    self.h265_process.wait()
                except:
                    pass
                self.h265_process = None
                self.h265_frame_count = 0
        
        self.last_frame = None
        self.frame_count = 0
        self.current_transformer = self.transformer_map.get(value, self.transformer_dummy)

    def compute_complexity_and_subdivide(self, frame, x, y, w, h, min_size=40):
        """Compute complexity and subdivide into 4 regions with colors, ensuring palette usage."""
        roi = frame[y:y+h, x:x+w]
        if w < min_size or h < min_size:
            # Assign a color based on dominance, considering prior assignments
            dominant = self.get_dominant_color(roi)
            return [(x, y, w, h, dominant)]

        # Count current lines to enforce limits
        if hasattr(self, 'mondrian_h_lines') and hasattr(self, 'mondrian_v_lines'):
            h_count = len(self.mondrian_h_lines)
            v_count = len(self.mondrian_v_lines)
        else:
            h_count = v_count = 0
            self.mondrian_h_lines = set()
            self.mondrian_v_lines = set()
            self.color_counts = {tuple(color): 0 for color in MONDRIAN_PALETTE}  # Track usage

        # Midpoint for potential 4-way split
        mid_x = x + w // 2
        mid_y = y + h // 2

        # Sample four quadrants around the midpoint
        patch_size = min(w, h) // 4
        if patch_size < 4:
            dominant = self.get_dominant_color(roi)
            return [(x, y, w, h, dominant)]

        top = frame[max(y, mid_y - patch_size):mid_y, mid_x - patch_size:mid_x + patch_size]
        bottom = frame[mid_y:min(y + h, mid_y + patch_size), mid_x - patch_size:mid_x + patch_size]
        left = frame[mid_y - patch_size:mid_y + patch_size, max(x, mid_x - patch_size):mid_x]
        right = frame[mid_y - patch_size:mid_y + patch_size, mid_x:min(x + w, mid_x + patch_size)]

        # Compute color variance for each quadrant
        variances = [
            np.var(top.reshape(-1, 3), axis=0).sum() if top.size > 0 else 0,
            np.var(bottom.reshape(-1, 3), axis=0).sum() if bottom.size > 0 else 0,
            np.var(left.reshape(-1, 3), axis=0).sum() if left.size > 0 else 0,
            np.var(right.reshape(-1, 3), axis=0).sum() if right.size > 0 else 0
        ]

        # Complexity check and 4-way split
        total_variance = sum(variances)
        min_variance_per_quadrant = 250
        if total_variance > 1000 and all(v > min_variance_per_quadrant for v in variances) and \
        h_count < 15 and v_count < 20:
            self.mondrian_h_lines.add(mid_y)
            self.mondrian_v_lines.add(mid_x)
            
            # Define sub-regions
            sub_regions = [
                (x, y, w // 2, h // 2),           # Top-left
                (mid_x, y, w - w // 2, h // 2),   # Top-right
                (x, mid_y, w // 2, h - h // 2),   # Bottom-left
                (mid_x, mid_y, w - w // 2, h - h // 2)  # Bottom-right
            ]
            
            # Assign colors to sub-regions
            result = []
            used_colors = set()  # Colors used in this split
            for rx, ry, rw, rh in sub_regions:
                roi_sub = frame[ry:ry+rh, rx:rx+rw]
                dominant = self.get_dominant_color(roi_sub)
                # Avoid duplicates within this split and check neighbors
                neighbors = []
                for prev_x, prev_y, prev_w, prev_h, prev_color in self.mondrian_map or []:
                    if (rx + rw == prev_x and ry < prev_y + prev_h and ry + rh > prev_y) or \
                    (prev_x + prev_w == rx and ry < prev_y + prev_h and ry + rh > prev_y) or \
                    (ry + rh == prev_y and rx < prev_x + prev_w and rx + rw > prev_x) or \
                    (prev_y + prev_h == ry and rx < prev_x + prev_w and rx + rw > prev_x):
                        neighbors.append(tuple(prev_color))
                
                candidate = dominant
                attempts = 0
                while (tuple(candidate) in used_colors or tuple(candidate) in neighbors) and attempts < len(MONDRIAN_PALETTE):
                    candidate_idx = (np.argmax(np.all(MONDRIAN_PALETTE == candidate, axis=1)) + 1) % len(MONDRIAN_PALETTE)
                    candidate = MONDRIAN_PALETTE[candidate_idx]
                    attempts += 1
                if attempts >= len(MONDRIAN_PALETTE):  # Fallback to random unused color
                    available = [c for c in MONDRIAN_PALETTE if tuple(c) not in used_colors and tuple(c) not in neighbors]
                    candidate = available[0] if available else dominant
                used_colors.add(tuple(candidate))
                self.color_counts[tuple(candidate)] = self.color_counts.get(tuple(candidate), 0) + 1
                result.append((rx, ry, rw, rh, candidate))
            return result
        else:
            dominant = self.get_dominant_color(roi)
            self.color_counts[tuple(dominant)] = self.color_counts.get(tuple(dominant), 0) + 1
            return [(x, y, w, h, dominant)]

    def get_dominant_color(self, roi):
        """Find the most dominant color in a region from MONDRIAN_PALETTE."""
        pixels = roi.reshape(-1, 3)
        distances = np.sqrt(np.sum((MONDRIAN_PALETTE - pixels[:, None]) ** 2, axis=2))
        closest_colors = np.argmin(distances, axis=1)
        hist = np.bincount(closest_colors, minlength=len(MONDRIAN_PALETTE))
        return MONDRIAN_PALETTE[np.argmax(hist)]

    def assign_colors(self, regions, dominant_colors):
        """Assign colors to regions, avoiding identical colors for neighbors."""
        assigned_colors = []
        for i, (x, y, w, h) in enumerate(regions):
            neighbors = []
            # Check 4-connectivity for adjacency
            for j, (x2, y2, w2, h2) in enumerate(regions[:i]):
                if (x + w == x2 and y < y2 + h2 and y + h > y2) or \
                (x2 + w2 == x and y < y2 + h2 and y + h > y2) or \
                (y + h == y2 and x < x2 + w2 and x + w > x2) or \
                (y2 + h2 == y and x < x2 + w2 and x + w > x2):
                    neighbors.append(j)
            
            # Avoid neighbor colors
            neighbor_colors = [assigned_colors[n] for n in neighbors if n < len(assigned_colors)]
            candidate_color = dominant_colors[i]
            while candidate_color.tolist() in [c.tolist() for c in neighbor_colors]:
                candidate_idx = (np.argmax(np.all(MONDRIAN_PALETTE == candidate_color, axis=1)) + 1) % len(MONDRIAN_PALETTE)
                candidate_color = MONDRIAN_PALETTE[candidate_idx]
            assigned_colors.append(candidate_color)
        return assigned_colors

    def create_region_palette(self, dominant_color):
        """Create a 5-color palette for a region, excluding black."""
        distances = np.sqrt(np.sum((MONDRIAN_PALETTE - dominant_color) ** 2, axis=1))
        sorted_indices = np.argsort(distances)
        # Pick 5 closest colors, excluding pure black (reserved for lines)
        palette = MONDRIAN_PALETTE[sorted_indices[:6]]  # Take 6 to filter black
        palette = palette[~np.all(palette == [0, 0, 0], axis=1)][:5]  # Exclude black, take 5
        if len(palette) < 5:  # Fallback if not enough colors
            palette = np.vstack([palette, MONDRIAN_PALETTE[:5 - len(palette)]])
        return palette

    def map_pixels(self, roi, palette):
        """Map region pixels to the nearest color in the 5-color palette."""
        pixels = roi.reshape(-1, 3)
        distances = np.sqrt(np.sum((palette - pixels[:, None]) ** 2, axis=2))
        closest_colors = np.argmin(distances, axis=1)
        return palette[closest_colors].reshape(roi.shape)

    def transformer_dummy(self, frame):
        return frame
    
    def transformer_prime_rgb(self, frame):
        # start_total = time.time()
        frame = frame.astype(np.uint8)
        output = self.prime_lookup[frame]
        # print(f"Total time: {(time.time() - start_total)*1000:.1f}ms")
        return output
    
    def transformer_fibonacci_rgb(self, frame):
        """
        Transform an RGB frame by rounding each component to the nearest Fibonacci number.
        
        Args:
            frame (np.ndarray): Input frame (RGB or BGR, uint8).
        
        Returns:
            np.ndarray: Processed frame with each RGB component rounded to the nearest Fibonacci number.
        """
        frame = frame.astype(np.uint8)
        output = self.fib_lookup[frame]
        return output

    def transformer_matrix(self, frame):
        """
        Apply a Matrix digital rain effect overlay on the frame.
        
        Args:
            self: The VideoApp instance.
            frame (np.ndarray): Input frame (BGR, uint8).
        
        Returns:
            np.ndarray: Processed frame with Matrix effect.
        """
        # Ensure frame is in uint8 format
        frame = frame.astype(np.uint8)
        height, width = frame.shape[:2]
        
        # Create a black overlay (semi-transparent)
        overlay = np.zeros_like(frame, dtype=np.uint8)
        overlay[:] = (3, 3, 3)  # Background color: Rich Black (#030303)
        
        # Colors for the characters
        bright_green = (1, 186, 54)  # #36BA01
        dark_green = (41, 133, 0)    # #008529
        
        # Initialize columns if not already done
        if self.transformer_matrix_columns is None:
            column_width = 20  # Pixels between columns
            num_columns = width // column_width
            self.transformer_matrix_columns = [
                MatrixColumn(x * column_width, height, random.randint(5, 15))
                for x in range(num_columns)
            ]
            self.transformer_matrix_font_scale = 0.6  # Adjust font size for readability
        
        # Update and draw each column
        for column in self.transformer_matrix_columns:
            column.update()
            column.draw(overlay, bright_green, dark_green, self.transformer_matrix_font_scale)
        
        # Blend the overlay with the original frame
        alpha = 0.9  # Transparency of the Matrix effect (0 = fully transparent, 1 = fully opaque)
        output = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0.0)
        
        return output
    def transformer_mondrian_2(self, frame):
        """Apply the Mondrian filter using a precomputed color-transform map."""
        h, w, _ = frame.shape
        output_frame = frame.copy()

        # Update the map every 30 frames
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
        if not hasattr(self, 'mondrian_map') or self.mondrian_map is None or self.frame_count % 30 == 0:
            self.mondrian_h_lines = set()
            self.mondrian_v_lines = set()
            self.color_counts = {tuple(color): 0 for color in MONDRIAN_PALETTE}  # Reset color counts
            
            # Step 1: Subdivide and assign colors
            regions = []
            grid_size = 8
            for i in range(0, h, h // grid_size):
                for j in range(0, w, w // grid_size):
                    self.mondrian_map = regions  # Temporarily store for neighbor checks
                    sub_regions = self.compute_complexity_and_subdivide(frame, j, i, w // grid_size, h // grid_size)
                    regions.extend(sub_regions)
            
            # Step 2: Enforce minimum 5 regions per color
            color_usage = self.color_counts.copy()
            min_regions_per_color = 5
            total_regions_needed = len(MONDRIAN_PALETTE) * min_regions_per_color  # 40 * 5 = 200
            if len(regions) < total_regions_needed:
                # Randomly assign remaining regions (won’t exceed 336 due to line limits)
                extra_needed = total_regions_needed - len(regions)
                if extra_needed > 0:
                    available_regions = [(x, y, w, h, c) for x, y, w, h, c in regions if w > 40 and h > 40]
                    for _ in range(min(extra_needed, len(available_regions))):
                        rx, ry, rw, rh, _ = available_regions.pop(0)
                        mid_x, mid_y = rx + rw // 2, ry + rh // 2
                        if len(self.mondrian_h_lines) < 15 and len(self.mondrian_v_lines) < 20:
                            self.mondrian_h_lines.add(mid_y)
                            self.mondrian_v_lines.add(mid_x)
                            sub_regions = [
                                (rx, ry, rw // 2, rh // 2),
                                (mid_x, ry, rw - rw // 2, rh // 2),
                                (rx, mid_y, rw // 2, rh - rh // 2),
                                (mid_x, mid_y, rw - rw // 2, rh - rh // 2)
                            ]
                            for srx, sry, srw, srh in sub_regions:
                                roi_sub = frame[sry:sry+srw, srx:srx+srh]
                                dominant = self.get_dominant_color(roi_sub)
                                regions.append((srx, sry, srw, srh, dominant))
                                color_usage[tuple(dominant)] += 1
            
            # Redistribute colors to meet minimum requirement
            final_regions = []
            for region in regions:
                x, y, w, h, color = region
                if color_usage[tuple(color)] >= min_regions_per_color:
                    final_regions.append(region)
                else:
                    # Find an underused color
                    underused = [c for c, count in color_usage.items() if count < min_regions_per_color]
                    if underused:
                        new_color = underused[0]
                        color_usage[tuple(new_color)] += 1
                        color_usage[tuple(color)] -= 1
                        final_regions.append((x, y, w, h, new_color))
                    else:
                        final_regions.append(region)
            
            # Store the final map with palettes
            self.mondrian_map = [(x, y, w, h, color, self.create_region_palette(color)) 
                                for x, y, w, h, color in final_regions]

        # Step 3: Apply the precomputed map
        for x, y, w, h, _, palette in self.mondrian_map:
            if w < 20 or h < 20:
                continue
            roi = output_frame[y:y+h, x:x+w]
            mapped_roi = self.map_pixels(roi, palette)
            output_frame[y:y+h, x:x+w] = mapped_roi
            
            # Draw black lines
            thickness = max(2, min(6, int((w + h) / 50)))
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 0), thickness=thickness)

        self.frame_count += 1
        return output_frame
    
    def transformer_mondrian(self, frame):
        h, w, _ = frame.shape

        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges to enhance divisions clearly
        edges_dilated = cv2.dilate(edges, np.ones((3,3)), iterations=2)

        # Find contours to identify rectangular regions
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare output frame
        mondrian_frame = np.full_like(frame, 255)  # start with white background

        # Process each region
        for cnt in contours:
            # Approximate contour to rectangles
            x, y, rect_w, rect_h = cv2.boundingRect(cnt)

            # Ignore very small regions (noise)
            if rect_w < 20 or rect_h < 20:
                continue

            # Extract region of interest
            roi = frame[y:y+rect_h, x:x+rect_w]

            # Calculate the average color of region
            avg_color = np.mean(roi.reshape(-1, 3), axis=0).astype(np.uint8)

            # Snap average color to closest Mondrian palette color
            mondrian_region_color = mondrian_color(avg_color)

            # Create subtle gradient within the rectangle
            grad_rect = np.zeros((rect_h, rect_w, 3), dtype=np.uint8)
            for i in range(rect_h):
                gradient_factor = 1 - (i / rect_h) * 0.2  # Slight vertical gradient
                grad_rect[i] = np.clip(mondrian_region_color * gradient_factor, 0, 255)

            # Apply colored rectangle with gradient
            mondrian_frame[y:y+rect_h, x:x+rect_w] = grad_rect

            # Draw black borders (bold outlines)
            cv2.rectangle(mondrian_frame, (x, y), (x+rect_w, y+rect_h), (0,0,0), thickness=3)

        return mondrian_frame
    
    def transformer_incremental(self, frame):
        # Encode the frame
        encoded_data = self.transformer_incremental_encode(frame)
        
        # Decode using the encoded data
        decoded_frame = self.transformer_incremental_decode(encoded_data)
        
        return decoded_frame
    
    def transformer_incremental_encode(self, frame):
        h, w, _ = frame.shape
        blocks_vertical = h // self.block_size
        blocks_horizontal = w // self.block_size

        # Flatten blocks from the frame
        blocks = np.array([
            frame[y * self.block_size:(y + 1) * self.block_size,
                x * self.block_size:(x + 1) * self.block_size].flatten()
            for y in range(blocks_vertical)
            for x in range(blocks_horizontal)
        ], dtype=np.uint8)

        # Compute XOR constraint across all blocks
        xor_constraint = np.bitwise_xor.reduce(blocks, axis=0)

        # Return this XOR constraint as encoded data
        return xor_constraint.tobytes()  # return bytes directly for compactness

    # Decoder function (for incremental reconstruction from constraints):
    def transformer_incremental_decode(self, encoded_data):
        h, w = self.frame_height, self.frame_width  # set these in __init__
        blocks_vertical = h // self.block_size
        blocks_horizontal = w // self.block_size
        num_blocks = blocks_vertical * blocks_horizontal
        block_pixel_count = self.block_size * self.block_size * 3

        # Convert encoded_data from bytes to numpy array
        xor_constraint = np.frombuffer(encoded_data, dtype=np.uint8)

        # Initialize decoding buffer if necessary
        if self.decoding_buffer is None:
            self.decoding_buffer = np.random.randint(
                0, 256,
                size=(num_blocks, block_pixel_count),
                dtype=np.uint8
            )

        # Incremental reconstruction: XOR constraint to each block
        self.decoding_buffer ^= xor_constraint

        # Reconstruct the frame from blocks
        reconstructed_frame = np.zeros((h, w, 3), dtype=np.uint8)
        idx = 0
        for y in range(blocks_vertical):
            for x in range(blocks_horizontal):
                block_data = self.decoding_buffer[idx].reshape(
                    self.block_size, self.block_size, 3
                )
                y_pos = y * self.block_size
                x_pos = x * self.block_size
                reconstructed_frame[
                    y_pos:y_pos+self.block_size,
                    x_pos:x_pos+self.block_size
                ] = block_data
                idx += 1

        return reconstructed_frame
    
    def transformer_h265_lowbitrate(self, frame):
        # Ensure frame is in uint8 format
        frame = frame.astype(np.uint8)
        
        # Initialize FFmpeg process on first frame
        if self.h265_process is None:
            output_video = "output_h265.mp4"
            try:
                # FFmpeg command to encode at 56 kbps with H.265
                self.h265_process = (
                    ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='640x480', r=15)
                    .output(
                        output_video,
                        vcodec='libx265',  # H.265 codec
                        b='56k',           # Target bitrate: 56 kbps
                        r=15,              # Frame rate: 15 FPS
                        pix_fmt='yuv420p', # Pixel format
                        preset='ultrafast',# Fast encoding
                        tune='zerolatency',# Low latency
                        f='mp4',           # Output format
                        movflags='frag_keyframe+empty_moov',  # For streaming
                        loglevel='error'
                    )
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )
            except ffmpeg.Error as e:
                print(f"FFmpeg error: {e.stderr.decode()}")
                self.h265_process = None
                return frame
        
        # Write the frame to FFmpeg's stdin
        try:
            self.h265_process.stdin.write(frame.tobytes())
            self.h265_frame_count += 1
        except Exception as e:
            print(f"Error writing to FFmpeg: {e}")
            self.h265_process.terminate()
            self.h265_process = None
        
        self.frame_count += 1
        return frame  # Return the original frame for display
    
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
    
    def transformer_cybergrid(self, frame, max_shapes=512, grid_spacing=20):
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find and limit contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_shapes]
        
        # Create output canvas
        height, width = frame.shape[:2]
        output = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw vectorized edges in neon cyan
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(output, [approx], -1, (0, 255, 255), 1)
        
        # Add pulsing grid
        grid_color = (255, 0, 255)  # Neon magenta
        pulse = int(128 + 127 * np.sin(self.frame_count * 0.1))  # Flicker effect
        grid_color = (pulse, 0, pulse)
        
        for y in range(0, height, grid_spacing):
            cv2.line(output, (0, y), (width, y), grid_color, 1)
        for x in range(0, width, grid_spacing):
            cv2.line(output, (x, 0), (x, height), grid_color, 1)
        
        # Optional: Shift grid over time
        offset = int(self.frame_count * 2) % grid_spacing
        for y in range(offset, height, grid_spacing):
            cv2.line(output, (0, y), (width, y), (0, 255, 0), 1)  # Green overlay
        
        self.frame_count += 1
        return output
    
    def transformer_macroblast(self, frame, block_size=8, quality=5):
        # Downscale frame to exaggerate blockiness (e.g., 1/4 original size)
        small_width = self.display_width // 4
        small_height = self.display_height // 4
        small_frame = cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_NEAREST)
        
        # Encode and decode with JPEG at low quality to introduce macroblocking
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', small_frame, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # Upscale back to original size with nearest-neighbor to keep blocks sharp
        blocky_frame = cv2.resize(decoded, (self.display_width, self.display_height), interpolation=cv2.INTER_NEAREST)
        
        # Enhance block edges for extra "JPEG-y" effect (fixed version)
        gray = cv2.cvtColor(blocky_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Convert edges to BGR and apply cyan mask directly
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_colored = np.zeros_like(blocky_frame, dtype=np.uint8)
        edges_colored[edges != 0] = [0, 255, 255]  # Cyan where edges exist
        
        # Blend with explicit type handling
        blocky_frame = cv2.addWeighted(blocky_frame, 0.8, edges_colored, 0.2, 0, dtype=cv2.CV_8U)
        
        return blocky_frame
    
    def transformer_tilecycle(self, frame, tile_size=32, max_depth=2):
        # Ensure frame is in uint8 format
        frame = frame.astype(np.uint8)
        height, width = frame.shape[:2]
        
        # Create output canvas
        output = np.zeros_like(frame, dtype=np.uint8)
        
        # Recursive tiling function
        def tile_region(x, y, w, h, depth):
            # Stop if depth exceeded or region too small
            if depth > max_depth or w < tile_size or h < tile_size:
                return
            
            # Ensure region is within bounds
            if x < 0 or y < 0 or x + w > width or y + h > height:
                return
            
            # Extract the region
            region = frame[y:y+h, x:x+w]
            if region.shape[0] == 0 or region.shape[1] == 0:
                return
            
            # Divide into 4 quadrants
            half_w, half_h = w // 2, h // 2
            if half_w < tile_size or half_h < tile_size:
                return
            
            quadrants = [
                (x, y, half_w, half_h),                    # Top-left
                (x + half_w, y, w - half_w, half_h),      # Top-right
                (x, y + half_h, half_w, h - half_h),      # Bottom-left
                (x + half_w, y + half_h, w - half_w, h - half_h)  # Bottom-right
            ]
            
            # Create a tile by sampling a central patch (faster than resizing quadrants)
            center_x, center_y = x + w // 4, y + h // 4
            tile_w, tile_h = min(tile_size, w // 2), min(tile_size, h // 2)
            tile = frame[center_y:center_y+tile_h, center_x:center_x+tile_w]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
            
            # Use the tile to approximate each quadrant with a bitwise operation
            operation = self.frame_count % 3  # Cycle through operations
            for qx, qy, qw, qh in quadrants:
                if qw < tile_size or qh < tile_size:
                    continue
                if qx + qw > width or qy + qh > height:
                    continue
                
                # Precompute the tiled region
                tiled_region = np.tile(tile, (qh // tile_size + 1, qw // tile_size + 1, 1))
                tiled_region = tiled_region[:qh, :qw]  # Crop to fit
                
                # Apply bitwise operation directly to the region
                region_patch = frame[qy:qy+qh, qx:qx+qw]
                if operation == 0:
                    tiled_region = cv2.bitwise_xor(region_patch, tiled_region)
                elif operation == 1:
                    tiled_region = cv2.bitwise_or(region_patch, tiled_region)
                else:
                    tiled_region = cv2.bitwise_and(region_patch, tiled_region)
                
                # Simple alpha blending instead of seamlessClone
                alpha = 0.5  # Adjust for desired blending strength
                output[qy:qy+qh, qx:qx+qw] = cv2.addWeighted(
                    output[qy:qy+qh, qx:qx+qw], 1 - alpha, tiled_region, alpha, 0
                )
                
                # Recursively tile this quadrant
                tile_region(qx, qy, qw, qh, depth + 1)
        
        # Start tiling from the whole frame
        tile_region(0, 0, width, height, 1)
        
        self.frame_count += 1
        return output

    def transformer_halftone_3d(self, frame, cell_size=32, palette=None):
        start_total = time.time()
        
        # Ensure frame is in uint8 format
        frame = frame.astype(np.uint8)
        height, width = frame.shape[:2]
        
        # Upload frame to GPU
        start_upload = time.time()
        frame_gpu = cv2.cuda_GpuMat()
        frame_gpu.upload(frame)
        print(f"GPU upload: {(time.time() - start_upload)*1000:.1f}ms")
        
        # Create a grid of dot centers (CPU)
        start_grid = time.time()
        grid_y, grid_x = np.mgrid[0:height:cell_size, 0:width:cell_size]
        centers = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        print(f"Grid setup: {(time.time() - start_grid)*1000:.1f}ms")
        
        # Precompute a shaded dot template (CPU)
        start_template = time.time()
        dot_size = cell_size
        dot_template = np.zeros((dot_size, dot_size), dtype=np.uint8)
        center = dot_size // 2
        max_radius = dot_size // 2
        for y in range(dot_size):
            for x in range(dot_size):
                dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
                if dist <= max_radius:
                    intensity = int(255 * (1 - dist / max_radius))
                    dot_template[y, x] = intensity
        dot_template_gpu = cv2.cuda_GpuMat()
        dot_template_gpu.upload(dot_template)
        print(f"Template creation: {(time.time() - start_template)*1000:.1f}ms")
        
        # Compute brightness for all cells on GPU
        start_brightness = time.time()
        gray_gpu = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2GRAY)
        num_cells_y, num_cells_x = grid_y.shape
        brightness = np.zeros((num_cells_y, num_cells_x), dtype=np.float32)
        gray = gray_gpu.download()
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                y, x = grid_y[i, j], grid_x[i, j]
                cell = gray[y:y+cell_size, x:x+cell_size]
                if cell.shape[0] > 0 and cell.shape[1] > 0:
                    brightness[i, j] = np.mean(cell) / 255.0
        radii = (brightness * (cell_size / 2)).astype(np.int32)
        print(f"Brightness calculation: {(time.time() - start_brightness)*1000:.1f}ms")
        
        # Initialize output canvas on GPU
        output_gpu = cv2.cuda_GpuMat(height, width, cv2.CV_8UC1, 0)
        
        # Draw dots on GPU
        start_dots = time.time()
        for radius in range(1, max_radius + 1):
            mask = radii == radius
            if not np.any(mask):
                continue
            centers_for_radius = centers[mask.ravel()]
            scaled_radius = min(radius, max_radius)
            scaled_template_gpu = cv2.cuda.resize(dot_template_gpu, (2 * scaled_radius, 2 * scaled_radius))
            scaled_template = scaled_template_gpu.download()
            for x, y in centers_for_radius:
                top_left_x = x + (cell_size - 2 * scaled_radius) // 2
                top_left_y = y + (cell_size - 2 * scaled_radius) // 2
                bottom_right_x = top_left_x + 2 * scaled_radius
                bottom_right_y = top_left_y + 2 * scaled_radius
                if top_left_x < 0 or top_left_y < 0 or bottom_right_x > width or bottom_right_y > height:
                    continue
                region_gpu = output_gpu.rowRange(top_left_y, bottom_right_y).colRange(top_left_x, bottom_right_x)
                if region_gpu.size() == scaled_template_gpu.size():
                    scaled_template_gpu.copyTo(region_gpu, cv2.cuda.max(region_gpu, scaled_template_gpu))
        print(f"Dot drawing: {(time.time() - start_dots)*1000:.1f}ms")
        
        # Convert to BGR on GPU
        start_convert = time.time()
        output_bgr_gpu = cv2.cuda.cvtColor(output_gpu, cv2.COLOR_GRAY2BGR)
        print(f"Color conversion: {(time.time() - start_convert)*1000:.1f}ms")
        
        # Download output
        start_download = time.time()
        output = output_bgr_gpu.download()
        print(f"GPU download: {(time.time() - start_download)*1000:.1f}ms")
        
        # Apply color palette on CPU (GPU palette mapping is complex)
        start_palette = time.time()
        if palette is not None:
            palette = np.array(palette, dtype=np.uint8)
            mask = output.sum(axis=2) > 0
            output_reshaped = output[mask].reshape(-1, 3)
            if len(output_reshaped) > 0:
                distances = np.linalg.norm(output_reshaped[:, np.newaxis, :] - palette[np.newaxis, :, :], axis=2)
                nearest = np.argmin(distances, axis=1)
                output[mask] = palette[nearest]
        print(f"Palette mapping: {(time.time() - start_palette)*1000:.1f}ms")
        
        print(f"Total time: {(time.time() - start_total)*1000:.1f}ms")
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
        frame = cv2.resize(frame, (28, 24))

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
    
    def transformer_magic_area(self, frame, cell_size=32, num_magic_areas=16, operations=None):
        """
        Apply a magic area effect to a video frame.
        
        Args:
            frame (np.ndarray): Input frame (RGB or BGR, uint8).
            cell_size (int): Size of each grid cell (e.g., 32 for 32x32 cells).
            num_magic_areas (int): Number of predefined magic areas to choose from.
            operations (list): List of operations to apply (e.g., ['invert', 'mirror']).
        
        Returns:
            np.ndarray: Processed frame with magic area effect.
        """
        start_total = time.time()
        
        # Ensure frame is in uint8 format
        frame = frame.astype(np.uint8)
        height, width = frame.shape[:2]
        
        # Compute grid dimensions
        grid_height = height // cell_size  # 480 / 32 = 15
        grid_width = width // cell_size    # 640 / 32 = 20
        
        # Convert to grayscale and compute cell intensities
        start_grayscale = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cell_intensities = np.zeros((grid_height, grid_width), dtype=np.float32)
        for i in range(grid_height):
            for j in range(grid_width):
                y, x = i * cell_size, j * cell_size
                cell = gray[y:y+cell_size, x:x+cell_size]
                if cell.shape[0] > 0 and cell.shape[1] > 0:
                    cell_intensities[i, j] = np.mean(cell)
        print(f"Grayscale and intensities: {(time.time() - start_grayscale)*1000:.1f}ms")
        
        # Generate predefined magic areas
        start_magic = time.time()
        magic_areas = []
        for seed in range(num_magic_areas):
            magic_area = generate_semi_magic_area(grid_height, grid_width, seed)
            magic_areas.append(magic_area)
        print(f"Magic areas generation: {(time.time() - start_magic)*1000:.1f}ms")
        
        # Find the closest magic area
        start_match = time.time()
        cell_intensities_flat = cell_intensities.flatten()
        cell_intensities_sorted = np.sort(cell_intensities_flat)
        best_match_idx = 0
        min_diff = float('inf')
        for idx, magic_area in enumerate(magic_areas):
            magic_flat = magic_area.flatten()
            magic_sorted = np.sort(magic_flat)
            diff = np.sum(np.abs(cell_intensities_sorted - magic_sorted))
            if diff < min_diff:
                min_diff = diff
                best_match_idx = idx
        selected_magic_area = magic_areas[best_match_idx].copy()
        print(f"Magic area matching: {(time.time() - start_match)*1000:.1f}ms")
        
        # Apply operations
        start_operations = time.time()
        if operations is None:
            operations = []
        for op in operations:
            if op == 'invert':
                selected_magic_area = (grid_height * grid_width + 1) - selected_magic_area
            elif op == 'multiply':
                selected_magic_area = (selected_magic_area * selected_magic_area) % (grid_height * grid_width + 1)
            elif op == 'mirror':
                selected_magic_area = selected_magic_area[:, ::-1]
            elif op == 'flip':
                selected_magic_area = selected_magic_area[::-1, :]
            elif op == 'rotate180':
                selected_magic_area = selected_magic_area[::-1, ::-1]
        print(f"Operations: {(time.time() - start_operations)*1000:.1f}ms")
        
        # Normalize magic area values to 0-255 for rendering
        start_normalize = time.time()
        magic_min, magic_max = selected_magic_area.min(), selected_magic_area.max()
        if magic_max > magic_min:
            selected_magic_area = (selected_magic_area - magic_min) * 255 / (magic_max - magic_min)
        selected_magic_area = selected_magic_area.astype(np.uint8)
        print(f"Normalization: {(time.time() - start_normalize)*1000:.1f}ms")
        
        # Tile the magic area across the frame
        start_tiling = time.time()
        output = np.zeros((height, width), dtype=np.uint8)
        for i in range(grid_height):
            for j in range(grid_width):
                y, x = i * cell_size, j * cell_size
                intensity = selected_magic_area[i, j]
                output[y:y+cell_size, x:x+cell_size] = intensity
        print(f"Tiling: {(time.time() - start_tiling)*1000:.1f}ms")
        
        # Convert to BGR for display
        start_convert = time.time()
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        print(f"Color conversion: {(time.time() - start_convert)*1000:.1f}ms")
        
        print(f"Total time: {(time.time() - start_total)*1000:.1f}ms")
        return output

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
        elif effect == "H.265 Low Bitrate":
            data_rate = 56  # 56 kbps as specified
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
        # Existing cleanup code...
        if self.h265_process:
            try:
                self.h265_process.stdin.close()
                self.h265_process.wait()
            except:
                pass
        self.camera_running = False
        if hasattr(self, 'camera_thread') and self.camera_thread:  # Check if attribute exists
            self.camera_thread.join()
        if self.cap:
            self.cap.release()

def main():
    root = Tk()
    app = VideoApp(root)
    fourcc = cv2.VideoWriter_fourcc(*'hevc')  # or 'h265'
    out = cv2.VideoWriter('test_h265.mp4', fourcc, 30, (640, 480))
    if not out.isOpened():
        print("OpenCV does not support H.265 encoding.")
    else:
        print("OpenCV supports H.265 encoding!")
        out.release()
    root.mainloop()

if __name__ == "__main__":
    main()