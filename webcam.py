from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
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
from sympy import factorint, primerange
import itertools

prime_list = list(primerange(2, 1000))
prime_index = {p: i for i, p in enumerate(prime_list)}
factor_lookup = {}

for n in range(8161):
    factors = factorint(n)
    factor_lookup[n] = {prime_index[p]: e for p, e in factors.items() if p in prime_index}

def factor_segment(segment_data, segment_size, prime_index):
    N = int(np.sum(segment_data))  # Sum is 0 to 8160

    return factor_lookup[N]

# def factor_segment(segment_data, segment_size, prime_index):
#     N = 0

#     for i, byte in enumerate(segment_data):
#         N += byte * (256 ** (segment_size - 1 - i))

#     factors = factorint(N)
#     segment_factors = {}

#     for prime, exponent in factors.items():
#         if prime in prime_index:
#             prime_idx = prime_index[prime]
#             segment_factors[prime_idx] = exponent

#     return segment_factors

def process_segment_task(task):
    """
    Process a single task for prime factor row encoding.
    Args:
        task: Tuple of (segment_data, segment_size, prime_index, channel, row, seg_idx)
    Returns:
        Tuple of (segment_factors, channel, row, seg_idx)
    """
    segment_data, segment_size, prime_index, channel, row, seg_idx = task

    return (factor_segment(segment_data, segment_size, prime_index), channel, row, seg_idx)

def get_pythagorean_numbers():
    """
    Returns a sorted list of unique numbers that appear in Pythagorean triples with c <= 255.
    """
    numbers = set([0])  # Include 0 for the trivial triple (0, 0, 0)
    
    # Generate primitive triples
    for m in range(2, 16):
        for n in range(1, m):
            if (m + n) % 2 == 1 and np.gcd(m, n) == 1:
                a = m * m - n * n
                b = 2 * m * n
                c = m * m + n * n
                if c > 255:
                    continue
                # Generate multiples
                k = 1
                while True:
                    ka, kb, kc = k * a, k * b, k * c
                    if kc > 255:
                        break
                    numbers.add(ka)
                    numbers.add(kb)
                    numbers.add(kc)
                    k += 1
    
    return sorted(numbers)

def generate_pythagorean_mappings():
    """
    Generates all Pythagorean triples with c <= 255, including all permutations.
    Returns:
    - mappings: Array of triples (a, b, c) with all permutations.
    - triple_map: Dictionary mapping (a, b, c) to its index.
    """
    triples = [(0, 0, 0)]
    for m in range(2, 16):
        for n in range(1, m):
            if (m + n) % 2 == 1 and np.gcd(m, n) == 1:
                a = m * m - n * n
                b = 2 * m * n
                c = m * m + n * n
                if c > 255:
                    continue
                k = 1
                while True:
                    ka, kb, kc = k * a, k * b, k * c
                    if kc > 255:
                        break
                    triples.append((ka, kb, kc))
                    k += 1
    # Generate all permutations
    mappings = []
    triple_map = {}
    index = 0
    for triple in triples:
        for perm in itertools.permutations(triple):
            mappings.append(perm)
            triple_map[perm] = index
            index += 1
    
    return np.array(mappings, dtype=np.uint8), triple_map

def generate_pythagorean_triples(max_value):
    triples = []

    for a in range(1, max_value + 1):
        for b in range(a, max_value + 1):
            c = int((a**2 + b**2)**0.5)

            if c <= max_value and a**2 + b**2 == c**2:
                triples.append((a, b, c))
    
    return triples

# def generate_pythagorean_triples():
#     """
#     Generates all Pythagorean triples with c <= 255, including all permutations.
#     Returns a list of triples (a, b, c).
#     """
#     triples = [(0, 0, 0)]  # Include the trivial triple
    
#     # Generate primitive triples using m, n
#     for m in range(2, 16):  # m <= sqrt(255) ~ 15.97
#         for n in range(1, m):
#             if (m + n) % 2 == 1 and np.gcd(m, n) == 1:  # m, n coprime, not both odd
#                 a = m * m - n * n
#                 b = 2 * m * n
#                 c = m * m + n * n
#                 if c > 255:
#                     continue
#                 # Generate multiples
#                 k = 1
#                 while True:
#                     ka, kb, kc = k * a, k * b, k * c
#                     if kc > 255:
#                         break
#                     # Add all permutations of the triple
#                     for perm in itertools.permutations([ka, kb, kc]):
#                         triples.append(perm)
#                     k += 1
    
#     return np.array(triples, dtype=np.uint8)

# Helper: Compute complexity and subdivide deterministically
def compute_complexity_and_subdivide(frame, x, y, w, h, threshold=5000, min_size=40):
    roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 50, 150)
    complexity = np.sum(edges)

    if complexity > threshold and w > min_size and h > min_size:
        # Deterministic split: always split at 40% and 60% to avoid randomness
        if w > h:  # Split vertically
            split1 = x + int(w * 0.4)
            split2 = x + int(w * 0.6)
            return [
                (x, y, split1 - x, h),
                (split1, y, split2 - split1, h),
                (split2, y, x + w - split2, h)
            ]
        else:  # Split horizontally
            split1 = y + int(h * 0.4)
            split2 = y + int(h * 0.6)
            return [
                (x, y, w, split1 - y),
                (x, split1, w, split2 - split1),
                (x, split2, w, y + h - split2)
            ]
    return [(x, y, w, h)]

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
        self.frame_w = 640  # Your frame width
        self.frame_h = 480  # Your frame height
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
        self.mondrian_map = None

        # Diagonal, Vertical, and Horizontal Vision
        # self.frame_width = 640
        # self.frame_height = 480
        self.initialize_reference_patterns()

        # Vertical Bitfield Pattern Prototype
        # Vertical Bitfield Pattern
        self.initialize_vertical_bitfield_reference_patterns()

        # Row / Column Persistent frame.
        self.persistent_frame_col = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)  # Initialize to black
        self.persistent_frame_col_current_row = 0  # Start at row 0

        self.persistent_frame_row = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)  # Initialize to black
        self.persistent_frame_row_current_row = 0  # Start at row 0

        # Transformer map
        self.transformer_map = {
            "Dummy": self.transformer_dummy,
            "Prime Factor Row": self.transformer_prime_factor_row,
            "Prime Factor Signature": self.transformer_prime_factor_signature,
            "Pythagorean Triple": self.transformer_pythagorean_triple,
            "Pythagorean Snap": self.transformer_pythagorean_snap,
            "Delta RGB Pix": self.transformer_delta_rgb_pix,
            "Even/Odd Color": self.transformer_even_odd_color,
            "Even/Odd Spatial": self.transformer_even_odd_spatial,
            "RGB Strobe": self.transformer_rgb_strobe,
            "RGB Even-Odd Strobe": self.transformer_rgb_even_odd_strobe,
            "RGB Matrix Strobe": self.transformer_rgb_matrix_strobe,
            "VBPV Prototype": self.transformer_vertical_bitfield_pattern_prototype,
            "VBPatternVision": self.transformer_vertical_bitfield_pattern,
            "DiagonalVision": self.transformer_diagonal_vision,
            "VerticalVision": self.transformer_vertical_bitfield,
            "Compressed Frame Prototype": self.transformer_compressed_frame_prototype,
            "Compressed Frame": self.transformer_compressed_frame,
            "Matrix Digital Rain": self.transformer_matrix,
            "Incremental Encode": self.transformer_incremental,
            "Bit Depth Strobe": self.transformer_bit_depth_strobe,
            "Bit Depth Resolve (prototype)": self.transformer_bit_depth_resolve_prototype,
            "Bit Depth Resolve": self.transformer_bit_depth_resolve,
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
            "Gradient Tiles": self.transformer_gradient_tiles,
            "Mondrian": self.transformer_mondrian,
            "Mondrian 2": self.transformer_mondrian_2,
            "Mondrian Next": self.transformer_mondrian_next,
            "Complexity Test": self.transformer_complexity_test,
            "Metrics": self.transformer_metrics,
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

        # RGB Even/Odd Strobe attributes
        self.rgb_even_odd_strobe_cycle = 0
        self.rgb_even_odd_strobe_cycle_order = [
            (0, True),  # R odd
            (1, False), # G even
            (2, True),  # B odd
            (0, False), # R even
            (1, True),  # G odd
            (2, False)  # B even
        ]
        self.encode_rgb_even_odd = None
        self.persistent_rgb_even_odd_strobe = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8) # Holds progressively refined frame

        self.persistent_rgb_matrix_strobe = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)

        # RGB Strobe attributes
        self.rgb_strobe_cycle = 0
        self.rgb_strobe_cycle_order = [
            (0,),  # R
            (1,),  # G
            (2,)   # B
        ]
        self.rgb_even_odd_strobe_cycle = 0
        self.rgb_strobe_persistent_frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)

        # Even/Odd attributes
        self.even_odd_color_cycle = True
        self.even_odd_color_persistent_frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)

        # Even/Odd Spatial attributes
        self.even_odd_spatial_cycle = 0
        self.even_odd_spatial_persistent_frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)

        # Delta RGB Pix attributes
        self.delta_rgb_pix_persistent_frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)

        # Prime Factor Row attributes
        self.prime_list = list(primerange(2, 1000))

        self.persistent_prime_factor = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        self.num_workers = mp.cpu_count()
        self.factor_lookup = {}
        for n in range(8161):
            factors = factorint(n)
            self.factor_lookup[n] = {prime_index[p]: e for p, e in factors.items() if p in prime_index}

        print(f"Using {self.num_workers} workers for Prime Factor Row transformer.")

        # Pythagorean Triple attributes
        self.pythagorean_triples = generate_pythagorean_triples(255)
        self.pythagorean_triples_legend, self.pythagorean_triples_map = generate_pythagorean_mappings()
        self.pythagorean_triples_persistent_frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        # self.pythagorean_triples_set = generate_pythagorean_triples()

        # Pythagorean Snap attributes
        self.pythagorean_snap_set = get_pythagorean_numbers()
        self.pythagorean_snap_persistent_frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)

        # Start Update Loop
        self.update()

    def init_camera(self):
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    def find_complexity_points(self, frame, num_points=375):
        """Identify points of high complexity across the image."""
        h, w, _ = frame.shape
        complexity_map = np.zeros((h, w), dtype=np.float32)

        # Compute complexity for the entire image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Compute entropy in a sliding window
        entropy_map = np.zeros((h, w), dtype=np.float32)
        window_size = 16
        for i in range(0, h - window_size, window_size // 2):
            for j in range(0, w - window_size, window_size // 2):
                roi = gray[i:i+window_size, j:j+window_size]
                if roi.size == 0:
                    continue
                hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
                hist = hist / hist.sum()
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                entropy_map[i:i+window_size, j:j+window_size] = entropy

        # Combine gradient and entropy
        complexity_map = grad_mag * entropy_map

        # Find top 375 points
        points = []
        complexity_flat = complexity_map.flatten()
        indices = np.argsort(-complexity_flat)[:num_points]  # Top values
        for idx in indices:
            y = idx // w
            x = idx % w
            points.append((x, y, complexity_map[y, x], grad_x[y, x], grad_y[y, x]))

        return sorted(points, key=lambda p: p[2], reverse=True)  # Sort by complexity

    def _assign_color(self, frame, x, y, w, h, exclude_colors=None):
        """Assign a color to a region, ensuring no adjacent duplicates and balanced distribution."""
        if exclude_colors is None:
            exclude_colors = set()

        # Check neighbor colors
        neighbors = set()
        for prev_x, prev_y, prev_w, prev_h, prev_color in self.mondrian_map or []:
            if (x + w == prev_x and y < prev_y + prev_h and y + h > prev_y) or \
            (prev_x + prev_w == x and y < prev_y + prev_h and y + h > prev_y) or \
            (y + h == prev_y and x < prev_x + prev_w and x + w > prev_x) or \
            (prev_y + prev_h == y and x < prev_x + prev_w and x + w > prev_x):
                neighbors.add(tuple(prev_color))

        # Combine excluded colors (from this split and neighbors)
        forbidden_colors = exclude_colors | neighbors

        # Prefer underused colors
        min_regions_per_color = 5
        underused = [color for color, count in self.color_counts.items() if count < min_regions_per_color]
        available_colors = [
            color for color in MONDRIAN_PALETTE 
            if tuple(color) not in forbidden_colors and tuple(color) != (0, 0, 0)  # Exclude black
        ]

        # Prioritize underused colors if available, otherwise use any available color
        if underused:
            candidates = [color for color in underused if tuple(color) in {tuple(c) for c in available_colors}]
            if not candidates:
                candidates = available_colors
        else:
            candidates = available_colors

        # If no candidates, relax the neighbor constraint but keep split constraint
        if not candidates:
            candidates = [
                color for color in MONDRIAN_PALETTE 
                if tuple(color) not in exclude_colors and tuple(color) != (0, 0, 0)
            ]

        # Choose the least used candidate
        if candidates:
            color_counts = [(self.color_counts.get(tuple(c), 0), c) for c in candidates]
            color = min(color_counts, key=lambda x: x[0])[1]
        else:
            # Fallback: use the least used color overall, ignoring constraints
            color_counts = [(count, color) for color, count in self.color_counts.items() 
                        if tuple(color) != (0, 0, 0)]
            color = min(color_counts, key=lambda x: x[0])[1]

        self.color_counts[tuple(color)] = self.color_counts.get(tuple(color), 0) + 1
        return color
        
    def _assign_tricolor(self, x, y, w, h, exclude_colors=None):
        """Assign one of three Mondrian colors, ensuring no adjacent duplicates."""
        if exclude_colors is None:
            exclude_colors = set()

        tricolor = [
            (227, 66, 52),   # Mondrian Red
            (238, 210, 20),  # Mondrian Yellow
            (39, 89, 180)    # Mondrian Blue
        ]

        neighbors = set()
        for prev_x, prev_y, prev_w, prev_h, prev_color, _ in self.mondrian_map or []:
            if (x + w == prev_x and y < prev_y + prev_h and y + h > prev_y) or \
            (prev_x + prev_w == x and y < prev_y + prev_h and y + h > prev_y) or \
            (y + h == prev_y and x < prev_x + prev_w and x + w > prev_x) or \
            (prev_y + prev_h == y and x < prev_x + prev_w and x + w > prev_x):
                neighbors.add(tuple(prev_color))

        forbidden_colors = exclude_colors | neighbors
        available_colors = [color for color in tricolor if tuple(color) not in forbidden_colors]

        if not available_colors:
            available_colors = [color for color in tricolor if tuple(color) not in exclude_colors]

        if not available_colors:
            color_counts = [(count, color) for color, count in self.color_counts.items()]
            color = min(color_counts, key=lambda x: x[0])[1]
        else:
            color_counts = [(self.color_counts.get(tuple(c), 0), c) for c in available_colors]
            color = min(color_counts, key=lambda x: x[0])[1]

        self.color_counts[tuple(color)] = self.color_counts.get(tuple(color), 0) + 1
        return color
    
    def grow_lines(self, frame, points):
        """Grow horizontal or vertical lines from complexity points to form rectangles."""
        h, w, _ = frame.shape
        h_lines = {0, h}  # Include image boundaries
        v_lines = {0, w}

        # Process each point
        for x, y, complexity, grad_x, grad_y in points:
            # Decide direction based on gradient
            if abs(grad_x) > abs(grad_y):
                # Vertical line (gradient is more horizontal, so split vertically)
                line_y = y
                if line_y in h_lines:
                    continue
                h_lines.add(line_y)

                # Find nearest boundaries
                left = max([vx for vx in v_lines if vx <= x], default=0)
                right = min([vx for vx in v_lines if vx >= x], default=w)

                # Extend if complexity is high
                if complexity > 150:  # Threshold for continuing past boundary
                    if left > 0:
                        v_lines.add(left)
                    if right < w:
                        v_lines.add(right)
            else:
                # Horizontal line
                line_x = x
                if line_x in v_lines:
                    continue
                v_lines.add(line_x)

                top = max([hy for hy in h_lines if hy <= y], default=0)
                bottom = min([hy for hy in h_lines if hy >= y], default=h)

                if complexity > 150:
                    if top > 0:
                        h_lines.add(top)
                    if bottom < h:
                        h_lines.add(bottom)

        # Convert lines to rectangles
        h_lines = sorted(h_lines)
        v_lines = sorted(v_lines)
        rectangles = []
        for i in range(len(h_lines) - 1):
            for j in range(len(v_lines) - 1):
                x = v_lines[j]
                y = h_lines[i]
                w = v_lines[j + 1] - x
                h = h_lines[i + 1] - y
                if w > 0 and h > 0:
                    rectangles.append((x, y, w, h))

        return rectangles

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

    # def get_dominant_color(self, roi):
    #     pixels = roi.reshape(-1, 3)
    #     distances = np.sqrt(np.sum((MONDRIAN_PALETTE - pixels[:, None]) ** 2, axis=2))
    #     closest_colors = np.argmin(distances, axis=1)
    #     hist = np.bincount(closest_colors, minlength=len(MONDRIAN_PALETTE))

    #     return MONDRIAN_PALETTE[np.argmax(hist)]

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
    
    def transformer_persistent_row(self, frame):
        """
        Update the persistent frame with a single row from the input frame.
        
        Args:
            frame: NumPy array of shape (480, 640, 3) with uint8 values (RGB).
        
        Returns:
            updated_frame: NumPy array of shape (480, 640, 3) with the updated persistent frame.
        """
        # Ensure the frame is the correct size
        assert frame.shape == (self.frame_h, self.frame_w, 3), "Frame must be 480x640x3"
        assert frame.dtype == np.uint8, "Frame must be uint8"

        # Update the current row in the persistent frame
        self.persistent_frame[self.current_row, :, :] = frame[self.current_row, :, :]

        # Increment the row index (wrap around at 480)
        self.current_row = (self.current_row + 1) % self.frame_h

        # Return a copy of the updated frame
        return self.persistent_frame.copy()

    def get_update_data(self, frame):
        """
        Simulate the data sent for the current update.
        
        Args:
            frame: NumPy array of shape (480, 640, 3) with uint8 values (RGB).
        
        Returns:
            tuple: (row_index, row_data)
                - row_index: Integer (9 bits).
                - row_data: Array of shape (640, 3) with uint8 values.
        """
        row_index = self.current_row
        row_data = frame[row_index, :, :].copy()
        return row_index, row_data

    # def reset(self):
    #     """
    #     Reset the persistent frame and row index.
    #     """
    #     self.persistent_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    #     self.current_row = 0

    def transformer_peristent_col(self, frame):
        """
        Update the persistent frame with a single column from the input frame.
        
        Args:
            frame: NumPy array of shape (480, 640, 3) with uint8 values (RGB).
        
        Returns:
            updated_frame: NumPy array of shape (480, 640, 3) with the updated persistent frame.
        """
        # Ensure the frame is the correct size
        assert frame.shape == (self.frame_h, self.frame_w, 3), "Frame must be 480x640x3"
        assert frame.dtype == np.uint8, "Frame must be uint8"

        # Update the current column in the persistent frame
        self.persistent_frame_col[:, self.persistent_col_current_col, :] = frame[:, self.persistent_col_current_col, :]

        # Create a copy for annotation
        annotated_frame = self.persistent_frame_col.copy()

        # Overlay metrics
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White text
        thickness = 1
        line_spacing = 20
        start_x, start_y = 10, 20  # Top-left corner

        metrics = [
            f"Current Column: {self.persistent_frame_col_current_col}",
            f"Data Sent: {11530} bits",
            f"Compression Ratio: {7372800 / 11530:.1f}:1"
        ]

        for i, metric in enumerate(metrics):
            y = start_y + i * line_spacing
            (text_w, text_h), _ = cv2.getTextSize(metric, font, font_scale, thickness)
            cv2.rectangle(
                annotated_frame,
                (start_x, y - text_h - 2),
                (start_x + text_w, y + 2),
                (0, 0, 0),  # Black background
                -1
            )
            cv2.putText(
                annotated_frame,
                metric,
                (start_x, y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )

        # Increment the column index (wrap around at 640)
        self.persistent_col_current_col = (self.persistent_col_current_col + 1) % self.frame_w

        return annotated_frame
    
    def helper_persistent_col_update_data(self, frame):
        """
        Simulate the data sent for the current update.
        
        Args:
            frame: NumPy array of shape (480, 640, 3) with uint8 values (RGB).
        
        Returns:
            tuple: (col_index, col_data)
                - col_index: Integer (10 bits).
                - col_data: Array of shape (480, 3) with uint8 values.
        """
        col_index = self.persistent_frame_col_current_col
        col_data = frame[:, col_index, :].copy()
        return col_index, col_data
    
    # def reset(self):
    #     """
    #     Reset the persistent frame and column index.
    #     """
    #     self.persistent_frame_col = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    #     self.persistent_frame_col_current_col = 0

    def transformer_delta_rgb_pix(self, frame):
        """
        Transforms the frame by transmitting only the delta of changed RGB values.
        1. Encodes the frame into a mask and new values.
        2. Decodes the mask and new values into the persistent frame.
        """
        # Encode the frame
        mask, new_values = self.encode_delta_rgb_pix(frame)

        mask, new_values = self.denoise_delta_rgb_pix(
            mask, new_values, frame, self.delta_rgb_pix_persistent_frame, threshold=15
        )

        # Decode and return the updated frame
        return self.decode_delta_rgb_pix(mask, new_values)
    
    def encode_delta_rgb_pix(self, frame):
        """
        Encodes the frame by identifying changed RGB values compared to the persistent frame.
        Returns:
        - mask: 2D array of 3-bit values (0-7) indicating which channels changed.
        - new_values: 3D array of new RGB values for the changed channels.
        """
        frame = frame.astype(np.uint8)
        
        # Compare the current frame with the persistent frame
        changed = frame != self.delta_rgb_pix_persistent_frame  # Shape: (height, width, 3), dtype: bool
        
        # Create the 3-bit mask
        # Bit 2 (4): R changed, Bit 1 (2): G changed, Bit 0 (1): B changed
        mask = (changed[:, :, 0].astype(np.uint8) * 4 +  # R channel
                changed[:, :, 1].astype(np.uint8) * 2 +  # G channel
                changed[:, :, 2].astype(np.uint8) * 1)   # B channel
        # Shape: (height, width), dtype: uint8, values 0-7
        
        # Create the new values array (only for changed channels)
        new_values = frame.copy()  # Shape: (height, width, 3), dtype: uint8
        # For channels that didn't change, the value doesn't matter (mask will ignore them)
        
        return mask, new_values
    
    def decode_delta_rgb_pix(self, mask, new_values):
        """
        Decodes the frame by applying the changed RGB values to the persistent frame.
        Overlays statistics on the frame about the number of pixels and channels updated.
        Args:
        - mask: 2D array of 3-bit values (0-7) indicating which channels changed.
        - new_values: 3D array of new RGB values for the changed channels.
        Returns:
        - Updated persistent frame with statistics overlaid as text.
        """
        # Extract channel-specific masks
        r_changed = (mask & 4) > 0  # Bit 2: R channel changed
        g_changed = (mask & 2) > 0  # Bit 1: G channel changed
        b_changed = (mask & 1) > 0  # Bit 0: B channel changed
        # Each is shape: (height, width), dtype: bool
        
        # Update the persistent frame
        updated_frame = self.delta_rgb_pix_persistent_frame.copy()
        updated_frame[:, :, 0] = np.where(r_changed, new_values[:, :, 0], updated_frame[:, :, 0])
        updated_frame[:, :, 1] = np.where(g_changed, new_values[:, :, 1], updated_frame[:, :, 1])
        updated_frame[:, :, 2] = np.where(b_changed, new_values[:, :, 2], updated_frame[:, :, 2])
        
        # Update the persistent frame
        self.delta_rgb_pix_persistent_frame = updated_frame.astype(np.uint8)
        
        # Compute statistics
        total_pixels_updated = np.sum(mask > 0)
        total_r_updated = np.sum(r_changed)
        total_g_updated = np.sum(g_changed)
        total_b_updated = np.sum(b_changed)
        
        # Prepare the frame for text overlay (ensure it's a copy so we don't modify the persistent frame)
        frame_with_text = self.delta_rgb_pix_persistent_frame.copy()
        
        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White text
        thickness = 1
        line_type = cv2.LINE_AA
        
        # Define starting position (top-left corner)
        text_pos_x = 10
        text_pos_y = 20
        line_spacing = 20  # Pixels between lines
        
        # Define the statistics text
        stats_text = [
            f"Total Pixels: {total_pixels_updated}",
            f"R: {total_r_updated}",
            f"G: {total_g_updated}",
            f"B: {total_b_updated}"
        ]
        
        # Add each line of text to the frame
        for i, line in enumerate(stats_text):
            pos = (text_pos_x, text_pos_y + i * line_spacing)
            # Add black outline for readability
            cv2.putText(frame_with_text, line, pos, font, font_scale, (0, 0, 0), thickness + 1, line_type)
            # Add white text on top
            cv2.putText(frame_with_text, line, pos, font, font_scale, font_color, thickness, line_type)
        
        return frame_with_text
    
    def denoise_delta_rgb_pix(self, mask, new_values, frame, persistent_frame, threshold=1):
        """
        Denoises the output of encode_delta_rgb_pix by excluding RGB changes below a threshold.
        Args:
        - mask: 2D array of 3-bit values (0-7) indicating which channels changed.
        - new_values: 3D array of new RGB values for the changed channels.
        - frame: The original input frame.
        - persistent_frame: The persistent frame to compare against.
        - threshold: The maximum allowed difference to consider a change as noise (default: 1).
        Returns:
        - Updated mask with small changes excluded.
        - Original new_values (unchanged).
        """
        # Compute the absolute difference between the frame and persistent frame
        diff = np.abs(frame.astype(np.int16) - persistent_frame.astype(np.int16))
        # Shape: (height, width, 3), dtype: int16
        
        # Identify channels where the difference is <= threshold (noise)
        is_noise = diff <= threshold  # Shape: (height, width, 3), dtype: bool
        
        # Extract the current channel-specific changes from the mask
        r_changed = (mask & 4) > 0  # Bit 2: R channel
        g_changed = (mask & 2) > 0  # Bit 1: G channel
        b_changed = (mask & 1) > 0  # Bit 0: B channel
        
        # Clear the bits for channels where the change is considered noise
        r_changed = r_changed & ~is_noise[:, :, 0]
        g_changed = g_changed & ~is_noise[:, :, 1]
        b_changed = b_changed & ~is_noise[:, :, 2]
        
        # Recompute the mask with the filtered changes
        updated_mask = (r_changed.astype(np.uint8) * 4 +
                        g_changed.astype(np.uint8) * 2 +
                        b_changed.astype(np.uint8) * 1)
        
        return updated_mask, new_values
    
    def factor_row(self, row_data, width, prime_index):
        """
        Factors a single row of data into its prime factorization.
        Args:
        - row_data: 1D array of bytes (e.g., 640 values for a row).
        - width: Number of pixels in the row.
        - prime_index: Dictionary mapping primes to their indices.
        Returns:
        - List of (prime_idx_bits, exp_bits) for the row.
        """
        # Convert to a big integer (base-256)
        N = 0
        for i, byte in enumerate(row_data):
            N += byte * (256 ** (width - 1 - i))
        
        # Compute the prime factorization
        factors = factorint(N)
        
        # Encode the factorization
        row_factors = []
        for prime, exponent in sorted(factors.items(), key=lambda x: x[0]):
            if prime not in prime_index:
                continue
            prime_idx = prime_index[prime]
            idx_bits = [int(b) for b in format(prime_idx, '08b')]
            exp_bits, _ = self.encode_variable_exponent(exponent)
            row_factors.append((idx_bits, exp_bits))
        
        return row_factors

    def transformer_prime_factor_row(self, frame):
        if not hasattr(self, 'processing') or not self.processing:
            self.processing = True
            threading.Thread(target=self._process_frame, args=(frame,), daemon=True).start()
        
        return self.last_display_frame if self.last_display_frame is not None else frame
    
    def _process_frame(self, frame):
        frame = frame.astype(np.uint8)
        height, width, _ = frame.shape
        segment_size = 32
        combined_factorizations = self.encode_prime_factor_segments(frame, segment_size)
        decoded_frame = self.decode_prime_factor_row(combined_factorizations, height, width, segment_size)
        self.last_display_frame = decoded_frame
        self.processing = False

    # def transformer_prime_factor_row(self, frame):
    #     """
    #     Encodes each row of each channel as a big integer, factorizes it, and transmits the factorization.
    #     Uses parallel processing to handle all rows and channels simultaneously.
    #     """
    #     frame = frame.astype(np.uint8)
    #     height, width, _ = frame.shape
        
    #     # Prepare tasks for parallel processing
    #     tasks = []
    #     for channel in range(3):
    #         for row in range(height):
    #             row_data = frame[row, :, channel]
    #             tasks.append((row_data, width, self.prime_index, channel, row))
        
    #     # Process all rows in parallel
    #     with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
    #         results = list(executor.map(lambda t: (self.factor_row(t[0], t[1], t[2]), t[3], t[4]), tasks))
        
    #     # Organize results into factorizations[channel][row]
    #     factorizations = [[[] for _ in range(height)] for _ in range(3)]
    #     for row_factors, channel, row in results:
    #         factorizations[channel][row] = row_factors
        
    #     # Decode and return the frame
    #     return self.decode_prime_factor_row(factorizations, height, width)
    
    def encode_prime_factor_row(self, frame, segment_size=32):
        frame = frame.astype(np.uint8)
        height, width, _ = frame.shape
        segments_per_row = width // segment_size

        # Reshape frame to (channels, height, segments_per_row, segment_size)
        frame_reshaped = frame.reshape(height, segments_per_row, segment_size, 3).transpose(3, 0, 1, 2)

        # Compute sums for all segments at once
        segment_sums = frame_reshaped.sum(axis=3).astype(np.int32)  # Shape: (3, height, segments_per_row)

        # Map sums to factorizations using a list comprehension (or NumPy vectorization if possible)
        combined_factorizations = np.empty((3, height, segments_per_row), dtype=object)

        for channel in range(3):
            for row in range(height):
                for seg_idx in range(segments_per_row):
                    N = segment_sums[channel, row, seg_idx]
                    combined_factorizations[channel, row, seg_idx] = self.factor_lookup[N]
        
        return combined_factorizations.tolist()
    
    def encode_prime_factor_segments(self, frame, segment_size=32):
        """
        Encode a frame into prime factorizations for each segment.
        
        Args:
            frame: NumPy array of shape (height, width, 3) with uint8 values.
            segment_size: Size of each segment (default: 32).
        
        Returns:
            combined_factorizations: Array of shape (3, height, segments_per_row) with factorization dicts.
        """
        frame = frame.astype(np.uint8)
        height, width, _ = frame.shape
        segments_per_row = width // segment_size
        
        # Reshape frame to (channels, height, segments_per_row, segment_size)
        frame_reshaped = frame.reshape(height, segments_per_row, segment_size, 3).transpose(3, 0, 1, 2)
        # Compute sums for all segments at once
        segment_sums = frame_reshaped.sum(axis=3).astype(np.int32)  # Shape: (3, height, segments_per_row)
        
        # Vectorized lookup
        flat_sums = segment_sums.flatten()  # Shape: (3 * height * segments_per_row,)
        factorizations_flat = [self.factor_lookup[sum_val] for sum_val in flat_sums]
        combined_factorizations = np.array(factorizations_flat, dtype=object).reshape(3, height, segments_per_row)
        
        return combined_factorizations
    
    def decode_prime_factor_row(self, combined_factorizations, height, width, segment_size):
        output = np.zeros((height, width, 3), dtype=np.uint8)
        segments_per_row = width // segment_size
        factor_array = np.array(combined_factorizations, dtype=object)  # Shape: (3, height, segments_per_row)

        # Flatten factorizations for vectorized processing
        flat_factors = factor_array.flatten()  # Shape: (3 * height * segments_per_row,)
        N_values_flat = np.zeros(len(flat_factors), dtype=np.int64)
        for i, factors in enumerate(flat_factors):
            N = 1
            for prime_idx, exponent in factors.items():
                prime = self.prime_list[prime_idx]
                N *= prime ** exponent
            N_values_flat[i] = min(N, 8160)

        # Reshape back
        N_values = N_values_flat.reshape(3, height, segments_per_row)

        # Compute avg_value and remainder for all segments
        avg_values = (N_values // segment_size).astype(np.uint8)
        remainders = N_values % segment_size

        # Expand to pixel level (vectorized)
        output_reshaped = output.reshape(height, segments_per_row, segment_size, 3).transpose(3, 0, 1, 2)
        output_reshaped[...] = avg_values[:, :, :, np.newaxis]  # Broadcast avg_values to all pixels in segment
        # Add remainder: set the first 'remainder' pixels to avg_value + 1
        for channel in range(3):
            for row in range(height):
                for seg_idx in range(segments_per_row):
                    remainder = remainders[channel, row, seg_idx]
                    if remainder > 0:
                        start = seg_idx * segment_size
                        output[row, start:start + remainder, channel] += 1

        return output
    
    def encode_variable_exponent(self, exponent):
        """
        Encodes an exponent using variable-bit encoding.
        - 1 bit: 0 if exponent is 1, 1 if >1.
        - If >1, additional bits to encode the exponent.
        Returns: (encoded_bits, num_bits)
        """
        if exponent == 1:
            return [0], 1
        else:
            bits = bin(exponent)[2:]
            return [1] + [int(b) for b in bits], 1 + len(bits)

    def decode_variable_exponent(self, bits):
        """
        Decodes a variable-bit exponent.
        Returns: (exponent, num_bits_consumed)
        """
        if bits[0] == 0:
            return 1, 1
        else:
            remaining_bits = bits[1:]
            exponent = int(''.join(map(str, remaining_bits)), 2)
            return exponent, 1 + len(remaining_bits)
        
    def factor_8bit(self, value, prime_index):
        """
        Factors an 8-bit number and returns its prime factor signature and factorization.
        """
        if value == 0:
            return 0, {}
        factors = factorint(value)
        signature = sum(prime_index.get(prime, 0) for prime in factors.keys()) % 16
        return signature, factors

    def transformer_prime_factor_signature(self, frame):
        frame = frame.astype(np.uint8)
        # height, width, _ = frame.shape
        segment_size = 32
        signatures = self.encode_prime_factor_signature(frame, segment_size)
        output = self.decode_prime_factor_signature(signatures, self.frame_h, self.frame_w, segment_size)

        return output

    def encode_prime_factor_signature(self, frame, segment_size=32):
        frame = frame.astype(np.uint8)
        # height, width, _ = frame.shape
        segments_per_row = self.frame_w // segment_size
        signatures = np.zeros((3, self.frame_h, segments_per_row), dtype=np.uint8)
        max_possible_idx = len(self.prime_list) - 1

        for channel in range(3):
            channel_data = frame[:, :, channel]
            reshaped = channel_data.reshape(self.frame_h, segments_per_row, segment_size)
            segment_sums = reshaped.sum(axis=2)
            factorizations = np.empty((self.frame_h, segments_per_row), dtype=object)

            for row in range(self.frame_h):
                for seg_idx in range(segments_per_row):
                    N = int(segment_sums[row, seg_idx])
                    factorizations[row, seg_idx] = self.factor_lookup[N]

            for row in range(self.frame_h):
                for seg_idx in range(segments_per_row):
                    factors = factorizations[row, seg_idx]

                    signatures[channel, row, seg_idx] = max(factors.keys()) * 255 // max_possible_idx if factors else 0

                    # if factors:
                    #     max_prime_idx = max(factors.keys())
                    #     signatures[channel, row, seg_idx] = (max_prime_idx * 255 // max_possible_idx)
                    # else:
                    #     signatures[channel, row, seg_idx] = 0

        return signatures

    def decode_prime_factor_signature(self, signatures, height, width, segment_size):
        output = np.zeros((height, width, 3), dtype=np.uint8)
        segments_per_row = width // segment_size

        for channel in range(3):
            for row in range(height):
                for seg_idx in range(segments_per_row):
                    start = seg_idx * segment_size
                    end = start + segment_size
                    output[row, start:end, channel] = signatures[channel, row, seg_idx]

        return output

    # def encode_prime_factor_row(self, frame):
    #     """
    #     Encodes each row of each channel using prime factorization.
    #     Returns a list of factorizations: [(prime_index, exponent), ...] for each row and channel.
    #     """
    #     height, width, _ = frame.shape
    #     factorizations = []
        
    #     for channel in range(3):
    #         channel_factorizations = []
    #         for row in range(height):
    #             # Extract the row for this channel
    #             row_data = frame[row, :, channel]  # Shape: (width,)
                
    #             # Convert to a big integer (base-256)
    #             N = 0
    #             for i, byte in enumerate(row_data):
    #                 N += byte * (256 ** (width - 1 - i))
                
    #             # Compute the prime factorization
    #             factors = factorint(N)
                
    #             # Encode the factorization
    #             row_factors = []
    #             for prime, exponent in sorted(factors.items(), key=lambda x: x[0]):
    #                 if prime not in self.prime_index:
    #                     # Skip primes not in our list (should be rare for small numbers)
    #                     continue
    #                 prime_idx = self.prime_index[prime]
    #                 # Encode the prime index (8 bits)
    #                 idx_bits = [int(b) for b in format(prime_idx, '08b')]
    #                 # Encode the exponent (variable bits)
    #                 exp_bits, exp_num_bits = self.encode_variable_exponent(exponent)
    #                 row_factors.append((idx_bits, exp_bits))
                
    #             channel_factorizations.append(row_factors)
            
    #         factorizations.append(channel_factorizations)
        
    #     return factorizations

    # def decode_prime_factor_row(self, factorizations, height, width):
    #     """
    #     Decodes the factorizations back into a frame.
    #     """
    #     frame = np.zeros((height, width, 3), dtype=np.uint8)
        
    #     for channel in range(3):
    #         for row in range(height):
    #             row_factors = factorizations[channel][row]
                
    #             # Reconstruct the big integer
    #             N = 1
    #             for idx_bits, exp_bits in row_factors:
    #                 prime_idx = int(''.join(map(str, idx_bits)), 2)
    #                 prime = self.prime_list[prime_idx]
    #                 exponent, _ = self.decode_variable_exponent(exp_bits)
    #                 N *= prime ** exponent
                
    #             # Convert the big integer back to bytes
    #             row_data = []
    #             temp_N = N
    #             for i in range(width):
    #                 byte = temp_N % 256
    #                 row_data.append(byte)
    #                 temp_N //= 256
    #             row_data = row_data[::-1]
                
    #             if len(row_data) < width:
    #                 row_data = [0] * (width - len(row_data)) + row_data
    #             frame[row, :, channel] = row_data
        
    #     self.persistent_prime_factor = frame
    #     return self.persistent_prime_factor

    # def decode_prime_factor_row(self, factorizations, height, width):
    #     """
    #     Decodes the factorizations back into a frame.
    #     """
    #     frame = np.zeros((height, width, 3), dtype=np.uint8)
        
    #     for channel in range(3):
    #         for row in range(height):
    #             row_factors = factorizations[channel][row]
                
    #             # Reconstruct the big integer
    #             N = 1
    #             for idx_bits, exp_bits in row_factors:
    #                 # Decode the prime index
    #                 prime_idx = int(''.join(map(str, idx_bits)), 2)
    #                 prime = self.prime_list[prime_idx]
    #                 # Decode the exponent
    #                 exponent, _ = self.decode_variable_exponent(exp_bits)
    #                 # Multiply
    #                 N *= prime ** exponent
                
    #             # Convert the big integer back to bytes
    #             row_data = []
    #             temp_N = N
    #             for i in range(width):
    #                 byte = temp_N % 256
    #                 row_data.append(byte)
    #                 temp_N //= 256
    #             row_data = row_data[::-1]  # Reverse to match the encoding order
                
    #             # Ensure the row_data is the correct length
    #             if len(row_data) < width:
    #                 row_data = [0] * (width - len(row_data)) + row_data
    #             frame[row, :, channel] = row_data
        
    #     # Update the persistent frame
    #     self.persistent_prime_factor = frame
        
    #     return self.persistent_prime_factor

    # transformer_pythagorean_triple
    def transformer_pythagorean_triple(self, frame):
        frame = frame.astype(np.uint8)
        height, width, _ = frame.shape

        # Downsample the frame to 240x320 for performance
        small_frame = cv2.resize(frame, (width//2, height//2), interpolation=cv2.INTER_LINEAR)
        small_height, small_width, _ = small_frame.shape

        # Encode: Find closest Pythagorean triples for the downsampled frame
        closest_triples = self.encode_pythagorean_triple(small_frame)

        # Decode: Reshape and upscale back to original resolution
        output = self.decode_pythagorean_triple(closest_triples, small_height, small_width, height, width)

        return output
    
    def encode_pythagorean_triple(self, frame):
        pixels = frame.reshape(-1, 3)
        triples = np.array(self.pythagorean_triples, dtype=np.uint8)
        diffs = np.abs(pixels[:, np.newaxis, :] - triples[np.newaxis, :, :])
        distances = diffs.sum(axis=2)
        closest_indices = np.argmin(distances, axis=1)
        closest_triples = triples[closest_indices]
        return closest_triples

    def decode_pythagorean_triple(self, closest_triples, small_height, small_width, height, width):
        small_output = closest_triples.reshape(small_height, small_width, 3)
        output = cv2.resize(small_output, (width, height), interpolation=cv2.INTER_NEAREST)

        return output
    
    def transformer_pythagorean_snap(self, frame):
        """
        Snaps each RGB channel value to the nearest number that appears in a Pythagorean triple.
        1. Encodes the frame by snapping each channel.
        2. Decodes the frame into the persistent buffer for progressive reconstruction.
        """
        frame = frame.astype(np.uint8)
        
        # Encode the frame
        transformed, changed_mask = self.encode_pythagorean_snap(frame)
        
        # Decode and return the progressively updated frame
        return self.decode_pythagorean_snap(transformed, changed_mask)

    def encode_pythagorean_snap(self, frame):
        """
        Encodes the frame by snapping each channel to the nearest Pythagorean number.
        Returns the transformed frame and a mask of changed pixels.
        """
        frame = frame.astype(np.uint8)
        height, width, _ = frame.shape
        transformed_frame = np.zeros_like(frame, dtype=np.uint8)
        
        # Process each channel independently
        for channel in range(3):
            channel_values = frame[:, :, channel].flatten()  # Shape: (height * width,)
            # Find the nearest Pythagorean number for each value
            indices = np.searchsorted(self.pythagorean_snap_set, channel_values)
            # Handle edge cases
            indices = np.clip(indices, 1, len(self.pythagorean_snap_set) - 1)
            # Compare distances to the two nearest numbers
            lower = np.array([self.pythagorean_snap_set[i-1] for i in indices])
            upper = np.array([self.pythagorean_snap_set[i] for i in indices])
            distances_lower = np.abs(channel_values - lower)
            distances_upper = np.abs(channel_values - upper)
            # Snap to the closer number
            snapped_values = np.where(
                distances_lower <= distances_upper,
                lower,
                upper
            )
            # Reshape back to image dimensions
            transformed_frame[:, :, channel] = snapped_values.reshape(height, width)
        
        # Create a mask of changed pixels
        changed_mask = np.any(frame != transformed_frame, axis=2)
        
        return transformed_frame, changed_mask

    def decode_pythagorean_snap(self, frame, changed_mask):
        """
        Updates the persistent buffer with the transformed frame where pixels changed.
        """
        # Update the persistent buffer
        self.pythagorean_snap_persistent_frame = np.where(
            changed_mask[:, :, np.newaxis],
            frame,
            self.pythagorean_snap_persistent_frame
        ).astype(np.uint8)
        
        return self.pythagorean_snap_persistent_frame

    def transformer_even_odd_color(self, frame):
        """
        Alternates between forcing odd and even values each frame.
        1. Encodes the frame with the current parity mode.
        2. Decodes the frame into the persistent buffer for progressive reconstruction.
        """
        frame = frame.astype(np.uint8)
        
        # Encode the frame
        transformed, changed_mask, isOdd = self.encode_even_odd_color(frame)
        
        # Decode and return the progressively updated frame
        return self.decode_even_odd_color(transformed, changed_mask, isOdd)

    def encode_even_odd_color(self, frame):
        """
        Encodes the frame by applying the odd/even transformation based on self.even_odd_cycle.
        Returns the transformed frame data and a mask of changed pixels.
        """
        frame = frame.astype(np.uint8)
        
        # Read the current parity mode
        odd = self.even_odd_color_cycle
        
        # Compute the mask of changed pixels
        target_parity = 1 if odd else 0
        boundary_value = 255 if odd else 0
        changed_mask = (frame % 2 != target_parity) & (frame != boundary_value)
        
        # Apply the transformation in one line
        adjustment = 2 * odd - 1  # 1 if odd=True, -1 if odd=False
        transformed_frame = np.where(changed_mask, frame + adjustment, frame)
        
        return transformed_frame.astype(np.uint8), changed_mask, odd

    def decode_even_odd_color(self, frame, changed_mask, isOdd):
        """
        Reverses the transformation only for pixels that were changed and updates the persistent buffer.
        Inverts self.even_odd_cycle for the next frame.
        """
        # Read the current parity mode
        odd = self.even_odd_color_cycle
        
        # Determine the target parity and adjustment
        target_parity = 1 if odd else 0
        adjustment = -1 if odd else 1  # Subtract 1 if odd=True, add 1 if odd=False
        boundary_check = (frame > 0) if odd else (frame < 255)
        
        # Reverse the transformation
        condition = changed_mask & (frame % 2 == target_parity) & boundary_check
        decoded_frame = np.where(condition, frame + adjustment, frame)
        
        # Update the persistent buffer
        self.even_odd_color_persistent_frame = np.where(changed_mask, decoded_frame, self.even_odd_color_persistent_frame).astype(np.uint8)
        
        # Invert even_odd_cycle for the next frame
        self.even_odd_color_cycle = not self.even_odd_color_cycle
        
        return self.even_odd_color_persistent_frame
    
    def transformer_rgb_even_odd_strobe(self, frame):
        """
        Applies the strobing transformation:
        1. Selects the next RGB channel and even/odd mode.
        2. Encodes the frame with only the selected channel and parity.
        3. Decodes the frame into the persistent buffer for progressive reconstruction.
        """
        # Ensure frame is uint8
        frame = frame.astype(np.uint8)

        # Step 1: Get next transmission cycle (numeric channel index and isOdd flag)
        channel_idx, isOdd = self.get_next_rgb_even_odd_strobe_params()

        # Step 2: Encode the frame (apply even/odd filtering to the selected channel)
        output_frame, update_mask, channel_idx, isOdd = self.encode_rgb_even_odd_strobe(frame, channel_idx, isOdd)

        # Step 3: Decode and return the progressively updated frame
        return self.decode_rgb_even_odd_strobe(output_frame, update_mask, channel_idx, isOdd)
    
    def encode_rgb_even_odd_strobe(self, frame, channel_idx, isOdd=True):
        """
        Extracts an RGB channel and keeps only even or odd values based on isOdd.
        - channel_idx: 0 (R), 1 (G), 2 (B)
        - isOdd: True (keeps odd values), False (keeps even values)
        Returns: (output_frame, update_mask, channel_idx, isOdd)
        """
        # Ensure frame is uint8
        frame = frame.astype(np.uint8)
        
        # Extract just the selected channel
        channel_data = frame[:, :, channel_idx].copy()

        # Compute the update mask based on parity
        update_mask = (channel_data % 2 == isOdd)  # Shape: (height, width)

        # Apply the update mask to keep only matching values
        channel_data = np.where(update_mask, channel_data, 0)

        # Create an output frame with only the selected channel
        output_frame = np.zeros_like(frame, dtype=np.uint8)
        output_frame[:, :, channel_idx] = channel_data
        
        return output_frame, update_mask, channel_idx, isOdd
    
    def decode_rgb_even_odd_strobe(self, encoded_frame, update_mask, channel_idx, isOdd):
        """
        Uses the incoming partial frame data to reconstruct the full frame progressively.
        - encoded_frame: The current incoming encoded frame
        - update_mask: A boolean mask indicating which pixels to update
        - channel_idx: The channel index (0 = R, 1 = G, 2 = B)
        - isOdd: Whether the data represents odd or even values
        """
        # Extract the transmitted channel data
        new_data = encoded_frame[:, :, channel_idx]

        # Apply updates only where the update mask is True
        self.persistent_rgb_even_odd_strobe[:, :, channel_idx] = np.where(
            update_mask, new_data, self.persistent_rgb_even_odd_strobe[:, :, channel_idx]
        ).astype(np.uint8)

        return self.persistent_rgb_even_odd_strobe
    
    def get_next_rgb_even_odd_strobe_params(self):
        channel_idx, isOdd = self.rgb_even_odd_strobe_cycle_order[self.rgb_even_odd_strobe_cycle]
        # print(f"Cycle {self.rgb_strobe_cycle}: Channel {channel_idx}, isOdd={isOdd}")
        self.rgb_even_odd_strobe_cycle = (self.rgb_even_odd_strobe_cycle + 1) % 6

        return channel_idx, isOdd
    
    def encode_rgb_matrix_strobe(self, frame, channel_idx, isOdd=True):
        """
        Extracts an RGB channel and keeps only even or odd pixels based on isOdd.
        - channel_idx: 0 (R), 1 (G), 2 (B)
        - isOdd: True (keeps odd pixels), False (keeps even pixels)
        Returns: (output_frame, channel_idx, isOdd)
        """
        # Ensure frame is uint8
        frame = frame.astype(np.uint8)
        
        # Extract just the selected channel
        channel_data = frame[:, :, channel_idx].copy()

        # Create a checkerboard mask for even/odd pixels
        height, width = channel_data.shape
        row_indices, col_indices = np.indices((height, width))
        checkerboard = (row_indices + col_indices) % 2  # 0 for even, 1 for odd
        update_mask = (checkerboard == (1 if isOdd else 0))  # True for pixels to update

        # Apply the update mask to keep only the selected pixels
        channel_data = np.where(update_mask, channel_data, 0)

        # Create an output frame with only the selected channel
        output_frame = np.zeros_like(frame, dtype=np.uint8)
        output_frame[:, :, channel_idx] = channel_data
        
        return output_frame, channel_idx, isOdd

    def decode_rgb_matrix_strobe(self, encoded_frame, channel_idx, isOdd):
        """
        Uses the incoming partial frame data to reconstruct the full frame progressively.
        - encoded_frame: The current incoming encoded frame
        - channel_idx: The channel index (0 = R, 1 = G, 2 = B)
        - isOdd: Whether the data represents odd or even pixels
        """
        # Extract the transmitted channel data
        new_data = encoded_frame[:, :, channel_idx]

        # Create the same checkerboard mask as in the encode function
        height, width = new_data.shape
        row_indices, col_indices = np.indices((height, width))
        checkerboard = (row_indices + col_indices) % 2
        update_mask = (checkerboard == (1 if isOdd else 0))

        # Apply updates only where the update mask is True
        self.persistent_rgb_matrix_strobe[:, :, channel_idx] = np.where(
            update_mask, new_data, self.persistent_rgb_matrix_strobe[:, :, channel_idx]
        ).astype(np.uint8)

        return self.persistent_rgb_matrix_strobe

    def transformer_rgb_matrix_strobe(self, frame):
        """
        Applies the matrix strobing transformation:
        1. Selects the next RGB channel and even/odd pixel mode.
        2. Encodes the frame with only the selected channel and pixels.
        3. Decodes the frame into the persistent buffer for progressive reconstruction.
        """
        # Ensure frame is uint8
        frame = frame.astype(np.uint8)

        # Step 1: Get next transmission cycle (numeric channel index and isOdd flag)
        channel_idx, isOdd = self.get_next_rgb_even_odd_strobe_params()

        # Step 2: Encode the frame (apply even/odd pixel filtering to the selected channel)
        output_frame, channel_idx, isOdd = self.encode_rgb_matrix_strobe(frame, channel_idx, isOdd)

        # Step 3: Decode and return the progressively updated frame
        return self.decode_rgb_matrix_strobe(output_frame, channel_idx, isOdd)
    
    def transformer_rgb_strobe(self, frame):
        """
        Applies the RGB split transformation:
        1. Selects the next RGB channel.
        2. Encodes the frame with only the selected channel.
        3. Decodes the frame into the persistent buffer for progressive reconstruction.
        """
        # Ensure frame is uint8
        frame = frame.astype(np.uint8)

        # Step 1: Get the next channel
        channel_idx = self.get_next_rgb_strobe_params()

        # Step 2: Encode the frame (extract the selected channel)
        output_frame, channel_idx = self.encode_rgb_strobe(frame, channel_idx)

        # Step 3: Decode and return the progressively updated frame
        return self.decode_rgb_strobe(output_frame, channel_idx)
    
    def encode_rgb_strobe(self, frame, channel_idx):
        """
        Extracts the specified RGB channel and sets the other channels to 0.
        - channel_idx: 0 (R), 1 (G), 2 (B)
        Returns: (output_frame, channel_idx)
        """
        # Ensure frame is uint8
        frame = frame.astype(np.uint8)
        
        # Create an output frame with all channels set to 0
        output_frame = np.zeros_like(frame, dtype=np.uint8)
        
        # Copy the selected channel to the output frame
        output_frame[:, :, channel_idx] = frame[:, :, channel_idx]
        
        return output_frame, channel_idx
    
    def decode_rgb_strobe(self, encoded_frame, channel_idx):
        """
        Updates the persistent buffer with the transmitted channel.
        - encoded_frame: The current incoming encoded frame
        - channel_idx: The channel index (0 = R, 1 = G, 2 = B)
        """
        # Extract the transmitted channel data
        new_data = encoded_frame[:, :, channel_idx]

        # Update the persistent buffer for the selected channel
        self.rgb_strobe_persistent_frame[:, :, channel_idx] = new_data.astype(np.uint8)

        return self.rgb_strobe_persistent_frame
    
    def transformer_even_odd_spatial(self, frame):
        """
        Applies the spatial even/odd transformation:
        1. Selects the next even/odd mode.
        2. Encodes the frame with the selected parity and position.
        3. Decodes the frame into the persistent buffer for progressive reconstruction.
        """
        frame = frame.astype(np.uint8)
        
        # Get the next mode (isOdd alternates between True and False)
        isOdd = (self.even_odd_spatial_cycle % 2 == 0)
        self.even_odd_spatial_cycle += 1
        
        # Encode the frame
        output_frame, update_mask, isOdd = self.encode_even_odd_spatial(frame, isOdd)
        
        # Decode and return the progressively updated frame
        return self.decode_even_odd_spatial(output_frame, update_mask, isOdd)
    
    def decode_even_odd_spatial(self, encoded_frame, update_mask, isOdd):
        """
        Updates the persistent buffer with the transmitted even/odd values.
        """
        # Update the persistent buffer where update_mask is True
        self.even_odd_spatial_persistent_frame = np.where(
            update_mask[:, :, None], encoded_frame, self.even_odd_spatial_persistent_frame
        ).astype(np.uint8)
        
        return self.even_odd_spatial_persistent_frame
    
    def encode_even_odd_spatial(self, frame, isOdd=True):
        """
        Extracts even or odd values based on a checkerboard pattern.
        - isOdd=True:  Even positions get even values, odd positions get odd values.
        - isOdd=False: Even positions get odd values, odd positions get even values.
        Returns: (output_frame, update_mask, isOdd)
        """
        frame = frame.astype(np.uint8)
        height, width = frame.shape[:2]
        
        # Create a checkerboard mask for even/odd positions
        row_indices, col_indices = np.indices((height, width))
        checkerboard = (row_indices + col_indices) % 2  # Shape: (480, 640)
        
        # Compute parity of frame values
        parity = frame % 2  # Shape: (480, 640, 3)
        
        # Add a new axis to checkerboard to broadcast across channels
        checkerboard = checkerboard[:, :, None]  # Shape: (480, 640, 1)
        
        # Determine which values to keep based on position and parity
        if isOdd:
            update_mask = ((checkerboard == 0) & (parity == 0)) | ((checkerboard == 1) & (parity == 1))
        else:
            update_mask = ((checkerboard == 0) & (parity == 1)) | ((checkerboard == 1) & (parity == 0))
        
        # Reduce the update_mask to a 2D array: update a pixel if any channel should be updated
        update_mask = np.any(update_mask, axis=2)  # Shape: (480, 640)
        
        # Create output frame: keep values where update_mask is True, set others to 0
        output_frame = np.where(update_mask[:, :, None], frame, 0)
        
        return output_frame, update_mask, isOdd
    
    def get_next_rgb_strobe_params(self):
        """
        Returns the next transmission cycle for RGB Split: (RGB index,)
        """
        channel_idx = self.rgb_strobe_cycle_order[self.rgb_strobe_cycle][0]
        self.rgb_strobe_cycle = (self.rgb_strobe_cycle + 1) % 3
        return channel_idx

    def transformer_mondrian_2(self, frame):
        h, w, _ = frame.shape
        output_frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Update the map every 30 frames
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
        if not hasattr(self, 'mondrian_map') or self.mondrian_map is None or self.frame_count % 30 == 0:
            self.mondrian_h_lines = set()
            self.mondrian_v_lines = set()
            self.color_counts = {tuple(color): 0 for color in MONDRIAN_PALETTE}  # Reset color counts
            
            # Subdivide and assign colors
            regions = []
            grid_size = 8
            for i in range(0, h, h // grid_size):
                for j in range(0, w, w // grid_size):
                    self.mondrian_map = regions  # Temporarily store for neighbor checks
                    sub_regions = compute_complexity_and_subdivide(frame, j, i, w // grid_size, h // grid_size)
                    regions.extend(sub_regions)
            
            # Enforce minimum 5 regions per color
            min_regions_per_color = 5
            total_regions_needed = len(MONDRIAN_PALETTE) * min_regions_per_color
            color_usage = self.color_counts.copy()
            if len(regions) < total_regions_needed:
                extra_needed = total_regions_needed - len(regions)
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
                        used_colors = set()
                        for srx, sry, srw, srh in sub_regions:
                            color = self._assign_color(frame, srx, sry, srw, srh, exclude_colors=used_colors)
                            used_colors.add(tuple(color))
                            regions.append((srx, sry, srw, srh, color))
                            color_usage[tuple(color)] += 1
            
            # Redistribute colors to meet minimum requirement
            final_regions = []
            for region in regions:
                x, y, w, h, color = region
                if color_usage[tuple(color)] >= min_regions_per_color:
                    final_regions.append(region)
                else:
                    underused = [c for c, count in color_usage.items() if count < min_regions_per_color]
                    if underused:
                        new_color = underused[0]
                        color_usage[tuple(new_color)] += 1
                        color_usage[tuple(color)] -= 1
                        final_regions.append((x, y, w, h, new_color))
                    else:
                        final_regions.append(region)
            
            self.mondrian_map = final_regions

        # Render the regions with their main colors
        for x, y, w, h, color in self.mondrian_map:
            if w < 20 or h < 20:
                continue
            output_frame[y:y+h, x:x+w] = color
            
            # Draw black lines
            thickness = max(2, min(6, int((w + h) / 50)))
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 0), thickness=thickness)

        self.frame_count += 1
        return output_frame
    
    def transformer_mondrian_next(self, frame):
        h, w, _ = frame.shape
        output_frame = frame.copy()

        # Step 1: Subdivide based on complexity
        regions = []
        grid_size = 8
        for i in range(0, h, h // grid_size):
            for j in range(0, w, w // grid_size):
                sub_regions = compute_complexity_and_subdivide(frame, j, i, w // grid_size, h // grid_size)
                regions.extend(sub_regions)

        # Step 2: Get dominant colors and assign with variation
        dominant_colors = [self.get_dominant_color(frame[y:y+h, x:x+w]) for (x, y, w, h) in regions]
        assigned_colors = self.assign_colors(regions, dominant_colors)

        # Step 3: Process each region
        for (x, y, w, h), color in zip(regions, assigned_colors):
            if w < 20 or h < 20:
                continue

            # Create 5-color palette for this region
            region_palette = self.create_region_palette(color)

            # Map pixels in the region to the limited palette
            roi = output_frame[y:y+h, x:x+w]
            mapped_roi = self.map_pixels(roi, region_palette)
            output_frame[y:y+h, x:x+w] = mapped_roi

            # Draw black lines (thickness scaled by region size)
            thickness = max(2, min(6, int((w + h) / 50)))
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 0), thickness=thickness)

        return output_frame
    
    def transformer_complexity_test(self, frame):
        """Visualize the new subdivisions with tricolor assignment."""
        h, w, _ = frame.shape
        output_frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Update the map every 30 frames
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
        if not hasattr(self, 'mondrian_map') or self.mondrian_map is None or self.frame_count % 30 == 0:
            # Find complexity points and grow lines
            points = self.find_complexity_points(frame)
            rectangles = self.grow_lines(frame, points)

            # Assign colors
            self.color_counts = {
                (227, 66, 52): 0,   # Mondrian Red
                (238, 210, 20): 0,  # Mondrian Yellow
                (39, 89, 180): 0    # Mondrian Blue
            }
            regions = []
            for x, y, w, h in rectangles:
                color = self._assign_tricolor(x, y, w, h)
                regions.append((x, y, w, h, color, None))  # No palette needed for test

            self.mondrian_map = regions

        # Render the regions with their main colors
        for x, y, w, h, color, _ in self.mondrian_map:
            if w < 20 or h < 20:
                continue
            output_frame[y:y+h, x:x+w] = color

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
    
    def encode_gradient_tiles(self, frame):
        """
        Encode a frame into reference colors and neighbor adjustments for 32x32 tiles.

        Args:
            frame: np.ndarray of shape (480, 640, 3) with uint8 values (RGB).

        Returns:
            tuple: (ref_colors, adjustments)
                - ref_colors: np.ndarray (15, 20, 3), uint8, reference colors per tile.
                - adjustments: np.ndarray (15, 20, 8), uint8, 4-bit adjustments per neighbor.
        """
        assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8, "Frame must be 480x640x3 uint8"
        
        tile_size = 32
        num_tiles_y, num_tiles_x = 15, 20
        
        ref_colors = np.zeros((num_tiles_y, num_tiles_x, 3), dtype=np.uint8)
        adjustments = np.zeros((num_tiles_y, num_tiles_x, 8), dtype=np.uint8)
        
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for ty in range(num_tiles_y):
            for tx in range(num_tiles_x):
                y_start = ty * tile_size
                x_start = tx * tile_size
                tile = frame[y_start:y_start + tile_size, x_start:x_start + tile_size, :]
                
                ref_color = tile[tile_size // 2, tile_size // 2, :]
                ref_colors[ty, tx] = ref_color
                
                for n_idx, (dy, dx) in enumerate(neighbor_offsets):
                    ny, nx = ty + dy, tx + dx
                    if 0 <= ny < num_tiles_y and 0 <= nx < num_tiles_x:
                        neighbor_ref = ref_colors[ny, nx]
                        diff = (neighbor_ref.astype(np.int16) - ref_color.astype(np.int16))
                        # Compute a single adjustment by averaging the difference across channels
                        adjustment = np.clip(np.mean(diff) // 16, -8, 7) + 8
                        adjustments[ty, tx, n_idx] = adjustment.astype(np.uint8)
                    else:
                        adjustments[ty, tx, n_idx] = 8
        
        return ref_colors, adjustments
    
    def decode_gradient_tiles(self, encoded_data):
        """
        Decode reference colors and adjustments into a full frame.

        Args:
            encoded_data: tuple (ref_colors, adjustments)

        Returns:
            np.ndarray: Reconstructed frame (480, 640, 3), uint8.
        """
        ref_colors, adjustments = encoded_data
        tile_size = 32
        num_tiles_y, num_tiles_x = 15, 20
        
        output_frame = np.zeros((480, 640, 3), dtype=np.float32)
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for ty in range(num_tiles_y):
            for tx in range(num_tiles_x):
                ref_color = ref_colors[ty, tx].astype(np.float32)
                adj = adjustments[ty, tx].astype(np.int16) - 8
                influences = adj * 16
                
                for y in range(tile_size):
                    for x in range(tile_size):
                        pos_y = ty * tile_size + y
                        pos_x = tx * tile_size + x
                        color = ref_color.copy()
                        
                        for n_idx, (dy, dx) in enumerate(neighbor_offsets):
                            dist_y = (y - tile_size // 2) - dy * tile_size
                            dist_x = (x - tile_size // 2) - dx * tile_size
                            dist = np.sqrt(dist_y**2 + dist_x**2)
                            if dist > 0 and 0 <= ty + dy < num_tiles_y and 0 <= tx + dx < num_tiles_x:
                                weight = 1 / (dist + 1)
                                # Apply the scalar influence to all channels
                                color += influences[n_idx] * weight
                        
                        output_frame[pos_y, pos_x] = color
        
        return np.clip(output_frame, 0, 255).astype(np.uint8)
    
    def transformer_gradient_tiles(self, frame):
        """
        Transform a frame using gradient tiles encoding and decoding.

        Args:
            frame: np.ndarray (480, 640, 3), uint8.

        Returns:
            np.ndarray: Reconstructed frame (480, 640, 3), uint8.
        """
        encoded_data = self.encode_gradient_tiles(frame)

        return self.decode_gradient_tiles(encoded_data)
    
    def get_diagonal_indices(self, start_x, start_y, width, height):
        """Compute the indices of a diagonal starting at (start_x, start_y) with wrapping."""
        indices = []
        x, y = start_x, start_y
        diagonal_length = max(width, height)  # Ensure we cover the longest dimension

        for _ in range(diagonal_length):
            indices.append((x, y))
            x = (x + 1) % width
            y = (y + 1) % height

        return indices

    def transformer_diagonal_vision(self, frame):
        """Apply the Diagonal Vision transformer by processing bit patterns along diagonals."""
        h, w, _ = frame.shape
        output_frame = frame.copy().astype(np.uint8)  # Ensure output_frame is uint8

        # Process diagonals starting from top row and left column
        diagonal_starts = [(0, j) for j in range(w)] + [(i, 0) for i in range(1, h)]

        for start_x, start_y in diagonal_starts:
            # Get the indices of the diagonal
            indices = self.get_diagonal_indices(start_x, start_y, w, h)
            
            # For each color channel (R, G, B)
            for channel in range(3):
                # For each bit position (0-7)
                for bit_pos in range(8):
                    # Extract the nth bit from the channel for all pixels in the diagonal
                    bits = []
                    for x, y in indices:
                        pixel_value = frame[y, x, channel]
                        bit = (pixel_value >> bit_pos) & 1
                        bits.append(bit)

                    # Compare to the two patterns: 101010... and 010101...
                    pattern_1 = [1 if i % 2 == 0 else 0 for i in range(len(bits))]  # 101010...
                    pattern_0 = [0 if i % 2 == 0 else 1 for i in range(len(bits))]  # 010101...

                    # Compute mismatches
                    mismatches_1 = sum(a != b for a, b in zip(bits, pattern_1))
                    mismatches_0 = sum(a != b for a, b in zip(bits, pattern_0))

                    # Choose the pattern with fewer mismatches
                    chosen_pattern = pattern_1 if mismatches_1 <= mismatches_0 else pattern_0

                    # Assign the chosen pattern to the nth bit of the channel for all pixels in the diagonal
                    for idx, (x, y) in enumerate(indices):
                        current_value = output_frame[y, x, channel]  # Already uint8
                        # Clear the nth bit
                        mask = np.uint8(255 - (1 << bit_pos))  # Compute mask as 255 - (1 << bit_pos)
                        current_value = current_value & mask
                        # Set the nth bit according to the chosen pattern
                        bit_value = np.uint8(chosen_pattern[idx] << bit_pos)
                        current_value = current_value | bit_value
                        output_frame[y, x, channel] = current_value

        return output_frame
    
    def initialize_reference_patterns(self):
        """Initialize two 640x480x3 frames with 10101010... and 01010101... patterns."""
        # Create a 480-length pattern for each
        pattern_1 = np.array([1 if i % 2 == 0 else 0 for i in range(self.frame_h)], dtype=np.uint8)  # 10101010...
        pattern_0 = np.array([0 if i % 2 == 0 else 1 for i in range(self.frame_h)], dtype=np.uint8)  # 01010101...

        # Convert patterns to 8-bit values (10101010 or 01010101)
        byte_1 = int('10101010', 2)  # 170
        byte_0 = int('01010101', 2)  # 85

        # Create 640x480x3 frames
        self.pattern_frame_1 = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        self.pattern_frame_0 = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)

        # Fill each column with the repeating pattern
        for col in range(self.frame_w):
            # For pattern_1 (10101010...), all channels get byte_1
            self.pattern_frame_1[:, col, :] = byte_1
            # For pattern_0 (01010101...), all channels get byte_0
            self.pattern_frame_0[:, col, :] = byte_0

    def transformer_vertical_bitfield(self, frame):
        """Encode the frame using a reversible process with pattern dominance."""
        h, w, _ = frame.shape
        assert w == 640 and h == 480, "Expected frame size 640x480"

        # Perform bitwise AND with each pattern frame
        matches_1 = frame & self.pattern_frame_1  # 640x480x3
        matches_0 = frame & self.pattern_frame_0  # 640x480x3

        # Sum matches along the height axis to get per-column totals
        total_matches_1 = np.sum(matches_1, axis=0)  # Shape: (640, 3)
        total_matches_0 = np.sum(matches_0, axis=0)  # Shape: (640, 3)

        # Compute matches for the full pixel
        full_pixel_matches_1 = np.sum(total_matches_1, axis=1)  # Shape: (640,)
        full_pixel_matches_0 = np.sum(total_matches_0, axis=1)  # Shape: (640,)

        # Create the 640x4 bitfield
        bitfield = np.zeros((w, 4), dtype=np.uint8)
        bitfield[:, 0] = (full_pixel_matches_1 >= full_pixel_matches_0).astype(np.uint8)
        bitfield[:, 1:4] = (total_matches_1 >= total_matches_0).astype(np.uint8)

        # Step 1: Build the base frame using the full pixel bitfield (column 0)
        base_frame = np.zeros((h, w, 3), dtype=np.uint8)
        pattern_1_value = 170  # 10101010
        pattern_0_value = 85   # 01010101

        for col in range(w):
            pattern_value = pattern_1_value if bitfield[col, 0] == 1 else pattern_0_value
            base_frame[:, col, :] = pattern_value

        # Step 2: Compute the difference frame
        difference_frame = (frame.astype(np.int16) - base_frame.astype(np.int16)) % 256
        difference_frame = difference_frame.astype(np.uint8)

        # Step 3: Scale the difference to a smaller range (0-15) for modulation
        # This ensures the pattern dominates the output
        scaled_difference = (difference_frame // 16).astype(np.uint8)  # 0-15 range

        # Step 4: Build the encoded frame by starting with the per-channel patterns
        encoded_frame = np.zeros((h, w, 3), dtype=np.uint8)
        for col in range(w):
            for channel in range(3):
                # Use the channel-specific bitfield to choose the pattern
                pattern_value = pattern_1_value if bitfield[col, channel + 1] == 1 else pattern_0_value
                # Start with the pattern, then add the scaled difference
                encoded_frame[:, col, channel] = (pattern_value + scaled_difference[:, col, channel]) % 256

        # Step 5: Create a 640-bit array for 4 groups of 20 8-bit colors
        color_group_bits = np.zeros(w, dtype=np.uint8)
        for col in range(w):
            group = (bitfield[col, 0] << 1) | bitfield[col, 1]  # Combine full pixel and red channel bits
            color_group_bits[col] = group  # Values 0-3

        # Store the color group bits and scaled difference for decoding
        self.color_group_bits = color_group_bits
        self.scaled_difference = scaled_difference  # Store for reversibility

        # Return the encoded frame
        return encoded_frame.astype(np.uint8)

# next, I want to actually change the frame. I think we have the performance we were looking for (these are limited bitwise operations after all!)

# Now that we have the bitfield arrays, we can do this. They are indexed like:
# # Create the 640x4 bitfield
# # Column 0: Full pixel (24 bits)
# # Column 1: Red channel (8 bits)
# # Column 2: Green channel (8 bits)
# # Column 3: Blue channel (8 bits)

# So we will use them to build a new frame. We'll take the bitfield and create a new frame where we start with populating the columns for the Full pixel, then we 

    # def initialize_reference_patterns(self):
    #     """Initialize two 640x480x3 frames with 10101010... and 01010101... patterns."""
    #     # Create a 480-length pattern for each
    #     pattern_1 = np.array([1 if i % 2 == 0 else 0 for i in range(self.height)], dtype=np.uint8)  # 10101010...
    #     pattern_0 = np.array([0 if i % 2 == 0 else 1 for i in range(self.height)], dtype=np.uint8)  # 01010101...

    #     # Convert patterns to 8-bit values (10101010 or 01010101)
    #     byte_1 = int('10101010', 2)  # 170
    #     byte_0 = int('01010101', 2)  # 85

    #     # Create 640x480x3 frames
    #     self.pattern_frame_1 = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    #     self.pattern_frame_0 = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    #     for col in range(self.width):
    #         self.pattern_frame_1[:, col, :] = byte_1
    #         self.pattern_frame_0[:, col, :] = byte_0

    # def initialize_vertical_bitfield_reference_patterns(self):
    #     # Existing code...
    #     byte_1 = int('10101010', 2)  # 170
    #     byte_0 = int('01010101', 2)  # 85
    #     self.pattern_frame_1 = np.full((self.height, self.width, 3), byte_1, dtype=np.uint8)
    #     self.pattern_frame_0 = np.full((self.height, self.width, 3), byte_0, dtype=np.uint8)

    def get_most_common_24bit_pattern(self, column, offsets=[0, 1, 2, 3]):
        """Find the most common 24-bit pattern in a column with bit offsets."""
        # Column shape: (480, 3)
        r, g, b = column[:, 0], column[:, 1], column[:, 2]

        # Combine RGB into a 24-bit integer
        rgb_values = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)

        # Consider offsets by shifting and masking
        max_count = 0
        best_pattern = None
        for offset in offsets:
            # Shift right by offset and mask to get 24-bit pattern
            shifted = (rgb_values >> offset) & 0xFFFFFF
            # Compute histogram of patterns
            unique, counts = np.unique(shifted, return_counts=True)
            most_common_idx = np.argmax(counts)
            count = counts[most_common_idx]
            pattern = unique[most_common_idx]
            if count > max_count:
                max_count = count
                best_pattern = pattern

        # Convert the 24-bit pattern back to RGB
        r_pattern = (best_pattern >> 16) & 0xFF
        g_pattern = (best_pattern >> 8) & 0xFF
        b_pattern = best_pattern & 0xFF
        return np.array([r_pattern, g_pattern, b_pattern], dtype=np.uint8), max_count
    
    def get_most_common_8bit_pattern(self, channel_values, offsets=[0, 1, 2, 3]):
        """Find the most common 8-bit pattern in a channel with bit offsets."""
        # Channel_values shape: (480,)
        max_count = 0
        best_pattern = None
        for offset in offsets:
            shifted = (channel_values >> offset) & 0xFF
            unique, counts = np.unique(shifted, return_counts=True)
            most_common_idx = np.argmax(counts)
            count = counts[most_common_idx]
            pattern = unique[most_common_idx]
            if count > max_count:
                max_count = count
                best_pattern = pattern
        return best_pattern, max_count
    
    def map_to_nearest_color(self, value, color_set):
        """Map a value to the nearest value in the color set."""
        return color_set[np.argmin(np.abs(color_set - value))]
    
    def get_top_80_colors(self, frame):
        """Extract the 80 most common 8-bit values across all channels in the frame."""
        # Frame shape: (480, 640, 3)
        all_values = frame.reshape(-1, 3)  # Shape: (480*640, 3)
        r_values = all_values[:, 0]
        g_values = all_values[:, 1]
        b_values = all_values[:, 2]
        all_channel_values = np.concatenate([r_values, g_values, b_values])
        unique, counts = np.unique(all_channel_values, return_counts=True)
        # Sort by count and take the top 80
        sorted_indices = np.argsort(-counts)
        top_80 = unique[sorted_indices[:80]]
        return top_80

    def transformer_vertical_bitfield_pattern_prototype(self, frame):
        """Create a new frame using the most common 24-bit and 8-bit patterns."""
        h, w, _ = frame.shape
        assert w == 640 and h == 480, "Expected frame size 640x480"

        # Step 1: Get the top 80 most common 8-bit values
        top_80_colors = self.get_top_80_colors(frame)

        # Step 2: Find the most common 24-bit pattern per column
        base_frame = np.zeros((h, w, 3), dtype=np.uint8)
        for col in range(w):
            column = frame[:, col, :]  # Shape: (480, 3)
            pattern, _ = self.get_most_common_24bit_pattern(column)
            # Tile the pattern across the column
            base_frame[:, col, :] = pattern

        # Step 3: Compute the difference frame
        difference_frame = (frame.astype(np.int16) - base_frame.astype(np.int16)) % 256
        difference_frame = difference_frame.astype(np.uint8)

        # Step 4: Find the most common 8-bit pattern per channel per column
        encoded_frame = base_frame.copy()
        for col in range(w):
            for channel in range(3):
                channel_values = difference_frame[:, col, channel]  # Shape: (480,)
                pattern, _ = self.get_most_common_8bit_pattern(channel_values)
                # Add the pattern to the base frame
                encoded_frame[:, col, channel] = (encoded_frame[:, col, channel].astype(np.int16) + pattern) % 256

        # Step 5: Map all values in the encoded frame to the nearest of the top 80 colors
        final_frame = np.zeros_like(encoded_frame)
        for col in range(w):
            for channel in range(3):
                final_frame[:, col, channel] = np.vectorize(lambda x: self.map_to_nearest_color(x, top_80_colors))(encoded_frame[:, col, channel])

        # Step 6: Create a 640-bit array for 4 groups of 20 8-bit colors
        # Use the same logic as before for consistency
        color_group_bits = np.zeros(w, dtype=np.uint8)
        for col in range(w):
            # Perform bitwise AND with pattern frames to compute bitfield
            col_matches_1 = frame[:, col, :] & self.pattern_frame_1[:, col, :]
            col_matches_0 = frame[:, col, :] & self.pattern_frame_0[:, col, :]
            total_matches_1 = np.sum(col_matches_1)
            total_matches_0 = np.sum(col_matches_0)
            bit = 1 if total_matches_1 >= total_matches_0 else 0
            # Use the bit and the next bit (e.g., from red channel match) to decide the group
            col_matches_1_r = frame[:, col, 0] & self.pattern_frame_1[:, col, 0]
            col_matches_0_r = frame[:, col, 0] & self.pattern_frame_0[:, col, 0]
            total_matches_1_r = np.sum(col_matches_1_r)
            total_matches_0_r = np.sum(col_matches_0_r)
            bit_r = 1 if total_matches_1_r >= total_matches_0_r else 0
            group = (bit << 1) | bit_r
            color_group_bits[col] = group

        # Store the color group bits for later use
        self.color_group_bits = color_group_bits

        # Return the final frame
        return final_frame.astype(np.uint8)
    
    def initialize_vertical_bitfield_reference_patterns(self):
        """Initialize transformations and pattern frames for vertical bitfield processing."""
        # 23-bit transformation: Expand from 480 to 483 pixels, trim back to 480
        self.height_23bit = 483  # 21 groups of 23 pixels
        self.expand_indices_23bit = np.arange(self.frame_h)
        self.trim_indices_23bit = np.arange(self.frame_h)  # First 480 pixels

        # 7-bit transformation: Expand from 480 to 483 values, trim back to 480
        self.height_7bit = 483  # 69 groups of 7 pixels
        self.expand_indices_7bit = np.arange(self.frame_h)
        self.trim_indices_7bit = np.arange(self.frame_h)  # First 480 pixels

        # Initialize pattern frames for color_group_bits
        byte_1 = int('10101010', 2)  # 170
        byte_0 = int('01010101', 2)  # 85
        self.pattern_frame_1 = np.full((self.frame_h, self.frame_w, 3), byte_1, dtype=np.uint8)
        self.pattern_frame_0 = np.full((self.frame_h, self.frame_w, 3), byte_0, dtype=np.uint8)
    
    def count_matching_bits_23(self, values, patterns, offset):
        """Count matching bits between 23-bit values and patterns with an offset."""
        # Values shape: (483, 640), patterns shape: (21, 640)
        # Shift the patterns by the offset
        shifted_patterns = ((patterns << offset) | (patterns >> (23 - offset))) & 0x7FFFFF  # Shape: (21, 640)
        # Broadcast to match values shape
        shifted_patterns = shifted_patterns[:, np.newaxis, :]  # Shape: (21, 1, 640)
        values_exp = values[np.newaxis, :, :]  # Shape: (1, 483, 640)
        # XOR to find differing bits
        xor_result = (values_exp ^ shifted_patterns) & 0x7FFFFF  # Shape: (21, 483, 640)
        # Count 0s (matches) using bitwise operations
        # We can approximate by counting 1s in each byte and subtracting
        byte1 = (xor_result >> 16) & 0xFF
        byte2 = (xor_result >> 8) & 0xFF
        byte3 = xor_result & 0xFF
        bits = (np.sum(np.unpackbits(byte1, axis=0), axis=0) +
                np.sum(np.unpackbits(byte2, axis=0), axis=0) +
                np.sum(np.unpackbits(byte3, axis=0), axis=0))  # Shape: (21, 483, 640)
        matching_bits = (23 - bits)  # Shape: (21, 483, 640)
        return np.sum(matching_bits, axis=1)  # Shape: (21, 640)

    def count_matching_bits_7(self, values, patterns, offset):
        """Count matching bits between 7-bit values and patterns with an offset."""
        # Values shape: (483, 640), patterns shape: (69, 640)
        shifted_patterns = ((patterns << offset) | (patterns >> (7 - offset))) & 0x7F  # Shape: (69, 640)
        shifted_patterns = shifted_patterns[:, np.newaxis, :]  # Shape: (69, 1, 640)
        values_exp = values[np.newaxis, :, :]  # Shape: (1, 483, 640)
        xor_result = (values_exp ^ shifted_patterns) & 0x7F  # Shape: (69, 483, 640)
        bits = np.sum(np.unpackbits(xor_result, axis=0)[:, 1:], axis=0)  # Shape: (69, 483, 640)
        matching_bits = (7 - bits)
        return np.sum(matching_bits, axis=1)  # Shape: (69, 640)

    def get_best_23bit_pattern(self, column):
        """Find the best 23-bit pattern for a column by sampling 21 groups."""
        # Column shape: (480, 3)
        r, g, b = column[:, 0], column[:, 1], column[:, 2]
        rgb_values = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
        rgb_values_23 = (rgb_values >> 1) & 0x7FFFFF  # Take 23 most significant bits

        # Divide into 21 groups (20 groups of 23 pixels, 1 group of 20 pixels)
        group_size = 23
        num_groups = 21
        patterns = []
        for i in range(num_groups):
            start = i * group_size
            end = min(start + group_size, 480)
            group_values = rgb_values_23[start:end]
            if len(group_values) == 0:
                continue
            # Take the most common pattern in the group
            unique, counts = np.unique(group_values, return_counts=True)
            most_common = unique[np.argmax(counts)]
            patterns.append(most_common)

        # Test each pattern with 23 offsets
        max_matches = 0
        best_pattern = patterns[0]
        best_offset = 0
        for pattern in patterns:
            for offset in range(23):
                matches = self.count_matching_bits_23(rgb_values_23, pattern, offset)
                if matches > max_matches:
                    max_matches = matches
                    best_pattern = pattern
                    best_offset = offset

        # Apply the offset to the pattern
        shifted_pattern = ((best_pattern << best_offset) | (best_pattern >> (23 - best_offset))) & 0x7FFFFF
        # Convert back to RGB
        r_pattern = ((shifted_pattern >> 15) & 0xFF) << 1
        g_pattern = ((shifted_pattern >> 7) & 0xFF) << 1
        b_pattern = ((shifted_pattern << 1) & 0xFF)
        return np.array([r_pattern, g_pattern, b_pattern], dtype=np.uint8)

    def get_best_7bit_pattern(self, channel_values):
        """Find the best 7-bit pattern for a channel by sampling 69 groups."""
        # Channel_values shape: (480,)
        values_7 = (channel_values >> 1) & 0x7F  # Take 7 most significant bits

        # Divide into 69 groups (68 groups of 7 pixels, 1 group of 4 pixels)
        group_size = 7
        num_groups = 69
        patterns = []
        for i in range(num_groups):
            start = i * group_size
            end = min(start + group_size, 480)
            group_values = values_7[start:end]
            if len(group_values) == 0:
                continue
            unique, counts = np.unique(group_values, return_counts=True)
            most_common = unique[np.argmax(counts)]
            patterns.append(most_common)

        # Test each pattern with 7 offsets
        max_matches = 0
        best_pattern = patterns[0]
        best_offset = 0
        for pattern in patterns:
            for offset in range(7):
                matches = self.count_matching_bits_7(values_7, pattern, offset)
                if matches > max_matches:
                    max_matches = matches
                    best_pattern = pattern
                    best_offset = offset

        # Apply the offset to the pattern
        shifted_pattern = ((best_pattern << best_offset) | (best_pattern >> (7 - best_offset))) & 0x7F
        return (shifted_pattern << 1).astype(np.uint8)
    
    def transformer_vertical_bitfield_pattern(self, frame):
        """Create a new frame using the best 23-bit and 7-bit patterns with improved performance."""
        h, w, _ = frame.shape
        assert w == 640 and h == 480, "Expected frame size 640x480"

        # Step 1: Expand the frame to 483 pixels for 23-bit processing
        frame_expanded = np.zeros((self.height_23bit, w, 3), dtype=np.uint8)
        frame_expanded[self.expand_indices_23bit, :, :] = frame

        # Step 2: Find the best 23-bit pattern for all columns
        r, g, b = frame_expanded[:, :, 0], frame_expanded[:, :, 1], frame_expanded[:, :, 2]
        rgb_values = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
        rgb_values_23 = (rgb_values >> 1) & 0x7FFFFF  # Shape: (483, 640)

        # Divide into 21 groups
        group_size = 23
        num_groups = 21
        patterns = np.zeros((num_groups, w), dtype=np.uint32)
        for i in range(num_groups):
            start = i * group_size
            end = min(start + group_size, self.height_23bit)
            group_values = rgb_values_23[start:end, :]  # Shape: (group_size, 640)
            # Vectorize the unique operation across columns
            group_values_flat = group_values.reshape(-1)  # Shape: (group_size * 640,)
            col_indices = np.repeat(np.arange(w), end - start)  # Shape: (group_size * 640,)
            unique, indices, counts = np.unique(
                group_values_flat,
                return_inverse=True,
                return_counts=True
            )
            # Find the most common value per column
            max_counts = np.zeros(w, dtype=np.int32)
            most_common = np.zeros(w, dtype=np.uint32)
            for j in range(len(unique)):
                mask = (indices == j)
                col_counts = np.bincount(col_indices[mask], minlength=w)
                mask_better = col_counts > max_counts
                max_counts[mask_better] = col_counts[mask_better]
                most_common[mask_better] = unique[j]
            patterns[i] = most_common

        # Test each pattern with 23 offsets
        best_patterns = np.zeros(w, dtype=np.uint32)
        best_offsets = np.zeros(w, dtype=np.uint8)
        max_matches = np.zeros(w, dtype=np.int32)
        for offset in range(23):
            matches = self.count_matching_bits_23(rgb_values_23, patterns, offset)  # Shape: (21, 640)
            for i in range(num_groups):
                mask = matches[i] > max_matches
                best_patterns[mask] = patterns[i][mask]
                best_offsets[mask] = offset
                max_matches[mask] = matches[i][mask]

        # Apply the best patterns and offsets
        base_frame_expanded = np.zeros((self.height_23bit, w, 3), dtype=np.uint8)
        shifted_patterns = ((best_patterns << best_offsets) | (best_patterns >> (23 - best_offsets))) & 0x7FFFFF
        r_pattern = ((shifted_patterns >> 15) & 0xFF) << 1
        g_pattern = ((shifted_patterns >> 7) & 0xFF) << 1
        b_pattern = (shifted_patterns << 1) & 0xFF
        base_frame_expanded[:, :, 0] = r_pattern[np.newaxis, :]
        base_frame_expanded[:, :, 1] = g_pattern[np.newaxis, :]
        base_frame_expanded[:, :, 2] = b_pattern[np.newaxis, :]

        # Trim the base frame back to 480 pixels
        base_frame = base_frame_expanded[self.trim_indices_23bit, :, :]

        # Step 3: Compute the difference frame
        difference_frame = (frame.astype(np.int16) - base_frame.astype(np.int16)) % 256
        difference_frame = difference_frame.astype(np.uint8)

        # Step 4: Find the best 7-bit pattern for each channel
        difference_expanded = np.zeros((self.height_7bit, w, 3), dtype=np.uint8)
        difference_expanded[self.expand_indices_7bit, :, :] = difference_frame

        encoded_frame_expanded = base_frame_expanded.copy()
        for channel in range(3):
            channel_values = difference_expanded[:, :, channel]  # Shape: (483, 640)
            values_7 = (channel_values >> 1) & 0x7F  # Shape: (483, 640)

            # Divide into 69 groups
            group_size = 7
            num_groups = 69
            patterns = np.zeros((num_groups, w), dtype=np.uint8)
            for i in range(num_groups):
                start = i * group_size
                end = min(start + group_size, self.height_7bit)
                group_values = values_7[start:end, :]  # Shape: (group_size, 640)
                group_values_flat = group_values.reshape(-1)
                col_indices = np.repeat(np.arange(w), end - start)
                unique, indices, counts = np.unique(
                    group_values_flat,
                    return_inverse=True,
                    return_counts=True
                )
                max_counts = np.zeros(w, dtype=np.int32)
                most_common = np.zeros(w, dtype=np.uint8)
                for j in range(len(unique)):
                    mask = (indices == j)
                    col_counts = np.bincount(col_indices[mask], minlength=w)
                    mask_better = col_counts > max_counts
                    max_counts[mask_better] = col_counts[mask_better]
                    most_common[mask_better] = unique[j]
                patterns[i] = most_common

            # Test each pattern with 7 offsets
            best_patterns = np.zeros(w, dtype=np.uint8)
            best_offsets = np.zeros(w, dtype=np.uint8)
            max_matches = np.zeros(w, dtype=np.int32)
            for offset in range(7):
                matches = self.count_matching_bits_7(values_7, patterns, offset)  # Shape: (69, 640)
                for i in range(num_groups):
                    mask = matches[i] > max_matches
                    best_patterns[mask] = patterns[i][mask]
                    best_offsets[mask] = offset
                    max_matches[mask] = matches[i][mask]

            # Apply the best patterns
            shifted_patterns = ((best_patterns << best_offsets) | (best_patterns >> (7 - best_offsets))) & 0x7F
            pattern_values = (shifted_patterns << 1).astype(np.uint8)
            # Scale down the 7-bit contribution to preserve the base pattern
            scaled_pattern = (pattern_values // 4).astype(np.uint8)  # Reduce impact
            encoded_frame_expanded[:, :, channel] = (encoded_frame_expanded[:, :, channel].astype(np.int16) + scaled_pattern[np.newaxis, :]) % 256

        # Trim the encoded frame back to 480 pixels
        encoded_frame = encoded_frame_expanded[self.trim_indices_7bit, :, :]

        # Step 5: Create a 640-bit array for 4 groups of 20 8-bit colors
        color_group_bits = np.zeros(w, dtype=np.uint8)
        for col in range(w):
            col_matches_1 = frame[:, col, :] & self.pattern_frame_1[:, col, :]
            col_matches_0 = frame[:, col, :] & self.pattern_frame_0[:, col, :]
            total_matches_1 = np.sum(col_matches_1)
            total_matches_0 = np.sum(col_matches_0)
            bit = 1 if total_matches_1 >= total_matches_0 else 0
            col_matches_1_r = frame[:, col, 0] & self.pattern_frame_1[:, col, 0]
            col_matches_0_r = frame[:, col, 0] & self.pattern_frame_0[:, col, 0]
            total_matches_1_r = np.sum(col_matches_1_r)
            total_matches_0_r = np.sum(col_matches_0_r)
            bit_r = 1 if total_matches_1_r >= total_matches_0_r else 0
            group = (bit << 1) | bit_r
            color_group_bits[col] = group

        # Store the color group bits for later use
        self.color_group_bits = color_group_bits

        # Return the final frame
        return encoded_frame.astype(np.uint8)
    
    def transformer_compressed_frame_prototype(self, frame):
        """
        Compress a 640x480x24-bit frame to 1,234,944 bits (5:1 ratio).
        
        Args:
            frame: NumPy array of shape (480, 640, 3) with uint8 values.
        
        Returns:
            compressed_frame: Reconstructed frame.
            seed_map: 4-bit seed map (480, 640).
            palette_r, palette_g, palette_b: 256-value palettes for each channel.
        """
        # h, w, _ = frame.shape
        # assert h == 480 and w == 640, "Frame must be 640x480"

        # Compute 4-bit seed map (simplified: average RGB mod 16)
        # In practice, optimize this to reflect frame content
        seed_map = np.mean(frame, axis=2).astype(np.uint8) % 16  # Shape: (480, 640)

        # Define palettes (256 values each, 0-255 for simplicity)
        # Could be optimized based on frame’s color distribution
        palette_r = np.arange(256, dtype=np.uint8)
        palette_g = np.arange(256, dtype=np.uint8)
        palette_b = np.arange(256, dtype=np.uint8)

        # Get row and column indices
        rows, cols = np.indices((self.frame_h, self.frame_w), dtype=np.uint32)
        seed_map = seed_map.astype(np.uint32)

        # Compute palette indices for each channel
        r_index = (seed_map * 640 + cols) % 256
        g_index = (seed_map * 480 + rows) % 256
        b_index = (seed_map * 307200 + rows * 640 + cols) % 256

        # Map indices to palette values
        compressed_frame = np.zeros_like(frame)
        compressed_frame[:, :, 0] = palette_r[r_index]
        compressed_frame[:, :, 1] = palette_g[g_index]
        compressed_frame[:, :, 2] = palette_b[b_index]

        # Compressed data: seed_map (1,228,800 bits) + palettes (6,144 bits)
        return compressed_frame.astype(np.uint8) # , seed_map, palette_r, palette_g, palette_b
    
    def transformer_compressed_frame(self, frame):
        """
        Compress and reconstruct a 640x480x24-bit frame using 5:1 compression.
        
        Args:
            frame: NumPy array of shape (480, 640, 3) with uint8 values.
        
        Returns:
            reconstructed_frame: NumPy array of shape (480, 640, 3) with uint8 values.
        """
        compressed_data = self.encode_compressed_frame(frame)
        return self.decode_compressed_frame(compressed_data)
    
    def encode_compressed_frame(self, frame):
        """
        Encode a 640x480x24-bit frame into a 4-bit seed map and three 256-value palettes.
        
        Args:
            frame: NumPy array of shape (480, 640, 3) with uint8 values.
        
        Returns:
            tuple: (seed_map, palette_r, palette_g, palette_b)
                - seed_map: 4-bit seed map (480, 640), values 0-15.
                - palette_r, palette_g, palette_b: 256-value palettes for each channel (uint8).
        """
        h, w, _ = frame.shape
        assert h == 480 and w == 640, "Frame must be 640x480"

        # Compute 4-bit seed map (average RGB mod 16)
        seed_map = np.mean(frame, axis=2).astype(np.uint8) % 16  # Shape: (480, 640)

        # Compute palettes: most frequent 256 values per channel
        def get_palette(channel_data):
            hist, _ = np.histogram(channel_data.flatten(), bins=256, range=(0, 256))
            indices = np.argsort(-hist)
            palette = indices[:256].astype(np.uint8)
            palette = np.sort(palette)
            return palette

        palette_r = get_palette(frame[:, :, 0])
        palette_g = get_palette(frame[:, :, 1])
        palette_b = get_palette(frame[:, :, 2])

        return seed_map, palette_r, palette_g, palette_b
    
    def decode_compressed_frame(self, compressed_data):
        """
        Decode a compressed frame from a 4-bit seed map and three 256-value palettes.
        
        Args:
            compressed_data: tuple (seed_map, palette_r, palette_g, palette_b)
                - seed_map: 4-bit seed map (480, 640), values 0-15, stored as uint8.
                - palette_r, palette_g, palette_b: 256-value palettes for each channel (uint8).
        
        Returns:
            reconstructed_frame: NumPy array of shape (480, 640, 3) with uint8 values.
        """
        seed_map, palette_r, palette_g, palette_b = compressed_data
        h, w = seed_map.shape
        assert h == 480 and w == 640, "Seed map must be 480x640"
        assert len(palette_r) == 256 and len(palette_g) == 256 and len(palette_b) == 256, "Palettes must have 256 values"
        assert seed_map.dtype == np.uint8, "Seed map must be uint8"
        assert np.all(seed_map < 16), "Seed map values must be in range 0-15"

        # Get row and column indices
        rows, cols = np.indices((h, w), dtype=np.uint16)  # Use uint16 to save memory

        # Compute odd/even masks for rows and columns
        odd_rows = rows % 2
        odd_cols = cols % 2

        # Compute palette indices with adjusted formulas
        r_index = (seed_map * 17 + rows * 3 + cols + odd_rows * 5 + odd_cols * 7) % 256
        g_index = (seed_map * 23 + rows + cols * 3 + odd_cols * 11 + odd_rows * 13) % 256
        b_index = (seed_map * 29 + (rows + cols) * 2 + (odd_rows ^ odd_cols) * 19) % 256

        # Reconstruct the frame
        reconstructed_frame = np.zeros((h, w, 3), dtype=np.uint8)
        reconstructed_frame[:, :, 0] = palette_r[r_index]
        reconstructed_frame[:, :, 1] = palette_g[g_index]
        reconstructed_frame[:, :, 2] = palette_b[b_index]

        return reconstructed_frame
    
    def transformer_metrics(self, frame):
        """
        Analyze a frame and overlay metrics about unique values in the top-left corner using sets.
        
        Args:
            frame: NumPy array of shape (h, w, 3) with uint8 values (RGB).
        
        Returns:
            annotated_frame: NumPy array of shape (h, w, 3) with metrics overlaid.
        """
        # Ensure the frame is in the correct format
        assert frame.dtype == np.uint8, "Frame must be uint8"
        assert frame.shape[2] == 3, "Frame must have 3 channels (RGB)"
        h, w, _ = frame.shape

        # Compute unique 24-bit RGB values using a set
        # Convert pixels to tuples for hashing
        pixels = frame.reshape(-1, 3)
        unique_rgb_set = set(map(tuple, pixels))
        total_unique_rgb = len(unique_rgb_set)

        # Compute unique 8-bit values for R, G, B individually using sets
        r_values = frame[:, :, 0].flatten()
        g_values = frame[:, :, 1].flatten()
        b_values = frame[:, :, 2].flatten()
        unique_r_set = set(r_values)
        unique_g_set = set(g_values)
        unique_b_set = set(b_values)
        unique_r = len(unique_r_set)
        unique_g = len(unique_g_set)
        unique_b = len(unique_b_set)

        # Compute total unique 8-bit values (R, G, B combined) using a set
        # Use a single set for all values
        unique_8bit_set = unique_r_set | unique_g_set | unique_b_set
        total_unique_8bit = len(unique_8bit_set)

        # Create a copy of the frame to annotate
        annotated_frame = frame.copy()

        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White text
        thickness = 1
        line_spacing = 20
        start_x, start_y = 10, 20  # Top-left corner

        # Prepare the metrics text
        metrics = [
            f"Unique 24-bit RGB: {total_unique_rgb}",
            f"Unique R: {unique_r}",
            f"Unique G: {unique_g}",
            f"Unique B: {unique_b}",
            f"Unique 8-bit (all): {total_unique_8bit}"
        ]

        # Overlay each metric on the frame
        for i, metric in enumerate(metrics):
            y = start_y + i * line_spacing
            # Add a black background for better readability
            (text_w, text_h), _ = cv2.getTextSize(metric, font, font_scale, thickness)
            cv2.rectangle(
                annotated_frame,
                (start_x, y - text_h - 2),
                (start_x + text_w, y + 2),
                (0, 0, 0),  # Black background
                -1
            )
            # Add the text
            cv2.putText(
                annotated_frame,
                metric,
                (start_x, y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )

        return annotated_frame
    
    def encode_24_15bit_compress(self, frame):
        """
        Compress a 640x480 frame using a 15-bit lookup index and a 2^15 color palette.
        
        Args:
            frame: NumPy array of shape (480, 640, 3) with uint8 values (RGB).
        
        Returns:
            tuple: (lookup_map, palette)
                - lookup_map: Array of shape (480, 640) with uint16 values (15-bit indices).
                - palette: Array of shape (32768, 3) with uint8 values (RGB colors).
        """
        h, w, _ = frame.shape
        assert h == 480 and w == 640, "Frame must be 640x480"

        # Step 1: Extract unique 24-bit RGB colors
        pixels = frame.reshape(-1, 3)  # Shape: (307200, 3)
        unique_colors = np.unique(pixels, axis=0)  # Shape: (31455, 3)
        num_unique_colors = len(unique_colors)
        print(f"Number of unique colors: {num_unique_colors}")

        # Step 2: Create the palette (2^15 = 32768 slots)
        palette = np.zeros((32768, 3), dtype=np.uint8)
        if num_unique_colors <= 32768:
            # Lossless: Store all unique colors
            palette[:num_unique_colors] = unique_colors
        else:
            # Lossy: Select the top 32768 colors by frequency
            # Compute frequency of each color
            pixels_tuples = [tuple(pixel) for pixel in pixels]
            from collections import Counter
            color_counts = Counter(pixels_tuples)
            # Sort by frequency (most common first)
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
            top_colors = [np.array(color, dtype=np.uint8) for color, _ in sorted_colors[:32768]]
            palette = np.array(top_colors, dtype=np.uint8)

        # Step 3: Create a mapping from RGB to palette index
        # For lossless case, this is straightforward
        color_to_index = {tuple(color): idx for idx, color in enumerate(unique_colors)}

        # Step 4: Create the lookup map
        lookup_map = np.zeros((h, w), dtype=np.uint16)  # 15-bit indices
        for i in range(h):
            for j in range(w):
                pixel = tuple(frame[i, j])
                if pixel in color_to_index:
                    # Direct match (lossless)
                    lookup_map[i, j] = color_to_index[pixel]
                else:
                    # Find closest color in palette (lossy case)
                    pixel_array = np.array(pixel, dtype=np.int32)
                    distances = np.sum((palette - pixel_array) ** 2, axis=1)
                    closest_idx = np.argmin(distances)
                    lookup_map[i, j] = closest_idx

        return lookup_map, palette

    def decode_24_15bit_compress(self, lookup_map, palette):
        """
        Decompress a frame using a 15-bit lookup map and palette.
        
        Args:
            lookup_map: Array of shape (480, 640) with uint16 values (15-bit indices).
            palette: Array of shape (32768, 3) with uint8 values (RGB colors).
        
        Returns:
            reconstructed_frame: NumPy array of shape (480, 640, 3) with uint8 values.
        """
        h, w = lookup_map.shape
        reconstructed_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                idx = lookup_map[i, j]
                reconstructed_frame[i, j] = palette[idx]
        
        return reconstructed_frame

    def transformer_24_15bit_compress(self, frame):
        """
        Simulate 24-bit to 15-bit + palette compression and decompression.
        
        Args:
            frame: NumPy array of shape (480, 640, 3) with uint8 values (RGB).
        
        Returns:
            reconstructed_frame: NumPy array of shape (480, 640, 3) with uint8 values.
        """
        lookup_map, palette = self.encode_24_15bit_compress(frame)
        reconstructed_frame = self.decode_24_15bit_compress(lookup_map, palette)
        return reconstructed_frame
    
    def transformer_analytics_frame_match_dummy(self, frame):
        """
        Adds info from the analytics_fram_match function to the frame and returns it.
        """
        return self.analytics_frame_match(frame, frame)
    
    def analytics_frame_match(self, original_frame, output_frame):
        """
        Compare original and output frames, compute bit-level and frame-level match percentages,
        and overlay the metrics on the output frame.
        
        Args:
            original_frame: NumPy array of shape (h, w, 3) with uint8 values (RGB).
            output_frame: NumPy array of shape (h, w, 3) with uint8 values (RGB).
        
        Returns:
            annotated_frame: NumPy array of shape (h, w, 3) with metrics overlaid.
        """
        # Ensure frames are compatible
        assert original_frame.shape == output_frame.shape, "Frames must have the same shape"
        assert original_frame.dtype == np.uint8 and output_frame.dtype == np.uint8, "Frames must be uint8"
        assert original_frame.shape[2] == 3, "Frames must have 3 channels (RGB)"
        h, w, _ = original_frame.shape

        # Total number of pixels and bits
        total_pixels = h * w
        total_bits = total_pixels * 24  # 24 bits per pixel (RGB)

        # Compute bit-level match
        # Convert RGB to 24-bit integers: R * 2^16 + G * 2^8 + B
        original_int = (original_frame[:, :, 0].astype(np.uint32) << 16) + \
                    (original_frame[:, :, 1].astype(np.uint32) << 8) + \
                        original_frame[:, :, 2].astype(np.uint32)
        output_int = (output_frame[:, :, 0].astype(np.uint32) << 16) + \
                    (output_frame[:, :, 1].astype(np.uint32) << 8) + \
                    output_frame[:, :, 2].astype(np.uint32)

        # XOR to find differing bits
        xor_result = original_int ^ output_int

        # Count matching bits (where XOR is 0)
        # We can use np.unpackbits to convert to bits, but it's faster to count 0s in the XOR
        # Use a loop over bits (0 to 23) to count matches
        matching_bits = 0
        for bit in range(24):
            bit_mask = 1 << bit
            matching_bits += np.sum((xor_result & bit_mask) == 0)
        bit_match_percent = (matching_bits / total_bits) * 100

        # Compute frame-level match per channel
        r_matches = np.sum(original_frame[:, :, 0] == output_frame[:, :, 0])
        g_matches = np.sum(original_frame[:, :, 1] == output_frame[:, :, 1])
        b_matches = np.sum(original_frame[:, :, 2] == output_frame[:, :, 2])
        r_match_percent = (r_matches / total_pixels) * 100
        g_match_percent = (g_matches / total_pixels) * 100
        b_match_percent = (b_matches / total_pixels) * 100

        # Create a copy of the output frame to annotate
        annotated_frame = output_frame.copy()

        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White text
        thickness = 1
        line_spacing = 20
        start_x, start_y = 10, 20  # Top-left corner

        # Prepare the metrics text
        metrics = [
            f"Bit Match: {bit_match_percent:.2f}%",
            f"R Match: {r_match_percent:.2f}%",
            f"G Match: {g_match_percent:.2f}%",
            f"B Match: {b_match_percent:.2f}%"
        ]

        # Overlay each metric on the frame
        for i, metric in enumerate(metrics):
            y = start_y + i * line_spacing
            (text_w, text_h), _ = cv2.getTextSize(metric, font, font_scale, thickness)
            cv2.rectangle(
                annotated_frame,
                (start_x, y - text_h - 2),
                (start_x + text_w, y + 2),
                (0, 0, 0),  # Black background
                -1
            )
            cv2.putText(
                annotated_frame,
                metric,
                (start_x, y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )

        return annotated_frame
    
    def transformer_incremental(self, frame):
        # Encode the frame
        encoded_data = self.encode_incremental(frame)
        
        # Decode using the encoded data
        decoded_frame = self.decode_incremental(encoded_data)
        
        return decoded_frame

    def encode_incremental(self, frame):
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

    # Decode function (for incremental reconstruction from constraints):
    def decode_incremental(self, encoded_data):
        h, w = self.frame_h, self.frame_w  # set these in __init__
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
    
    def transformer_bit_depth_strobe(self, frame):
        """
        Transform the frame by transmitting one bit layer per frame for all RGB channels.
        
        Args:
            frame: np.ndarray of shape (480, 640, 3) with uint8 values (RGB).
        
        Returns:
            np.ndarray: Updated frame with the current bit layer applied.
        """
        # Initialize persistent state if not set
        if not hasattr(self, 'bit_depth_strobe_persistent_frame'):
            self.bit_depth_strobe_persistent_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.bit_depth_strobe_current_bit = 0
        
        # Encode the current bit layer
        encoded_data = self.encode_bit_depth_strobe(frame)
        
        # Decode into the persistent frame
        output_frame = self.decode_bit_depth_strobe(encoded_data)
        
        # Increment the bit layer, cycling back to 0 after 7
        self.bit_depth_strobe_current_bit = (self.bit_depth_strobe_current_bit + 1) % 8
        
        return output_frame
    
    def encode_bit_depth_strobe(self, frame):
        """
        Encode the frame by extracting the current bit layer for all RGB channels.
        
        Args:
            frame: np.ndarray of shape (480, 640, 3) with uint8 values (RGB).
        
        Returns:
            tuple: (bit_layer, mask)
                - bit_layer: Integer (0-7) indicating the current bit being transmitted.
                - mask: np.ndarray of shape (480, 640, 3) with 1-bit values (0 or 1) for R, G, B.
        """
        assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8, "Frame must be 480x640x3 uint8"
        
        # Extract the current bit layer from all channels
        bit_layer = self.bit_depth_strobe_current_bit
        mask = (frame >> bit_layer) & 1  # Shape: (480, 640, 3), values 0 or 1
        
        return bit_layer, mask
    
    def decode_bit_depth_strobe(self, encoded_data):
        """
        Decode the bit layer into the persistent frame, replacing the specified bit.
        
        Args:
            encoded_data: tuple (bit_layer, mask)
                - bit_layer: Integer (0-7) indicating the bit being updated.
                - mask: np.ndarray of shape (480, 640, 3) with 1-bit values (0 or 1).
        
        Returns:
            np.ndarray: Updated persistent frame with the new bit layer applied.
        """
        bit_layer, mask = encoded_data
        assert mask.shape == (480, 640, 3) and mask.dtype == np.uint8, "Mask must be 480x640x3 uint8"
        assert 0 <= bit_layer <= 7, "Bit layer must be between 0 and 7"
        
        # Compute bit_mask directly in uint8 to avoid deprecation warning
        bit_mask = (~np.uint8(1 << bit_layer)).astype(np.uint8)  # e.g., ~0b100 = 0b11111011 = 251
        
        # Clear the current bit layer in the persistent frame
        self.bit_depth_strobe_persistent_frame = (
            self.bit_depth_strobe_persistent_frame & bit_mask
        ).astype(np.uint8)
        
        # Set the new bit layer
        self.bit_depth_strobe_persistent_frame = (
            self.bit_depth_strobe_persistent_frame | (mask << bit_layer)
        ).astype(np.uint8)
        
        return self.bit_depth_strobe_persistent_frame.copy()
    
    def transformer_bit_depth_resolve_prototype(self, frame):
        """
        Transform the frame by updating the bit layer with the most changes for one RGB channel.
        
        Args:
            frame: np.ndarray of shape (480, 640, 3) with uint8 values (RGB).
        
        Returns:
            np.ndarray: Updated frame with the selected bit layer applied.
        """
        # Initialize persistent state if not set
        if not hasattr(self, 'bit_depth_resolve_persistent_frame'):
            self.bit_depth_resolve_persistent_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Encode the frame, choosing the most changed bit layer
        encoded_data = self.encode_bit_depth_resolve_prototype(frame)
        
        # Decode into the persistent frame
        output_frame = self.decode_bit_depth_resolve_prototype(encoded_data)
        
        return output_frame
    
    def encode_bit_depth_resolve_prototype(self, frame):
        """
        Encode the frame by selecting the RGB channel and bit layer with the most changes.
        
        Args:
            frame: np.ndarray of shape (480, 640, 3) with uint8 values (RGB).
        
        Returns:
            tuple: (channel, bit_layer, mask)
                - channel: Integer (0=R, 1=G, 2=B) indicating the updated channel.
                - bit_layer: Integer (0-7) indicating the bit being updated.
                - mask: np.ndarray of shape (480, 640) with 1-bit values (0 or 1) for the chosen channel.
        """
        assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8, "Frame must be 480x640x3 uint8"
        
        # Compute differences for each channel and bit layer
        changes = np.zeros((3, 8), dtype=np.int32)  # [channel, bit] → number of differing bits

        for channel in range(3):  # R, G, B
            current_bits = self.bit_depth_resolve_persistent_frame[:, :, channel]
            new_bits = frame[:, :, channel]

            for bit in range(8):
                current_mask = (current_bits >> bit) & 1
                new_mask = (new_bits >> bit) & 1
                changes[channel, bit] = np.sum(current_mask != new_mask)
        
        # Find the channel and bit with the most changes
        channel, bit_layer = np.unravel_index(np.argmax(changes), changes.shape)
        
        # Extract the bit mask for the chosen channel and layer
        mask = (frame[:, :, channel] >> bit_layer) & 1  # Shape: (480, 640)
        
        return channel, bit_layer, mask
    
    def decode_bit_depth_resolve_prototype(self, encoded_data):
        """
        Decode the bit layer into the persistent frame for the specified channel.
        
        Args:
            encoded_data: tuple (channel, bit_layer, mask)
                - channel: Integer (0=R, 1=G, 2=B) indicating the updated channel.
                - bit_layer: Integer (0-7) indicating the bit being updated.
                - mask: np.ndarray of shape (480, 640) with 1-bit values (0 or 1).
        
        Returns:
            np.ndarray: Updated persistent frame with the new bit layer applied.
        """
        channel, bit_layer, mask = encoded_data
        assert mask.shape == (480, 640) and mask.dtype == np.uint8, "Mask must be 480x640 uint8"
        assert 0 <= channel <= 2, "Channel must be 0, 1, or 2"
        assert 0 <= bit_layer <= 7, "Bit layer must be between 0 and 7"
        
        # Compute bit_mask as uint8
        bit_mask = np.uint8(~(1 << bit_layer))  # e.g., for bit 2: ~0b100 = 0b11111011 = 251
        
        # Clear the specified bit layer in the chosen channel
        self.bit_depth_resolve_persistent_frame[:, :, channel] = (
            self.bit_depth_resolve_persistent_frame[:, :, channel] & bit_mask
        ).astype(np.uint8)
        
        # Set the new bit layer
        self.bit_depth_resolve_persistent_frame[:, :, channel] = (
            self.bit_depth_resolve_persistent_frame[:, :, channel] | (mask << bit_layer)
        ).astype(np.uint8)
        
        return self.bit_depth_resolve_persistent_frame.copy()
    
    def encode_bit_depth_resolve(self, frame):
        """
        Encode the frame by selecting the bit layer for each RGB channel with the most
        perceptually significant changes compared to the previous original frame.
        
        Args:
            frame: np.ndarray of shape (480, 640, 3) with uint8 values (RGB).
        
        Returns:
            tuple: (update_map, bit_layers)
                - update_map: np.ndarray of shape (480, 640, 3) with 1-bit values (0 or 1) for R, G, B.
                - bit_layers: np.ndarray of shape (3,) with integers (0-7) for R, G, B bit layers.
        """
        assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8, "Frame must be 480x640x3 uint8"
        
        # Use zeros for the first frame if no previous frame exists
        prev_frame = (self.bit_depth_resolve_prev_frame if self.bit_depth_resolve_prev_frame is not None 
                    else np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Initialize arrays
        changes = np.zeros((3, 8), dtype=np.int32)  # [channel, bit] → number of differing bits
        weighted_changes = np.zeros((3, 8), dtype=np.float32)  # Weighted by bit significance
        bit_layers = np.zeros(3, dtype=np.uint8)    # Chosen bit layer for R, G, B
        update_map = np.zeros((480, 640, 3), dtype=np.uint8)  # 1-bit update map
        
        # Compute changes and weight them by bit significance
        for channel in range(3):  # R, G, B
            prev_bits = prev_frame[:, :, channel]
            new_bits = frame[:, :, channel]
            for bit in range(8):
                prev_mask = (prev_bits >> bit) & 1
                new_mask = (new_bits >> bit) & 1
                changes[channel, bit] = np.sum(prev_mask != new_mask)
                # Weight by the bit's magnitude (2^bit)
                weighted_changes[channel, bit] = changes[channel, bit] * (1 << bit)
            bit_layers[channel] = np.argmax(weighted_changes[channel])  # Most significant changes
            update_map[:, :, channel] = (new_bits >> bit_layers[channel]) & 1  # Update map for this channel
        
        return update_map, bit_layers
    
    def decode_bit_depth_resolve(self, encoded_data):
        """
        Decode the update map into the persistent frame for all three RGB channels.
        
        Args:
            encoded_data: tuple (update_map, bit_layers)
                - update_map: np.ndarray of shape (480, 640, 3) with 1-bit values (0 or 1) for R, G, B.
                - bit_layers: np.ndarray of shape (3,) with integers (0-7) for R, G, B bit layers.
        
        Returns:
            np.ndarray: Updated persistent frame with the new bit layers applied to all channels.
        """
        update_map, bit_layers = encoded_data
        assert update_map.shape == (480, 640, 3) and update_map.dtype == np.uint8, "Update map must be 480x640x3 uint8"
        assert bit_layers.shape == (3,) and np.all((0 <= bit_layers) & (bit_layers <= 7)), "Bit layers must be 0-7"
        
        # Update each channel
        for channel in range(3):
            bit_layer = bit_layers[channel]
            # Compute bit_mask directly in uint8 to avoid deprecation warning
            bit_mask = (~np.uint8(1 << bit_layer)).astype(np.uint8)  # e.g., ~0b100 = 0b11111011 = 251
            
            # Clear the bit layer in the persistent frame
            self.bit_depth_resolve_persistent_frame[:, :, channel] = (
                self.bit_depth_resolve_persistent_frame[:, :, channel] & bit_mask
            ).astype(np.uint8)
            
            # Set the new bit layer from the update map
            self.bit_depth_resolve_persistent_frame[:, :, channel] = (
                self.bit_depth_resolve_persistent_frame[:, :, channel] | 
                (update_map[:, :, channel] << bit_layer)
            ).astype(np.uint8)
        
        return self.bit_depth_resolve_persistent_frame.copy()
    
    def transformer_bit_depth_resolve(self, frame):
        if not hasattr(self, 'bit_depth_resolve_persistent_frame'):
            self.bit_depth_resolve_persistent_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.bit_depth_resolve_prev_frame = None
        
        # Encode using the previous original frame for comparison
        encoded_data = self.encode_bit_depth_resolve(frame)
        output_frame = self.decode_bit_depth_resolve(encoded_data)
        
        # # Debug output
        # update_map, bit_layers = encoded_data
        # # Recompute changes for debug
        # prev_frame = (self.bit_depth_resolve_prev_frame if self.bit_depth_resolve_prev_frame is not None 
        #             else np.zeros((480, 640, 3), dtype=np.uint8))
        # changes = np.zeros((3, 8), dtype=np.int32)
        # weighted_changes = np.zeros((3, 8), dtype=np.float32)
        # for channel in range(3):
        #     prev_bits = prev_frame[:, :, channel]
        #     new_bits = frame[:, :, channel]
        #     for bit in range(8):
        #         prev_mask = (prev_bits >> bit) & 1
        #         new_mask = (new_bits >> bit) & 1
        #         changes[channel, bit] = np.sum(prev_mask != new_mask)
        #         weighted_changes[channel, bit] = changes[channel, bit] * (1 << bit)
        # print(f"R Bit {bit_layers[0]} (Raw: {changes[0, bit_layers[0]]}, Weighted: {weighted_changes[0, bit_layers[0]]}), "
        #     f"G Bit {bit_layers[1]} (Raw: {changes[1, bit_layers[1]]}, Weighted: {weighted_changes[1, bit_layers[1]]}), "
        #     f"B Bit {bit_layers[2]} (Raw: {changes[2, bit_layers[2]]}, Weighted: {weighted_changes[2, bit_layers[2]]})")
        
        self.bit_depth_resolve_prev_frame = frame.copy()
        return output_frame
    
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