"""
Webcam application using the PF-Compression Effects Library.

This is a simplified version of the original webcam.py that uses the new library structure.
"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import importlib.util
import sys
import os
import time
from PIL import Image, ImageTk

# Add the parent directory to the path so we can import the library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Helper function to import modules with hyphens in their path
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import effect modules
delta_rgb_pix = import_from_path('delta_rgb_pix', '../research-effects-library/effects/delta_rgb_pix.py')
prime_rgb = import_from_path('prime_rgb', '../research-effects-library/effects/prime_rgb.py')
fibonacci_rgb = import_from_path('fibonacci_rgb', '../research-effects-library/effects/fibonacci_rgb.py')
color_negative = import_from_path('color_negative', '../research-effects-library/effects/color_negative.py')
middle_four_bits = import_from_path('middle_four_bits', '../research-effects-library/effects/middle_four_bits.py')
panavision_1970s = import_from_path('panavision_1970s', '../research-effects-library/effects/panavision_1970s.py')
cybergrid = import_from_path('cybergrid', '../research-effects-library/effects/cybergrid.py')
macroblast = import_from_path('macroblast', '../research-effects-library/effects/macroblast.py')
retro_flashy = import_from_path('retro_flashy', '../research-effects-library/effects/retro_flashy.py')

# Import effect classes
DeltaRGBPixEffect = delta_rgb_pix.DeltaRGBPixEffect
PrimeRGBEffect = prime_rgb.PrimeRGBEffect
FibonacciRGBEffect = fibonacci_rgb.FibonacciRGBEffect
ColorNegativeEffect = color_negative.ColorNegativeEffect
MiddleFourBitsEffect = middle_four_bits.MiddleFourBitsEffect
Panavision1970sEffect = panavision_1970s.Panavision1970sEffect
CybergridEffect = cybergrid.CybergridEffect
MacroblastEffect = macroblast.MacroblastEffect
RetroFlashyEffect = retro_flashy.RetroFlashyEffect

class VideoApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Set up video capture
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_shape = (self.height, self.width, 3)
        
        # Create a canvas for displaying the video
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create a frame for controls
        control_frame = ttk.Frame(window)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Create effect selection dropdown
        ttk.Label(control_frame, text="Effect:").pack(side=tk.LEFT, padx=5)
        self.effect_var = tk.StringVar()
        
        # Create effect instances
        self.effects = {
            "None": None,
            "Delta RGB": DeltaRGBPixEffect(self.frame_shape),
            "Prime RGB": PrimeRGBEffect(),
            "Fibonacci RGB": FibonacciRGBEffect(),
            "Color Negative": ColorNegativeEffect(),
            "Middle Four Bits": MiddleFourBitsEffect(),
            "1970s Panavision": Panavision1970sEffect(),
            "Cybergrid": CybergridEffect(),
            "Macroblast": MacroblastEffect(),
            "Retro Flashy": RetroFlashyEffect()
        }
        
        effect_dropdown = ttk.Combobox(control_frame, textvariable=self.effect_var, 
                                       values=list(self.effects.keys()))
        effect_dropdown.pack(side=tk.LEFT, padx=5)
        effect_dropdown.current(0)
        
        # Reset button
        reset_button = ttk.Button(control_frame, text="Reset Effect", command=self.reset_effect)
        reset_button.pack(side=tk.LEFT, padx=5)
        
        # Quit button
        quit_button = ttk.Button(control_frame, text="Quit", command=self.quit)
        quit_button.pack(side=tk.RIGHT, padx=5)
        
        # Start the video loop
        self.delay = 15
        self.update()
        
        self.window.protocol("WM_DELETE_WINDOW", self.quit)
        self.window.mainloop()
    
    def update(self):
        # Read a frame from the video source
        ret, frame = self.cap.read()
        
        if ret:
            # Apply the selected effect
            effect_name = self.effect_var.get()
            effect = self.effects.get(effect_name)
            
            if effect:
                try:
                    frame = effect.transform(frame)
                except Exception as e:
                    print(f"Error applying effect {effect_name}: {e}")
            
            # Convert the frame to a format suitable for tkinter
            self.photo = self.convert_frame_to_photo(frame)
            
            # Update the canvas with the new frame
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        # Schedule the next update
        self.window.after(self.delay, self.update)
    
    def convert_frame_to_photo(self, frame):
        # Convert from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Convert to PhotoImage
        return ImageTk.PhotoImage(image=pil_image)
    
    def reset_effect(self):
        effect_name = self.effect_var.get()
        effect = self.effects.get(effect_name)
        
        if effect:
            effect.reset()
            print(f"Reset effect: {effect_name}")
    
    def quit(self):
        # Release the video source when the object is destroyed
        if self.cap.isOpened():
            self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    # Create a window and pass it to the VideoApp
    VideoApp(tk.Tk(), "PF-Compression Effects Demo")
