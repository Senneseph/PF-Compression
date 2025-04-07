"""
Demo script for the PF-Compression Effects Library.
"""
import cv2
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import effects from the research-effects-library
from research_effects_library.effects.delta_rgb_pix import DeltaRGBPixEffect
from research_effects_library.effects.prime_rgb import PrimeRGBEffect
from research_effects_library.effects.fibonacci_rgb import FibonacciRGBEffect
from research_effects_library.effects.color_negative import ColorNegativeEffect
from research_effects_library.effects.middle_four_bits import MiddleFourBitsEffect
from research_effects_library.effects.panavision_1970s import Panavision1970sEffect

def main():
    # Open a video capture device (webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get the frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_shape = (height, width, 3)
    
    # Create instances of our effects
    effects = [
        DeltaRGBPixEffect(frame_shape),
        PrimeRGBEffect(),
        FibonacciRGBEffect(),
        ColorNegativeEffect(),
        MiddleFourBitsEffect(),
        Panavision1970sEffect()
    ]
    
    # Create windows for displaying the results
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    for effect in effects:
        cv2.namedWindow(effect.name, cv2.WINDOW_NORMAL)
    
    # Current effect index
    current_effect_idx = 0
    
    print("Press 'n' to cycle to the next effect")
    print("Press 'p' to cycle to the previous effect")
    print("Press 'r' to reset the current effect")
    print("Press 'q' to quit")
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Apply the current effect
        current_effect = effects[current_effect_idx]
        transformed_frame = current_effect.transform(frame)
        
        # Display the results
        cv2.imshow('Original', frame)
        cv2.imshow(current_effect.name, transformed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Quit
            break
        elif key == ord('n'):
            # Next effect
            current_effect_idx = (current_effect_idx + 1) % len(effects)
            print(f"Switched to effect: {effects[current_effect_idx].name}")
        elif key == ord('p'):
            # Previous effect
            current_effect_idx = (current_effect_idx - 1) % len(effects)
            print(f"Switched to effect: {effects[current_effect_idx].name}")
        elif key == ord('r'):
            # Reset the current effect
            effects[current_effect_idx].reset()
            print(f"Reset effect: {effects[current_effect_idx].name}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
