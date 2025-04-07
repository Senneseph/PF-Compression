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
import importlib.util
import sys

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
        Panavision1970sEffect(),
        CybergridEffect(),
        MacroblastEffect(),
        RetroFlashyEffect()
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
