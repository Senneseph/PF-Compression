"""
Basic usage example for the PF-Compression library.
"""
import cv2
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lib.transformer.delta_rgb_pix import DeltaRGBPixTransformer
from src.lib.transformer.pythagorean_triple import PythagoreanTripleTransformer
from src.lib.transformer.even_odd_color import EvenOddColorTransformer
from src.lib.transformer.prime_rgb import PrimeRGBTransformer
from src.lib.transformer.fibonacci_rgb import FibonacciRGBTransformer
from src.lib.transformer.fibonacci import FibonacciTransformer
from src.lib.transformer.retro import RetroTransformer
from src.lib.transformer.dummy import DummyTransformer
from src.lib.filter.middle_four_bits import MiddleFourBitsFilter
from src.lib.filter.color_negative import ColorNegativeFilter
from src.lib.filter.panavision_1970s import Panavision1970sFilter

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

    # Create instances of our transformers and filters
    delta_rgb_transformer = DeltaRGBPixTransformer(frame_shape)
    pythagorean_triple_transformer = PythagoreanTripleTransformer()
    even_odd_color_transformer = EvenOddColorTransformer(frame_shape)
    prime_rgb_transformer = PrimeRGBTransformer()
    fibonacci_rgb_transformer = FibonacciRGBTransformer()
    fibonacci_transformer = FibonacciTransformer()
    retro_transformer = RetroTransformer()
    dummy_transformer = DummyTransformer()
    middle_four_bits_filter = MiddleFourBitsFilter()
    color_negative_filter = ColorNegativeFilter()
    panavision_filter = Panavision1970sFilter()

    # Create windows for displaying the results
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Delta RGB', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Pythagorean Triple', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Even/Odd Color', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Prime RGB', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Fibonacci RGB', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Fibonacci', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Retro', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Middle Four Bits', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Color Negative', cv2.WINDOW_NORMAL)
    cv2.namedWindow('1970s Panavision', cv2.WINDOW_NORMAL)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Apply the transformers and filters
        delta_rgb_frame = delta_rgb_transformer.transform(frame)
        pythagorean_triple_frame = pythagorean_triple_transformer.transform(frame)
        even_odd_color_frame = even_odd_color_transformer.transform(frame)
        prime_rgb_frame = prime_rgb_transformer.transform(frame)
        fibonacci_rgb_frame = fibonacci_rgb_transformer.transform(frame)
        fibonacci_frame = fibonacci_transformer.transform(frame)
        retro_frame = retro_transformer.transform(frame)
        dummy_frame = dummy_transformer.transform(frame)  # This should be identical to the original frame
        middle_four_bits_frame = middle_four_bits_filter.filter(frame)
        color_negative_frame = color_negative_filter.filter(frame)
        panavision_frame = panavision_filter.filter(frame)

        # Display the results
        cv2.imshow('Original', frame)
        cv2.imshow('Delta RGB', delta_rgb_frame)
        cv2.imshow('Pythagorean Triple', pythagorean_triple_frame)
        cv2.imshow('Even/Odd Color', even_odd_color_frame)
        cv2.imshow('Prime RGB', prime_rgb_frame)
        cv2.imshow('Fibonacci RGB', fibonacci_rgb_frame)
        cv2.imshow('Fibonacci', fibonacci_frame)
        cv2.imshow('Retro', retro_frame)
        cv2.imshow('Middle Four Bits', middle_four_bits_frame)
        cv2.imshow('Color Negative', color_negative_frame)
        cv2.imshow('1970s Panavision', panavision_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
