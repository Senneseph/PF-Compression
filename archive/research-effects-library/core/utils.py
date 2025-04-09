"""
Utility functions for the PF-Compression library.
"""
import numpy as np
import cv2
import importlib.util
import sys
import os

def validate_frame(frame, expected_shape=None, expected_dtype=np.uint8):
    """
    Validate that the frame has the expected shape and data type.
    
    Args:
        frame: NumPy array to validate.
        expected_shape: Tuple of expected shape (height, width, channels) or None.
        expected_dtype: Expected data type of the frame.
        
    Returns:
        frame: The validated frame, converted to the expected data type if necessary.
        
    Raises:
        ValueError: If the frame does not have the expected shape.
    """
    # Ensure the frame is a NumPy array
    if not isinstance(frame, np.ndarray):
        raise ValueError("Frame must be a NumPy array")
    
    # Ensure the frame has the expected shape
    if expected_shape is not None:
        if frame.shape != expected_shape:
            raise ValueError(f"Frame must have shape {expected_shape}, got {frame.shape}")
    
    # Ensure the frame has the expected data type
    if frame.dtype != expected_dtype:
        frame = frame.astype(expected_dtype)
    
    return frame

def precompute_nearest_primes(max_value):
    """
    Precompute a lookup table for the nearest prime number for values from 0 to max_value.
    
    Args:
        max_value: Maximum value to precompute.
        
    Returns:
        np.ndarray: Lookup table of nearest prime numbers.
    """
    # List of primes up to 255
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 
              73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 
              157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 
              239, 241, 251]
    
    # Create lookup table
    lookup = np.zeros(max_value + 1, dtype=np.uint8)
    
    # For each value, find the nearest prime
    for i in range(max_value + 1):
        # Find the nearest prime
        nearest_prime = min(primes, key=lambda x: abs(x - i))
        lookup[i] = nearest_prime
    
    return lookup

def precompute_nearest_fibonacci(max_value):
    """
    Precompute a lookup table for the nearest Fibonacci number for values from 0 to max_value.
    
    Args:
        max_value: Maximum value to precompute.
        
    Returns:
        np.ndarray: Lookup table of nearest Fibonacci numbers.
    """
    # Generate Fibonacci numbers up to max_value
    fibs = [0, 1]
    while fibs[-1] + fibs[-2] <= max_value:
        fibs.append(fibs[-1] + fibs[-2])
    
    # Create lookup table
    lookup = np.zeros(max_value + 1, dtype=np.uint8)
    
    # For each value, find the nearest Fibonacci number
    for i in range(max_value + 1):
        # Find the nearest Fibonacci number
        nearest_fib = min(fibs, key=lambda x: abs(x - i))
        lookup[i] = nearest_fib
    
    return lookup

def generate_pythagorean_triples(max_value=255):
    """
    Generate Pythagorean triples (a, b, c) where a^2 + b^2 = c^2 and a, b, c <= max_value.
    
    Args:
        max_value: Maximum value for a, b, c.
        
    Returns:
        list: List of Pythagorean triples as tuples (a, b, c).
    """
    triples = []
    
    # Generate primitive triples using Euclid's formula
    for m in range(2, int(np.sqrt(max_value)) + 1):
        for n in range(1, m):
            if (m - n) % 2 == 1 and np.gcd(m, n) == 1:
                a = m*m - n*n
                b = 2*m*n
                c = m*m + n*n
                
                if c <= max_value:
                    # Add primitive triple
                    triples.append((a, b, c))
                    
                    # Add multiples of primitive triple
                    k = 2
                    while k*c <= max_value:
                        triples.append((k*a, k*b, k*c))
                        k += 1
    
    return triples

def import_from_path(module_name, file_path):
    """
    Import a module from a file path.
    
    Args:
        module_name: Name to give the imported module.
        file_path: Path to the module file.
        
    Returns:
        module: The imported module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def load_effects(effects_dir):
    """
    Load all effect modules from a directory.
    
    Args:
        effects_dir: Directory containing effect modules.
        
    Returns:
        dict: Dictionary mapping effect names to effect classes.
    """
    effects = {}
    
    # Get all Python files in the directory
    for file_name in os.listdir(effects_dir):
        if file_name.endswith('.py') and not file_name.startswith('__'):
            # Get the module name (file name without extension)
            module_name = os.path.splitext(file_name)[0]
            
            # Import the module
            file_path = os.path.join(effects_dir, file_name)
            module = import_from_path(module_name, file_path)
            
            # Find effect classes in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                # Check if it's a class that ends with 'Effect'
                if isinstance(attr, type) and attr_name.endswith('Effect'):
                    # Create an instance of the effect
                    effect = attr()
                    effects[effect.name] = effect
    
    return effects
