import sys
from collections import defaultdict

def analyze_bit_patterns(file_path, chunk_size=1):
    """
    Read a file as binary, count frequency of chunk_size-byte patterns.
    Args:
        file_path (str): Path to input file.
        chunk_size (int): Size of patterns in bytes (e.g., 1 = bytes, 4 = 32-bit).
    Returns:
        dict: {pattern (bytes): count (int)}
    """
    # Use defaultdict to avoid key checks
    patterns = defaultdict(int)
    
    try:
        with open(file_path, 'rb') as f:
            while True:
                # Read chunk_size bytes at a time
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                # If last chunk is smaller, pad or skip (here, skip)
                if len(chunk) == chunk_size:
                    patterns[chunk] += 1
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    return dict(patterns)

def analyze_counts(patterns):
    gt2_entries = [(p, c) for p, c in patterns.items() if c > 2]
    eq2_entries = [(p, c) for p, c in patterns.items() if c == 2]
    eq1_entries = [(p, c) for p, c in patterns.items() if c == 1]
    
    gt2_total = sum(count for _, count in gt2_entries)
    eq2_total = sum(count for _, count in eq2_entries)
    eq1_total = len(eq1_entries)  # Same as sum since all are 1
    
    print(f"\nPattern count analysis:")
    print(f"Patterns occurring >2 times: {len(gt2_entries)} patterns, {gt2_total} total occurrences")
    print(f"Patterns occurring =2 times: {len(eq2_entries)} patterns, {eq2_total} total occurrences")
    print(f"Patterns occurring =1 time:  {len(eq1_entries)} patterns, {eq1_total} total occurrences")

def print_pattern_frequencies(patterns):
    """Print patterns and their frequencies, sorted by count descending."""
    if not patterns:
        print("No patterns to display.")
        return
    
    # Sort by frequency (descending), then pattern
    sorted_patterns = sorted(patterns.items(), key=lambda x: (-x[1], x[0])) #[:256]  # Take top 100
    total_count = sum(patterns.values())
    print(f"Total chunks: {total_count}")
    print("Pattern (hex) : Count : Frequency (%)")
    for pattern, count in sorted_patterns:
        freq = (count / total_count) * 100
        # Convert bytes to hex for readability
        pattern_hex = pattern.hex()
        print(f"{pattern_hex:>12} : {count:>5} : {freq:.2f}%")

def main():
    # Check command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Analyze with 1-byte chunks (8-bit patterns)
    patterns = analyze_bit_patterns(file_path, chunk_size=1)
    if patterns:
        analyze_counts(patterns)
        print_pattern_frequencies(patterns)

if __name__ == "__main__":
    main()