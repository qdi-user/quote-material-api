import pandas as pd
import argparse
from collections import Counter

def analyze_printing_method_sizes(file_path, printing_method):
    """
    Analyzes the most common sizes (width x height) for a specific printing method.
    
    Args:
        file_path (str): Path to the CSV file
        printing_method (str): The printing method to search for
    
    Returns:
        A list of tuples containing the most common sizes and their counts
    """
    # Read the CSV file
    df = pd.read_csv("QuoteDetails.csv")
    
    # Filter by printing method (case insensitive search)
    filtered_df = df[df['Print Method'].str.lower() == printing_method.lower()]
    
    if filtered_df.empty:
        print(f"No records found for printing method: {printing_method}")
        return []
    
    # Create size strings in format 'width x height'
    filtered_df['size'] = filtered_df.apply(lambda row: f"{row['width']}x{row['height']}", axis=1)
    
    # Count occurrences of each size
    size_counts = Counter(filtered_df['size'])
    
    # Get the most common sizes
    most_common_sizes = size_counts.most_common()
    
    return most_common_sizes

def main():
    parser = argparse.ArgumentParser(description='Analyze the most common sizes for a specific printing method.')
    parser.add_argument('file_path', help='Path to the CSV file')
    parser.add_argument('printing_method', help='Printing method to search for')
    parser.add_argument('-n', '--num_results', type=int, default=5, help='Number of top results to display (default: 5)')
    
    args = parser.parse_args()
    
    most_common_sizes = analyze_printing_method_sizes(args.file_path, args.printing_method)
    
    if most_common_sizes:
        print(f"\nMost common sizes for {args.printing_method} printing method:")
        print("=" * 50)
        print(f"{'Size (width x height)':<25} {'Count':<10} {'Percentage':<10}")
        print("-" * 50)
        
        total_items = sum(count for _, count in most_common_sizes)
        
        for i, (size, count) in enumerate(most_common_sizes[:args.num_results]):
            percentage = (count / total_items) * 100
            print(f"{size:<25} {count:<10} {percentage:.2f}%")
            
        print("=" * 50)
        print(f"Total items with {args.printing_method} printing method: {total_items}")
    
if __name__ == "__main__":
    main()