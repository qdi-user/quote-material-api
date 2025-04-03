import pandas as pd
import argparse
from typing import Dict, List, Tuple
import json

def analyze_printing_method_sizes(csv_path: str, printing_method: str) -> Dict:
    """
    Analyze the most common sizes (width and height) for a specific printing method
    
    Args:
        csv_path: Path to the CSV file
        printing_method: The printing method to filter by
    
    Returns:
        Dictionary with analysis results
    """
    # Load CSV data
    df = pd.read_csv("QuoteDetails.csv")
    
    # Check if 'print method' column exists
    if 'print method' not in df.columns:
        # Try to find a similar column
        potential_columns = [col for col in df.columns if 'print' in col.lower()]
        if potential_columns:
            print(f"'Print Method' column not found. Similar columns: {potential_columns}")
        else:
            print("No 'Print Method' column found in the CSV")
        return {"error": "Print Method column not found"}
    
    # Print unique printing methods to help user
    unique_methods = df['print method'].unique()
    print(f"Available printing methods in the data: {unique_methods}")
    
    # Filter quotes by printing method (case-insensitive)
    filtered_df = df[df['print method'].str.lower() == printing_method.lower()]
    
    # Check if any matching records found
    if filtered_df.empty:
        print(f"No records found for printing method: {printing_method}")
        return {
            "printing_method": printing_method,
            "total_items": 0,
            "sizes": []
        }
    
    # Check if width and height columns exist
    if 'width' not in df.columns or 'height' not in df.columns:
        print("Width or Height columns not found in CSV")
        return {"error": "Width or Height columns not found"}
    
    # Convert to numeric to ensure proper grouping
    filtered_df['width'] = pd.to_numeric(filtered_df['width'], errors='coerce')
    filtered_df['height'] = pd.to_numeric(filtered_df['height'], errors='coerce')
    
    # Drop rows with NaN width or height
    filtered_df = filtered_df.dropna(subset=['width', 'height'])
    
    # Group by width and height and count occurrences
    size_counts = filtered_df.groupby(['width', 'height']).size().reset_index(name='count')
    
    # Sort by frequency (descending)
    size_counts = size_counts.sort_values(by='count', ascending=False)
    
    # Calculate percentage
    total_items = len(filtered_df)
    size_counts['percentage'] = (size_counts['count'] / total_items * 100).round(2)
    
    # Convert to list of dictionaries
    sizes_list = []
    for _, row in size_counts.iterrows():
        sizes_list.append({
            "width": float(row['width']),
            "height": float(row['height']),
            "count": int(row['count']),
            "percentage": float(row['percentage'])
        })
    
    result = {
        "printing_method": printing_method,
        "total_items": total_items,
        "sizes": sizes_list
    }
    
    print(f"Found {total_items} items with printing method '{printing_method}'")
    print(f"Top 5 most common sizes:")
    for i, size in enumerate(sizes_list[:5], 1):
        print(f"{i}. Width: {size['width']}, Height: {size['height']} - Count: {size['count']} ({size['percentage']}%)")
    
    return result

def get_printing_method_counts(csv_path: str) -> List[Dict]:
    """Get counts of all printing methods in the dataset"""
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
        
        if 'print method' not in df.columns:
            print("Print Method column not found in CSV")
            return []
        
        # Get counts of each printing method
        counts = df['print method'].value_counts().reset_index()
        counts.columns = ['printing_method', 'count']
        
        return counts.to_dict('records')
    except Exception as e:
        print(f"Error analyzing printing methods: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Analyze printing method sizes in a CSV file")
    parser.add_argument('--csv', default='example-QuoteDetails.csv', help='Path to the CSV file')
    parser.add_argument('--method', help='Printing method to analyze')
    parser.add_argument('--list-methods', action='store_true', help='List all printing methods in the CSV')
    parser.add_argument('--output', help='Output file path for results (JSON format)')
    
    args = parser.parse_args()
    
    if args.list_methods:
        methods = get_printing_method_counts(args.csv)
        print("\nAvailable printing methods:")
        for m in methods:
            print(f"- {m['printing_method']}: {m['count']} items")
        return
    
    if not args.method:
        print("Please specify a printing method with --method or use --list-methods to see available options")
        return
    
    result = analyze_printing_method_sizes(args.csv, args.method)
    
    # Save to file if output path provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()