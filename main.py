from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load CSV data
try:
    quotes_df = pd.read_csv("QuoteDetails.csv")
    # Standardize column names to lowercase and strip spaces
    quotes_df.columns = quotes_df.columns.str.strip().str.lower()
    logger.info(f"Successfully loaded CSV with {len(quotes_df)} rows")
except Exception as e:
    logger.error(f"Error loading CSV file: {e}")
    quotes_df = pd.DataFrame()  # Create empty DataFrame in case of error

# Define input model
class PrintingMethodInput(BaseModel):
    printing_method: str

# Define output model for size frequencies
class SizeFrequency(BaseModel):
    width: float
    height: float
    count: int
    percentage: float

class PrintingMethodSizes(BaseModel):
    printing_method: str
    total_items: int
    sizes: List[SizeFrequency]

@app.get("/")
def read_root():
    """Root endpoint with welcome message"""
    return {"message": "Welcome to the Printing Method Size Analysis API!"}

@app.post("/analyze-printing-method-sizes", response_model=PrintingMethodSizes)
def analyze_printing_method_sizes(input_data: PrintingMethodInput):
    """
    Analyze the most common sizes (width and height) for a specific printing method
    """
    printing_method = input_data.printing_method.strip()
    logger.info(f"Analyzing sizes for printing method: {printing_method}")
    
    # Check if DataFrame is empty
    if quotes_df.empty:
        raise HTTPException(status_code=500, detail="CSV data could not be loaded")
    
    # Check if 'print method' column exists
    if 'print method' not in quotes_df.columns:
        raise HTTPException(status_code=400, detail="'Print Method' column not found in CSV")
    
    # Filter quotes by printing method (case-insensitive)
    filtered_quotes = quotes_df[quotes_df['print method'].str.lower() == printing_method.lower()]
    
    # Check if any matching records found
    if filtered_quotes.empty:
        return PrintingMethodSizes(
            printing_method=printing_method,
            total_items=0,
            sizes=[]
        )
    
    # Group by width and height and count occurrences
    if 'width' in filtered_quotes.columns and 'height' in filtered_quotes.columns:
        # Convert to numeric to ensure proper grouping
        filtered_quotes['width'] = pd.to_numeric(filtered_quotes['width'], errors='coerce')
        filtered_quotes['height'] = pd.to_numeric(filtered_quotes['height'], errors='coerce')
        
        # Drop rows with NaN width or height
        filtered_quotes = filtered_quotes.dropna(subset=['width', 'height'])
        
        # Group by width and height
        size_counts = filtered_quotes.groupby(['width', 'height']).size().reset_index(name='count')
        
        # Sort by frequency (descending)
        size_counts = size_counts.sort_values(by='count', ascending=False)
        
        # Calculate percentage
        total_items = len(filtered_quotes)
        size_counts['percentage'] = (size_counts['count'] / total_items * 100).round(2)
        
        # Convert to list of dictionaries
        sizes_list = []
        for _, row in size_counts.iterrows():
            sizes_list.append(
                SizeFrequency(
                    width=float(row['width']),
                    height=float(row['height']),
                    count=int(row['count']),
                    percentage=float(row['percentage'])
                )
            )
        
        return PrintingMethodSizes(
            printing_method=printing_method,
            total_items=total_items,
            sizes=sizes_list
        )
    else:
        raise HTTPException(status_code=400, detail="Width or Height columns not found in CSV")

@app.get("/available-printing-methods")
def get_available_printing_methods():
    """
    Get a list of all available printing methods in the dataset
    """
    if quotes_df.empty:
        raise HTTPException(status_code=500, detail="CSV data could not be loaded")
    
    if 'print method' not in quotes_df.columns:
        raise HTTPException(status_code=400, detail="'Print Method' column not found in CSV")
    
    # Get unique printing methods and their counts
    printing_methods = quotes_df['print method'].value_counts().reset_index()
    printing_methods.columns = ['printing_method', 'count']
    
    return {"printing_methods": printing_methods.to_dict(orient='records')}

# Run FastAPI app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)