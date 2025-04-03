from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define input model
class PrintMethodQuery(BaseModel):
    print_method: str

# Define response model
class SizeData(BaseModel):
    width: float
    height: float
    count: int
    percentage: float

class PrintMethodSizeResponse(BaseModel):
    print_method: str
    sizes: List[SizeData]
    total_count: int

def load_quote_data():
    """
    Load and preprocess the QuoteDetails.csv data
    """
    try:
        quotes_df = pd.read_csv("QuoteDetails.csv")
        
        # Standardize column names to lowercase and strip spaces
        quotes_df.columns = quotes_df.columns.str.strip().str.lower()
        
        # Ensure print method, width, and height columns exist
        required_columns = ['print method', 'width', 'height']
        if not all(col in quotes_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in quotes_df.columns]
            logger.error(f"Required columns missing from CSV: {missing}")
            
            # Map column names if needed
            if 'printing' in quotes_df.columns and 'print method' not in quotes_df.columns:
                quotes_df['print method'] = quotes_df['printing']
            
        # Convert width and height to numeric, handling errors
        quotes_df['width'] = pd.to_numeric(quotes_df['width'], errors='coerce')
        quotes_df['height'] = pd.to_numeric(quotes_df['height'], errors='coerce')
        
        # Drop rows with invalid width or height
        quotes_df = quotes_df.dropna(subset=['width', 'height'])
        
        return quotes_df
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

@app.get("/")
def read_root():
    """Root endpoint with welcome message"""
    return {"message": "Welcome to the Print Method Size Analyzer API!"}

@app.post("/get-material-details/")
def get_material_details(query: PrintMethodQuery):
    """
    Query the most commonly requested sizes for a specific printing method
    """
    logger.info(f"Received request for print method: {query.print_method}")
    
    # Load the quote data
    quotes_df = load_quote_data()
    
    # Standardize the input print method (case-insensitive matching)
    target_print_method = query.print_method.strip().lower()
    
    # Get all available print methods for logging
    available_methods = quotes_df['print method'].str.lower().unique().tolist()
    logger.info(f"Available print methods: {available_methods}")
    
    # Filter quotes by the requested print method
    filtered_quotes = quotes_df[quotes_df['print method'].str.lower() == target_print_method]
    
    if filtered_quotes.empty:
        logger.warning(f"No quotes found for print method: {query.print_method}")
        return PrintMethodSizeResponse(
            print_method=query.print_method,
            sizes=[],
            total_count=0
        )
    
    # Count occurrences of each width-height combination
    size_counts = (
        filtered_quotes
        .groupby(['width', 'height'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
    )
    
    # Calculate percentage of each size
    total_count = size_counts['count'].sum()
    size_counts['percentage'] = (size_counts['count'] / total_count * 100).round(2)
    
    # Convert to list of dictionaries
    sizes_list = []
    for _, row in size_counts.iterrows():
        sizes_list.append(SizeData(
            width=float(row['width']),
            height=float(row['height']),
            count=int(row['count']),
            percentage=float(row['percentage'])
        ))
    
    logger.info(f"Found {len(sizes_list)} different sizes for {query.print_method}")
    
    # Return the response
    return PrintMethodSizeResponse(
        print_method=query.print_method,
        sizes=sizes_list,
        total_count=int(total_count)
    )

@app.get("/print-methods")
def get_print_methods():
    """
    List all available print methods in the dataset
    """
    quotes_df = load_quote_data()
    
    # Get unique print methods and counts
    print_method_counts = (
        quotes_df['print method']
        .str.lower()
        .value_counts()
        .reset_index()
        .rename(columns={'index': 'print_method', 'print method': 'count'})
    )
    
    return {
        "print_methods": print_method_counts.to_dict(orient='records'),
        "total_methods": len(print_method_counts)
    }
from fastapi import Body
from fastapi.encoders import jsonable_encoder

@app.post("/query")
def dynamic_query(filters: Dict[str, Any] = Body(...)):
    """
    Accepts a dictionary of filters and returns matching rows from QuoteDetails.csv.
    Filters can be exact matches or support simple conditions like gt, lt, etc.
    Width greater than 10 and price less than 2
    Example input:
    {
        "product_line": "Secure Sack CR Bags",
        "width": { "gt": 10 },
        "price": { "lt": 2.0 }
    }
    """
    try:
        df = load_quote_data()

        for key, condition in filters.items():
            col = key.strip().lower()
            if isinstance(condition, dict):
                for op, val in condition.items():
                    if op == "gt":
                        df = df[df[col] > val]
                    elif op == "lt":
                        df = df[df[col] < val]
                    elif op == "gte":
                        df = df[df[col] >= val]
                    elif op == "lte":
                        df = df[df[col] <= val]
                    elif op == "ne":
                        df = df[df[col] != val]
                    elif op == "eq":
                        df = df[df[col] == val]
            else:
                df = df[df[col] == condition]

        result = df.to_dict(orient="records")
        return jsonable_encoder({"count": len(result), "results": result[:100]})  # limit return size

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run FastAPI app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)