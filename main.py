from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
from material_service import fetch_material_details

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load CSV data
materials_df = pd.read_csv("Materials.csv")
quotes_df = pd.read_csv("QuoteDetails.csv")

# Standardize column names to lowercase and strip spaces
materials_df.columns = materials_df.columns.str.strip().str.lower()
quotes_df.columns = quotes_df.columns.str.strip().str.lower()

# Convert material_id to integers, replacing non-numeric values with NaN
materials_df["material_id"] = pd.to_numeric(materials_df["material_id"], errors='coerce')
quotes_df["material_id"] = pd.to_numeric(quotes_df["material_id"], errors='coerce')

# Drop rows with NaN or zero material_id in both dataframes
materials_df = materials_df[materials_df["material_id"].notna() & (materials_df["material_id"] != 0)]
quotes_df = quotes_df[quotes_df["material_id"].notna() & (quotes_df["material_id"] != 0)]

# Clean and standardize the values in other columns
materials_df["recyclable"] = materials_df["recyclable"].str.strip().str.lower()
materials_df["finish"] = materials_df["finish"].str.strip().str.lower()
materials_df["opacity"] = materials_df["opacity"].str.strip().str.lower()
materials_df["factory"] = materials_df["factory"].str.strip().str.lower()

# Define input models
class QueryInput(BaseModel):
    Recyclable: str
    Finish: str
    Opacity: str
    Factory: str = None

class MaterialQuery(BaseModel):
    recyclable: str
    finish: str
    opacity: str

@app.get("/")
def read_root():
    """Root endpoint with welcome message"""
    return {"message": "Welcome to the Quote Material API!"}

@app.post("/query-materials")
def query_materials(input_data: QueryInput):
    """
    Query materials based on input criteria and sort by popularity
    """
    # Convert input data to lowercase and strip spaces to avoid mismatches
    recyclable = input_data.Recyclable.strip().lower()
    finish = input_data.Finish.strip().lower()
    opacity = input_data.Opacity.strip().lower()
    factory = input_data.Factory.strip().lower() if input_data.Factory else None

    logger.info(f"Sanitized Input: Recyclable={recyclable}, Finish={finish}, Opacity={opacity}, Factory={factory}")

    # Filter materials based on input criteria
    filtered_materials = materials_df[
        (materials_df["recyclable"] == recyclable) &
        (materials_df["finish"] == finish) &
        (materials_df["opacity"] == opacity)
    ]

    # Filter by factory if provided
    if factory:
        if "factory" in filtered_materials.columns:
            filtered_materials = filtered_materials[filtered_materials["factory"] == factory]
        else:
            logger.warning("'factory' column not found in materials_df.")

    # Count occurrences of materials in quotes
    if "material_id" in filtered_materials.columns and "material_id" in quotes_df.columns:
        material_ids = filtered_materials["material_id"].astype(int).tolist()
        
        material_counts = (
            quotes_df[quotes_df["material_id"].isin(material_ids)]["material_id"]
            .value_counts()
            .to_dict()
        )
    else:
        logger.warning("'material_id' column not found in one or both CSVs.")
        material_counts = {}

    # Map count of occurrences to the filtered materials
    if "material_id" in filtered_materials.columns:
        filtered_materials["count"] = (
            filtered_materials["material_id"]
            .astype(int)
            .map(material_counts)
            .fillna(0)
            .astype(int)
        )
    else:
        logger.warning("'material_id' column not found in materials_df.")
        filtered_materials["count"] = 0

    # Sort materials by count of occurrences (from most to least found)
    sorted_materials = filtered_materials.sort_values(by="count", ascending=False)

    # Handle invalid values to avoid JSON conversion issues
    sorted_materials.replace([np.inf, -np.inf], 0, inplace=True)
    sorted_materials.fillna(0, inplace=True)

    # Drop 'count' column before returning the final result
    result = sorted_materials.drop(columns=["count"], errors="ignore").to_dict(orient="records")

    logger.info(f"Query results: {len(result)} materials found")
    return result

@app.post("/get-material-details/")
def get_material_details(query: MaterialQuery):
    """
    Fetch material details using external API.
    Returns all matching materials.
    """
    logging.info(f"Received input: {query.dict()}")
    result = fetch_material_details(
        query.recyclable, query.finish, query.opacity
    )
    logging.info(f"API Response: {result}")
    return {"result": result}

# Run FastAPI app with uvicorn explicitly on port 10000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)