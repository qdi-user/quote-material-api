from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load CSV data
try:
    materials_df = pd.read_csv("Materials.csv")
    quotes_df = pd.read_csv("QuoteDetails.csv")
except FileNotFoundError as e:
    logger.error(f"Error loading CSV files: {e}")
    exit(1)

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
if 'printing_method' in quotes_df.columns:
    quotes_df["printing_method"] = quotes_df["printing_method"].str.strip().str.lower()

# Define input models
class QueryInput(BaseModel):
    Recyclable: str
    Finish: str
    Opacity: str
    Factory: Optional[str] = None
    Printing_Method: Optional[str] = None

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
    recyclable = input_data.Recyclable.strip().lower()
    finish = input_data.Finish.strip().lower()
    opacity = input_data.Opacity.strip().lower()
    factory = input_data.Factory.strip().lower() if input_data.Factory else None
    printing_method = input_data.Printing_Method.strip().lower() if input_data.Printing_Method else None

    logger.info(f"Sanitized Input: Recyclable={recyclable}, Finish={finish}, Opacity={opacity}, Factory={factory}, Printing_Method={printing_method}")

    filtered_materials = materials_df[
        (materials_df["recyclable"] == recyclable) &
        (materials_df["finish"] == finish) &
        (materials_df["opacity"] == opacity)
    ]

    if factory and "factory" in filtered_materials.columns:
        filtered_materials = filtered_materials[filtered_materials["factory"] == factory]

    if "material_id" in filtered_materials.columns and "material_id" in quotes_df.columns:
        material_ids = filtered_materials["material_id"].astype(int).tolist()

        filtered_quotes = quotes_df[quotes_df["material_id"].isin(material_ids)]

        if printing_method and 'printing_method' in filtered_quotes.columns:
            filtered_quotes = filtered_quotes[filtered_quotes["printing_method"] == printing_method]

        material_counts = filtered_quotes["material_id"].value_counts().to_dict()
    else:
        logger.warning("'material_id' column not found in one or both CSVs.")
        material_counts = {}

    if "material_id" in filtered_materials.columns:
        filtered_materials["count"] = filtered_materials["material_id"].astype(int).map(material_counts).fillna(0).astype(int)
    else:
        logger.warning("'material_id' column not found in materials_df.")
        filtered_materials["count"] = 0

    sorted_materials = filtered_materials.sort_values(by="count", ascending=False)
    sorted_materials.replace([np.inf, -np.inf], 0, inplace=True)
    sorted_materials.fillna(0, inplace=True)

    if not sorted_materials.empty and "material_id" in sorted_materials.columns and "material_id" in quotes_df.columns:
        most_popular_material_id = sorted_materials["material_id"].iloc[0]
        popular_quotes = quotes_df[quotes_df["material_id"] == most_popular_material_id]
        if printing_method and 'printing_method' in popular_quotes.columns:
            popular_quotes = popular_quotes[popular_quotes["printing_method"] == printing_method]
        if not popular_quotes.empty and "width" in popular_quotes.columns and "height" in popular_quotes.columns:
            most_common_size = popular_quotes.groupby(["width", "height"]).size().idxmax()
            most_common_width, most_common_height = most_common_size
        else:
            most_common_width, most_common_height = None, None
    else:
        most_common_width, most_common_height = None, None

    result = sorted_materials.drop(columns=["count"], errors="ignore").to_dict(orient="records")

    logger.info(f"Query results: {len(result)} materials found")
    if most_common_width is not None and most_common_height is not None:
        return {"materials": result, "most_common_width": most_common_width, "most_common_height": most_common_height}
    else:
        return {"materials": result, "most_common_width": None, "most_common_height": None}

@app.post("/get-material-details/")
def get_material_details(query: MaterialQuery):
    """
    Fetch material details using external API.
    Returns all matching materials.
    """
    logging.info(f"Received input: {query.dict()}")
    # result = fetch_material_details(
    #     query.recyclable, query.finish, query.opacity
    # )
    # logging.info(f"API Response: {result}")
    return {"result": "API call to fetch_material_details() is currently commented out"}

# Run FastAPI app with uvicorn explicitly on port 10000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)