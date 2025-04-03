from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
from material_service import fetch_material_details # Assume this function exists or is replaced

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load CSV data
materials_df = pd.read_csv("Materials.csv") # Replace with actual path if needed
quotes_df = pd.read_csv("QuoteDetails.csv")  # Replace with actual path if needed

# Standardize column names
materials_df.columns = materials_df.columns.str.strip().str.lower()
quotes_df.columns = quotes_df.columns.str.strip().str.lower()

# Convert material_id to integers and drop invalid rows
materials_df["material_id"] = pd.to_numeric(materials_df["material_id"], errors='coerce').dropna().astype(int)
quotes_df["material_id"] = pd.to_numeric(quotes_df["material_id"], errors='coerce').dropna().astype(int)

materials_df = materials_df[materials_df["material_id"] != 0]
quotes_df = quotes_df[quotes_df["material_id"] != 0]

# Clean string columns
for col in materials_df.columns:
    if materials_df[col].dtype == 'object':
        materials_df[col] = materials_df[col].str.strip().str.lower()
for col in quotes_df.columns:
    if quotes_df[col].dtype == 'object':
        quotes_df[col] = quotes_df[col].str.strip().str.lower()


class MaterialQuery(BaseModel):
    recyclable: str
    finish: str
    opacity: str

def get_material_counts(quotes_df: pd.DataFrame) -> dict:
    """
    Counts the occurrences of each material_id in the quotes dataframe.
    """
    return quotes_df['material_id'].value_counts().to_dict()

@app.get("/materials/")
def get_materials():
    """
    Fetch all materials with their quote counts.
    """
    material_counts = get_material_counts(quotes_df)
    materials_with_counts = materials_df.copy()
    materials_with_counts['count'] = materials_with_counts['material_id'].map(material_counts).fillna(0).astype(int)
    return materials_with_counts.to_dict(orient='records')


@app.post("/get-material-details/")
def get_material_details(query: MaterialQuery):
    """
    Fetch material details using external API.
    Returns all matching materials.
    """
    logging.info(f"Received input: {query.dict()}")
    # Assuming fetch_material_details now works correctly or is replaced
    result = fetch_material_details(query.recyclable, query.finish, query.opacity, materials_df) 
    logging.info(f"API Response: {result}")
    return {"result": result}


# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)