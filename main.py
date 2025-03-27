from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

# Load the CSV data
materials_df = pd.read_csv("Materials.csv")
quotes_df = pd.read_csv("QuoteDetails.csv")


# Define input model
class QueryInput(BaseModel):
    Recyclable: str
    Finish: str
    Opacity: str
    Factory: str = None


@app.post("/query-materials")
def query_materials(input_data: QueryInput):
    # Filter materials based on criteria
    filtered_materials = materials_df[
        (materials_df["Recyclable"] == input_data.Recyclable) &
        (materials_df["Finish"] == input_data.Finish) &
        (materials_df["Opacity"] == input_data.Opacity)
    ]

    # Filter by factory if provided
    if input_data.Factory:
        filtered_materials = filtered_materials[filtered_materials["Factory"] == input_data.Factory]
        quotes_filtered = quotes_df[quotes_df["Factory"] == input_data.Factory]
    else:
        quotes_filtered = quotes_df.copy()

    # Count material popularity
    material_counts = quotes_filtered["Material"].value_counts().to_dict()
    filtered_materials["Popularity"] = filtered_materials["Material"].map(material_counts).fillna(0)

    # Sort materials by popularity
    sorted_materials = filtered_materials.sort_values(by="Popularity", ascending=False)

    # Return sorted materials without popularity column
    return sorted_materials.drop(columns=["Popularity"]).to_dict(orient="records")


# Run FastAPI app with uvicorn explicitly on port 10000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
