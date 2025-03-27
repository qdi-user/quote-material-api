from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os

app = FastAPI()

# Use relative path
materials_df = pd.read_csv("Materials.csv")
quotes_df = pd.read_csv("QuoteDetails.csv")

class QueryInput(BaseModel):
    Recyclable: str
    Finish: str
    Opacity: str
    Factory: str = None

@app.post("/query-materials")
def query_materials(input_data: QueryInput):
    filtered_materials = materials_df[
        (materials_df["Recyclable"] == input_data.Recyclable) &
        (materials_df["Finish"] == input_data.Finish) &
        (materials_df["Opacity"] == input_data.Opacity)
    ]

    if input_data.Factory:
        filtered_materials = filtered_materials[filtered_materials["Factory"] == input_data.Factory]
        quotes_filtered = quotes_df[quotes_df["Factory"] == input_data.Factory]
    else:
        quotes_filtered = quotes_df.copy()

    material_counts = quotes_filtered["Material"].value_counts().to_dict()
    filtered_materials["Popularity"] = filtered_materials["Material"].map(material_counts).fillna(0)

    sorted_materials = filtered_materials.sort_values(by="Popularity", ascending=False)

    return sorted_materials.drop(columns=["Popularity"]).to_dict(orient="records")
