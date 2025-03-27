from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

# ✅ Load CSV data
materials_df = pd.read_csv("Materials.csv")
quotes_df = pd.read_csv("QuoteDetails.csv")

# ✅ Debug: Print column names to logs
print("✅ Materials.csv Columns:", list(materials_df.columns))
print("✅ QuoteDetails.csv Columns:", list(quotes_df.columns))

# Define input model
class QueryInput(BaseModel):
    Recyclable: str
    Finish: str
    Opacity: str
    Factory: str = None


@app.get("/")
def read_root():
    # ✅ Simple welcome message for root endpoint
    return {"message": "Welcome to the Quote Material API!"}


@app.post("/query-materials")
def query_materials(input_data: QueryInput):
    # ✅ Debug: Print received input data
    print(f"✅ Received input data: {input_data.dict()}")

    # ✅ Debug: Print shape of materials_df before filtering
    print(f"✅ Initial materials_df shape: {materials_df.shape}")

    # Filter materials based on criteria
    filtered_materials = materials_df[
        (materials_df["Recyclable"] == input_data.Recyclable) &
        (materials_df["Finish"] == input_data.Finish) &
        (materials_df["Opacity"] == input_data.Opacity)
    ]

    # ✅ Debug: Print shape of filtered materials
    print(f"✅ Filtered materials_df shape: {filtered_materials.shape}")

    # Filter by factory if provided
    if input_data.Factory:
        if "Factory" in filtered_materials.columns:
            filtered_materials = filtered_materials[filtered_materials["Factory"] == input_data.Factory]
            quotes_filtered = quotes_df[quotes_df["Factory"] == input_data.Factory]
        else:
            print("❗️ Warning: 'Factory' column not found in materials_df or quotes_df.")
            quotes_filtered = quotes_df.copy()
    else:
        quotes_filtered = quotes_df.copy()

    # ✅ Debug: Print shape of quotes_filtered
    print(f"✅ Quotes filtered shape: {quotes_filtered.shape}")

    # Count material popularity
    if "Material" in quotes_filtered.columns:
        material_counts = quotes_filtered["Material"].value_counts().to_dict()
    else:
        print("❗️ Warning: 'Material' column not found in quotes_df.")
        material_counts = {}

    # Add popularity to filtered materials
    if "Material" in filtered_materials.columns:
        filtered_materials["Popularity"] = filtered_materials["Material"].map(material_counts).fillna(0)
    else:
        print("❗️ Warning: 'Material' column not found in materials_df.")
        filtered_materials["Popularity"] = 0

    # Sort materials by popularity
    sorted_materials = filtered_materials.sort_values(by="Popularity", ascending=False)

    # ✅ Debug: Print result shape
    print(f"✅ Result shape after sorting: {sorted_materials.shape}")

    # Drop 'Popularity' column before returning
    result = sorted_materials.drop(columns=["Popularity"], errors="ignore").to_dict(orient="records")

    # ✅ Debug: Print final result
    print(f"✅ Final result: {result}")

    return result


# Run FastAPI app with uvicorn explicitly on port 10000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
