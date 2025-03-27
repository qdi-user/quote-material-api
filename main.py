from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI()

# ✅ Load CSV data
materials_df = pd.read_csv("Materials.csv")
quotes_df = pd.read_csv("QuoteDetails.csv")

# ✅ Standardize column names and strip whitespace
materials_df.columns = materials_df.columns.str.strip().str.lower()
quotes_df.columns = quotes_df.columns.str.strip().str.lower()

# ✅ Clean and standardize the values in key columns
materials_df["recyclable"] = materials_df["recyclable"].str.strip().str.lower()
materials_df["finish"] = materials_df["finish"].str.strip().str.lower()
materials_df["opacity"] = materials_df["opacity"].str.strip().str.lower()
materials_df["factory"] = materials_df["factory"].str.strip().str.lower()

# ✅ Ensure 'material_id' is treated as a string to avoid mismatches
materials_df["material_id"] = materials_df["material_id"].astype(str).str.strip()
quotes_df["material_id"] = quotes_df["material_id"].astype(str).str.strip()
quotes_df["factory"] = quotes_df["factory"].str.strip().str.lower()

# ✅ Debug: Print unique values to ensure no mismatches
print("✅ Unique values in 'recyclable':", materials_df["recyclable"].unique())
print("✅ Unique values in 'finish':", materials_df["finish"].unique())
print("✅ Unique values in 'opacity':", materials_df["opacity"].unique())
print("✅ Unique values in 'factory':", materials_df["factory"].unique())


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

    # ✅ Convert input data to lowercase and strip spaces to avoid mismatches
    recyclable = input_data.Recyclable.strip().lower()
    finish = input_data.Finish.strip().lower()
    opacity = input_data.Opacity.strip().lower()
    factory = input_data.Factory.strip().lower() if input_data.Factory else None

    # ✅ Debug: Print received data after sanitizing
    print(f"✅ Sanitized Input: Recyclable={recyclable}, Finish={finish}, Opacity={opacity}, Factory={factory}")

    # ✅ Filter materials based on input criteria
    filtered_materials = materials_df[
        (materials_df["recyclable"] == recyclable) &
        (materials_df["finish"] == finish) &
        (materials_df["opacity"] == opacity)
    ]

    # ✅ Debug: Print shape after basic filters
    print(f"✅ Filtered materials_df shape (before factory): {filtered_materials.shape}")

    # Filter by factory if provided
    if factory:
        if "factory" in filtered_materials.columns:
            filtered_materials = filtered_materials[filtered_materials["factory"] == factory]
        else:
            print("❗️ Warning: 'factory' column not found in materials_df.")
    
    # ✅ Debug: Print shape after factory filtering
    print(f"✅ Filtered materials_df shape (after factory filter): {filtered_materials.shape}")

    # Check if 'material_id' exists in both dataframes
    if "material_id" in filtered_materials.columns and "material_id" in quotes_df.columns:
        # Get list of relevant Material_IDs as strings
        material_ids = filtered_materials["material_id"].astype(str).tolist()

        # ✅ Count occurrences of each Material_ID in QuoteDetails.csv
        material_counts = (
            quotes_df[quotes_df["material_id"].isin(material_ids)]["material_id"]
            .value_counts()
            .to_dict()
        )

        # ✅ Debug: Print material counts for verification
        print(f"✅ Material counts from QuoteDetails.csv: {material_counts}")

    else:
        print("❗️ Warning: 'material_id' column not found in one or both CSVs.")
        material_counts = {}

    # ✅ Map count of occurrences to the filtered materials using 'material_id'
    if "material_id" in filtered_materials.columns:
        filtered_materials["count"] = (
            filtered_materials["material_id"]
            .astype(str)
            .map(material_counts)
            .fillna(0)
            .astype(int)
        )
    else:
        print("❗️ Warning: 'material_id' column not found in materials_df.")
        filtered_materials["count"] = 0

    # ✅ Debug: Print mapped counts to check before sorting
    print(f"✅ Mapped counts for filtered materials:\n{filtered_materials[['material_id', 'count']].sort_values(by='count', ascending=False).head()}")

    # ✅ Sort materials by count of occurrences (from most to least found)
    sorted_materials = filtered_materials.sort_values(by="count", ascending=False)

    # ✅ Handle invalid values (NaN, Inf, -Inf) to avoid JSON conversion issues
    sorted_materials.replace([np.inf, -np.inf], 0, inplace=True)
    sorted_materials.fillna(0, inplace=True)

    # ✅ Debug: Print result shape after sorting
    print(f"✅ Result shape after sorting: {sorted_materials.shape}")

    # Drop 'count' column before returning the final result
    result = sorted_materials.drop(columns=["count"], errors="ignore").to_dict(orient="records")

    # ✅ Debug: Print final result to check before returning
    print(f"✅ Final result: {result}")

    # ✅ Additional Debug: Check for invalid data types
    print(f"🔎 Data types after final cleaning:\n{sorted_materials.dtypes}")
    print(f"🔎 Any remaining NaN values? {sorted_materials.isnull().sum().sum()}")

    return result


# Run FastAPI app with uvicorn explicitly on port 10000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
