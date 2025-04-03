
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from typing import List, Dict, Any
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MAX_RESULTS = 100
SUPPORTED_OPS = {"gt", "lt", "gte", "lte", "eq", "ne"}

class PrintMethodQuery(BaseModel):
    print_method: str

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
    try:
        df = pd.read_csv("QuoteDetails.csv")
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

@app.get("/")
def health_check():
    return {"status": "OK"}

@app.post("/print_method", response_model=PrintMethodSizeResponse)
def get_sizes_by_print_method(query: PrintMethodQuery):
    try:
        df = load_quote_data()
        if 'print method' not in df.columns:
            raise HTTPException(status_code=400, detail="Missing 'print method' column")

        df_filtered = df[df['print method'].str.lower() == query.print_method.lower()]

        if df_filtered.empty:
            return {
                "print_method": query.print_method,
                "sizes": [],
                "total_count": 0
            }

        grouped = df_filtered.groupby(["width", "height"]).size().reset_index(name="count")
        total = grouped["count"].sum()
        grouped["percentage"] = grouped["count"] / total * 100
        grouped = grouped.sort_values(by="count", ascending=False)

        sizes = [
            {
                "width": row["width"],
                "height": row["height"],
                "count": row["count"],
                "percentage": row["percentage"]
            }
            for _, row in grouped.iterrows()
        ]

        return {
            "print_method": query.print_method,
            "sizes": sizes,
            "total_count": total
        }

    except Exception as e:
        logger.error(f"Error processing print method query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def dynamic_query(filters: Dict[str, Any] = Body(...)):
    try:
        logger.info(f"Received filters: {filters}")
        if not filters:
            raise HTTPException(status_code=400, detail="At least one filter is required.")

        df = load_quote_data()

        for key, condition in filters.items():
            col = key.strip().lower()
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in data")

            if isinstance(condition, dict):
                for op, val in condition.items():
                    if op not in SUPPORTED_OPS:
                        raise HTTPException(status_code=400, detail=f"Unsupported operation '{op}' on '{col}'")
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

        result = (
            df.replace({float("inf"): None, float("-inf"): None})
            .fillna("")
            .head(MAX_RESULTS)
            .to_dict(orient="records")
        )

        return jsonable_encoder({"count": len(result), "results": result})

    except Exception as e:
        logger.error(f"Query error with filters {filters}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/print_methods")
def get_print_methods():
    try:
        df = load_quote_data()
        methods = df["print method"].dropna().unique().tolist()
        return {"print_methods": sorted(methods)}
    except Exception as e:
        logger.error(f"Error retrieving print methods: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/material_details")
def get_material_details():
    try:
        df = load_quote_data()
        if "material" not in df.columns:
            raise HTTPException(status_code=400, detail="Missing 'material' column")
        materials = df["material"].value_counts().to_dict()
        return {"materials": materials}
    except Exception as e:
        logger.error(f"Error retrieving material details: {e}")
        raise HTTPException(status_code=500, detail=str(e))
