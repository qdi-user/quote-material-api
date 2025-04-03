import pandas as pd
from fastapi import FastAPI, HTTPException
from collections import Counter
import uvicorn

app = FastAPI()

# Load the CSV file
CSV_FILE_PATH = "QuoteDetails.csv"

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found at path: {file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while loading the CSV file: {e}")

df = load_data(CSV_FILE_PATH)

@app.get("/get-material-details/")
async def get_common_sizes(printing_method: str):
    """
    Finds the most commonly requested sizes (width x height) for a given printing method.

    Args:
        printing_method (str): The printing method to filter by.

    Returns:
        dict: A dictionary containing the most common sizes.
              Returns an empty list if no matching printing method is found.
    """

    # Filter the DataFrame by the specified printing method
    filtered_df = df[df['Print Method'] == printing_method]

    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="Printing method not found")

    # Combine width and height into a single string format "width x height"
    filtered_df['Size'] = filtered_df['width'].astype(str) + 'x' + filtered_df['height'].astype(str)

    # Count the occurrences of each unique size
    size_counts = Counter(filtered_df['Size'])

    # Find the most common sizes
    most_common_sizes = size_counts.most_common()

    return {"most_common_sizes": most_common_sizes}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)