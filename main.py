from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create a simple function to replace the missing material_service module
def fetch_material_details(recyclable, finish, opacity):
    """
    Simplified placeholder for the missing material_service function.
    In production, this would make an API call to an external service.
    """
    logger.info(f"Fetching material details for: {recyclable}, {finish}, {opacity}")
    return {
        "status": "success",
        "message": "Mock material details returned",
        "data": {
            "recyclable": recyclable,
            "finish": finish,
            "opacity": opacity
        }
    }

# Load CSV data with error handling
def load_csv_data():
    try:
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
        for col in ['recyclable', 'finish', 'opacity', 'factory']:
            if col in materials_df.columns:
                materials_df[col] = materials_df[col].astype(str).str.strip().str.lower()
        
        return materials_df, quotes_df
    except FileNotFoundError as e:
        logger.warning(f"CSV file not found: {e}")
        # Return empty DataFrames with expected columns
        materials_df = pd.DataFrame(columns=["material_id", "recyclable", "finish", "opacity", "factory"])
        quotes_df = pd.DataFrame(columns=["material_id"])
        return materials_df, quotes_df

# Load the data
materials_df, quotes_df = load_csv_data()

# Define input models
class QueryInput(BaseModel):
    Recyclable: str
    Finish: str
    Opacity: str
    Factory: Optional[str] = None

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

# Define input model for price prediction
class PredictionInput(BaseModel):
    width: Optional[float] = None
    height: Optional[float] = None
    length: Optional[float] = None
    print_method: Optional[str] = None
    material_id: Optional[str] = None
    ship_via: Optional[str] = None
    factory: Optional[str] = None
    total_quantity: Optional[int] = None
    options: Optional[str] = None

# Define output model
class PredictionOutput(BaseModel):
    min_price: float
    max_price: float
    price_factors: List[Dict[str, Union[str, float]]]

# Load the dataset from CSV
def load_data():
    # In production, this path would be configured properly
    try:
        df = pd.read_csv('example-QuoteDetails.csv')
        # Clean the data
        # Remove $ and convert to float if price is string
        if 'price' in df.columns and isinstance(df['price'].iloc[0], str):
            df['price'] = df['price'].str.replace('$', '').astype(float)
        
        # Parse options to extract main features
        if 'Options' in df.columns:
            df['options_list'] = df['Options'].str.split(',')
        
        return df
    except FileNotFoundError:
        # Sample data for testing when file is not found
        logger.warning("Warning: Dataset not found. Using sample data for initialization.")
        return pd.DataFrame({
            'width': [8.75, 16.0, 17.5],
            'height': [17.0, 24.0, 32.0],
            'length': [4.75, 3.75, 5.0],
            'print_method': ['Flexographic', 'Flexographic', 'Flexographic'],
            'material_id': ['3598163000015277617', '3598163000015277617', '3598163000015277617'],
            'ship_via': ['None', 'None', 'None'],
            'factory': ['USA AP', 'USA AP', 'USA AP'],
            'total_quantity': [35000, 36000, 45000],
            'Options': ['Tear Notch', 'Tear Notch', 'Tear Notch'],
            'price': [1.274, 2.051, 1.944]
        })

# Create and train the model
def train_model(data):
    # Features to consider
    numeric_features = ['width', 'height', 'length', 'total_quantity']
    categorical_features = ['print_method', 'material_id', 'ship_via', 'factory']
    
    # Check if all features exist in the data
    available_numeric = [f for f in numeric_features if f in data.columns]
    available_categorical = [f for f in categorical_features if f in data.columns]
    
    # Define preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    transformers = []
    if available_numeric:
        transformers.append(('num', numeric_transformer, available_numeric))
    if available_categorical:
        transformers.append(('cat', categorical_transformer, available_categorical))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop columns not specified
    )
    
    # Create training pipeline with Random Forest regressor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Prepare X and y
    X = data[available_numeric + available_categorical]
    if 'Options' in data.columns:
        X['Options'] = data['Options'].astype(str)
    
    if 'price' not in data.columns:
        logger.warning("No 'price' column found in data. Using dummy values for training.")
        y = np.ones(len(data))
    else:
        y = data['price']
    
    # Train the model
    pipeline.fit(X, y)
    
    # Save the model
    model_path = 'price_prediction_model.joblib'
    joblib.dump(pipeline, model_path)
    
    return pipeline, available_numeric, available_categorical

# Calculate feature importances and their impact on price
def calculate_price_factors(model, numeric_features, categorical_features, input_data, prediction):
    try:
        # Extract the actual random forest model
        rf_model = model.named_steps['regressor']
        
        # Get feature importances
        feature_importances = rf_model.feature_importances_
        
        # Get feature names (considering one-hot encoding)
        preprocessor = model.named_steps['preprocessor']
        
        # Convert input dictionary to DataFrame for consistent handling
        input_df = pd.DataFrame([input_data])
        
        # Get transformed feature names
        transformed_features = []
        
        # Extract feature names based on what's available in the input
        for feature in numeric_features:
            if feature in input_data:
                transformed_features.append(feature)
                
        # For categorical features, we need to handle one-hot encoding
        for feature in categorical_features:
            if feature in input_data:
                value = input_data[feature]
                transformed_features.append(f"{feature}={value}")
        
        # Limit feature importances to the number of transformed features
        limited_importances = feature_importances[:len(transformed_features)] if len(transformed_features) <= len(feature_importances) else feature_importances
        
        # Create a list of (feature name, importance) tuples
        importance_pairs = list(zip(transformed_features, limited_importances))
        
        # Sort by importance
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Create a list of price factors
        price_factors = []
        
        # Include only top 3 factors or all if less than 3
        top_n = min(3, len(importance_pairs))
        
        for i in range(top_n):
            feature, importance = importance_pairs[i]
            
            # Generate explanation based on feature type
            if feature in numeric_features:
                explanation = f"{feature} has significant impact on price"
                impact = f"{importance * 100:.1f}% influence on price"
            else:
                try:
                    feature_name, value = feature.split('=', 1)  # Use maxsplit=1 to handle values with '='
                    explanation = f"{feature_name}={value} affects pricing"
                except ValueError:
                    explanation = f"{feature} affects pricing"
                impact = f"{importance * 100:.1f}% influence on price"
            
            price_factors.append({
                "feature": feature,
                "explanation": explanation,
                "impact": impact
            })
        
        return price_factors
    except Exception as e:
        logger.error(f"Error calculating price factors: {e}")
        # Return default price factors if calculation fails
        return [
            {"feature": "dimensions", "explanation": "Product dimensions affect pricing", "impact": "30.0% influence on price"},
            {"feature": "quantity", "explanation": "Order quantity affects pricing", "impact": "25.0% influence on price"},
            {"feature": "material", "explanation": "Material selection affects pricing", "impact": "20.0% influence on price"}
        ]

# Function to analyze the dataset and provide a price range
def analyze_and_predict(model, numeric_features, categorical_features, input_data, data):
    try:
        # Create prediction input with user provided fields
        pred_input = {}
        
        # Filter out None values
        for key, value in input_data.dict().items():
            if value is not None:
                pred_input[key] = value
        
        # Make sure all required columns exist in pred_df
        for feature in numeric_features + categorical_features:
            if feature not in pred_df.columns:
                pred_df[feature] = None  # or some default value

        # Filter dataset based on provided constraints
        filtered_df = data.copy()
        
        for key, value in pred_input.items():
            if key in data.columns:
                if key in numeric_features:
                    # For numeric features, allow a reasonable range (±10%)
                    lower_bound = value * 0.9
                    upper_bound = value * 1.1
                    filtered_df = filtered_df[(filtered_df[key] >= lower_bound) & (filtered_df[key] <= upper_bound)]
                elif key in categorical_features or key == 'Options':
                    # For categorical features, exact match
                    filtered_df = filtered_df[filtered_df[key] == value]
        
        # If no similar records found, make a model prediction
        if filtered_df.empty or 'price' not in filtered_df.columns:
            # Convert the input dictionary to a DataFrame for prediction
            # This fixes the "Expected 2D array, got 1D array instead" error
            pred_df = pd.DataFrame([pred_input])
            
            # Make prediction based on input
            prediction = model.predict(pred_df)[0]
            
            # Use model's feature importances to calculate price range
            # For simplicity, we'll use ±15% as the price range
            min_price = max(0, prediction * 0.85)
            max_price = prediction * 1.15
        else:
            # Calculate price range from filtered data
            min_price = filtered_df['price'].min()
            max_price = filtered_df['price'].max()
        
        # Calculate price factors
        price_factors = calculate_price_factors(model, numeric_features, categorical_features, pred_input, (min_price + max_price) / 2)
        
        return min_price, max_price, price_factors
    except Exception as e:
        logger.error(f"Error in analyze_and_predict: {e}")
        # Return default values if prediction fails
        return 1.0, 2.0, [{"feature": "error", "explanation": "Error in prediction", "impact": "100% influence on price"}]

# Global variables for model and data
global_data = None
global_model = None
global_numeric_features = None
global_categorical_features = None

@app.on_event("startup")
async def startup_event():
    global global_data, global_model, global_numeric_features, global_categorical_features
    
    # Load data
    global_data = load_data()
    
    # Train or load model
    model_path = 'price_prediction_model.joblib'
    if os.path.exists(model_path):
        try:
            global_model = joblib.load(model_path)
            # Set feature lists (these would need to be saved separately in production)
            global_numeric_features = ['width', 'height', 'length', 'total_quantity']
            global_categorical_features = ['print_method', 'material_id', 'ship_via', 'factory']
            logger.info("Successfully loaded existing model")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            global_model, global_numeric_features, global_categorical_features = train_model(global_data)
    else:
        logger.info("Training new model")
        global_model, global_numeric_features, global_categorical_features = train_model(global_data)

@app.post("/predict", response_model=PredictionOutput)
async def predict_price(input_data: PredictionInput):
    logger.info(f"Received data: {input_data.dict()}")
    # Check if at least one parameter is provided
    if all(value is None for value in input_data.dict().values()):
        raise HTTPException(status_code=400, detail="At least one parameter must be provided")
    
    # Make prediction
    min_price, max_price, price_factors = analyze_and_predict(
        global_model, 
        global_numeric_features, 
        global_categorical_features,
        input_data,
        global_data
    )
    
    return PredictionOutput(
        min_price=round(min_price, 3),
        max_price=round(max_price, 3),
        price_factors=price_factors
    )

# Run FastAPI app with uvicorn explicitly on port 10000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)