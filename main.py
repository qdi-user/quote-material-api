from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
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

# Add this function to your code, near the top with your other utility functions
def log_dataset_info(df, description="Dataset", max_records_to_print=5):
    """
    Always logs dataset stats, but only prints all values when dataset is small
    
    Args:
        df: DataFrame to log information about
        description: Description to include in log messages
        max_records_to_print: Maximum number of records to print in full
    """
    # Always log basic stats about the dataset
    logger.info(f"{description} - Record count: {len(df)}")
    
    # Log column information
    if not df.empty:
        # Get count of non-null values for each column
        non_null_counts = df.count()
        logger.info(f"{description} - Column counts:")
        for col in df.columns:
            logger.info(f"  {col}: {non_null_counts[col]} non-null values")
        
        # Log numeric column stats
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            logger.info(f"{description} - Numeric column stats:")
            for col in numeric_cols:
                stats = df[col].describe()
                logger.info(f"  {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
    
    # Only print all values when dataset is small
    if len(df) <= max_records_to_print and not df.empty:
        logger.info(f"{description} - Printing all {len(df)} records:")
        for i, row in df.iterrows():
            logger.info(f"Record {i}:")
            for col in row.index:
                logger.info(f"  {col}: {row[col]}")
            logger.info("-" * 40)  # Separator between records


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

# 1. Fix the material_id handling to work with both string and number formats
def normalize_material_id(material_id):
    """
    Normalize material_id to ensure consistent handling regardless of format.
    Handles integer, string with leading apostrophe, and regular string formats.
    """
    if material_id is None:
        return None
        
    # Convert to string and strip any leading apostrophes or spaces
    if isinstance(material_id, (int, float)):
        return str(int(material_id))
    else:
        # Strip any leading apostrophes and spaces
        cleaned = str(material_id).strip().lstrip("'")
        return cleaned

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

# 2. Modify the analyze_and_predict function to properly handle optional fields
def analyze_and_predict(model, numeric_features, categorical_features, input_data, data):
    try:
        # Create prediction input dictionary with only provided fields
        pred_input = {}
        
        # Extract non-None values from input
        for key, value in input_data.dict().items():
            if value is not None:
                # Normalize material_id if present
                if key == 'material_id':
                    pred_input[key] = normalize_material_id(value)
                else:
                    pred_input[key] = value

        logger.info(f"Filtered input data (non-None values only): {pred_input}")
        
        # Start with the complete dataset - don't filter by default
        filtered_df = data.copy()
        
        # Log the initial dataset size
        logger.info(f"Initial dataset size: {len(filtered_df)} records")
        log_dataset_info(filtered_df, "Initial dataset")
        
        # Check if we have required fields (width, height)
        required_fields = ['width', 'height']
        has_required = all(field in pred_input for field in required_fields)
        
        # Only proceed with filtering if we have required fields
        if has_required:
            # ------- MUCH WIDER RANGES FOR NUMERIC FEATURES -------
            # Apply filters ONLY for fields that are present in pred_input
            for key, value in pred_input.items():
                if key in data.columns:
                    if key in numeric_features:
                        # For numeric features, use a MUCH wider range (±50%)
                        # Small values like 0 need special handling
                        if key == 'length' and (value == 0 or value == 0.0):
                            # For length=0, match any length <= 1
                            before_count = len(filtered_df)
                            filtered_df = filtered_df[(filtered_df[key] <= 1.0)]
                        elif value == 0 or value == 0.0:
                            # For other zero values, match small values
                            before_count = len(filtered_df)
                            filtered_df = filtered_df[(filtered_df[key] <= 1.0)]
                        elif key == 'total_quantity':
                            # For quantities, use an even wider range (±70%)
                            before_count = len(filtered_df)
                            lower_bound = float(value) * 0.3  # 70% less
                            upper_bound = float(value) * 3.0  # 300% more
                            filtered_df = filtered_df[(filtered_df[key] >= lower_bound) & 
                                                    (filtered_df[key] <= upper_bound)]
                        else:
                            # For other dimensions, use a 50% range
                            before_count = len(filtered_df)
                            lower_bound = float(value) * 0.5  # 50% less
                            upper_bound = float(value) * 1.5  # 50% more
                            filtered_df = filtered_df[(filtered_df[key] >= lower_bound) & 
                                                    (filtered_df[key] <= upper_bound)]
                        
                        # Log after filtering
                        after_count = len(filtered_df)
                        logger.info(f"Filtered by {key}: {before_count} → {after_count} records")
                        
                        # IMPORTANT: If filtering by this dimension eliminated all records,
                        # restore the previous state and skip this filter
                        if after_count == 0 and before_count > 0:
                            logger.info(f"Skipping {key} filter as it eliminated all matches")
                            filtered_df = data[data.index.isin(filtered_df.index.tolist() + 
                                                               data.iloc[:before_count].index.tolist())]
                    
                    elif key in categorical_features:
                        # For categorical features, if we have very few records (<5),
                        # be more lenient with matching
                        if len(filtered_df) < 5:
                            logger.info(f"Skipping {key} filter due to low record count ({len(filtered_df)})")
                            continue
                            
                        if key == 'material_id':
                            # Special handling for material_id
                            before_count = len(filtered_df)
                            
                            # Create normalized version for comparison
                            normalized_value = normalize_material_id(value)
                            filtered_df['normalized_material_id'] = filtered_df[key].apply(normalize_material_id)
                            filtered_df = filtered_df[filtered_df['normalized_material_id'] == normalized_value]
                            
                            after_count = len(filtered_df)
                            logger.info(f"Filtered by {key}: {before_count} → {after_count} records")
                            
                            # If no matches, restore previous state
                            if after_count == 0 and before_count > 0:
                                logger.info(f"Skipping {key} filter as it eliminated all matches")
                                filtered_df = data[data.index.isin(filtered_df.index.tolist() + 
                                                                 data.iloc[:before_count].index.tolist())]
                        else:
                            # For other categorical features
                            before_count = len(filtered_df)
                            
                            filtered_df = filtered_df[filtered_df[key] == value]
                            
                            after_count = len(filtered_df)
                            logger.info(f"Filtered by {key}: {before_count} → {after_count} records")
                            
                            # If no matches, restore previous state
                            if after_count == 0 and before_count > 0:
                                logger.info(f"Skipping {key} filter as it eliminated all matches")
                                filtered_df = data[data.index.isin(filtered_df.index.tolist() + 
                                                                 data.iloc[:before_count].index.tolist())]
        
        # Create a DataFrame from pred_input for model prediction
        # This will only contain fields explicitly provided by the user
        pred_df = pd.DataFrame([pred_input])
        
        # Add missing required columns for the model with None values
        # This is only for the model prediction, not for filtering
        for feature in numeric_features + categorical_features:
            if feature not in pred_df.columns:
                pred_df[feature] = None
        
        # Log the final filtered dataset size
        logger.info(f"Final filtered dataset size: {len(filtered_df)} records")
        
        # Log the final filtered dataset
        log_dataset_info(filtered_df, "Final filtered dataset")

        # If we have similar records with price data, use their range
        if not filtered_df.empty and 'price' in filtered_df.columns and len(filtered_df) >= 1:
            min_price = filtered_df['price'].min()
            max_price = filtered_df['price'].max()
            
            # If min and max are too close, add some variability
            if abs(max_price - min_price) < 0.01 or len(filtered_df) == 1:
                min_price = min_price * 0.85
                max_price = max_price * 1.15
                
            logger.info(f"Using data-based price range: {min_price} to {max_price} (from {len(filtered_df)} records)")
        else:
            # If no similar records found, make a model prediction
            logger.info("No similar records found or filtered dataset empty, using model prediction")
            
            # Make prediction based on input
            prediction = model.predict(pred_df)[0]
            
            # Use a wider range for model predictions to reflect uncertainty
            min_price = max(0, prediction * 0.8)
            max_price = prediction * 1.2
            
            logger.info(f"Model prediction: {prediction} (range: {min_price} to {max_price})")
        
        # Calculate price factors
        price_factors = calculate_price_factors(model, numeric_features, categorical_features, 
                                                pred_input, (min_price + max_price) / 2)
        
        return min_price, max_price, price_factors
    except Exception as e:
        logger.error(f"Error in analyze_and_predict: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return default values if prediction fails
        return 1.0, 2.0, [{"feature": "error", "explanation": f"Error in prediction: {str(e)}", 
                          "impact": "100% influence on price"}]

# 3. Update the predict_price endpoint to have only width, height, and product_line as required
@app.post("/predict", response_model=PredictionOutput)
async def predict_price(input_data: PredictionInput):
    logger.info(f"Received data: {input_data.dict()}")
    
    # Check if required parameters are provided
    required_fields = ['width', 'height']
    missing_fields = [field for field in required_fields 
                     if field not in input_data.dict() or input_data.dict()[field] is None]
    
    if missing_fields:
        raise HTTPException(status_code=400, 
                           detail=f"Required parameters missing: {', '.join(missing_fields)}")
    
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

# 4. Update the PredictionInput model to better handle different types
class PredictionInput(BaseModel):
    width: Optional[Union[float, str]] = None
    height: Optional[Union[float, str]] = None
    length: Optional[Union[float, str]] = None
    print_method: Optional[str] = None
    material_id: Optional[Union[str, int]] = None
    ship_via: Optional[str] = None
    factory: Optional[str] = None
    total_quantity: Optional[Union[int, str]] = None
    options: Optional[str] = None
    product_line: Optional[str] = None
    
    # Add a validator to convert string numeric values to proper types
    @validator('width', 'height', 'length', 'total_quantity', pre=True)
    def parse_numeric_values(cls, v):
        if isinstance(v, str) and v.strip():
            try:
                return float(v.strip())
            except ValueError:
                pass
        return v

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

# Run FastAPI app with uvicorn explicitly on port 10000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)