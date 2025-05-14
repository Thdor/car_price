import pandas as pd
import joblib
import os
from model_pipeline import build_pipeline
from data_cleaning import clean_data

def ensure_directories():
    """Create necessary directories if they don't exist"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(current_dir, 'models')
    
    try:
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        return data_dir, models_dir
    except PermissionError:
        raise PermissionError("Cannot create directories. Please run the script with appropriate permissions.")

def load_data(filepath):
    """Load data with error handling"""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {filepath}. Please ensure the file exists.")

def main():
    # Ensure directories exist
    data_dir, models_dir = ensure_directories()
    
    # Define data path using absolute path
    data_path = os.path.join(data_dir, 'car_price.csv')
    
    # Load dataset
    df = load_data(data_path)
    df = clean_data(df)    
    # Define features
    numeric_features = ['carlength', 'enginesize', 'boreratio', 'highwaympg']
    categorical_features = ['aspiration', 'carbody', 'drivewheel', 'enginelocation',
                          'enginetype', 'cylindernumber', 'fuelsystem', 'carbrand']
    
    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Build and train pipeline
    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X, y)
    
    # Save model
    model_path = os.path.join(models_dir, 'car_price_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Model saved successfully at {model_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")