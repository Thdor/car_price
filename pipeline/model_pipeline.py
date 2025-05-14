from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

def build_pipeline(numeric_features, categorical_features):
    """Build the preprocessing and modeling pipeline"""
    
    # Numeric preprocessing
    numeric_transformer = StandardScaler()
    
    # Categorical preprocessing
    categorical_transformer = OneHotEncoder(
        sparse_output=False,  # Changed from sparse=False
        handle_unknown='ignore'
    )
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    return pipeline