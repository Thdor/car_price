import pandas as pd
import numpy as np

def clean_data(df):
    """Clean and validate the car price dataset"""
    
    # Create a copy of the dataframe
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    #carbrand 
    df['carbrand'] = df['CarName'].str.lower().str.split(' ', expand=True)[0]
    replace_dict = {
        'porcshce': 'porsche',
        'maxda': 'mazda',
        'toyouta':'toyota',
        'vokswagen':'volkswagen',
        'vw':'volkswagen'
    }
    df['carbrand'] = df['carbrand'].replace(replace_dict)
    
    # Handle missing values
    numeric_cols = ['carlength', 'enginesize', 'boreratio', 'highwaympg', 'price']
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = ['aspiration', 'carbody', 'drivewheel', 'enginelocation',
                       'enginetype', 'cylindernumber', 'fuelsystem', 'carbrand']
    df[categorical_cols] = df[categorical_cols].fillna('unknown')
    
    # Validate data types
    df[numeric_cols] = df[numeric_cols].astype(float)
    df[categorical_cols] = df[categorical_cols].astype(str)
    
    return df