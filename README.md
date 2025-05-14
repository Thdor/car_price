# Car Price Prediction Project

## Overview
A machine learning project that predicts car prices using various features like engine specifications, car dimensions, and brand categories. The model achieves 86.5% accuracy (R² score) using Linear Regression.

## Project Structure
```
no2/
├── README.md
├── app.py                 # Streamlit web application
├── pyproject.toml     # Project dependencies
├── data/
│   └── car_price.csv     # Dataset
├── pipeline/
│   ├── models/           # Saved model files
│   ├── train_model.py    # Training pipeline
│   └── model_pipeline.py # Model architecture
│   └── data_cleaning.py # Preparing data
└── notebooks/
    └── car.ipynb        # EDA and model development
```

## Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering and selection
- Model training and evaluation
- Interactive web interface using Streamlit

## Key Findings
- Achieved 86.5% prediction accuracy using Linear Regression
- Mean Absolute Error: $3,435
- Root Mean Square Error: $4,968
- Key price drivers: engine size, bore ratio, and highway MPG

## Technical Details
- **Languages & Tools**: Python, Streamlit
- **Libraries**: scikit-learn, pandas, numpy, seaborn
- **Machine Learning**: Linear Regression with cross-validation
- **Data Processing**: StandardScaler, OneHotEncoder
- **Visualization**: Matplotlib, Seaborn

## Model Features
### Numeric Features
- Car length
- Engine size
- Bore ratio
- Highway MPG

### Categorical Features
- Aspiration
- Car body
- Drive wheel
- Engine location
- Engine type
- Cylinder number
- Fuel system
- Car brand

## Project Highlights
- Comprehensive data cleaning and feature engineering
- Robust model validation using 5-fold cross-validation
- Interactive web interface for real-time predictions
- Systematic handling of categorical variables

**Car Price Prediction System** | Python, Scikit-learn, Streamlit
- Developed an end-to-end machine learning pipeline achieving 86.5% accuracy in car price predictions
- Implemented feature engineering and selection techniques reducing input dimensions while maintaining model performance
- Built an interactive web application using Streamlit for real-time price predictions
- Applied statistical analysis and data visualization to identify key price drivers in the automotive market
