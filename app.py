import streamlit as st
import pandas as pd
import joblib
import os

def get_model_path():
    """Get absolute path to the model file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, 'pipeline', 'models', 'car_price_model.pkl')

def validate_model_exists():
    """Check if model file exists and show appropriate messages"""
    model_path = get_model_path()
    if not os.path.exists(model_path):
        st.error(f"""
        Model file not found at {model_path}
        """)
        return False
    return True

def load_model():
    """Load the trained model from disk"""
    try:
        model_path = get_model_path()
        if not validate_model_exists():
            return None
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Set page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("Car Price Prediction")

# Load the model
model = load_model()

if model:
    # Collect user input
    st.header("Enter Car Details")
    
    # Numeric inputs
    with st.container():
        st.subheader("Car Specifications")
        carlength = st.number_input("Car Length", min_value=100.0, max_value=300.0, step=0.5)
        boreratio = st.number_input("Bore Ratio", min_value=2.0, max_value=5.0, step=0.01)
        highwaympg = st.number_input("Highway MPG", min_value=10, max_value=80, step=1)
        enginesize = st.number_input("Engine Size", min_value=50, max_value=400, step=1)

    # Categorical inputs
    with st.container():
        st.subheader("Car Features")
        carbody = st.selectbox("Car Body", ['sedan', 'hatchback', 'wagon', 'hardtop', 'convertible'])
        carbrand = st.selectbox("Car Brand", ['toyota', 'honda', 'nissan', 'mazda', 'mitsubishi', 'other'])
        drivewheel = st.selectbox("Drive Wheel", ['fwd', 'rwd', '4wd'])
        aspiration = st.selectbox("Aspiration", ['std', 'turbo'])
        enginelocation = st.selectbox("Engine Location", ['front', 'rear'])
        enginetype = st.selectbox("Engine Type", ['ohc', 'dohc', 'ohcf', 'ohcv', 'l', 'rotor', 'dohcv'])
        cylindernumber = st.selectbox("Cylinder Number", ['four', 'six', 'five', 'eight', 'two', 'three', 'twelve'])
        fuelsystem = st.selectbox("Fuel System", ['mpfi', '2bbl', '1bbl', '4bbl', 'idi', 'spdi', 'spfi', 'mfi'])

    # Prepare input data
    input_df = pd.DataFrame([{
        'carlength': carlength,
        'boreratio': boreratio,
        'highwaympg': highwaympg,
        'enginesize': enginesize,
        'carbody': carbody,
        'carbrand': carbrand,
        'drivewheel': drivewheel,
        'aspiration': aspiration,
        'enginelocation': enginelocation,
        'enginetype': enginetype,
        'cylindernumber': cylindernumber,
        'fuelsystem': fuelsystem
    }])

    # Make prediction
    if st.button("Predict Price"):
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Car Price: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")