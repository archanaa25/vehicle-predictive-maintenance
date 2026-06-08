
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import json

# Define model repository details
model_repo_id = "arss25/VehiclePredictiveMaintenance-Model"
model_filename = "best_engine_model.pkl"
metadata_filename = "model_metadata.json"

# Download and load the model
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename, repo_type="model")
    return joblib.load(model_path)

# Download and load model metadata
@st.cache_resource
def load_metadata():
    metadata_path = hf_hub_download(repo_id=model_repo_id, filename=metadata_filename, repo_type="model")
    with open(metadata_path, 'r') as f:
        return json.load(f)

model = load_model()
metadata = load_metadata()

# Streamlit UI for Engine Condition Prediction
st.title("Vehicle Predictive Maintenance: Engine Condition Prediction")
st.write("""
This application predicts whether an engine is operating normally or requires maintenance
based on real-time sensor data.
Please enter the engine sensor readings below to get a prediction.
""")

st.subheader("Engine Sensor Readings")

# Define feature ranges and default values based on EDA
feature_configs = {
    'Engine rpm': {'min': 61.0, 'max': 2239.0, 'default': 791.0, 'step': 1.0},
    'Lub oil pressure': {'min': 0.003, 'max': 7.266, 'default': 3.3, 'step': 0.01},
    'Fuel pressure': {'min': 0.003, 'max': 21.138, 'default': 6.6, 'step': 0.01},
    'Coolant pressure': {'min': 0.002, 'max': 7.479, 'default': 2.3, 'step': 0.01},
    'lub oil temp': {'min': 71.322, 'max': 89.581, 'default': 77.6, 'step': 0.01},
    'Coolant temp': {'min': 61.673, 'max': 195.528, 'default': 78.4, 'step': 0.01}
}

numeric_features = metadata.get("features", [
    'Engine rpm', 'Lub oil pressure', 'Fuel pressure',
    'Coolant pressure', 'lub oil temp', 'Coolant temp'
])

input_values = {}
for feature in numeric_features:
    config = feature_configs.get(feature, {'min': 0.0, 'max': 1000.0, 'default': 500.0, 'step': 0.1}) # Fallback
    input_values[feature] = st.number_input(
        f"{feature.replace('_', ' ').title()}",
        min_value=config['min'],
        max_value=config['max'],
        value=config['default'],
        step=config['step']
    )

# Provide guidance on typical sensor ranges and faulty conditions
st.markdown("""
### Sensor Reading Guidance
- **Engine RPM**: Typical range is 61 - 2239 RPM. High RPMs might indicate stress.
- **Lub Oil Pressure**: Normal range is generally 2.5 - 4.0 bar. Low pressure (below 2.5 bar) can indicate issues.
- **Fuel Pressure**: Normal range is typically 4.0 - 8.0 bar. Low pressure can lead to fuel delivery problems.
- **Coolant Pressure**: Normal range is around 1.5 - 3.0 bar. Low pressure indicates a leak, while very high pressure can mean overheating.
- **Lub Oil Temp**: Optimal operating range 75 - 80°C. Higher temperatures (above 80°C) could indicate overheating.
- **Coolant Temp**: Optimal operating range 70 - 85°C. Temperatures significantly above 90°C (like 195°C) are critical overheating indicators.

**Engine Faulty Condition Indicators (from EDA):**
- High Engine RPM
- Low Lub Oil Pressure
- Low Fuel Pressure
- High Coolant Temperature
- High Lub Oil Temperature
""")

# Assemble input into DataFrame
input_data = pd.DataFrame([input_values])

if st.button("Predict Engine Condition"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "The engine is likely in a **faulty** condition and requires maintenance."
        st.error(f"Prediction Result: {result}")
    else:
        result = "The engine is likely operating **normally**."
        st.success(f"Prediction Result: {result}")
