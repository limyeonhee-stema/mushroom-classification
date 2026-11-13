import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Mushroom Classifier", layout="centered")

st.title("🍄 Mushroom Edibility Classifier")
st.write("Enter the traits of your mushroom to check if it is **Edible** or **Poisonous**.")

# Load the saved model and encoders
with open('mushroom_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('encoders.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)

# Define the features our model expects
features = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]
def mushroom_selectbox(feature, label):
    codes = list(CODE_TO_NAME[feature].keys())

    return st.selectbox(
        label,
        options=codes,
        format_func=lambda x: CODE_TO_NAME[feature][x]
    )

# Construct the input form
inputs = {}
st.subheader("Specify Mushroom Traits:")

col1, col2 = st.columns(2)

for i, col in enumerate(features):
    # Get original classes/labels for dropdown options
    options = list(encoders[col].classes_)
    
    # Divide the features visually into two columns
    with col1 if i % 2 == 0 else col2:
        selected_val = st.selectbox(f"{col.replace('-', ' ').title()}", options)
        # Encode the selected value to its corresponding numeric representation
        inputs[col] = encoders[col].transform([selected_val])[0]

# Predict button
if st.button("Predict Edibility", use_container_width=True):
    # Format inputs for model
    input_df = pd.DataFrame([inputs])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    st.markdown("---")
    if prediction == 0:
        st.success(f"🟢 **Edible!** (Confidence: {probability[0]*100:.1f}%)")
    else:
        st.error(f"🔴 **Poisonous!** (Confidence: {probability[1]*100:.1f}%)")
