import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import os

# --- Page Configuration ---
st.set_page_config(page_title="AI Disease Predictor", page_icon="🏥", layout="wide")

# --- Supported Models Configuration ---
SUPPORTED_MODELS = {
    "diabetes.pkl": {"name": "Diabetes", "type": "tabular", "features": 8},
    "breast_cancer.pkl": {"name": "Breast Cancer", "type": "tabular", "features": 26},
    "heart.pkl": {"name": "Heart Disease", "type": "tabular", "features": 13},
    "kidney.pkl": {"name": "Kidney Disease", "type": "tabular", "features": 18},
    "liver.pkl": {"name": "Liver Disease", "type": "tabular", "features": 10},
    "malaria.h5": {"name": "Malaria", "type": "image"},
    "pneumonia.h5": {"name": "Pneumonia", "type": "image"}
}

# --- Exact Feature Names Mapped from Datasets ---
FEATURE_DICT = {
    "Diabetes": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
    "Heart Disease": ["Age", "Sex (1=M, 0=F)", "Chest Pain Type (cp)", "Resting Blood Pressure (trestbps)", "Cholesterol (chol)", "Fasting Blood Sugar (fbs)", "Resting ECG (restecg)", "Max Heart Rate (thalach)", "Exercise Induced Angina (exang)", "Oldpeak", "Slope", "Number of Major Vessels (ca)", "Thalassemia (thal)"],
    "Liver Disease": ["Age", "Gender (1=M, 0=F)", "Total Bilirubin", "Direct Bilirubin", "Alkaline Phosphotase", "Alamine Aminotransferase", "Aspartate Aminotransferase", "Total Proteins", "Albumin", "Albumin and Globulin Ratio"],
    "Breast Cancer": ["Radius (Mean)", "Texture (Mean)", "Perimeter (Mean)", "Area (Mean)", "Smoothness (Mean)", "Compactness (Mean)", "Concavity (Mean)", "Concave Points (Mean)", "Symmetry (Mean)", "Radius (SE)", "Perimeter (SE)", "Area (SE)", "Compactness (SE)", "Concavity (SE)", "Concave Points (SE)", "Fractal Dimension (SE)", "Radius (Worst)", "Texture (Worst)", "Perimeter (Worst)", "Area (Worst)", "Smoothness (Worst)", "Compactness (Worst)", "Concavity (Worst)", "Concave Points (Worst)", "Symmetry (Worst)", "Fractal Dimension (Worst)"],
    "Kidney Disease": ["Age", "Blood Pressure (bp)", "Specific Gravity (sg)", "Albumin (al)", "Sugar (su)", "Red Blood Cells (rbc)", "Pus Cell (pc)", "Pus Cell Clumps (pcc)", "Bacteria (ba)", "Blood Glucose Random (bgr)", "Blood Urea (bu)", "Serum Creatinine (sc)", "Sodium (sod)", "Potassium (pot)", "Hemoglobin (hemo)", "Packed Cell Volume (pcv)", "White Blood Cell Count (wc)", "Red Blood Cell Count (rc)"]
}

# --- Dynamic File Detection ---
available_pages = ["Home"]
active_models = {}

if os.path.exists("models"):
    for file in os.listdir("models"):
        if file in SUPPORTED_MODELS:
            disease_name = SUPPORTED_MODELS[file]["name"]
            available_pages.append(disease_name)
            active_models[disease_name] = SUPPORTED_MODELS[file]
            active_models[disease_name]["path"] = os.path.join("models", file)

# --- Resource Loading (Cached) ---
@st.cache_resource
def load_tabular_model(path):
    return pickle.load(open(path, 'rb'))

@st.cache_resource
def load_image_model(path):
    from tensorflow.keras.models import load_model 
    return load_model(path)

# --- Sidebar Navigation ---
st.sidebar.title("AI Disease Predictor")
st.sidebar.subheader("Navigation")

if len(available_pages) == 1:
    st.sidebar.warning("⚠️ No valid model files found in the 'models/' folder.")
else:
    page = st.sidebar.radio("Select a module:", available_pages)

# --- Page Routing & UI Logic ---
if page == "Home":
    st.title("Welcome to AI Disease Predictor")
    st.write(f"**System Status:** Found {len(available_pages)-1} active models in your folder.")
    st.write("""
    Select an available disease from the sidebar to input your data and get a prediction.
    """)

# --- Handle Tabular Models ---
elif active_models[page]["type"] == "tabular":
    disease_info = active_models[page]
    st.header(f"{page} Prediction")
    st.write(f"Please enter the {disease_info['features']} required clinical parameters below.")
    
    try:
        model = load_tabular_model(disease_info["path"])
        feature_labels = FEATURE_DICT.get(page, [])
        
        with st.form(f"{page}_form"):
            cols = st.columns(3)
            input_values = []
            
            for i in range(disease_info["features"]):
                with cols[i % 3]:
                    label = feature_labels[i] if i < len(feature_labels) else f"Clinical Feature {i+1}"
                    val = st.number_input(label, value=0.0, format="%.4f")
                    input_values.append(val)
                    
            if st.form_submit_button("Predict"):
                values_array = np.asarray(input_values).reshape(1, -1)
                prediction = model.predict(values_array)[0]
                
                st.markdown("---")
                st.subheader("🧠 AI Analysis Results")

                # Display the primary text alert
                if prediction == 1:
                    st.error(f"**Primary Diagnosis:** Positive for {page}")
                else:
                    st.success(f"**Primary Diagnosis:** Negative for {page}")
                
                # Check if the model supports probability tracking
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(values_array)[0]
                    neg_prob, pos_prob = proba[0], proba[1]
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    col1.metric("Likelihood of being Positive (Risk)", f"{pos_prob * 100:.1f}%")
                    col2.metric("Likelihood of being Negative (Clear)", f"{neg_prob * 100:.1f}%")
                    
                    # Display progress bar for visual risk
                    st.write("**Risk Level Bar:**")
                    st.progress(float(pos_prob))
                    
                    # Display Bar Chart
                    st.write("**Probability Breakdown Chart:**")
                    chart_data = pd.DataFrame(
                        {"Probability (%)": [neg_prob * 100, pos_prob * 100]},
                        index=["Negative", "Positive"]
                    )
                    st.bar_chart(chart_data)
                else:
                    st.info("Note: This specific model type does not support percentage breakdowns.")
                    
    except Exception as e:
        st.error(f"Failed to load the model or make a prediction. Error: {e}")

# --- Handle Image Models ---
elif active_models[page]["type"] == "image":
    disease_info = active_models[page]
    st.header(f"{page} Prediction")
    st.write("Upload the required medical scan/image for analysis.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "jfif"])
    
    if uploaded_file is not None:
        is_grayscale = (page == "Pneumonia")
        img = Image.open(uploaded_file).convert('L') if is_grayscale else Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', width=300)
        
        if st.button(f"Predict {page}"):
            with st.spinner('Analyzing Image...'):
                try:
                    model = load_image_model(disease_info["path"])
                    
                    img = img.resize((36, 36))
                    img = np.asarray(img)
                    
                    if is_grayscale:
                        img = img.reshape((1, 36, 36, 1))
                        img = img / 255.0 
                    else:
                        img = img.reshape((1, 36, 36, 3))
                        img = img.astype(np.float64)
                        
                    raw_pred = model.predict(img)[0]
                    
                    # Deep learning models return raw probabilities natively
                    if len(raw_pred) >= 2:
                        neg_prob, pos_prob = raw_pred[0], raw_pred[1]
                        pred = np.argmax(raw_pred)
                    else:
                        pos_prob = raw_pred[0]
                        neg_prob = 1 - pos_prob
                        pred = 1 if pos_prob > 0.5 else 0

                    st.markdown("---")
                    st.subheader("🧠 AI Analysis Results")

                    if pred == 1:
                        st.error(f"**Primary Diagnosis:** Positive (Detected)")
                    else:
                        st.success(f"**Primary Diagnosis:** Negative (Normal/Uninfected)")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Likelihood of being Positive (Risk)", f"{pos_prob * 100:.1f}%")
                    col2.metric("Likelihood of being Negative (Clear)", f"{neg_prob * 100:.1f}%")
                    
                    st.write("**Risk Level Bar:**")
                    st.progress(float(pos_prob))
                    
                    st.write("**Probability Breakdown Chart:**")
                    chart_data = pd.DataFrame(
                        {"Probability (%)": [neg_prob * 100, pos_prob * 100]},
                        index=["Negative", "Positive"]
                    )
                    st.bar_chart(chart_data)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")