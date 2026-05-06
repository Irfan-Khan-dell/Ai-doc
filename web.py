import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import os

# --- Page Configuration & Branding ---
st.set_page_config(
    page_title="AI Disease Predictor | LAR", 
    page_icon="⚕️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Supported Models Configuration ---
SUPPORTED_MODELS = {
    "diabetes.pkl": {"name": "Diabetes Assessment", "type": "tabular", "features": 8},
    "breast_cancer.pkl": {"name": "Breast Cancer Analysis", "type": "tabular", "features": 26},
    "heart.pkl": {"name": "Heart Disease Risk", "type": "tabular", "features": 13},
    "kidney.pkl": {"name": "Kidney Disease Screen", "type": "tabular", "features": 18},
    "liver.pkl": {"name": "Liver Function Test", "type": "tabular", "features": 10},
    "malaria.h5": {"name": "Malaria Cell Scan", "type": "image"},
    "pneumonia.h5": {"name": "Pneumonia X-Ray", "type": "image"}
}

# --- Clinical Feature Mapping ---
FEATURE_DICT = {
    "Diabetes Assessment": ["Pregnancies", "Glucose Level", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree Function", "Age"],
    "Heart Disease Risk": ["Age", "Sex (1=M, 0=F)", "Chest Pain Type (0-3)", "Resting Blood Pressure", "Cholesterol (mg/dl)", "Fasting Blood Sugar > 120 (1=T, 0=F)", "Resting ECG Results (0-2)", "Max Heart Rate Achieved", "Exercise Induced Angina (1=Y, 0=N)", "ST Depression (Oldpeak)", "Slope of Peak ST Segment (0-2)", "Number of Major Vessels (0-3)", "Thalassemia (0-3)"],
    "Liver Function Test": ["Age", "Gender (1=M, 0=F)", "Total Bilirubin", "Direct Bilirubin", "Alkaline Phosphotase", "Alamine Aminotransferase", "Aspartate Aminotransferase", "Total Proteins", "Albumin", "Albumin and Globulin Ratio"],
    "Breast Cancer Analysis": ["Radius (Mean)", "Texture (Mean)", "Perimeter (Mean)", "Area (Mean)", "Smoothness (Mean)", "Compactness (Mean)", "Concavity (Mean)", "Concave Points (Mean)", "Symmetry (Mean)", "Radius (SE)", "Perimeter (SE)", "Area (SE)", "Compactness (SE)", "Concavity (SE)", "Concave Points (SE)", "Fractal Dimension (SE)", "Radius (Worst)", "Texture (Worst)", "Perimeter (Worst)", "Area (Worst)", "Smoothness (Worst)", "Compactness (Worst)", "Concavity (Worst)", "Concave Points (Worst)", "Symmetry (Worst)", "Fractal Dimension (Worst)"],
    "Kidney Disease Screen": ["Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar", "Red Blood Cells (1=Normal)", "Pus Cell (1=Normal)", "Pus Cell Clumps (1=Present)", "Bacteria (1=Present)", "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium", "Potassium", "Hemoglobin", "Packed Cell Volume", "White Blood Cell Count", "Red Blood Cell Count"]
}

# --- Dynamic File Detection ---
available_pages = ["Dashboard (Home)"]
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

# --- Sidebar Navigation & Styling ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
    st.title("MedAI Diagnostics")
    st.markdown("---")
    page = st.radio("Select Diagnostic Module:", available_pages)
    
    st.markdown("---")
    st.warning("**⚠️ MEDICAL DISCLAIMER**\n\nThis application is powered by Artificial Intelligence. It is intended for educational and screening purposes only. **Do not use this tool to self-diagnose.** Always consult a qualified healthcare professional for medical advice.")

# --- Page Routing & UI Logic ---
if page == "Dashboard (Home)":
    st.title("🏥 Medical AI Diagnostic Hub")
    st.markdown("Welcome to the clinical prediction dashboard. Please select a diagnostic module from the sidebar to begin screening.")
    
    # Dashboard Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Active ML Models", len([m for m in active_models.values() if m['type'] == 'tabular']))
    col2.metric("Active Deep Learning Models", len([m for m in active_models.values() if m['type'] == 'image']))
    col3.metric("System Status", "Online & Ready")
    
    st.info("👈 Use the navigation panel on the left to select a specific disease predictor.")

# --- Handle Tabular Models ---
elif page in active_models and active_models[page]["type"] == "tabular":
    disease_info = active_models[page]
    st.title(f"📊 {page}")
    st.markdown(f"Enter the patient's clinical parameters below. All **{disease_info['features']}** fields are required for an accurate algorithmic assessment.")
    st.markdown("---")
    
    try:
        model = load_tabular_model(disease_info["path"])
        feature_labels = FEATURE_DICT.get(page, [])
        
        with st.form(f"{page}_form"):
            st.subheader("Patient Clinical Data")
            # Group inputs into a clean 4-column grid
            cols = st.columns(4)
            input_values = []
            
            for i in range(disease_info["features"]):
                with cols[i % 4]:
                    label = feature_labels[i] if i < len(feature_labels) else f"Feature {i+1}"
                    val = st.number_input(label, value=0.0, format="%.2f")
                    input_values.append(val)
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("Run AI Diagnostic Analysis", type="primary", use_container_width=True)
            
            if submit_btn:
                st.markdown("---")
                values_array = np.asarray(input_values).reshape(1, -1)
                prediction = model.predict(values_array)[0]
                
                st.subheader("Diagnostic Results")

                if prediction == 1:
                    st.error(f"🚨 **Alert:** The model indicates a **High Risk** (Positive) for {page.replace(' Assessment', '').replace(' Analysis', '')}.")
                else:
                    st.success(f"✅ **Clear:** The model indicates a **Low Risk** (Negative) for {page.replace(' Assessment', '').replace(' Analysis', '')}.")
                
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(values_array)[0]
                    neg_prob, pos_prob = proba[0], proba[1]
                    
                    st.markdown("##### Confidence Metrics")
                    col_chart, col_metrics = st.columns([2, 1])
                    
                    with col_metrics:
                        st.metric("Probability (Positive)", f"{pos_prob * 100:.1f}%")
                        st.metric("Probability (Negative)", f"{neg_prob * 100:.1f}%")
                    
                    with col_chart:
                        chart_data = pd.DataFrame({"Probability (%)": [neg_prob * 100, pos_prob * 100]}, index=["Negative (Healthy)", "Positive (At Risk)"])
                        st.bar_chart(chart_data, color=["#1f77b4"])
                    
    except Exception as e:
        st.error("System encountered an error loading the predictive model. Please check the integrity of the .pkl files.")
# --- Handle Image Models ---
elif page in active_models and active_models[page]["type"] == "image":
    disease_info = active_models[page]
    st.title(f"🔬 {page}")
    st.markdown("Upload the required medical scan to run it through the Deep Learning vision model.")
    st.markdown("---")
    
    # 1. Ask for Vitals ONLY if it's the advanced Pneumonia model
    patient_age, patient_temp = 30, 37.0 
    is_pneumonia = ("Pneumonia" in page)
    
    if is_pneumonia:
        st.subheader("Step 1: Patient Vitals")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=30)
        with col_v2:
            patient_temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=43.0, value=37.0)
        st.markdown("---")
        st.subheader("Step 2: X-Ray Scan")
    
    # 2. Handle Image Upload
    col_upload, col_preview = st.columns([1, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader("Select Scan File (JPG, PNG)", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Convert to Grayscale for X-Rays, keep RGB for Malaria
            img = Image.open(uploaded_file).convert('L') if is_pneumonia else Image.open(uploaded_file)
            
            with col_preview:
                st.image(img, caption='Uploaded File Ready for Analysis', use_container_width=True)
            
            if st.button("Run AI Vision Analysis", type="primary", use_container_width=True):
                with st.spinner('Processing neural network layers...'):
                    try:
                        model = load_image_model(disease_info["path"])
                        
                        # Preprocess image
                        img = img.resize((36, 36))
                        img_array = np.asarray(img)
                        
                        st.markdown("---")
                        st.subheader("Diagnostic Results")
                        
                        # --- ROUTE A: Multi-Modal Pneumonia (4 Classes) ---
                        if is_pneumonia:
                            img_array = img_array.reshape((1, 36, 36, 1)) / 255.0 
                            meta_array = np.array([[patient_age, patient_temp]])
                            
                            # Pass BOTH inputs to the model
                            raw_pred = model.predict([img_array, meta_array])[0]
                            
                            pred_idx = np.argmax(raw_pred)
                            confidence = raw_pred[pred_idx]
                            disease_classes = ["Normal", "Bacterial Pneumonia", "COVID-19", "Tuberculosis"]
                            predicted_disease = disease_classes[pred_idx]
                            
                            if pred_idx == 0:
                                st.success(f"✅ **Clear:** Scan appears {predicted_disease} ({confidence*100:.1f}% Confidence).")
                            else:
                                st.error(f"🚨 **Alert:** Detected signs of {predicted_disease} ({confidence*100:.1f}% Confidence).")
                                
                            st.markdown("##### Full Probability Breakdown")
                            chart_data = pd.DataFrame({"Probability (%)": raw_pred * 100}, index=disease_classes)
                            st.bar_chart(chart_data)
                        
                        # --- ROUTE B: Standard Malaria Model (Binary) ---
                        else:
                            img_array = img_array.reshape((1, 36, 36, 3)).astype(np.float64)
                            
                            # Pass ONLY the image
                            raw_pred = model.predict(img_array)[0]
                            
                            # Handle different output shapes safely
                            if len(raw_pred) >= 2:
                                neg_prob, pos_prob = raw_pred[0], raw_pred[1]
                                pred = np.argmax(raw_pred)
                            else:
                                pos_prob = raw_pred[0]
                                neg_prob = 1 - pos_prob
                                pred = 1 if pos_prob > 0.5 else 0

                            if pred == 1:
                                st.error(f"🚨 **Alert:** Scan shows anomalies consistent with Positive Diagnosis.")
                            else:
                                st.success(f"✅ **Clear:** Scan appears Normal / Uninfected.")
                            
                            st.markdown("##### AI Confidence Metrics")
                            st.progress(float(pos_prob), text=f"Calculated Risk Threshold: {pos_prob * 100:.1f}%")
                            
                    except Exception as e:
                        st.error(f"System Error during analysis: {e}")

# --- Footer Branding ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey; font-size: small;'>Developed by Irfan Khan | LAR Technologies &copy; 2026</p>", unsafe_allow_html=True)