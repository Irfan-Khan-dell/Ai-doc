import streamlit as st
from google import genai
from PIL import Image
import os

# --- Page Configuration & LAR Branding ---
st.set_page_config(
    page_title="AI Doctor | LAR Diagnostics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Configuration ---
# Securely fetch the API key from Streamlit secrets or environment variables
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

def analyze_medical_case(image, patient_data):
    """
    Passes the medical report image and Indian patient data to Gemini for analysis
    using the new google-genai SDK.
    """
    if not api_key:
        return "Error: API Key not found. Please configure GEMINI_API_KEY."

    # Initialize the new Client object
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an advanced AI Medical Assistant integrated into the LAR diagnostic hub. 
    Analyze the provided medical scan/report image along with the patient's data.

    Patient Profile:
    - Age/Gender: {patient_data['age']} years old, {patient_data['gender']}
    - Location: {patient_data['district']}, {patient_data['state']}, India
    - Symptoms/Context: {patient_data['symptoms']}

    Please provide a structured response containing:
    1. **Report Analysis:** A clear, plain-English summary of what the uploaded report or scan indicates.
    2. **Care Guidelines:** General management advice and standard approaches for these indications. (Crucial: Explicitly state that this is not a formal prescription and a licensed doctor must be consulted).
    3. **Hospital Recommendations:** Based on the diagnosis, recommend 3 highly-rated specialized hospitals or clinics in or near {patient_data['district']}, {patient_data['state']}. Include the hospital name, specialty, and why it is a good fit.

    Format the output cleanly using Markdown. Maintain a professional, empathetic, and clinical tone.
    """
    
    try:
        # If an image is provided, pass both text and image; otherwise, just text.
        content_payload = [prompt, image] if image else [prompt]
        
        # Use the stateless client.models.generate_content method
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=content_payload
        )
        return response.text
    except Exception as e:
        return f"An error occurred during AI analysis: {e}"

# --- Sidebar ---
with st.sidebar:
    st.title("LAR Diagnostics")
    st.markdown("---")
    st.warning(
        "**⚠️ MEDICAL DISCLAIMER**\n\n"
        "This application is powered by Artificial Intelligence. It is intended for educational and screening "
        "purposes only. **Do not use this tool to self-diagnose or self-prescribe.** Always consult a "
        "registered medical practitioner in your area."
    )

# --- Main UI ---
st.title("👨‍⚕️ LAR AI Doctor: Clinical Analysis")
st.markdown("Complete the patient registration and upload your scan to receive a preliminary AI analysis and localized specialist recommendations.")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Patient Registration (India)")
    
    with st.form("patient_registration_form"):
        # Standard Indian patient intake fields
        patient_name = st.text_input("Full Name")
        abha_id = st.text_input("ABHA ID (14-digit Health Account Number)", max_chars=14, placeholder="e.g., 12-3456-7890-1234")
        
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=0, max_value=120, value=25)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
        c3, c4 = st.columns(2)
        with c3:
            state = st.selectbox("State", ["Rajasthan", "Delhi", "Gujarat", "Haryana", "Maharashtra", "Punjab", "Uttar Pradesh"], index=0)
        with c4:
            district = st.text_input("City/District", value="Jaipur")
            
        symptoms = st.text_area("Chief Complaints / Symptoms", placeholder="Describe current symptoms, duration, and medical history...")
        
        submit_patient_data = st.form_submit_button("Save Patient Data", type="secondary")

with col2:
    st.subheader("🖼️ Upload Medical Report")
    uploaded_file = st.file_uploader("Upload Scan or Lab Report (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
    
    img = None
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Report Ready for Analysis", use_container_width=True)

st.markdown("---")

# --- AI Execution Block ---
if st.button("Run AI Diagnostic Analysis", type="primary", use_container_width=True):
    if not symptoms.strip():
        st.error("Please enter the patient's symptoms in the registration form before running the analysis.")
    elif uploaded_file is None:
        st.warning("Running analysis without an image. Please upload a scan for a complete assessment.")
        
    if symptoms.strip():
        # Package data into a clean dictionary
        patient_data = {
            "name": patient_name,
            "age": age,
            "gender": gender,
            "state": state,
            "district": district,
            "symptoms": symptoms
        }
        
        with st.spinner("Analyzing clinical data and cross-referencing local healthcare facilities..."):
            result = analyze_medical_case(img, patient_data)
            
            st.subheader("📊 AI Clinical Report")
            st.markdown(result)

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey; font-size: small;'>Engineered by LAR Technologies &copy; 2026</p>", unsafe_allow_html=True)