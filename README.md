# 🏥 AI Disease Predictor

A comprehensive, Streamlit-based web application that utilizes Machine Learning and Deep Learning models to predict the likelihood of multiple diseases. 

By unifying 7 different prediction models into a single, dynamic interface, this tool allows users to input clinical parameters or upload medical imagery to receive instant AI-driven diagnostic analysis and risk probability breakdowns.

---

### 🔬 Supported Disease Predictions
**Tabular Clinical Models (Machine Learning):**
* 🩸 **Diabetes**
* 🫀 **Heart Disease**
* 🎗️ **Breast Cancer**
* 🧬 **Kidney Disease**
* 🧪 **Liver Disease**

**Image-Based Models (Deep Learning):**
* 🦠 **Malaria** (Microscopic Cell Imaging)
* 🫁 **Pneumonia** (Chest X-Ray Imaging)

---

### ✨ Key Features
* **Dynamic Model Routing:** The app automatically scans the `models/` directory upon startup. It natively builds the UI and navigation only for the models it successfully detects, preventing crashes from missing files.
* **Probability Visualizations:** Beyond simple positive/negative outputs, the app calculates confidence scores and renders interactive Risk Level bars and Probability Breakdown charts using Pandas.
* **Image Processing:** Deep learning modules automatically resize, normalize, and process uploaded medical scans (RGB and Grayscale) in-memory before passing them to the TensorFlow `.h5` models.
* **Responsive UI:** Built entirely in Python using Streamlit, featuring a clean sidebar navigation system and dynamic input forms mapped to specific medical terminology.

---

### 🛠️ Technology Stack
* **Frontend/Backend:** [Streamlit](https://streamlit.io/) (Replaced Flask)
* **Deep Learning:** TensorFlow / Keras (`.h5` models)
* **Machine Learning:** Scikit-Learn (`.pkl` models)
* **Data Processing & Visualization:** Pandas, NumPy, Pillow (PIL)

---

### 🚀 Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/AIDiseasePredictor.git](https://github.com/yourusername/AIDiseasePredictor.git)
cd AIDiseasePredictor-main