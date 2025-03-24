import os
import numpy as np
import torch
import cv2
from PIL import Image
import pydicom
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import SimpleITK as sitk

# Configure application
st.set_page_config(
    page_title="AI-Powered Medical Diagnosis System",
    page_icon="🏥",
    layout="wide"
)

# Application title and description
st.title("AI-Powered Medical Diagnosis System")
st.markdown("""
    This system uses deep learning to assist in medical diagnoses through image analysis.
    Upload medical images (CT scans, MRIs) to get instant AI-powered analysis.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Diagnosis", "Model Performance", "About"])

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Load pre-trained models
@st.cache_resource
def load_models():
    # PyTorch model for CT scan analysis
    torch_model = torch.load("models/ct_analyzer.pt", map_location=torch.device('cpu')) # Ensure CPU usage
    torch_model.eval()

    return {"torch_model": torch_model}

# Image preprocessing
def preprocess_image(image, model_type):
    if model_type == "ct":
        # Preprocessing for CT scans
        img = cv2.resize(image, (256, 256))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img.unsqueeze(0)

# Diagnosis function
def run_diagnosis(image, image_type):
    models = load_models()

    if image_type == "CT Scan":
        # Process CT scan with PyTorch model
        processed_img = preprocess_image(image, "ct")
        with torch.no_grad():
            prediction = models["torch_model"](processed_img)
        classes = ["Normal", "Hemorrhage", "Stroke", "Tumor", "Fracture"]
        result = {classes[i]: float(prediction[0][i]) for i in range(len(classes))}

    elif image_type == "MRI":
        # Process MRI with PyTorch model (same preprocessing as CT)
        processed_img = preprocess_image(image, "ct")
        with torch.no_grad():
            prediction = models["torch_model"](processed_img)
        classes = ["Normal", "Brain Tumor", "Alzheimer's", "Multiple Sclerosis", "Stroke"]
        result = {classes[i]: float(prediction[0][i]) for i in range(len(classes))}

    return result

# Home page
if page == "Home":
    st.header("Welcome to the AI Medical Diagnosis System")
    st.write("This platform uses state-of-the-art deep learning models to assist medical professionals in diagnosing various conditions.")

    st.subheader("Features")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            - **Multiple imaging modalities**: CT scans, MRIs
            - **Real-time analysis**: Get results in seconds
            - **High accuracy**: Models trained on large medical datasets
            - **Detailed reports**: Get probability scores for various conditions
        """)

    with col2:
        # Load the image
        try:
            image = Image.open("pic.jpg")  # Ensure pic.jpg is in the same directory or provide the correct path
            st.image(image, caption="AI-powered analysis in action")
        except FileNotFoundError:
            st.image("https://via.placeholder.com/400x300", caption="AI-powered analysis in action") #fallback image

# Diagnosis page
elif page == "Diagnosis":
    st.header("Medical Image Analysis")

    # Image upload
    uploaded_file = st.file_uploader("Upload medical image", type=["jpg", "png", "dcm"])

    # Image type selection
    image_type = st.selectbox("Select imaging type", ["CT Scan", "MRI"])

    if uploaded_file is not None:
        # Handle DICOM files
        if uploaded_file.name.endswith('.dcm'):
            dicom_data = pydicom.dcmread(uploaded_file)
            image = dicom_data.pixel_array
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # Handle regular image files
            image = np.array(Image.open(uploaded_file))

        # Display the uploaded image
        st.image(image, caption=f"Uploaded {image_type}", width=400)

        # Run diagnosis on button click
        if st.button("Run Diagnosis"):
            with st.spinner("Analyzing image..."):
                results = run_diagnosis(image, image_type)

            # Display results
            st.subheader("Diagnosis Results")

            # Create a bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            conditions = list(results.keys())
            probabilities = list(results.values())

            # Sort results by probability
            sorted_indices = np.argsort(probabilities)[::-1]
            sorted_conditions = [conditions[i] for i in sorted_indices]
            sorted_probabilities = [probabilities[i] for i in sorted_indices]

            ax.barh(sorted_conditions, sorted_probabilities, color='skyblue')
            ax.set_xlabel('Probability')
            ax.set_title('Diagnosis Probabilities')

            st.pyplot(fig)

            # Primary diagnosis (highest probability)
            primary_condition = sorted_conditions[0]
            primary_prob = sorted_probabilities[0]

            st.success(f"Primary diagnosis: {primary_condition} (Confidence: {primary_prob:.2%})")

            # Add to history
            st.session_state.history.append({
                "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "image_type": image_type,
                "diagnosis": primary_condition,
                "confidence": primary_prob
            })

            # Recommendations based on diagnosis
            st.subheader("Recommendations")
            st.write("Based on the AI analysis, consider the following steps:")

            if primary_prob > 0.8:
                st.write("- High confidence diagnosis. Recommend confirming with specialist.")
            else:
                st.write("- Moderate confidence. Additional tests recommended.")

            # Additional specific recommendations based on condition
            if "Tumor" in primary_condition or "Stroke" in primary_condition:
                st.write("- Urgent referral to neurology recommended")
                st.write("- Further imaging and possible biopsy advised")

# Model Performance page
elif page == "Model Performance":
    st.header("Model Performance Metrics")

    st.subheader("Accuracy on Test Sets")

    # Create dummy performance data
    metrics = {
        "CT Scan Model": {"accuracy": 0.88, "precision": 0.86, "recall": 0.85, "f1": 0.85},
        "MRI Model": {"accuracy": 0.90, "precision": 0.89, "recall": 0.87, "f1": 0.88}
    }

    # Display metrics
    for model, metric in metrics.items():
        st.subheader(model)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metric['accuracy']:.2%}")
        col2.metric("Precision", f"{metric['precision']:.2%}")
        col3.metric("Recall", f"{metric['recall']:.2%}")
        col4.metric("F1 Score", f"{metric['f1']:.2%}")

    # Sample confusion matrix
    st.subheader("Confusion Matrix (CT Scan Model)")
    confusion = np.array([
        [118, 7, 3, 2, 0],
        [5, 125, 8, 2, 0],
        [2, 6, 110, 2, 0],
        [1, 3, 2, 104, 0],
        [0, 1, 2, 0, 97]
    ])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion, cmap='Blues')
    ax.set_xticks(np.arange(5))
    ax.set