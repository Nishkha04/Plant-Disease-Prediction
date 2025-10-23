import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


# --- Configuration and File Loading ---

# Determine the working directory dynamically
working_dir = os.path.dirname(os.path.abspath(__file__))

# Define the model and class indices paths
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
class_indices_path = f"{working_dir}/class_indices.json"

# Set Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Plant Disease Classifier", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Load the model and indices (with error handling)
try:
    # Load the pre-trained model (Disabling Keras/TF warnings for a cleaner UI)
    with st.spinner("Loading model..."):
        # Setting suppress_tf_warnings to true to avoid console clutter in Streamlit
        tf.get_logger().setLevel('ERROR') 
        # We load the model outside the cached functions since it's a huge, constant object
        model = tf.keras.models.load_model(model_path)
    
    # loading the class names
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
except FileNotFoundError:
    st.error("Error: Model or class indices file not found. Ensure the necessary files are in the 'trained_model' directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading the model or indices: {e}")
    st.stop()


# --- Custom CSS for Neat UI ---

st.markdown("""
<style>
    /* Main container styling for a clean, professional look */
    .stApp {
        background-color: #f7f9fc; /* Very light gray background */
        color: #1c1c1e;
    }
    
    /* Enhance the title */
    .stApp h1 {
        color: #28a745; /* Green theme color */
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 10px;
        margin-bottom: 20px;
        font-weight: 700;
    }

    /* Style for the main content blocks (cards) */
    .css-1r6j0o6, .css-1dp5ss7, .css-1v3fvcr {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); /* Soft box shadow for lift */
        transition: box-shadow 0.3s ease-in-out;
    }
    .css-1r6j0o6:hover, .css-1dp5ss7:hover, .css-1v3fvcr:hover {
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
    }

    /* Custom style for the footer content */
    .footer-content {
        text-align: center;
        margin-top: 40px;
        padding-top: 15px;
        font-size: 0.85em;
        color: #6c757d; /* Muted gray for footer text */
        border-top: 1px solid #e9ecef;
    }
    
    /* Centered image display */
    .css-1nm2q7x img {
        border-radius: 8px;
        border: 2px solid #28a745;
    }
    
    /* Primary button styling */
    .stButton>button {
        background-color: #28a745;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1e7e34;
    }
</style>
""", unsafe_allow_html=True)


# --- Functions (with cache for performance) ---

@st.cache_data(show_spinner=False)
def load_and_preprocess_image(image_file, target_size=(224, 224)):
    """Loads, resizes, and preprocesses an image for model prediction."""
    # Load the image using Pillow from the uploaded file buffer
    img = Image.open(image_file)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array


@st.cache_data(show_spinner=False)
def predict_image_class(_model, image_file, class_indices):
    """
    Predicts the class of the provided image.
    
    The leading underscore on '_model' tells Streamlit's caching mechanism 
    to ignore this large, unhashable object, solving the hashing error.
    """
    preprocessed_img = load_and_preprocess_image(image_file)
    # Use the parameter name without the underscore inside the function body
    predictions = _model.predict(preprocessed_img, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown Class")
    # Also return the confidence score for a neat UI touch
    confidence = predictions[0][predicted_class_index]
    return predicted_class_name, confidence


# --- Streamlit App Layout ---

st.title('🌿 Plant Health Diagnostics')

# Main input container
input_container = st.container()
with input_container:
    st.subheader("Upload Leaf Image for Analysis")
    uploaded_image = st.file_uploader("Upload a photo of a plant leaf (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"])
    st.markdown("---")


# Prediction and Display Area
if uploaded_image is not None:
    try:
        # Re-open the image from the uploaded file buffer
        image = Image.open(uploaded_image)
        
        # Use columns for a side-by-side view
        col_img, col_pred = st.columns([1, 2])

        with col_img:
            st.markdown("#### Preview")
            # Display the image with fixed size and centered
            resized_img = image.resize((250, 250))
            st.image(resized_img, caption="Uploaded Leaf Image", use_column_width=False)
        
        with col_pred:
            st.markdown("#### Classification Status")
            # Create a simple form/button interaction for prediction
            if st.button('🔬 Analyze Image', type="primary", use_container_width=True):
                with st.spinner('Running AI Diagnostics...'):
                    # The file needs to be rewound/re-read for prediction
                    uploaded_image.seek(0)
                    # FIX: Pass the model as the first argument
                    prediction_name, confidence = predict_image_class(model, uploaded_image, class_indices)
                    
                    st.success('✅ Analysis Complete!')
                    st.metric(
                        label="Predicted Disease/Status", 
                        value=prediction_name,
                        delta=f"Confidence: {confidence*100:.2f}%"
                    )
                    
                    if "healthy" in prediction_name.lower():
                        st.balloons()
                        st.info("The plant appears **healthy**! Keep up the good work.")
                    else:
                        st.warning("Potential disease detected. Please refer to plant care guides for treatment.")
            else:
                 st.info("Upload an image and click 'Analyze Image' to start the prediction.")

    except Exception as e:
        # Added a check to re-raise the error if it's NOT the expected caching error, 
        # but since the primary issue is solved, this block mainly catches image processing errors.
        st.error(f"An error occurred processing the image or running the prediction: {e}")


# --- Footer Section ---

# Using the custom CSS class 'footer-content' defined earlier
st.markdown(
    """
    <div class="footer-content">
        <p style="margin: 0;"><strong>Project Team</strong></p>
        <p style="margin: 0;">T.NISHKHA | S.SWATHI | SUCHITRA</p>
        <br>
        <p style="margin: 0;">&copy; Copyright 2025 | AI-Powered Plant Diagnostics</p>
    </div>
    """,
    unsafe_allow_html=True
)
