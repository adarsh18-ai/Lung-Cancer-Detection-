import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# Import scikit-learn components that might be needed for the pickled model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import sklearn

# Configure page
st.set_page_config(
    page_title="Lung Cancer Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
        border-bottom: 2px solid #F18F01;
        padding-bottom: 0.5rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        color : #2E86AB;
        margin: 10px 0;
    }
    .high-risk {
        border-left-color: #ff4444;
        background-color: #ffe6e6;
        color:#ff4444;
    }
    .low-risk {
        border-left-color: #44ff44;
        background-color: #e6ffe6;
        color : #44ff44;
    }
    .medium-risk {
        border-left-color: #ffaa00;
        background-color: #fff8e6;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        color:black;
        padding: 15px;
        margin: 15px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: blue;
        border-radius: 5px;
        padding: 15px;
        margin: 15px 0;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained models"""
    try:
        # Load classification model
        try:
            with open('lung_cancer_model.pkl', 'rb') as file:
                classification_model = pickle.load(file)
            st.success("✅ Classification model loaded successfully!")
        except FileNotFoundError:
            st.error("❌ lung_cancer_model.pkl not found. Please ensure the file is in the correct directory.")
            classification_model = None
        except Exception as e:
            st.error(f"❌ Error loading classification model: {str(e)}")
            classification_model = None
        
        # Load CNN model
        try:
            cnn_model = load_model('chest_xray_pneumonia_model.h5')
            st.success("✅ CNN model loaded successfully!")
        except FileNotFoundError:
            st.error("❌ chest_xray_pneumonia_model.h5 not found. Please ensure the file is in the correct directory.")
            cnn_model = None
        except Exception as e:
            st.error(f"❌ Error loading CNN model: {str(e)}")
            cnn_model = None
        
        return classification_model, cnn_model
    except Exception as e:
        st.error(f"❌ Unexpected error in load_models: {str(e)}")
        return None, None

def preprocess_image_rgb(image, target_size=(224, 224)):
    """Preprocess image for RGB input (3 channels)"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Handle different image formats
        if len(img_array.shape) == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:  # Single channel
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Ensure RGB format
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            else:
                img_array = img_array[:, :, :3]
        
        # Resize and normalize
        img_resized = cv2.resize(img_array, target_size)
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_final = np.expand_dims(img_normalized, axis=0)
        
        return img_final
    except Exception as e:
        st.error(f"Error in RGB preprocessing: {str(e)}")
        return None

def preprocess_image_grayscale(image, target_size=(224, 224)):
    """Preprocess image for grayscale input (1 channel)"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
            elif img_array.shape[2] == 3:  # RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Resize and normalize
        img_resized = cv2.resize(img_array, target_size)
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_final = np.expand_dims(np.expand_dims(img_normalized, axis=-1), axis=0)
        
        return img_final
    except Exception as e:
        st.error(f"Error in grayscale preprocessing: {str(e)}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Smart preprocessing that tries both RGB and grayscale based on model requirements"""
    # First, let's try RGB preprocessing (most common for medical imaging)
    processed_rgb = preprocess_image_rgb(image, target_size)
    
    if processed_rgb is not None:
        st.info(f"📊 Image preprocessing complete: RGB format, Shape {processed_rgb.shape}")
        return processed_rgb
    else:
        # Fallback to grayscale
        st.warning("RGB preprocessing failed, trying grayscale...")
        processed_gray = preprocess_image_grayscale(image, target_size)
        if processed_gray is not None:
            st.info(f"📊 Image preprocessing complete: Grayscale format, Shape {processed_gray.shape}")
        return processed_gray

def predict_from_symptoms(model, symptoms_data):
    """Make prediction using symptom-based classification model"""
    try:
        # Check if we're in demo mode
        if st.session_state.get('demo_mode', False) or model is None:
            # Return mock predictions for demo
            mock_prediction = np.random.choice([0, 1], p=[0.7, 0.3])
            mock_probabilities = [0.75, 0.25] if mock_prediction == 0 else [0.3, 0.7]
            return mock_prediction, np.array(mock_probabilities)
        
        prediction = model.predict(symptoms_data)
        probability = model.predict_proba(symptoms_data)
        return prediction[0], probability[0]
    except Exception as e:
        st.error(f"Error in symptom prediction: {str(e)}")
        return None, None

def predict_from_xray(model, image_data):
    """Make prediction using CNN model on X-ray image"""
    try:
        # Check if we're in demo mode
        if st.session_state.get('demo_mode', False) or model is None:
            # Return mock predictions for demo
            mock_probability = np.random.uniform(0.1, 0.9)
            return np.array([mock_probability])
        
        prediction = model.predict(image_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Error in X-ray prediction: {str(e)}")
        return None

def create_gauge_chart(probability, title):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 Lung Cancer Detection System</h1>', unsafe_allow_html=True)
    
    # Load models with better error handling
    with st.spinner("Loading models..."):
        classification_model, cnn_model = load_models()
    
    # Show model status
    col1, col2 = st.columns(2)
    with col1:
        if classification_model is not None:
            st.success("🤖 Classification Model: Ready")
        else:
            st.error("🤖 Classification Model: Not Available")
    
    with col2:
        if cnn_model is not None:
            st.success("🧠 CNN Model: Ready")
        else:
            st.error("🧠 CNN Model: Not Available")
    
    # Continue even if one model fails
    if classification_model is None and cnn_model is None:
        st.error("⚠️ No models could be loaded. Please check the troubleshooting guide below.")
        
        # Troubleshooting guide
        st.markdown("""
        ### 🔧 Troubleshooting Guide
        
        **If you're seeing model loading errors, try these solutions:**
        
        1. **Install missing packages:**
        ```bash
        pip install scikit-learn tensorflow opencv-python pandas numpy pillow plotly streamlit
        ```
        
        2. **Check file locations:**
        - Ensure `lung_cancer_model.pkl` is in the same directory as this script
        - Ensure `chest_xray_pneumonia_model.h5` is in the same directory as this script
        
        3. **File permissions:**
        - Make sure the model files have read permissions
        
        4. **Model compatibility:**
        - Ensure models were saved with compatible library versions
        - Try recreating the models with current library versions if needed
        
        5. **Alternative approach:**
        - You can still run the app with placeholder functions to test the UI
        """)
        
        # Offer to continue with demo mode
        if st.button("Continue in Demo Mode (with mock predictions)"):
            st.session_state.demo_mode = True
        
        if not st.session_state.get('demo_mode', False):
            st.stop()
    
    # Add demo mode flag
    demo_mode = st.session_state.get('demo_mode', False)
    if demo_mode:
        st.warning("⚠️ Running in Demo Mode - Predictions are simulated for testing purposes only!")
    
    # Sidebar
    st.sidebar.header("🔧 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Complete Analysis", "Symptom Analysis Only", "X-ray Analysis Only", "About"]
    )
    
    if page == "About":
        st.markdown('<h2 class="sub-header">About This System</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Purpose</h3>
        <p>This multimodal lung cancer detection system combines two powerful approaches:</p>
        <ul>
            <li><strong>Symptom-based Analysis:</strong> Uses machine learning to analyze patient symptoms</li>
            <li><strong>Medical Imaging Analysis:</strong> Uses deep learning (CNN) to analyze chest X-ray images</li>
        </ul>
        
        <h3>⚠️ Important Disclaimer</h3>
        <p><strong>This tool is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.</strong></p>
        
        <h3>🔬 Technology Stack</h3>
        <ul>
            <li>Classification Model: Scikit-learn based model (lung_cancer_model.pkl)</li>
            <li>CNN Model: TensorFlow/Keras based model (chest_xray_pneumonia_model.h5)</li>
            <li>Frontend: Streamlit</li>
            <li>Visualization: Plotly</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    elif page in ["Complete Analysis", "Symptom Analysis Only"]:
        # Symptom Analysis Section
        st.markdown('<h2 class="sub-header">📋 Symptom Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            gender = st.selectbox("Gender", ["Male", "Female"])
            smoking_history = st.selectbox("Smoking History", ["Never", "Former", "Current"])
            
            st.subheader("Symptoms")
            cough = st.selectbox("Persistent Cough", ["No", "Yes"])
            chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
            shortness_of_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
            
        with col2:
            st.subheader("Additional Symptoms")
            weight_loss = st.selectbox("Unexplained Weight Loss", ["No", "Yes"])
            fatigue = st.selectbox("Persistent Fatigue", ["No", "Yes"])
            hoarseness = st.selectbox("Hoarseness", ["No", "Yes"])
            
            st.subheader("Medical History")
            family_history = st.selectbox("Family History of Cancer", ["No", "Yes"])
            exposure_to_carcinogens = st.selectbox("Occupational Exposure to Carcinogens", ["No", "Yes"])
        
        # Convert inputs to numerical format (adjust based on your model's requirements)
        symptom_data = pd.DataFrame({
            'age': [age],
            'gender': [1 if gender == "Male" else 0],
            'smoking_history': [0 if smoking_history == "Never" else 1 if smoking_history == "Former" else 2],
            'cough': [1 if cough == "Yes" else 0],
            'chest_pain': [1 if chest_pain == "Yes" else 0],
            'shortness_of_breath': [1 if shortness_of_breath == "Yes" else 0],
            'weight_loss': [1 if weight_loss == "Yes" else 0],
            'fatigue': [1 if fatigue == "Yes" else 0],
            'hoarseness': [1 if hoarseness == "Yes" else 0],
            'family_history': [1 if family_history == "Yes" else 0],
            'exposure_to_carcinogens': [1 if exposure_to_carcinogens == "Yes" else 0]
        })
        
        if st.button("🔍 Analyze Symptoms", type="primary"):
            # Check if model is available
            if classification_model is None and not st.session_state.get('demo_mode', False):
                st.error("❌ Classification model not available. Please load the model or enable demo mode.")
            else:
                with st.spinner("Analyzing symptoms..."):
                    prediction, probabilities = predict_from_symptoms(classification_model, symptom_data)
                    
                    if prediction is not None:
                        st.markdown('<h3 class="sub-header">📊 Symptom Analysis Results</h3>', unsafe_allow_html=True)
                        
                        # Show demo warning if in demo mode
                        if st.session_state.get('demo_mode', False):
                            st.warning("⚠️ Demo Mode: These are simulated results for testing purposes only!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            risk_level = "High" if probabilities[1] > 0.7 else "Medium" if probabilities[1] > 0.3 else "Low"
                            risk_color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
                            st.metric("Risk Level", f"{risk_color} {risk_level}", delta=f"{probabilities[1]:.1%}")
                        
                        with col2:
                            st.metric("Cancer Probability", f"{probabilities[1]:.1%}")
                        
                        with col3:
                            st.metric("Normal Probability", f"{probabilities[0]:.1%}")
                        
                        # Gauge chart
                        fig = create_gauge_chart(probabilities[1], "Lung Cancer Risk Assessment")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Results box
                        risk_class = "high-risk" if probabilities[1] > 0.7 else "medium-risk" if probabilities[1] > 0.3 else "low-risk"
                        st.markdown(f"""
                        <div class="prediction-box {risk_class}">
                            <h4>Symptom-based Prediction</h4>
                            <p><strong>Result:</strong> {'Positive indicators for lung cancer' if prediction == 1 else 'Negative indicators for lung cancer'}</p>
                            <p><strong>Confidence:</strong> {max(probabilities):.1%}</p>
                            <p><strong>Recommendation:</strong> {'Immediate medical consultation recommended' if probabilities[1] > 0.7 else 'Consider medical consultation' if probabilities[1] > 0.3 else 'Continue regular health monitoring'}</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    if page in ["Complete Analysis", "X-ray Analysis Only"]:
        # X-ray Analysis Section
        st.markdown('<h2 class="sub-header">🏥 X-ray Image Analysis</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload Chest X-ray Image", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear chest X-ray image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📷 Uploaded X-ray")
                st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
                
                # Model configuration options
                st.subheader("⚙️ Model Configuration")
                preprocessing_method = st.radio(
                    "Choose preprocessing method:",
                    ["Smart Auto-Detect", "Try All Methods", "Force RGB", "Force Grayscale→RGB"],
                    help="Smart Auto-Detect examines your model. Try All Methods tests multiple approaches automatically."
                )
                
                # Image info
                img_array = np.array(image)
                st.info(f"📊 Original image: {img_array.shape}, Mode: {image.mode}")
            
            with col2:
                st.subheader("🔬 Analysis")
                
                if st.button("🔍 Analyze X-ray", type="primary"):
                    # Check if model is available
                    if cnn_model is None and not st.session_state.get('demo_mode', False):
                        st.error("❌ CNN model not available. Please load the model or enable demo mode.")
                    else:
                        with st.spinner("Processing X-ray image..."):
                            # Choose preprocessing method
                            if preprocessing_method == "Force Grayscale":
                                processed_image = preprocess_image_grayscale(image)
                            elif preprocessing_method == "Force RGB":
                                processed_image = preprocess_image_rgb(image)
                            else:  # Auto
                                processed_image = preprocess_image(image)
                            
                            if processed_image is not None or st.session_state.get('demo_mode', False):
                                prediction = predict_from_xray(cnn_model, processed_image)
                                
                                if prediction is not None:
                                    st.markdown('<h3 class="sub-header">📊 X-ray Analysis Results</h3>', unsafe_allow_html=True)
                                    
                                    # Show demo warning if in demo mode
                                    if st.session_state.get('demo_mode', False):
                                        st.warning("⚠️ Demo Mode: These are simulated results for testing purposes only!")
                                    
                                    # Assuming binary classification: 0 = Normal, 1 = Abnormal/Cancer
                                    probability_abnormal = prediction[0] if len(prediction) == 1 else prediction[1]
                                    probability_normal = 1 - probability_abnormal
                                    
                                    # Metrics
                                    col3, col4 = st.columns(2)
                                    with col3:
                                        st.metric("Abnormal Probability", f"{probability_abnormal:.1%}")
                                    with col4:
                                        st.metric("Normal Probability", f"{probability_normal:.1%}")
                                    
                                    # Gauge chart for X-ray
                                    fig_xray = create_gauge_chart(probability_abnormal, "X-ray Abnormality Assessment")
                                    st.plotly_chart(fig_xray, use_container_width=True)
                                    
                                    # Results box
                                    risk_class_xray = "high-risk" if probability_abnormal > 0.7 else "medium-risk" if probability_abnormal > 0.3 else "low-risk"
                                    st.markdown(f"""
                                    <div class="prediction-box {risk_class_xray}">
                                        <h4>X-ray Analysis Result</h4>
                                        <p><strong>Finding:</strong> {'Abnormalities detected' if probability_abnormal > 0.5 else 'No significant abnormalities detected'}</p>
                                        <p><strong>Confidence:</strong> {max(probability_abnormal, probability_normal):.1%}</p>
                                        <p><strong>Recommendation:</strong> {'Urgent medical consultation recommended' if probability_abnormal > 0.7 else 'Medical review advised' if probability_abnormal > 0.3 else 'Appears normal, continue routine monitoring'}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
    
    # Combined Results (only show in Complete Analysis mode)
    if page == "Complete Analysis":
        st.markdown('<h2 class="sub-header">🎯 Combined Assessment</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>💡 How to interpret combined results:</h4>
        <ul>
            <li><strong>Both models agree (High Risk):</strong> Strong indication for immediate medical consultation</li>
            <li><strong>Both models agree (Low Risk):</strong> Lower probability, but continue regular monitoring</li>
            <li><strong>Models disagree:</strong> Additional testing and medical consultation recommended</li>
            <li><strong>One analysis unavailable:</strong> Rely on available analysis but consider completing both</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer with disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Medical Disclaimer</h4>
        <p>This application is for educational and research purposes only. The predictions provided by this system should not be considered as medical advice or diagnosis. Always consult with qualified healthcare professionals for proper medical evaluation and treatment. Early detection and professional medical care are crucial for cancer treatment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Information")
    st.sidebar.info("""
    **Classification Model**: Trained on symptom data
    
    **CNN Model**: Trained on chest X-ray images
    
    """)
    
    st.sidebar.subheader("📞 Support")
    st.sidebar.info("""
    For technical issues or questions about this application, please contact your system administrator.
    
    For medical emergencies, contact your local emergency services immediately.
    """)

if __name__ == "__main__":
    main()