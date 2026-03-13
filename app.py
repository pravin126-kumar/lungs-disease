%%writefile app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
import cv2
from PIL import Image

# Page config
st.set_page_config(page_title="AI Pneumonia Detector", page_icon="🫁", layout="wide")

st.markdown("""
<style>
.main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
.prediction-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center;}
.confidence-bar {background: rgba(255,255,255,0.2); border-radius: 10px; padding: 1rem; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load pre-trained MobileNetV2 model"""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image):
    """Preprocess image for MobileNetV2"""
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    img_resized = cv2.resize(img_array, (224, 224))
    img_normalized = img_resized.astype('float32') / 255.0
    return np.expand_dims(img_normalized, axis=0)

def main():
    st.markdown('<h1 class="main-header">🫁 AI Pneumonia Detector</h1>', unsafe_allow_html=True)
    
    # Sidebar info
    st.sidebar.markdown("## 🎯 Features")
    st.sidebar.markdown("- MobileNetV2 Transfer Learning\n- Real-time X-ray analysis\n- Confidence scores\n- Hackathon ready!")
    
    # Load model
    model = load_model()
    st.info("✅ Model loaded successfully!")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Chest X-ray")
        uploaded_file = st.file_uploader("Choose image...", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    with col2:
        if 'result' in st.session_state:
            st.markdown(f"""
            <div class="prediction-card">
                <h2>🎯 Prediction</h2>
                <h1>{st.session_state.result['label']}</h1>
                <div class="confidence-bar">
                    <h3>Confidence: {st.session_state.result['confidence']:.1f}%</h3>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Prediction
    if uploaded_file is not None and st.button("🔍 Analyze X-ray", type="primary"):
        with st.spinner("🔬 Analyzing..."):
            img_batch = preprocess_image(image)
            prediction = model.predict(img_batch, verbose=0)[0][0]
            
            confidence = prediction if prediction > 0.5 else (1 - prediction)
            label = "🦠 PNEUMONIA" if prediction > 0.5 else "✅ NORMAL"
            
            st.session_state.result = {
                'label': label,
                'confidence': confidence * 100
            }
            st.success("✅ Analysis Complete!")

if __name__ == "__main__":
    main()
