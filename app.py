import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from minio import Minio
import io

# ==========================================
# 1. Configuration
# ==========================================
st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️‍♂️")

# Hardcoded path to your best model
# Ensure this matches the filename you push to GitHub
MODEL_PATH = "models/best_finetuned.pth" 

# ==========================================
# 2. Model Loading (EfficientNet-B4 Only)
# ==========================================
@st.cache_resource
def load_model():
    try:
        # Use CPU for deployment to avoid CUDA errors
        device = torch.device('cpu')
        
        # Initialize strictly EfficientNet-B4
        # weights=None means we are creating a skeleton to load your weights into
        model = models.efficientnet_b4(weights=None)
        
        # Modify the final layer for 2 classes (Real vs Fake)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
        
        # Load your trained weights
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
            return model
        else:
            st.error(f"❌ Model file not found: {MODEL_PATH}")
            st.warning("Please make sure you pushed 'models/best_finetuned.pth' to GitHub.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ==========================================
# 3. MinIO Connection (Auto-Detect)
# ==========================================
def get_minio_client():
    try:
        # Attempt to connect to local Docker MinIO
        # This will succeed on your Mac, but fail on Streamlit Cloud
        client = Minio(
            "localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )
        client.list_buckets() # Test connection
        return client
    except:
        return None

minio_client = get_minio_client()

# ==========================================
# 4. User Interface
# ==========================================
st.title("🕵️‍♂️ Deepfake Detection System")
st.markdown("""
**Model Architecture:** EfficientNet-B4  
**Status:** Ready to analyze
""")

# File Uploader
uploaded_file = st.file_uploader("Upload an image for analysis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Analysis Button
    if st.button("🔍 Analyze Image"):
        if model:
            with st.spinner("Processing with EfficientNet-B4..."):
                # Preprocessing
                transform = transforms.Compose([
                    transforms.Resize((380, 380)), # B4 specific size
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_tensor = transform(image).unsqueeze(0) # Add batch dimension
                
                # Inference
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                
                # Logic (0=Real, 1=Fake) - Adjust if your dataset logic is swapped
                label_map = {0: "REAL", 1: "FAKE"} 
                result_label = label_map[predicted_class.item()]
                score = confidence.item() * 100
                
                # Display Result
                color = "green" if result_label == "REAL" else "red"
                st.markdown(f"### Prediction: <span style='color:{color}'>{result_label}</span>", unsafe_allow_html=True)
                st.progress(int(score))
                st.write(f"Confidence: **{score:.2f}%**")
                
                # Feedback Section
                st.divider()
                if minio_client:
                    # LOCAL MODE
                    st.markdown("#### 🛠️ MLOps Feedback Loop")
                    st.info("System is connected to MinIO. You can report errors.")
                    if st.button("🚩 Report Wrong Prediction"):
                        try:
                            uploaded_file.seek(0)
                            minio_client.put_object(
                                "hard-samples",
                                uploaded_file.name,
                                io.BytesIO(uploaded_file.read()),
                                length=uploaded_file.size,
                                content_type=uploaded_file.type
                            )
                            st.success("Image reported to Data Lake!")
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
                else:
                    # CLOUD DEMO MODE
                    st.warning("⚠️ **Demo Mode Active**: Data collection (MinIO) is disabled in this public demo.")