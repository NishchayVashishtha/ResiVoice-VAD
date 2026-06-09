import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import time
import os

# ==========================================
# 1. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="ResiVoice-VAD", page_icon="🎙️", layout="centered")
st.title("🎙️ ResiVoice-VAD Engine")
st.markdown("**Noise-Resilient Child Speech Classification & VAD Pipeline**")
st.write("Upload a noisy audio file (`.wav`) to evaluate child vs. adult voice presence and track inference latency.")

# ==========================================
# 2. LOAD TRAINED PYTORCH MODEL
# ==========================================
class VoiceClassifierCNN(nn.Module):
    def __init__(self):
        super(VoiceClassifierCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.network(x)

@st.cache_resource
def load_model():
    model = VoiceClassifierCNN()
    # Loading the weights you saved earlier
    model.load_state_dict(torch.load("resivoice_model.pth", weights_only=True))
    model.eval()
    return model

classifier_model = load_model()

# ==========================================
# 3. INTERACTIVE UPLOAD & INFERENCE PIPELINE
# ==========================================
uploaded_file = st.file_uploader("Drop an audio file here...", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Run Evaluation Benchmark"):
        with st.spinner("Initializing VAD & Extracting MFCCs..."):
            
            # Temporary save to prevent Librosa memory buffer errors
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            start_time = time.time()
            
            # Feature Extraction Pipeline
            audio, sr = librosa.load(temp_path, sr=16000)
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
            mfccs = librosa.feature.mfcc(y=audio_trimmed, sr=sr, n_mfcc=13)
            mfccs_processed = np.mean(mfccs.T, axis=0)
            
            # Model Inference
            tensor_input = torch.tensor([mfccs_processed], dtype=torch.float32)
            with torch.no_grad():
                output = classifier_model(tensor_input)
                prediction = torch.argmax(output, dim=1).item()
                
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            os.remove(temp_path) # Clean up
            
            # Display Results (Omli metrics format)
            st.success(f"⚡ End-to-End Pipeline Latency: **{latency_ms:.2f} ms**")
            
            if prediction == 1:
                st.info("🧒 **Classification Result:** Child Voice Detected")
            else:
                st.warning("🧑 **Classification Result:** Adult Voice Detected")