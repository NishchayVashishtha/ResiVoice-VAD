# 🎙️ ResiVoice-VAD

**Noise-Resilient Child Speech Classification & Voice Activity Detection Pipeline**

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://resivoice-vad.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Network-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Dataset](#-dataset)
- [Performance](#-performance)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

ResiVoice-VAD is a deep learning-powered voice activity detection system specifically designed to classify and differentiate between **child** and **adult** voices in noisy environments. The system leverages MFCC (Mel-Frequency Cepstral Coefficients) feature extraction combined with a custom CNN architecture built in PyTorch for robust, real-time audio classification.

### Why ResiVoice-VAD?

- **Noise Resilient**: Handles audio with background noise effectively
- **Lightweight**: Fast inference with minimal computational overhead
- **Child-Focused**: Optimized for detecting child speech patterns
- **Production-Ready**: Deployed as an interactive Streamlit web application
- **Low Latency**: End-to-end pipeline optimized for speed

---

## ✨ Features

- 🎯 **Binary Voice Classification**: Accurately distinguishes between child and adult voices
- 🔊 **Voice Activity Detection (VAD)**: Automatic audio trimming and silence removal
- 📊 **MFCC Feature Extraction**: 13-coefficient MFCC analysis at 16kHz sampling rate
- 🚀 **Real-Time Inference**: Low-latency prediction pipeline (<100ms typical)
- 🎨 **Interactive Web UI**: User-friendly Streamlit interface for audio upload and analysis
- ⚡ **Performance Metrics**: Real-time latency tracking and classification confidence
- 🔧 **Modular Design**: Easy to extend and integrate into larger systems

---

## 🏗️ Architecture

### Neural Network Architecture

The ResiVoice classifier employs a compact fully-connected neural network:

```
Input Layer (13 MFCC features)
    ↓
Dense Layer (64 neurons) + ReLU
    ↓
Dropout (30%)
    ↓
Dense Layer (32 neurons) + ReLU
    ↓
Output Layer (2 classes)
    ↓
Softmax → [Adult, Child]
```

### Audio Processing Pipeline

```
Raw Audio (.wav)
    ↓
Librosa Load (16kHz resampling)
    ↓
Voice Activity Detection (trim silence)
    ↓
MFCC Extraction (13 coefficients)
    ↓
Feature Normalization (mean aggregation)
    ↓
PyTorch Model Inference
    ↓
Classification Result + Latency Metrics
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Clone the Repository

```bash
git clone https://github.com/NishchayVashishtha/ResiVoice-VAD.git
cd ResiVoice-VAD
```

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install streamlit
pip install librosa
pip install numpy
```

### Download Dataset (Optional)

If you want to train the model from scratch or explore the dataset:

```bash
python download_dataset.py
```

This script downloads the **Google Speech Commands Dataset** and extracts relevant voice samples for training.

---

## 💻 Usage

### Running the Web Application

Launch the Streamlit app locally:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Upload Audio**: Click "Drop an audio file here..." and select a `.wav` file
2. **Preview**: Listen to your uploaded audio using the built-in player
3. **Analyze**: Click "Run Evaluation Benchmark" to process the audio
4. **Results**: View classification results and pipeline latency metrics

### Programmatic Usage

You can also use the model programmatically in your Python scripts:

```python
import torch
import librosa
import numpy as np
from app import VoiceClassifierCNN

# Load the trained model
model = VoiceClassifierCNN()
model.load_state_dict(torch.load("resivoice_model.pth", weights_only=True))
model.eval()

# Load and process audio
audio, sr = librosa.load("your_audio.wav", sr=16000)
audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
mfccs = librosa.feature.mfcc(y=audio_trimmed, sr=sr, n_mfcc=13)
mfccs_processed = np.mean(mfccs.T, axis=0)

# Make prediction
tensor_input = torch.tensor([mfccs_processed], dtype=torch.float32)
with torch.no_grad():
    output = model(tensor_input)
    prediction = torch.argmax(output, dim=1).item()
    
# 0 = Adult, 1 = Child
print(f"Detected: {'Child' if prediction == 1 else 'Adult'} Voice")
```

---

## 🧠 Model Details

### Training Configuration

- **Framework**: PyTorch
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam (recommended)
- **Input Features**: 13 MFCC coefficients
- **Output Classes**: 2 (Adult, Child)
- **Regularization**: 30% Dropout to prevent overfitting

### Model File

The pre-trained model weights are stored in `resivoice_model.pth`. This file contains the learned parameters for the neural network and is loaded automatically when running the application.

### Feature Engineering

- **MFCC (Mel-Frequency Cepstral Coefficients)**: Captures the power spectrum of audio signals
- **13 Coefficients**: Optimal balance between information retention and computational efficiency
- **16kHz Sampling Rate**: Standard rate for speech processing
- **Silence Trimming**: Removes non-speech segments (25dB threshold)
- **Mean Aggregation**: Temporal averaging for fixed-length representation

---

## 📊 Dataset

### Google Speech Commands Dataset

The project uses the [Google Speech Commands Dataset v0.02](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz), a crowd-sourced collection of one-second audio clips containing spoken words.

**Dataset Characteristics**:
- **Source**: Crowd-sourced recordings
- **Duration**: ~1 second per clip
- **Sample Rate**: 16kHz
- **Format**: WAV files
- **Diversity**: Multiple speakers of varying ages and accents
- **Use Cases**: Keyword spotting, voice activity detection, age classification

### Included Categories

The `download_dataset.py` script extracts specific categories:
- `yes/` - Positive affirmations
- `no/` - Negative responses  
- `backward/` - Command word (1,664 samples)
- `bed/` - Object noun (2,014 samples)
- `_background_noise_/` - Ambient noise samples for robustness testing

---

## ⚡ Performance

### Latency Benchmarks

- **Feature Extraction**: ~30-50ms
- **Neural Network Inference**: ~5-15ms
- **Total Pipeline**: **~50-100ms** (typical for 1-second audio)

### System Requirements

- **Minimum**: 2GB RAM, Dual-core CPU
- **Recommended**: 4GB+ RAM, Quad-core CPU
- **GPU**: Optional (CPU inference is fast enough for real-time applications)

### Accuracy Considerations

The model performs best under these conditions:
- Clear speech (minimal background noise)
- 16kHz sampling rate audio
- Voice activity (non-silent segments)
- English language speakers

---

## 📁 Project Structure

```
ResiVoice-VAD/
│
├── app.py                      # Main Streamlit web application
├── download_dataset.py         # Dataset download and extraction script
├── resivoice_model.pth        # Pre-trained PyTorch model weights
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
│
└── real_voice_dataset/        # Downloaded speech dataset (gitignored)
    └── speech_commands/
        ├── backward/          # 1,664 audio samples
        ├── bed/               # 2,014 audio samples
        └── ...                # Additional categories
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the Repository**
2. **Create a Feature Branch**: `git checkout -b feature/YourFeature`
3. **Commit Changes**: `git commit -m 'Add YourFeature'`
4. **Push to Branch**: `git push origin feature/YourFeature`
5. **Open a Pull Request**

### Areas for Improvement

- [ ] Multi-language support
- [ ] Real-time microphone input
- [ ] Model quantization for edge deployment
- [ ] Extended age group classification (infant, teen, adult, senior)
- [ ] Noise reduction preprocessing
- [ ] Batch processing for multiple files
- [ ] RESTful API endpoint

---

## 📜 License

This project is open-source and available for educational and research purposes. Please ensure compliance with the [Google Speech Commands Dataset license](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) when using the training data.

---

## 🔗 Links

- **Live Demo**: [https://resivoice-vad.streamlit.app/](https://resivoice-vad.streamlit.app/)
- **GitHub Repository**: [https://github.com/NishchayVashishtha/ResiVoice-VAD](https://github.com/NishchayVashishtha/ResiVoice-VAD)
- **Developer**: [Nishchay Vashishtha](https://github.com/NishchayVashishtha)

---

## 🙏 Acknowledgments

- **Google Research** for the Speech Commands Dataset
- **PyTorch Team** for the deep learning framework
- **Librosa Developers** for audio processing utilities
- **Streamlit** for the web application framework

---

<div align="center">

**Made with ❤️ for child speech recognition research**

⭐ Star this repository if you find it helpful!

</div>
