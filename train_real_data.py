import os
import time
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("[INFO] Booting up ResiVoice-VAD Real Data Pipeline...")

# ==========================================
# 1. REAL DATA INGESTION & FEATURE ENGINEERING
# ==========================================
def load_real_data(data_path, max_files_per_class=200):
    X, y = [], []
    # Hum 'yes' aur 'no' folders ko do alag voice classes (e.g., Child vs Adult) maan kar train kar rahe hain
    classes = {'yes': 1, 'no': 0} 
    
    print(f"[INFO] Extracting MFCCs from '{data_path}' (Max {max_files_per_class} files per class)...")
    
    for class_name, label in classes.items():
        folder_path = os.path.join(data_path, class_name)
        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder {folder_path} missing!")
            continue
            
        files = os.listdir(folder_path)
        count = 0
        
        for file in files:
            if file.endswith('.wav') and count < max_files_per_class:
                file_path = os.path.join(folder_path, file)
                try:
                    # Load audio aur basic VAD (Noise Trimming)
                    audio, sr = librosa.load(file_path, sr=16000)
                    audio_trimmed, _ = librosa.effects.trim(audio, top_db=25) 
                    
                    # 13 MFCC Channels extract karna
                    mfccs = librosa.feature.mfcc(y=audio_trimmed, sr=sr, n_mfcc=13)
                    mfccs_processed = np.mean(mfccs.T, axis=0)
                    
                    X.append(mfccs_processed)
                    y.append(label)
                    count += 1
                except Exception as e:
                    pass
                    
        print(f" -> Processed {count} files for class '{class_name}'")
        
    return np.array(X, dtype=np.float32), np.array(y)

# Loading data from your extracted folder
DATA_FOLDER = "real_voice_dataset/speech_commands"
X_data, y_data = load_real_data(DATA_FOLDER, max_files_per_class=250)

# Converting to PyTorch Tensors and Splitting (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train), torch.tensor(X_test)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

# ==========================================
# 2. PYTORCH CLASSIFICATION NETWORK
# ==========================================
class VoiceClassifierCNN(nn.Module):
    def __init__(self):
        super(VoiceClassifierCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # Overfitting rokne ke liye
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        return self.network(x)

model = VoiceClassifierCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
print("\n[INFO] Starting Model Training on Real Audio Features...")
epochs = 25
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0:
        print(f" - Epoch {epoch}/{epochs} | Loss Optimization: {loss.item():.4f}")

# ==========================================
# 4. OMLI BENCHMARK ENGINE (Evaluation & Latency)
# ==========================================
print("\n[INFO] Running Production Inference Benchmark...")
model.eval()

start_time = time.time()
with torch.no_grad():
    raw_preds = model(X_test)
    predictions = torch.argmax(raw_preds, dim=1).numpy()
end_time = time.time()

# Calculations
actuals = y_test.numpy()
total_latency_ms = (end_time - start_time) * 1000
per_sample_latency = total_latency_ms / len(X_test)

print("\n" + "="*50)
print("     RESI-VOICE REAL DATA EVALUATION REPORT     ")
print("="*50)
print(f"[LATENCY] Total Batch Inference: {total_latency_ms:.2f} ms")
print(f"[LATENCY] Per-Sample Speed:      {per_sample_latency:.4f} ms/sample")
print("\n[METRICS] Confusion Matrix:")
print(confusion_matrix(actuals, predictions))
print("\n[METRICS] Detailed Classification Report:")
print(classification_report(actuals, predictions, target_names=["Class 0", "Class 1"]))
print("="*50)

# Save the trained model weights
torch.save(model.state_dict(), "resivoice_model.pth")
print("[SUCCESS] Model weights saved to resivoice_model.pth")