import librosa
import librosa.display
import numpy as np
import soundfile as sf
import time

def process_audio_pipeline(file_path):
    print(f"[INFO] Ingesting Messy Audio File: {file_path}")
    start_time = time.time()
    
    # 1. Load Audio File (Resampling to standard 16kHz for voice profiling)
    # real-world algorithms prefer 16000Hz for speech processing metrics
    audio, sample_rate = librosa.load(file_path, sr=16000)
    
    # 2. Simple Noise Handling: Trim silent/low-amplitude edges programmatically
    # Strips out non-vocal static chunks before matrix processing
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
    
    # 3. Feature Engineering: Extract 13 standard MFCC channels
    # These represent vocal tract parameters crucial for identifying a Child's voice pitch
    mfccs = librosa.feature.mfcc(y=audio_trimmed, sr=sample_rate, n_mfcc=13)
    
    # Calculate Mean of MFCCs across time axis to get a consistent structural footprint
    mfccs_processed = np.mean(mfccs.T, axis=0)
    
    # 4. Latency and Production Tracking Metrics (Omli Constraint Tracker)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    print(f"[SUCCESS] Feature Extraction Completed in {latency_ms:.2f} ms")
    return mfccs_processed, latency_ms

# --- RUNNING LOCAL TEST LOGIC ---
if __name__ == "__main__":
    # Bhai, testing ke liye folder mein koi bhi ek sample .wav file rakh lena aur uska naam yahan badal dena
    sample_file = "test_child_voice.wav" 
    
    try:
        features, execution_latency = process_audio_pipeline(sample_file)
        print(f"\n--- Omli Pipeline Benchmark Results ---")
        print(f"Extracted Vector Shape (13 MFCC Channels): {features.shape}")
        print(f"Extracted Feature Array Vector:\n{features}")
        print(f"System Latency Met: {execution_latency:.2f} ms")
    except Exception as e:
        print(f"[ERROR] Testing failed. Make sure a valid '{sample_file}' exists in this directory. Details: {e}")