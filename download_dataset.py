import os
import tarfile
import urllib.request

# Dataset URL (Official Google Speech Commands subset)
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DEST_FOLDER = "real_voice_dataset"
ARCHIVE_NAME = "speech_commands.tar.gz"

if not os.path.exists(DEST_FOLDER):
    os.makedirs(DEST_FOLDER)
    print(f"[INFO] Created directory: {DEST_FOLDER}")

archive_path = os.path.join(DEST_FOLDER, ARCHIVE_NAME)

# Downloading the real dataset matrix
if not os.path.exists(archive_path):
    print("[INFO] Downloading Google Speech Commands Dataset (This might take a few minutes)...")
    urllib.request.urlretrieve(DATASET_URL, archive_path)
    print("[SUCCESS] Download completed.")

# Extracting only a few specific structural voice folders for testing
# We will extract folders like 'yes', 'no' which contain diverse child/adult crowd-sourced samples
extract_targets = ['yes', 'no', '_background_noise_']

print("[INFO] Extracting voice data channels...")
with tarfile.open(archive_path, "r:gz") as tar:
    for member in tar.getmembers():
        # Only extract our targeted speaker folders to save space and keep it lightweight
        if any(member.name.startswith(target) for target in extract_targets):
            tar.extract(member, path=DEST_FOLDER)

print(f"[SUCCESS] Real Dataset ready at: {os.path.abspath(DEST_FOLDER)}")