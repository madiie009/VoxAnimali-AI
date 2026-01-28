import librosa
import numpy as np
import soundfile as sf
import os

def augment_audio(file_path, output_folder, file_name):
    y, sr = librosa.load(file_path)
    
    # 1. Original save karein
    sf.write(f"{output_folder}/orig_{file_name}", y, sr)
    
    # 2. Pitch Shift (High pitch)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    sf.write(f"{output_folder}/pitch_{file_name}", y_pitch, sr)
    
    # 3. Time Stretch (Slow speed)
    y_slow = librosa.effects.time_stretch(y, rate=0.8)
    sf.write(f"{output_folder}/slow_{file_name}", y_slow, sr)
    
    # 4. Add Noise
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise
    sf.write(f"{output_folder}/noise_{file_name}", y_noise, sr)

# Folders create karein
for label in ['dog', 'cat']:
    input_dir = f'data/{label}'
    output_dir = f'augmented_data/{label}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Augmenting {label} files...")
    for file in os.listdir(input_dir):
        augment_audio(f"{input_dir}/{file}", output_dir, file)

print("Data Augmentation Complete! Check 'augmented_data' folder.")