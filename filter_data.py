import pandas as pd
import os
import shutil

audio_src = 'audio/audio/44100' 
csv_path = 'esc50.csv'
output_dir = 'data'

if not os.path.exists(audio_src):
    print(f"Error: Folder nahi mila! Path check karein: {audio_src}")
else:
    df = pd.read_csv(csv_path)
    target_animals = ['dog', 'cat']

    for animal in target_animals:
        animal_folder = os.path.join(output_dir, animal)
        os.makedirs(animal_folder, exist_ok=True)
        
        animal_files = df[df['category'] == animal]['filename']
        print(f"{animal} ke liye processing shuru...")

        count = 0
        for file_name in animal_files:
            src_path = os.path.join(audio_src, file_name)
            dst_path = os.path.join(animal_folder, file_name)
            
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                count += 1
        
        print(f"Done! {count} files '{animal}' folder copied.")

