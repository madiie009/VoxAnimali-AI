import customtkinter as ctk
import sounddevice as sd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import requests
import threading
import librosa
import time
import os
from dotenv import load_dotenv

load_dotenv()  
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

try:
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    classifier = tf.keras.models.load_model('animal_finetuned_model_v2.h5')
except Exception as e:
    print(f"Error loading models: {e}")

class VoxAnimaliAdvanced(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("VoxAnimali AI v2.0 - Advanced Neural Translator")
        self.geometry("950x750")
        ctk.set_appearance_mode("dark")
        
        # Theme Colors
        self.primary_blue = "#1f6aa5"
        self.accent_green = "#2ecc71"
        self.bg_dark = "#121212"
        self.card_bg = "#1e1e1e"

        self.configure(fg_color=self.bg_dark)
        self.grid_columnconfigure(0, weight=1)

        # 1. Top Navigation Bar
        self.top_bar = ctk.CTkFrame(self, height=80, fg_color=self.card_bg, corner_radius=0)
        self.top_bar.pack(fill="x")
        
        self.title_label = ctk.CTkLabel(self.top_bar, text="VOX ANIMALI AI", 
                                       font=ctk.CTkFont(family="Rajdhani", size=32, weight="bold"),
                                       text_color=self.primary_blue)
        self.title_label.pack(pady=10)

        # 2. Results Dashboard
        self.dashboard = ctk.CTkFrame(self, fg_color="transparent")
        self.dashboard.pack(pady=30, padx=40, fill="x")

        # Species Display Card
        self.species_card = self.create_glass_card(self.dashboard, "SPECIES DETECTED", "Waiting...", self.primary_blue)
        self.species_card.pack(side="left", padx=15, expand=True, fill="both")

        # Mood Display Card
        self.mood_card = self.create_glass_card(self.dashboard, "EMOTIONAL STATE", "Waiting...", self.accent_green)
        self.mood_card.pack(side="right", padx=15, expand=True, fill="both")

        # 3. Status & Progress
        self.status_label = ctk.CTkLabel(self, text="System Ready - Tap to Translate", font=("Arial", 12), text_color="gray")
        self.status_label.pack(pady=(10, 0))
        
        self.progress = ctk.CTkProgressBar(self, width=700, height=4, progress_color=self.primary_blue)
        self.progress.set(0)
        self.progress.pack(pady=10)

        # 4. AI Interpretation Area
        self.output_container = ctk.CTkFrame(self, fg_color=self.card_bg, corner_radius=25, border_width=1, border_color="#333333")
        self.output_container.pack(pady=10, padx=60, fill="both", expand=True)
        
        self.output_title = ctk.CTkLabel(self.output_container, text="NEURAL INTERPRETATION", font=("Orbitron", 10), text_color="#555555")
        self.output_title.pack(pady=(15, 0))

        self.text_output = ctk.CTkTextbox(self.output_container, width=700, height=150, 
                                          font=("Inter", 20), fg_color="transparent", 
                                          text_color="#E0E0E0", wrap="word", border_width=0)
        self.text_output.pack(pady=15, padx=30)

        # 5. Main Action Button
        self.listen_btn = ctk.CTkButton(self, text="🎙 START NEURAL CAPTURE", 
                                        command=self.start_thread, 
                                        height=65, width=400, corner_radius=32,
                                        font=("Arial", 20, "bold"),
                                        fg_color=self.primary_blue,
                                        hover_color="#144870")
        self.listen_btn.pack(pady=40)

    def create_glass_card(self, master, title, val, color):
        card = ctk.CTkFrame(master, height=180, corner_radius=25, fg_color=self.card_bg, border_width=2, border_color=color)
        card.pack_propagate(False)
        
        ctk.CTkLabel(card, text=title, font=("Arial", 11, "bold"), text_color="gray").pack(pady=(25, 5))
        label = ctk.CTkLabel(card, text=val, font=("Arial", 32, "bold"), text_color=color)
        label.pack(pady=10)
        
        if "SPECIES" in title: self.species_val_label = label
        else: self.mood_val_label = label
        return card

    def start_thread(self):
        # UI freeze se bachne ke liye threading
        threading.Thread(target=self.run_logic, daemon=True).start()

    def run_logic(self):
        try:
            self.listen_btn.configure(state="disabled", text="👂 LISTENING...")
            self.progress.set(0.1)
            self.status_label.configure(text="Capturing acoustic signatures...", text_color=self.primary_blue)
            
            # --- 1. Audio Recording ---
            fs, duration = 16000, 4
            rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            audio_raw = rec.flatten()

            self.progress.set(0.4)
            self.status_label.configure(text="Cleaning and Normalizing Audio...")
            
            audio_trimmed, _ = librosa.effects.trim(audio_raw, top_db=20)
            audio_data = librosa.util.normalize(audio_trimmed) if len(audio_trimmed) > 0 else audio_raw

            self.status_label.configure(text="Extracting Neural Features...")
            _, embeddings, _ = yamnet(audio_data)
            features = np.mean(embeddings, axis=0).reshape(1, -1)
            
            prediction = classifier.predict(features)
            confidence = np.max(prediction)
            class_idx = np.argmax(prediction)
            
            if confidence < 0.60:
                self.show_result("N/A", "QUIET", "Sound was too faint. Please bring the microphone closer to the animal.")
                return

            species_name = "🐶 DOG" if class_idx == 0 else "🐱 CAT"
            rms = np.sqrt(np.mean(audio_data**2))
            emotion_name = "💢 ALERT" if rms > 0.07 else "💖 HAPPY"

            # --- 4. LLM Translation with Timeout Control ---
            self.status_label.configure(text="Synthesizing Human Speech...", text_color=self.accent_green)
            self.progress.set(0.7)
            
            try:
                prompt = f"<|system|>\nYou are a professional animal translator. Translate the {species_name}'s feeling into a short human thought.</s>\n<|user|>\nA {species_name} is sounding {emotion_name}.</s>\n<|assistant|>\n"
                
                payload = {
                    "inputs": prompt, 
                    "parameters": {"max_new_tokens": 45, "temperature": 0.7},
                    "options": {"wait_for_model": True} # Wait for model to load
                }
                
                # Added timeout of 20 seconds
                response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=20)
                
                if response.status_code == 200:
                    human_text = response.json()[0]['generated_text'].split("<|assistant|>\n")[-1].strip()
                else:
                    human_text = "The AI translator is a bit busy. But I can tell they are feeling " + emotion_name.lower() + "!"
            
            except requests.exceptions.Timeout:
                human_text = "Neural link timed out. The connection is slow, but the signal is clear!"
            except Exception as e:
                human_text = "Translation failed, but the animal seems to be in a " + emotion_name.lower() + " mood."

            self.show_result(species_name, emotion_name, human_text)

        except Exception as e:
            print(f"Main Loop Error: {e}")
            self.show_result("ERROR", "N/A", "An unexpected error occurred. Please restart the app.")

    def show_result(self, sp, em, txt):
        self.species_val_label.configure(text=sp)
        self.mood_val_label.configure(text=em)
        self.text_output.delete("1.0", "end")
        self.text_output.insert("1.0", f"\"{txt}\"")
        self.progress.set(1.0)
        self.status_label.configure(text="Analysis Complete", text_color="gray")
        self.listen_btn.configure(state="normal", text="🎙 START NEURAL CAPTURE")

if __name__ == "__main__":
    app = VoxAnimaliAdvanced()
    app.mainloop()