import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load Pre-trained YAMNet
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def get_yamnet_embeddings(file_path):
    try:
        wav_data, sr = librosa.load(file_path, sr=16000)
        scores, embeddings, spectrogram = yamnet_model(wav_data)
        return np.mean(embeddings, axis=0)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

X, y = [], []
print("Extracting features... Please wait.")

for label_idx, label in enumerate(['dog', 'cat']):
    folder = f"augmented_data/{label}"
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            emb = get_yamnet_embeddings(os.path.join(folder, file))
            if emb is not None:
                X.append(emb)
                y.append(label_idx)

X = np.array(X)
y = np.array(y)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,)), # YAMNet embedding size
    
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dropout(0.5),          
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),          
    

    tf.keras.layers.Dense(2, activation='softmax')
])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)

plt.figure(figsize=(12, 4))

# Accuracy Graph
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()

# Loss Graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

# 5. Confusion Matrix (Detail Reporting)
y_pred = np.argmax(model.predict(X), axis=1)
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Dog', 'Cat'], yticklabels=['Dog', 'Cat'])
plt.title('Confusion Matrix')
plt.show()

model.save('animal_finetuned_model_v2.h5')
print("New Model Saved as 'animal_finetuned_model_v2.h5'")
# Training ke aakhir mein ye lines add karein
final_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100

print(f"\nTraining mukammal!")
print(f"Sikhne ki Accuracy (Train): {final_acc:.2f}%")
print(f"Asli Accuracy (Validation): {final_val_acc:.2f}%")