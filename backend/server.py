from flask import Flask, request, jsonify
import os
import numpy as np
import librosa
import noisereduce as nr
import pickle
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the pre-trained model
with open("Forest_model.pkl", "rb") as file:
    forest_model = pickle.load(file)

# Define the class labels
class_labels = {
    0: "belly_pain",
    1: "burping",
    2: "discomfort",
    3: "hungry",
    4: "tired"
}

def extract_mfcc(audio_file_path, sample_rate=44100, fixed_length=100):
    y, sr = librosa.load(audio_file_path, sr=sample_rate)
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=20)
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)

    if mfcc.shape[1] < fixed_length:
        pad_width = fixed_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :fixed_length]

    return mfcc.flatten()

@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['audio']
    audio_path = os.path.join("temp", "audio.wav")
    audio_file.save(audio_path)

    mfcc_features = extract_mfcc(audio_path)
    extended_features = np.tile(mfcc_features, 1200 // len(mfcc_features) + 1)[:1200]
    extended_features = extended_features.reshape(1, -1)

    prediction = forest_model.predict(extended_features)
    predicted_class = class_labels.get(prediction[0])

    return jsonify({"class": predicted_class})

if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run(debug=True)
