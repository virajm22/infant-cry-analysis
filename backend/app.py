from flask import Flask, render_template, request, jsonify
import os
import librosa
import numpy as np
import pickle
import soundfile as sf
import noisereduce as nr
from scipy.fftpack import dct

#path to templates folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "templates"))
MODEL_PATH = os.path.join(BASE_DIR, "Forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Upload folder setup
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model and scaler
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Class labels
class_labels = {
    0: "belly_pain",
    1: "burping",
    2: "discomfort",
    3: "hungry",
    4: "tired"
}

# MFCC Extraction Function
def extract_mfcc(audio_file_path, fixed_length=100):
    y, sr = librosa.load(audio_file_path, sr=None)
    emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])
    frame_length = int(round(0.025 * sr))
    frame_step = int(round(0.01 * sr))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil((np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros(pad_signal_length - signal_length)
    pad_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    mag_frames = np.absolute(np.fft.rfft(frames, 512))
    pow_frames = (1.0 / 512) * (mag_frames ** 2)
    fbank = np.zeros((40, int(np.floor(512 / 2 + 1))))
    mel_points = np.linspace(0, 2595 * np.log10(1 + (sr / 2) / 700), 42)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin = np.floor((512 + 1) * hz_points / sr)
    for m in range(1, 41):
        f_m_minus, f_m, f_m_plus = int(bin[m - 1]), int(bin[m]), int(bin[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(12 + 1)]
    if mfcc.shape[0] < fixed_length:
        pad_width = fixed_length - mfcc.shape[0]
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:fixed_length]
    return mfcc

# Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['audio_data']
    filepath = os.path.join(UPLOAD_FOLDER, "temp.wav")
    file.save(filepath)

    # Read and denoise
    try:
        y, sr = sf.read(filepath)
        if y.dtype != 'float32':
            y = y.astype('float32')
    except Exception as e:
        print("soundfile failed, trying librosa:", e)
        try:
            y, sr = librosa.load(filepath, sr=None)
        except Exception as e2:
            print("librosa also failed:", e2)
            return jsonify({'error': 'Unsupported audio format'}), 400

    y_denoised = nr.reduce_noise(y=y, sr=sr)
    y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=20)
    cleaned_path = os.path.join(UPLOAD_FOLDER, "cleaned_temp.wav")
    sf.write(cleaned_path, y_trimmed, sr)

    # Feature extraction
    mfcc = extract_mfcc(cleaned_path)
    features_flatten = mfcc.flatten()
    features_flatten = np.tile(features_flatten, 1200 // len(features_flatten) + 1)[:1200]
    features_scaled = scaler.transform([features_flatten])
    probs = model.predict_proba(features_scaled)[0]
    results = {class_labels[i]: round(float(probs[i]) * 100, 2) for i in range(len(probs))}
    top_prediction = class_labels[np.argmax(probs)]

    return jsonify({'prediction': top_prediction, 'probabilities': results})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

