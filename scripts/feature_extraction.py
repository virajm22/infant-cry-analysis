import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from tqdm import tqdm
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

def extract_mfcc(
    audio_file_path,
    sample_rate=None,
    pre_emphasis=0.97,
    frame_size=0.025,
    frame_stride=0.01,
    NFFT=512,
    nfilt=40, 
    num_ceps=12,
    fixed_length=100
):
    y, sr = librosa.load(audio_file_path, sr=sample_rate)
    emphasized_signal = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    frame_length, frame_step = frame_size * sr, frame_stride * sr
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    frames *= np.hamming(frame_length)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sr)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]

    if mfcc.shape[0] < fixed_length:
        pad_width = fixed_length - mfcc.shape[0]
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:fixed_length]

    return mfcc

def extract_mfcc_for_folders(data_folder, subfolders, **kwargs):
    features = []
    labels = []

    for label, subfolder in enumerate(subfolders):
        folder_path = os.path.join(data_folder, subfolder)
        files = os.listdir(folder_path)
        for filename in tqdm(files, desc=f"Processing {subfolder}", unit="file"):
            if filename.endswith('.wav'):
                file_path = os.path.join(folder_path, filename)
                mfccs = extract_mfcc(file_path, **kwargs)
                features.append(mfccs.flatten().tolist())
                labels.append(label)

    return features, labels

data_folder = "augmentaudiocry"
subfolders = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

features, labels = extract_mfcc_for_folders(data_folder, subfolders, fixed_length=100)

df = pd.DataFrame(features)
df['label'] = labels
df.to_csv('mfcc_features.csv', index=False)

data = pd.read_csv("mfcc_features.csv")

X = data.drop(columns=['label'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=62)

forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=50,
    criterion="entropy",
    max_depth=32
)
forest_model.fit(X_train, y_train)

forest_predictions = forest_model.predict(X_test)
print('Random Forest Classifier Performance:')
print(classification_report(y_test, forest_predictions))
print("Accuracy:", accuracy_score(y_test, forest_predictions))

conf_matrix = confusion_matrix(y_test, forest_predictions)
print("\nConfusion Matrix:\n", conf_matrix)

with open("Forest_model.pkl", "wb") as file:
    pickle.dump(forest_model, file)
