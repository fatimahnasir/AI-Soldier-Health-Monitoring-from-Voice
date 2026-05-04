import numpy as np
import librosa

def extract_features(file_path, max_duration=5):
    """
    Extract audio features (MFCC, Chroma, Spectral Contrast, ZCR)
    """
    audio, sample_rate = librosa.load(file_path, sr=None)
    
    # Pad or truncate to max_duration
    max_len = int(max_duration * sample_rate)
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
    else:
        audio = audio[:max_len]
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    contrast_mean = np.mean(contrast.T, axis=0)
    
    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr.T, axis=0)
    
    # Combine all features
    features = np.concatenate((mfccs_mean, chroma_mean, contrast_mean, zcr_mean))
    return features
