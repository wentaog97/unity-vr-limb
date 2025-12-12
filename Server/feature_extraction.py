import pandas as pd
import numpy as np
import glob
import os
from scipy.fftpack import fft
import sys


# Set parameters
WINDOW_SIZE = 120  # Number of samples per window (~600 ms at 200 Hz)
OVERLAP = 60       # Overlap between windows (50%)
SAMPLING_RATE = 200  # EMG sampling rate (Hz)


# 1️⃣ Load and process raw EMG data
def load_emg_data(pattern='myo_raw_*_emg.csv'):
    files = glob.glob(pattern, recursive=True)
    if not files:
        print("No EMG files found.")
        return pd.DataFrame()


    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# 2️⃣ Compute Time-Domain Features
def compute_rms(signal):
    return np.sqrt(np.mean(signal**2))


def compute_mav(signal):
    return np.mean(np.abs(signal))


def compute_wl(signal):
    return np.sum(np.abs(np.diff(signal)))


def compute_zc(signal):
    return np.sum(np.diff(np.sign(signal)) != 0)


def compute_ssc(signal):
    return np.sum((signal[1:-1] - signal[:-2]) * (signal[1:-1] - signal[2:]) > 0)


# 3️⃣ Compute Frequency-Domain Features
def compute_fft(signal):
    fft_vals = np.abs(fft(signal))[:len(signal)//2]  # Keep positive frequencies
    return np.mean(fft_vals), np.max(fft_vals)  # Mean & Max FFT Power




# 4️⃣ Process data into feature windows
def extract_features(df):
    feature_cols = [col for col in df.columns if col.startswith('emg')]
    label_col = "pose"


    feature_data = []
   
    for start in range(0, len(df) - WINDOW_SIZE, OVERLAP):
        end = start + WINDOW_SIZE
        segment = df.iloc[start:end]
       
        if len(segment) < WINDOW_SIZE:
            continue
       
        features = []
       
        # Compute features for each EMG channel
        for col in feature_cols:
            signal = segment[col].values
            features.extend([
                compute_rms(signal),
                compute_mav(signal),
                compute_wl(signal),
                compute_zc(signal),
                compute_ssc(signal),
                *compute_fft(signal),  # Mean & Max FFT Power
               
            ])
       
        # Add label (majority vote of window labels)
        #if segment[label_col].nunique() == 1:
        label = segment[label_col].iloc[0]
        #else:
        #    label = 'unlabeled'


        features.append(label)
        feature_data.append(features)
   
    # Create feature DataFrame
    feature_names = []
    for col in feature_cols:
        feature_names.extend([
            f"{col}_RMS", f"{col}_MAV", f"{col}_WL", f"{col}_ZC", f"{col}_SSC",
            f"{col}_FFT_Mean", f"{col}_FFT_Max"
        ])
    feature_names.append("pose")  # Add label column
   
    return pd.DataFrame(feature_data, columns=feature_names)


# 5️⃣ Save processed data
def main():
    if len(sys.argv)==3:
        input_csv=sys.argv[1]
        output_csv=sys.argv[2]
        if not os.path.exists(input_csv):
            print(f"Input file not found: {input_csv}")
            sys.exit(1)
        raw_data = pd.read_csv(input_csv)
    else:
        raw_data = load_emg_data(pattern='**/*_emg.csv')
        output_csv = "myo_emg_features.csv"


    if raw_data.empty:
        print("No data available. Exiting.")
        sys.exit(1)


    feature_df = extract_features(raw_data)
    feature_df.to_csv(output_csv, index=False)
    print(f"Feature extraction completed. Saved to {output_csv}")


# ==================== REAL-TIME INFERENCE FUNCTION ====================


def extract_features_from_window(window_data):
    """
    Extract features from a single window of EMG data for real-time inference.
   
    Args:
        window_data: numpy array of shape (window_size, n_channels) containing EMG samples
       
    Returns:
        features: 1D numpy array of extracted features
    """
    window_size, n_channels = window_data.shape
    features = []
   
    # Process each EMG channel
    for ch in range(n_channels):
        signal = window_data[:, ch].astype(float)
       
        # Time-domain features (5 per channel)
        features.extend([
            compute_rms(signal),
            compute_mav(signal),
            compute_wl(signal),
            compute_zc(signal),
            compute_ssc(signal),
        ])
       
        # Frequency-domain features (2 per channel)
        fft_mean, fft_max = compute_fft(signal)
        features.extend([fft_mean, fft_max])
   
    return np.array(features)




if __name__ == "__main__":
    main()




