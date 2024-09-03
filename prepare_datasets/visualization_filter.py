import numpy as np
import matplotlib.pyplot as plt
from mne.io import read_raw_edf
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# Function for filtering and preprocessing
def preprocess_eeg(file_path, channel_name):
    # Read EDF file using MNE
    raw = read_raw_edf(file_path, preload=True)
    sampling_rate = raw.info['sfreq']  # Get the sampling frequency
    
    # Select only the desired channel using the new recommended method
    raw.pick([channel_name])

    # Skip applying Common Average Reference (CAR) for a single channel
    # Check if there's more than one channel before applying CAR
    if len(raw.ch_names) > 1:
        raw.set_eeg_reference('average', projection=True)
        raw.apply_proj()

    # High-pass filter the data (e.g., 1.0 Hz)
    raw.filter(1.0, None)
    
    # **Skip ICA for single-channel data**: ICA is not applicable in this scenario.

    # Band-pass filter (0.5–40 Hz)
    raw.filter(0.5, 40.0)
    
    # Extract data from the selected channel
    data = raw.get_data(picks=channel_name)[0]  # Pick specific channel, e.g., "EEG Fpz-Cz"
    
    return data, sampling_rate

# Function to visualize the filtered signal
def plot_filtered_signal(data, sampling_rate, title="Filtered EEG Signal", channel_name="EEG Fpz-Cz"):
    plot_raw_signal(data, sampling_rate, title, channel_name)
    plot_fft(data, sampling_rate, title=f"FFT of {title}", channel_name=channel_name)
    identify_artifacts(data, sampling_rate)
    signal_statistics(data)

# Main function to read, process, and visualize
def visualize_filtered_eeg(file_path, channel_name):
    data, sampling_rate = preprocess_eeg(file_path, channel_name)
    
    # Visualize the signal after filtering
    plot_filtered_signal(data, sampling_rate, title="Filtered EEG Signal", channel_name=channel_name)
    
    # Calculate SNR (using simulated noise)
    noise = data * 0.1  # Assume noise as 10% of the signal amplitude
    snr = calculate_snr(data, noise)
    print(f'Signal-to-Noise Ratio (SNR): {snr:.2f} dB')

# Example usage
file_path = '/content/drive/MyDrive/TugasAkhir/Dataset/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001E0-PSG.edf'  # Replace with the path to your EDF file
channel_name = "EEG Fpz-Cz"
visualize_filtered_eeg(file_path, channel_name)
