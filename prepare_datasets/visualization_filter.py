import numpy as np
import matplotlib.pyplot as plt
from mne.io import read_raw_edf
from mne.preprocessing import ICA
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# Fungsi untuk melakukan filtering dan preprocessing
def preprocess_eeg(file_path, channel_name):
    # Membaca file EDF menggunakan MNE
    raw = read_raw_edf(file_path, preload=True)
    sampling_rate = raw.info['sfreq']  # Mendapatkan frekuensi sampling
    
    # Step 1: Apply Common Average Reference (CAR)
    raw.set_eeg_reference('average', projection=True)
    
    # Step 2: Apply Independent Component Analysis (ICA) for artifact removal
    ica = ICA(n_components=20, random_state=97)
    ica.fit(raw)
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ica.exclude = eog_indices
    raw = ica.apply(raw)
    
    # Step 3: Band-pass filter (0.5–40 Hz)
    raw.filter(0.5, 40.0)
    
    # Mengambil data dari channel yang dipilih
    data = raw.get_data(picks=channel_name)[0]  # Ambil channel tertentu, misal "EEG Fpz-Cz"
    
    return data, sampling_rate

# Menggunakan fungsi yang sama untuk visualisasi
def plot_filtered_signal(data, sampling_rate, title="Filtered EEG Signal", channel_name="EEG Fpz-Cz"):
    plot_raw_signal(data, sampling_rate, title, channel_name)
    plot_fft(data, sampling_rate, title=f"FFT of {title}", channel_name=channel_name)
    identify_artifacts(data, sampling_rate)
    signal_statistics(data)

# Fungsi utama untuk membaca, memproses, dan visualisasi
def visualize_filtered_eeg(file_path, channel_name):
    data, sampling_rate = preprocess_eeg(file_path, channel_name)
    
    # Visualisasi sinyal setelah filtering
    plot_filtered_signal(data, sampling_rate, channel_name=channel_name)
    
    # Menghitung SNR (gunakan noise yang disimulasikan)
    noise = data * 0.1  # Asumsi noise sebagai 10% dari amplitudo sinyal
    snr = calculate_snr(data, noise)
    print(f'Signal-to-Noise Ratio (SNR): {snr:.2f} dB')

# Contoh penggunaan
file_path = 'path/to/your/file.edf'  # Ganti dengan path ke file EDF Anda
channel_name = "EEG Fpz-Cz"
visualize_filtered_eeg(file_path, channel_name)
