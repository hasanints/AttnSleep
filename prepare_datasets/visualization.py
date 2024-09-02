import numpy as np
import matplotlib.pyplot as plt
from mne.io import read_raw_edf
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# Fungsi untuk memvisualisasikan sinyal mentah dari EDF
def plot_raw_signal(data, sampling_rate, title="Raw EEG Signal", channel_name="EEG Fpz-Cz"):
    time = np.arange(0, len(data)) / sampling_rate
    plt.figure(figsize=(15, 5))
    plt.plot(time, data, color='blue')
    plt.title(f'{title} - {channel_name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (µV)')
    plt.show()

# Fungsi untuk menghitung dan menampilkan FFT
def plot_fft(data, sampling_rate, title="FFT of Raw EEG Signal", channel_name="EEG Fpz-Cz"):
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / sampling_rate)
    
    plt.figure(figsize=(15, 5))
    plt.plot(xf[:N//2], np.abs(yf[:N//2]))
    plt.title(f'{title} - {channel_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

# Fungsi untuk menghitung SNR
def calculate_snr(signal, noise):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Fungsi untuk mengidentifikasi dan menampilkan artefak umum dalam sinyal
def identify_artifacts(data, sampling_rate):
    f, Pxx = welch(data, fs=sampling_rate)
    plt.figure(figsize=(15, 5))
    plt.semilogy(f, Pxx)
    plt.title('Power Spectral Density - Welch’s method')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V^2/Hz)')
    plt.show()

# Fungsi untuk menghitung statistik dasar sinyal
def signal_statistics(data):
    mean = np.mean(data)
    variance = np.var(data)
    skewness = skew(data)
    kurt = kurtosis(data)
    print(f'Statistik Dasar Sinyal:\nMean: {mean:.2f}, Variance: {variance:.2f}, Skewness: {skewness:.2f}, Kurtosis: {kurt:.2f}')

# Fungsi utama untuk membaca file EDF dan melakukan visualisasi
def read_and_visualize_edf(file_path, channel_name):
    # Membaca file EDF menggunakan MNE
    raw = read_raw_edf(file_path, preload=True)
    sampling_rate = raw.info['sfreq']  # Mendapatkan frekuensi sampling
    
    # Mengambil data dari channel yang dipilih
    data = raw.get_data(picks=channel_name)[0]  # Ambil channel tertentu, misal "EEG Fpz-Cz"
    
    # Visualisasi sinyal mentah
    plot_raw_signal(data, sampling_rate, channel_name=channel_name)
    
    # Visualisasi FFT
    plot_fft(data, sampling_rate)
    
    # Identifikasi Artefak
    identify_artifacts(data, sampling_rate)
    
    # Menghitung Statistik Dasar
    signal_statistics(data)

    # Menghitung SNR (gunakan noise yang disimulasikan)
    noise = data * 0.1  # Asumsi noise sebagai 10% dari amplitudo sinyal
    snr = calculate_snr(data, noise)
    print(f'Signal-to-Noise Ratio (SNR): {snr:.2f} dB')

# Contoh penggunaan
file_path = 'path/to/your/file.edf'  # Ganti dengan path ke file EDF Anda
channel_name = "EEG Fpz-Cz"
read_and_visualize_edf(file_path, channel_name)
