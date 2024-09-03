import numpy as np
import matplotlib.pyplot as plt
from mne.io import read_raw_edf
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import argparse
import glob
import math
import ntpath
import os
import shutil
from datetime import datetime
import dhedfreader

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

EPOCH_SEC_SIZE = 30

# Fungsi untuk memfilter dan memproses sinyal EEG dari file EDF
def preprocess_eeg(file_path, channel_name):
    # Read EDF file using MNE
    raw = read_raw_edf(file_path, preload=True)
    sampling_rate = raw.info['sfreq']  # Get the sampling frequency
    
    # Select only the desired channel using the new recommended method
    raw.pick([channel_name])

    # Apply band-pass filter (0.5–40 Hz)
    raw.filter(0.5, 40.0, fir_design='firwin')
    
    # Extract data from the selected channel
    data = raw.get_data(picks=channel_name)[0]  # Pick specific channel, e.g., "EEG Fpz-Cz"
    
    return data, sampling_rate

# Fungsi untuk menyimpan data yang sudah difilter ke dalam format NPZ
def save_filtered_data(file_path, channel_name, output_dir):
    # Read EDF file and preprocess it
    data, sampling_rate = preprocess_eeg(file_path, channel_name)
    
    # Save the filtered data into NPZ format
    filename = ntpath.basename(file_path).replace("-PSG.edf", "-filtered.npz")
    save_dict = {
        "data": data,
        "sampling_rate": sampling_rate,
        "channel_name": channel_name
    }
    np.savez(os.path.join(output_dir, filename), **save_dict)
    print(f"Filtered data saved to {os.path.join(output_dir, filename)}")

# Fungsi untuk memvisualisasikan sinyal yang telah difilter
def plot_filtered_signal(data, sampling_rate, title="Filtered EEG Signal", channel_name="EEG Fpz-Cz", duration=5):
    # Menghitung jumlah sampel untuk durasi yang diinginkan
    num_samples = int(duration * sampling_rate)
    
    # Memotong data sesuai durasi yang diinginkan
    data = data[:num_samples]
    
    # Mengatur waktu untuk plot
    time = np.arange(0, len(data)) / sampling_rate
    
    # Membuat plot
    plt.figure(figsize=(15, 5))
    plt.plot(time, data, color='blue')
    plt.title(f'{title} - {channel_name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (µV)')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_edf_20",
                        help="File path to the PSG and annotation files.")
    parser.add_argument("--output_dir", type=str, default="data_edf_20_npz/fpzcz",
                        help="Directory where to save numpy files outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG Fpz-Cz",
                        help="The selected channel")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Select channel
    select_ch = args.select_ch

    # Read raw and annotation EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    psg_fnames.sort()

    for i in range(len(psg_fnames)):
        # Preprocess and filter the data
        filtered_data, sampling_rate = preprocess_eeg(psg_fnames[i], select_ch)

        # Save the filtered data to NPZ format
        save_filtered_data(psg_fnames[i], select_ch, args.output_dir)

        # Visualize the filtered signal
        plot_filtered_signal(filtered_data, sampling_rate, title="Filtered EEG Signal", channel_name=select_ch, duration=5)

        print("\n=======================================\n")

if __name__ == "__main__":
    main()
