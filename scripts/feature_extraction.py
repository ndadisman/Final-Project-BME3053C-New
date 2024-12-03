import os
import numpy as np
from scipy.signal import welch
import pandas as pd

# Function to calculate band power in specified frequency bands
def calculate_band_power(data, sf, bands):
    """
    Calculate power for each EEG band in the signal.
    
    Parameters:
    - data: 2D array of EEG signals (channels x samples)
    - sf: Sampling frequency
    - bands: Dictionary of frequency bands (e.g., {'Delta': (0.5, 4)})
    
    Returns:
    - band_powers: Dictionary with band power values
    """
    band_powers = {band: [] for band in bands.keys()}
    for channel_data in data:
        f, psd = welch(channel_data, sf, nperseg=sf*2)
        for band, (low, high) in bands.items():
            band_power = np.trapz(psd[(f >= low) & (f <= high)], f[(f >= low) & (f <= high)])
            band_powers[band].append(band_power)
    return band_powers

# Function to process all processed .npy files
def extract_features(input_dir, output_dir, sf, bands):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file in os.listdir(input_dir):
        if file.endswith('_processed_data.npy'):
            file_path = os.path.join(input_dir, file)
            data = np.load(file_path)
            file_name = os.path.splitext(file)[0]

            # Calculate band powers
            band_powers = calculate_band_power(data, sf, bands)
            df = pd.DataFrame(band_powers)

            # Save the features as a CSV file
            output_file_path = os.path.join(output_dir, f"{file_name}_features.csv")
            df.to_csv(output_file_path, index=False)
            print(f"Features saved to {output_file_path}")

# Define parameters
input_dir = r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\outputs"
output_dir = r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\features"
sampling_frequency = 128  # Update with your data's sampling frequency
frequency_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 50)
}

# Run the feature extraction
extract_features(input_dir, output_dir, sampling_frequency, frequency_bands)
