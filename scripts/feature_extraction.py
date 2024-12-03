import os
import numpy as np
import pandas as pd
from scipy.signal import welch

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

# Function to classify sleep stages based on band powers
def classify_sleep_stage(band_powers):
    """
    Classify sleep stage based on power in different frequency bands.
    
    Parameters:
    - band_powers: Dictionary with the mean band power for each EEG frequency band
    
    Returns:
    - sleep_stage: The sleep stage classification as a string
    """
    delta_power = np.mean(band_powers['Delta'])
    theta_power = np.mean(band_powers['Theta'])
    alpha_power = np.mean(band_powers['Alpha'])
    beta_power = np.mean(band_powers['Beta'])

    # Debug: Print band power values to verify their magnitude
    print(f"Delta Power: {delta_power}, Theta Power: {theta_power}, Alpha Power: {alpha_power}, Beta Power: {beta_power}")

    # Adjust thresholds based on observed values
    # Experimenting with thresholds, based on magnitude of band powers.
    if beta_power > 1E-8 and theta_power < 1E-8:  # Wake (High beta, low theta)
        return "Wake"
    elif delta_power > 1E-8 and theta_power < 1E-8:  # Deep Sleep (High delta, low theta)
        return "Stage 3/4 (Deep Sleep)"
    elif theta_power > 1E-8 and delta_power < 1E-8:  # Stage 1/2 (Moderate theta, low delta)
        if alpha_power > 1E-8:
            return "Stage 2 (Light Sleep)"
        else:
            return "Stage 1 (Light Sleep)"
    elif theta_power > 1E-8 and beta_power < 1E-8:  # REM Sleep (Moderate theta, low beta)
        return "REM Sleep"
    else:
        return "Unknown"

# Function to process all processed .npy files and extract features
def extract_features(input_dir, output_dir, sf, bands):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith("_processed_data.npy"):
            file_path = os.path.join(input_dir, file)
            data = np.load(file_path)  # Load EEG data
            file_name = os.path.splitext(file)[0]  # Get the file name without extension

            # Calculate band powers
            band_powers = calculate_band_power(data, sf, bands)
            sleep_stage = classify_sleep_stage(band_powers)  # Classify sleep stage

            # Convert band powers to a DataFrame
            df = pd.DataFrame(band_powers)
            df['Sleep Stage'] = sleep_stage  # Add the sleep stage to the DataFrame

            # Save the features and sleep stage as a CSV file
            output_file_path = os.path.join(output_dir, f"{file_name}_features.csv")
            df.to_csv(output_file_path, index=False)
            print(f"Features and sleep stage saved to {output_file_path}")

# Combine all CSV files into a single DataFrame
def combine_csv(input_dir, output_file):
    dataframes = []
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            # Read the CSV file
            df = pd.read_csv(os.path.join(input_dir, file))
            # Assuming the label is in the filename, e.g., 'class1_data.csv' -> label 'class1'
            label = file.split('_')[0]  # Adjust this based on your filename convention
            df['Signal'] = label  # Add the label as a new column
            dataframes.append(df)

    # Combine all dataframes into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined dataset with the added label column
    combined_df.to_csv(output_file, index=False)
    print(f"Combined dataset saved at {output_file}")

# Define parameters
input_dir = r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\outputs"
output_dir = r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\features"
combined_csv_path = r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\features_dataset.csv"
sampling_frequency = 128  # Update with your data's sampling frequency
frequency_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 50)
}

# Run the feature extraction and classification
extract_features(input_dir, output_dir, sampling_frequency, frequency_bands)

# Combine all extracted features and save them as a single dataset
combine_csv(output_dir, combined_csv_path)
