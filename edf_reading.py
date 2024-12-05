import os
import numpy as np
import mne

# Function to process a single EDF file
def process_edf_file(file_path, output_dir):
    try:
        print(f"Processing {file_path}...")
        # Load the EDF file
        raw = mne.io.read_raw_edf(file_path, preload=True)

        # Apply bandpass filtering (1-50 Hz)
        raw.filter(1., 50., fir_design='firwin')

        # Extract the data and save it as a NumPy array
        data, times = raw.get_data(return_times=True)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Save the filtered data and timestamps
        output_data_path = os.path.join(output_dir, f"{file_name}_processed_data.npy")
        output_time_path = os.path.join(output_dir, f"{file_name}_processed_times.npy")
        np.save(output_data_path, data)
        np.save(output_time_path, times)

        print(f"Processed data saved to {output_data_path}")
        print(f"Processed timestamps saved to {output_time_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Function to process all EDF files in a directory
def process_all_edf_files(folder_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it doesn't exist

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.edf'):
                file_path = os.path.join(root, file)
                process_edf_file(file_path, output_dir)
            else:
                print(f"Skipping non-EDF file: {file}")

# Define input folder and output folder
input_folder = r"c:\Users\nldad\Documents\Final-Project-BME3053C\data\st-vincents-data\files"  # Update with your folder path
output_folder = r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\outputs"  # Specify where processed files should be saved

# Run the batch processing
process_all_edf_files(input_folder, output_folder)
