import mne
import matplotlib.pyplot as plt

# Load the EDF file
edf_file = r"c:\Users\nldad\Documents\Final-Project-BME3053C\Final-Project-BME3053C-New\data\st-vincents-data\files\ucddb002_lifecard.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)

# Plot the signal
raw.plot()
plt.show()
