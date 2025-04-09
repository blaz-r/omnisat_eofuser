from PIL import Image
import h5py
import numpy as np
import os

# Directory containing .tif files
input_dir = "/home/filip/OmniSat/data/SocaDataset/sentinel_tif"
output_dir = "/home/filip/OmniSat/data/SocaDataset/sentinel_h5"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through all .tif files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".tif"):
        tif_path = os.path.join(input_dir, filename)
        h5_path = os.path.join(output_dir, filename.replace(".tif", ".h5"))

        # Open the .tif file and convert it to a NumPy array
        with Image.open(tif_path) as img:
            image_array = np.array(img)

        # Save the NumPy array to an .h5 file
        with h5py.File(h5_path, "w") as h5_file:
            h5_file.create_dataset("image", data=image_array)

        print(f"Converted {tif_path} to {h5_path}")
