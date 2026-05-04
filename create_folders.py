import os

# Path to your 'data' folder
data_path = "data"

# List of subfolders to create
subfolders = ["normal", "stressed", "fatigued"]

# Create each folder
for folder in subfolders:
    folder_path = os.path.join(data_path, folder)
    os.makedirs(folder_path, exist_ok=True)  # exist_ok=True avoids errors if folder exists
    print(f"Created folder: {folder_path}")
