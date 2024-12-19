import os

import kagglehub

import re

# Set the directory where you want to save the dataset
destination_dir = os.path.join(os.getcwd(), 'dataset_from_kaggle')



# Download the dataset to the specified directory
path = kagglehub.dataset_download("alessiocorrado99/animals10")
path = os.path.join(path, 'raw-img', 'cane')
path = repr(path)

print("Path to dataset files:", path)

# Path to the .py file
file_path = os.path.join(os.getcwd(), 'variables', 'variables.py')


# Read the original file
with open(file_path, "r") as file:
    content = file.read()

# Update the variable value using regex
# Matches: variable_name = <any value>
updated_content = re.sub(r"kaggle_data_path\s*=\s*.*", r"kaggle_data_path = r{}".format(path), content)

# Write the updated content back to the file
with open(file_path, "w") as file:
    file.write(updated_content)

print(f"Updated 'variable_name' in {file_path} to {path}!")