import os
import json
import re
import yaml
import numpy as np

def load_yaml(file_path):
    """
    Loads a YAML file and returns its contents as a Python dictionary.
    
    Parameters:
        file_path (str): The path to the YAML file.
    
    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)  # Use safe_load for security
    return data

def load_json(file_path):
    """
    Load the entire JSON content from a file.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_number(filename):
    """
    Extracts the numeric part of the filename.
    Assumes that there is a number in the filename. Modify the regex pattern as needed.
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')  # Return 'inf' if no number is found to handle such cases

def get_sorted_files(folder_path):
    """
    Get all files in the folder and sort them by the numeric part of the filenames.
    """
    files = os.listdir(folder_path)
    
    # Filter only files (ignoring directories)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

    # Sort files by the extracted numeric part of the filename
    sorted_files = sorted(files, key=extract_number)

    return sorted_files

def list_subfolders(folder_path):
    """
    List all subfolders in the given folder path.
    """
    return [f.name for f in os.scandir(folder_path) if f.is_dir()]

def get_order_diagonals(matrix, k):
    """
    Extracts the main diagonal and the k-1 diagonals below the main diagonal.
    Trims the diagonals so that all diagonals have the same length as the lowest diagonal.

    Parameters:
    - matrix: Input 2D NumPy array.
    - k: Number of diagonals to extract (main diagonal + k-1 below).

    Returns:
    - diagonals: List of diagonals with matching lengths.
    """
    diagonals = []
    n = matrix.shape[0]
    
    # Extract diagonals from the main diagonal (k=0) to the (k-1) diagonals below
    for i in range(k):
        diag = np.diag(matrix, k=-i)
        diag = diag[k-i-1:]
        diagonals.append(diag)
    
    return diagonals

def get_all_diagonals(matrix):
    """
    Extracts the main diagonal and the k-1 diagonals below the main diagonal.
    Trims the diagonals so that all diagonals have the same length as the lowest diagonal.

    Parameters:
    - matrix: Input 2D NumPy array.
    - k: Number of diagonals to extract (main diagonal + k-1 below).

    Returns:
    - diagonals: List of diagonals with matching lengths.
    """
    diagonals = []
    n = matrix.shape[0]
    
    # Extract diagonals from the main diagonal (k=0) to the (k-1) diagonals below
    for i in range(n):
        diag = np.diag(matrix, k=-i)
        diagonals.append(diag)
    
    return diagonals

def compute_mean_std(diagonals):
    """
    Computes the mean and standard deviation for each diagonal.

    Parameters:
    - diagonals: List of diagonals.

    Returns:
    - means: List of means for each diagonal.
    - stds: List of standard deviations for each diagonal.
    """
    means = [np.mean(diag) for diag in diagonals]
    stds = [np.std(diag) for diag in diagonals]
    
    return means, stds

def write_to_json(data, filename):
    """
    Writes the given data to a JSON file.

    Parameters:
    - data: The data to be written (usually a dictionary or list).
    - filename: The name of the file to write the JSON data to.
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # `indent=4` makes the output readabl
