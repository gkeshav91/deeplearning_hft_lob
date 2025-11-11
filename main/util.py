
import os
import re
import pandas as pd
import multiprocessing
from functools import partial

def read_and_add_date(file_path, sampling_fraction=None):
    """Reads a CSV file, optionally samples it, and adds a date column extracted from the filename."""
    
    data = pd.read_csv(file_path)  # Read CSV file
    
    # Optional Sampling
    if sampling_fraction is not None and sampling_fraction < 1:
        data = data.sample(frac=sampling_fraction, random_state=123)  # Sample rows
    
    # Extract date from filename
    file_name = os.path.basename(file_path)
    match = re.search(r"\d{8}", file_name)  # Find 8-digit date
    if not match:
        raise ValueError(f"Could not extract date from file: {file_path}")
    
    data["date"] = match.group(0)  # Add date column
    return data

def combine_csv(folder_path, regex_pattern, cores=1, sampling_fraction=None):
    """Reads multiple CSV files in parallel, processes them, and combines into one DataFrame."""
    
    # List all matching files
#    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if re.search(regex_pattern, f)]
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if re.search(regex_pattern, f):
                full_path = os.path.join(root, f)
                all_files.append(full_path)

    
    if not all_files:
        raise ValueError("No files found matching the specified pattern!")
    
    # Parallel Processing
    with multiprocessing.Pool(cores) as pool:
        read_func = partial(read_and_add_date, sampling_fraction=sampling_fraction)
        combined_data = pd.concat(pool.map(read_func, all_files), ignore_index=True)

    return combined_data

# Example Usage:
# df = combine_csv("data_folder", r"\d{8}.*\.csv", cores=4, sampling_fraction=0.5)


def trimtails(df, cutoff):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    def trim_column(col):
        if pd.api.types.is_numeric_dtype(col):
            lower_limit = col.quantile(1 - cutoff)
            upper_limit = col.quantile(cutoff)
            return col.clip(lower=lower_limit, upper=upper_limit)
        return col

    return df.apply(trim_column)




