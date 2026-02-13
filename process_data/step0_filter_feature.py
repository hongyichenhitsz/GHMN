import os
import pandas as pd
from glob import glob
from datetime import datetime
import time
from tqdm import tqdm

base_dir = "dataset"
output_dir = "daily_processed_dataset"
os.makedirs(output_dir, exist_ok=True)

def fahrenheit_to_kelvin(fahrenheit):
    return (5/9) * (fahrenheit - 32) + 273.15

variables = {
    "MXSPD": 999.9,
    "DEWP": 9999.9,
    "SLP": 9999.9,  
    "WDSP": 999.9,
    "MAX": 9999.9, 
    "MIN": 9999.9, 
}

for var in variables:
    os.makedirs(os.path.join(output_dir, var), exist_ok=True)

overall_start = time.time()

for year in range(2017, 2025):
    year_path = os.path.join(base_dir, str(year))
    if not os.path.exists(year_path):
        continue  

    csv_files = glob(os.path.join(year_path, "*.csv"))
    if not csv_files:
        continue

    dfs = []
    cols = ['DATE', 'STATION', 'LONGITUDE', 'LATITUDE'] + list(variables.keys())
    
    for file in tqdm(csv_files, desc=f"Loading {year}", unit="file"):
        df = pd.read_csv(file, dtype={'STATION': str}, usecols=cols, parse_dates=['DATE'])
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['DATE'] = pd.to_datetime(combined_df['DATE']).dt.date
    
    year_start = datetime(year, 1, 1).date()
    year_end = datetime(year, 12, 31).date()
    combined_df = combined_df[combined_df['DATE'].between(year_start, year_end)]
    
    date_groups = list(combined_df.groupby('DATE'))
    
    for date, group in tqdm(date_groups, desc=f"Processing {year}", unit="day"):
        date_str = date.strftime("%Y-%m-%d")
        
        for var, missing_value in variables.items():
            valid_data = group[group[var] != missing_value]
            
            if valid_data.empty:
                continue
                
            output_data = valid_data[['LONGITUDE', 'LATITUDE', 'STATION', var]].copy()
            output_data.columns = ["longitude", "latitude", "station_id", "observation_value"]
            
            output_data = output_data.dropna(how='any').drop_duplicates(subset=['station_id'])
            
            if var in ["MAX", "MIN", "TEMP", "DEWP"]:
                output_data['observation_value'] = fahrenheit_to_kelvin(output_data['observation_value'])
            
            file_path = os.path.join(output_dir, var, f"{date_str}.csv")
            output_data.to_csv(file_path, index=False)

print(f"\nProcessing complete. Total time: {time.time() - overall_start:.2f}s")
