import pandas as pd
from pathlib import Path
from tqdm import tqdm

folder_path = Path('daily_processed_dataset')

variables = ["DEWP", "MXSPD", "SLP", "WDSP", "MAX", "MIN"]

dtype_spec = {
    'station_id': str,
    'latitude': float,
    'longitude': float
}

for variable in variables:
    var_path = folder_path / variable
    
    if not var_path.exists():
        continue
    
    csv_files = sorted(var_path.glob('*.csv'))
    
    if not csv_files:
        continue
    
    all_stations = []
    
    for file in tqdm(csv_files, desc=f"Extracting stations: {variable}"):
        df = pd.read_csv(
            file, 
            usecols=['station_id', 'latitude', 'longitude'], 
            dtype=dtype_spec
        )
        all_stations.append(df)
    
    station_info = pd.concat(all_stations, ignore_index=True)
    station_info = station_info.drop_duplicates(subset=['station_id'], keep='first')
    
    output_csv = folder_path / f"unique_stations_{variable}.csv"
    station_info.to_csv(output_csv, index=False)

print("Station extraction complete.")
