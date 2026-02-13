import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path

def main(args):
    node_types = ['MAX', 'MIN', 'DEWP', 'SLP', 'WDSP', 'MXSPD']
    node_data = {}
    base_path = Path(args.base_path)

    for node_type in node_types:
        node_data[node_type] = {}
        
        station_file = base_path / f'unique_stations_{node_type}.csv'
        if not station_file.exists():
            continue
            
        node_info_df = pd.read_csv(station_file, dtype={'station_id': str})
        station_ids = node_info_df['station_id'].tolist()
        num_stations = len(station_ids)
        
        if num_stations == 0:
            continue
        
        station_to_idx = {station_id: idx for idx, station_id in enumerate(station_ids)}
        
        var_path = base_path / node_type
        csv_files = sorted(var_path.glob('*.csv'))
        num_dates = len(csv_files)
        
        if num_dates == 0:
            continue
        
        feature_matrix = np.full((num_stations, num_dates), np.nan, dtype=np.float32)
        
        for date_idx, file in enumerate(tqdm(csv_files, desc=f"Processing {node_type}")):
            try:
                daily_df = pd.read_csv(file, dtype={'station_id': str}, usecols=['station_id', 'observation_value'])
                
                mask = daily_df['station_id'].isin(station_to_idx)
                daily_df = daily_df[mask]
                
                indices = [station_to_idx[sid] for sid in daily_df['station_id']]
                feature_matrix[indices, date_idx] = daily_df['observation_value'].values
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        
        node_data[node_type]['mean'] = float(np.nanmean(feature_matrix))
        node_data[node_type]['std'] = float(np.nanstd(feature_matrix))
        
        print(f"{node_type} - Stations: {num_stations}, Dates: {num_dates}, Mean: {node_data[node_type]['mean']:.2f}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(node_data, f)
    
    print(f"Climatology statistics saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='daily_processed_dataset')
    parser.add_argument('--output_path', type=str, default='daily_processed_dataset/climatology.pkl')
    args = parser.parse_args()
    main(args)