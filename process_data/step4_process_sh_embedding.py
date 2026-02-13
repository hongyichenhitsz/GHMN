import os
import numpy as np
import pandas as pd
import torch
from typing import Dict
from pathlib import Path

def load_station_lonlat(station_csv_path: str) -> torch.Tensor:
    """
    Loads station coordinates from CSV and returns an (N, 2) tensor.
    Expects 'longitude' and 'latitude' columns.
    """
    df = pd.read_csv(station_csv_path)
    required_cols = ["longitude", "latitude"]
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV missing required columns: {required_cols}")
    
    lonlat = np.stack([df["longitude"].values, df["latitude"].values], axis=1)
    return torch.tensor(lonlat, dtype=torch.float32)

def main():
    config = {
        "vars": ["DEWP", "MAX", "MIN", "MXSPD", "SLP", "WDSP"],
        "station_csv_dir": "daily_processed_dataset",
        "embedding_save_dir": "daily_processed_dataset/embedded_dir",
        "existing_healpix_npy_path": "daily_processed_dataset/healpix_sh_refine_5_9.npy",
        "legendre_polys": 3
    }

    os.makedirs(config["embedding_save_dir"], exist_ok=True)
    
    # Import custom location encoder module
    from locationencoder.pe import SphericalHarmonics
    sh_calculator = SphericalHarmonics(
        legendre_polys=config["legendre_polys"], 
        harmonics_calculation="closed-form"
    )

    print(f"{'='*50}\nGenerating Source Node Embeddings (Stations)\n{'='*50}")
    
    for var in config["vars"]:
        station_csv_path = os.path.join(
            config["station_csv_dir"], 
            f"unique_stations_{var}.csv"
        )
        
        if not os.path.exists(station_csv_path):
            print(f"Skipping {var}: File not found at {station_csv_path}")
            continue

        station_lonlat = load_station_lonlat(station_csv_path)
        embedded_station = sh_calculator(station_lonlat)

        save_path = os.path.join(config["embedding_save_dir"], f"{var}_embeddings.pt")
        torch.save(embedded_station, save_path)
        print(f"Variable {var}: Saved to {os.path.basename(save_path)} | Shape: {list(embedded_station.shape)}")

    print(f"\n{'='*50}\nConverting HEALPix Embeddings (.npy -> .pt)\n{'='*50}")
    
    if os.path.exists(config["existing_healpix_npy_path"]):
        healpix_np = np.load(config["existing_healpix_npy_path"])
        healpix_pt = torch.tensor(healpix_np, dtype=torch.float32)
        
        h_save_path = os.path.join(config["embedding_save_dir"], "healpix_embeddings_5.pt")
        torch.save(healpix_pt, h_save_path)
        print(f"HEALPix: Saved to {os.path.basename(h_save_path)} | Shape: {list(healpix_pt.shape)}")
    else:
        print(f"Warning: Reference HEALPix file not found at {config['existing_healpix_npy_path']}")

    print(f"\n{'='*50}\nProcessing Complete\n{'='*50}")

if __name__ == "__main__":
    main()