import dgl
import torch
import pickle
import numpy as np
from dgl.dataloading import GraphDataLoader, batch_graphs
from dataset.dataset import TemporalHeterogeneousDataset  

def custom_collate_fn(batch):
    """
    Handles merging of heterogeneous graphs and sample indices for batched processing.
    """
    graphs, time_embeddings, sh_embeddings, sample_idx_per_vars = zip(*batch)
    
    batched_graph = batch_graphs(graphs)
    batched_time = torch.stack(time_embeddings, dim=0)
    
    batched_sh = {}
    for key in sh_embeddings[0].keys():
        batched_sh[key] = torch.stack([sh[key] for sh in sh_embeddings], dim=0)
    
    batched_sample_idx_per_var = {}
    for key in sample_idx_per_vars[0].keys():
        batched_sample_idx_per_var[key] = [
            sample_idx_per_var[key] for sample_idx_per_var in sample_idx_per_vars
        ]
    
    return batched_graph, batched_time, batched_sh, batched_sample_idx_per_var

# --- Configuration ---
data_args = {
    "data_dir": 'daily_processed_dataset',
    "train_years": [2017, 2018, 2019, 2020, 2021, 2022],
    "input_length": 1,
    "output_length": 1,
    "predict_vars": ['DEWP', 'MAX', 'MIN', 'MXSPD', 'SLP', 'WDSP'],
    "climatology_dir": 'daily_processed_dataset/climatology.pkl',
    "batch_size": 32,
}

model_args = {
    "num_workers": 4,
}

PERCENTILE_VALUES = [0.5, 2, 3, 10, 90, 97, 98, 99.5]
SAVE_PATH = "daily_processed_dataset/percentile.pkl"

def calculate_percentile_manual(data_args, model_args, percentile_values, save_path):
    """
    Iterates through the training set to extract labels, performs inverse normalization, 
    and calculates statistical percentiles for extreme event detection.
    """
    with open(data_args['climatology_dir'], 'rb') as f:
        climatology = pickle.load(f)
    
    var_names = data_args['predict_vars']
    var_data_store = {var: [] for var in var_names}
    
    print("Initializing Training Dataset...")
    train_dataset = TemporalHeterogeneousDataset(
        data_dir=data_args["data_dir"],
        years=data_args["train_years"],
        input_length=data_args["input_length"],
        output_length=data_args["output_length"],
        climatology=climatology
    )
    
    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=data_args["batch_size"],
        shuffle=False,
        num_workers=model_args["num_workers"],
        collate_fn=custom_collate_fn
    )
    
    print(f"Starting traversal ({len(train_loader)} batches)...")
    for batch_idx, batch in enumerate(train_loader):
        graph, _, _, _ = batch
        input_len = data_args['input_length']
        output_len = data_args['output_length']
        
        for var in var_names:
            var_label_list = [graph.nodes[var].data[f't{input_len + t}'] for t in range(output_len)]
            var_label_tensor = torch.stack(var_label_list, dim=1) 
            
            mean_val = climatology[var]['mean']
            std_val = climatology[var]['std']
            
            mean = torch.tensor(mean_val, dtype=torch.float32, device=var_label_tensor.device)
            std = torch.tensor(std_val, dtype=torch.float32, device=var_label_tensor.device)
            
            # Inverse Normalization: (x * std) + mean
            var_label_denorm = var_label_tensor * std.view(1, 1, 1) + mean.view(1, 1, 1)
            
            var_label_flat = var_label_denorm.detach().cpu().numpy().flatten()
            var_data_store[var].append(var_label_flat)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"Processed {batch_idx + 1}/{len(train_loader)} batches")

    print("\nAggregating data and calculating percentiles...")
    percentile_tensor = torch.zeros((1, 1, len(var_names), len(percentile_values)), dtype=torch.float32)
    
    for var_idx, var in enumerate(var_names):
        var_all_data = np.concatenate(var_data_store[var], axis=0)
        var_percentiles = np.nanpercentile(var_all_data, percentile_values)
        
        print(f"Variable: {var} | Data points: {var_all_data.shape[0]:,}")
        print(f"Results: {dict(zip(percentile_values, var_percentiles.round(4)))}")
        
        percentile_tensor[0, 0, var_idx, :] = torch.tensor(var_percentiles, dtype=torch.float32)
    
    with open(save_path, 'wb') as f:
        pickle.dump(percentile_tensor, f)
    
    print(f"\nCalculation complete. Saved to: {save_path}")

if __name__ == "__main__":
    calculate_percentile_manual(
        data_args=data_args,
        model_args=model_args,
        percentile_values=PERCENTILE_VALUES,
        save_path=SAVE_PATH
    )