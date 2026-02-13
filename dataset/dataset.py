import dgl
import torch
import pickle
import numpy as np
import re
import warnings
from torch.utils.data import Dataset
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


class TemporalHeterogeneousDataset(Dataset):
    def __init__(self, data_dir,years,input_length,output_length,climatology=None):
        self.data_dir = Path(f"{data_dir}/dgl_neighbor_10_step_1_refine_5")
        self.years = [str(year) for year in years]
        self.all_file=[]
        self.input_length=input_length
        self.output_length=output_length
        for year in self.years:
            self.all_file.extend(list(self.data_dir.glob(f'*{year}*.bin')))
        with open(f"{data_dir}/climatology.pkl", 'rb') as f:
            self.climatology = pickle.load(f)
        self.all_file.sort()

    def __len__(self):
        return len(self.all_file)

    def normalize(self, graphs):
        for idx, graph in enumerate(graphs):
            for node_type in ['DEWP', 'MAX', 'MIN', 'MXSPD', 'SLP', 'WDSP']:
                for t in range(self.input_length+self.output_length):
                    graph.nodes[node_type].data[f't{t}']=(graph.nodes[node_type].data[f't{t}']-torch.tensor(self.climatology[node_type]['mean']).float())/torch.tensor(self.climatology[node_type]['std']).float()

        return graphs

    def get_time_embedding(self,date_str):
        date = datetime.strptime(date_str, "%Y-%m-%d")

        year = date.year
        month = date.month
        day = date.day
        start_of_year = datetime(year, 1, 1)
        end_of_year = datetime(year + 1, 1, 1)
        total_days_in_year = (end_of_year - start_of_year).days
        
        day_of_year = (date - start_of_year).days
        
        day_ratio = day_of_year / total_days_in_year
        
        time_embedding = [date.year-2000, date.month,date.day,np.cos(2 * np.pi * day_ratio), np.sin(2 * np.pi * day_ratio)]
        
        return time_embedding

    def get_embeddings_for_next_n_days(self,start_date_str, n_days):
        embeddings = []
        

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        

        for i in range(n_days):
            current_date = start_date + timedelta(days=i)
            current_date_str = current_date.strftime("%Y-%m-%d")
            embedding = self.get_time_embedding(current_date_str)
            embeddings.append(embedding)
        
        return embeddings

    def __getitem__(self, idx):
        graph,sh_embedding=dgl.load_graphs(str(self.all_file[idx]))
        date_str = re.search(r'(\d{4}-\d{2}-\d{2})', str(self.all_file[idx])).group(0)
        time_embeddings = torch.tensor(self.get_embeddings_for_next_n_days(date_str,self.input_length+self.output_length), dtype=torch.float32)
        graph=self.normalize(graph)

        sample_idx_per_var = {}
        for node_type in ['DEWP', 'MAX', 'MIN', 'MXSPD', 'SLP', 'WDSP']:
            num_nodes = graph[0].num_nodes(node_type)
            sample_idx_per_var[node_type] = [None] * num_nodes  

        
        return graph[0], time_embeddings, sh_embedding, sample_idx_per_var


