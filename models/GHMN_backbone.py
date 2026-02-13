import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
import dgl.nn.pytorch as dglnn
from dgl.dataloading import GraphDataLoader, batch_graphs
from einops import rearrange
from models.GHMN_flowmatchinghead import FlowMatchingHead

from models.unet_processor import init_mesh_model
from models.layers.hetero import HeteroGraphConv
from models.layers.readout import AttentionReadout
from dataset.dataset import TemporalHeterogeneousDataset
from utils import criterion
from utils.crps import crps_from_samples
from utils.sedi import ProbabilisticMultiMetricsCalculator

class GHMNBackbone(nn.Module):
    def __init__(self, model_args,data_args):
        super(GHMNBackbone, self).__init__()

        self.input_length=data_args['input_length']
        self.output_length=data_args['output_length']
        self.hidden_size = model_args['hidden_size']
        
        self.model_args=model_args
        self.data_args=data_args

        with open(self.data_args['climatology_dir'], 'rb') as f:
            self.climatology = pickle.load(f)

        self.Encoder =Graph_Encoder(self.model_args,self.data_args)
        self.Processor=init_mesh_model(self.model_args,self.data_args)
        self.Decoder = Graph_Decoder(self.model_args,self.data_args)
        
        if self.model_args['sh_before'] or self.model_args['sh_after']:
            self.position_embedding_feature =nn.ParameterDict({node_type : nn.Parameter(torch.randn(9)) for node_type in self.data_args['vars']})
            self.position_embedding_healpix = nn.Parameter(torch.randn(9))


    def forward(self, g,timestamp,sh_embedding):

        if self.model_args['sh_before'] or self.model_args['sh_after']:

            x=self.Encoder(g,timestamp,sh_embedding,self.position_embedding_feature,self.position_embedding_healpix)
            x=x.unsqueeze(0).reshape(-1,self.data_args['num_healpix_nodes'],self.data_args['var_num'],self.input_length,self.hidden_size).permute(0,2,3,1,4)
            x=self.Processor(x)
            x=x.permute(0,3,1,2,4).reshape(-1,self.output_length,self.data_args['var_num'],self.hidden_size)
            g=self.Decoder(x,g,sh_embedding,self.position_embedding_feature,self.position_embedding_healpix )

        return g

class Graph_Encoder(nn.Module):
    def __init__(self, model_args,data_args):
        super(Graph_Encoder, self).__init__()

        self.edge_types=[]
        self.model_args=model_args
        self.data_args=data_args
        for node_type in self.data_args['vars']:
            for t in range(self.data_args['input_length']):
                edge_type = (node_type, f't{t}_to_healpix', 'healpix')
                self.edge_types.append(edge_type)
        if self.model_args['sh_before']:
            self.convs = HeteroGraphConv(
                {
                    edge_type: dglnn.GraphConv(1+9, self.model_args['hidden_size'],activation=nn.LeakyReLU())  
                    for edge_type in self.edge_types
                },
                aggregate='sum' 
            )
        else:
            self.convs = HeteroGraphConv(
                {
                    edge_type: dglnn.GraphConv(1, self.model_args['hidden_size'],activation=nn.LeakyReLU())  
                    for edge_type in self.edge_types
                },
                aggregate='sum' 
            )

        self.norm  = dglnn.EdgeWeightNorm(norm='right')

        self.model_args=model_args
        self.data_args=data_args
        self.input_length=self.data_args['input_length']
        if self.model_args['sh_before']:
            self.position_embedding_src ={var: torch.load(f"{self.data_args['embedding_dir']}/{var}_embeddings.pt").to('cuda:0') for var in self.data_args['vars']}
            self.position_embedding_dst = torch.load(f"{self.data_args['embedding_dir']}/healpix_embeddings_5.pt").to('cuda:0')

    def forward(self, g,timestamp,sh_embedding=None,position_embedding_feature=None,position_embedding_healpix=None):
        
        for node_type in self.data_args['vars']:
            
            for t in range(self.input_length):
                
                x_src={}
                x_dst={}
                x_src[node_type]=torch.nan_to_num(g.nodes[node_type].data[f't{t}'], nan=0.0)
                x_dst['healpix'] = g.nodes['healpix'].data[f'{node_type}_t{t}']
                subgraph = g.edge_type_subgraph([(node_type, f't{t}_to_healpix', 'healpix') for t in range(self.input_length)])
                if self.model_args['sh_before']:
                    src_position_embeddings = torch.cat([self.position_embedding_src[node_type][sh_embedding[node_type][i].bool()] for i in range(sh_embedding[node_type].shape[0])],dim=0)
                    dst_position_embeddings = torch.cat([self.position_embedding_dst for i in range(sh_embedding[node_type].shape[0])],dim=0)
                    x_src[node_type]=torch.concat([x_src[node_type],src_position_embeddings*position_embedding_feature[node_type]],dim=1)
                    x_dst['healpix']=torch.concat([x_dst['healpix'],dst_position_embeddings*position_embedding_healpix],dim=1)
                g.nodes['healpix'].data[f'{node_type}_t{t}']=self.convs(subgraph,(x_src,x_dst))['healpix']
            

        concatenated_features = []
        for node_type in self.data_args['vars']:
            features = [g.nodes['healpix'].data[f'{node_type}_t{t}'] for t in range(self.input_length)]
            concatenated_feature = torch.stack(features, dim=1)
            concatenated_features.append(concatenated_feature)
        final_feature_tensor = torch.stack(concatenated_features, dim=1)

        return final_feature_tensor
    
class Graph_Decoder(nn.Module):
    def __init__(self, model_args,data_args):
        super(Graph_Decoder, self).__init__()

        self.model_args=model_args
        self.data_args=data_args
        self.input_length=self.data_args['input_length']
        self.output_length=self.data_args['output_length']
        self.hidden_size=self.model_args['hidden_size']
        self.edge_types=[]
        for node_type in self.data_args['vars']:
            for t in range(self.data_args['output_length']):
                edge_type = ('healpix', f't{self.input_length+t}_to_{node_type}', node_type)
                self.edge_types.append(edge_type)

        if self.model_args['sh_after']:
            self.convs = HeteroGraphConv(
                {
                    edge_type: dglnn.GraphConv(self.model_args['hidden_size']+9,self.model_args['hidden_size'] ,activation=nn.LeakyReLU())
                    for edge_type in self.edge_types
                },
                aggregate='sum' 
            )
        else:
            self.convs = HeteroGraphConv(
                {
                    edge_type: dglnn.GraphConv(self.model_args['hidden_size'],self.model_args['hidden_size'] ,activation=nn.LeakyReLU())
                    for edge_type in self.edge_types
                },
                aggregate='sum' 
            )
        self.norm  = dglnn.EdgeWeightNorm(norm='right')

        if self.model_args['sh_after']:
            self.position_embedding_src = torch.load(f"{self.data_args['embedding_dir']}/healpix_embeddings_5.pt").to('cuda:0')
            self.position_embedding_dst = {var: torch.load(f"{self.data_args['embedding_dir']}/{var}_embeddings.pt").to('cuda:0') for var in self.data_args['vars']}
        self.hidden_size=self.model_args['hidden_size']
        if self.model_args['sh_after']:
            self.linears = nn.ModuleDict({node_type: nn.Linear(1+9,self.hidden_size) for node_type in self.data_args['vars']})
        else:
            self.linears = nn.ModuleDict({node_type: nn.Linear(1,self.hidden_size) for node_type in self.data_args['vars']})
        self.readout = AttentionReadout(
            input_size=self.model_args['hidden_size'],
            hidden_size=self.model_args['hidden_size']*2,
            output_size=1,
            dim_size=2,
            horizon=1,
            dim=1,
            fully_connected=False)

    def forward(self, x,g,sh_embedding=None,position_embedding_feature=None,position_embedding_healpix=None):
        for idx, node_type in enumerate(self.data_args['vars']):
            for t in range(self.output_length):
                

                g.nodes['healpix'].data[f'{node_type}_t{self.input_length+t}']=x[:,t,idx]
                x_src={}
                x_dst={}
                x_src['healpix']=g.nodes['healpix'].data[f'{node_type}_t{self.input_length+t}']
                g.nodes[node_type].data[f't{self.input_length+t}'] = torch.zeros_like(g.nodes[node_type].data[f't{self.input_length+t}'])
                x_dst[node_type]=g.nodes[node_type].data[f't{self.input_length+t}']


                subgraph = g.edge_type_subgraph([('healpix', f't{self.input_length+t}_to_{node_type}', node_type) for t in range(self.output_length)])

                if self.model_args['sh_after']:
                    src_position_embeddings = torch.cat([self.position_embedding_src for i in range(sh_embedding[node_type].shape[0])],dim=0)
                    dst_position_embeddings = torch.cat([self.position_embedding_dst[node_type][sh_embedding[node_type][i].bool()] for i in range(sh_embedding[node_type].shape[0])],dim=0)

                    x_src['healpix']=torch.concat([x_src['healpix'],src_position_embeddings*position_embedding_healpix],dim=1)
                    x_dst[node_type]=torch.concat([x_dst[node_type],dst_position_embeddings*position_embedding_feature[node_type]],dim=1)

                space_1 = self.convs(subgraph,(x_src,x_dst))[node_type].unsqueeze(0).unsqueeze(0).unsqueeze(0)

                if self.model_args['sh_after']:
                    space_2 = self.linears[node_type](torch.concat([g.nodes[node_type].data[f't{t}'],dst_position_embeddings*position_embedding_feature[node_type]],dim=1).unsqueeze(0).unsqueeze(0).unsqueeze(0))
                else:
                    space_2 = self.linears[node_type](g.nodes[node_type].data[f't{t}'].unsqueeze(0).unsqueeze(0).unsqueeze(0))
                out_space = torch.cat([space_1,space_2],dim=1)
                out_space = rearrange(out_space, 'b s t ... -> b (s t) ...')
                out, states, alpha = self.readout(out_space)
                g.nodes[node_type].data[f't{self.input_length+t}'] = out[0][0]
        return g


class GHMN(pl.LightningModule):
    def __init__(self, model_args, data_args):
        super().__init__()
        self.save_hyperparameters()  
        self.model_args = model_args
        self.data_args = data_args

        with open(self.data_args['climatology_dir'], 'rb') as f:
            self.climatology = pickle.load(f)

        self.flow_model = FlowMatchingHead(model_args, data_args)
        self.mesh_model=GHMNBackbone(model_args, data_args)

        self.rmse_loss = criterion.RMSE()
        self.mse_loss = criterion.MSE()
        self.mae_loss = criterion.MAE()
        self.input_length=1
        self.output_length=1

        self.num_vars = len(self.data_args['predict_vars'])  
        self.num_percentile_pairs = 4  
        
        with open(self.data_args['percentile_dir'], 'rb') as f:
            self.percentile = pickle.load(f).to(self.device)  
        
        self.sedi_calculator=ProbabilisticMultiMetricsCalculator(num_vars=self.num_vars)
        self.crps_num_samples = 10
        self.test_crps_list = []
        self.test_crps_per_var = {var: [] for var in self.data_args['predict_vars']}
    
    def training_step(self, batch, batch_idx):
        
        graph, timestamp, sh_embedding = batch[0], batch[1], batch[2]
        
        batch_num_nodes_per_var = {
            node_type: graph.batch_num_nodes(node_type)  
            for node_type in self.data_args['predict_vars']
        }

        concatenated_idxs = []
        for node_type in self.data_args['predict_vars']:
            var_batch_indices = []
            for sample_in_batch_idx in range(len(batch_num_nodes_per_var[node_type])):
                num_nodes = batch_num_nodes_per_var[node_type][sample_in_batch_idx]
                var_batch_indices.append(
                    torch.full((num_nodes, 1), sample_in_batch_idx, dtype=torch.long, device=graph.device)
                )
            concatenated_idxs.append(torch.cat(var_batch_indices, dim=0))
        batch_sample_idx = torch.cat(concatenated_idxs, dim=0)

        loss = 0
        concatenated_features = []
        for node_type in self.data_args['predict_vars']:
            features = [graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
            concatenated_feature = torch.stack(features, dim=1)
            concatenated_features.append(concatenated_feature)
        label=torch.concatenate(concatenated_features,dim=0)

        output_graph = self.mesh_model(graph,timestamp,sh_embedding)
        
        output_concatenated_features = []
        for node_type in self.data_args['predict_vars']:
            features = [output_graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
            output_concatenated_feature = torch.stack(features, dim=1)
            output_concatenated_features.append(output_concatenated_feature)
        predict=torch.concatenate(output_concatenated_features,dim=0)

        original_loss=self.mse_loss(predict,label)

        _,noise_loss=self.flow_model.flow_loss(x_0=label, cond_latent=predict,batch_sample_idx=batch_sample_idx)
        loss=original_loss+noise_loss
        self.log(
            "train_loss",
            loss,
            on_step=True, on_epoch=True, prog_bar=True, logger=True,
            batch_size=self.data_args["batch_size"]
        ) 
        return loss

    def validation_step(self, batch, batch_idx):
         
        graph, timestamp, sh_embedding = batch[0], batch[1], batch[2]

        loss = 0
        concatenated_features = []
        for node_type in self.data_args['predict_vars']:
            features = [graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
            concatenated_feature = torch.stack(features, dim=1)
            concatenated_features.append(concatenated_feature)
        label=torch.concatenate(concatenated_features,dim=0)

        output_graph = self.mesh_model(graph,timestamp,sh_embedding)
        
        output_concatenated_features = []
        for node_type in self.data_args['predict_vars']:
            features = [output_graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
            output_concatenated_feature = torch.stack(features, dim=1)
            output_concatenated_features.append(output_concatenated_feature)
        predict=torch.concatenate(output_concatenated_features,dim=0)

        label_predict=self.flow_model.sample(
            cond_latent=predict,num_sample=1
        )
        
        loss=self.mse_loss(label_predict,label)
        self.log(
            "val_diffusion_loss",
            loss,
            on_step=True, on_epoch=True, prog_bar=True, logger=True,
            batch_size=self.data_args["batch_size"]
        )  
        return loss

    def test_step(self, batch, batch_idx):
        graph, timestamp, sh_embedding = batch[0], batch[1], batch[2]

        concatenated_features = []
        var_sample_counts = []  
        for node_type in self.data_args['predict_vars']:
            features = [graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
            concatenated_feature = torch.stack(features, dim=1)  
            concatenated_feature = (concatenated_feature*self.climatology[node_type]['std']) + self.climatology[node_type]['mean']
            concatenated_features.append(concatenated_feature)
            var_sample_counts.append(concatenated_feature.shape[0])  
        
        label = torch.cat(concatenated_features, dim=0)  

        output_graph = self.mesh_model(graph,timestamp,sh_embedding)
        output_concatenated_features = []
        label_predict_features=[]
        for node_type in self.data_args['predict_vars']:
            features = [output_graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
            output_concatenated_feature = torch.stack(features, dim=1)  
            label_predict_features.append((output_concatenated_feature*self.climatology[node_type]['std']) + self.climatology[node_type]['mean'])
            output_concatenated_features.append(output_concatenated_feature)
        predict = torch.cat(output_concatenated_features, dim=0)  
        label_predict = torch.cat(label_predict_features, dim=0)

        crps_pred_samples_norm = self.flow_model.sample(
            cond_latent=predict, 
            num_sample=self.crps_num_samples, 
            return_all_samples=True
        )
        
        crps_pred_samples = []
        flag = 0
        for i, node_type in enumerate(self.data_args['predict_vars']):
            var_sample_count = var_sample_counts[i]
            var_samples_norm = crps_pred_samples_norm[:, flag:flag+var_sample_count, :]
            var_samples = (var_samples_norm * self.climatology[node_type]['std']) + self.climatology[node_type]['mean']
            crps_pred_samples.append(var_samples)
            flag += var_sample_count
        crps_pred_samples = torch.cat(crps_pred_samples, dim=1) 

        loss=self.mse_loss(label_predict,label) 
        assert not torch.isnan(loss).any()

        flag=0
        for i,node_type in enumerate(self.data_args['predict_vars']):
            for day in range(self.output_length):
                each_loss=self.mse_loss(label_predict[flag:flag+concatenated_features[i].shape[0],day],label[flag:flag+concatenated_features[i].shape[0],day])
                each_loss_mae=self.mae_loss(label_predict[flag:flag+concatenated_features[i].shape[0],day],label[flag:flag+concatenated_features[i].shape[0],day])
                each_loss_rmse=self.rmse_loss(label_predict[flag:flag+concatenated_features[i].shape[0],day],label[flag:flag+concatenated_features[i].shape[0],day])
                assert not torch.isnan(each_loss).any()
                self.log(f"{node_type} in day {day+1} rmse",each_loss_rmse, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
                self.log(f"{node_type} in day {day+1} mse",each_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
                self.log(f"{node_type} in day {day+1} mae",each_loss_mae, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
            flag=flag+concatenated_features[i].shape[0]

        flow_overall_mse = self.mse_loss(label_predict, label)
        flow_overall_mae = self.mae_loss(label_predict, label)
        flow_overall_rmse = self.rmse_loss(label_predict, label)
        
        self.log("overall_mse", flow_overall_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.data_args['batch_size'])
        self.log("overall_mae", flow_overall_mae, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.data_args['batch_size'])
        self.log("overall_rmse", flow_overall_rmse, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.data_args['batch_size'])
        
        crps_pred_samples, _ = torch.sort(crps_pred_samples, dim=0)  

        batch_percentile = self.percentile.to(label.device)
        self.sedi_calculator.update(
            predicted_samples=crps_pred_samples,  
            true_values=label,               
            percentile=batch_percentile,
            var_sample_counts=var_sample_counts  
        )

        true_values = label  
        crps = crps_from_samples(crps_pred_samples, true_values, dim=0)  
        overall_crps = crps.mean()  
        self.test_crps_list.append(overall_crps.item())
        
        flag = 0
        for i, node_type in enumerate(self.data_args['predict_vars']):
            var_sample_count = var_sample_counts[i]
            var_crps = crps[flag:flag+var_sample_count, :]  
            for day in range(self.output_length):
                var_day_crps = var_crps[:, day].mean()  
                self.log(f"{node_type} in day {day+1} CRPS", var_day_crps, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.data_args['batch_size'])
                self.test_crps_per_var[node_type].append(var_day_crps.item())
            flag += var_sample_count
        
        self.log("overall_CRPS", overall_crps, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.data_args['batch_size'])
        return loss

    def on_test_epoch_end(self):
        SEDI = self.sedi_calculator.get_metrics()  
        var_names = self.data_args['predict_vars']
        percentile_values = [0.5,2,3,10,90,97,98,99.5]  
        percentile_pairs = [
            f"({percentile_values[i]}%,{percentile_values[7-i]}%)" 
            for i in range(4)
        ] 
        
        for pair_idx, pair_name in enumerate(percentile_pairs):
            for var_idx, var_name in enumerate(var_names):
                sedi_value = SEDI[pair_idx, var_idx]
                self.log(f"SEDI_{pair_name}_{var_name}", sedi_value, on_epoch=True, logger=True)
        
        for pair_idx, pair_name in enumerate(percentile_pairs):
            pair_avg_sedi = np.mean(SEDI[pair_idx, :])
            self.log(f"SEDI_{pair_name}_avg", pair_avg_sedi, on_epoch=True, logger=True)
        
        global_avg_sedi = np.mean(SEDI)
        self.log("global_avg_SEDI", global_avg_sedi, on_epoch=True, logger=True)

        epoch_overall_crps = np.mean(self.test_crps_list)
        self.log("epoch_overall_CRPS", epoch_overall_crps, on_epoch=True, logger=True)
        
        for var_name in self.data_args['predict_vars']:
            var_avg_crps = np.mean(self.test_crps_per_var[var_name])
            self.log(f"epoch_{var_name}_CRPS", var_avg_crps, on_epoch=True, logger=True)
        
        self.test_crps_list.clear()
        for var_name in self.data_args['predict_vars']:
            self.test_crps_per_var[var_name].clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.model_args["learning_rate"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.model_args["t_max"],
            eta_min=self.model_args["learning_rate"] / 10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def setup(self, stage=None):
        self.train_dataset = TemporalHeterogeneousDataset(
            data_dir=self.data_args["data_dir"],
            years=self.data_args["train_years"],
            input_length=self.data_args["input_length"],
            output_length=self.data_args["output_length"],
            climatology=self.climatology
        )
        self.val_dataset = TemporalHeterogeneousDataset(
            data_dir=self.data_args["data_dir"],
            years=self.data_args["val_years"],
            input_length=self.data_args["input_length"],
            output_length=self.data_args["output_length"],
            climatology=self.climatology
        )
        self.test_dataset = TemporalHeterogeneousDataset(
            data_dir=self.data_args["data_dir"],
            years=self.data_args["test_years"],
            input_length=self.data_args["input_length"],
            output_length=self.data_args["output_length"],
            climatology=self.climatology
        )

    def train_dataloader(self):
        return GraphDataLoader(
            self.train_dataset,
            batch_size=self.data_args["batch_size"],
            shuffle=True,
            num_workers=self.model_args["num_workers"],
            collate_fn=custom_collate_fn  
        )

    def val_dataloader(self):
        return GraphDataLoader(
            self.val_dataset,
            batch_size=self.data_args["batch_size"],
            shuffle=False,
            num_workers=self.model_args["num_workers"],
            collate_fn=custom_collate_fn  
        )

    def test_dataloader(self):
        return GraphDataLoader(
            self.test_dataset,
            batch_size=self.data_args["batch_size"],
            shuffle=False,
            num_workers=self.model_args["num_workers"],
            collate_fn=custom_collate_fn  
        )
    
    def predict_dataloader(self):
        return GraphDataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.model_args["num_workers"],
            collate_fn=custom_collate_fn
        )

def custom_collate_fn(batch):
    """
    Collate function for handling heterogeneous graphs, temporal embeddings, 
    spherical harmonics, and variable-specific sample indices in a single batch.
    
    Args:
        batch (list): A list of tuples from Dataset.__getitem__, where each tuple 
                      contains (graph, time_emb, sh_emb, sample_idx_per_var).
    
    Returns:
        tuple: (batched_graph, batched_time, batched_sh, batched_sample_idx_per_var)
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