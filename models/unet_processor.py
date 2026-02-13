import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
import healpy as hp
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from typing import Dict, List, Tuple, Optional, Union


# --------------------------
# 1. 类型定义与常量
# --------------------------
Tensor = torch.Tensor
SparseTensor = torch.Tensor  
HealpixHierarchy = Dict[int, Dict[str, Union[int, Tensor, None]]]
AdjMatrices = Dict[int, Tensor]


# --------------------------
# 2. 工具函数：图拉普拉斯矩阵计算
# --------------------------
def calculate_normalized_laplacian(adj: np.ndarray) -> sp.coo_matrix:
    """计算归一化拉普拉斯矩阵 L = D^-1/2 (D-A) D^-1/2"""
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5, where=d != 0)  
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)).tocoo()


def calculate_random_walk_matrix(adj_mx: np.ndarray) -> sp.coo_matrix:
    """计算随机游走矩阵 D^-1 A"""
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1)).flatten()
    d_inv = np.power(d, -1, where=d != 0)  
    d_inv[np.isinf(d_inv)] = 0.0
    return sp.diags(d_inv).dot(adj_mx).tocoo()


def calculate_scaled_laplacian(adj_mx: np.ndarray, lambda_max: Optional[float] = 2, 
                             undirected: bool = True) -> sp.coo_matrix:
    """计算缩放拉普拉斯矩阵 (2/Lambda_max * L) - I"""
    if undirected:
        adj_mx = np.maximum(adj_mx, adj_mx.T)
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max = eigsh(L, 1, which='LM')[0][0]  
    I = sp.identity(L.shape[0], format='coo', dtype=L.dtype)
    return ((2 / lambda_max) * L - I).tocoo()


def sparse_matrix_to_torch(mat: sp.coo_matrix, device: torch.device) -> SparseTensor:
    """将scipy稀疏矩阵转换为PyTorch稀疏张量"""
    values = mat.data
    indices = np.vstack((mat.row, mat.col))
    i = torch.LongTensor(indices).to(device)
    v = torch.FloatTensor(values).to(device)
    return torch.sparse_coo_tensor(i, v, torch.Size(mat.shape), requires_grad=False)


# --------------------------
# 3. 自定义扩散图卷积层
# --------------------------
class DiffusionGraphConv(nn.Module):
    def __init__(self, supports: List[SparseTensor], input_dim: int, output_dim: int, 
                 num_nodes: int, max_diffusion_step: int = 2, bias_start: float = 0.0):
        super().__init__()
        self.num_matrices = len(supports) * max_diffusion_step + 1  
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._supports = supports  
        
        self.weight = nn.Parameter(torch.FloatTensor(input_dim * self.num_matrices, output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(output_dim))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播：支持批次输入的扩散图卷积
        Args:
            x: 输入特征，形状 (B, N, D) 其中 B=批次, N=节点数, D=特征维度
        Returns:
            输出特征，形状 (B, N, output_dim)
        """
        batch_size, num_nodes, input_dim = x.shape
        x0 = rearrange(x, "b n d -> n (b d)")  
        x = torch.unsqueeze(x0, dim=0)  

        if self._max_diffusion_step > 0:
            for support in self._supports:
                support = support.to(x.device)
                x1 = torch.sparse.mm(support, x0)  
                x = torch.cat([x, torch.unsqueeze(x1, 0)], dim=0)
                
                x_prev, x_curr = x0, x1
                for k in range(2, self._max_diffusion_step + 1):
                    x_next = 2 * torch.sparse.mm(support, x_curr) - x_prev
                    x = torch.cat([x, torch.unsqueeze(x_next, 0)], dim=0)
                    x_prev, x_curr = x_curr, x_next

        x = rearrange(x, "k n (b d) -> b n d k", b=batch_size, d=input_dim)
        x = rearrange(x, "b n d k -> (b n) (d k)")
        x = torch.matmul(x, self.weight) + self.biases
        return rearrange(x, "(b n) d -> b n d", b=batch_size, n=num_nodes)

# --------------------------
# 4. HEALPix采样器
# --------------------------
class HEALPix_Sampler(nn.Module):
    def __init__(self, hierarchy: HealpixHierarchy, hidden_sizes: Dict[int, int]):
        super().__init__()
        self.hierarchy = hierarchy
        self.base_level = min(hierarchy.keys())
        self.max_level = max(hierarchy.keys())
        self.levels = sorted(hierarchy.keys())
        self.hidden_sizes = hidden_sizes

        self.down_convs = nn.ModuleDict()
        self.up_convs = nn.ModuleDict()

        # 下采样卷积层（高→低）
        for i in range(1, len(self.levels)):
            current_level = self.levels[i]
            target_level = self.levels[i-1]
            in_dim = self.hidden_sizes[current_level]
            out_dim = self.hidden_sizes[current_level]
            self.down_convs[f"{current_level}→{target_level}"] = nn.Conv1d(
                in_channels=in_dim, out_channels=out_dim, kernel_size=1
            )

        # 上采样卷积层（低→高）
        for i in range(len(self.levels)-1):
            current_level = self.levels[i]
            target_level = self.levels[i+1]
            in_dim = self.hidden_sizes[target_level]
            out_dim = self.hidden_sizes[target_level]
            self.up_convs[f"{current_level}→{target_level}"] = nn.Conv1d(
                in_channels=in_dim, out_channels=out_dim, kernel_size=1
            )

    def downsample(self, x: Tensor, current_level: int, target_level: int) -> Tensor:
        """高分辨率→低分辨率（均值聚合+卷积）"""
        assert current_level > target_level and target_level >= self.base_level, "层级关系错误"
        current_data = self.hierarchy[current_level]
        target_data = self.hierarchy[target_level]
        parent_map = current_data["parent_map"]
        B, C, T, _, D = x.shape
        target_npix = target_data["npix"]
        
        # 重塑与聚合
        x_reshaped = rearrange(x, "b c t n d -> (b c t) n d")  # (BCT, N, D)
        batch_size = x_reshaped.shape[0]

        # 高效聚合
        parent_indices = repeat(parent_map, 'n -> b n', b=batch_size).to(x.device)
        x_down_reshaped = torch.zeros((batch_size, target_npix, D), device=x.device, dtype=x.dtype)
        x_down_reshaped.scatter_add_(
            dim=1,
            index=parent_indices.unsqueeze(-1).expand(-1, -1, D),
            src=x_reshaped
        )
        
        # 均值计算
        count = torch.bincount(parent_map, minlength=target_npix).to(x.device)
        count = count.unsqueeze(0).unsqueeze(-1)
        x_down_reshaped = x_down_reshaped / count.clamp(min=1e-6)

        # 层级卷积
        conv_key = f"{current_level}→{target_level}"
        conv_input = x_down_reshaped.transpose(1, 2)  # (BCT, D, target_npix)
        conv_output = self.down_convs[conv_key](conv_input).transpose(1, 2)
        return rearrange(conv_output, "(b c t) n d -> b c t n d", b=B, c=C, t=T)

    def upsample(self, x: Tensor, current_level: int, target_level: int, mode: str = "nearest") -> Tensor:
        """低分辨率→高分辨率（插值+卷积）"""
        assert current_level < target_level and target_level <= self.max_level, "层级关系错误"
        x_up = x
        device = x.device

        # 逐层上采样
        for level in range(current_level, target_level):
            next_level = level + 1
            current_data = self.hierarchy[level]
            next_data = self.hierarchy[next_level]
            child_map = current_data["child_map"]
            B, C, T, N, D = x_up.shape
            next_npix = next_data["npix"]

            # 重塑输入
            x_reshaped = rearrange(x_up, "b c t n d -> (b c t) n d")  # (BCT, N, D)
            batch_size = x_reshaped.shape[0]

            # 子像素索引
            child_indices = repeat(child_map, 'n k -> b n k', b=batch_size).to(device)
            x_next_reshaped = torch.zeros((batch_size, next_npix, D), device=device, dtype=x.dtype)

            # 插值
            if mode == "nearest":
                index = child_indices.unsqueeze(-1).expand(-1, -1, -1, D)
                index = rearrange(index, 'b n k d -> b (n k) d')
                src = repeat(x_reshaped, 'b n d -> b (n k) d', k=4)
                x_next_reshaped.scatter_(dim=1, index=index, src=src)

            elif mode == "linear":
                # 球面距离权重计算
                parent_lat = current_data["lat"].to(device).squeeze()
                parent_lon = current_data["lon"].to(device).squeeze()
                child_lat = next_data["lat"][child_map].to(device).squeeze()
                child_lon = next_data["lon"][child_map].to(device).squeeze()

                # 哈弗辛公式计算距离
                lon1, lat1 = map(torch.deg2rad, [parent_lon, parent_lat])
                lon2, lat2 = map(torch.deg2rad, [child_lon, child_lat])
                dlon = lon2 - lon1.unsqueeze(1)
                dlat = lat2 - lat1.unsqueeze(1)
                a = torch.sin(dlat/2)**2 + torch.cos(lat1.unsqueeze(1)) * torch.cos(lat2) * torch.sin(dlon/2)** 2
                dist = 2 * torch.arcsin(torch.sqrt(a))
                weight = 1.0 / (dist + 1e-6)
                weight = weight / weight.sum(dim=1, keepdim=True)

                # 加权插值
                src = x_reshaped.unsqueeze(2) * weight.unsqueeze(-1)
                src = rearrange(src, 'b n k d -> b (n k) d')
                index = child_indices.unsqueeze(-1).expand(-1, -1, -1, D)
                index = rearrange(index, 'b n k d -> b (n k) d')
                x_next_reshaped.scatter_(dim=1, index=index, src=src)

            else:
                raise ValueError(f"不支持的上采样模式：{mode}")

            # 层级卷积
            conv_key = f"{level}→{next_level}"
            conv_input = x_next_reshaped.transpose(1, 2)
            conv_output = self.up_convs[conv_key](conv_input).transpose(1, 2)
            x_up = rearrange(conv_output, "(b c t) n d -> b c t n d", b=B, c=C, t=T)

        return x_up


# --------------------------
# 5. HEALPix + U-Net 处理器
# --------------------------
class HEALPix_UNet_Processor(nn.Module):
    def __init__(self, model_args: Dict, data_args: Dict, adj_matrices: AdjMatrices, 
                 hierarchy: HealpixHierarchy, filter_type: str = "laplacian", 
                 max_diffusion_step: int = 2):
        super().__init__()
        self.model_args = model_args
        self.data_args = data_args
        self.hierarchy = hierarchy
        self.filter_type = filter_type
        self.max_diffusion_step = max_diffusion_step

        self.hidden_size = model_args["hidden_size"]
        self.input_length = data_args["input_length"]
        self.output_length = data_args["output_length"]
        self.num_vars = len(data_args["vars"])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.unet_levels = sorted(hierarchy.keys(), reverse=True)
        self.down_levels = self.unet_levels[:-1]
        self.up_levels = sorted(self.unet_levels[1:])
        self.bottleneck_level = self.unet_levels[-1]

        self.hidden_size = model_args["hidden_size"]  
        self.hidden_sizes = {
            3: self.hidden_size * 8,   
            4: self.hidden_size * 4,   
            5: self.hidden_size * 2    
        }

        self.supports = self._precompute_supports(adj_matrices)

        self.hp_sampler = HEALPix_Sampler(hierarchy, self.hidden_sizes)

        self.time_proj_encoder = nn.ModuleDict()
        self.time_proj_bottleneck = nn.Sequential(
            nn.Linear(self.input_length, self.output_length),
            nn.ReLU()
        )
        self.time_proj_decoder = nn.ModuleDict()

        self.encoder_blocks = nn.ModuleDict()
        level_hidden_size = self.hidden_size
        for level in self.down_levels:
            npix = hierarchy[level]["npix"]
            self.time_proj_encoder[f"level_{level}"] = nn.Sequential(
                nn.Linear(self.input_length, self.output_length),
                nn.ReLU()
            )
            self.encoder_blocks[f"level_{level}"] = nn.ModuleDict({
                "graph_conv": DiffusionGraphConv(
                    supports=self.supports[level],
                    input_dim=level_hidden_size,
                    output_dim=level_hidden_size * 2,
                    num_nodes=npix,
                    max_diffusion_step=max_diffusion_step
                ),
                "var_attn": nn.MultiheadAttention(
                    embed_dim=level_hidden_size * 2,
                    num_heads=model_args["head"],
                    batch_first=True
                ),
                "residual_adjust": nn.Linear(level_hidden_size, level_hidden_size * 2)
            })
            level_hidden_size *= 2

        bottleneck_npix = hierarchy[self.bottleneck_level]["npix"]
        self.bottleneck_block = nn.ModuleDict({
            "graph_conv": DiffusionGraphConv(
                supports=self.supports[self.bottleneck_level],
                input_dim=level_hidden_size,
                output_dim=level_hidden_size,
                num_nodes=bottleneck_npix,
                max_diffusion_step=max_diffusion_step
            ),
            "var_attn": nn.MultiheadAttention(
                embed_dim=level_hidden_size,
                num_heads=model_args["head"],
                batch_first=True
            )
        })

        self.decoder_blocks = nn.ModuleDict()
        for level in self.up_levels:
            npix = hierarchy[level+1]["npix"]

            self.time_proj_decoder[f"level_{level}"] = nn.Sequential(
                nn.Linear(self.input_length, self.output_length),
                nn.ReLU()
            )
            self.decoder_blocks[f"level_{level}"] = nn.ModuleDict({
                "feat_fuse": nn.Linear(level_hidden_size * 2, level_hidden_size),
                "graph_conv": DiffusionGraphConv(
                    supports=self.supports[level+1],
                    input_dim=level_hidden_size,
                    output_dim=level_hidden_size // 2,
                    num_nodes=npix,
                    max_diffusion_step=max_diffusion_step
                ),
                "var_attn": nn.MultiheadAttention(
                    embed_dim=level_hidden_size // 2,
                    num_heads=model_args["head"],
                    batch_first=True
                ),
                "residual_adjust": nn.Linear(level_hidden_size, level_hidden_size // 2)
            })
            level_hidden_size //= 2

    def _precompute_supports(self, adj_matrices: AdjMatrices) -> Dict[int, List[SparseTensor]]:
        """为每个层级预计算图支持矩阵"""
        supports = {}
        for level in adj_matrices:
            adj = adj_matrices[level].cpu().numpy()
            if self.filter_type == "laplacian":
                support = calculate_scaled_laplacian(adj, lambda_max=None)
                supports[level] = [sparse_matrix_to_torch(support, self.device)]
            elif self.filter_type == "random_walk":
                support = calculate_random_walk_matrix(adj).T
                supports[level] = [sparse_matrix_to_torch(support, self.device)]
            elif self.filter_type == "dual_random_walk":
                support1 = calculate_random_walk_matrix(adj)
                support2 = calculate_random_walk_matrix(adj.T)
                supports[level] = [
                    sparse_matrix_to_torch(support1, self.device),
                    sparse_matrix_to_torch(support2, self.device)
                ]
            else:
                raise ValueError(f"不支持的过滤类型: {self.filter_type}")
        return supports

    def encoder_forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """编码器前向传播"""
        skip_features = []
        current_x = x
        current_level = self.unet_levels[0]

        for level in self.down_levels:
            block = self.encoder_blocks[f"level_{level}"]
            time_proj = self.time_proj_encoder[f"level_{level}"]
            next_level = self.unet_levels[self.unet_levels.index(level) + 1]

            # 图卷积
            B, C, T, N, D = current_x.shape
            conv_input = rearrange(current_x, "b c t n d -> (b c t) n d")
            conv_output = block["graph_conv"](conv_input)
            conv_output = rearrange(conv_output, "(b c t) n d -> b c t n d", b=B, c=C, t=T)

            # 变量注意力
            attn_input = rearrange(conv_output, "b c t n d -> (b t n) c d")
            attn_output, _ = block["var_attn"](attn_input, attn_input, attn_input)
            attn_output = rearrange(attn_output, "(b t n) c d -> b c t n d", b=B, t=T, n=N)

            # 时间维度映射（新增）
            time_input = rearrange(attn_output, "b c t n d -> b c n d t")
            time_output = time_proj(time_input)
            attn_output = rearrange(time_output, "b c n d t -> b c t n d")

            # 残差连接
            residual = block["residual_adjust"](current_x)
            attn_output = attn_output + residual
            skip_features.append(attn_output)

            # 下采样
            current_x = self.hp_sampler.downsample(attn_output, current_level, next_level)
            current_level = next_level  

        # 瓶颈层处理    
        B, C, T, N, D = current_x.shape
        conv_input = rearrange(current_x, "b c t n d -> (b c t) n d")
        conv_output = self.bottleneck_block["graph_conv"](conv_input)
        conv_output = rearrange(conv_output, "(b c t) n d -> b c t n d", b=B, c=C, t=T)

        attn_input = rearrange(conv_output, "b c t n d -> (b t n) c d")
        attn_output, _ = self.bottleneck_block["var_attn"](attn_input, attn_input, attn_input)
        attn_output = rearrange(attn_output, "(b t n) c d -> b c t n d", b=B, t=T, n=N)

        # 瓶颈层时间维度映射
        time_input = rearrange(attn_output, "b c t n d -> b c n d t")
        time_output = self.time_proj_bottleneck(time_input)
        bottleneck_output = rearrange(time_output, "b c n d t -> b c t n d")
        
        return bottleneck_output, skip_features

    def decoder_forward(self, x: Tensor, skip_features: List[Tensor]) -> Tensor:
        """解码器前向传播"""
        current_x = x
        current_level = self.bottleneck_level

        for i, level in enumerate(self.up_levels):
            block = self.decoder_blocks[f"level_{level}"]
            time_proj = self.time_proj_decoder[f"level_{level}"]
            skip_x = skip_features[-(i + 1)]
            prev_level = self.unet_levels[self.unet_levels.index(level) - 1]

            # 上采样与特征融合
            current_x = self.hp_sampler.upsample(current_x, current_level, prev_level, mode="linear")
            current_level = prev_level
            fused_x = torch.cat([current_x, skip_x], dim=-1)
            fused_x = block["feat_fuse"](fused_x)

            # 图卷积与注意力
            B, C, T, N, D = fused_x.shape
            conv_input = rearrange(fused_x, "b c t n d -> (b c t) n d")
            conv_output = block["graph_conv"](conv_input)
            conv_output = rearrange(conv_output, "(b c t) n d -> b c t n d", b=B, c=C, t=T)

            attn_input = rearrange(conv_output, "b c t n d -> (b t n) c d")
            attn_output, _ = block["var_attn"](attn_input, attn_input, attn_input)
            attn_output = rearrange(attn_output, "(b t n) c d -> b c t n d", b=B, t=T, n=N)

            # 时间维度映射
            time_input = rearrange(attn_output, "b c t n d -> b c n d t")
            time_output = time_proj(time_input)
            attn_output = rearrange(time_output, "b c n d t -> b c t n d")

            # 残差连接
            residual = block["residual_adjust"](fused_x)
            current_x = attn_output + residual

        return current_x

    def forward(self, x: Tensor) -> Tensor:
        """
        输入：(B, C, T, N, D)  B=批次, C=变量数, T=时间步, N=像素数, D=特征维度
        输出：(B, T_out, C, N, D)
        """
        # 编码器
        bottleneck_x, skip_features = self.encoder_forward(x)
        
        # 解码器
        decoder_x = self.decoder_forward(bottleneck_x, skip_features)
        
        # 最终输出形状调整
        return rearrange(decoder_x, "b c t n d -> b t c n d")


# --------------------------
# 6. HEALPix辅助数据生成
# --------------------------
def haversine(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """计算球面两点间距离（哈弗辛公式）"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1[:, None]
    dlat = lat2 - lat1[:, None]
    a = np.sin(dlat / 2)** 2 + np.cos(lat1[:, None]) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * np.arcsin(np.sqrt(a)) * 6371000 / 1000000  


def get_healpix_hierarchy(refinement_level: int, max_refinement_level: Optional[int] = None) -> HealpixHierarchy:
    """生成HEALPix层级结构数据"""
    if max_refinement_level is None:
        max_refinement_level = refinement_level + 2
    hierarchy = {}
    for level in range(refinement_level, max_refinement_level + 1):
        nside = 2 ** level
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix), nest=True)
        lat = 90 - np.degrees(theta)  
        lon = np.degrees(phi) - 180   

        # 父子映射
        child_map = None
        if level < max_refinement_level:
            child_nside = 2 **(level + 1)
            child_map = np.zeros((npix, 4), dtype=int)
            for parent_idx in range(npix):
                child_indices = hp.nest2ring(
                    child_nside, 
                    hp.ring2nest(nside, parent_idx) * 4 + np.arange(4)
                )
                child_map[parent_idx] = child_indices

        # 子父映射
        parent_map = None
        if level > refinement_level:
            parent_nside = 2** (level - 1)
            parent_map = np.zeros(npix, dtype=int)
            for child_idx in range(npix):
                parent_idx = hp.nest2ring(
                    parent_nside, 
                    hp.ring2nest(nside, child_idx) // 4
                )
                parent_map[child_idx] = parent_idx
        
        hierarchy[level] = {
            "nside": nside,
            "npix": npix,
            "lat": torch.tensor(lat, dtype=torch.float32).unsqueeze(1),
            "lon": torch.tensor(lon, dtype=torch.float32).unsqueeze(1),
            "child_map": torch.tensor(child_map, dtype=torch.long) if child_map is not None else None,
            "parent_map": torch.tensor(parent_map, dtype=torch.long) if parent_map is not None else None
        }
    return hierarchy


def get_healpix_auxiliary_data(refinement_level: int, num_neighbors: int, 
                              max_refinement_level: Optional[int] = None) -> Tuple[AdjMatrices, HealpixHierarchy]:
    """生成HEALPix邻接矩阵和层级数据"""
    hierarchy = get_healpix_hierarchy(refinement_level, max_refinement_level)
    adj_matrices = {}
    for level in hierarchy:
        data = hierarchy[level]
        npix = data["npix"]
        lon = data["lon"].numpy().flatten()
        lat = data["lat"].numpy().flatten()
        distances = haversine(lon, lat, lon, lat)
        adj = np.zeros((npix, npix), dtype=np.float32)
        nearest_indices = np.argsort(distances, axis=1)[:, :num_neighbors + 1]  
        for i in range(npix):
            adj[i, nearest_indices[i]] = 1.0
        adj_matrices[level] = torch.tensor(adj)
    return adj_matrices, hierarchy


def init_mesh_model(model_args: Dict, data_args: Dict) -> HEALPix_UNet_Processor:
    """初始化模型，自动缓存HEALPix辅助数据"""
    cache_filename = (
        f"healpix_cache_level_{data_args['healpix_base_level']}_"
        f"max_{data_args['healpix_max_level']}_"
        f"neighbors_{data_args['healpix_num_neighbors']}.pkl"
    )
    cache_dir = os.path.join(data_args.get('cache_dir', 'healpix_cache'), '')
    cache_path = cache_dir + cache_filename

    if os.path.exists(cache_path):
        print(f"加载HEALPix缓存: {cache_path}")
        with open(cache_path, 'rb') as f:
            adj_matrices, hierarchy = pickle.load(f)
    else:
        print(f"生成HEALPix数据并缓存: {cache_path}")
        adj_matrices, hierarchy = get_healpix_auxiliary_data(
            refinement_level=data_args["healpix_base_level"],
            num_neighbors=data_args["healpix_num_neighbors"],
            max_refinement_level=data_args["healpix_max_level"]
        )
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump((adj_matrices, hierarchy), f, protocol=pickle.HIGHEST_PROTOCOL)

    return HEALPix_UNet_Processor(
        model_args=model_args,
        data_args=data_args,
        adj_matrices=adj_matrices,
        hierarchy=hierarchy,
        filter_type="laplacian",
        max_diffusion_step=2
    )


if __name__ == "__main__":
    # 配置参数
    model_args = {"hidden_size": 64, "head": 4}
    data_args = {
        "input_length": 1,
        "output_length": 1,
        "vars": ["MAX", "MIN", "SLP", "WDSP", "MXSPD", "DEWP"],
        "healpix_base_level": 3,
        "healpix_max_level": 5,
        "healpix_num_neighbors": 5,
        "cache_dir": "daily_processed_dataset"
    }

    # 初始化模型
    model = init_mesh_model(model_args, data_args)
    model = model.to(model.device)
    print(f"使用设备: {model.device}")

    # 测试前向传播
    dummy_x = torch.randn(2, 6, 1, 12288, 64).to(model.device)
    dummy_out = model(dummy_x)
    print(f"输入形状: {dummy_x.shape}")
    print(f"输出形状: {dummy_out.shape}")