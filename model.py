import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn import Parameter
import numpy as np

# Try importing torch_scatter; provide a fallback if unavailable
try:
    from torch_scatter import scatter_softmax, scatter_add
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False
    print("Warning: torch_scatter not found. Using fallback implementation (slower).")
    print("To install: pip install torch-scatter")


class SimplicialMessagePassing(nn.Module):
    """
    Simplicial-aware message passing layer (optional batched triangle processing).

    Three steps:
    1. Step1: Pairwise stream — attention over 1-hop neighbors.
    2. Step2: Higher-order stream — messages from triangles (2-simplices) to nodes (batched optional).
    3. Step3: Multimodal fusion — combine self, local, and higher-order signals.

    Args:
        in_dim (int): Input node feature dimension.
        out_dim (int): Output node feature dimension.
        triangle_dim (int): Triangle feature dimension; default None means same as in_dim.
        lambda_high (float): Strength of higher-order injection; default 1.0.
        dropout (float): Dropout rate; default 0.0.
        negative_slope (float): LeakyReLU negative slope; default 0.2.
        batch_size_triangles (int, optional): Node batch size for triangle aggregation (GPU memory).
            If None, no batching. Typical range: 1000–5000.
    """
    
    def __init__(self, in_dim, out_dim, triangle_dim=None, lambda_high=1.0, dropout=0.0, 
                 negative_slope=0.2, batch_size_triangles=None, batch_size_edges=None):
        super(SimplicialMessagePassing, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.triangle_dim = triangle_dim if triangle_dim is not None else in_dim
        self.lambda_high = lambda_high
        self.negative_slope = negative_slope
        self.batch_size_triangles = batch_size_triangles
        self.batch_size_edges = batch_size_edges
        
        # Step1: Pairwise stream parameters
        # W_1: pairwise feature transform
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        # a_1: pairwise attention vector
        self.a1 = Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        
        # Step2: Higher-order stream parameters
        # W_2: higher-order transform (input dim = triangle feature dim)
        self.W2 = nn.Linear(self.triangle_dim, out_dim, bias=False)
        # a_2: higher-order attention vector
        self.a2 = Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        
        # Step3: Fusion parameters
        # W_0: self-loop transform
        self.W0 = nn.Linear(in_dim, out_dim, bias=True)
        # Learnable gate: w_pair * m_pairwise + w_high * lambda * m_high_order
        self.gate_linear = nn.Linear(3 * out_dim, 2)
        # Residual projection when in_dim != out_dim
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize network parameters."""
        nn.init.xavier_uniform_(self.W1.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.W2.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.W0.weight.data, gain=1.414)
        if self.W0.bias is not None:
            nn.init.zeros_(self.W0.bias.data)
        if hasattr(self.gate_linear, 'weight'):
            nn.init.xavier_uniform_(self.gate_linear.weight.data, gain=0.5)
            if self.gate_linear.bias is not None:
                # Bias init so gate is near [0.5, 0.5]
                nn.init.zeros_(self.gate_linear.bias.data)
        if hasattr(self.residual_proj, 'weight'):
            nn.init.xavier_uniform_(self.residual_proj.weight.data, gain=1.0)
            if self.residual_proj.bias is not None:
                nn.init.zeros_(self.residual_proj.bias.data)
    
    def forward(self, h, edge_index, triangles_dict, h_triangles):
        """
        Forward pass.

        Args:
            h (torch.Tensor): Node features, shape [num_nodes, in_dim].
            edge_index (torch.Tensor): Edge indices, shape [num_edges, 2].
            triangles_dict (dict): Node -> list of triangle indices containing that node.
            h_triangles (torch.Tensor): Triangle features, shape [num_triangles, in_dim].

        Returns:
            torch.Tensor: Updated node features, shape [num_nodes, out_dim].
        """
        num_nodes = h.shape[0]
        device = h.device
        
        # Step1: Pairwise stream — aggregate 1-hop neighbors (vectorized)
        h_transformed = self.W1(h)  # [num_nodes, out_dim]
        
        # Edge endpoints; edge_index is [num_edges, 2]
        row, col = edge_index[:, 0].contiguous(), edge_index[:, 1].contiguous()  # [num_edges]
        
        # Use all edges (assume edge_index already has both directions for undirected graphs)
        row_all, col_all = row, col  # [num_edges]
        
        a_left, a_right = self.a1[:self.out_dim], self.a1[self.out_dim:]
        edge_chunk_size = max(1, int(32_000_000 // max(1, self.out_dim)))
        
        # Grouped softmax over edges by target node
        if HAS_TORCH_SCATTER:
            m_pairwise = torch.zeros(num_nodes, self.out_dim, dtype=h_transformed.dtype, device=device)
            if self.batch_size_edges is not None and self.batch_size_edges > 0:
                for node_start in range(0, num_nodes, self.batch_size_edges):
                    node_end = min(node_start + self.batch_size_edges, num_nodes)
                    left = torch.searchsorted(
                        row_all,
                        torch.tensor(node_start, dtype=row_all.dtype, device=device),
                        right=False,
                    )
                    right = torch.searchsorted(
                        row_all,
                        torch.tensor(node_end - 1, dtype=row_all.dtype, device=device),
                        right=True,
                    )
                    if right <= left:
                        continue
                    batch_row = row_all[left:right]
                    batch_col = col_all[left:right]
                    attention_scores_batch = (
                        torch.matmul(h_transformed[batch_row], a_left)
                        + torch.matmul(h_transformed[batch_col], a_right)
                    )
                    attention_scores_batch = F.leaky_relu(
                        attention_scores_batch, negative_slope=self.negative_slope
                    ).squeeze(1)
                    attention_weights_batch = scatter_softmax(attention_scores_batch, batch_row, dim=0)
                    weighted_features = attention_weights_batch.unsqueeze(1) * h_transformed[batch_col]
                    m_pairwise.index_add_(0, batch_row, weighted_features)
                    del batch_row, batch_col, attention_scores_batch, attention_weights_batch, weighted_features
            else:
                attention_score_chunks = []
                for start in range(0, row_all.numel(), edge_chunk_size):
                    end = min(start + edge_chunk_size, row_all.numel())
                    score_chunk = (
                        torch.matmul(h_transformed[row_all[start:end]], a_left)
                        + torch.matmul(h_transformed[col_all[start:end]], a_right)
                    )
                    attention_score_chunks.append(score_chunk)
                attention_scores_all = torch.cat(attention_score_chunks, dim=0)
                del attention_score_chunks
                attention_scores_all = F.leaky_relu(attention_scores_all, negative_slope=self.negative_slope)
                attention_scores_all = attention_scores_all.squeeze(1)  # [num_edges]
                attention_weights_all = scatter_softmax(attention_scores_all, row_all, dim=0)  # [num_edges]
                for start in range(0, row_all.numel(), edge_chunk_size):
                    end = min(start + edge_chunk_size, row_all.numel())
                    weighted_features = (
                        attention_weights_all[start:end].unsqueeze(1)
                        * h_transformed[col_all[start:end]]
                    )
                    m_pairwise.index_add_(0, row_all[start:end], weighted_features)
                    del weighted_features
        else:
            # Fallback: per-target softmax and sum
            m_pairwise = torch.zeros(num_nodes, self.out_dim, dtype=h_transformed.dtype, device=device)
            unique_nodes = torch.unique(row_all)
            
            for v in unique_nodes:
                mask = (row_all == v)
                if not mask.any():
                    continue
                
                neighbors_idx = col_all[mask]
                scores_v = (
                    torch.matmul(h_transformed[v].unsqueeze(0), a_left)
                    + torch.matmul(h_transformed[neighbors_idx], a_right)
                ).squeeze(1)
                scores_v = F.leaky_relu(scores_v, negative_slope=self.negative_slope)
                neighbors_v = h_transformed[neighbors_idx]  # [num_neighbors_v, out_dim]
                weights_v = F.softmax(scores_v, dim=0)  # [num_neighbors_v]
                m_pairwise[v] = torch.sum(weights_v.unsqueeze(1) * neighbors_v, dim=0)  # [out_dim]
        
        precomputed_pairs = (
            isinstance(triangles_dict, dict)
            and '_node_triangle_pairs' in triangles_dict
        )

        # Step2: Higher-order stream — triangles to nodes (optional batching)
        if self.batch_size_triangles is None or self.batch_size_triangles >= num_nodes:
            # No batching: transform all triangle features first
            if h_triangles.device.type == 'cpu':
                h_triangles_gpu = h_triangles.to(device)
                h_triangles_transformed = self.W2(h_triangles_gpu)  # [num_triangles, out_dim]
                del h_triangles_gpu
            else:
                h_triangles_transformed = self.W2(h_triangles)  # [num_triangles, out_dim]
            if precomputed_pairs:
                node_indices_tensor, triangle_indices_tensor = triangles_dict['_node_triangle_pairs']
                node_indices_tensor = node_indices_tensor.to(device, non_blocking=True)
                triangle_indices_tensor = triangle_indices_tensor.to(device, non_blocking=True)
                valid_pair_mask = triangle_indices_tensor < h_triangles_transformed.shape[0]
                if not valid_pair_mask.all().item():
                    node_indices_tensor = node_indices_tensor[valid_pair_mask]
                    triangle_indices_tensor = triangle_indices_tensor[valid_pair_mask]
            else:
                node_indices = []
                triangle_indices_list = []
                
                for v, tri_indices in triangles_dict.items():
                    if isinstance(v, str):
                        continue
                    for tri_idx in tri_indices:
                        if tri_idx < h_triangles_transformed.shape[0]:
                            node_indices.append(v)
                            triangle_indices_list.append(tri_idx)
                
                if len(node_indices) == 0:
                    node_indices_tensor = None
                    triangle_indices_tensor = None
                else:
                    node_indices_tensor = torch.tensor(node_indices, dtype=torch.long, device=device)
                    triangle_indices_tensor = torch.tensor(triangle_indices_list, dtype=torch.long, device=device)

            if node_indices_tensor is None or node_indices_tensor.numel() == 0:
                m_high_order = torch.zeros(num_nodes, self.out_dim, dtype=h_transformed.dtype, device=device)
            else:
                h_v_all = h_transformed[node_indices_tensor]  # [num_pairs, out_dim]
                h_T_all = h_triangles_transformed[triangle_indices_tensor]  # [num_pairs, out_dim]
                
                concat_feat_all = torch.cat([h_v_all, h_T_all], dim=1)  # [num_pairs, 2*out_dim]
                
                attention_scores_all = torch.matmul(concat_feat_all, self.a2)  # [num_pairs, 1]
                attention_scores_all = F.leaky_relu(attention_scores_all, negative_slope=self.negative_slope)
                attention_scores_all = attention_scores_all.squeeze(1)  # [num_pairs]
                
                if HAS_TORCH_SCATTER:
                    attention_weights_all = scatter_softmax(attention_scores_all, node_indices_tensor, dim=0)  # [num_pairs]
                    weighted_features = attention_weights_all.unsqueeze(1) * h_T_all  # [num_pairs, out_dim]
                    m_high_order = scatter_add(weighted_features, node_indices_tensor, dim=0, dim_size=num_nodes)  # [num_nodes, out_dim]
                else:
                    m_high_order = torch.zeros(num_nodes, self.out_dim, dtype=h_transformed.dtype, device=device)
                    unique_nodes = torch.unique(node_indices_tensor)
                    
                    for v in unique_nodes:
                        mask = (node_indices_tensor == v)
                        if not mask.any():
                            continue
                        
                        scores_v = attention_scores_all[mask]  # [num_triangles_v]
                        triangles_v = h_T_all[mask]  # [num_triangles_v, out_dim]
                        weights_v = F.softmax(scores_v, dim=0)  # [num_triangles_v]
                        m_high_order[v] = torch.sum(weights_v.unsqueeze(1) * triangles_v, dim=0)  # [out_dim]
        else:
            # Batched: aggregate triangle messages per node batch (memory saving)
            m_high_order = torch.zeros(num_nodes, self.out_dim, dtype=h_transformed.dtype, device=device)
            nodes_with_triangles = triangles_dict.get('_nodes_with_triangles') if precomputed_pairs else None
            if nodes_with_triangles is None:
                nodes_with_triangles = [node for node, tris in triangles_dict.items() if not isinstance(node, str) and len(tris) > 0]
            
            if len(nodes_with_triangles) == 0:
                pass
            else:
                num_batches = (len(nodes_with_triangles) + self.batch_size_triangles - 1) // self.batch_size_triangles
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.batch_size_triangles
                    end_idx = min((batch_idx + 1) * self.batch_size_triangles, len(nodes_with_triangles))
                    batch_nodes = nodes_with_triangles[start_idx:end_idx]

                    if precomputed_pairs:
                        node_pairs_cpu, triangle_pairs_cpu = triangles_dict['_node_triangle_pairs']
                        start_node = batch_nodes[0]
                        end_node = batch_nodes[-1]
                        left = torch.searchsorted(node_pairs_cpu, torch.tensor(start_node, dtype=node_pairs_cpu.dtype), right=False)
                        right = torch.searchsorted(node_pairs_cpu, torch.tensor(end_node, dtype=node_pairs_cpu.dtype), right=True)
                        batch_node_tensor = node_pairs_cpu[left:right].to(device, non_blocking=True)
                        batch_triangle_tensor = triangle_pairs_cpu[left:right].to(device, non_blocking=True)
                        valid_mask = batch_triangle_tensor < h_triangles.shape[0]
                        if not valid_mask.all().item():
                            batch_node_tensor = batch_node_tensor[valid_mask]
                            batch_triangle_tensor = batch_triangle_tensor[valid_mask]
                    else:
                        batch_node_indices = []
                        batch_triangle_indices = []
                        
                        for v in batch_nodes:
                            if v in triangles_dict:
                                for tri_idx in triangles_dict[v]:
                                    if tri_idx < h_triangles.shape[0]:
                                        batch_node_indices.append(v)
                                        batch_triangle_indices.append(tri_idx)
                        
                        if len(batch_node_indices) == 0:
                            continue
                        
                        batch_node_tensor = torch.tensor(batch_node_indices, dtype=torch.long, device=device)
                        batch_triangle_tensor = torch.tensor(batch_triangle_indices, dtype=torch.long, device=device)

                    if batch_node_tensor.numel() == 0:
                        continue
                    
                    unique_triangle_indices, batch_triangle_indices_mapped = torch.unique(
                        batch_triangle_tensor, sorted=True, return_inverse=True
                    )
                    if h_triangles.device.type == 'cpu':
                        h_triangles_batch = h_triangles[unique_triangle_indices.cpu()].to(
                            device, dtype=h_transformed.dtype
                        )  # [unique_triangles_in_batch, triangle_dim]
                    else:
                        h_triangles_batch = h_triangles[unique_triangle_indices].to(dtype=h_transformed.dtype)  # [unique_triangles_in_batch, triangle_dim]
                    h_triangles_transformed_batch = self.W2(h_triangles_batch)  # [unique_triangles_in_batch, out_dim]
                    
                    h_v_batch = h_transformed[batch_node_tensor]  # [batch_pairs, out_dim]
                    h_T_batch = h_triangles_transformed_batch[batch_triangle_indices_mapped]  # [batch_pairs, out_dim]
                    
                    concat_feat_batch = torch.cat([h_v_batch, h_T_batch], dim=1)  # [batch_pairs, 2*out_dim]
                    
                    attention_scores_batch = torch.matmul(concat_feat_batch, self.a2)  # [batch_pairs, 1]
                    attention_scores_batch = F.leaky_relu(attention_scores_batch, negative_slope=self.negative_slope)
                    attention_scores_batch = attention_scores_batch.squeeze(1)  # [batch_pairs]
                    
                    if HAS_TORCH_SCATTER:
                        attention_weights_batch = scatter_softmax(attention_scores_batch, batch_node_tensor, dim=0)
                        weighted_features_batch = attention_weights_batch.unsqueeze(1) * h_T_batch
                        m_high_order.index_add_(0, batch_node_tensor, weighted_features_batch)
                    else:
                        unique_batch_nodes = torch.unique(batch_node_tensor)
                        for v in unique_batch_nodes:
                            mask = (batch_node_tensor == v)
                            if not mask.any():
                                continue
                            
                            scores_v = attention_scores_batch[mask]
                            triangles_v = h_T_batch[mask]
                            weights_v = F.softmax(scores_v, dim=0)
                            m_high_order[v] += torch.sum(weights_v.unsqueeze(1) * triangles_v, dim=0)
                    
                    del h_v_batch, h_T_batch, concat_feat_batch, attention_scores_batch
                    del h_triangles_batch, h_triangles_transformed_batch, unique_triangle_indices, batch_triangle_indices_mapped
                    del batch_node_tensor, batch_triangle_tensor
                    if 'weighted_features_batch' in locals():
                        del weighted_features_batch
                    if 'attention_weights_batch' in locals():
                        del attention_weights_batch
                    if h.is_cuda:
                        torch.cuda.empty_cache()
                        if batch_idx > 0 and (batch_idx + 1) % 5 == 0:
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
        
        # Step3: Fusion (learnable gate + residual)
        h_self = self.W0(h)  # [num_nodes, out_dim]
        # gate_weights [N, 2] -> (w_pair, w_high) after softmax
        gate_in = torch.cat([h_self, m_pairwise, m_high_order], dim=1)  # [num_nodes, 3*out_dim]
        gate_weights = F.softmax(self.gate_linear(gate_in), dim=1)  # [num_nodes, 2]
        w_pair = gate_weights[:, 0:1]   # [num_nodes, 1]
        w_high = gate_weights[:, 1:2]   # [num_nodes, 1]
        m_fused = w_pair * m_pairwise + w_high * (self.lambda_high * m_high_order)
        # h_updated = residual(h) + ReLU(h_self + m_fused)
        h_updated = self.residual_proj(h) + self.activation(h_self + m_fused)
        
        del h_self, m_pairwise, m_high_order, gate_in, gate_weights, w_pair, w_high, m_fused
        
        h_updated = self.dropout(h_updated)
        
        return h_updated


class SimplicialGNN(nn.Module):
    """
    Simplicial GNN for node classification.

    Stacks L SimplicialMessagePassing layers for multi-hop higher-order reasoning,
    then applies a classification head.

    Args:
        input_dim (int): Input feature dimension (initial node features).
        hidden_dim (int): Hidden dimension; default 64.
        output_dim (int): Number of classes; default 12.
        num_layers (int): Number of message-passing layers; default 2.
        lambda_high (float): Higher-order injection strength; default 1.0.
        dropout (float): Dropout rate; default 0.5.
        negative_slope (float): LeakyReLU negative slope; default 0.2.
        batch_size_triangles (int, optional): Node batch size for triangle aggregation (GPU memory).
            If None, no batching. Typical range: 1000–5000.
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=12, num_layers=2, 
                 lambda_high=1.0, dropout=0.5, negative_slope=0.2, batch_size_triangles=None,
                 batch_size_edges=None, gradient_checkpointing=False):
        super(SimplicialGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing
        
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()
        
        # h_triangles should match input_dim; W2 projects triangles to hidden_dim inside each layer.
        
        self.message_passing_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layer_input_dim = hidden_dim
            else:
                layer_input_dim = hidden_dim
            self.message_passing_layers.append(
                SimplicialMessagePassing(
                    in_dim=layer_input_dim,
                    out_dim=hidden_dim,
                    triangle_dim=input_dim,
                    lambda_high=lambda_high,
                    dropout=dropout,
                    negative_slope=negative_slope,
                    batch_size_triangles=batch_size_triangles,
                    batch_size_edges=batch_size_edges,
                )
            )
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.norm_before_cls = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, h_0, edge_index, triangles_dict, h_triangles):
        """
        Forward pass.

        Args:
            h_0 (torch.Tensor): Initial node features, [num_nodes, input_dim].
            edge_index (torch.Tensor): Edge indices, [num_edges, 2].
            triangles_dict (dict): Node -> triangle indices.
            h_triangles (torch.Tensor): Triangle features, [num_triangles, input_dim]
                (same feature dim as nodes).

        Returns:
            torch.Tensor: Class logits, [num_nodes, output_dim].
        """
        h = self.input_proj(h_0)  # [num_nodes, hidden_dim]
        
        for i, layer in enumerate(self.message_passing_layers):
            h_res = h
            if self.gradient_checkpointing and self.training:
                h = checkpoint(
                    lambda x, layer=layer: layer(x, edge_index, triangles_dict, h_triangles),
                    h,
                    use_reentrant=True,
                )
            else:
                h = layer(h, edge_index, triangles_dict, h_triangles)
            h = h_res + h
            h = self.norm_layers[i](h)
        
        h = self.norm_before_cls(h)
        logits = self.classifier(h)  # [num_nodes, output_dim]
        
        return logits


def build_node_to_triangles_dict(triangles, num_nodes, include_precomputed_pairs=True,
                                 max_triangles_per_node=None, seed=42):
    """
    Build node -> list of triangle indices.

    Args:
        triangles (set): Triangles as (u, v, w) with u < v < w.
        num_nodes (int): Number of nodes.

    Returns:
        dict: Node index -> list of triangle indices that include that node.
    """
    triangles_list = sorted(list(triangles))
    node_to_triangles = {}
    
    for tri_idx, (u, v, w) in enumerate(triangles_list):
        for node in (u, v, w):
            node_to_triangles.setdefault(node, []).append(tri_idx)

    if max_triangles_per_node is not None and max_triangles_per_node > 0:
        rng = np.random.default_rng(seed)
        for node, tri_indices in list(node_to_triangles.items()):
            if isinstance(node, str) or len(tri_indices) <= max_triangles_per_node:
                continue
            selected = rng.choice(
                np.asarray(tri_indices, dtype=np.int64),
                size=max_triangles_per_node,
                replace=False,
            )
            node_to_triangles[node] = selected.tolist()

    if include_precomputed_pairs:
        nodes_with_triangles = sorted(
            [node for node in node_to_triangles.keys() if not isinstance(node, str)]
        )
        if nodes_with_triangles:
            node_indices = []
            triangle_indices = []
            for node in nodes_with_triangles:
                tris = node_to_triangles[node]
                node_indices.extend([node] * len(tris))
                triangle_indices.extend(tris)
            node_to_triangles['_node_triangle_pairs'] = (
                torch.tensor(node_indices, dtype=torch.long),
                torch.tensor(triangle_indices, dtype=torch.long),
            )
            node_to_triangles['_nodes_with_triangles'] = nodes_with_triangles
        else:
            node_to_triangles['_node_triangle_pairs'] = (
                torch.empty((0,), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
            )
            node_to_triangles['_nodes_with_triangles'] = []
    
    return node_to_triangles


def compute_loss(logits, labels, train_idx, weight_decay=0.0, model=None,
                 label_smoothing=0.0, class_weights=None):
    """
    Cross-entropy loss plus optional L2 regularization.

    Args:
        logits (torch.Tensor): Predictions, [num_nodes, num_classes].
        labels (torch.Tensor): Ground-truth labels, [num_nodes].
        train_idx (torch.Tensor): Training node indices.
        weight_decay (float): L2 coefficient; default 0.0.
        model (nn.Module): Model whose parameters are regularized; default None.

    Returns:
        torch.Tensor: Total loss.
    """
    train_logits = logits[train_idx]
    train_labels = labels[train_idx]
    ce_loss = F.cross_entropy(
        train_logits,
        train_labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    
    l2_reg = torch.tensor(0.0, device=logits.device)
    if weight_decay > 0 and model is not None:
        for param in model.parameters():
            l2_reg += torch.sum(param ** 2)
        l2_reg = weight_decay * l2_reg
    
    total_loss = ce_loss + l2_reg
    return total_loss


def evaluate(model, h_0, edge_index, triangles_dict, h_triangles, labels, idx):
    """
    Evaluate accuracy on a subset of nodes.

    Args:
        model (nn.Module): Model.
        h_0 (torch.Tensor): Node features.
        edge_index (torch.Tensor): Edge indices.
        triangles_dict (dict): Node -> triangle indices.
        h_triangles (torch.Tensor): Triangle features.
        labels (torch.Tensor): Ground-truth labels.
        idx (torch.Tensor): Evaluation node indices.

    Returns:
        float: Accuracy.
    """
    model.eval()
    with torch.no_grad():
        logits = model(h_0, edge_index, triangles_dict, h_triangles)
        pred = logits[idx].argmax(dim=1)
        acc = (pred == labels[idx]).float().mean().item()
    return acc
