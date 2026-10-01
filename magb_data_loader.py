"""
MAGB-format data loader — loads MAGB multimodal graph data (Grocery / Toys and same-format datasets).
Follows MAGB-master/GNN/GraphData.py and converts DGL graphs to the required format.
Infers graph and feature file names from the last directory component of data_path
(e.g. Toys -> ToysGraph.pt, Grocery -> GroceryGraph.pt).
"""

import os
import glob
import ast
import csv
import numpy as np
import torch


def _dataset_name_from_path(data_path):
    """Infer dataset name from data directory path, e.g. /path/to/Data/Toys -> Toys."""
    return os.path.basename(os.path.normpath(os.path.abspath(data_path)))


def _split_graph(nodes_num, train_ratio, val_ratio, labels, fewshots=None, split_seed=45):
    """Split train/val/test indices; matches MAGB GraphData.split_graph."""
    rng = np.random.default_rng(split_seed)
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    indices = rng.permutation(nodes_num)
    if fewshots is not None:
        train_ids = []
        for label in np.unique(labels_np):
            label_indices = np.where(labels_np == label)[0]
            rng.shuffle(label_indices)
            train_ids.extend(label_indices[:fewshots])
        remaining = np.setdiff1d(indices, train_ids)
        rng.shuffle(remaining)
        val_size = int(len(remaining) * val_ratio)
        val_ids, test_ids = remaining[:val_size], remaining[val_size:]
    else:
        train_size = int(nodes_num * train_ratio)
        val_size = int(nodes_num * val_ratio)
        train_ids = indices[:train_size]
        val_ids = indices[train_size : train_size + val_size]
        test_ids = indices[train_size + val_size :]
    return train_ids, val_ids, test_ids


def _to_long_tensor_indices(value):
    if isinstance(value, torch.Tensor):
        return value.long()
    return torch.LongTensor(value)


def _load_edges_labels_without_dgl(data_path, dataset_name, undirected=True):
    """Load MAGB Grocery/Toys graph inputs without requiring DGL."""
    cache_candidates = [
        os.path.join(data_path, f"{dataset_name.lower()}_nc_cache.pt"),
        os.path.join(data_path, f"{dataset_name}_nc_cache.pt"),
    ]
    for cache_path in cache_candidates:
        if os.path.exists(cache_path):
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            edges = cache["edges"].long()
            labels = cache["labels"].long()
            num_nodes = int(cache.get("num_nodes", labels.shape[0]))
            return edges, labels, num_nodes

    csv_path = os.path.join(data_path, f"{dataset_name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"DGL is unavailable and no fallback cache/CSV was found under {data_path}"
        )

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    labels = torch.LongTensor([int(row["label"]) for row in rows])
    num_nodes = len(rows)
    edges_set = set()
    for row in rows:
        src = int(row["id"])
        for col in ("also_buy", "also_view"):
            value = row.get(col, "[]")
            if not isinstance(value, str) or not value:
                continue
            try:
                neighbors = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                neighbors = []
            for dst in neighbors:
                dst = int(dst)
                if 0 <= src < num_nodes and 0 <= dst < num_nodes and src != dst:
                    edges_set.add((src, dst))
                    if undirected:
                        edges_set.add((dst, src))
    edges = torch.LongTensor(sorted(edges_set)) if edges_set else torch.empty((0, 2), dtype=torch.long)
    return edges, labels, num_nodes


def load_completed_graph_data(data_path, feat_path=None, train_ratio=0.6, val_ratio=0.2,
                              fewshots=None, undirected=True, load_edges=True, split_seed=45):
    """
    Load Multimodal-Graph-Completed-Graph datasets such as ele-fashion/books-nc.
    The split.pt shipped with the dataset is used by default for reproducible results.
    """
    labels_raw = torch.load(os.path.join(data_path, "labels-w-missing.pt"), map_location="cpu", weights_only=False)
    labels = torch.LongTensor(labels_raw)
    valid_mask = labels >= 0

    if load_edges:
        edges_raw = torch.load(os.path.join(data_path, "nc_edges-nodeid.pt"), map_location="cpu", weights_only=False)
        edges = torch.LongTensor(edges_raw)
        if undirected and edges.numel() > 0:
            rev_edges = edges[:, [1, 0]]
            edges = torch.unique(torch.cat([edges, rev_edges], dim=0), dim=0)
    else:
        edges = torch.empty((0, 2), dtype=torch.long)

    split_path = os.path.join(data_path, "split.pt")
    if os.path.exists(split_path) and fewshots is None:
        split = torch.load(split_path, map_location="cpu", weights_only=False)
        splits = {
            "train_idx": _to_long_tensor_indices(split["train_idx"]),
            "val_idx": _to_long_tensor_indices(split["val_idx"]),
            "test_idx": _to_long_tensor_indices(split["test_idx"]),
        }
    else:
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).view(-1)
        compact_labels = labels[valid_idx]
        tr, va, te = _split_graph(len(valid_idx), train_ratio, val_ratio, compact_labels, fewshots, split_seed=split_seed)
        splits = {
            "train_idx": valid_idx[_to_long_tensor_indices(tr)],
            "val_idx": valid_idx[_to_long_tensor_indices(va)],
            "test_idx": valid_idx[_to_long_tensor_indices(te)],
        }

    for key in list(splits):
        idx = splits[key]
        splits[key] = idx[valid_mask[idx]]

    feature_path = feat_path
    if feature_path is None:
        feature_path = os.path.join(data_path, "clip_feat.pt")
    elif not os.path.isabs(feature_path):
        feature_path = os.path.join(data_path, feature_path)
    feat = torch.load(feature_path, map_location="cpu", weights_only=False).float()
    mid = feat.shape[1] // 2
    t_feat, v_feat = feat[:, :mid], feat[:, mid:]

    num_classes = int(labels[valid_mask].max().item()) + 1
    return edges, labels, splits, v_feat, t_feat, num_classes


def load_magb_grocery_data(
    data_path,
    graph_name=None,
    feat_path=None,
    text_feat_path=None,
    image_feat_path=None,
    train_ratio=0.6,
    val_ratio=0.2,
    fewshots=None,
    undirected=True,
    load_edges=True,
    split_seed=45,
):
    """
    Load MAGB data. Supports Grocery, Toys, and same-format datasets.
    If graph_name is omitted, it is inferred from the last directory name of data_path
    (e.g. Toys -> ToysGraph.pt).
    Returns: edges, labels, splits, v_feat, t_feat, num_classes
    """
    dataset_name = _dataset_name_from_path(data_path)
    completed_graph_files = ("labels-w-missing.pt", "nc_edges-nodeid.pt", "split.pt")
    if all(os.path.exists(os.path.join(data_path, name)) for name in completed_graph_files):
        return load_completed_graph_data(
            data_path,
            feat_path=feat_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            fewshots=fewshots,
            undirected=undirected,
            load_edges=load_edges,
            split_seed=split_seed,
        )

    if graph_name is None:
        graph_name = f"{dataset_name}Graph.pt"

    graph_path = os.path.join(data_path, graph_name)
    try:
        import dgl
    except ImportError:
        edges, labels, nodes_num = _load_edges_labels_without_dgl(data_path, dataset_name, undirected=undirected)
    else:
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

        graph_list, _ = dgl.load_graphs(graph_path)
        graph = graph_list[0]
        labels = graph.ndata["label"].long()

        src, dst = graph.edges()
        src, dst = src.numpy(), dst.numpy()
        edges = np.stack([src, dst], axis=1)
        if undirected:
            edges = np.vstack([edges, np.stack([dst, src], axis=1)])
        edges = torch.LongTensor(edges)
        nodes_num = graph.num_nodes()

    train_ids, val_ids, test_ids = _split_graph(
        nodes_num, train_ratio, val_ratio, labels, fewshots, split_seed=split_seed
    )
    splits = {
        "train_idx": torch.LongTensor(train_ids),
        "val_idx": torch.LongTensor(val_ids),
        "test_idx": torch.LongTensor(test_ids),
    }

    if feat_path is not None and os.path.exists(feat_path):
        clip_feat = np.load(feat_path).astype(np.float32)
        clip_feat = torch.from_numpy(clip_feat).float()
        mid = clip_feat.shape[1] // 2
        t_feat, v_feat = clip_feat[:, :mid], clip_feat[:, mid:]
    elif text_feat_path is not None and image_feat_path is not None:
        t_raw = np.load(text_feat_path).astype(np.float32)
        v_raw = np.load(image_feat_path).astype(np.float32)
        if t_raw.shape[0] != v_raw.shape[0]:
            raise ValueError("Text and image features have inconsistent numbers of nodes")
        t_feat = torch.from_numpy(t_raw).float()
        v_feat = torch.from_numpy(v_raw).float()
    else:
        default_mm = os.path.join(data_path, "MMFeature", f"{dataset_name}_LLAMA8B_CLIP.npy")
        default_t = os.path.join(data_path, "TextFeature", f"{dataset_name}_roberta_base_256_mean.npy")
        default_v = os.path.join(data_path, "ImageFeature", f"{dataset_name}_openai_clip-vit-large-patch14.npy")
        if not os.path.exists(default_t) and os.path.isdir(os.path.join(data_path, "TextFeature")):
            text_candidates = glob.glob(os.path.join(data_path, "TextFeature", f"{dataset_name}_roberta*.npy"))
            if text_candidates:
                default_t = text_candidates[0]
        if os.path.exists(default_mm):
            clip_feat = torch.from_numpy(np.load(default_mm).astype(np.float32)).float()
            mid = clip_feat.shape[1] // 2
            t_feat, v_feat = clip_feat[:, :mid], clip_feat[:, mid:]
        elif os.path.exists(default_t) and os.path.exists(default_v):
            t_feat = torch.from_numpy(np.load(default_t).astype(np.float32)).float()
            v_feat = torch.from_numpy(np.load(default_v).astype(np.float32)).float()
        else:
            raise FileNotFoundError(
                f"No feature files found; pass --feat_path or ensure {default_mm} exists, or both {default_t} and {default_v}"
            )

    num_classes = int(labels.max().item()) + 1
    return edges, labels, splits, v_feat, t_feat, num_classes
