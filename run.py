import os
import sys
import copy
# Set before importing torch to reduce CUDA fragmentation and help avoid OOM
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import argparse
import numpy as np
import time
import warnings
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.exceptions import UndefinedMetricWarning
from model import SimplicialGNN, build_node_to_triangles_dict, compute_loss, evaluate
from magb_data_loader import load_magb_grocery_data


def set_seed(seed, deterministic=True):
    """Set all local RNGs used by this training script."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_data_path(dataset, data_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(os.path.join(script_dir, "Data")) or os.path.isdir(os.path.join(script_dir, "Multimodal-Graph-Completed-Graph")):
        project_root = script_dir
    else:
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
    if data_path is not None:
        return os.path.abspath(data_path)

    dataset_paths = {
        'grocery': os.path.join(project_root, "Data", "Grocery"),
        'toys': os.path.join(project_root, "Data", "Toys"),
        'ele-fashion': os.path.join(project_root, "Multimodal-Graph-Completed-Graph", "ele-fashion"),
        'books-nc': os.path.join(project_root, "Multimodal-Graph-Completed-Graph", "books-nc"),
        'books': os.path.join(project_root, "Multimodal-Graph-Completed-Graph", "books-nc"),
    }
    if dataset not in dataset_paths:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return os.path.abspath(dataset_paths[dataset])


def is_books_dataset(dataset):
    return dataset in {'books', 'books-nc'}


def _flag_was_passed(flag_name):
    aliases = {
        'repeat_times': ['times'],
    }
    flag_names = [flag_name] + aliases.get(flag_name, [])
    for arg in sys.argv[1:]:
        for name in flag_names:
            prefix = f"--{name.replace('_', '-')}"
            underscore_prefix = f"--{name}"
            if arg == prefix or arg.startswith(prefix + "="):
                return True
            if arg == underscore_prefix or arg.startswith(underscore_prefix + "="):
                return True
    return False


def _apply_profile_defaults(args, defaults):
    """Apply dataset-tuned defaults while preserving explicit CLI overrides."""
    for attr, value in defaults.items():
        if not _flag_was_passed(attr):
            setattr(args, attr, value)


def apply_agsp_dataset_profile(args):
    """Use tuned AGSP defaults for each dataset unless a CLI flag overrides them."""
    if getattr(args, 'no_dataset_profile', False):
        return
    if args.model_type != 'agsp' or is_books_dataset(args.dataset):
        return

    profiles = {
        'grocery': {
            'hidden_dim': 352,
            'num_layers': 2,
            'dropout': 0.4,
            'lr': 0.001,
            'lambda_high': 2.0,
            'weight_decay': 0.0001,
            'selection_metric': 'f1',
            'epochs': 220,
            'patience': 45,
            'repeat_times': 1,
            'seed': 42,
            'split_seed': 45,
            'metric_average': 'weighted',
            'class_weight': 'balanced',
            'class_weight_power': 0.5,
            'eval_every': 1,
            'logit_bias_calibration': 'val_coordinate',
            'checkpoint_tag': 'grocery_books_hparams_weighted_h352_f1weighted_cw05',
            'skip_train_eval': True,
        },
        'toys': {
            'hidden_dim': 352,
            'num_layers': 2,
            'dropout': 0.4,
            'lr': 0.001,
            'lambda_high': 2.0,
            'weight_decay': 0.0001,
            'selection_metric': 'f1',
            'eval_every': 1,
            'epochs': 220,
            'patience': 45,
            'repeat_times': 1,
            'seed': 42,
            'split_seed': 11,
            'metric_average': 'weighted',
            'class_weight': 'balanced',
            'class_weight_power': 0.5,
            'logit_bias_calibration': 'val_coordinate',
            'checkpoint_tag': 'toys_books_hparams_weighted_h352_f1weighted_cw05',
            'skip_train_eval': True,
        },
        'ele-fashion': {
            'hidden_dim': 352,
            'num_layers': 2,
            'dropout': 0.4,
            'lr': 0.001,
            'lambda_high': 2.0,
            'weight_decay': 0.0001,
            'selection_metric': 'f1',
            'epochs': 220,
            'patience': 45,
            'repeat_times': 1,
            'seed': 42,
            'split_seed': 45,
            'metric_average': 'weighted',
            'class_weight': 'balanced',
            'class_weight_power': 0.5,
            'eval_every': 1,
            'logit_bias_calibration': 'val_coordinate',
            'compact_split_classes': True,
            'checkpoint_tag': 'ele-fashion_books_hparams_weighted_h352_f1weighted_cw05',
            'skip_train_eval': True,
        },
    }
    profile = profiles.get(args.dataset)
    if profile is None:
        return

    _apply_profile_defaults(args, profile)
    print(f"\n{args.dataset} AGSP dataset profile enabled")
    print("  Explicit command-line flags override profile defaults")


def apply_books_agsp_profile(args):
    """Lock the tuned two-stage Books AGSP recipe without touching other datasets."""
    if getattr(args, 'no_books_profile', False):
        return
    if not (is_books_dataset(args.dataset) and args.model_type == 'agsp'):
        return

    fixed_defaults = {
        'epochs': 220,
        'repeat_times': 1,
        'patience': 45,
        'train_batch_size': 999999,
        'eval_batch_size': 999999,
        'hidden_dim': 352,
        'num_layers': 2,
        'dropout': 0.4,
        'lambda_high': 2.0,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'selection_metric': 'f1',
        'metric_average': 'weighted',
        'class_weight': 'balanced',
        'class_weight_power': 0.5,
        'skip_train_eval': True,
        'simplex_cache_path': 'AGSP/cache/books_simplex_fp16.pt',
        'batch_size_triangles': 1000,
        'batch_size_edges': 200000,
        'gradient_checkpointing': True,
        'eval_every': 1,
        'logit_bias_calibration': 'val_coordinate',
        'checkpoint_tag': 'books_books_hparams_weighted_h352_f1weighted_cw05',
        'compact_split_classes': True,
    }
    _apply_profile_defaults(args, fixed_defaults)

    if not _flag_was_passed('keep_data_cpu'):
        args.keep_data_cpu = True
    if not _flag_was_passed('simplex_cache_fp16'):
        args.simplex_cache_fp16 = True
    if not _flag_was_passed('sort_edges_by_target'):
        args.sort_edges_by_target = True

    print("\nBooks AGSP resource profile enabled")
    print("  Unified defaults: h352, f1 selection, weighted metrics, sqrt-balanced CE")
    print("  Books resources: CPU data, simplex fp16 cache, sorted edge batching")


def print_effective_hyperparameters(args):
    fields = [
        'dataset', 'model_type', 'epochs', 'repeat_times', 'patience',
        'hidden_dim', 'num_layers', 'dropout', 'lambda_high', 'lr',
        'weight_decay', 'label_smoothing', 'selection_metric',
        'metric_average', 'class_weight', 'class_weight_power',
        'skip_train_eval', 'batch_size_triangles', 'sort_edges_by_target',
        'batch_size_edges', 'eval_every', 'logit_bias_calibration',
        'logit_bias_selection_metric', 'logit_bias_grid',
        'logit_bias_rounds', 'seed', 'split_seed', 'compact_split_classes',
        'keep_data_cpu', 'simplex_cache_path', 'simplex_cache_fp16',
        'train_batch_size', 'eval_batch_size', 'gradient_checkpointing',
        'mixed_precision', 'two_stage',
        'init_checkpoint', 'freeze_backbone',
    ]
    print("\nEffective hyperparameters:")
    for field in fields:
        print(f"  {field}: {getattr(args, field, None)}")


def calculate_metrics(y_true, y_pred, y_proba=None, num_classes=None, average='weighted'):
    """
    Compute classification metrics.

    Args:
        y_true: Ground-truth labels (numpy or torch).
        y_pred: Predicted labels (numpy or torch).
        y_proba: Probabilities or logits (numpy or torch, shape [n_samples, n_classes]).
        num_classes: Number of classes.

    Returns:
        dict with accuracy, precision, recall, f1, auc.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    
    metrics = {}
    
    accuracy = float((y_true == y_pred).mean())
    metrics['accuracy'] = accuracy
    
    precision = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    recall = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    if y_proba is not None:
        try:
            if isinstance(y_proba, torch.Tensor):
                y_proba_np = y_proba.cpu().numpy()
            else:
                y_proba_np = np.array(y_proba).copy()
            
            if y_proba_np.size == 0:
                metrics['auc'] = 0.0
                return metrics
            
            if len(y_proba_np.shape) != 2:
                raise ValueError(f"y_proba must be 2D [n_samples, n_classes], got shape {y_proba_np.shape}")
            
            if len(y_true) != y_proba_np.shape[0]:
                raise ValueError(f"y_true length ({len(y_true)}) != y_proba rows ({y_proba_np.shape[0]})")
            
            if np.any(np.isnan(y_proba_np)) or np.any(np.isinf(y_proba_np)):
                raise ValueError("y_proba contains NaN or Inf; cannot compute AUC")
            
            if num_classes is None:
                num_classes = y_proba_np.shape[1]
            
            if y_proba_np.shape[1] != num_classes:
                raise ValueError(f"y_proba columns ({y_proba_np.shape[1]}) != num_classes ({num_classes})")
            
            row_sums = np.sum(y_proba_np, axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-3):
                exp_scores = np.exp(y_proba_np - np.max(y_proba_np, axis=1, keepdims=True))
                y_proba_np = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                if np.any(np.isnan(y_proba_np)):
                    raise ValueError("NaN after softmax on y_proba; cannot compute AUC")
            
            unique_labels = np.unique(y_true)
            if len(unique_labels) == 0:
                metrics['auc'] = 0.0
                return metrics
            
            min_label = np.min(unique_labels)
            max_label = np.max(unique_labels)
            
            if min_label < 0 or max_label >= num_classes:
                raise ValueError(f"Labels in [{min_label}, {max_label}] outside [0, {num_classes-1}]")
            
            if len(unique_labels) < 2:
                metrics['auc'] = 0.0
                return metrics
            
            if num_classes == 2:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                    auc = float(roc_auc_score(y_true, y_proba_np[:, 1]))
            else:
                existing_classes = unique_labels
                
                if len(existing_classes) < num_classes:
                    y_true_filtered_binarized = label_binarize(y_true, classes=existing_classes)
                    y_proba_filtered = y_proba_np[:, existing_classes]
                    
                    proba_sums = np.sum(y_proba_filtered, axis=1, keepdims=True)
                    proba_sums = np.where(proba_sums == 0, 1.0, proba_sums)
                    y_proba_filtered = y_proba_filtered / proba_sums
                    
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                        try:
                            auc = float(roc_auc_score(y_true_filtered_binarized, y_proba_filtered, 
                                                     multi_class='ovr', average=average))
                        except ValueError as e:
                            try:
                                auc = float(roc_auc_score(y_true_filtered_binarized, y_proba_filtered, 
                                                         multi_class='ovr', average='macro'))
                            except ValueError as e2:
                                print(f"Warning: multiclass AUC failed (classes present: {existing_classes}). Error: {e2}")
                                auc = 0.0
                else:
                    y_true_binarized = label_binarize(y_true, classes=np.arange(num_classes))
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                        try:
                            auc = float(roc_auc_score(y_true_binarized, y_proba_np, 
                                                     multi_class='ovr', average=average))
                        except ValueError as e:
                            try:
                                auc = float(roc_auc_score(y_true_binarized, y_proba_np, 
                                                         multi_class='ovr', average='weighted'))
                            except ValueError as e2:
                                print(f"Warning: multiclass AUC failed. Error: {e2}")
                                auc = 0.0
            
            if np.isnan(auc) or np.isinf(auc):
                print(f"Warning: invalid AUC {auc}, setting to 0.0")
                auc = 0.0
            
            metrics['auc'] = auc
        except Exception as e:
            print(f"Warning: could not compute AUC: {e}")
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    return metrics


def log_label_distribution(labels, train_idx, val_idx, test_idx, num_classes, name="dataset"):
    """
    Print per-split label counts and ratios for class imbalance analysis.
    labels and *_idx may be numpy or torch tensors.
    """
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    labels = np.asarray(labels).astype(np.int64)
    train_idx = np.asarray(train_idx) if hasattr(train_idx, 'cpu') else np.asarray(train_idx)
    val_idx = np.asarray(val_idx) if hasattr(val_idx, 'cpu') else np.asarray(val_idx)
    test_idx = np.asarray(test_idx) if hasattr(test_idx, 'cpu') else np.asarray(test_idx)

    def _count_and_ratio(idx):
        if len(idx) == 0:
            return {}, {}
        y = labels[idx]
        cnt = {}
        for c in range(num_classes):
            cnt[c] = int((y == c).sum())
        total = len(y)
        ratio = {c: cnt[c] / total for c in range(num_classes)}
        return cnt, ratio

    print(f"\n[{name}] label distribution ({num_classes} classes)")
    print("=" * 70)
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        cnt, ratio = _count_and_ratio(idx)
        total = sum(cnt.values())
        print(f"  {split_name}: total samples = {total}")
        for c in range(num_classes):
            pct = ratio.get(c, 0) * 100
            print(f"    class {c:3d}: count = {cnt.get(c, 0):6d}, pct = {pct:6.2f}%")
        if total > 0:
            max_c = max(range(num_classes), key=lambda k: cnt.get(k, 0))
            min_c = min(range(num_classes), key=lambda k: cnt.get(k, 0) if cnt.get(k, 0) > 0 else float('inf'))
            print(f"    (max: class {max_c} n={cnt.get(max_c, 0)}, min: class {min_c} n={cnt.get(min_c, 0)})")
    print("=" * 70)


def sample_split_by_ratio(labels, split_idx, ratio, seed, num_classes):
    """Stratified split subsampling for memory-bound diagnostic runs."""
    if ratio >= 1.0:
        return split_idx
    if ratio <= 0.0:
        raise ValueError("--split_sample_ratio must be > 0")

    idx = split_idx.detach().cpu().long() if isinstance(split_idx, torch.Tensor) else torch.LongTensor(split_idx)
    labels_cpu = labels.detach().cpu().long() if isinstance(labels, torch.Tensor) else torch.LongTensor(labels)
    gen = torch.Generator().manual_seed(seed)
    sampled = []
    for cls in range(num_classes):
        cls_idx = idx[labels_cpu[idx] == cls]
        if cls_idx.numel() == 0:
            continue
        keep = max(1, int(cls_idx.numel() * ratio))
        perm = torch.randperm(cls_idx.numel(), generator=gen)[:keep]
        sampled.append(cls_idx[perm])
    if not sampled:
        return idx[:0]
    return torch.cat(sampled).sort().values


def compact_labels_to_split_classes(raw_data):
    """Remap labels that appear in the active split to contiguous ids."""
    split_tensors = [
        torch.as_tensor(raw_data['splits'][key], dtype=torch.long)
        for key in ('train_idx', 'val_idx', 'test_idx')
    ]
    active_idx = torch.cat(split_tensors)
    labels = raw_data['labels']
    labels_cpu = labels.detach().cpu().long() if isinstance(labels, torch.Tensor) else torch.LongTensor(labels)
    active_classes = torch.unique(labels_cpu[active_idx]).sort().values
    old_num_classes = int(raw_data['num_classes'])
    new_num_classes = int(active_classes.numel())
    if new_num_classes == old_num_classes and torch.equal(active_classes, torch.arange(old_num_classes)):
        return

    mapping = torch.full((int(labels_cpu.max().item()) + 1,), -1, dtype=torch.long)
    mapping[active_classes] = torch.arange(new_num_classes, dtype=torch.long)
    valid = labels_cpu >= 0
    remapped = labels_cpu.clone()
    remapped[valid] = mapping[labels_cpu[valid]]
    if isinstance(labels, torch.Tensor):
        raw_data['labels'] = remapped.to(labels.device)
    else:
        raw_data['labels'] = remapped
    raw_data['num_classes'] = new_num_classes
    print(
        "Compacted split classes: "
        f"{active_classes.tolist()} -> 0..{new_num_classes - 1} "
        f"(num_classes {old_num_classes} -> {new_num_classes})",
        flush=True,
    )


def log_per_class_metrics(y_true, y_pred, num_classes, logits=None):
    """
    Per-class test counts, per-class accuracy, and predicted label histogram.
    y_true/y_pred are numpy or torch; logits reserved for extensions.
    """
    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, 'cpu'):
        y_pred = y_pred.cpu().numpy()
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    print("\n[test] per-class metrics and prediction histogram")
    print("=" * 70)
    for c in range(num_classes):
        mask = y_true == c
        n = mask.sum()
        if n == 0:
            print(f"  class {c:3d}: no test samples")
            continue
        correct = (y_pred[mask] == c).sum()
        acc_c = correct / n
        print(f"  class {c:3d}: n = {n:6d}, correct = {correct:6d}, acc = {acc_c:.4f}")

    pred_counts = {}
    for c in range(num_classes):
        pred_counts[c] = int((y_pred == c).sum())
    print("  predicted label counts:")
    for c in range(num_classes):
        print(f"    pred class {c:3d}: {pred_counts[c]:6d}")
    print("=" * 70)


def load_raw_dataset(
    dataset='grocery',
    device=None,
    data_path=None,
    feat_path=None,
    train_ratio=0.6,
    val_ratio=0.2,
    load_edges=True,
    split_seed=45,
):
    """
    Load raw MAGB-format data (Grocery / Toys).

    Args:
        dataset (str): 'grocery' or 'toys'; default 'grocery'.
        device: torch.device for tensors, or None (CPU).
        data_path (str): If set, overrides dataset and uses this directory.
        feat_path (str): Optional merged feature file (preferred if set).
        train_ratio (float): Train fraction; default 0.6.
        val_ratio (float): Val fraction; default 0.2.

    Returns:
        dict: edges, labels, splits, v_feat, t_feat, num_nodes, num_classes, data_path.
    """
    DATA_PATH = resolve_data_path(dataset, data_path)
    if data_path is not None:
        print(f"Using data path: {DATA_PATH}")
    else:
        print(f"Dataset: {dataset}, data path: {DATA_PATH}")

    if feat_path is not None and not os.path.isabs(feat_path) and not os.path.exists(feat_path):
        feat_path = os.path.join(DATA_PATH, feat_path)

    edges, labels, splits, v_feat, t_feat, num_classes = load_magb_grocery_data(
        DATA_PATH,
        feat_path=feat_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        fewshots=None,
        undirected=True,
        load_edges=load_edges,
        split_seed=split_seed,
    )
    num_nodes = labels.shape[0]

    if device is not None:
        v_feat = v_feat.to(device)
        t_feat = t_feat.to(device)

    print(f"num_nodes: {num_nodes}, num_classes: {num_classes}")
    print("Raw data loaded.")

    return {
        'edges': edges,
        'labels': labels,
        'splits': splits,
        'v_feat': v_feat,
        't_feat': t_feat,
        'num_nodes': num_nodes,
        'num_classes': num_classes,
        'data_path': DATA_PATH,
    }



def build_triangles_and_B2(edges, num_nodes):
    """
    Build triangle set and B2 boundary operator from edge data.

    Args:
        edges: torch.Tensor [num_edges, 2], numpy array, or list of (u, v).
        num_nodes: Number of nodes.

    Returns:
        dict with 'triangles', 'B2', 'edges_index', 'num_edges', 'num_triangles',
        'edge_index_tensor'.

    Example:
        >>> edges = torch.tensor([[0, 1], [1, 2], [0, 2]])
        >>> num_nodes = 3
        >>> result = build_triangles_and_B2(edges, num_nodes)
        >>> print(result['num_triangles'])  # 1
        >>> print(result['B2'].shape)  # [3, 1]
    """
    
    if isinstance(edges, list):
        if len(edges) > 0 and isinstance(edges[0], tuple):
            edges_array = np.array(edges)
            edge_index_tensor = torch.from_numpy(edges_array)
        else:
            edge_index_tensor = torch.tensor(edges)
    elif isinstance(edges, np.ndarray):
        edge_index_tensor = torch.from_numpy(edges)
    elif isinstance(edges, torch.Tensor):
        edge_index_tensor = edges.clone()
    else:
        raise TypeError(f"Unsupported edge type: {type(edges)}")
    
    if edge_index_tensor.dim() != 2 or edge_index_tensor.shape[1] != 2:
        raise ValueError(f"Edges must be [num_edges, 2], got {edge_index_tensor.shape}")
    
    num_edges = edge_index_tensor.shape[0]
    
    edge_np = edge_index_tensor.cpu().numpy()
    graph = defaultdict(set)
    edges_index = {}
    for i in range(num_edges):
        u, v = int(edge_np[i, 0]), int(edge_np[i, 1])
        if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
            continue
        graph[u].add(v)
        graph[v].add(u)
        key = (min(u, v), max(u, v))
        if key not in edges_index:
            edges_index[key] = i
    
    triangles = set()
    triangle_num = 0
    B2_row = []
    B2_column = []
    values = []
    
    for u in range(num_nodes):
        if u not in graph:
            continue
            
        neighbors_u = graph[u]
        
        for v in neighbors_u:
            if u < v:
                neighbors_v = graph[v]
                common_neighbors = neighbors_u.intersection(neighbors_v)
                
                for w in common_neighbors:
                    if u < v < w:
                        triangles.add((u, v, w))
                        
                        edge_index_uv = edges_index.get((u, v), None)
                        if edge_index_uv is None:
                            edge_index_uv = edges_index.get((v, u), None)
                        if edge_index_uv is None:
                            triangles.discard((u, v, w))
                            continue
                        
                        edge_index_uw = edges_index.get((u, w), None)
                        if edge_index_uw is None:
                            edge_index_uw = edges_index.get((w, u), None)
                        if edge_index_uw is None:
                            triangles.discard((u, v, w))
                            continue
                        
                        edge_index_vw = edges_index.get((v, w), None)
                        if edge_index_vw is None:
                            edge_index_vw = edges_index.get((w, v), None)
                        if edge_index_vw is None:
                            triangles.discard((u, v, w))
                            continue
                        
                        B2_row.extend([edge_index_uv, edge_index_uw, edge_index_vw])
                        B2_column.extend([triangle_num, triangle_num, triangle_num])
                        values.extend([1, -1, 1])
                        triangle_num += 1
    
    num_triangles = len(triangles)
    if num_triangles == 0:
        B2 = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float),
            torch.Size([num_edges, 0])
        )
    else:
        B2_indices = torch.tensor([B2_row, B2_column], dtype=torch.long)
        B2_values = torch.tensor(values, dtype=torch.float)
        B2_shape = torch.Size([num_edges, num_triangles])
        B2 = torch.sparse_coo_tensor(B2_indices, B2_values, B2_shape)
        B2 = B2.coalesce()
    
    return {
        'triangles': triangles,
        'B2': B2,
        'edges_index': edges_index,
        'num_edges': num_edges,
        'num_triangles': num_triangles,
        'edge_index_tensor': edge_index_tensor
    }


def extract_k_hop_subgraph(center_nodes, edge_index, num_nodes, k=2, return_node_mapping=True):
    """
    Extract a k-hop subgraph around center nodes to save memory.

    Args:
        center_nodes: Center node indices (often train nodes).
        edge_index: [num_edges, 2].
        num_nodes: Total nodes in the full graph.
        k: Number of hops; default 2 (typical for 2-layer GNN).
        return_node_mapping: If True, include original->local index map.

    Returns:
        dict with subgraph_nodes, subgraph_edge_index, reverse_mapping,
        num_subgraph_nodes, num_subgraph_edges, and optionally node_mapping.
    """
    device = edge_index.device
    
    if not isinstance(center_nodes, torch.Tensor):
        if isinstance(center_nodes, (list, np.ndarray)):
            center_nodes = torch.tensor(center_nodes, dtype=torch.long, device=device)
        else:
            center_nodes = torch.tensor([center_nodes], dtype=torch.long, device=device)
    else:
        center_nodes = center_nodes.to(device)
    
    center_nodes = center_nodes.unique()
    
    subgraph_nodes_set = set(center_nodes.cpu().tolist())
    
    current_nodes = center_nodes.clone()
    for hop in range(k):
        mask_forward = torch.isin(edge_index[:, 0], current_nodes)
        neighbors_forward = edge_index[mask_forward, 1].unique()
        
        mask_backward = torch.isin(edge_index[:, 1], current_nodes)
        neighbors_backward = edge_index[mask_backward, 0].unique()
        
        all_neighbors = torch.cat([neighbors_forward, neighbors_backward]).unique()
        
        new_neighbors = all_neighbors[~torch.isin(all_neighbors, torch.tensor(list(subgraph_nodes_set), device=device))]
        
        if len(new_neighbors) == 0:
            break
        
        subgraph_nodes_set.update(new_neighbors.cpu().tolist())
        current_nodes = new_neighbors
    
    subgraph_nodes_list = sorted(list(subgraph_nodes_set))
    subgraph_nodes = torch.tensor(subgraph_nodes_list, dtype=torch.long, device=device)
    num_subgraph_nodes = len(subgraph_nodes_list)
    
    node_mapping = {orig_idx: subgraph_idx for subgraph_idx, orig_idx in enumerate(subgraph_nodes_list)}
    reverse_mapping = {subgraph_idx: orig_idx for orig_idx, subgraph_idx in node_mapping.items()}
    
    subgraph_tensor = torch.tensor(subgraph_nodes_list, dtype=torch.long, device=device)
    edge_mask = torch.isin(edge_index[:, 0], subgraph_tensor) & torch.isin(edge_index[:, 1], subgraph_tensor)
    subgraph_edge_index_raw = edge_index[edge_mask]
    num_subgraph_edges = subgraph_edge_index_raw.shape[0]
    
    mapping_tensor = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    mapping_tensor[subgraph_nodes] = torch.arange(num_subgraph_nodes, dtype=torch.long, device=device)
    subgraph_edge_index = mapping_tensor[subgraph_edge_index_raw].clone()
    
    result = {
        'subgraph_nodes': subgraph_nodes,
        'subgraph_edge_index': subgraph_edge_index,
        'reverse_mapping': reverse_mapping,
        'num_subgraph_nodes': num_subgraph_nodes,
        'num_subgraph_edges': num_subgraph_edges
    }
    
    if return_node_mapping:
        result['node_mapping'] = node_mapping
    
    return result


def compute_0_simplex_features(v_feat, t_feat):
    """
    0-simplex (node) features: concatenate text and vision along dim=1.

    Args:
        v_feat: [num_nodes, v_feat_dim]
        t_feat: [num_nodes, t_feat_dim]

    Returns:
        h_0: [num_nodes, v_feat_dim + t_feat_dim]
    """
    h_0 = torch.cat([t_feat, v_feat], dim=1)
    return h_0


def standardize_modal_features(v_feat, t_feat, train_idx, eps=1e-6):
    """Standardize text and visual features using train nodes only."""
    if not isinstance(train_idx, torch.Tensor):
        train_idx = torch.as_tensor(train_idx, dtype=torch.long, device=v_feat.device)
    else:
        train_idx = train_idx.to(v_feat.device)

    v_mean = v_feat[train_idx].mean(dim=0, keepdim=True)
    v_std = v_feat[train_idx].std(dim=0, keepdim=True).clamp_min(eps)
    t_mean = t_feat[train_idx].mean(dim=0, keepdim=True)
    t_std = t_feat[train_idx].std(dim=0, keepdim=True).clamp_min(eps)
    return (v_feat - v_mean) / v_std, (t_feat - t_mean) / t_std


def compute_1_simplex_features(h_0, edge_index_tensor, chunk_size=None):
    """
    1-simplex (edge) features: Hadamard product of endpoint features, h_e = h_u * h_v.

    Args:
        h_0: [num_nodes, feat_dim]
        edge_index_tensor: [num_edges, 2], rows [u, v].

    Returns:
        h_1: [num_edges, feat_dim]
    """
    num_edges = edge_index_tensor.shape[0]
    
    if chunk_size is None or chunk_size <= 0 or chunk_size >= num_edges:
        u_indices = edge_index_tensor[:, 0]
        v_indices = edge_index_tensor[:, 1]
        h_1 = h_0[u_indices] * h_0[v_indices]
        return h_1

    h_1 = torch.empty((num_edges, h_0.shape[1]), dtype=h_0.dtype, device=h_0.device)
    for start in range(0, num_edges, chunk_size):
        end = min(start + chunk_size, num_edges)
        u_indices = edge_index_tensor[start:end, 0]
        v_indices = edge_index_tensor[start:end, 1]
        h_1[start:end] = h_0[u_indices] * h_0[v_indices]
    
    return h_1


def compute_2_simplex_features(h_1, triangles, B2, is_binary=False, chunk_size=None,
                               h_0=None, edge_index_tensor=None):
    """
    2-simplex (triangle) features via max-pool over incident edges (or OR if binary).

    Args:
        h_1: [num_edges, feat_dim]
        triangles: set of (u, v, w) with u < v < w
        B2: sparse COO [num_edges, num_triangles]
        is_binary: if True, treat as binary / OR path inside aggregation
    """
    num_triangles = len(triangles)
    source_features = h_1 if h_1 is not None else h_0
    feat_dim = source_features.shape[1]
    device = source_features.device
    dtype = source_features.dtype

    if num_triangles == 0:
        return torch.empty((0, feat_dim), dtype=dtype, device=device)

    out = torch.full((num_triangles, feat_dim), float('-inf') if not is_binary else 0.0, dtype=dtype, device=device)
    edge_idx_all = B2.indices()[0].to(device)   # [nnz]
    tri_idx_all = B2.indices()[1].to(device)    # [nnz]
    nnz = edge_idx_all.numel()
    if chunk_size is None or chunk_size <= 0:
        chunk_size = nnz

    for start in range(0, nnz, chunk_size):
        end = min(start + chunk_size, nnz)
        edge_idx = edge_idx_all[start:end]
        tri_idx = tri_idx_all[start:end]
        if h_1 is None:
            endpoints = edge_index_tensor[edge_idx]
            h_1_edges = h_0[endpoints[:, 0]] * h_0[endpoints[:, 1]]
        else:
            h_1_edges = h_1[edge_idx]
        if is_binary:
            h_1_edges = (h_1_edges > 0).to(dtype)
        index_2d = tri_idx.unsqueeze(1).expand(-1, feat_dim)
        out.scatter_reduce_(0, index_2d, h_1_edges, reduce='amax')
        del h_1_edges, index_2d, edge_idx, tri_idx
    if not is_binary:
        out[out == float('-inf')] = 0.0
    return out


def compute_all_simplex_features(v_feat, t_feat, edge_index_tensor, triangles, B2, 
                                  is_binary=False, verbose=True, chunk_size=None):
    """
    Compute 0/1/2-simplex features: node concat, edge Hadamard, triangle pool.

    Returns dict with 'h_0', 'h_1', 'h_2'.
    """
    if verbose:
        print("\nComputing simplex features...")
    
    h_0 = compute_0_simplex_features(v_feat, t_feat)
    if verbose:
        print(f"0-simplex (node) shape: {h_0.shape}")
    
    direct_triangle_features = chunk_size is not None and chunk_size > 0
    h_1 = None
    if direct_triangle_features:
        if verbose:
            print(f"1-simplex (edge) shape: computed in chunks of {chunk_size:,}")
    else:
        h_1 = compute_1_simplex_features(h_0, edge_index_tensor)
        if verbose:
            print(f"1-simplex (edge) shape: {h_1.shape}")
    
    h_2 = compute_2_simplex_features(
        h_1, 
        triangles, 
        B2,
        is_binary=is_binary,
        chunk_size=chunk_size,
        h_0=h_0 if direct_triangle_features else None,
        edge_index_tensor=edge_index_tensor if direct_triangle_features else None
    )
    if verbose:
        feat_type = "binary" if is_binary else "continuous"
        print(f"2-simplex (triangle) shape ({feat_type}): {h_2.shape}")
    
    return {
        'h_0': h_0,
        'h_1': h_1,
        'h_2': h_2
    }


# Helper: convert indices to torch.Tensor on global `device` (set in main)
def to_tensor(idx):
    if isinstance(idx, torch.Tensor):
        return idx.to(device)
    elif isinstance(idx, list):
        return torch.LongTensor(idx).to(device)
    elif isinstance(idx, np.ndarray):
        return torch.LongTensor(idx).to(device)
    else:
        return torch.LongTensor(idx).to(device)


def train_model(model, h_0, edge_index, triangles_dict, h_triangles, labels, 
                train_idx, val_idx, device, args):
    """
    Train with optional micro-batching (loss only) and mixed precision.

    Returns dict with best_model_state, best_val_acc, train_history.
    """
    print("\nStarting training...", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_classes_for_metrics = model.classifier[-1].out_features
    class_weights = None
    if getattr(args, 'class_weight', 'none') == 'balanced':
        train_labels = labels[train_idx]
        num_classes_for_loss = model.classifier[-1].out_features
        counts = torch.bincount(train_labels, minlength=num_classes_for_loss).float()
        present = counts > 0
        weights = torch.zeros_like(counts)
        weights[present] = counts[present].sum() / counts[present]
        class_weight_power = getattr(args, 'class_weight_power', 1.0)
        if class_weight_power != 1.0:
            weights[present] = weights[present].pow(class_weight_power)
        weights = weights / weights.mean().clamp_min(1e-12)
        class_weights = weights.to(device)
        print(f"Class-balanced CE weights enabled (power={class_weight_power})", flush=True)
    
    use_amp = args.mixed_precision and device.type == 'cuda' and torch.cuda.is_available()
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print("Mixed precision (AMP) enabled", flush=True)
    else:
        scaler = None
    
    if not use_amp and device.type == 'cuda' and torch.cuda.is_available():
        print("Tip: enable --mixed_precision to save GPU memory", flush=True)
    
    features_on_cpu = device.type == 'cuda' and h_0.device.type == 'cpu'
    
    train_batch_size = args.train_batch_size
    if train_batch_size is None and features_on_cpu:
        num_nodes = h_0.shape[0]
        if num_nodes > 90000:
            train_batch_size = 200
        elif num_nodes > 50000:
            train_batch_size = 300
        elif len(train_idx) > 50000:
            train_batch_size = 400
        elif len(train_idx) > 20000:
            train_batch_size = 600
        elif len(train_idx) > 10000:
            train_batch_size = 800
        else:
            train_batch_size = 1000
        print(f"\nAuto train_batch_size={train_batch_size} (num_nodes={num_nodes:,}, save GPU mem)", flush=True)
    
    use_batch_training = train_batch_size is not None and train_batch_size < len(train_idx)
    
    if use_batch_training:
        print(f"Micro-batch training (grad accumulation), train_batch_size={train_batch_size}", flush=True)
        print(f"  Note: train_batch_size only affects which nodes contribute to loss; forward is still full graph.", flush=True)
        print(f"  To cut peak VRAM: --batch_size_triangles, --keep_data_cpu, --use_subgraph, --mixed_precision", flush=True)
        num_train_batches = (len(train_idx) + train_batch_size - 1) // train_batch_size
        print(f"Train batches per epoch: {num_train_batches}", flush=True)
        if num_train_batches > 1000:
            print(f"Warning: many train batches ({num_train_batches}); epochs may be slow", flush=True)
            print(f"  Try larger --train_batch_size (e.g. --train_batch_size {min(1000, len(train_idx) // 10)})", flush=True)
    else:
        num_train_batches = 1
        print("Full-graph training (single loss pass)", flush=True)
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    train_history = {
        'loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    oom_count = 0
    current_train_batch_size = train_batch_size if use_batch_training else len(train_idx)

    if getattr(args, 'keep_init_best', False):
        print("\nScoring current weights before fine-tuning (--keep_init_best)...", flush=True)
        if getattr(args, 'skip_train_eval', False):
            train_acc = 0.0
        else:
            train_acc = evaluate_batched(
                model, h_0, edge_index, triangles_dict, h_triangles, labels,
                train_idx, device, args.eval_batch_size, features_on_cpu
            )
        val_metrics = evaluate_metrics_batched(
            model, h_0, edge_index, triangles_dict, h_triangles, labels,
            val_idx, num_classes_for_metrics, device, args, features_on_cpu
        )
        best_val_acc = val_metrics.get(args.selection_metric, val_metrics['accuracy'])
        best_model_state = copy.deepcopy(model.state_dict())
        train_history['train_acc'].append(train_acc)
        train_history['val_acc'].append(val_metrics['accuracy'])
        print(f"Initial checkpoint | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | Val AUC: {val_metrics['auc']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | Best {args.selection_metric}: {best_val_acc:.4f}", flush=True)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{args.epochs}...", flush=True)
        if use_batch_training:
            print(f"  {num_train_batches} train batches (size {current_train_batch_size})", flush=True)
        
        model.train()
        if getattr(args, 'freeze_backbone', False):
            model.input_proj.eval()
            model.message_passing_layers.eval()
            model.norm_layers.eval()
            model.norm_before_cls.eval()
        optimizer.zero_grad(set_to_none=True)
        
        try:
            total_loss = 0.0
            
            for batch_idx in range(num_train_batches):
                if use_batch_training:
                    progress_interval = max(100, num_train_batches // 20)
                    if (batch_idx + 1) % progress_interval == 0 or batch_idx == 0:
                        progress_pct = (batch_idx + 1) / num_train_batches * 100
                        elapsed = time.time() - epoch_start_time
                        print(f"  Epoch {epoch+1} progress: {batch_idx+1}/{num_train_batches} ({progress_pct:.1f}%) | "
                              f"elapsed {elapsed:.1f}s", flush=True)
                if use_batch_training:
                    start_idx = batch_idx * current_train_batch_size
                    end_idx = min((batch_idx + 1) * current_train_batch_size, len(train_idx))
                    batch_train_idx = train_idx[start_idx:end_idx]
                else:
                    batch_train_idx = train_idx
                
                if batch_idx > 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                if features_on_cpu:
                    if batch_idx > 0:
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    h_0_gpu = h_0.to(
                        device,
                        dtype=torch.float16 if use_amp else h_0.dtype,
                        non_blocking=False,
                    )
                    h_triangles_gpu = h_triangles
                else:
                    h_0_gpu = h_0
                    h_triangles_gpu = h_triangles
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        if args.gradient_checkpointing:
                            logits = model(h_0_gpu, edge_index, triangles_dict, h_triangles_gpu)
                        else:
                            logits = model(h_0_gpu, edge_index, triangles_dict, h_triangles_gpu)
                        batch_loss = compute_loss(
                            logits,
                            labels,
                            batch_train_idx,
                            weight_decay=0.0,
                            model=None,
                            label_smoothing=args.label_smoothing,
                            class_weights=class_weights,
                        )
                        if use_batch_training:
                            batch_loss = batch_loss / num_train_batches
                    
                    scaler.scale(batch_loss).backward()
                    total_loss += batch_loss.item() * (num_train_batches if use_batch_training else 1)
                else:
                    logits = model(h_0_gpu, edge_index, triangles_dict, h_triangles_gpu)
                    batch_loss = compute_loss(
                        logits,
                        labels,
                        batch_train_idx,
                        weight_decay=0.0,
                        model=None,
                        label_smoothing=args.label_smoothing,
                        class_weights=class_weights,
                    )
                    if use_batch_training:
                        batch_loss = batch_loss / num_train_batches
                    
                    batch_loss.backward()
                    total_loss += batch_loss.item() * (num_train_batches if use_batch_training else 1)
                
                del logits, batch_loss
                
                if features_on_cpu:
                    del h_0_gpu
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                if features_on_cpu and batch_idx > 0:
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
            
            if args.weight_decay > 0:
                l2_reg = torch.tensor(0.0, device=device)
                for param in model.parameters():
                    l2_reg += torch.sum(param ** 2)
                l2_reg = args.weight_decay * l2_reg
                if use_amp:
                    scaler.scale(l2_reg).backward()
                else:
                    l2_reg.backward()
                total_loss += l2_reg.item()
            
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            loss_value = total_loss
            
            eval_every = max(1, args.eval_every)
            do_eval = epoch < 20 or (epoch + 1) % eval_every == 0 or epoch == args.epochs - 1
            if do_eval:
                if getattr(args, 'skip_train_eval', False):
                    train_acc = train_history['train_acc'][-1] if len(train_history['train_acc']) > 0 else 0.0
                else:
                    train_acc = evaluate_batched(
                        model, h_0, edge_index, triangles_dict, h_triangles, labels,
                        train_idx, device, args.eval_batch_size, features_on_cpu
                    )
                val_metrics = evaluate_metrics_batched(
                    model, h_0, edge_index, triangles_dict, h_triangles, labels,
                    val_idx, num_classes_for_metrics, device, args, features_on_cpu
                )
                val_acc = val_metrics['accuracy']
                selection_metric_value = val_metrics.get(args.selection_metric, val_acc)
            else:
                train_acc = train_history['train_acc'][-1] if len(train_history['train_acc']) > 0 else 0.0
                val_acc = best_val_acc
                selection_metric_value = best_val_acc
            
            train_history['loss'].append(loss_value)
            train_history['train_acc'].append(train_acc)
            train_history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            
            if do_eval:
                print(f"Epoch {epoch+1:4d}/{args.epochs} | Loss: {loss_value:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Val AUC: {val_metrics['auc']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
                      f"Time: {epoch_time:.2f}s | Best {args.selection_metric}: {best_val_acc:.4f}", flush=True)
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"Progress: Epoch {epoch+1}/{args.epochs} (Loss: {loss_value:.4f}, Best {args.selection_metric}: {best_val_acc:.4f})", flush=True)
            
            if selection_metric_value > best_val_acc:
                best_val_acc = selection_metric_value
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    total_time = time.time() - start_time
                    print(f"\nEarly stopping at epoch {epoch+1} (total time: {total_time:.2f}s)", flush=True)
                    break
        
        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA" in str(e):
                oom_count += 1
                print(f"\nEpoch {epoch+1}: CUDA OOM: {e}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                if not use_batch_training:
                    use_batch_training = True
                    current_train_batch_size = min(400, len(train_idx))
                    num_train_batches = (len(train_idx) + current_train_batch_size - 1) // current_train_batch_size
                    print(f"Full-graph OOM; switching to micro-batches: size={current_train_batch_size}, num_batches={num_train_batches}", flush=True)
                    oom_count = 0
                elif use_batch_training and current_train_batch_size > 100:
                    old_batch_size = current_train_batch_size
                    current_train_batch_size = max(100, current_train_batch_size // 2)
                    num_train_batches = (len(train_idx) + current_train_batch_size - 1) // current_train_batch_size
                    print(f"OOM: train_batch_size {old_batch_size} -> {current_train_batch_size} (next epoch)", flush=True)
                    print(f"   New num_batches: {num_train_batches}", flush=True)
                    oom_count = 0
                
                if oom_count >= 5:
                    print("\nRepeated OOM; options that reduce peak VRAM:", flush=True)
                    print("   1. --keep_data_cpu", flush=True)
                    print("   2. --use_subgraph", flush=True)
                    print("   3. Smaller --batch_size_triangles (e.g. 1000-2000)", flush=True)
                    print("   4. --mixed_precision", flush=True)
                    print("   5. Smaller --hidden_dim (e.g. 32)", flush=True)
                    print("   Note: --train_batch_size only scales loss; it does not cut peak VRAM.", flush=True)
                    print("Stopping this run after repeated OOM before any stable epoch.", flush=True)
                    break
                print("Skipping this epoch after cache clear; smaller batch next time.", flush=True)
                continue
            else:
                raise e
        except Exception as e:
            print(f"\nEpoch {epoch+1}: error: {e}", flush=True)
            print("Skipping epoch, continuing...", flush=True)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue
    
    total_time = time.time() - start_time
    print(f"\nTraining done. Total time: {total_time:.2f}s ({total_time/60:.2f} min)", flush=True)
    print(f"Best validation {args.selection_metric}: {best_val_acc:.4f}", flush=True)
    
    return {
        'best_model_state': best_model_state,
        'best_val_acc': best_val_acc,
        'train_history': train_history
    }


def evaluate_batched(model, h_0, edge_index, triangles_dict, h_triangles, labels, idx, 
                     device, eval_batch_size=None, features_on_cpu=False):
    """
    Evaluate accuracy; optional batching over idx after one full-graph forward.

    Returns float accuracy.
    """
    model.eval()
    
    if eval_batch_size is None or eval_batch_size >= len(idx):
        if features_on_cpu:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            h_0_gpu = h_0.to(device, non_blocking=False)
            h_triangles_gpu = h_triangles
        else:
            h_0_gpu = h_0
            h_triangles_gpu = h_triangles
        
        with torch.no_grad():
            logits = model(h_0_gpu, edge_index, triangles_dict, h_triangles_gpu)
            pred = logits[idx].argmax(dim=1)
            acc = (pred == labels[idx]).float().mean().item()
        
        if features_on_cpu:
            del h_0_gpu, logits, pred
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        
        return acc
    
    if features_on_cpu:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        h_0_gpu = h_0.to(device, non_blocking=False)
        h_triangles_gpu = h_triangles
    else:
        h_0_gpu = h_0
        h_triangles_gpu = h_triangles
    
    with torch.no_grad():
        logits = model(h_0_gpu, edge_index, triangles_dict, h_triangles_gpu)
        num_batches = (len(idx) + eval_batch_size - 1) // eval_batch_size
        all_pred = []
        all_labels = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * eval_batch_size
            end_idx = min((batch_idx + 1) * eval_batch_size, len(idx))
            batch_idx_tensor = idx[start_idx:end_idx]
            batch_pred = logits[batch_idx_tensor].argmax(dim=1)
            batch_labels = labels[batch_idx_tensor]
            all_pred.append(batch_pred.cpu())
            all_labels.append(batch_labels.cpu())
        all_pred = torch.cat(all_pred, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        acc = (all_pred == all_labels).float().mean().item()
    
    if features_on_cpu:
        del h_0_gpu, logits
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return acc


def evaluate_metrics_batched(model, h_0, edge_index, triangles_dict, h_triangles, labels,
                             idx, num_classes, device, args, features_on_cpu=False):
    """Evaluate full metric dict on a subset; used for checkpoint selection."""
    eval_batch_size = getattr(args, 'eval_batch_size', None)
    y_true, y_pred, logits_idx = get_predictions_and_logits(
        model, h_0, edge_index, triangles_dict, h_triangles, labels,
        idx, device, eval_batch_size, features_on_cpu
    )
    return calculate_metrics(
        y_true,
        y_pred,
        logits_idx,
        num_classes=num_classes,
        average=getattr(args, 'metric_average', 'weighted'),
    )


def get_predictions_and_logits(model, h_0, edge_index, triangles_dict, h_triangles, labels, idx,
                               device, eval_batch_size=None, features_on_cpu=False):
    """
    Predictions and logits for idx (CPU tensors). logits shape [len(idx), num_classes].
    """
    model.eval()
    if features_on_cpu:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        h_0_gpu = h_0.to(device, non_blocking=False)
        h_triangles_gpu = h_triangles
    else:
        h_0_gpu = h_0
        h_triangles_gpu = h_triangles

    with torch.no_grad():
        logits = model(h_0_gpu, edge_index, triangles_dict, h_triangles_gpu)
        if eval_batch_size is None or eval_batch_size >= len(idx):
            logits_idx = logits[idx].cpu()
        else:
            chunks = []
            for start in range(0, len(idx), eval_batch_size):
                end = min(start + eval_batch_size, len(idx))
                chunks.append(logits[idx[start:end]].cpu())
            logits_idx = torch.cat(chunks, dim=0)
    y_true = labels[idx].cpu()
    y_pred = logits_idx.argmax(dim=1)
    if features_on_cpu:
        del h_0_gpu, logits
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return y_true, y_pred, logits_idx


def parse_float_grid(grid_text):
    values = []
    for raw_value in str(grid_text).split(','):
        raw_value = raw_value.strip()
        if raw_value:
            values.append(float(raw_value))
    return values or [0.0]


def calibrate_classwise_logit_bias(val_labels, val_logits, num_classes, average='macro',
                                   selection_metric='f1', alpha_grid=None, eps=1e-6):
    """
    Estimate additive per-class logit bias from validation labels and validation logits.

    The bias matches the validation label prior against the model's mean predicted
    probability, then picks a scalar strength on validation metrics only.
    """
    if alpha_grid is None:
        alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.cpu()
    if isinstance(val_logits, torch.Tensor):
        val_logits = val_logits.cpu()

    val_labels = val_labels.long()
    val_logits = val_logits.float()
    val_probs = torch.softmax(val_logits, dim=1)
    label_counts = torch.bincount(val_labels, minlength=num_classes).float()
    label_prior = (label_counts + eps) / (label_counts.sum() + eps * num_classes)
    pred_prior = val_probs.mean(dim=0).clamp_min(eps)
    raw_bias = torch.log(label_prior.clamp_min(eps)) - torch.log(pred_prior)

    best_alpha = 0.0
    best_metrics = None
    best_score = None
    best_bias = torch.zeros(num_classes)
    for alpha in alpha_grid:
        bias = raw_bias * float(alpha)
        calibrated_logits = val_logits + bias.view(1, -1)
        calibrated_pred = calibrated_logits.argmax(dim=1)
        metrics = calculate_metrics(
            val_labels,
            calibrated_pred,
            calibrated_logits,
            num_classes=num_classes,
            average=average,
        )
        score = metrics.get(selection_metric, metrics['f1'])
        if best_score is None or score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_metrics = metrics
            best_bias = bias.clone()

    return {
        'bias': best_bias,
        'alpha': best_alpha,
        'raw_bias': raw_bias,
        'val_metrics': best_metrics,
        'label_prior': label_prior,
        'pred_prior': pred_prior,
    }


def calibrate_classwise_logit_bias_coordinate(val_labels, val_logits, num_classes,
                                              average='macro', selection_metric='f1',
                                              bias_grid=None, rounds=3):
    """Coordinate-search class-wise additive logits on validation metrics only."""
    if bias_grid is None:
        bias_grid = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.cpu()
    if isinstance(val_logits, torch.Tensor):
        val_logits = val_logits.cpu()

    val_labels = val_labels.long()
    val_logits = val_logits.float()
    val_labels_np = val_labels.numpy()

    def _fast_score(pred):
        pred_np = pred.numpy()
        if selection_metric == 'accuracy':
            return float((val_labels_np == pred_np).mean())
        if selection_metric == 'precision':
            return float(precision_score(val_labels_np, pred_np, average=average, zero_division=0))
        if selection_metric == 'recall':
            return float(recall_score(val_labels_np, pred_np, average=average, zero_division=0))
        return float(f1_score(val_labels_np, pred_np, average=average, zero_division=0))

    bias = torch.zeros(num_classes)
    best_logits = val_logits
    best_pred = best_logits.argmax(dim=1)
    best_score = _fast_score(best_pred)

    for _ in range(max(1, rounds)):
        improved = False
        for cls in range(num_classes):
            cls_best_delta = 0.0
            cls_best_score = best_score
            for delta in bias_grid:
                if float(delta) == 0.0:
                    continue
                trial_bias = bias.clone()
                trial_bias[cls] += float(delta)
                trial_logits = val_logits + trial_bias.view(1, -1)
                trial_pred = trial_logits.argmax(dim=1)
                trial_score = _fast_score(trial_pred)
                if trial_score > cls_best_score:
                    cls_best_score = trial_score
                    cls_best_delta = float(delta)
            if cls_best_delta != 0.0:
                bias[cls] += cls_best_delta
                best_score = cls_best_score
                improved = True
        if not improved:
            break

    best_logits = val_logits + bias.view(1, -1)
    best_pred = best_logits.argmax(dim=1)
    best_metrics = calculate_metrics(
        val_labels,
        best_pred,
        best_logits,
        num_classes=num_classes,
        average=average,
    )

    return {
        'bias': bias,
        'alpha': None,
        'raw_bias': bias.clone(),
        'val_metrics': best_metrics,
        'label_prior': None,
        'pred_prior': None,
    }


def evaluate_model(model, h_0, edge_index, triangles_dict, h_triangles, labels,
                   train_idx, val_idx, test_idx, num_classes, best_model_state=None, device=None, args=None):
    """
    Load best checkpoint if given, then report train/val/test metrics.
    """
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best checkpoint for evaluation")
    
    print("\nEvaluating...")
    
    features_on_cpu = device.type == 'cuda' and h_0.device.type == 'cpu'
    
    if device is None:
            device = h_0.device
    eval_batch_size = getattr(args, 'eval_batch_size', None) if args is not None else None
    
    if getattr(args, 'skip_train_eval', False):
        train_acc = 0.0
    else:
        train_acc = evaluate_batched(
            model, h_0, edge_index, triangles_dict, h_triangles, labels,
            train_idx, device, eval_batch_size, features_on_cpu
        )
    print(f"Train accuracy: {train_acc:.4f}")
    
    val_true, val_pred, logits_val = get_predictions_and_logits(
        model, h_0, edge_index, triangles_dict, h_triangles, labels,
        val_idx, device, eval_batch_size, features_on_cpu
    )
    val_acc = (val_pred == val_true).float().mean().item()
    print(f"Val accuracy: {val_acc:.4f}")
    
    y_true, y_pred, logits_test = get_predictions_and_logits(
        model, h_0, edge_index, triangles_dict, h_triangles, labels,
        test_idx, device, eval_batch_size, features_on_cpu
    )
    metric_average = getattr(args, 'metric_average', 'weighted') if args is not None else 'weighted'
    test_metrics = calculate_metrics(y_true, y_pred, logits_test, num_classes=num_classes, average=metric_average)
    test_acc = test_metrics['accuracy']
    print(f"Test - Accuracy: {test_acc:.4f}, Precision: {test_metrics['precision']:.4f}, "
          f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
    log_per_class_metrics(y_true, y_pred, num_classes, logits=logits_test)

    calibration_result = None
    calibration_mode = getattr(args, 'logit_bias_calibration', 'none') if args is not None else 'none'
    if calibration_mode in {'val_prior', 'val_coordinate'}:
        selection_metric = getattr(args, 'logit_bias_selection_metric', 'f1')
        if calibration_mode == 'val_coordinate':
            bias_grid = parse_float_grid(getattr(args, 'logit_bias_grid', '-2,-1.5,-1,-0.5,-0.25,0,0.25,0.5,1,1.5,2'))
            rounds = getattr(args, 'logit_bias_rounds', 3)
            calibration_result = calibrate_classwise_logit_bias_coordinate(
                val_true,
                logits_val,
                num_classes,
                average=metric_average,
                selection_metric=selection_metric,
                bias_grid=bias_grid,
                rounds=rounds,
            )
        else:
            alpha_grid = parse_float_grid(getattr(args, 'logit_bias_alpha_grid', '0,0.25,0.5,0.75,1.0'))
            eps = getattr(args, 'logit_bias_eps', 1e-6)
            calibration_result = calibrate_classwise_logit_bias(
                val_true,
                logits_val,
                num_classes,
                average=metric_average,
                selection_metric=selection_metric,
                alpha_grid=alpha_grid,
                eps=eps,
            )
        bias = calibration_result['bias']
        calibrated_test_logits = logits_test.float() + bias.view(1, -1)
        calibrated_pred = calibrated_test_logits.argmax(dim=1)
        calibrated_metrics = calculate_metrics(
            y_true,
            calibrated_pred,
            calibrated_test_logits,
            num_classes=num_classes,
            average=metric_average,
        )
        calibration_result['test_metrics'] = calibrated_metrics
        calibration_result['bias'] = bias.tolist()
        calibration_result['raw_bias'] = calibration_result['raw_bias'].tolist()
        if calibration_result['label_prior'] is not None:
            calibration_result['label_prior'] = calibration_result['label_prior'].tolist()
        if calibration_result['pred_prior'] is not None:
            calibration_result['pred_prior'] = calibration_result['pred_prior'].tolist()
        test_metrics = calibrated_metrics
        test_acc = calibrated_metrics['accuracy']
        print("\nValidation-based class-wise logit bias calibration")
        alpha = calibration_result['alpha']
        alpha_text = "n/a" if alpha is None else f"{alpha:.4f}"
        print(f"  mode={calibration_mode}, alpha={alpha_text}, selected_by=val_{selection_metric}")
        print(f"  calibrated val {selection_metric}: {calibration_result['val_metrics'][selection_metric]:.4f}")
        print(f"Calibrated Test - Accuracy: {test_acc:.4f}, Precision: {test_metrics['precision']:.4f}, "
              f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'test_metrics': test_metrics,
        'logit_bias_calibration': calibration_result,
    }



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='grocery',
                        choices=['grocery', 'toys', 'ele-fashion', 'books-nc', 'books'],
                        help='Dataset: grocery, toys, ele-fashion, or books-nc/books')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Data directory; overrides --dataset if set')
    parser.add_argument('--feat_path', type=str, default=None, help='Optional merged feature file')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Train ratio (MAGB split)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Val ratio (MAGB split)')
    parser.add_argument('--split_sample_ratio', type=float, default=1.0,
                        help='Optional stratified subsample ratio applied to train/val/test splits after loading')
    
    parser.add_argument('--model_type', type=str, default='agsp', choices=['agsp'],
                        help='Run the original AGSP simplicial GNN')
    parser.add_argument('--metric_average', type=str, default='weighted',
                        choices=['weighted', 'macro', 'micro'],
                        help='Average used for Precision/Recall/F1')
    parser.add_argument('--selection_metric', type=str, default='f1',
                        choices=['accuracy', 'auc', 'f1', 'precision', 'recall'],
                        help='Validation metric used to select the best AGSP checkpoint')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate validation metrics every N epochs after the first 20 epochs')
    parser.add_argument('--hidden_dim', type=int, default=352, help='Hidden dim')
    parser.add_argument('--num_layers', type=int, default=2, help='Message passing layers (default 2)')
    parser.add_argument('--lambda_high', type=float, default=2.0, help='Higher-order strength')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout for AGSP')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay (default 1e-4)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Cross-entropy label smoothing for AGSP training')
    parser.add_argument('--class_weight', type=str, default='balanced', choices=['none', 'balanced'],
                        help='Optional class weighting for AGSP cross-entropy')
    parser.add_argument('--class_weight_power', type=float, default=0.5,
                        help='Exponent applied to balanced class weights; 0.5 gives sqrt-balanced weights')
    parser.add_argument('--epochs', type=int, default=220, help='Max epochs')
    parser.add_argument('--patience', type=int, default=45, help='Early stopping patience')
    parser.add_argument('--repeat_times', '--times', dest='repeat_times', type=int, default=1,
                        help='Repeat runs for mean±std on test')
    parser.add_argument('--checkpoint_tag', type=str, default=None,
                        help='Optional tag appended to saved AGSP checkpoint names')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--split_seed', type=int, default=45, help='Seed for random train/val/test splits when no official split is provided')
    parser.add_argument('--non_deterministic', action='store_true',
                        help='Keep fixed seeds but do not force deterministic CUDA kernels')
    parser.add_argument('--device', type=str, default=None,
                        help='Device string, e.g. cuda:0 or cpu. Default: cuda:0 if available else cpu')
    
    parser.add_argument('--batch_size_triangles', type=int, default=1000, 
                        help='Node batch size for triangle aggregation (GPU mem); None disables. Try 1000-5000')
    parser.add_argument('--batch_size_edges', type=int, default=200000,
                        help='Target-node batch size for pairwise edge attention; requires --sort_edges_by_target')
    parser.add_argument('--sort_edges_by_target', action='store_true', default=True,
                        help='Sort edge_index by target node so pairwise edge attention can be node-batched')
    parser.add_argument('--auto_batch_size', action='store_true', 
                        help='Auto-set triangle batch size from triangle count')
    parser.add_argument('--train_batch_size', type=int, default=None,
                        help='Micro-batch for loss only (full-graph forward). For VRAM use --batch_size_triangles, --keep_data_cpu, --use_subgraph')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='Slice eval indices; None = full idx. Try 2000-10000')
    parser.add_argument('--keep_data_cpu', action='store_true',
                        help='Keep features on CPU; load to GPU on demand')
    parser.add_argument('--preload_triangles_gpu', action='store_true',
                        help='With --keep_data_cpu, keep h_0 on CPU but preload triangle features to GPU in fp16 under AMP')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='AMP training (~50%% less GPU memory)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Gradient checkpointing (slower, saves memory)')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='Optional AGSP checkpoint path used to initialize model weights before training')
    parser.add_argument('--eval_only', action='store_true',
                        help='Load --init_checkpoint, run evaluation/calibration, and exit without training')
    parser.add_argument('--keep_init_best', action='store_true',
                        help='When using --init_checkpoint, treat initial weights as the first best checkpoint')
    parser.add_argument('--skip_train_eval', action='store_true', default=True,
                        help='Skip train accuracy forward passes during AGSP tuning; validation metrics still run')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze AGSP message-passing backbone and train only the classifier head')
    parser.add_argument('--two_stage', action='store_true',
                        help='Run a second fine-tuning stage after loading the first-stage best checkpoint')
    parser.add_argument('--stage2_epochs', type=int, default=None,
                        help='Epochs for stage 2; default keeps --epochs')
    parser.add_argument('--stage2_patience', type=int, default=None,
                        help='Early stopping patience for stage 2; default keeps --patience')
    parser.add_argument('--stage2_lr', type=float, default=None,
                        help='Learning rate for stage 2; default keeps --lr')
    parser.add_argument('--stage2_selection_metric', type=str, default=None,
                        choices=['accuracy', 'auc', 'f1', 'precision', 'recall'],
                        help='Validation metric for stage 2; default keeps --selection_metric')
    parser.add_argument('--stage2_class_weight_power', type=float, default=None,
                        help='Class-weight power for stage 2; default keeps --class_weight_power')
    parser.add_argument('--stage2_label_smoothing', type=float, default=None,
                        help='Label smoothing for stage 2; default keeps --label_smoothing')
    parser.add_argument('--logit_bias_calibration', type=str, default='val_coordinate',
                        choices=['none', 'val_prior', 'val_coordinate'],
                        help='Post-hoc class-wise logit bias calibration; books profile enables val_prior')
    parser.add_argument('--logit_bias_alpha_grid', type=str, default='0,0.25,0.5,0.75,1.0',
                        help='Comma-separated calibration strengths selected on validation set')
    parser.add_argument('--logit_bias_grid', type=str, default='-2,-1.5,-1,-0.5,-0.25,0,0.25,0.5,1,1.5,2',
                        help='Comma-separated per-class bias deltas for coordinate logit-bias search')
    parser.add_argument('--logit_bias_rounds', type=int, default=3,
                        help='Coordinate-search rounds for validation-based logit-bias calibration')
    parser.add_argument('--logit_bias_selection_metric', type=str, default='f1',
                        choices=['accuracy', 'auc', 'f1', 'precision', 'recall'],
                        help='Validation metric used to choose logit-bias strength')
    parser.add_argument('--logit_bias_eps', type=float, default=1e-6,
                        help='Smoothing epsilon for validation prior calibration')
    parser.add_argument('--no_books_profile', action='store_true',
                        help='Disable the fixed Books AGSP profile')
    parser.add_argument('--no_dataset_profile', action='store_true',
                        help='Disable tuned dataset defaults for Grocery/Toys/Ele-fashion')
    parser.add_argument('--normalize_features', action='store_true',
                        help='Standardize AGSP text/vision features from train split before simplex construction')
    parser.add_argument('--simplex_chunk_size', type=int, default=None,
                        help='Chunk size for simplex feature construction; saves memory on large graphs')
    parser.add_argument('--simplex_cache_path', type=str, default=None,
                        help='Optional torch cache path for computed h_0/h_2 simplex features')
    parser.add_argument('--simplex_cache_fp16', action='store_true',
                        help='Store/use h_2 simplex cache in fp16 to reduce CPU memory and IO for AMP runs')
    parser.add_argument('--compact_split_classes', action='store_true',
                        help='Remap classes present in train/val/test to contiguous ids and drop absent output classes')
    parser.add_argument('--max_triangles_per_node', type=int, default=None,
                        help='Optional cap for node-to-triangle high-order neighborhoods; default keeps all triangles')
    parser.add_argument('--max_edges_per_node', type=int, default=None,
                        help='Optional cap for pairwise incoming edges per target node; default keeps all edges')
    
    parser.add_argument('--use_subgraph', action='store_true',
                        help='k-hop subgraph around centers (saves memory)')
    parser.add_argument('--subgraph_hops', type=int, default=2,
                        help='Subgraph hops (default 2 for 2-layer GNN)')
    parser.add_argument('--subgraph_include_val_test', action='store_true',
                        help='Also use val/test nodes as subgraph centers')
    
    args = parser.parse_args()
    apply_agsp_dataset_profile(args)
    apply_books_agsp_profile(args)
    print_effective_hyperparameters(args)
    
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"Warning: requested {args.device}, but this PyTorch build has no CUDA; falling back to CPU")
            device = torch.device("cpu")
    print(f"Device: {device}")
    set_seed(args.seed, deterministic=not args.non_deterministic)
    
    load_device = torch.device("cpu") if args.keep_data_cpu else device
    raw_data = load_raw_dataset(
        dataset=args.dataset,
        device=load_device,
        data_path=args.data_path,
        feat_path=args.feat_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        load_edges=True,
        split_seed=args.split_seed,
    )
    if args.keep_data_cpu:
        print(f"Data loaded on CPU (--keep_data_cpu)")
    
    print(f"\nRaw data summary:")
    print(f"Edges: {len(raw_data['edges'])}")
    print(f"Nodes: {raw_data['num_nodes']}")
    print(f"Classes: {raw_data['num_classes']}")
    print(f"Train nodes: {len(raw_data['splits']['train_idx'])}")
    print(f"Val nodes: {len(raw_data['splits']['val_idx'])}")
    print(f"Test nodes: {len(raw_data['splits']['test_idx'])}")
    print(f"Vision feat shape: {raw_data['v_feat'].shape}")
    print(f"Text feat shape: {raw_data['t_feat'].shape}")


    if isinstance(raw_data['edges'], torch.Tensor):
        edge_index_full = raw_data['edges'].clone()
    elif isinstance(raw_data['edges'], np.ndarray):
        edge_index_full = torch.from_numpy(raw_data['edges'])
    else:
        edge_index_full = torch.tensor(raw_data['edges'], dtype=torch.long)
    
    train_idx_raw = raw_data['splits']['train_idx']
    val_idx_raw = raw_data['splits']['val_idx']
    test_idx_raw = raw_data['splits']['test_idx']

    if args.split_sample_ratio < 1.0:
        print(f"\nApplying stratified split sampling: ratio={args.split_sample_ratio}")
        train_idx_raw = sample_split_by_ratio(raw_data['labels'], train_idx_raw, args.split_sample_ratio, args.split_seed, raw_data['num_classes'])
        val_idx_raw = sample_split_by_ratio(raw_data['labels'], val_idx_raw, args.split_sample_ratio, args.split_seed + 1, raw_data['num_classes'])
        test_idx_raw = sample_split_by_ratio(raw_data['labels'], test_idx_raw, args.split_sample_ratio, args.split_seed + 2, raw_data['num_classes'])
        raw_data['splits'] = {
            'train_idx': train_idx_raw,
            'val_idx': val_idx_raw,
            'test_idx': test_idx_raw
        }
        print(f"Sampled train/val/test: {len(train_idx_raw):,}/{len(val_idx_raw):,}/{len(test_idx_raw):,}")

    if args.compact_split_classes:
        compact_labels_to_split_classes(raw_data)
        train_idx_raw = raw_data['splits']['train_idx']
        val_idx_raw = raw_data['splits']['val_idx']
        test_idx_raw = raw_data['splits']['test_idx']
    
    subgraph_info = None
    original_num_nodes = raw_data['num_nodes']
    if args.use_subgraph:
        print(f"\n{'='*60}")
        print("Subgraph sampling: {}-hop around centers...".format(args.subgraph_hops))
        print(f"{'='*60}")
        
        if args.subgraph_include_val_test:
            center_nodes = torch.cat([
                torch.tensor(train_idx_raw, dtype=torch.long),
                torch.tensor(val_idx_raw, dtype=torch.long),
                torch.tensor(test_idx_raw, dtype=torch.long)
            ]).unique()
            print(f"Centers: train+val+test ({len(center_nodes):,} nodes)")
        else:
            center_nodes = torch.tensor(train_idx_raw, dtype=torch.long)
            print(f"Centers: train only ({len(center_nodes):,} nodes)")
            print("   Val/test nodes included if within {} hops of train.".format(args.subgraph_hops))
        
        subgraph_info = extract_k_hop_subgraph(
            center_nodes=center_nodes,
            edge_index=edge_index_full,
            num_nodes=raw_data['num_nodes'],
            k=args.subgraph_hops,
            return_node_mapping=True
        )
        
        print(f"Full graph nodes: {raw_data['num_nodes']:,}")
        print(f"Subgraph nodes: {subgraph_info['num_subgraph_nodes']:,}")
        print(f"Node reduction: {(1 - subgraph_info['num_subgraph_nodes'] / raw_data['num_nodes']) * 100:.1f}%")
        print(f"Full graph edges: {edge_index_full.shape[0]:,}")
        print(f"Subgraph edges: {subgraph_info['num_subgraph_edges']:,}")
        print(f"Edge reduction: {(1 - subgraph_info['num_subgraph_edges'] / edge_index_full.shape[0]) * 100:.1f}%")
        
        edge_index_for_simplicial = subgraph_info['subgraph_edge_index']
        num_nodes_subgraph = subgraph_info['num_subgraph_nodes']
        
        node_mapping = subgraph_info['node_mapping']
        train_idx_subgraph = []
        val_idx_subgraph = []
        test_idx_subgraph = []
        
        for idx in train_idx_raw:
            idx_int = int(idx.item()) if hasattr(idx, 'item') else int(idx)
            if idx_int in node_mapping:
                train_idx_subgraph.append(node_mapping[idx_int])
        for idx in val_idx_raw:
            idx_int = int(idx.item()) if hasattr(idx, 'item') else int(idx)
            if idx_int in node_mapping:
                val_idx_subgraph.append(node_mapping[idx_int])
        for idx in test_idx_raw:
            idx_int = int(idx.item()) if hasattr(idx, 'item') else int(idx)
            if idx_int in node_mapping:
                test_idx_subgraph.append(node_mapping[idx_int])
        
        train_idx_subgraph = torch.tensor(train_idx_subgraph, dtype=torch.long)
        val_idx_subgraph = torch.tensor(val_idx_subgraph, dtype=torch.long)
        test_idx_subgraph = torch.tensor(test_idx_subgraph, dtype=torch.long)
        
        print(f"Train in subgraph: {len(train_idx_subgraph):,} / {len(train_idx_raw):,}")
        print(f"Val in subgraph: {len(val_idx_subgraph):,} / {len(val_idx_raw):,}")
        print(f"Test in subgraph: {len(test_idx_subgraph):,} / {len(test_idx_raw):,}")
        
        if len(train_idx_subgraph) < len(train_idx_raw):
            missing_count = len(train_idx_raw) - len(train_idx_subgraph)
            print(f"ERROR: {missing_count} train nodes missing from subgraph (unexpected).")
            print(f"   Train nodes should be centers; check subgraph code.")
        else:
            print(f"OK: all {len(train_idx_subgraph):,} train nodes in subgraph")
        
        if len(val_idx_subgraph) < len(val_idx_raw):
            missing_count = len(val_idx_raw) - len(val_idx_subgraph)
            missing_pct = (missing_count / len(val_idx_raw)) * 100
            print(f"Warning: {missing_count} val nodes ({missing_pct:.1f}%) not in subgraph")
            if missing_pct > 10:
                print(f"   Low val acc may be due to missing val nodes.")
                print(f"   Try: 1) --subgraph_hops 3 or 4")
                print(f"         2) --subgraph_include_val_test")
        else:
            print(f"OK: all {len(val_idx_subgraph):,} val nodes in subgraph")
        
        if len(test_idx_subgraph) < len(test_idx_raw):
            missing_count = len(test_idx_raw) - len(test_idx_subgraph)
            missing_pct = (missing_count / len(test_idx_raw)) * 100
            print(f"Warning: {missing_count} test nodes ({missing_pct:.1f}%) not in subgraph")
            if missing_pct > 10:
                print(f"   Low test acc may be due to missing test nodes.")
                print(f"   Try: 1) --subgraph_hops 3 or 4")
                print(f"         2) --subgraph_include_val_test")
        else:
            print(f"OK: all {len(test_idx_subgraph):,} test nodes in subgraph")
        
        subgraph_nodes_orig = subgraph_info['subgraph_nodes']
        h_0_subgraph = raw_data['v_feat'][subgraph_nodes_orig.cpu()].clone()
        t_feat_subgraph = raw_data['t_feat'][subgraph_nodes_orig.cpu()].clone()
        labels_subgraph = raw_data['labels'][subgraph_nodes_orig.cpu()].clone()
        
        raw_data['num_nodes'] = num_nodes_subgraph
        raw_data['v_feat'] = h_0_subgraph
        raw_data['t_feat'] = t_feat_subgraph
        raw_data['labels'] = labels_subgraph
        raw_data['splits'] = {
            'train_idx': train_idx_subgraph,
            'val_idx': val_idx_subgraph,
            'test_idx': test_idx_subgraph
        }
        
        print(f"\nSubgraph sampling done; using induced subgraph.")
        print(f"{'='*60}\n")
    else:
        edge_index_for_simplicial = edge_index_full
        num_nodes_subgraph = raw_data['num_nodes']

    if args.normalize_features:
        print("\nStandardizing AGSP modal features with train split statistics...")
        raw_data['v_feat'], raw_data['t_feat'] = standardize_modal_features(
            raw_data['v_feat'],
            raw_data['t_feat'],
            raw_data['splits']['train_idx'],
        )

    simplicial_data = build_triangles_and_B2(edge_index_for_simplicial, num_nodes_subgraph)

    print(f"Edges: {simplicial_data['num_edges']}")
    print(f"Triangles: {simplicial_data['num_triangles']}")

    v_feat_for_compute = raw_data['v_feat']
    t_feat_for_compute = raw_data['t_feat']
    edge_index_for_compute = simplicial_data['edge_index_tensor']
    B2_for_compute = simplicial_data['B2']
    
    if args.keep_data_cpu:
        if v_feat_for_compute.device.type != 'cpu':
            v_feat_for_compute = v_feat_for_compute.cpu()
        if t_feat_for_compute.device.type != 'cpu':
            t_feat_for_compute = t_feat_for_compute.cpu()
        if edge_index_for_compute.device.type != 'cpu':
            edge_index_for_compute = edge_index_for_compute.cpu()
        if B2_for_compute.device.type != 'cpu':
            B2_for_compute = B2_for_compute.cpu()
        print("Computing features on CPU (--keep_data_cpu)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if args.simplex_cache_path and os.path.exists(args.simplex_cache_path):
        print(f"Loading simplex feature cache: {args.simplex_cache_path}")
        simplex_cache = torch.load(args.simplex_cache_path, map_location='cpu')
        h_0 = simplex_cache['h_0']
        h_2_continuous = simplex_cache['h_2']
        if device.type == 'cpu' and h_2_continuous.dtype == torch.float16:
            print("Converting cached h_2 from fp16 to fp32 for CPU training")
            h_2_continuous = h_2_continuous.float()
        cache_meta = simplex_cache.get('meta', {})
        if cache_meta:
            print(f"Cache meta: {cache_meta}")
        del simplex_cache
    else:
        simplex_features = compute_all_simplex_features(
            v_feat=v_feat_for_compute,
            t_feat=t_feat_for_compute,
            edge_index_tensor=edge_index_for_compute,
            triangles=simplicial_data['triangles'],
            B2=B2_for_compute,
            is_binary=False,
            verbose=True,
            chunk_size=args.simplex_chunk_size
        )
        
        h_0 = simplex_features['h_0']
        h_2_continuous = simplex_features['h_2']
        if args.simplex_cache_fp16:
            print("Converting h_2 simplex features to fp16 for cache/runtime")
            h_2_continuous = h_2_continuous.half()
        if simplex_features.get('h_1') is not None:
            del simplex_features['h_1']
        if args.simplex_cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(args.simplex_cache_path)), exist_ok=True)
            print(f"Saving simplex feature cache: {args.simplex_cache_path}")
            torch.save({
                'h_0': h_0.cpu(),
                'h_2': h_2_continuous.cpu(),
                'meta': {
                    'dataset': args.dataset,
                    'num_nodes': int(num_nodes_subgraph),
                    'num_edges': int(simplicial_data['num_edges']),
                    'num_triangles': int(simplicial_data['num_triangles']),
                    'normalize_features': bool(args.normalize_features),
                    'simplex_chunk_size': args.simplex_chunk_size,
                }
            }, args.simplex_cache_path)
    del v_feat_for_compute, t_feat_for_compute, edge_index_for_compute, B2_for_compute
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Feature computation done; GPU cache cleared")
    
    print("\nBuilding node-to-triangle map...")
    if args.max_triangles_per_node is not None:
        print(f"Capping high-order triangles per node at {args.max_triangles_per_node}")
    triangles_dict = build_node_to_triangles_dict(
        simplicial_data['triangles'], 
        num_nodes_subgraph,
        max_triangles_per_node=args.max_triangles_per_node,
        seed=args.seed,
    )
    print(f"Node-to-triangle map built")
    
    print("\nPreparing training data...")
    
    train_idx = to_tensor(raw_data['splits']['train_idx'])
    val_idx = to_tensor(raw_data['splits']['val_idx'])
    test_idx = to_tensor(raw_data['splits']['test_idx'])

    labels_for_log = raw_data['labels']
    if hasattr(labels_for_log, 'cpu'):
        labels_for_log = labels_for_log.cpu()
    log_label_distribution(
        labels_for_log,
        raw_data['splits']['train_idx'],
        raw_data['splits']['val_idx'],
        raw_data['splits']['test_idx'],
        raw_data['num_classes'],
        name="label distribution (current split)"
    )
    
    if args.keep_data_cpu:
        print("\nKeeping features on CPU (staged GPU load)...")
        h_0 = h_0.cpu()
        h_2_continuous = h_2_continuous.cpu()
        edge_index = simplicial_data['edge_index_tensor'].to(device)
        labels = raw_data['labels'].to(device)
        if args.preload_triangles_gpu:
            tri_dtype = torch.float16 if args.mixed_precision and device.type == 'cuda' else h_2_continuous.dtype
            print(f"Preloading triangle features to {device} ({tri_dtype})...")
            h_2_continuous = h_2_continuous.to(device, dtype=tri_dtype)
        print("Features stay on CPU; moved to GPU per train/eval step as needed")
        if args.use_subgraph:
            print("With subgraph: only subgraph features touch GPU")
            print(f"   Subgraph nodes: {h_0.shape[0]:,} ({(1 - h_0.shape[0] / original_num_nodes) * 100:.1f}% fewer than full graph)")
        else:
            print("Note: full-graph GNN still needs full h_0 on GPU during forward.")
            print("   Consider --use_subgraph to shrink working set.")
    else:
        print(f"\nLoading tensors to {device}...")
        if args.use_subgraph:
            print(f"Subgraph: loading {h_0.shape[0]:,} node feats (full graph had {original_num_nodes:,})")
        else:
            print("If OOM, try --keep_data_cpu or --use_subgraph")
        h_0 = h_0.to(device)
        h_2_continuous = h_2_continuous.to(device)
        edge_index = simplicial_data['edge_index_tensor'].to(device)
        labels = raw_data['labels'].to(device)

    if args.sort_edges_by_target:
        print("\nSorting edge_index by target node for pairwise batching...")
        edge_order = torch.argsort(edge_index[:, 0])
        edge_index = edge_index[edge_order].contiguous()
        del edge_order
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.batch_size_edges is not None and not args.sort_edges_by_target:
        print("Warning: --batch_size_edges requires --sort_edges_by_target; disabling pairwise batching.")
        args.batch_size_edges = None

    if args.max_edges_per_node is not None and args.max_edges_per_node > 0:
        if not args.sort_edges_by_target:
            print("\nSorting edge_index by target node for edge cap...")
            edge_order = torch.argsort(edge_index[:, 0])
            edge_index = edge_index[edge_order].contiguous()
            del edge_order
        print(f"Capping pairwise incoming edges per target node at {args.max_edges_per_node}...")
        capped_parts = []
        row_cpu = edge_index[:, 0].cpu()
        unique_rows, counts = torch.unique_consecutive(row_cpu, return_counts=True)
        offset = 0
        for count in counts.tolist():
            end = offset + count
            if count > args.max_edges_per_node:
                keep_rel = torch.linspace(
                    0,
                    count - 1,
                    steps=args.max_edges_per_node,
                    dtype=torch.long,
                    device=edge_index.device,
                )
                capped_parts.append(edge_index[offset:end][keep_rel])
            else:
                capped_parts.append(edge_index[offset:end])
            offset = end
        edge_index = torch.cat(capped_parts, dim=0).contiguous()
        print(f"Pairwise edges after cap: {edge_index.shape[0]:,}")
        del capped_parts, row_cpu, unique_rows, counts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\nInitializing model...")
    
    batch_size_triangles = args.batch_size_triangles
    num_triangles = h_2_continuous.shape[0]
    
    if args.keep_data_cpu and batch_size_triangles is None:
        if num_triangles > 50000:
            num_nodes_current = h_0.shape[0]
            if num_nodes_current > 90000:
                if num_triangles > 5000000:
                    batch_size_triangles = 500
                elif num_triangles > 2000000:
                    batch_size_triangles = 800
                elif num_triangles > 500000:
                    batch_size_triangles = 1000
                elif num_triangles > 100000:
                    batch_size_triangles = 1500
                else:
                    batch_size_triangles = 2000
            else:
                if num_triangles > 5000000:
                    batch_size_triangles = 1000
                elif num_triangles > 2000000:
                    batch_size_triangles = 1500
                elif num_triangles > 500000:
                    batch_size_triangles = 2000
                elif num_triangles > 100000:
                    batch_size_triangles = 2500
                else:
                    batch_size_triangles = 4000
            print(f"\nWarning: --keep_data_cpu without --batch_size_triangles")
            print(f"   Auto triangle batch size: {batch_size_triangles} (num_nodes={num_nodes_current:,})")
            print(f"   Avoids loading all triangle feats to GPU at once")
    
    if args.auto_batch_size and batch_size_triangles is None:
        if num_triangles > 500000:
            if num_triangles > 5000000:
                auto_batch_size = 1000
            elif num_triangles > 2000000:
                auto_batch_size = 1500
            elif num_triangles > 500000:
                auto_batch_size = 2000
            else:
                auto_batch_size = 4000
            batch_size_triangles = auto_batch_size
            print(f"Many triangles ({num_triangles:,}); auto batch_size_triangles={batch_size_triangles}")
    elif batch_size_triangles is None:
        if num_triangles > 50000:
            if num_triangles > 500000:
                suggested_batch_size = 4000
            elif num_triangles > 100000:
                suggested_batch_size = 5000
            else:
                suggested_batch_size = 8000
            print(f"\nTip: {num_triangles:,} triangles; consider --batch_size_triangles {suggested_batch_size} or --auto_batch_size")
    
    if batch_size_triangles is not None:
        print(f"Batched triangles, node batch size: {batch_size_triangles}")
    
    num_classes = raw_data['num_classes']
    all_test_metrics = []

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoint dir: {save_dir}\n")

    for run in range(args.repeat_times):
        if args.repeat_times > 1:
            print(f"\n{'='*60}")
            print(f"Run {run + 1}/{args.repeat_times}")
            print(f"{'='*60}")
        set_seed(args.seed + run)
        
        model = SimplicialGNN(
            input_dim=h_0.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers=args.num_layers,
            lambda_high=args.lambda_high,
            dropout=args.dropout,
            batch_size_triangles=batch_size_triangles,
            batch_size_edges=args.batch_size_edges,
            gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)
        
        if run == 0:
            print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")
            print(f"Model:")
            print(model)

        if args.init_checkpoint:
            checkpoint = torch.load(args.init_checkpoint, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            print(f"Loaded initial checkpoint: {args.init_checkpoint}", flush=True)

        if args.freeze_backbone:
            for module in [model.input_proj, model.message_passing_layers, model.norm_layers, model.norm_before_cls]:
                for param in module.parameters():
                    param.requires_grad = False
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Frozen AGSP backbone; trainable classifier params: {trainable_params}", flush=True)

        if args.eval_only:
            eval_results = evaluate_model(
                model=model,
                h_0=h_0,
                edge_index=edge_index,
                triangles_dict=triangles_dict,
                h_triangles=h_2_continuous,
                labels=labels,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                num_classes=num_classes,
                best_model_state=None,
                device=device,
                args=args
            )
            all_test_metrics.append(eval_results['test_metrics'])
            continue

        if args.two_stage:
            print("\n" + "=" * 60)
            print("Stage 1/2 training")
            print("=" * 60)
        train_results = train_model(
            model=model,
            h_0=h_0,
            edge_index=edge_index,
            triangles_dict=triangles_dict,
            h_triangles=h_2_continuous,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            device=device,
            args=args
        )

        stage_results = {'stage1': train_results}
        if args.two_stage:
            if train_results['best_model_state'] is not None:
                model.load_state_dict(train_results['best_model_state'])
                print("\nLoaded stage-1 best checkpoint for stage 2", flush=True)
            else:
                print("\nStage 1 did not produce a best checkpoint; stage 2 continues current weights", flush=True)

            stage2_args = copy.copy(args)
            if args.stage2_epochs is not None:
                stage2_args.epochs = args.stage2_epochs
            if args.stage2_patience is not None:
                stage2_args.patience = args.stage2_patience
            if args.stage2_lr is not None:
                stage2_args.lr = args.stage2_lr
            if args.stage2_selection_metric is not None:
                stage2_args.selection_metric = args.stage2_selection_metric
            if args.stage2_class_weight_power is not None:
                stage2_args.class_weight_power = args.stage2_class_weight_power
            if args.stage2_label_smoothing is not None:
                stage2_args.label_smoothing = args.stage2_label_smoothing

            print("\n" + "=" * 60)
            print("Stage 2/2 fine-tuning")
            print("=" * 60)
            print(f"Stage 2 hyperparams: epochs={stage2_args.epochs}, patience={stage2_args.patience}, "
                  f"lr={stage2_args.lr}, selection_metric={stage2_args.selection_metric}, "
                  f"class_weight_power={stage2_args.class_weight_power}, "
                  f"label_smoothing={stage2_args.label_smoothing}", flush=True)
            train_results = train_model(
                model=model,
                h_0=h_0,
                edge_index=edge_index,
                triangles_dict=triangles_dict,
                h_triangles=h_2_continuous,
                labels=labels,
                train_idx=train_idx,
                val_idx=val_idx,
                device=device,
                args=stage2_args
            )
            stage_results['stage2'] = train_results
        
        eval_results = evaluate_model(
            model=model,
            h_0=h_0,
            edge_index=edge_index,
            triangles_dict=triangles_dict,
            h_triangles=h_2_continuous,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            num_classes=num_classes,
            best_model_state=train_results['best_model_state'],
            device=device,
            args=args
        )

        best_model_state = train_results['best_model_state']
        if best_model_state is not None:
            tag = f"_{args.checkpoint_tag}" if args.checkpoint_tag else ""
            save_path = os.path.join(save_dir, f"best_model_{args.dataset}{tag}_run{run+1}.pt")
            torch.save({
                'model_state_dict': best_model_state,
                'run': run + 1,
                'init_checkpoint': args.init_checkpoint,
                'best_val_acc': train_results['best_val_acc'],
                'test_metrics': eval_results['test_metrics'],
                'stage_results': {
                    name: {'best_val_acc': result['best_val_acc']}
                    for name, result in stage_results.items()
                },
                'logit_bias_calibration': eval_results.get('logit_bias_calibration'),
            }, save_path)
            print(f"Saved best model run {run+1} to: {save_path}")

        all_test_metrics.append(eval_results['test_metrics'])
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    print(f"\nFinal test metrics (mean ± std over runs):")
    print("="*60)
    for name in metrics_names:
        values = [m[name] for m in all_test_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{name.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")
    print("===================================")
    
    
