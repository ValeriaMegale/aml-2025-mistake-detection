"""
Prepare Graph Data for Substep 4: GNN Classification.

Questo script converte task graphs matched in formato PyTorch Geometric Data.

Processo:
1. Per ogni video, caricare visual step embeddings (da substep 1)
2. Caricare task graph corrispondente
3. Calcolare matching visual steps → graph nodes (usando Hungarian algorithm)
4. Per ogni nodo del grafo:
   - Text feature: CLIP encoding della descrizione
   - Matched visual feature: embedding dello step visivo matched (se presente)
   - Node feature: proiezione learnable di [text_feat, visual_feat]
5. Creare PyTorch Geometric Data object con:
   - x: node features [N_nodes, feature_dim]
   - edge_index: edge connectivity [2, N_edges]
   - y: graph-level label [1] (0=correct, 1=error)
"""

import argparse
import json
import os
import csv
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

# Import utilities from substep 3
try:
    from extension.substep3_task_graph_matching.model.task_graph_matcher import TextEncoder
    from extension.substep3_task_graph_matching.train_task_graph_matching_gpu import (
        load_recording_to_activity_mapping,
        load_activity_to_taskgraph,
        load_task_graph,
        load_step_annotations,
        get_binary_label
    )
except ImportError:
    # Fallback per import locali
    import sys
    sys.path.append('../substep3_task_graph_matching')
    from model.task_graph_matcher import TextEncoder
    from train_task_graph_matching_gpu import (
        load_recording_to_activity_mapping,
        load_activity_to_taskgraph,
        load_task_graph,
        load_step_annotations,
        get_binary_label
    )


def compute_matching(visual_emb, text_emb):
    """
    Calcola matching Hungarian tra visual steps e graph nodes.
    
    Args:
        visual_emb: [N_steps, visual_dim] - visual step embeddings
        text_emb: [N_nodes, text_dim] - task graph node embeddings
    
    Returns:
        matching: dict con:
            - visual_to_node: {visual_idx: node_idx} mapping
            - node_to_visual: {node_idx: visual_idx} mapping
            - match_scores: {node_idx: similarity_score} per nodi matched
    """
    if isinstance(visual_emb, torch.Tensor):
        visual_emb = visual_emb.numpy()
    if isinstance(text_emb, torch.Tensor):
        text_emb = text_emb.numpy()
    
    visual_emb = np.array(visual_emb, dtype=np.float32)
    text_emb = np.array(text_emb, dtype=np.float32)
    
    N_steps, visual_dim = visual_emb.shape
    N_nodes, text_dim = text_emb.shape
    
    # Align dimensions for matching (simple approach: pad or truncate text to match visual)
    # For cosine similarity, we can project both to a common space or pad
    if text_dim < visual_dim:
        # Pad text embeddings with zeros
        padding = np.zeros((N_nodes, visual_dim - text_dim), dtype=np.float32)
        text_emb = np.concatenate([text_emb, padding], axis=1)
    elif text_dim > visual_dim:
        # Truncate text embeddings
        text_emb = text_emb[:, :visual_dim]
    
    # Normalizza per cosine similarity
    vis_norm = F.normalize(torch.tensor(visual_emb, dtype=torch.float32), p=2, dim=-1)
    txt_norm = F.normalize(torch.tensor(text_emb, dtype=torch.float32), p=2, dim=-1)
    
    similarity = torch.matmul(vis_norm, txt_norm.t()).numpy()
    
    cost_matrix = -similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    visual_to_node = {}
    node_to_visual = {}
    match_scores = {}
    
    for r, c in zip(row_ind, col_ind):
        visual_to_node[int(r)] = int(c)
        node_to_visual[int(c)] = int(r)
        match_scores[int(c)] = float(similarity[r, c])
    
    return {
        'visual_to_node': visual_to_node,
        'node_to_visual': node_to_visual,
        'match_scores': match_scores
    }


def create_node_features(task_graph, text_emb, visual_emb, matching, feature_dim=256):
    """
    Crea node features combinando text e visual features matched.
    
    Args:
        task_graph: dict con 'steps' (node descriptions)
        text_emb: [N_nodes, text_dim] - text embeddings dei nodi
        visual_emb: [N_steps, visual_dim] - visual step embeddings
        matching: dict da compute_matching()
        feature_dim: dimensione output node features
    
    Returns:
        node_features: [N_nodes, feature_dim] - features finali dei nodi
    """
    N_nodes = text_emb.shape[0]
    text_dim = text_emb.shape[1]
    visual_dim = visual_emb.shape[1]
    
    # Simple projection layer (può essere sostituita con una learnable nel modello)
    # Per ora usiamo una proiezione lineare semplice
    node_to_visual = matching['node_to_visual']
    
    # Per ogni nodo, combina text + visual features
    combined_features = []
    for node_idx in range(N_nodes):
        # Text feature
        text_feat = text_emb[node_idx]  # [text_dim]
        
        # Visual feature (se matched)
        if node_idx in node_to_visual:
            visual_idx = node_to_visual[node_idx]
            visual_feat = visual_emb[visual_idx]  # [visual_dim]
        else:
            visual_feat = np.zeros(visual_dim, dtype=np.float32)
        
        # Concatenate
        combined = np.concatenate([text_feat, visual_feat])  # [text_dim + visual_dim]
        combined_features.append(combined)
    
    combined_features = np.array(combined_features)  # [N_nodes, text_dim + visual_dim]
    
    # Proiezione a feature_dim (simple linear projection, normalizzata)
    # In pratica, questo sarà fatto dal modello DAGNN, ma qui prepariamo l'input
    # Per ora manteniamo la concatenazione, il modello farà la proiezione
    return combined_features


def task_graph_to_edge_index(task_graph, num_nodes=None):
    """
    Converte task graph edges in formato PyTorch Geometric edge_index.
    
    Args:
        task_graph: dict con 'edges': [[src, dst], ...]
        num_nodes: numero nodi (per validazione)
    
    Returns:
        edge_index: [2, N_edges] tensor
    """
    edges = task_graph['edges']
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    
    edge_list = []
    for edge in edges:
        if len(edge) >= 2:
            src, dst = int(edge[0]), int(edge[1])
            # Validazione: gli indici devono essere 0-based e < num_nodes
            if num_nodes is not None:
                if src < 0 or src >= num_nodes or dst < 0 or dst >= num_nodes:
                    continue  # Skip edge invalido
            edge_list.append([src, dst])
    
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


def create_graph_data(video_id, step_embeddings, task_graph, text_emb, text_encoder, 
                      annotation_map, feature_dim=256):
    """
    Crea un PyTorch Geometric Data object per un video.
    
    Args:
        video_id: str - ID del video
        step_embeddings: list o array - visual step embeddings del video
        task_graph: dict - task graph JSON
        text_emb: [N_nodes, text_dim] - precomputed text embeddings
        text_encoder: TextEncoder instance
        annotation_map: dict - annotations per labels
        feature_dim: int - dimensione node features (output)
    
    Returns:
        data: torch_geometric.data.Data object
    """
    if isinstance(step_embeddings, list):
        visual_emb = np.array([s['embedding'] if isinstance(s, dict) else s 
                              for s in step_embeddings], dtype=np.float32)
    else:
        visual_emb = np.array(step_embeddings, dtype=np.float32)
    
    if len(visual_emb) == 0:
        return None
    
    matching = compute_matching(visual_emb, text_emb)
    
    node_features = create_node_features(
        task_graph, text_emb, visual_emb, matching, feature_dim=feature_dim
    )
    
    x = torch.tensor(node_features, dtype=torch.float32)
    
    edge_index = task_graph_to_edge_index(task_graph, num_nodes=x.size(0))
    
    label = int(get_binary_label(video_id, annotation_map))
    y = torch.tensor([label], dtype=torch.long)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        video_id=video_id,
        num_nodes=x.size(0)
    )
    
    return data


def prepare_all_graphs(args):
    """Prepara tutti i grafi per training e test."""
    
    print("Loading data...")
    
    step_embeddings_dict = np.load(args.step_embeddings_npy, allow_pickle=True).item()
    print(f"Loaded step embeddings for {len(step_embeddings_dict)} videos")
    
    annotation_map = load_step_annotations(args.annotations)
    
    recording_to_activity = load_recording_to_activity_mapping(args.recording_csv)
    activity_to_taskgraph = load_activity_to_taskgraph(args.activity_mapping)
    
    print("Initializing text encoder...")
    text_encoder = TextEncoder(args.text_model)
    
    print("Precomputing task graph text embeddings...")
    precomputed_text_emb = {}
    for act_id, tg_info in activity_to_taskgraph.items():
        task_graph = load_task_graph(args.task_graph_dir, tg_info['task_graph_file'])
        _, text_emb = text_encoder.encode_task_graph(task_graph)
        precomputed_text_emb[act_id] = text_emb
        print(f"  Activity {act_id}: {len(text_emb)} nodes")
    
    all_video_ids = list(step_embeddings_dict.keys())
    
    recipes = sorted(list(set([
        v.split('_')[0] for v in all_video_ids if '_' in v
    ])), key=lambda x: int(x) if x.isdigit() else 0)
    
    print(f"\nFound {len(recipes)} recipes")
    print(f"Total videos: {len(all_video_ids)}")
    
    output_dir = Path(args.output_dir)
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'recipes': recipes,
        'total_videos': len(all_video_ids),
        'graphs': {}
    }
    
    # Process each video
    print("\nProcessing videos...")
    for video_id in tqdm(all_video_ids, desc="Creating graphs"):
        if video_id not in step_embeddings_dict:
            continue
        
        step_embeddings = step_embeddings_dict[video_id]
        
        activity_id = recording_to_activity.get(video_id)
        if activity_id is None:
            continue
        
        tg_info = activity_to_taskgraph.get(activity_id)
        if tg_info is None:
            continue
        
        task_graph = load_task_graph(args.task_graph_dir, tg_info['task_graph_file'])
        text_emb = precomputed_text_emb[activity_id]
        
        data = create_graph_data(
            video_id, step_embeddings, task_graph, text_emb, text_encoder,
            annotation_map, feature_dim=args.feature_dim
        )
        
        if data is None:
            continue
        
        recipe_id = video_id.split('_')[0]
        
        graph_filename = f"{video_id}.pt"
        if recipe_id in args.test_recipes.split(','):
            save_path = test_dir / graph_filename
            split = 'test'
        else:
            save_path = train_dir / graph_filename
            split = 'train'
        
        torch.save(data, save_path)
        
        metadata['graphs'][video_id] = {
            'file': graph_filename,
            'split': split,
            'recipe_id': recipe_id,
            'activity_id': activity_id,
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.size(1) if data.edge_index.size(1) > 0 else 0,
            'label': int(data.y.item())
        }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGraphs saved to {output_dir}")
    print(f"  Train graphs: {len(list(train_dir.glob('*.pt')))}")
    print(f"  Test graphs: {len(list(test_dir.glob('*.pt')))}")
    print(f"  Metadata saved to {metadata_file}")
    
    train_labels = [g['label'] for g in metadata['graphs'].values() if g['split'] == 'train']
    test_labels = [g['label'] for g in metadata['graphs'].values() if g['split'] == 'test']
    
    print(f"\nStatistics:")
    if len(train_labels) > 0:
        print(f"  Train: {len(train_labels)} graphs, {sum(train_labels)} with errors ({sum(train_labels)/len(train_labels)*100:.1f}%)")
    if len(test_labels) > 0:
        print(f"  Test: {len(test_labels)} graphs, {sum(test_labels)} with errors ({sum(test_labels)/len(test_labels)*100:.1f}%)")
    else:
        print(f"  Test: 0 graphs (all in train - leave-one-out split will be done during training)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare graph data for Substep 4 GNN classification'
    )
    
    parser.add_argument('--step_embeddings_npy', type=str,
                        default='../substep1_step_localization/data/step_embeddings_perception.npy',
                        help='Path to step embeddings .npy file (from substep 1)')
    parser.add_argument('--annotations', type=str,
                        default='annotations/annotation_json/step_annotations.json',
                        help='Path to step annotations JSON')
    parser.add_argument('--recording_csv', type=str,
                        default='annotations/annotation_csv/recording_id_step_idx.csv',
                        help='Path to recording_id_step_idx.csv')
    parser.add_argument('--activity_mapping', type=str,
                        default='../substep3_task_graph_matching/activity_to_taskgraph.json',
                        help='Path to activity_to_taskgraph.json')
    parser.add_argument('--task_graph_dir', type=str,
                        default='annotations/task_graphs',
                        help='Directory containing task graph JSON files')
    parser.add_argument('--text_model', type=str,
                        default='clip',
                        help='Text encoder: "clip" or "all-MiniLM-L6-v2"')
    
    parser.add_argument('--output_dir', type=str,
                        default='extension/substep4_gnn_classification/graph_classification_extension/substep4_gnn_classification/data',
                        help='Output directory for graph data')
    parser.add_argument('--feature_dim', type=int, default=768,
                        help='Node feature dimension (text_dim + visual_dim, before projection)')
    parser.add_argument('--test_recipes', type=str, default='',
                        help='Comma-separated recipe IDs for test split (leave-one-out style, empty = all in train for initial prep)')
    
    args = parser.parse_args()
    
    prepare_all_graphs(args)
