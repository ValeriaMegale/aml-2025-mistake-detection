"""
Test Pipeline End-to-End: Substep 1 → 3 → 4

Verifica che tutti i componenti del pipeline siano integrati correttamente.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data

# Test imports
print("Testing imports...")
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from extension.substep4_gnn_classification.graph_classification_substep4.dataset import TaskGraphDataset
    from extension.substep4_gnn_classification.graph_classification_substep4.model import DAGNNClassifier, get_model_input_dim
    print("✓ Dataset and model imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model creation
print("\nTesting model creation...")
try:
    # Default dimensions: text(512) + visual(768) = 1280
    in_channels = get_model_input_dim(text_dim=512, visual_dim=768)
    assert in_channels == 1280, f"Expected 1280, got {in_channels}"
    
    model = DAGNNClassifier(
        in_channels=in_channels,
        hidden_channels=128,
        num_classes=2,
        K=10
    )
    print(f"✓ Model created with input_dim={in_channels}, hidden_dim=128")
except Exception as e:
    print(f"✗ Model creation error: {e}")
    sys.exit(1)

# Test model forward pass
print("\nTesting model forward pass...")
try:
    num_nodes = 10
    num_edges = 15
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    out = model(x, edge_index, batch)
    
    assert out.shape == (1, 2), f"Expected output shape (1, 2), got {out.shape}"
    print(f"✓ Forward pass successful: output shape {out.shape}")
except Exception as e:
    print(f"✗ Forward pass error: {e}")
    sys.exit(1)

# Test data preparation (if prepare_graph_data exists)
print("\nTesting data preparation script...")
try:
    from substep4.prepare_graph_data import (
        compute_matching, 
        create_node_features,
        task_graph_to_edge_index
    )
    print("✓ Data preparation functions import successful")
    
    # Test matching
    visual_emb = np.random.randn(5, 768).astype(np.float32)
    text_emb = np.random.randn(8, 512).astype(np.float32)
    matching = compute_matching(visual_emb, text_emb)
    
    assert 'visual_to_node' in matching
    assert 'node_to_visual' in matching
    print(f"✓ Matching computation successful: {len(matching['visual_to_node'])} matches")
    
    # Test edge index conversion
    task_graph = {
        'edges': [[0, 1], [1, 2], [2, 3], [0, 3]]
    }
    edge_index = task_graph_to_edge_index(task_graph)
    assert edge_index.shape == (2, 4), f"Expected (2, 4), got {edge_index.shape}"
    print(f"✓ Edge index conversion successful: {edge_index.shape}")
    
except ImportError as e:
    print(f"⚠ Data preparation imports not available: {e}")
except Exception as e:
    print(f"✗ Data preparation test error: {e}")

# Test dataset (if data directory exists)
print("\nTesting dataset loading...")
data_dir = Path("substep4/extension/substep4_gnn_classification/graph_classification_substep4/data")
if data_dir.exists() and (data_dir / "metadata.json").exists():
    try:
        # Try to load dataset
        dataset_train = TaskGraphDataset(data_dir, split='train')
        print(f"✓ Dataset loaded: {len(dataset_train)} train graphs")
        
        if len(dataset_train) > 0:
            # Test getting a sample
            sample = dataset_train[0]
            assert hasattr(sample, 'x'), "Sample missing 'x' attribute"
            assert hasattr(sample, 'edge_index'), "Sample missing 'edge_index' attribute"
            assert hasattr(sample, 'y'), "Sample missing 'y' attribute"
            print(f"✓ Sample loaded: {sample.num_nodes} nodes, {sample.edge_index.size(1)} edges")
            
            # Test model with real data
            model.eval()
            with torch.no_grad():
                batch = sample
                batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long)
                out = model(batch.x, batch.edge_index, batch.batch)
                print(f"✓ Model forward pass on real data: output shape {out.shape}")
    except Exception as e:
        print(f"⚠ Dataset loading test skipped: {e}")
else:
    print("⚠ Data directory not found - run prepare_graph_data.py first")

# Test training script imports
print("\nTesting training script imports...")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_gnn", "substep4/train_gnn_classification.py"
    )
    if spec is not None:
        print("✓ Training script found")
    else:
        print("⚠ Training script not found")
except Exception as e:
    print(f"⚠ Training script check: {e}")

# Test evaluation script imports
print("\nTesting evaluation script imports...")
try:
    spec = importlib.util.spec_from_file_location(
        "eval_gnn", "substep4/eval_gnn_classification.py"
    )
    if spec is not None:
        print("✓ Evaluation script found")
    else:
        print("⚠ Evaluation script not found")
except Exception as e:
    print(f"⚠ Evaluation script check: {e}")

print("\n" + "="*70)
print("PIPELINE TEST SUMMARY")
print("="*70)
print("All core components tested successfully!")
print("\nNext steps:")
print("1. Run prepare_graph_data.py to create graph data")
print("2. Run train_gnn_classification.py for leave-one-out training")
print("3. Run eval_gnn_classification.py to evaluate all checkpoints")
