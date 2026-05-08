import joblib
import torch
import numpy as np
from rdkit import Chem
import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.architecture import DDIGNNModel, smiles_to_graph, NODE_FEAT_DIM

def smoke_test_inference():
    path = "models/ddi_advanced_gnn_integrated.joblib"
    if not os.path.exists(path):
        print(f"Skipping inference test: {path} not found.")
        return
    
    print(f"Loading model from {path}...")
    data = joblib.load(path)
    config = data['config']
    model_state = data['model_state']
    standardizer = data['descriptor_standardizer']
    
    model = DDIGNNModel(
        node_dim=config.get('node_dim', NODE_FEAT_DIM),
        hidden_dim=config.get('hidden_dim', 128),
        num_labels=config.get('num_outputs', 6),
        descriptor_dim=8 if config.get('use_descriptors') else 0,
        gnn_type=config.get('gnn_type', 'gcn')
    )
    model.load_state_dict(model_state)
    model.eval()
    
    s1 = "CCO" # Ethanol
    s2 = "CC(=O)O" # Acetic Acid
    
    A1, X1, m1 = smiles_to_graph(s1, config.get('max_nodes', 70))
    A2, X2, m2 = smiles_to_graph(s2, config.get('max_nodes', 70))
    
    d1, d2 = None, None
    if standardizer:
        d1 = torch.tensor(standardizer.transform(Chem.MolFromSmiles(s1)), dtype=torch.float32).unsqueeze(0)
        d2 = torch.tensor(standardizer.transform(Chem.MolFromSmiles(s2)), dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(
            torch.tensor(A1).unsqueeze(0),
            torch.tensor(X1).unsqueeze(0),
            torch.tensor(m1).unsqueeze(0),
            torch.tensor(A2).unsqueeze(0),
            torch.tensor(X2).unsqueeze(0),
            torch.tensor(m2).unsqueeze(0),
            d1, d2
        )
        probs = torch.softmax(logits, dim=1).numpy()
        print(f"Predictions: {probs}")
        print("Inference smoke test passed!")

if __name__ == "__main__":
    smoke_test_inference()
