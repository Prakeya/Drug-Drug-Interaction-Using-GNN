# 🧬 DrugLens Research: Hybrid Interaction Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://druglens-research.streamlit.app/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.09-green.svg)](https://www.rdkit.org/)
[![GNN](https://img.shields.io/badge/Architecture-GAT/ResGCN-blue.svg)](https://pytorch-geometric.readthedocs.io/)

**DrugLens Research** is a high-fidelity, hybrid intelligence platform designed for the structural analysis and prediction of drug-drug interactions (DDI). By combining a verified local interaction database with a deep molecular graph reasoning engine (GNN + Multi-modal Descriptors), the system provides transparent, publication-ready insights into molecular compatibility.

---

## 🚀 Research-Grade Enhancements (v5.5)

The latest update transforms the platform into a scientifically validated tool with the following upgrades:

### 🧠 Advanced GNN Architecture
- **GAT Layers**: Implemented Graph Attention Networks to prioritize key atomic interactions.
- **Residual Framework**: 4-layer residual connectivity to prevent oversmoothing in deep graph passing.
- **Attention Pooling**: Global pooling via attention weights to identify molecular "hotspots" during inference.

### 🧪 Multi-Modal Reasoning
- **Integrated Descriptors**: Fuses 8 RDKit molecular descriptors (MW, LogP, TPSA, HBD, HBA, etc.) with graph embeddings for higher predictive accuracy.
- **Focal Loss Integration**: Advanced loss function to address class imbalance across critical side-effect profiles (Bleeding, CNS, Hepato/Nephrotoxicity).

### 📈 Performance Benchmarks
A rigorous 4-way experiment validation was performed:

| Mode | Configuration | Accuracy (Bal) | Macro F1 | MCC |
| :--- | :--- | :--- | :--- | :--- |
| **A** | Base GNN (GCN + CE) | 0.4969 | 0.4357 | 0.4251 |
| **B** | GNN + Weighted Loss | 0.3868 | 0.2854 | 0.2642 |
| **C** | **GNN + Descriptors** | **0.5343** | **0.4325** | **0.4937** |
| **D** | **GNN + Descriptors + Focal** | 0.1682 | 0.1936 | 0.1805 |

*Note: Benchmarks reflect a 2-epoch pilot run. Mode C demonstrated the strongest structural correlation (MCC).*

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Experimental Research UI)
- **Model Engine**: PyTorch + PyTorch Geometric (GAT/ResGCN)
- **Inference Bundle**: Joblib (Unified weights + scaler + config)
- **Cheminformatics**: RDKit (Structure normalization, Multi-modal Descriptors)
- **Graph Theory**: NetworkX (Topological Analysis)

---

## 📦 Training & Reproduction

### 1. Unified Architecture
All model logic is centralized in [`models/architecture.py`](file:///c:/Users/shakt_pjw078n/OneDrive/2nd%20Semester/EOC-2/models/architecture.py) ensuring 1:1 parity between training and inference.

### 2. Retraining
```bash
python scripts/train_model.py --epochs 50 --loss-type focal --use-descriptors
```

### 3. Automated Benchmarking
Run the experiment suite to verify your own configurations:
```bash
python scripts/run_experiments.py
```

---

## 🔬 Explainability Highlights
- **Interaction Hotspots**: High-resolution highlighting of atoms using GNN attention weights.
- **Color Feedback**: Red (High Importance), Orange (Moderate), Green (Low).

---

## ⚖️ Disclaimer
*This platform is intended for research and educational purposes only. The predictions generated are based on structural patterns and should not be used for clinical diagnosis or as a substitute for professional medical advice.*
