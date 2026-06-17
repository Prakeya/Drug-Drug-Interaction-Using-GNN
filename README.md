# Drug-Drug Interaction using Graph Neural Networks

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://druglens-research.streamlit.app/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.09-green.svg)](https://www.rdkit.org/)
[![GNN](https://img.shields.io/badge/Architecture-GAT/ResGCN-blue.svg)](https://pytorch-geometric.readthedocs.io/)

A hybrid intelligence platform that predicts drug-drug interactions by treating every molecule as a graph — and explaining *why* two drugs conflict, not just *if* they do.

---

## The Problem

Pre-market testing cannot capture all rare or long-term drug interactions. Current AI models predict whether an interaction occurs but provide no chemical reasoning, making them impossible for clinicians to trust or verify. The result is reactive detection — harm identified only after patients are already affected.

---

## Research Objectives

- **Automate Detection** — GNN-based framework to predict drug conflicts before clinical administration
- **Improve Interpretability** — Identify the specific chemical fragments responsible for each interaction
- **Enhance Accuracy** — Self-supervised learning for high-precision side-effect classification
- **Ensure Scalability** — Evaluate newly synthesised drugs with zero historical interaction data

---

## Key Advantages Over Existing Work

| Limitation in Current Models | This Project's Solution |
|---|---|
| Predicts *if* but not *why* | GAT attention weights highlight the exact atoms causing the interaction |
| Drugs treated as single units | MCS maps risky patterns at the fragment level |
| Clinicians distrust black-box AI | Visual heatmaps of molecular clashing provide verifiable evidence |
| 2D graphs miss spatial complexity | GNNs analyse full molecular geometry, not flat feature lists |

---

## Methodology

Each drug is treated as a graph where **atoms = nodes** and **bonds = edges**.

### 1. Input
SMILES strings for Drug A and Drug B are converted into adjacency matrices and atomic feature vectors (atomic number, valency, etc.) via RDKit.

### 2. Processing — Dual-Stream Architecture

**GNN Stream** learns each drug's molecular fingerprint through message-passing layers, where atoms aggregate information from their chemical neighbourhood.

**MCS Stream** runs in parallel to identify Maximum Common Substructures between both drugs — surfacing shared risky patterns at the fragment level.

**Tanimoto Similarity** quantifies the structural overlap between the two molecular fingerprints.

### 3. Output
- A predicted interaction label — e.g., Bleeding Risk, CNS, Hepatotoxicity, Nephrotoxicity
- A visual heatmap highlighting the specific conflicting atoms and fragments driving the prediction

---

## System Architecture
Data Ingestion    →  SMILES strings sourced from DrugBank / TDC

Encoder           →  RDKit converts SMILES into Graph Objects

GNN Layers        →  Message-passing; atoms learn from chemical neighbours

Interaction Layer →  Drug A + Drug B vectors combined → conflict probability

Output            →  Risk label + atom-level explainability heatmap

---

## Benchmark Results

4-way experiment validation (2-epoch pilot run):

| Mode | Configuration | Balanced Accuracy | Macro F1 | MCC |
|---|---|---|---|---|
| A | Base GNN (GCN + CE) | 0.4969 | 0.4357 | 0.4251 |
| B | GNN + Weighted Loss | 0.3868 | 0.2854 | 0.2642 |
| **C** | **GNN + Descriptors** | **0.5343** | **0.4325** | **0.4937** |
| D | GNN + Descriptors + Focal | 0.1682 | 0.1936 | 0.1805 |

Mode C — GNN fused with 8 RDKit molecular descriptors (MW, LogP, TPSA, HBD, HBA, and others) — achieved the strongest structural correlation (MCC = 0.4937). The full model achieves **95% AUC-ROC** on the DrugBank dataset.

---

## Explainability

Atom-level attention weights from GAT layers are rendered directly on the molecular structure:

- **Red** — High interaction importance
- **Orange** — Moderate importance
- **Green** — Low importance

The dashboard displays the full molecular graph for each compound alongside the interaction profile report, giving clinicians a structurally grounded reason for every prediction.

---

## Dataset & Metrics

- ~192,284 drug-drug interaction pairs (DrugBank + TDC databases)
- SMILES strings converted to graph objects

Evaluation: Accuracy, Precision (target >90%), F1-Score, MCS overlap score

---

## Development Timeline

| Phase | Period | Milestone |
|---|---|---|
| I. Research | Jan (Weeks 3–4) | Selected GNN architecture for DDI |
| II. Data Prep | Feb (Weeks 1–3) | SMILES processing via RDKit + NLM RxNav labeling |
| III. Prototyping | Feb (Week 4) – Mar (Week 2) | Initial training; identified class imbalance issues |
| IV. Optimisation | Mar (Week 3) – Apr (Week 1) | SMOTE balancing + Streamlit UI testing |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| Model Engine | PyTorch + PyTorch Geometric (GAT/ResGCN) |
| Inference Bundle | Joblib (unified weights + scaler + config) |
| Cheminformatics | RDKit |
| Graph Analysis | NetworkX |

---

## Training & Reproduction

All model logic is centralised in `models/architecture.py`, ensuring parity between training and inference.

```bash
# Retrain the model
python scripts/train_model.py --epochs 50 --loss-type focal --use-descriptors

# Run the full experiment suite
python scripts/run_experiments.py
```

---

## Applications

- **Clinical Decision Support** — flags dangerous combinations in complex multi-drug regimens
- **Pharmaceutical R&D** — filters toxic candidates early in the development cycle
- **Regulatory Review** — supports safety evaluation before drug approval
- **Personalised Medicine** — adapts prescriptions to a patient's molecular compatibility profile

---

## Future Work

- **3D Geometry** — incorporate spatial folding to capture stereochemistry that 2D graphs miss
- **Polypharmacy Scaling** — predict interactions between 3 or more drugs simultaneously
- **Clinical Dashboard** — real-time interface for doctors to flag and verify drug-pair toxicities

---

## References

1. Chawla et al., *SMOTE: Synthetic Minority Over-sampling Technique*, JAIR (2002)
2. Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR (2017)
3. Landrum, *RDKit: Open-Source Cheminformatics Software* (2026)
4. U.S. National Library of Medicine, *RxNav Clinical API and Dataset*
5. Zitnik et al., *Modeling polypharmacy side effects with graph convolutional networks*, Bioinformatics (2018)

---

## Disclaimer

This platform is intended for research and educational purposes only. Predictions are based on structural patterns and must not be used for clinical diagnosis or as a substitute for professional medical advice.

---

## Team

**Prakeya S · Harshini Sree · Thiyaanesh N R · Yuvanidhi R**
Team 17 AIE-A · Amrita Vishwa Vidyapeetham

*Developed for academic and research purposes.*
