# 🧬 DrugLens Research: Hybrid Interaction Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://druglens-research.streamlit.app/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.09-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DrugLens Research** is a high-fidelity, hybrid intelligence platform designed for the structural analysis and prediction of drug-drug interactions (DDI). By combining a verified local interaction database with a deep molecular graph reasoning engine, the system provides transparent, publication-ready insights into molecular compatibility.

---

## 🚀 Key Features

### 🧠 Hybrid Retrieval-Prediction Pipeline
The core architecture prioritizes empirical evidence:
- **Database-First Lookup**: Scans the verified DrugBank repository for known interaction maps using canonical structural normalization.
- **Model Fallback**: If no match is found, the system deploys a Graph Neural Network (GNN)-backed model to predict novel interactions based on 2048-bit Morgan fingerprints and topological features.

### 🧪 Structural Reasoning Engine
Unlike traditional 'black-box' models, DrugLens explains *why* an interaction is likely:
- **Automated Inference**: Connects Tanimoto similarity, MCS (Maximum Common Substructure) overlap, and physicochemical descriptors (MW, LogP) into natural language reasoning.
- **Confidence Calibration**: Interprets predictive probability into standardized Low/Moderate/High confidence tiers.

### 📊 Advanced Cheminformatics Visualization
- **Dynamic MCS Highlighting**: Real-time visualization of shared molecular scaffolds.
- **Topological Graph Analytics**: High-resolution interactive graphs showing atomic connectivity and bond order.
- **Pharmacological Signaling**: Profiles molecules for CYP interference, QT risk, and blood-brain barrier permeability markers.

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Experimental Research UI)
- **Cheminformatics**: RDKit (Structure normalization, Fingerprinting, MCS)
- **Machine Learning**: PyTorch Geometric (Optional Graph Backend), Joblib (Main Inference Engine)
- **Graph Theory**: NetworkX (Topological Graph Analytics)
- **Data Science**: Pandas, NumPy, Scikit-Learn

---

## 📦 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/druglens-research.git
   cd druglens-research
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For Windows users, the system includes a hardened PyTorch fallback for stability.*

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

---

## 🔬 Use Cases
- **Clinical Research**: Rapidly screening known contraindications during drug discovery.
- **Academic Study**: Visualizing structural overlap (MCS) between analog compounds.
- **Pharmacology Education**: Demonstrating the impact of structural similarity on metabolism.

---

## ⚖️ Disclaimer
*This platform is intended for research and educational purposes only. The predictions generated are based on structural patterns and should not be used for clinical diagnosis or as a substitute for professional medical advice.*
