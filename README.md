# 🧬 DrugLens: Hybrid GNN-Clinical Drug Interaction Diagnostic System

**A biology-aware ML system for predicting drug-drug interactions using molecular graphs, pharmacological features, and clinical knowledge.**

---

## 🎯 Project Overview

DrugLens is a hybrid prediction system that combines:
- **Graph Neural Networks (GNNs)** for molecular structure analysis
- **8 Core Pharmacological Signals** (CYP inhibitor/inducer, substrates, QT risk, etc.)
- **KNN-based similarity matching** against curated clinical database
- **Light safety filters** to prevent nonsense predictions
- **Interactive Streamlit dashboard** for drug interaction exploration

---

## 🚀 March 2026 Update: Biology-Aware Feature Engineering

### What Changed

#### **V2: Enhanced Pharmacological Signal Extraction**

Instead of hardcoded interaction rules, DrugLens now extracts **8 core pharmacological signals** that the ML model learns to combine:

| Signal | What It Detects | Example |
|--------|-----------------|---------|
| **CYP Inhibitor** | Blocks cytochrome P450 enzyme | Fluconazole, Ketoconazole |
| **CYP Inducer** | Accelerates CYP metabolism | Rifampicin, Phenobarbital |
| **CYP Substrate** | Metabolized by CYP enzymes | Warfarin, Simvastatin |
| **Prodrug** | Requires activation (ester/amide) | Codeine, Enalapril |
| **QT Prolongation Risk** | Cardiac electrolyte effects | Citalopram, Amiodarone |
| **CNS Depressant** | Crosses blood-brain barrier | Diazepam, Propofol |
| **Narrow Therapeutic Index** | Small dose = toxicity | Digoxin, Warfarin |
| **P-gp Substrate** | Transporter-limited absorption | Digoxin, Talinolol |

### Architecture

```
SMILES Input
    ↓
Molecular Descriptors + Functional Groups
    ↓
8 Pharmacological Signals (Features)
    ↓
Morgan Fingerprints + Mechanism Vectors + Class Proxies
    ↓
ML Model (learns interactions from data)
    ↓
Light Safety Filter (2-3 rules only)
    ↓
Clinical-Grade Prediction
```

### Key Improvements

1. **Biology-Informed**: Signals extract real pharmacological mechanisms
2. **Model-Driven**: ML learns interaction patterns, not hardcoded rules
3. **Explainable**: Each prediction backed by detected mechanisms
4. **Directional**: Distinguishes A→B from B→A interactions
5. **Safer**: Light safety layer prevents extreme outliers

---

## 📊 Pharmacokinetic & Pharmacodynamic Patterns

### PK Interactions Detected
- **CYP Inhibition**: `Inhibitor → Substrate` → increased exposure → toxicity
- **CYP Induction**: `Inducer → Substrate` → decreased exposure → efficacy loss
- **Transporter Modulation**: `Inhibitor → P-gp Substrate` → bioavailability changes
- **Prodrug Blocking**: `Inhibitor → Prodrug` → loss of activation

### PD Interactions Detected
- **QT Additive Risk**: Two QT-prolonging drugs → arrhythmia risk
- **CNS Synergy**: CNS depressants → enhanced sedation/respiratory risk
- **Narrow TI Caution**: Both drugs have narrow margins → dosing critical
- **Bleeding Risk**: NSAID-like + Anticoagulant-like → hemorrhage risk

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- PyTorch 2.1+ (with GPU optional)
- RDKit for molecular cheminformatics
- Streamlit for web UI

### Quick Start

```bash
# Clone / enter workspace
cd /home/hxman/Work/DSA

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Validate drug-level split constraints
python scripts/validate_drug_level_split.py

# Train model (writes models/ and reports/ artifacts)
python scripts/train_model.py --data-path data/drugbank.tab --epochs 20 --batch-size 128 --num-workers 4 --no-resume --checkpoint-path models/drugbank_ddi_simple_gnn.ckpt --out models/drugbank_ddi_simple_gnn.pt --report-dir reports

# Run Streamlit app
streamlit run app.py
```

**Access at**: `http://localhost:8501`

---

## 📂 Project Structure

```
DSA/
├── app.py                              # Main Streamlit app
├── requirements.txt                    # Python dependencies
├── scripts/
│   ├── train_model.py                  # Main training/retraining entrypoint
│   └── validate_drug_level_split.py    # Split integrity checker
├── data/
│   └── drugbank.tab                   # DrugBank interaction database
├── models/
│   ├── drugbank_ddi_simple_gnn.ckpt   # GNN checkpoint
│   └── drugbank_ddi_simple_gnn.pt     # GNN state dict (.pt format)
├── reports/
│   ├── test_confusion_matrix.csv      # Model evaluation metrics
│   ├── test_confusion_matrix.npy      # Confusion matrix (numpy)
│   └── test_per_class_metrics.csv     # Per-class precision/recall
├── plan-drugLensPredictionOutput.prompt.md  # Historical design note
└── README.md                           # This file
```

---

## 🧬 Core Features

### 1. **GNN-Based Prediction**
- Trained on DrugBank molecular graphs
- Encodes: atom types, bond order, aromaticity
- Outputs: multiclass interaction severity

### 2. **KNN Similarity Fallback**
- Builds hybrid feature vectors (Morgan + mechanism features)
- Finds nearest neighbors in clinical knowledge base
- Returns curated interaction descriptions + monitoring guidance

### 3. **Pharmacology-Aware Features**
- 8 core mechanism signals
- 10 functional group counts (carboxylic acid, ester, amide, etc.)
- 6 drug class proxies (NSAID, statin, azole, macrolide, etc.)
- 8 molecular descriptors (MW, LogP, TPSA, H-donors, etc.)

### 4. **Clinical Safety Layer**
Only **2-3 lightweight rules** prevent obvious errors:
- `LOW_CONFIDENCE`: Prediction confidence < 55%
- `WEAK_MECHANISM_BASIS`: High confidence but no pharmacological signals detected
- `DESCRIPTOR_MISMATCH`: Extreme molecular weight difference (>800 Da)

---

## 🔬 ML Models Used

### Models
| Name | Type | Input | Output | Status |
|------|------|-------|--------|--------|
| **GnnDDIModel** | Graph Neural Network | Molecular graphs | Severity class | Primary |
| **KNNModel** | K-Nearest Neighbors | Hybrid feature vectors | Similarity scores | Fallback |

### Feature Dimensions
- **Morgan Fingerprint**: 2048-bit
- **Mechanism Vector**: 8 dimensions (pharmacological signals)
- **Functional Group Vector**: 10 dimensions
- **Class Vector**: 6 dimensions
- **Descriptor Vector**: 8 dimensions
- **Total Hybrid Vector**: ~2,080 dimensions (+ optional ChemBERTa embeddings)

---

## 📈 Prediction Flow

```
User Input (Drug A SMILES, Drug B SMILES)
    ↓
Validate SMILES & Load Models
    ↓
Extract Pharmacological Features
    ├─ Descriptors (MW, LogP, TPSA, etc.)
    ├─ Functional Groups (amide, ester, etc.)
    ├─ Mechanism Signals (8 core signals)
    └─ Drug Classes (NSAID-like, azole-like, etc.)
    ↓
Build Hybrid Pair Vector
    ↓
GNN Prediction (if available)
    └─ Output: Severity class + confidence
    ↓
KNN Fallback (alternative)
    └─ Find similar interactions in knowledge base
    ↓
Refine Mechanism (from predicted signals)
    ├─ CYP inhibitor + substrate? → CYP inhibition
    ├─ Both QT risk? → Additive QT prolongation
    └─ etc.
    ↓
Apply Safety Filter
    └─ Check confidence, mechanism plausibility, descriptor range
    ↓
Return Prediction
    ├─ Interaction type
    ├─ Mechanism explanation
    ├─ Clinical impact
    ├─ Monitoring recommendations
    └─ Confidence score
```

---

## 🧪 Example Usage

### Interactive Dashboard

1. **Enter Drug SMILES** or search by name
2. **Select Prediction Engine** (GNN or KNN)
3. **View Results**:
   - Predicted interaction type
   - Mechanism explanation (CYP inhibition, QT risk, etc.)
   - Clinical impact & monitoring guidance
   - Confidence & reliability metrics
   - Similar known interactions

### Programmatic API

```python
from app import ai_predict_interaction, gnn_predict_interaction

# KNN-based prediction
result = ai_predict_interaction(
    smiles_a="CC(=O)OC1=CC=CC(=C1)C(=O)O",  # Aspirin
    smiles_b="CC(C)CC1=CC=C(C=C1)C(C)C(O)=O",  # Ibuprofen
    name_a="Aspirin",
    name_b="Ibuprofen"
)

# GNN-based prediction
result = gnn_predict_interaction(
    smiles_a="...",
    smiles_b="...",
    name_a="Drug A",
    name_b="Drug B"
)
```

---

## 📚 Data Sources

- **DrugBank**: ~1,700+ curated drug-drug interactions + molecular structures
- **NIH RxNav**: Real-time clinical interaction database
- **Custom Annotations**: Mechanism labels, severity rankings, monitoring guidelines

---

## 🔍 Model Training & Validation

### GNN Training
- **Dataset**: DrugBank DDI dataset + molecular graphs
- **Architecture**: 2-layer GCN + MLP head
- **Loss**: Cross-entropy (multiclass) or binary cross-entropy
- **Metrics**: Accuracy, Precision, Recall, F1-score per class
- **Results**: See `reports/test_per_class_metrics.csv`

### Hybridization Strategy
- Combines graph structure (GNN captures subgraph patterns)
- + Pharmacological mechanisms (8 signals guide learning)
- + Functional class proxies (semantic similarity)
- Result: More transferable, explainable predictions

---

## ⚠️ Limitations & Future Work

### Current Limitations
1. **SMILES Validation**: Invalid SMILES will fail
2. **Model Scope**: Trained on known DrugBank pairs, unknown drug classes may be OOD
3. **No Real-Time FDA Updates**: Database is snapshot-based
4. **No Protein Binding Data**: Assumes similar proteins bind similar drugs

### Future Enhancements
- [ ] Add protein target prediction
- [ ] Integrate real-time PubMed literature mining
- [ ] Support 3+ drug combinations
- [ ] Fine-tune on institution-specific outcome data
- [ ] Dosage/formulation adjustments
- [ ] Temporal interaction modeling (absorption timing)

---

## 📖 References

### Key Papers
1. **Graph Neural Networks for Drug Discovery**: Gilmer et al. (2017)
2. **Molecular Representation Learning**: Schwaller et al. (2019)
3. **DrugBank 5.0**: Wishart et al. (2018)

### Software & Frameworks
- **RDKit**: Cheminformatics toolkit → molecular descriptors, fingerprints
- **PyTorch**: Deep learning framework → GNN implementation
- **scikit-learn**: KNN, feature normalization
- **Streamlit**: Interactive web dashboard
- **TransFormers**: ChemBERTa embeddings (optional)

### Clinical References
- **FDA Approved Drug Interactions**: FDA Orange Book
- **PharmGKB**: Pharmacogenomics Knowledge Base
- **UpToDate**: Clinical evidence summaries

---

## 🔐 Safety & Ethics

### Responsible AI Practices
✅ **Confidence Thresholds**: Low-confidence predictions flagged  
✅ **Mechanism Transparency**: Every prediction explained  
✅ **Clinical Validation**: Backed by literature + expert annotations  
✅ **Comprehensive Output**: Never silent failures  

### Disclaimer
> **This system is a research tool, not a clinical decision system.**  
> Always consult pharmacists, clinical pharmacologists, and FDA resources for critical decisions.

---

## 📞 Support & Contact

- **Project Lead**: [Your Name]
- **Issues**: Check `app.py` error logs and `models/` directory
- **Model Loading**: Ensure `drugbank_ddi_simple_gnn.ckpt` exists in `models/`

---

## 📜 Citation

```bibtex
@software{druglens2026,
  title={DrugLens: Hybrid GNN-Clinical Drug Interaction Diagnostic System},
  author={[Your Team]},
  year={2026},
  url={https://github.com/[repo]},
  note={Biology-aware ML for drug safety}
}
```

---

**Last Updated**: March 27, 2026  
**Version**: 2.0 (Biology-Aware Feature Engineering)  
**Status**: 🟢 Active Development

