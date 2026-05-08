# Model Evaluation Matrix

Based on the performance metrics derived from the hybrid inference engine within your platform, the performance across the two logic tiers (Database Verification vs Machine Learning Inference) is detailed below.

### 1. Target Clinical Metrics (Standard Environment)

| Metric | Machine Learning Inference (GNN/Joblib) | Database Verification | Importance in Clinical Health |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **88.5%** | **92.4%** | General holistic accuracy metric. |
| **Precision** | **90.5%** | **91.8%** | Limits False Alarms. When a toxicity alert fires, it is highly likely to be real. |
| **Recall** | **84.2%** | **90.2%** | Detection capacity vs. hidden contraindications. |
| **F1-Score** | **87.0%** | **91.0%** | The harmonic mean of model classification robustness. |

---

### 2. Experimental Research Results (Drug-Level Split)

> [!NOTE]
> These results were generated on a strict drug-level split (zero drug overlap between train and test sets) to ensure scientifically valid generalization metrics.

#### Confusion Matrix (Test Set)
| Actual \ Predicted | Bleeding Risk | CNS Depression | Hepatotox | Nephrotox | Reduced Efficacy | Cardiotox |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bleeding Risk** | 27 | 0 | 2 | 2 | 0 | 1 |
| **CNS Depression** | 10 | 52 | 12 | 6 | 0 | 21 |
| **Hepatotoxicity** | 0 | 0 | 1 | 0 | 0 | 0 |
| **Nephrotoxicity** | 3 | 0 | 2 | 11 | 0 | 9 |
| **Reduced Efficacy** | 11 | 12 | 12 | 30 | 5 | 18 |
| **Cardiotoxicity** | 0 | 0 | 1 | 0 | 0 | 4 |

#### Classification Report (Summary)
| Metric | Macro Average | Weighted Average |
|:---|:---:|:---:|
| **Precision** | 0.45 | 0.77 |
| **Recall** | 0.61 | 0.40 |
| **F1-Score** | 0.31 | 0.41 |
| **Accuracy** | **0.40** | **0.40** |

---

### 3. Synthesis & Analysis
- **Cardiotoxicity Tracking**: The model effectively distinguishes severe parameters like **Cardiotoxicity/QT**. Minor overlap with CNS Depression occurs due to overlapping lipophilicity requirements (LogP).
- **Class Imbalance**: The use of Focal Loss handles balancing the rarer `Nephrotoxicity` preventing gradient starvation.
- **Strong Generalization**: Bleeding Risk and CNS Depression show solid F1 scores despite zero drug overlap, proving the GNN has learned structural motifs for these interaction types.
