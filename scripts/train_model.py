"""Train a pure-PyTorch GNN for DrugBank DDI with clinical side-effect targets.

Enhancements:
- Filters invalid-SMILES rows before dataset construction.
- Exports confusion matrix and per-class F1 reports after test.
- Supports resume-from-checkpoint training.
"""

import argparse
import os
import random
import re
import time
from typing import Dict, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from sklearn.metrics import (
	confusion_matrix,
	f1_score,
	precision_recall_fscore_support,
	roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


ATOM_LIST = [
	"C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al",
	"I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn",
	"H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb",
]
NODE_FEAT_DIM = len(ATOM_LIST) + 1 + 6 + 2

SIDE_EFFECT_LABELS = [
	"Bleeding risk",
	"QT prolongation",
	"CNS depression",
	"Reduced therapeutic efficacy",
	"Hepatotoxicity",
	"Nephrotoxicity",
]

TARGETED_QT_TEST_SMILES = {
	"Azithromycin": "CC[C@H]1C[C@@H](O[C@@H]2[C@@H](C)[C@H](O[C@H]3C[C@@H](N(C)C)[C@@H](O)[C@H](C)O3)[C@@H](C)O[C@H]2C)O[C@@H](C)[C@H](O)[C@@H](C)C(=O)O1",
	"Haloperidol": "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
}

SIDE_EFFECT_PATTERNS = {
	"Bleeding risk": [
		r"bleed",
		r"hemorr",
		r"haemorr",
		r"anticoagul",
		r"platelet",
		r"inr",
	],
	"QT prolongation": [
		r"\bqt\b",
		r"torsad",
		r"arrhythm",
		r"ventricular tachy",
		r"cardiac rhythm",
	],
	"CNS depression": [
		r"\bcns\b",
		r"sedat",
		r"drows",
		r"somnol",
		r"respiratory depression",
		r"central nervous system",
	],
	"Reduced therapeutic efficacy": [
		r"efficacy.*decreas",
		r"decreas.*efficacy",
		r"reduce.*effect",
		r"lower.*concentration",
		r"decreas.*serum concentration",
		r"diminish",
		r"subtherapeutic",
	],
	"Hepatotoxicity": [
		r"hepatotox",
		r"liver",
		r"hepatic",
		r"transaminase",
		r"\balt\b",
		r"\bast\b",
	],
	"Nephrotoxicity": [
		r"nephrotox",
		r"renal",
		r"kidney",
		r"creatinin",
	],
}

QT_DRUG_FAMILY_KEYWORDS = {
	"antibiotics": [
		"antibiotic",
		"macrolide",
		"azithromycin",
		"clarithromycin",
		"erythromycin",
		"fluoroquinolone",
		"moxifloxacin",
		"levofloxacin",
		"ciprofloxacin",
	],
	"antipsychotics": [
		"antipsychotic",
		"haloperidol",
		"quetiapine",
		"risperidone",
		"ziprasidone",
		"olanzapine",
		"chlorpromazine",
	],
	"antiarrhythmics": [
		"antiarrhythmic",
		"amiodarone",
		"sotalol",
		"quinidine",
		"dofetilide",
		"procainamide",
	],
}


def atom_feature(atom) -> np.ndarray:
	symbol = atom.GetSymbol()
	symbol_idx = ATOM_LIST.index(symbol) if symbol in ATOM_LIST else len(ATOM_LIST)
	symbol_feat = np.zeros(len(ATOM_LIST) + 1, dtype=np.float32)
	symbol_feat[symbol_idx] = 1.0

	degree = atom.GetDegree()
	degree_feat = np.zeros(6, dtype=np.float32)
	degree_feat[min(degree, 5)] = 1.0

	formal_charge = atom.GetFormalCharge()
	charge_feat = np.array([formal_charge], dtype=np.float32)
	aromatic_feat = np.array([float(atom.GetIsAromatic())], dtype=np.float32)
	return np.concatenate([symbol_feat, degree_feat, charge_feat, aromatic_feat])


def smiles_to_graph(smiles: str, max_nodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		raise ValueError("Invalid SMILES reached graph conversion after filtering.")

	num_atoms = mol.GetNumAtoms()
	num_use = min(num_atoms, max_nodes)

	X = np.zeros((max_nodes, NODE_FEAT_DIM), dtype=np.float32)
	for i in range(num_use):
		X[i] = atom_feature(mol.GetAtomWithIdx(i))

	A = np.zeros((max_nodes, max_nodes), dtype=np.float32)
	for bond in mol.GetBonds():
		i = bond.GetBeginAtomIdx()
		j = bond.GetEndAtomIdx()
		if i < max_nodes and j < max_nodes:
			A[i, j] = 1.0
			A[j, i] = 1.0

	for i in range(num_use):
		A[i, i] = 1.0

	mask = np.zeros((max_nodes,), dtype=np.float32)
	mask[:num_use] = 1.0
	return A, X, mask


def _is_valid_smiles(smiles: str, cache: Dict[str, bool]) -> bool:
	if smiles in cache:
		return cache[smiles]
	if not isinstance(smiles, str) or not smiles.strip():
		cache[smiles] = False
		return False
	cache[smiles] = Chem.MolFromSmiles(smiles) is not None
	return cache[smiles]


def _normalize_interaction_text(text: str) -> str:
	t = str(text or "").strip().lower()
	t = t.replace("#drug1", "drug a").replace("#drug2", "drug b")
	return t


def map_description_to_side_effect(description: str) -> Optional[str]:
	text = _normalize_interaction_text(description)
	if not text:
		return None

	for label in SIDE_EFFECT_LABELS:
		patterns = SIDE_EFFECT_PATTERNS[label]
		if any(re.search(pat, text) for pat in patterns):
			return label

	return None


def build_clinical_side_effect_dataset(data_path: str) -> pd.DataFrame:
	"""
	Build multiclass training rows with direct clinical side-effect targets.

	Mapping path: molecular structure pair -> clinical side-effect class (Y).
	No mechanism-class intermediary labels are retained.
	"""
	df = pd.read_csv(data_path, sep="\t")
	required_cols = {"X1", "X2", "Map"}
	if not required_cols.issubset(df.columns):
		raise ValueError(f"Missing required columns {required_cols} in {data_path}")

	out = pd.DataFrame({
		"Drug1": df["X1"].astype(str).str.strip('"').str.strip(),
		"Drug2": df["X2"].astype(str).str.strip('"').str.strip(),
		"Map": df["Map"].astype(str),
	})
	out = out[(out["Drug1"] != "") & (out["Drug2"] != "")].copy()
	out["Y"] = out["Map"].map(map_description_to_side_effect)

	total = len(out)
	out = out[out["Y"].notna()].reset_index(drop=True)
	dropped = total - len(out)
	print(f"Mapped {len(out)} rows to clinical side-effect labels; dropped {dropped} unmapped rows.")

	return out[["Drug1", "Drug2", "Y", "Map"]]


def print_class_distribution(df: pd.DataFrame, split_name: str, ratio_warn_threshold: float = 8.0) -> Dict[str, float]:
	counts = df["Y"].value_counts().reindex(SIDE_EFFECT_LABELS, fill_value=0)
	total = int(counts.sum())

	print(f"\n[{split_name}] Class distribution")
	for label, count in counts.items():
		pct = (100.0 * count / total) if total else 0.0
		print(f"  - {label:<30} {int(count):>7} ({pct:6.2f}%)")

	nonzero = counts[counts > 0]
	imbalance_ratio = float(nonzero.max() / nonzero.min()) if len(nonzero) > 0 else float("inf")
	missing = counts[counts == 0].index.tolist()

	if missing:
		print(f"  ! Missing classes: {missing}")
	if np.isfinite(imbalance_ratio) and imbalance_ratio >= ratio_warn_threshold:
		print(f"  ! Class imbalance warning: max/min ratio={imbalance_ratio:.2f} (threshold={ratio_warn_threshold:.2f})")

	return {
		"total": float(total),
		"imbalance_ratio": float(imbalance_ratio),
		"num_missing_classes": float(len(missing)),
	}


def get_fixed_class_counts(df: pd.DataFrame) -> pd.Series:
	return df["Y"].value_counts().reindex(SIDE_EFFECT_LABELS, fill_value=0).astype(np.int64)


def _min_additions_for_fraction(current_class_count: int, current_total: int, target_fraction: float) -> int:
	"""Minimum additional rows needed so class_count/total reaches target_fraction."""
	if not (0.0 < target_fraction < 1.0):
		raise ValueError("target_fraction must be in (0, 1).")
	numerator = (target_fraction * current_total) - current_class_count
	if numerator <= 0:
		return 0
	return int(np.ceil(numerator / (1.0 - target_fraction)))


def augment_qt_targeted_rows(
	train_df: pd.DataFrame,
	qt_min_fraction: float = 0.15,
	seed: int = 42,
) -> pd.DataFrame:
	"""Boost QT rows using family-targeted duplication for antibiotics/antipsychotics/antiarrhythmics."""
	if qt_min_fraction <= 0.0 or qt_min_fraction >= 1.0:
		raise ValueError("qt_min_fraction must be in (0, 1).")

	rng = np.random.default_rng(seed)
	augmented = train_df.copy().reset_index(drop=True)

	qt_label = "QT prolongation"
	qt_rows = augmented[augmented["Y"] == qt_label]
	if qt_rows.empty:
		raise ValueError("QT class has zero samples in training split; cannot enforce QT >= 15%.")

	current_qt = int(len(qt_rows))
	current_total = int(len(augmented))
	need = _min_additions_for_fraction(current_qt, current_total, qt_min_fraction)
	if need <= 0:
		print(
			f"QT representation already meets target: {current_qt}/{current_total} "
			f"({100.0 * current_qt / max(current_total, 1):.2f}%)."
		)
		return augmented

	map_text = qt_rows["Map"].astype(str).str.lower().fillna("")
	mask = pd.Series(False, index=qt_rows.index)
	family_hits = {}
	for family, keywords in QT_DRUG_FAMILY_KEYWORDS.items():
		family_mask = map_text.str.contains("|".join(re.escape(k) for k in keywords), regex=True)
		family_hits[family] = int(family_mask.sum())
		mask = mask | family_mask

	targeted_qt = qt_rows[mask]
	if targeted_qt.empty:
		targeted_qt = qt_rows
		print(
			"No explicit QT family keywords found in interaction text; "
			"falling back to all QT rows for augmentation."
		)

	take_idx = rng.integers(0, len(targeted_qt), size=need)
	dup = targeted_qt.iloc[take_idx].copy()
	augmented = pd.concat([augmented, dup], ignore_index=True)

	new_qt = int((augmented["Y"] == qt_label).sum())
	new_total = int(len(augmented))
	print("\nApplied targeted QT augmentation:")
	print(f"  - Source QT rows: {len(qt_rows)}")
	print(
		"  - Family keyword matches: "
		+ ", ".join([f"{k}={v}" for k, v in family_hits.items()])
	)
	print(f"  - Duplicated QT rows: {need}")
	print(f"  - QT ratio after augmentation: {new_qt}/{new_total} ({100.0 * new_qt / max(new_total, 1):.2f}%)")
	return augmented


def rebalance_min_fraction(
	train_df: pd.DataFrame,
	min_fraction: float = 0.10,
	seed: int = 42,
	max_rounds: int = 25,
) -> pd.DataFrame:
	"""Duplicate minority-class rows until each class reaches min_fraction of training data."""
	if min_fraction <= 0.0 or min_fraction >= 1.0:
		raise ValueError("min_fraction must be in (0, 1).")

	rng = np.random.default_rng(seed)
	rebalanced = train_df.copy().reset_index(drop=True)

	for _ in range(max_rounds):
		counts = get_fixed_class_counts(rebalanced)
		total = int(counts.sum())
		min_required = int(np.ceil(min_fraction * total))

		deficits = {
			label: int(max(0, min_required - int(counts[label])))
			for label in SIDE_EFFECT_LABELS
		}
		deficits = {label: need for label, need in deficits.items() if need > 0}

		if not deficits:
			return rebalanced

		new_chunks = [rebalanced]
		for label, need in deficits.items():
			subset = rebalanced[rebalanced["Y"] == label]
			if subset.empty:
				raise ValueError(
					f"Cannot rebalance class '{label}' because it has zero samples in training split."
				)
			take_idx = rng.integers(0, len(subset), size=need)
			new_chunks.append(subset.iloc[take_idx].copy())

		rebalanced = pd.concat(new_chunks, ignore_index=True)

	raise RuntimeError(
		"Class-balance rebalancing did not converge within max_rounds; "
		"try increasing max_rounds or lowering min_fraction."
	)


def rebalance_min_fraction_by_class(
	train_df: pd.DataFrame,
	min_fraction_by_class: Dict[str, float],
	default_min_fraction: float = 0.10,
	seed: int = 42,
	max_rounds: int = 30,
) -> pd.DataFrame:
	"""Duplicate rows until each class reaches its configured minimum fraction."""
	if not (0.0 <= default_min_fraction < 1.0):
		raise ValueError("default_min_fraction must be in [0, 1).")

	for label, frac in min_fraction_by_class.items():
		if label not in SIDE_EFFECT_LABELS:
			raise ValueError(f"Unknown class label in min_fraction_by_class: {label}")
		if not (0.0 < frac < 1.0):
			raise ValueError(f"Fraction for class '{label}' must be in (0, 1).")

	rng = np.random.default_rng(seed)
	rebalanced = train_df.copy().reset_index(drop=True)

	for _ in range(max_rounds):
		counts = get_fixed_class_counts(rebalanced)
		total = int(counts.sum())

		deficits = {}
		for label in SIDE_EFFECT_LABELS:
			target = float(min_fraction_by_class.get(label, default_min_fraction))
			need = _min_additions_for_fraction(int(counts[label]), total, target)
			if need > 0:
				deficits[label] = int(need)

		if not deficits:
			return rebalanced

		new_chunks = [rebalanced]
		for label, need in deficits.items():
			subset = rebalanced[rebalanced["Y"] == label]
			if subset.empty:
				raise ValueError(
					f"Cannot rebalance class '{label}' because it has zero samples in training split."
				)
			take_idx = rng.integers(0, len(subset), size=need)
			new_chunks.append(subset.iloc[take_idx].copy())

		rebalanced = pd.concat(new_chunks, ignore_index=True)

	raise RuntimeError(
		"Class-specific rebalancing did not converge within max_rounds; "
		"try increasing max_rounds or lowering target fractions."
	)


def log_qt_misclassifications(
	cm: np.ndarray,
	class_to_idx: dict,
	phase_label: str,
	epoch: Optional[int] = None,
) -> dict:
	"""Summarize where QT samples are misclassified in a confusion matrix."""
	qt_idx = class_to_idx["QT prolongation"]
	idx_to_class = {v: k for k, v in class_to_idx.items()}

	qt_row = cm[qt_idx, :]
	qt_total = int(qt_row.sum())
	qt_correct = int(qt_row[qt_idx])
	qt_misclassified_total = int(qt_total - qt_correct)

	record = {
		"phase": phase_label,
		"epoch": int(epoch) if epoch is not None else -1,
		"qt_total": qt_total,
		"qt_correct": qt_correct,
		"qt_misclassified_total": qt_misclassified_total,
		"qt_recall_pct": (100.0 * qt_correct / qt_total) if qt_total > 0 else 0.0,
	}

	print(f"  {phase_label} QT confusion row (true QT -> predicted class):")
	for pred_idx in range(cm.shape[1]):
		label = idx_to_class[pred_idx]
		count = int(qt_row[pred_idx])
		rate = (100.0 * count / qt_total) if qt_total > 0 else 0.0
		record[f"qt_to_{label}"] = count
		record[f"qt_to_{label}_pct"] = rate
		print(f"    - {label:<30} {count:>5} ({rate:5.1f}%)")

	return record


def enforce_drug_level_split_df(
	df: pd.DataFrame,
	train_ratio: float = 0.8,
	valid_ratio: float = 0.1,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Split by unique drugs so train/valid/test have zero drug overlap."""
	if train_ratio <= 0 or valid_ratio < 0 or train_ratio + valid_ratio >= 1.0:
		raise ValueError("Require train_ratio > 0, valid_ratio >= 0, and train_ratio + valid_ratio < 1.0")

	unique_drugs = pd.unique(pd.concat([df["Drug1"], df["Drug2"]], axis=0))
	rng = np.random.default_rng(seed)
	rng.shuffle(unique_drugs)

	n = len(unique_drugs)
	n_train = int(n * train_ratio)
	n_valid = int(n * valid_ratio)

	train_drugs = set(unique_drugs[:n_train])
	valid_drugs = set(unique_drugs[n_train:n_train + n_valid])
	test_drugs = set(unique_drugs[n_train + n_valid:])

	if train_drugs & valid_drugs or train_drugs & test_drugs or valid_drugs & test_drugs:
		raise AssertionError("Drug-level split overlap detected across partitions.")

	train_mask = df["Drug1"].isin(train_drugs) & df["Drug2"].isin(train_drugs)
	valid_mask = df["Drug1"].isin(valid_drugs) & df["Drug2"].isin(valid_drugs)
	test_mask = df["Drug1"].isin(test_drugs) & df["Drug2"].isin(test_drugs)

	train_df = df.loc[train_mask].reset_index(drop=True)
	valid_df = df.loc[valid_mask].reset_index(drop=True)
	test_df = df.loc[test_mask].reset_index(drop=True)

	discarded = len(df) - (len(train_df) + len(valid_df) + len(test_df))
	print("\nDrug-level split summary")
	print(f"  Unique drugs total: {n}")
	print(f"  Train drugs: {len(train_drugs)} | Valid drugs: {len(valid_drugs)} | Test drugs: {len(test_drugs)}")
	print(f"  Rows -> train: {len(train_df)}, valid: {len(valid_df)}, test: {len(test_df)}, discarded-mixed: {discarded}")

	if len(valid_df) <= 0:
		raise AssertionError(
			"Drug-level split produced an empty validation set. "
			"Adjust split ratios or data coverage to ensure real evaluation."
		)
	if len(test_df) <= 0:
		raise AssertionError(
			"Drug-level split produced an empty test set. "
			"Adjust split ratios or data coverage to ensure real evaluation."
		)

	return train_df, valid_df, test_df


def filter_invalid_smiles_rows(df: pd.DataFrame, split_name: str, cache: Dict[str, bool]) -> pd.DataFrame:
	mask = df["Drug1"].map(lambda s: _is_valid_smiles(s, cache)) & df["Drug2"].map(lambda s: _is_valid_smiles(s, cache))
	dropped = int((~mask).sum())
	if dropped > 0:
		print(f"Filtered {dropped} invalid-SMILES rows from {split_name} split.")
	return df.loc[mask].reset_index(drop=True)


class DDIGNNDataset(Dataset):
	def __init__(
		self,
		df: pd.DataFrame,
		max_nodes: int,
		task_type: str,
		label_cols: Optional[list[str]],
		graph_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
		class_to_idx: Optional[dict] = None,
	):
		self.df = df.reset_index(drop=True)
		self.max_nodes = max_nodes
		self.task_type = task_type
		self.label_cols = label_cols
		self.graph_cache = graph_cache
		self.class_to_idx = class_to_idx or {}

	def __len__(self) -> int:
		return len(self.df)

	def _get_graph(self, smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		if smiles not in self.graph_cache:
			self.graph_cache[smiles] = smiles_to_graph(smiles, self.max_nodes)
		return self.graph_cache[smiles]

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		s1 = row["Drug1"]
		s2 = row["Drug2"]

		A1, X1, m1 = self._get_graph(s1)
		A2, X2, m2 = self._get_graph(s2)

		if self.task_type == "multilabel":
			y = row[self.label_cols].values.astype(np.float32)
			y_tensor = torch.tensor(y, dtype=torch.float32)
		elif self.task_type == "multiclass":
			y = np.int64(self.class_to_idx[row["Y"]])
			y_tensor = torch.tensor(y, dtype=torch.long)
		else:
			y = np.array(row["Y"], dtype=np.float32)
			if y.ndim == 0:
				y = np.array([y], dtype=np.float32)
			y_tensor = torch.tensor(y, dtype=torch.float32)

		return (
			torch.tensor(A1, dtype=torch.float32),
			torch.tensor(X1, dtype=torch.float32),
			torch.tensor(m1, dtype=torch.float32),
			torch.tensor(A2, dtype=torch.float32),
			torch.tensor(X2, dtype=torch.float32),
			torch.tensor(m2, dtype=torch.float32),
			y_tensor,
		)


def collate_fn(batch):
	A1, X1, m1, A2, X2, m2, y = zip(*batch)
	return (
		torch.stack(A1, dim=0),
		torch.stack(X1, dim=0),
		torch.stack(m1, dim=0),
		torch.stack(A2, dim=0),
		torch.stack(X2, dim=0),
		torch.stack(m2, dim=0),
		torch.stack(y, dim=0),
	)


class SimpleGCNLayer(nn.Module):
	def __init__(self, in_dim: int, out_dim: int):
		super().__init__()
		self.lin = nn.Linear(in_dim, out_dim)

	def forward(self, A, X):
		AX = torch.bmm(A, X)
		H = self.lin(AX)
		return F.relu(H)


class AttentionPooling(nn.Module):
	def __init__(self, hidden_dim: int):
		super().__init__()
		self.attention_layer = nn.Linear(hidden_dim, 1)

	def forward(self, H, mask):
		attn_logits = self.attention_layer(H).squeeze(-1)
		attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
		attn_weights = F.softmax(attn_logits, dim=1)
		pooled = torch.bmm(attn_weights.unsqueeze(1), H).squeeze(1)
		return pooled, attn_weights


class DrugGNN(nn.Module):
	def __init__(self, node_dim: int, hidden_dim: int, out_dim: int):
		super().__init__()
		self.gcn1 = SimpleGCNLayer(node_dim, hidden_dim)
		self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
		self.attention = AttentionPooling(hidden_dim)
		self.lin = nn.Linear(hidden_dim, out_dim)

	def forward(self, A, X, mask, return_attn: bool = False):
		H = self.gcn1(A, X)
		H = self.gcn2(A, H)
		graph_emb, attn_weights = self.attention(H, mask)
		out = self.lin(graph_emb)
		if return_attn:
			return out, attn_weights, H
		return out


class DDIGNNModel(nn.Module):
	def __init__(self, node_dim: int, hidden_dim: int, num_labels: int):
		super().__init__()
		self.gnn_a = DrugGNN(node_dim, hidden_dim, hidden_dim)
		self.gnn_b = DrugGNN(node_dim, hidden_dim, hidden_dim)
		self.fc1 = nn.Linear(hidden_dim * 4, 256)
		self.fc2 = nn.Linear(256, num_labels)
		self.dropout = nn.Dropout(0.3)

	def forward(self, A1, X1, m1, A2, X2, m2):
		h1 = self.gnn_a(A1, X1, m1)
		h2 = self.gnn_b(A2, X2, m2)
		h_diff = torch.abs(h1 - h2)
		h_prod = h1 * h2
		h = torch.cat([h1, h2, h_diff, h_prod], dim=1)
		h = F.relu(self.fc1(h))
		h = self.dropout(h)
		return self.fc2(h)


def infer_task_config(train_df: pd.DataFrame):
	if "Y" in train_df.columns:
		ex_y = train_df["Y"].iloc[0]
		if isinstance(ex_y, (list, np.ndarray)):
			return {
				"task_type": "multilabel",
				"num_outputs": len(ex_y),
				"label_cols": None,
				"class_to_idx": None,
			}

		y_unique = sorted(train_df["Y"].dropna().unique().tolist())
		if set(y_unique).issubset({0, 1}):
			return {
				"task_type": "binary",
				"num_outputs": 1,
				"label_cols": None,
				"class_to_idx": None,
			}

		class_to_idx = {label: idx for idx, label in enumerate(y_unique)}
		return {
			"task_type": "multiclass",
			"num_outputs": len(y_unique),
			"label_cols": None,
			"class_to_idx": class_to_idx,
		}

	label_cols = [c for c in train_df.columns if c not in ["Drug1", "Drug2"]]
	return {
		"task_type": "multilabel",
		"num_outputs": len(label_cols),
		"label_cols": label_cols,
		"class_to_idx": None,
	}


def summarize_drug_overlap(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
	train_drugs = set(train_df["Drug1"]).union(set(train_df["Drug2"]))
	valid_drugs = set(valid_df["Drug1"]).union(set(valid_df["Drug2"]))
	test_drugs = set(test_df["Drug1"]).union(set(test_df["Drug2"]))

	return {
		"train_val_overlap": len(train_drugs.intersection(valid_drugs)),
		"train_test_overlap": len(train_drugs.intersection(test_drugs)),
		"valid_test_overlap": len(valid_drugs.intersection(test_drugs)),
		"train_unique_drugs": len(train_drugs),
		"valid_unique_drugs": len(valid_drugs),
		"test_unique_drugs": len(test_drugs),
	}


def safe_multiclass_auc(y_true: np.ndarray, y_score: np.ndarray, num_outputs: int) -> float:
	y_true_bin = label_binarize(y_true, classes=np.arange(num_outputs))
	valid_cols = []
	for c in range(num_outputs):
		col = y_true_bin[:, c]
		if col.sum() > 0 and col.sum() < len(col):
			valid_cols.append(c)

	if not valid_cols:
		return float("nan")

	try:
		return float(roc_auc_score(y_true_bin[:, valid_cols], y_score[:, valid_cols], average="macro"))
	except ValueError:
		return float("nan")


def collect_predictions(model, loader, device, task_type: str):
	model.eval()
	y_true_chunks = []
	y_score_chunks = []

	with torch.no_grad():
		for A1, X1, m1, A2, X2, m2, y in loader:
			A1, X1, m1 = A1.to(device), X1.to(device), m1.to(device)
			A2, X2, m2 = A2.to(device), X2.to(device), m2.to(device)
			logits = model(A1, X1, m1, A2, X2, m2)

			if task_type == "multiclass":
				scores = torch.softmax(logits, dim=1)
			else:
				scores = torch.sigmoid(logits)

			y_true_chunks.append(y.cpu().numpy())
			y_score_chunks.append(scores.cpu().numpy())

	y_true = np.concatenate(y_true_chunks, axis=0)
	y_score = np.concatenate(y_score_chunks, axis=0)

	if task_type == "multiclass":
		y_pred = y_score.argmax(axis=1)
	elif task_type == "binary":
		y_pred = (y_score.reshape(-1) >= 0.5).astype(np.int64)
	else:
		y_pred = (y_score >= 0.5).astype(np.int64)

	return y_true, y_pred, y_score


def evaluate(model, loader, device, task_type: str, num_outputs: int) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray]]:
	y_true, y_pred, y_score = collect_predictions(model, loader, device, task_type)

	if task_type == "binary":
		yt = y_true.reshape(-1)
		ys = y_score.reshape(-1)
		metrics = {
			"acc": float((y_pred.reshape(-1) == yt.astype(np.int64)).mean()),
			"macro_f1": float(f1_score(yt.astype(np.int64), y_pred.reshape(-1), average="macro", zero_division=0)),
			"auc": float(roc_auc_score(yt, ys)) if len(np.unique(yt)) > 1 else float("nan"),
		}
		return metrics, (yt, y_pred.reshape(-1), ys)

	if task_type == "multiclass":
		yt = y_true.reshape(-1).astype(np.int64)
		precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
			yt,
			y_pred,
			average="macro",
			zero_division=0,
		)
		metrics = {
			"acc": float((y_pred == yt).mean()),
			"precision_macro": float(precision_macro),
			"recall_macro": float(recall_macro),
			"macro_f1": float(f1_macro),
			"auc_ovr": safe_multiclass_auc(yt, y_score, num_outputs),
		}
		return metrics, (yt, y_pred, y_score)

	metrics = {
		"macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
	}
	try:
		metrics["auc_macro"] = float(roc_auc_score(y_true, y_score, average="macro"))
	except ValueError:
		metrics["auc_macro"] = float("nan")
	return metrics, (y_true, y_pred, y_score)


def per_class_multiclass_report(y_true: np.ndarray, y_pred: np.ndarray, class_to_idx: dict) -> tuple[np.ndarray, pd.DataFrame]:
	labels = np.arange(len(class_to_idx))
	idx_to_class = {v: k for k, v in class_to_idx.items()}
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	precision, recall, f1, support = precision_recall_fscore_support(
		y_true,
		y_pred,
		labels=labels,
		zero_division=0,
	)
	per_class_df = pd.DataFrame(
		{
			"class_index": labels,
			"class_label": [idx_to_class[i] for i in labels],
			"precision": precision,
			"recall": recall,
			"f1": f1,
			"support": support,
		}
	)
	return cm, per_class_df


def resolve_smiles_from_pubchem(drug_name: str, timeout: int = 10) -> Optional[str]:
	url = (
		"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
		f"{quote(drug_name, safe='')}/property/CanonicalSMILES/JSON"
	)
	try:
		resp = requests.get(url, timeout=timeout)
		if not resp.ok:
			return None
		props = resp.json().get("PropertyTable", {}).get("Properties", [])
		if not props:
			return None
		smiles = str(props[0].get("CanonicalSMILES", "")).strip()
		return smiles or None
	except Exception:
		return None


def predict_pair_multiclass(
	model: nn.Module,
	device,
	smiles_a: str,
	smiles_b: str,
	max_nodes: int,
	idx_to_class: dict,
) -> Optional[dict]:
	g1 = smiles_to_graph(smiles_a, max_nodes)
	g2 = smiles_to_graph(smiles_b, max_nodes)

	A1, X1, m1 = g1
	A2, X2, m2 = g2

	with torch.no_grad():
		A1_t = torch.tensor(A1, dtype=torch.float32, device=device).unsqueeze(0)
		X1_t = torch.tensor(X1, dtype=torch.float32, device=device).unsqueeze(0)
		m1_t = torch.tensor(m1, dtype=torch.float32, device=device).unsqueeze(0)
		A2_t = torch.tensor(A2, dtype=torch.float32, device=device).unsqueeze(0)
		X2_t = torch.tensor(X2, dtype=torch.float32, device=device).unsqueeze(0)
		m2_t = torch.tensor(m2, dtype=torch.float32, device=device).unsqueeze(0)

		logits = model(A1_t, X1_t, m1_t, A2_t, X2_t, m2_t)
		probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
		pred_idx = int(np.argmax(probs))

	return {
		"pred_idx": pred_idx,
		"pred_label": str(idx_to_class[pred_idx]),
		"pred_prob": float(probs[pred_idx]),
		"probs": probs,
	}


def export_multiclass_reports(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	class_to_idx: dict,
	report_dir: str,
) -> None:
	os.makedirs(report_dir, exist_ok=True)
	labels = np.arange(len(class_to_idx))
	idx_to_class = {v: k for k, v in class_to_idx.items()}

	cm = confusion_matrix(y_true, y_pred, labels=labels)
	class_names = [str(idx_to_class[i]) for i in labels]
	cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

	cm_csv_path = os.path.join(report_dir, "test_confusion_matrix.csv")
	cm_npy_path = os.path.join(report_dir, "test_confusion_matrix.npy")
	cm_df.to_csv(cm_csv_path)
	np.save(cm_npy_path, cm)

	precision, recall, f1, support = precision_recall_fscore_support(
		y_true,
		y_pred,
		labels=labels,
		zero_division=0,
	)
	per_class_df = pd.DataFrame(
		{
			"class_index": labels,
			"class_label": class_names,
			"precision": precision,
			"recall": recall,
			"f1": f1,
			"support": support,
		}
	)
	per_class_path = os.path.join(report_dir, "test_per_class_metrics.csv")
	per_class_df.to_csv(per_class_path, index=False)

	print(f"Saved confusion matrix CSV: {cm_csv_path}")
	print(f"Saved confusion matrix NPY: {cm_npy_path}")
	print(f"Saved per-class metrics CSV: {per_class_path}")


def save_checkpoint(
	ckpt_path: str,
	epoch: int,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	scheduler,
	best_score: float,
	best_epoch: int,
	patience_counter: int,
	task_type: str,
	class_to_idx: Optional[dict],
	max_nodes: int,
	hidden_dim: int,
	num_outputs: int,
) -> None:
	ckpt = {
		"epoch": epoch,
		"model_state": model.state_dict(),
		"optimizer_state": optimizer.state_dict(),
		"scheduler_state": scheduler.state_dict() if scheduler is not None else None,
		"best_score": best_score,
		"best_epoch": best_epoch,
		"patience_counter": patience_counter,
		"task_type": task_type,
		"class_to_idx": class_to_idx,
		"max_nodes": max_nodes,
		"hidden_dim": hidden_dim,
		"num_outputs": num_outputs,
	}
	out_dir = os.path.dirname(ckpt_path)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
	torch.save(ckpt, ckpt_path)


def load_checkpoint(
	ckpt_path: str,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	scheduler,
	device,
):
	ckpt = torch.load(ckpt_path, map_location=device)
	model.load_state_dict(ckpt["model_state"])
	optimizer.load_state_dict(ckpt["optimizer_state"])
	if scheduler is not None and ckpt.get("scheduler_state") is not None:
		scheduler.load_state_dict(ckpt["scheduler_state"])
	return (
		int(ckpt.get("epoch", 0)),
		float(ckpt.get("best_score", -float("inf"))),
		int(ckpt.get("best_epoch", 0)),
		int(ckpt.get("patience_counter", 0)),
	)


def main():
	parser = argparse.ArgumentParser(description="Train simple DrugBank DDI GNN (pure PyTorch)")
	parser.add_argument("--data-path", type=str, default="data/drugbank.tab")
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--max-nodes", type=int, default=70)
	parser.add_argument("--hidden-dim", type=int, default=128)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--grad-clip", type=float, default=1.0)
	parser.add_argument("--qt-loss-boost", type=float, default=1.5)
	parser.add_argument("--patience", type=int, default=5)
	parser.add_argument("--min-delta", type=float, default=1e-4)
	parser.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--checkpoint-path", type=str, default="models/drugbank_ddi_simple_gnn.ckpt")
	parser.add_argument("--report-dir", type=str, default="reports")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--out", type=str, default="models/drugbank_ddi_simple_gnn.pt")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Device:", device)

	set_seed(args.seed)
	RDLogger.DisableLog("rdApp.error")

	print(f"Loading DrugBank tab from: {args.data_path}")
	clinical_df = build_clinical_side_effect_dataset(args.data_path)
	if clinical_df.empty:
		raise ValueError("No rows mapped to clinical side-effect classes. Check mapping rules and source data.")

	print_class_distribution(clinical_df, "All mapped rows")

	# Ensure direct molecular-structure -> clinical-side-effect targets via drug-level split.
	train_df, valid_df, test_df = enforce_drug_level_split_df(
		clinical_df,
		train_ratio=0.8,
		valid_ratio=0.1,
		seed=args.seed,
	)

	smiles_valid_cache: Dict[str, bool] = {}
	train_df = filter_invalid_smiles_rows(train_df, "train", smiles_valid_cache)
	valid_df = filter_invalid_smiles_rows(valid_df, "valid", smiles_valid_cache)
	test_df = filter_invalid_smiles_rows(test_df, "test", smiles_valid_cache)

	if train_df.empty or valid_df.empty or test_df.empty:
		raise ValueError("One or more splits are empty after filtering; adjust split ratios or mapping coverage.")

	print("Train:", train_df.shape)
	print("Valid:", valid_df.shape)
	print("Test :", test_df.shape)

	print_class_distribution(train_df, "Train")
	print_class_distribution(valid_df, "Valid")
	print_class_distribution(test_df, "Test")
	overlap_summary = summarize_drug_overlap(train_df, valid_df, test_df)
	print("Drug overlap summary:", overlap_summary)
	if (
		overlap_summary["train_val_overlap"] > 0
		or overlap_summary["train_test_overlap"] > 0
		or overlap_summary["valid_test_overlap"] > 0
	):
		raise ValueError("Drug-level split leakage detected after filtering.")

	cfg = infer_task_config(train_df)
	task_type = cfg["task_type"]
	label_cols = cfg["label_cols"]
	class_to_idx = {label: idx for idx, label in enumerate(SIDE_EFFECT_LABELS)}
	num_outputs = len(SIDE_EFFECT_LABELS)

	if task_type != "multiclass":
		raise ValueError(f"Expected multiclass clinical-side-effect targets, got task_type={task_type}")

	observed_labels = set(clinical_df["Y"].dropna().unique().tolist())
	if observed_labels != set(SIDE_EFFECT_LABELS):
		missing = sorted(set(SIDE_EFFECT_LABELS) - observed_labels)
		extra = sorted(observed_labels - set(SIDE_EFFECT_LABELS))
		raise ValueError(
			"Clinical label-space mismatch. "
			f"missing={missing}, extra={extra}."
		)

	if num_outputs != 6:
		raise ValueError(f"Expected exactly 6 outputs for app compatibility, got {num_outputs}")

	# Boost QT representation first with targeted family-aware augmentation.
	train_df = augment_qt_targeted_rows(train_df, qt_min_fraction=0.15, seed=args.seed)

	# Class-balance check with class-specific targets: QT >=15%, others >=10%.
	train_counts_before = get_fixed_class_counts(train_df)
	print("\nFixed class counts before balancing:")
	for label in SIDE_EFFECT_LABELS:
		print(f"  - {label:<30} {int(train_counts_before[label]):>7}")

	min_fraction_by_class = {label: 0.10 for label in SIDE_EFFECT_LABELS}
	min_fraction_by_class["QT prolongation"] = 0.15
	train_df = rebalance_min_fraction_by_class(
		train_df,
		min_fraction_by_class=min_fraction_by_class,
		default_min_fraction=0.10,
		seed=args.seed,
	)
	print_class_distribution(train_df, "Train (Rebalanced: QT>=15%, others>=10%)")

	train_counts_after = get_fixed_class_counts(train_df)
	train_total_after = float(train_counts_after.sum())
	qt_ratio_after = float(train_counts_after["QT prolongation"] / train_total_after) if train_total_after > 0 else 0.0
	if qt_ratio_after < 0.15:
		raise ValueError(
			"QT class representation requirement failed after balancing: "
			f"{100.0 * qt_ratio_after:.2f}% < 15.00%."
		)
	print(f"QT representation check passed: {100.0 * qt_ratio_after:.2f}% >= 15.00%.")

	print("Task type:", task_type)
	print("Number of outputs:", num_outputs)
	if task_type == "multiclass":
		print("Detected clinical side-effect classes:", num_outputs)

	graph_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

	train_set = DDIGNNDataset(train_df, args.max_nodes, task_type, label_cols, graph_cache, class_to_idx)
	valid_set = DDIGNNDataset(valid_df, args.max_nodes, task_type, label_cols, graph_cache, class_to_idx)
	test_set = DDIGNNDataset(test_df, args.max_nodes, task_type, label_cols, graph_cache, class_to_idx)

	loader_kwargs = {
		"batch_size": args.batch_size,
		"num_workers": args.num_workers,
		"pin_memory": device.type == "cuda",
		"persistent_workers": args.num_workers > 0,
		"collate_fn": collate_fn,
	}
	# Mandatory oversampling of minority classes via WeightedRandomSampler.
	y_train_idx = train_df["Y"].map(class_to_idx).values
	class_counts_for_sampling = np.bincount(y_train_idx, minlength=num_outputs).astype(np.float64)
	per_class_sample_weight = len(y_train_idx) / (num_outputs * np.clip(class_counts_for_sampling, 1.0, None))
	sample_weights = per_class_sample_weight[y_train_idx]
	train_sampler = WeightedRandomSampler(
		weights=torch.as_tensor(sample_weights, dtype=torch.double),
		num_samples=len(sample_weights),
		replacement=True,
	)

	train_loader = DataLoader(train_set, shuffle=False, sampler=train_sampler, **loader_kwargs)
	valid_loader = DataLoader(valid_set, shuffle=False, **loader_kwargs)
	test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

	model = DDIGNNModel(NODE_FEAT_DIM, args.hidden_dim, num_outputs).to(device)
	print(model)

	if task_type == "multiclass":
		# Mandatory class weights: weight_i = total_samples / (num_classes * class_count_i)
		y_idx = train_df["Y"].map(class_to_idx).values
		class_counts = np.bincount(y_idx, minlength=num_outputs).astype(np.float32)
		class_weights_np = len(y_idx) / (num_outputs * np.clip(class_counts, 1.0, None))
		class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
		qt_idx = int(class_to_idx["QT prolongation"])
		print("\nCrossEntropy class weights:")
		for i, label in enumerate(SIDE_EFFECT_LABELS):
			print(
				f"  - {label:<30} count={int(class_counts[i]):>7} weight={float(class_weights_np[i]):.4f}"
			)
		print(f"QT loss boost factor: {args.qt_loss_boost:.2f}")
		criterion = None
	else:
		criterion = nn.BCEWithLogitsLoss()

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="max",
		factor=0.5,
		patience=2,
		min_lr=1e-6,
	)

	if task_type == "multiclass":
		monitor_key = "macro_f1"
	elif task_type == "binary":
		monitor_key = "auc"
	else:
		monitor_key = "auc_macro"

	start_epoch = 1
	best_score = -float("inf")
	best_epoch = 0
	patience_counter = 0

	if args.resume and os.path.exists(args.checkpoint_path):
		epoch_loaded, best_score, best_epoch, patience_counter = load_checkpoint(
			args.checkpoint_path,
			model,
			optimizer,
			scheduler,
			device,
		)
		start_epoch = epoch_loaded + 1
		print(f"Resumed from checkpoint {args.checkpoint_path} at epoch {epoch_loaded}.")

	if start_epoch > args.epochs:
		print("Checkpoint already reached requested epochs; skipping training loop.")

	qt_val_confusion_records = []

	for epoch in range(start_epoch, args.epochs + 1):
		model.train()
		total_loss = 0.0
		t0 = time.time()

		for A1, X1, m1, A2, X2, m2, y in train_loader:
			A1, X1, m1 = A1.to(device), X1.to(device), m1.to(device)
			A2, X2, m2 = A2.to(device), X2.to(device), m2.to(device)
			y = y.to(device)

			optimizer.zero_grad()
			logits = model(A1, X1, m1, A2, X2, m2)
			if task_type == "multiclass":
				y_long = y.long()
				per_sample_loss = F.cross_entropy(
					logits,
					y_long,
					weight=class_weights,
					reduction="none",
				)
				sample_boost = torch.ones_like(per_sample_loss)
				sample_boost = torch.where(
					y_long == qt_idx,
					torch.full_like(sample_boost, float(args.qt_loss_boost)),
					sample_boost,
				)
				loss = (per_sample_loss * sample_boost).mean()
			else:
				loss = criterion(logits, y.float())
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
			optimizer.step()

			total_loss += loss.item() * A1.size(0)

		train_loss = total_loss / len(train_set)
		val_metrics, (val_y_true, val_y_pred, _) = evaluate(model, valid_loader, device, task_type, num_outputs)
		monitor_val = val_metrics.get(monitor_key, float("nan"))
		if np.isnan(monitor_val):
			monitor_val = val_metrics.get("acc", float("nan"))

		if not np.isnan(monitor_val):
			scheduler.step(monitor_val)

		improved = (not np.isnan(monitor_val)) and (monitor_val > best_score + args.min_delta)
		if improved:
			best_score = monitor_val
			best_epoch = epoch
			patience_counter = 0

			out_dir = os.path.dirname(args.out)
			if out_dir:
				os.makedirs(out_dir, exist_ok=True)
			torch.save(model.state_dict(), args.out)
		else:
			patience_counter += 1

		save_checkpoint(
			args.checkpoint_path,
			epoch,
			model,
			optimizer,
			scheduler,
			best_score,
			best_epoch,
			patience_counter,
			task_type,
			class_to_idx,
			args.max_nodes,
			args.hidden_dim,
			num_outputs,
		)

		lr_now = optimizer.param_groups[0]["lr"]
		metrics_text = " | ".join([f"Val {k}: {v:.4f}" for k, v in val_metrics.items()])
		print(
			f"Epoch {epoch:02d} | Train loss: {train_loss:.4f} | {metrics_text} | "
			f"LR: {lr_now:.6f} | Time: {time.time() - t0:.1f}s"
		)

		if task_type == "multiclass":
			val_cm, val_per_class = per_class_multiclass_report(val_y_true, val_y_pred, class_to_idx)
			val_class_text = " | ".join(
				[
					(
						f"{row['class_label']}:P={row['precision']:.2f},"
						f"R={row['recall']:.2f},F1={row['f1']:.2f}"
					)
					for _, row in val_per_class.iterrows()
				]
			)
			print(f"  Val per-class -> {val_class_text}")
			qt_conf = log_qt_misclassifications(val_cm, class_to_idx, phase_label="Val", epoch=epoch)
			qt_val_confusion_records.append(qt_conf)

		if patience_counter >= args.patience:
			print(
				f"Early stopping at epoch {epoch} "
				f"(best epoch: {best_epoch}, best {monitor_key}: {best_score:.4f})"
			)
			break

	if os.path.exists(args.out):
		model.load_state_dict(torch.load(args.out, map_location=device))

	test_metrics, (y_true, y_pred, _y_score) = evaluate(model, test_loader, device, task_type, num_outputs)
	metrics_text = " | ".join([f"Test {k}: {v:.4f}" for k, v in test_metrics.items()])
	print(f"\n{metrics_text}")

	if task_type == "multiclass" and class_to_idx is not None:
		export_multiclass_reports(y_true, y_pred, class_to_idx, args.report_dir)
		test_cm, _ = per_class_multiclass_report(y_true, y_pred, class_to_idx)
		_ = log_qt_misclassifications(test_cm, class_to_idx, phase_label="Test", epoch=None)
		if qt_val_confusion_records:
			os.makedirs(args.report_dir, exist_ok=True)
			qt_epoch_path = os.path.join(args.report_dir, "val_qt_confusion_per_epoch.csv")
			pd.DataFrame(qt_val_confusion_records).to_csv(qt_epoch_path, index=False)
			print(f"Saved per-epoch QT confusion trace: {qt_epoch_path}")

	# Targeted QT validation requested by product requirement.
	idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx is not None else {}
	azithromycin_smiles = resolve_smiles_from_pubchem("Azithromycin")
	haloperidol_smiles = resolve_smiles_from_pubchem("Haloperidol")
	if not azithromycin_smiles:
		azithromycin_smiles = TARGETED_QT_TEST_SMILES["Azithromycin"]
	if not haloperidol_smiles:
		haloperidol_smiles = TARGETED_QT_TEST_SMILES["Haloperidol"]
	print("\nTargeted QT validation (Azithromycin + Haloperidol):")
	if azithromycin_smiles and haloperidol_smiles and idx_to_class:
		targeted = predict_pair_multiclass(
			model,
			device,
			azithromycin_smiles,
			haloperidol_smiles,
			args.max_nodes,
			idx_to_class,
		)
		if targeted is not None:
			expected_label = "QT prolongation"
			pred_label = targeted["pred_label"]
			pred_prob = targeted["pred_prob"]
			passed = pred_label == expected_label
			print(f"  Expected: {expected_label}")
			print(f"  Predicted: {pred_label} (p={pred_prob:.4f})")
			print(f"  QT-target test passed: {passed}")
		else:
			print("  Targeted test skipped: could not run model prediction for resolved SMILES.")
	else:
		print("  Targeted test skipped: could not resolve SMILES from PubChem for one or both drugs.")

	print("Best model saved to:", args.out)
	print("Latest checkpoint saved to:", args.checkpoint_path)


if __name__ == "__main__":
	main()
