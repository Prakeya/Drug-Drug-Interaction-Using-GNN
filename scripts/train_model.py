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
import joblib
from sklearn.metrics import (
	confusion_matrix,
	f1_score,
	precision_recall_fscore_support,
	roc_auc_score,
	classification_report,
	balanced_accuracy_score,
	matthews_corrcoef,
	precision_recall_curve,
	auc,
)
from sklearn.preprocessing import label_binarize, StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset


def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# Enforce deterministic algorithms where possible
	torch.use_deterministic_algorithms(True, warn_only=True)
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


import sys
import os

# Add root directory to sys.path for internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.architecture import (
    NODE_FEAT_DIM, ATOM_LIST, DDIGNNModel, FocalLoss, 
    MolecularDescriptorStandardizer, smiles_to_graph
)


# ==========================================
# 🧬 DATA CONFIG & CONSTANTS
# ==========================================

SIDE_EFFECT_LABELS = [
	"Bleeding Risk",
	"CNS Depression",
	"Hepatotoxicity",
	"Nephrotoxicity",
	"Reduced Therapeutic Effect",
	"Cardiotoxicity",
]

TARGETED_QT_TEST_SMILES = {
	"Azithromycin": "CC[C@H]1C[C@@H](O[C@@H]2[C@@H](C)[C@H](O[C@H]3C[C@@H](N(C)C)[C@@H](O)[C@H](C)O3)[C@@H](C)O[C@H]2C)O[C@@H](C)[C@H](O)[C@@H](C)C(=O)O1",
	"Haloperidol": "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
}

SIDE_EFFECT_PATTERNS = {
	"Bleeding Risk": [
		r"bleed", r"hemorr", r"haemorr", r"anticoagul", r"platelet", r"inr",
	],
	"Cardiotoxicity": [
		r"\bqt\b", r"torsad", r"arrhythm", r"ventricular tachy", r"cardiac rhythm", r"heart", r"myocard", r"cardiotox", r"congestive", r"bradycard", r"tachycard",
	],
	"CNS Depression": [
		r"\bcns\b", r"sedat", r"drows", r"somnol", r"respiratory depression", r"central nervous system",
	],
	"Reduced Therapeutic Effect": [
		r"efficacy.*decreas", r"decreas.*efficacy", r"reduce.*effect", r"lower.*concentration", r"decreas.*serum concentration", r"diminish", r"subtherapeutic",
	],
	"Hepatotoxicity": [
		r"hepatotox", r"liver", r"hepatic", r"transaminase", r"\balt\b", r"\bast\b",
	],
	"Nephrotoxicity": [
		r"nephrotox", r"renal", r"kidney", r"creatinin",
	],
}

QT_DRUG_FAMILY_KEYWORDS = {
	"antibiotics": ["antibiotic", "macrolide", "azithromycin", "clarithromycin", "erythromycin", "fluoroquinolone", "moxifloxacin", "levofloxacin", "ciprofloxacin"],
	"antipsychotics": ["antipsychotic", "haloperidol", "quetiapine", "risperidone", "ziprasidone", "olanzapine", "chlorpromazine"],
	"antiarrhythmics": ["antiarrhythmic", "amiodarone", "sotalol", "quinidine", "dofetilide", "procainamide"],
}


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

	qt_label = "Cardiotoxicity"
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
	qt_idx = class_to_idx["Cardiotoxicity"]
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
		descriptor_standardizer: Optional[MolecularDescriptorStandardizer] = None,
	):
		self.df = df.reset_index(drop=True)
		self.max_nodes = max_nodes
		self.task_type = task_type
		self.label_cols = label_cols
		self.graph_cache = graph_cache
		self.class_to_idx = class_to_idx or {}
		self.descriptor_standardizer = descriptor_standardizer

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

		d1_tensor, d2_tensor = None, None
		if self.descriptor_standardizer:
			m1_obj = Chem.MolFromSmiles(s1)
			m2_obj = Chem.MolFromSmiles(s2)
			d1 = self.descriptor_standardizer.transform(m1_obj)
			d2 = self.descriptor_standardizer.transform(m2_obj)
			d1_tensor = torch.tensor(d1, dtype=torch.float32)
			d2_tensor = torch.tensor(d2, dtype=torch.float32)

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
			d1_tensor,
			d2_tensor,
			y_tensor,
		)


def collate_fn(batch):
	A1, X1, m1, A2, X2, m2, D1, D2, y = zip(*batch)
	res = [
		torch.stack(A1, dim=0),
		torch.stack(X1, dim=0),
		torch.stack(m1, dim=0),
		torch.stack(A2, dim=0),
		torch.stack(X2, dim=0),
		torch.stack(m2, dim=0),
	]
	# Handle optional descriptors
	if D1[0] is not None:
		res.append(torch.stack(D1, dim=0))
		res.append(torch.stack(D2, dim=0))
	else:
		res.append(None)
		res.append(None)
	
	res.append(torch.stack(y, dim=0))
	return tuple(res)


# Redundant DDIGNNModel and GNN components removed


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
		for batch in loader:
			# batch: (A1, X1, m1, A2, X2, m2, D1, D2, y)
			A1, X1, m1, A2, X2, m2, D1, D2, y = batch
			A1, X1, m1 = A1.to(device), X1.to(device), m1.to(device)
			A2, X2, m2 = A2.to(device), X2.to(device), m2.to(device)
			if D1 is not None and D2 is not None:
				D1, D2 = D1.to(device), D2.to(device)
			
			logits = model(A1, X1, m1, A2, X2, m2, D1, D2)

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
		y_pred = (y_score >= 0.5).astype(np.float32)

	return y_true, y_pred, y_score


class EarlyStopping:
	"""Closes training if validation metric does not improve."""
	def __init__(self, patience=10, min_delta=0, mode='max'):
		self.patience = patience
		self.min_delta = min_delta
		self.mode = mode
		self.counter = 0
		self.best_score = None
		self.early_stop = False

	def __call__(self, score):
		if self.best_score is None:
			self.best_score = score
		elif self.mode == 'max' and score < self.best_score + self.min_delta:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
		elif self.mode == 'min' and score > self.best_score - self.min_delta:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.counter = 0


def train_epoch(model, loader, optimizer, criterion, device, task_type: str, grad_clip: float = 1.0):
	model.train()
	total_loss = 0.0

	for batch in loader:
		A1, X1, m1, A2, X2, m2, D1, D2, y = batch
		A1, X1, m1 = A1.to(device), X1.to(device), m1.to(device)
		A2, X2, m2 = A2.to(device), X2.to(device), m2.to(device)
		if D1 is not None and D2 is not None:
			D1, D2 = D1.to(device), D2.to(device)
		y = y.to(device)

		optimizer.zero_grad()
		logits = model(A1, X1, m1, A2, X2, m2, D1, D2)
		loss = criterion(logits, y)
		loss.backward()
		
		if grad_clip > 0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			
		optimizer.step()
		total_loss += loss.item()

	return total_loss / len(loader)


def validate(model, loader, criterion, device, task_type: str, num_outputs: int):
	model.eval()
	total_loss = 0.0
	y_true_list, y_pred_list, y_score_list = [], [], []

	with torch.no_grad():
		for batch in loader:
			A1, X1, m1, A2, X2, m2, D1, D2, y = batch
			A1, X1, m1 = A1.to(device), X1.to(device), m1.to(device)
			A2, X2, m2 = A2.to(device), X2.to(device), m2.to(device)
			if D1 is not None and D2 is not None:
				D1, D2 = D1.to(device), D2.to(device)
			y = y.to(device)

			logits = model(A1, X1, m1, A2, X2, m2, D1, D2)
			loss = criterion(logits, y)
			total_loss += loss.item()

			if task_type == "multiclass":
				scores = torch.softmax(logits, dim=1)
				preds = scores.argmax(dim=1)
			else:
				scores = torch.sigmoid(logits)
				preds = (scores >= 0.5).float()

			y_true_list.append(y.cpu().numpy())
			y_pred_list.append(preds.cpu().numpy())
			y_score_list.append(scores.cpu().numpy())

	y_true = np.concatenate(y_true_list, axis=0)
	y_pred = np.concatenate(y_pred_list, axis=0)
	y_score = np.concatenate(y_score_list, axis=0)

	avg_loss = total_loss / len(loader)
	
	if task_type == "multiclass":
		f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
		f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
		auc_val = safe_multiclass_auc(y_true, y_score, num_outputs)
		mcc = matthews_corrcoef(y_true, y_pred)
		bal_acc = balanced_accuracy_score(y_true, y_pred)
	else:
		f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
		f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
		auc_val = roc_auc_score(y_true, y_score, average="macro")
		mcc = matthews_corrcoef(y_true.flatten(), y_pred.flatten())
		bal_acc = balanced_accuracy_score(y_true.flatten(), y_pred.flatten())

	return {
		"loss": avg_loss,
		"f1": f1_macro,
		"f1_weighted": f1_weighted,
		"auc": auc_val,
		"mcc": mcc,
		"bal_acc": bal_acc
	}


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
		p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(yt, y_pred, average="macro", zero_division=0)
		f1_weighted = f1_score(yt, y_pred, average="weighted", zero_division=0)
		mcc = matthews_corrcoef(yt, y_pred)
		bal_acc = balanced_accuracy_score(yt, y_pred)
		
		# Multiclass ROC-AUC and PR-AUC
		y_true_bin = label_binarize(yt, classes=np.arange(num_outputs))
		try:
			auc_roc = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")
		except:
			auc_roc = float("nan")
			
		from sklearn.metrics import average_precision_score
		try:
			auc_pr = average_precision_score(y_true_bin, y_score, average="macro")
		except:
			auc_pr = float("nan")

		metrics = {
			"acc": float((y_pred == yt).mean()),
			"balanced_acc": float(bal_acc),
			"precision_macro": float(p_macro),
			"recall_macro": float(r_macro),
			"macro_f1": float(f1_macro),
			"weighted_f1": float(f1_weighted),
			"mcc": float(mcc),
			"auc_roc": float(auc_roc),
			"auc_pr": float(auc_pr),
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
	parser = argparse.ArgumentParser(description="Advanced DDI GNN Training Pipeline")
	parser.add_argument("--data-path", type=str, default="data/drugbank.tab")
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--max-nodes", type=int, default=70)
	parser.add_argument("--hidden-dim", type=int, default=128)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--grad-clip", type=float, default=1.0)
	parser.add_argument("--patience", type=int, default=10)
	parser.add_argument("--loss-type", type=str, choices=["ce", "weighted_ce", "focal"], default="focal")
	parser.add_argument("--gnn-type", type=str, choices=["gcn", "gat"], default="gcn")
	parser.add_argument("--use-descriptors", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--checkpoint-path", type=str, default="models/ddi_advanced_gnn.ckpt")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--out", type=str, default="models/ddi_advanced_gnn.pt")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device} | Loss: {args.loss_type} | Descriptors: {args.use_descriptors}")

	set_seed(args.seed)
	RDLogger.DisableLog("rdApp.error")

	print(f"Loading data from: {args.data_path}")
	clinical_df = build_clinical_side_effect_dataset(args.data_path)
	
	# DATA QUALITY CHECKS
	print("\n[DATA QUALITY] Running integrity checks...")
	all_smiles = pd.concat([clinical_df["Drug1"], clinical_df["Drug2"]])
	dupes = all_smiles.duplicated().sum()
	print(f"  - Duplicate SMILES in raw dataset: {dupes}")
	
	# Drug-level split
	train_df, valid_df, test_df = enforce_drug_level_split_df(clinical_df, train_ratio=0.8, valid_ratio=0.1, seed=args.seed)

	# Leakage Verification
	overlap = summarize_drug_overlap(train_df, valid_df, test_df)
	print(f"  - Split Leakage Check: {overlap['train_val_overlap'] + overlap['train_test_overlap'] + overlap['valid_test_overlap']} drugs overlap (target: 0)")
	if overlap["train_test_overlap"] > 0:
		raise ValueError("CRITICAL: Leakage detected between Train and Test sets.")

	smiles_valid_cache: Dict[str, bool] = {}
	train_df = filter_invalid_smiles_rows(train_df, "train", smiles_valid_cache)
	valid_df = filter_invalid_smiles_rows(valid_df, "valid", smiles_valid_cache)
	test_df = filter_invalid_smiles_rows(test_df, "test", smiles_valid_cache)

	class_to_idx = {label: idx for idx, label in enumerate(SIDE_EFFECT_LABELS)}
	num_outputs = len(SIDE_EFFECT_LABELS)
	task_type = "multiclass"

	# DESCRIPTOR STANDARDIZATION
	standardizer = None
	if args.use_descriptors:
		print("\n[FE] Fitting Molecular Descriptor Standardizer on Training Set...")
		standardizer = MolecularDescriptorStandardizer()
		train_smiles = list(set(train_df["Drug1"].tolist() + train_df["Drug2"].tolist()))
		standardizer.fit(train_smiles)

	graph_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
	train_set = DDIGNNDataset(train_df, args.max_nodes, task_type, None, graph_cache, class_to_idx, standardizer)
	valid_set = DDIGNNDataset(valid_df, args.max_nodes, task_type, None, graph_cache, class_to_idx, standardizer)
	test_set = DDIGNNDataset(test_df, args.max_nodes, task_type, None, graph_cache, class_to_idx, standardizer)

	# LOSS & WEIGHTS
	y_idx = train_df["Y"].map(class_to_idx).values
	class_counts = np.bincount(y_idx, minlength=num_outputs).astype(np.float32)
	weights_np = np.sqrt(1.0 / np.clip(class_counts / class_counts.sum(), 1e-6, None))
	weights_np /= weights_np.sum()
	weights = torch.tensor(weights_np, dtype=torch.float32, device=device)

	if args.loss_type == "focal":
		criterion = FocalLoss(weight=weights, gamma=2.0)
	elif args.loss_type == "weighted_ce":
		criterion = nn.CrossEntropyLoss(weight=weights)
	else:
		criterion = nn.CrossEntropyLoss()

	loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "collate_fn": collate_fn}
	train_loader = DataLoader(train_set, sampler=WeightedRandomSampler(
		[weights_np[y] for y in y_idx], len(train_set)), **loader_kwargs)
	valid_loader = DataLoader(valid_set, shuffle=False, **loader_kwargs)
	test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

	model = DDIGNNModel(
		NODE_FEAT_DIM, args.hidden_dim, num_outputs, 
		descriptor_dim=8 if args.use_descriptors else 0,
		gnn_type=args.gnn_type
	).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
	early_stopping = EarlyStopping(patience=args.patience, mode="max")

	start_epoch = 1
	if args.resume and os.path.exists(args.checkpoint_path):
		start_epoch, _, _, _ = load_checkpoint(args.checkpoint_path, model, optimizer, scheduler, device)
		print(f"Resumed from epoch {start_epoch}")

	print("\n[TRAIN] Beginning training loop...")
	for epoch in range(start_epoch, args.epochs + 1):
		t_loss = train_epoch(model, train_loader, optimizer, criterion, device, task_type, args.grad_clip)
		val_res = validate(model, valid_loader, criterion, device, task_type, num_outputs)
		
		print(f"Epoch {epoch:02d} | T-Loss: {t_loss:.4f} | V-Loss: {val_res['loss']:.4f} | V-F1: {val_res['f1']:.4f} | V-MCC: {val_res['mcc']:.4f}")
		
		scheduler.step(val_res["f1"])
		early_stopping(val_res["f1"])
		
		if early_stopping.best_score == val_res["f1"]:
			torch.save(model.state_dict(), args.out)

		save_checkpoint(args.checkpoint_path, epoch, model, optimizer, scheduler, 
						early_stopping.best_score, epoch, early_stopping.counter, 
						task_type, class_to_idx, args.max_nodes, args.hidden_dim, num_outputs)
		
		if early_stopping.early_stop:
			print(f"Early stopping at epoch {epoch}")
			break

	# FINAL EVALUATION
	model.load_state_dict(torch.load(args.out))
	test_res = validate(model, test_loader, criterion, device, task_type, num_outputs)
	y_true, y_pred, y_score = collect_predictions(model, test_loader, device, task_type)
	
	print("\n" + "="*40)
	print("FINAL PERFORMANCE REPORT (TEST SET)")
	print("="*40)
	print(f"Accuracy: {test_res['bal_acc']:.4f} (Balanced)")
	print(f"Macro F1: {test_res['f1']:.4f}")
	print(f"MCC:      {test_res['mcc']:.4f}")
	print(f"ROC-AUC:  {test_res['auc']:.4f}")
	
	report = classification_report(y_true, y_pred, target_names=SIDE_EFFECT_LABELS, zero_division=0)
	print("\nPer-Class Detailed Report:")
	print(report)

	# EXPORT INTEGRATED PACKAGE
	print("\n[EXPORT] Saving integrated inference package...")
	joblib.dump({
		"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
		"class_to_idx": class_to_idx,
		"idx_to_class": {v: k for k, v in class_to_idx.items()},
		"descriptor_standardizer": standardizer,
		"config": {
			"node_dim": NODE_FEAT_DIM,
			"hidden_dim": args.hidden_dim,
			"num_outputs": num_outputs,
			"max_nodes": args.max_nodes,
			"use_descriptors": args.use_descriptors,
			"gnn_type": args.gnn_type
		}
	}, "models/ddi_advanced_gnn_integrated.joblib")

if __name__ == "__main__":
	main()
