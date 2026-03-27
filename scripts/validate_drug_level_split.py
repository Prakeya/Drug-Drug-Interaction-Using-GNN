#!/usr/bin/env python3
"""
Test script: Drug-Level Split Validation
- Loads interactions from drugbank.tab
- Extracts unique drug pairs
- Applies drug-level split (no overlap)
- Verifies overlap count is exactly 0
- Asserts constraint compliance
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def enforce_drug_level_split(pairs: list, train_ratio: float = 0.8) -> tuple:
    """
    REQUIREMENT #2: Split by UNIQUE drugs, NOT pairs.
    
    CRITICAL IMPLEMENTATION:
    1. Extract unique drugs from all pairs
    2. Split unique drugs into train/test (no overlap)
    3. Build train/test pairs from split drug lists
    4. Verify ZERO overlap (failure condition if violated)
    
    Returns: (train_pairs, test_pairs) with ZERO drug overlap
    Raises: AssertionError if any overlap detected (REQUIREMENT #10)
    """
    if not pairs:
        return [], []
    
    # Step 1: Extract unique drugs
    all_drugs = {}
    for smiles_a, smiles_b in pairs:
        if smiles_a not in all_drugs:
            all_drugs[smiles_a] = []
        if smiles_b not in all_drugs:
            all_drugs[smiles_b] = []
    
    # Step 2: Split unique drugs
    unique_drugs = list(all_drugs.keys())
    np.random.shuffle(unique_drugs)
    split_idx = int(len(unique_drugs) * train_ratio)
    
    train_drugs = set(unique_drugs[:split_idx])
    test_drugs = set(unique_drugs[split_idx:])
    
    # CRITICAL: Verify zero overlap (REQUIREMENT #10 - FAILURE CONDITION)
    overlap = train_drugs & test_drugs
    if overlap:
        raise AssertionError(
            f"\n{'='*70}\n"
            f"FAILURE CONDITION VIOLATED (REQUIREMENT #10)\n"
            f"{'='*70}\n"
            f"Drug-level split FAILED!\n\n"
            f"❌ {len(overlap)} drugs appear in BOTH train AND test sets!\n\n"
            f"This is CRITICAL DATA LEAKAGE that causes overfitting.\n"
            f"Model will memorize these drugs instead of learning structure.\n\n"
            f"Example overlaps: {list(overlap)[:5]}\n"
            f"{'='*70}\n"
        )
    
    # Step 3: Build pairs from split drugs
    train_pairs = []
    test_pairs = []
    
    for smiles_a, smiles_b in pairs:
        a_in_train = smiles_a in train_drugs
        b_in_train = smiles_b in train_drugs
        
        if a_in_train and b_in_train:
            train_pairs.append((smiles_a, smiles_b))
        elif not a_in_train and not b_in_train:
            test_pairs.append((smiles_a, smiles_b))
        # Mixed pairs (one drug in train, one in test) are discarded
    
    # Print explicit report
    print("\n" + "="*70)
    print("DRUG-LEVEL SPLIT VERIFICATION (REQUIREMENT #2)")
    print("="*70)
    print(f"Total unique drugs: {len(unique_drugs)}")
    print(f"Train drugs: {len(train_drugs)} (unique)")
    print(f"Test drugs: {len(test_drugs)} (unique)")
    print(f"Overlap: {len(overlap)} (MUST BE ZERO) ✓" if not overlap else f"Overlap: {len(overlap)} ✗ FAILED")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    print("="*70 + "\n")
    
    return train_pairs, test_pairs


def validate_drug_level_split(train_pairs: list, test_pairs: list, split_name: str = "Dataset"):
    """
    CONSTRAINT #2: Verify that train and test have ZERO drug overlap.
    
    Args:
        train_pairs: List of (smiles_a, smiles_b) tuples
        test_pairs: List of (smiles_a, smiles_b) tuples
        split_name: Name of split for reporting
    
    Returns:
        dict with validation results and overlap info
    """
    report = {
        "split_name": split_name,
        "is_valid": False,
        "train_unique_drugs": 0,
        "test_unique_drugs": 0,
        "overlap_count": 0,
        "overlap_drugs": [],
        "error": None,
    }
    
    try:
        # Extract unique drugs from each set
        train_drugs = set()
        for smiles_a, smiles_b in train_pairs:
            train_drugs.add(smiles_a)
            train_drugs.add(smiles_b)
        
        test_drugs = set()
        for smiles_a, smiles_b in test_pairs:
            test_drugs.add(smiles_a)
            test_drugs.add(smiles_b)
        
        # Check overlap
        overlap = train_drugs & test_drugs
        
        report["train_unique_drugs"] = len(train_drugs)
        report["test_unique_drugs"] = len(test_drugs)
        report["overlap_count"] = len(overlap)
        report["overlap_drugs"] = list(overlap)[:10]  # First 10 overlapping drugs
        report["is_valid"] = len(overlap) == 0
        
    except Exception as e:
        report["error"] = str(e)
    
    return report


def load_drugbank_pairs():
    """Load all drug pairs from drugbank.tab"""
    pairs = []
    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / "data" / "drugbank.tab"
    
    if not db_path.exists():
        print(f"❌ ERROR: Data file not found: {db_path}")
        return pairs
    
    try:
        df = pd.read_csv(db_path, sep="\t")
        print(f"Loaded DrugBank file with {len(df)} rows\n")
        
        # Vectorized approach: much faster than iterrows()
        valid_df = df[df["Y"] == 1].copy()
        valid_df["X1"] = valid_df["X1"].astype(str).str.strip('"')
        valid_df["X2"] = valid_df["X2"].astype(str).str.strip('"')
        
        # Filter out empty SMILES
        valid_df = valid_df[(valid_df["X1"] != "") & (valid_df["X2"] != "")]
        
        # Convert to pairs list
        pairs = list(zip(valid_df["X1"], valid_df["X2"]))
    except Exception as e:
        print(f"❌ ERROR loading drugbank.tab: {e}")
    
    return pairs


def test_drug_level_split():
    """Test drug-level split with explicit overlap verification"""
    
    print("="*80)
    print("DRUG-LEVEL SPLIT VALIDATION TEST")
    print("="*80)
    print()
    
    # Load pairs
    print("📂 Loading training data...")
    pairs = load_drugbank_pairs()
    
    if not pairs:
        print("❌ ERROR: No valid pairs loaded!")
        sys.exit(1)
    
    print(f"✓ Loaded {len(pairs)} interaction pairs")
    
    # Extract unique drugs from full dataset
    all_drugs = set()
    for smiles_a, smiles_b in pairs:
        all_drugs.add(smiles_a)
        all_drugs.add(smiles_b)
    print(f"✓ Total unique drugs: {len(all_drugs)}")
    print()
    
    # Apply enforce_drug_level_split
    print("🔄 Applying enforce_drug_level_split()...")
    try:
        train_pairs, test_pairs = enforce_drug_level_split(pairs, train_ratio=0.8)
    except AssertionError as e:
        print(f"\n❌ ASSERTION FAILED:\n{e}")
        sys.exit(1)
    
    print()
    
    # Validate with validate_drug_level_split
    print("✓ Validating split with validate_drug_level_split()...")
    validation_report = validate_drug_level_split(train_pairs, test_pairs, "DrugBank")
    
    print()
    print("="*80)
    print("VALIDATION REPORT")
    print("="*80)
    print(f"Train unique drugs: {validation_report['train_unique_drugs']}")
    print(f"Test unique drugs: {validation_report['test_unique_drugs']}")
    print(f"Overlap count: {validation_report['overlap_count']}")
    
    # EXPLICIT OVERLAP COUNT AND ASSERTION
    print()
    print("="*80)
    print("OVERLAP VERIFICATION (REQUIREMENT #2)")
    print("="*80)
    overlap_count = validation_report['overlap_count']
    print(f"Overlap count: {overlap_count}")
    
    if overlap_count > 0:
        print(f"\n❌ FAILURE: {overlap_count} drugs appear in BOTH train AND test!")
        print(f"Example overlaps: {validation_report['overlap_drugs']}")
        print("\n⚠️  DATA LEAKAGE DETECTED - Model will overfit on these drugs!")
        sys.exit(1)
    else:
        print("✓ Overlap count is 0")
    
    # ASSERTION (REQUIREMENT #10)
    assert overlap_count == 0, f"Drug-level split FAILED: {overlap_count} drugs overlap!"
    print("✓ Assertion passed: overlap_count == 0")
    
    print()
    print("="*80)
    print("✓ ALL CHECKS PASSED - Pure drug-level split verified!")
    print("="*80)
    print()
    
    # Summary
    print("SUMMARY:")
    print(f"  • Train pairs: {len(train_pairs)}")
    print(f"  • Test pairs: {len(test_pairs)}")
    print(f"  • Train drugs: {validation_report['train_unique_drugs']}")
    print(f"  • Test drugs: {validation_report['test_unique_drugs']}")
    print(f"  • Drug overlap: {validation_report['overlap_count']} (MUST BE 0) ✓")
    print(f"  • Valid: {validation_report['is_valid']} (MUST BE True) ✓")
    print()


if __name__ == "__main__":
    test_drug_level_split()
