import subprocess
import pandas as pd
import os
import json

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def extract_metrics(output):
    metrics = {}
    lines = output.split('\n')
    in_report = False
    for line in lines:
        if "FINAL PERFORMANCE REPORT" in line:
            in_report = True
        if not in_report:
            continue
            
        if "Accuracy:" in line:
            metrics['Accuracy'] = float(line.split(':')[1].split('(')[0].strip())
        if "Macro F1:" in line:
            metrics['Macro F1'] = float(line.split(':')[1].strip())
        if "MCC:" in line:
            metrics['MCC'] = float(line.split(':')[1].strip())
    return metrics

def main():
    experiments = {
        "A: Base GNN": [
            "python", "scripts/train_model.py", 
            "--data-path", "Project/DSA (1)/DSA/data/drugbank.tab",
            "--epochs", "2", "--batch-size", "256", 
            "--loss-type", "ce", "--no-use-descriptors",
            "--out", "models/exp_a.pt"
        ],
        "B: GNN + Weighted Loss": [
            "python", "scripts/train_model.py", 
            "--data-path", "Project/DSA (1)/DSA/data/drugbank.tab",
            "--epochs", "2", "--batch-size", "256", 
            "--loss-type", "weighted_ce", "--no-use-descriptors",
            "--out", "models/exp_b.pt"
        ],
        "C: GNN + Descriptors": [
            "python", "scripts/train_model.py", 
            "--data-path", "Project/DSA (1)/DSA/data/drugbank.tab",
            "--epochs", "2", "--batch-size", "256", 
            "--loss-type", "weighted_ce", "--use-descriptors",
            "--out", "models/exp_c.pt"
        ],
        "D: GNN + Descriptors + Focal": [
            "python", "scripts/train_model.py", 
            "--data-path", "Project/DSA (1)/DSA/data/drugbank.tab",
            "--epochs", "2", "--batch-size", "256", 
            "--loss-type", "focal", "--use-descriptors",
            "--out", "models/exp_d.pt"
        ]
    }

    results = []
    for name, cmd in experiments.items():
        print(f"\n=== Running Experiment {name} ===")
        output = run_cmd(cmd)
        metrics = extract_metrics(output)
        metrics['Experiment'] = name
        results.append(metrics)

    df = pd.DataFrame(results)
    print("\n=== Experiment Comparison Table ===")
    print(df.to_string(index=False))
    
    df.to_csv("reports/experiment_comparison.csv", index=False)
    print("\nSaved comparison to reports/experiment_comparison.csv")

if __name__ == "__main__":
    main()
