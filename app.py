import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import joblib
import os
import base64
import json
from datetime import datetime
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS, Descriptors, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib.pyplot as plt

# ==========================================
# 🛡️ SAFE TORCH IMPORT (WINDOWS HARDENING)
# ==========================================
try:
    import torch
    from torch_geometric.data import Data
    HAS_PYG = True
    TORCH_AVAILABLE = True
except Exception:
    HAS_PYG = False
    TORCH_AVAILABLE = False

# ==========================================
# 🧬 RESEARCH-GRADE BIOMEDICAL DASHBOARD UI (CSS)
# ==========================================
st.set_page_config(
    page_title="DrugLens Research | Interaction Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize Session State
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# COLOR PALETTE
# Background: #FFE6E6
# Soft card tone: #E1AFD1
# Muted accent: #AD88C6
# Primary accent: #7469B6

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* ── Global Theme ── */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: #FFE6E6 !important;
        color: #2D3436 !important;
    }}
    
    .stApp {{ background: #FFE6E6; }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ── Hide Streamlit Elements ── */
    header, footer, #MainMenu {{ visibility: hidden !important; height: 0 !important; }}

    /* ── Typography ── */
    .main-title {{
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        color: #7469B6;
        margin-bottom: 0.1rem;
        letter-spacing: -0.06em;
    }}
    .sub-title {{
        font-size: 1.1rem;
        font-weight: 500;
        text-align: center;
        color: #AD88C6;
        margin-bottom: 3rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}

    /* ── Result Container Control ── */
    .result-container {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 0 1rem;
    }}

    /* ── Cards ── */
    .research-card {{
        background: #ffffff;
        border-radius: 28px;
        padding: 2.2rem;
        box-shadow: 0 12px 35px rgba(116, 105, 182, 0.07);
        border: 1px solid rgba(173, 136, 198, 0.08);
        margin-bottom: 1.5rem;
        animation: fadeIn 0.5s ease-out;
    }}
    .accent-card {{
        background: #7469B6;
        color: white;
        border-radius: 26px;
        padding: 2.5rem;
        box-shadow: 0 15px 40px rgba(116, 105, 182, 0.3);
    }}

    /* ── Input Styling ── */
    .stTextInput > div > div > input {{
        background-color: #2D3436 !important;
        color: #ffffff !important;
        border-radius: 16px !important;
        border: 2px solid #7469B6 !important;
        padding: 14px 22px !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-align: center;
        font-size: 1.05rem !important;
    }}
    .stTextInput [data-testid="stWidgetLabel"] p {{
        color: #7469B6 !important;
        font-weight: 800 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        margin-bottom: 10px !important;
        letter-spacing: 0.04em;
    }}

    /* ── Optimized Button ── */
    .stButton > button {{
        background: linear-gradient(135deg, #7469B6 0%, #AD88C6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 60px !important;
        padding: 0.8rem 3.5rem !important;
        font-weight: 800 !important;
        font-size: 1.15rem !important;
        box-shadow: 0 8px 25px rgba(116, 105, 182, 0.25) !important;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(116, 105, 182, 0.35) !important;
    }}
    .back-btn button {{
        background: transparent !important;
        color: #7469B6 !important;
        border: 2px solid #7469B6 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 2rem !important;
    }}

    /* ── Components ── */
    .status-chip {{
        display: inline-flex;
        align-items: center;
        background: #FDF0F0;
        color: #7469B6;
        padding: 0.5rem 1.2rem;
        border-radius: 60px;
        font-size: 0.85rem;
        font-weight: 700;
        margin: 5px;
        border: 1px solid rgba(116, 105, 182, 0.15);
    }}
    .mol-frame {{
        background: #FAFAFA;
        border-radius: 20px;
        padding: 15px;
        display: flex;
        justify-content: center;
        align-items: center;
        border: 1px solid rgba(0,0,0,0.02);
    }}
    .pred-label {{
        font-size: 2.5rem;
        font-weight: 900;
        color: #ffffff;
        margin: 0.5rem 0;
    }}
    .conf-badge {{
        background: rgba(255,255,255,0.2);
        padding: 0.4rem 1rem;
        border-radius: 12px;
        font-size: 0.95rem;
        font-weight: 700;
    }}
    .descriptor-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 800;
        color: #7469B6;
    }}
    .descriptor-label {{
        font-size: 0.75rem;
        font-weight: 700;
        color: #AD88C6;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}
    .bar-container {{
        width: 100%;
        background-color: #FDF0F0;
        border-radius: 10px;
        margin: 5px 0 15px 0;
        height: 9px;
    }}
    .bar-fill {{
        height: 100%;
        border-radius: 10px;
        background: #7469B6;
    }}
    .signal-pill {{
        background: rgba(173, 136, 198, 0.12);
        color: #7469B6;
        padding: 0.3rem 0.7rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 700;
        margin: 2px;
        display: inline-block;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 BIOMEDICAL RESEARCH ENGINE
# ==========================================

def get_atom_features(atom):
    hybridization_map = {
        Chem.rdchem.HybridizationType.S: 1,
        Chem.rdchem.HybridizationType.SP: 2,
        Chem.rdchem.HybridizationType.SP2: 3,
        Chem.rdchem.HybridizationType.SP3: 4,
        Chem.rdchem.HybridizationType.SP3D: 5,
        Chem.rdchem.HybridizationType.SP3D2: 6,
    }
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        float(hybridization_map.get(atom.GetHybridization(), 0)),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        atom.GetExplicitValence() + atom.GetImplicitValence()
    ]

def get_bond_features(bond):
    bt = bond.GetBondType()
    order = 0.0
    if bt == Chem.rdchem.BondType.SINGLE: order = 1.0
    elif bt == Chem.rdchem.BondType.DOUBLE: order = 2.0
    elif bt == Chem.rdchem.BondType.TRIPLE: order = 3.0
    elif bt == Chem.rdchem.BondType.AROMATIC: order = 1.5
    
    return [
        order,
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ]

def mol_to_graph_data(mol):
    if not mol: return None
    try:
        node_feats = [get_atom_features(a) for a in mol.GetAtoms()]
        x = np.array(node_feats, dtype=np.float32)
        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = get_bond_features(bond)
            edge_index.extend([[u, v], [v, u]])
            edge_attr.extend([bf, bf])
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_attr = np.array(edge_attr, dtype=np.float32)
        if HAS_PYG and TORCH_AVAILABLE:
            return Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index), edge_attr=torch.from_numpy(edge_attr))
        return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}
    except: return None

def compute_graph_metrics(mol):
    if not mol: return {}
    counts = {"SINGLE": 0, "DOUBLE": 0, "TRIPLE": 0, "AROMATIC": 0}
    for b in mol.GetBonds():
        bt = b.GetBondType()
        if bt == Chem.rdchem.BondType.SINGLE: counts["SINGLE"] += 1
        elif bt == Chem.rdchem.BondType.DOUBLE: counts["DOUBLE"] += 1
        elif bt == Chem.rdchem.BondType.TRIPLE: counts["TRIPLE"] += 1
        elif bt == Chem.rdchem.BondType.AROMATIC: counts["AROMATIC"] += 1
    return {
        "nodes": mol.GetNumAtoms(),
        "edges": mol.GetNumBonds(),
        "rings": mol.GetRingInfo().NumRings(),
        "avg_degree": np.mean([a.GetDegree() for a in mol.GetAtoms()]) if mol.GetNumAtoms() > 0 else 0,
        "bond_counts": counts
    }

def get_tanimoto_similarity(mol_a, mol_b):
    try:
        if mol_a is None or mol_b is None: return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
        return round(DataStructs.TanimotoSimilarity(fp1, fp2), 3)
    except: return 0.0

@st.cache_resource
def load_research_model():
    path = "models/main_multiclass.joblib"
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def get_safe_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        try: AllChem.ComputeGasteigerCharges(mol)
        except: pass
        chg = [a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms() if a.HasProp("_GasteigerCharge")]
        return {
            "MW": Descriptors.MolWt(mol), "LogP": Descriptors.MolLogP(mol), "TPSA": Descriptors.TPSA(mol),
            "HBD": Descriptors.NumHDonors(mol), "HBA": Descriptors.NumHAcceptors(mol), "RotB": Descriptors.NumRotatableBonds(mol),
            "HA": Descriptors.HeavyAtomCount(mol), "Aro": Descriptors.NumAromaticRings(mol),
            "Charge": f"{min(chg):.2f}/{max(chg):.2f}" if chg else "0.00"
        }
    except: return None

@st.cache_data
def get_pharmacological_signals(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return []
    sigs = []
    pats = {
        "CYP Inhibitor": ["n1ccnc1", "c1ccc(cc1)C[N+]"], "CYP Inducer": ["c1ccc2c(c1)ccc3c2ccc4c3cccc4"],
        "CYP Substrate": ["C(=O)O", "C(=O)N"], "Prodrug": ["C(=O)OC", "C(=O)NC"], "QT Risk": ["N1CCN(CC1)CC", "C1CCN(CC1)CC"],
    }
    for n, s in pats.items():
        for sm in s:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(sm)): sigs.append(n); break
    lp, mw = Descriptors.MolLogP(mol), Descriptors.MolWt(mol)
    if lp > 3.0 and mw < 450: sigs.append("CNS Depressant")
    if mw < 200 or mw > 800: sigs.append("Narrow TI")
    return list(set(sigs))

def draw_mol_clean(smiles, highlights=None):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    d = Draw.MolDraw2DCairo(500, 450)
    o = d.drawOptions()
    o.backgroundColour = (1,1,1,0); o.bondLineWidth = 3; o.padding = 0.1
    o.highlightRadius = 0.4; o.setHighlightColour((0.4, 0.8, 0.4, 0.45))
    Draw.PrepareMolForDrawing(mol)
    d.DrawMolecule(mol, highlightAtoms=list(highlights) if highlights else [])
    d.FinishDrawing()
    return base64.b64encode(d.GetDrawingText()).decode()

def render_topo_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None, {}
    G = nx.Graph()
    for atom in mol.GetAtoms(): G.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
    edge_list, edge_widths, edge_styles = [], [], []
    for bond in mol.GetBonds():
        edge_list.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        bt = bond.GetBondType()
        if bt == Chem.rdchem.BondType.SINGLE: edge_widths.append(1.5); edge_styles.append('solid')
        elif bt == Chem.rdchem.BondType.DOUBLE: edge_widths.append(3.5); edge_styles.append('solid')
        elif bt == Chem.rdchem.BondType.TRIPLE: edge_widths.append(5.5); edge_styles.append('solid')
        elif bt == Chem.rdchem.BondType.AROMATIC: edge_widths.append(2.5); edge_styles.append('dashed')
        else: edge_widths.append(1.0); edge_styles.append('solid')
    fig, ax = plt.subplots(figsize=(5, 5)); fig.patch.set_facecolor('none')
    pos = nx.kamada_kawai_layout(nx.Graph(edge_list)) if edge_list else {}
    nx.draw_networkx_nodes(nx.Graph(edge_list), pos, ax=ax, node_color='#7469B6', alpha=0.9, node_size=500)
    labels = {atom.GetIdx(): atom.GetSymbol() for atom in mol.GetAtoms()}
    nx.draw_networkx_labels(nx.Graph(edge_list), pos, labels=labels, ax=ax, font_size=9, font_color='white', font_weight='bold')
    for i, (u, v) in enumerate(edge_list):
        nx.draw_networkx_edges(nx.Graph([(u, v)]), pos, ax=ax, width=edge_widths[i], style=edge_styles[i], edge_color='#AD88C6', alpha=0.6)
    buf = BytesIO(); plt.savefig(buf, format="png", transparent=True, bbox_inches='tight', dpi=140); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode(), compute_graph_metrics(mol)

# ==========================================
# 🗺️ NAVIGATION & FLOW CONTROL
# ==========================================

def switch_to_results(): st.session_state.analysis_mode = True
def switch_to_input(): st.session_state.analysis_mode = False

# ==========================================
# 🏠 SCREEN 1: MOLECULAR INPUT
# ==========================================

if not st.session_state.analysis_mode:
    st.markdown("<div class='main-title'>Drug Interaction Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>A biomedical research dashboard for molecular interaction analysis</div>", unsafe_allow_html=True)

    with st.container():
        col_inp1, col_inp2 = st.columns(2)
        with col_inp1:
            smi_a = st.text_input("Compound A SMILES", placeholder="CC(=O)OC1=CC=CC=C1C(=O)O", key="smi_a")
            if smi_a:
                img_a = draw_mol_clean(smi_a)
                if img_a: st.markdown(f"<div class='research-card'><div class='mol-frame'><img src='data:image/png;base64,{img_a}' width='100%'></div></div>", unsafe_allow_html=True)
                else: st.error("SMILES Architecture Invalid")
            else: st.markdown("<div class='research-card' style='height:300px; display:flex; align-items:center; justify-content:center; border:2px dashed #AD88C6; background:#FDF0F0; color:#AD88C6; font-weight:700;'>Compound A Slot</div>", unsafe_allow_html=True)
        with col_inp2:
            smi_b = st.text_input("Compound B SMILES", placeholder="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", key="smi_b")
            if smi_b:
                img_b = draw_mol_clean(smi_b)
                if img_b: st.markdown(f"<div class='research-card'><div class='mol-frame'><img src='data:image/png;base64,{img_b}' width='100%'></div></div>", unsafe_allow_html=True)
                else: st.error("SMILES Architecture Invalid")
            else: st.markdown("<div class='research-card' style='height:300px; display:flex; align-items:center; justify-content:center; border:2px dashed #AD88C6; background:#FDF0F0; color:#AD88C6; font-weight:700;'>Compound B Slot</div>", unsafe_allow_html=True)

    _, center_col, _ = st.columns([1, 1.5, 1])
    with center_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("INITIATE INTERACTION ANALYTICS", use_container_width=True):
            if smi_a and smi_b and Chem.MolFromSmiles(smi_a) and Chem.MolFromSmiles(smi_b):
                st.session_state.analysis_results = {"smi_a": smi_a, "smi_b": smi_b}
                switch_to_results(); st.rerun()
            else: st.error("Dual molecules required.")

# ==========================================
# 📊 SCREEN 2: RESEARCH ANALYSIS
# ==========================================

else:
    res, sa, sb = st.session_state.analysis_results, st.session_state.analysis_results['smi_a'], st.session_state.analysis_results['smi_b']
    m1, m2 = Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)
    model_data = load_research_model()
    
    with st.spinner("Decoding Molecular Patterns..."):
        def get_v(s):
            m = Chem.MolFromSmiles(s)
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
            f = np.zeros((2048,)); DataStructs.ConvertToNumpyArray(fp, f)
            d = [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m), Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m), Descriptors.NumRotatableBonds(m), Descriptors.RingCount(m), Descriptors.NumAromaticRings(m), Descriptors.HeavyAtomCount(m), Descriptors.FractionCSP3(m), 0.0, 0.0]
            return f, np.array(d, dtype=np.float32)

        f1, d1 = get_v(sa); f2, d2 = get_v(sb)
        feats = np.concatenate([f1, f2, d1, d2]).reshape(1, -1)
        probs = model_data['model'].predict_proba(feats)[0] if model_data else [0.0]
        classes = model_data.get('class_names', ["Unknown"]) if model_data else ["Error"]
        sorted_preds = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        top_pred, mcs, sim_score = sorted_preds[0], rdFMCS.FindMCS([m1, m2], timeout=3), get_tanimoto_similarity(m1, m2)
        d_a, d_b = get_safe_descriptors(sa), get_safe_descriptors(sb)
        sig_a, sig_b = get_pharmacological_signals(sa), get_pharmacological_signals(sb)

    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
    head_col1, head_col2 = st.columns([4, 1])
    with head_col1:
        st.markdown(f"<div style='color:#7469B6; font-size:2.5rem; font-weight:800;'>Interaction Profile Analysis</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#AD88C6; font-weight:600; margin-bottom:2rem;'>REPORT ID: {datetime.now().strftime('%Y%m%d')}-{abs(hash(sa))%1000}</div>", unsafe_allow_html=True)
    with head_col2:
        st.markdown("<div class='back-btn'>", unsafe_allow_html=True)
        if st.button("← NEW PROTOCOL", on_click=switch_to_input): pass
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='accent-card' style='text-align:center; margin-bottom:3rem;'>
        <div style='font-size:0.9rem; font-weight:700; text-transform:uppercase; opacity:0.8;'>Target Prediction</div>
        <div class='pred-label'>{top_pred[0]}</div>
        <div class='conf-badge' style='display:inline-block;'>Confidence: {top_pred[1]*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='research-card'>
        <div style='color:#7469B6; font-weight:800; font-size:1.2rem; margin-bottom:1.5rem;'>STRUCTURAL EVIDENCE & GNN METRICS</div>
        <div style='display:grid; grid-template-columns: 1fr 1fr 1fr; gap:20px; text-align:center;'>
            <div style='border-right:1px solid #FDF0F0;'>
                <div class='descriptor-label'>Similarity</div>
                <div class='descriptor-val' style='font-size:1.8rem;'>{sim_score:.3f}</div>
            </div>
            <div style='border-right:1px solid #FDF0F0;'>
                <div class='descriptor-label'>MCS Atoms</div>
                <div class='descriptor-val' style='font-size:1.8rem;'>{mcs.numAtoms}</div>
            </div>
            <div>
                <div class='descriptor-label'>Backend State</div>
                <div class='descriptor-val' style='font-size:1rem; margin-top:0.5rem;'>{ "Ready (Torch)" if TORCH_AVAILABLE else "Structural Mode" }</div>
            </div>
        </div>
        {f'<div class="status-chip" style="background:#FFF3CD; color:#856404; border:1px solid #FFEEBA; width:100%; justify-content:center; margin-top:1.5rem;">⚠️ Torch backend unavailable, running in structural fallback mode.</div>' if not TORCH_AVAILABLE else ''}
    </div>
    """, unsafe_allow_html=True)

    m_col1, m_col2 = st.columns(2)
    def render_p(mol, smi, title, signals):
        h = mol.GetSubstructMatch(Chem.MolFromSmarts(mcs.smartsString)) if mcs.numAtoms else None
        img = draw_mol_clean(smi, h)
        sigs_html = ' '.join([f'<span class="signal-pill">{s}</span>' for s in signals])
        st.markdown(f"<div class='research-card'><div class='descriptor-label' style='margin-bottom:1rem;'>{title} Profile</div><div class='mol-frame'><img src='data:image/png;base64,{img}' width='100%'></div><div style='margin-top:1rem;'>{sigs_html}</div></div>", unsafe_allow_html=True)
    with m_col1: render_p(m1, sa, "Compound A", sig_a)
    with m_col2: render_p(m2, sb, "Compound B", sig_b)

    st.markdown(f"""
    <div class='research-card'>
        <div style='color:#7469B6; font-weight:800; font-size:1.2rem; margin-bottom:1.5rem;'>INTERPRETATION & RATIONALE</div>
        <div style='display:grid; grid-template-columns: 1fr 1fr; gap:30px;'>
            <div style='border-left:4px solid #7469B6; padding-left:15px;'>
                <div style='font-weight:700; color:#2D3436; margin-bottom:5px;'>Structural Evidence</div>
                <ul style="font-size:0.9rem; color:#4E5D6C;"><li>Conserved core of <strong>{mcs.numAtoms} atoms</strong>.</li><li>Tanimoto similarity: <strong>{sim_score:.2f}</strong>.</li></ul>
            </div>
            <div style='border-left:4px solid #AD88C6; padding-left:15px;'>
                <div style='font-weight:700; color:#2D3436; margin-bottom:5px;'>Metabolic Proxy</div>
                <div style='font-size:0.9rem; color:#4E5D6C;'>Similarity of {sim_score*100:.1f}% indicates structural kinship.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color:#7469B6; font-weight:800; font-size:1.2rem; margin:2rem 0 1.5rem 0;'>DESCRIPTOR COMPARISON</div>", unsafe_allow_html=True)
    d_col1, d_col2 = st.columns(2)
    def render_d(data, title):
        if not data: st.markdown("<div class='research-card'>Data Unavailable</div>", unsafe_allow_html=True); return
        bars = ""
        for k, v, mx, lbl in [("MW", data['MW'], 800, "Molecular Weight"), ("LogP", data['LogP'], 8, "Lipophilicity"), ("TPSA", data['TPSA'], 200, "TPSA")]:
            p = min(100, (v/mx)*100 if mx > 0 else 0)
            bars += f"<div class='descriptor-label'>{lbl}</div><div class='bar-container'><div class='bar-fill' style='width:{p}%;'></div></div>"
        st.markdown(f"<div class='research-card'><div style='font-weight:800; color:#7469B6; margin-bottom:1.5rem;'>{title}</div>{bars}</div>", unsafe_allow_html=True)
    with d_col1: render_d(d_a, "Compound A")
    with d_col2: render_d(d_b, "Compound B")

    st.markdown("<div style='color:#7469B6; font-weight:800; font-size:1.2rem; margin:2rem 0 1.5rem 0;'>TOPOLOGICAL GRAPH ANALYTICS</div>", unsafe_allow_html=True)
    g_col1, g_col2 = st.columns(2)
    def render_g(smi, title):
        img, met = render_topo_graph(smi)
        st.markdown(f"<div class='research-card'><div style='font-weight:800; color:#7469B6; margin-bottom:1rem;'>{title} Graph</div><div class='mol-frame' style='background:white;'><img src='data:image/png;base64,{img}' width='100%'></div><div style='display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:1.5rem;'><div><div class='descriptor-label'>Nodes</div><div class='descriptor-val'>{met['nodes']}</div></div><div><div class='descriptor-label'>Edges</div><div class='descriptor-val'>{met['edges']}</div></div></div></div>", unsafe_allow_html=True)
    with g_col1: render_g(sa, "Compound A")
    with g_col2: render_g(sb, "Compound B")

    st.markdown("<div style='display:flex; justify-content:center; gap:20px; margin: 3rem 0;'>", unsafe_allow_html=True)
    st.download_button("EXPORT JSON", data=json.dumps(top_pred[0]), file_name="DDI.json")
    st.download_button("EXPORT CSV", data=pd.DataFrame(sorted_preds).to_csv(), file_name="DDI.csv")
    st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown("<div style='text-align: center; color: #AD88C6; margin: 5rem 0 3rem 0; font-size: 0.9rem; font-weight: 700; opacity:0.6;'>DrugLens Pro | v5.1 Stable Research Platform</div>", unsafe_allow_html=True)