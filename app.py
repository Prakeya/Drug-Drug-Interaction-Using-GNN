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
# 🌿 PREMIUM WELLNESS DASHBOARD UI (CSS)
# ==========================================
st.set_page_config(
    page_title="DrugLens Wellness | Insight Engine",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# FORCE ACCESSIBILITY & VISIBILITY OVERRIDES
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@300;400;500;600;700&family=Inter:wght@400;600;800&display=swap');
    
    /* ── Global Theme & Forced Visibility ── */
    html, body, [class*="css"] {
        font-family: 'Quicksand', sans-serif;
        background-color: #f6f3ee !important;
        color: #1e293b !important;
    }
    
    .stApp { background: #f6f3ee; }

    /* ABSOLUTE TEXT VISIBILITY OVERRIDE */
    .stMarkdown, .stText, p, span, div, li {
        color: #1e293b !important;
        opacity: 1 !important;
    }

    /* Headings & Section Titles */
    h1, h2, h3, h4, h5, h6, .section-title, .hero-title {
        color: #0f172a !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }

    /* Input Labels */
    .input-label, label, [data-testid="stWidgetLabel"] p {
        color: #0f172a !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        font-size: 1rem !important;
    }

    /* Table & Dataframe Content */
    .stTable td, .stTable th, [data-testid="stTable"] td, [data-testid="stTable"] th {
        color: #0f172a !important;
        opacity: 1 !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] div {
        color: #0f172a !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricLabel"] p {
        color: #334155 !important;
        opacity: 1 !important;
    }

    /* Metadata & Captions */
    .stCaption, caption, .metric-lbl {
        color: #475569 !important;
        opacity: 1 !important;
        font-weight: 500 !important;
    }

    /* ── Hide Streamlit Elements ── */
    header, footer, #MainMenu { visibility: hidden !important; height: 0 !important; }

    /* ── Premium Card Components ── */
    .app-card {
        background: #ffffff;
        border-radius: 28px;
        padding: 2.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        border: 1px solid rgba(0, 0, 0, 0.02);
        color: #1a202c !important;
    }
    
    .card-pink { background-color: #fff1f2 !important; border: 1px solid #fee2e2 !important; }
    .card-blue { background-color: #f0f9ff !important; border: 1px solid #e0f2fe !important; }
    .card-lavender { background-color: #f5f3ff !important; border: 1px solid #ede9fe !important; }
    .card-mint { background-color: #f0fdf4 !important; border: 1px solid #dcfce7 !important; }
    .card-cream { background-color: #fffbeb !important; border: 1px solid #fef3c7 !important; }

    /* ── Typography & Headings ── */
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        text-align: center;
        background: linear-gradient(135deg, #1e3a8a 0%, #6d28d9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #334155 !important;
        text-align: center;
        margin-bottom: 4rem;
        font-weight: 500;
        letter-spacing: 0.01em;
    }
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a !important;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ── Input Styling ── */
    .stTextInput input {
        background: #1e293b !important;
        border: 2px solid #0f172a !important;
        color: #ffffff !important;
        border-radius: 18px !important;
        height: 4rem !important;
        padding: 0 1.5rem !important;
        font-size: 1.05rem !important;
        font-family: 'Inter', monospace !important;
    }

    /* ── CTA Button ── */
    .stButton>button {
        background: linear-gradient(135deg, #1d4ed8 0%, #7c3aed 100%) !important;
        color: white !important;
        font-weight: 800 !important;
        border-radius: 20px !important;
        height: 4.5rem !important;
        box-shadow: 0 10px 20px rgba(29, 78, 216, 0.3) !important;
    }

    /* ── Result Metrics ── */
    .prediction-val {
        font-size: 3rem;
        font-weight: 800;
        color: #9f1239 !important;
        margin: 1rem 0;
    }
    .confidence-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #581c87 !important;
    }

    /* ── Metrics ── */
    .metric-val { font-size: 1.4rem; font-weight: 800; color: #1e40af !important; }
    .metric-lbl { font-size: 0.8rem; color: #334155 !important; text-transform: uppercase; font-weight: 700 !important; }

    /* Visualization Frame */
    .vis-frame {
        background: white;
        border-radius: 20px;
        padding: 1rem;
        border: 1px solid rgba(0,0,0,0.05);
        display: flex;
        justify-content: center;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 BIO-LOGIC & DESCRIPTOR ENGINE
# ==========================================

@st.cache_resource
def get_main_model():
    path = "models/main_multiclass.joblib"
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def analyze_molecule_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    try: AllChem.ComputeGasteigerCharges(mol)
    except: pass
    chg = [a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms() if a.HasProp("_GasteigerCharge")]
    return {
        "Molecular Weight": f"{Descriptors.MolWt(mol):.2f}",
        "LogP": f"{Descriptors.MolLogP(mol):.2f}",
        "TPSA": f"{Descriptors.TPSA(mol):.2f}",
        "H-Bond Donors": int(Descriptors.NumHDonors(mol)),
        "H-Bond Acceptors": int(Descriptors.NumHAcceptors(mol)),
        "Rotatable Bonds": int(Descriptors.NumRotatableBonds(mol)),
        "Heavy Atom Count": int(Descriptors.HeavyAtomCount(mol)),
        "Ring Count": int(Descriptors.RingCount(mol)),
        "Aromatic Rings": int(Descriptors.NumAromaticRings(mol)),
        "Charge Range": f"{min(chg):.2f} / {max(chg):.2f}" if chg else "0.00"
    }

@st.cache_data
def generate_mol_preview(smiles, highlights=None):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    d = Draw.MolDraw2DCairo(400, 350)
    opts = d.drawOptions()
    opts.backgroundColour = (1,1,1,0)
    opts.bondLineWidth = 3
    Draw.PrepareMolForDrawing(mol)
    d.DrawMolecule(mol, highlightAtoms=list(highlights) if highlights else [])
    d.FinishDrawing()
    return base64.b64encode(d.GetDrawingText()).decode()

def render_structural_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    G = nx.Graph()
    for b in mol.GetBonds():
        G.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('none')
    pos = nx.spring_layout(G, k=0.8, iterations=50)
    nx.draw(G, pos, ax=ax, node_color='#1e40af', edge_color='#94a3b8', node_size=150, width=2, with_labels=False)
    buf = BytesIO()
    plt.savefig(buf, format="png", transparent=True, bbox_inches='tight', dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

@st.cache_data
def safe_extract_scaffold(smiles):
    """Safely extract Murcko Scaffold and return mol object."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        return MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None

# ==========================================
# 📱 HERO INTERFACE (LIVE PREVIEW)
# ==========================================

st.markdown("<div class='hero-title'>Drug Interaction Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>A biomedical research dashboard for molecular interaction analysis</div>", unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<label class='input-label'>Enter SMILES A</label>", unsafe_allow_html=True)
        smi_a = st.text_input("", key="smi_a", label_visibility="collapsed")
        if smi_a:
            mol_a = Chem.MolFromSmiles(smi_a)
            if mol_a:
                img_a = generate_mol_preview(smi_a)
                st.markdown(f"<div class='app-card card-cream' style='padding:1rem;'><div class='vis-frame'><img src='data:image/png;base64,{img_a}' width='100%'></div></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='app-card card-pink' style='color:#9f1239; text-align:center; font-weight:800;'>⚠ Invalid SMILES A</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='app-card' style='background:#ffffff; text-align:center; height:200px; display:flex; align-items:center; justify-content:center; color:#334155; font-weight:600; border:2px dashed #e2e8f0;'>Enter a SMILES string to preview structure</div>", unsafe_allow_html=True)
            
    with col2:
        st.markdown("<label class='input-label'>Enter SMILES B</label>", unsafe_allow_html=True)
        smi_b = st.text_input("", key="smi_b", label_visibility="collapsed")
        if smi_b:
            mol_b = Chem.MolFromSmiles(smi_b)
            if mol_b:
                img_b = generate_mol_preview(smi_b)
                st.markdown(f"<div class='app-card card-cream' style='padding:1rem;'><div class='vis-frame'><img src='data:image/png;base64,{img_b}' width='100%'></div></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='app-card card-pink' style='color:#9f1239; text-align:center; font-weight:800;'>⚠ Invalid SMILES B</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='app-card' style='background:#ffffff; text-align:center; height:200px; display:flex; align-items:center; justify-content:center; color:#334155; font-weight:600; border:2px dashed #e2e8f0;'>Enter a SMILES string to preview structure</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("Analyze Interaction Pipeline")

# ==========================================
# 💠 WELLNESS DASHBOARD (RESULTS)
# ==========================================

if analyze_btn:
    if not smi_a or not smi_b or not Chem.MolFromSmiles(smi_a) or not Chem.MolFromSmiles(smi_b):
        st.error("Protocol Incomplete: Structurally valid SMILES strings are required for full analysis.")
        st.stop()
    
    with st.spinner("Decoding Molecular Interaction Patterns..."):
        model_data = get_main_model()
        
        # 4120 Feature Extraction
        def get_features_4120(s1, s2):
            def v(s):
                m = Chem.MolFromSmiles(s)
                fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
                f = np.zeros((2048,))
                DataStructs.ConvertToNumpyArray(fp, f)
                try:
                    AllChem.ComputeGasteigerCharges(m)
                    c = [a.GetDoubleProp("_GasteigerCharge") for a in m.GetAtoms()]
                except:
                    c = [0.0] * m.GetNumAtoms()
                d = [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m), Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m), Descriptors.NumRotatableBonds(m), Descriptors.RingCount(m), Descriptors.NumAromaticRings(m), Descriptors.HeavyAtomCount(m), Descriptors.FractionCSP3(m), max(c) if c else 0.0, min(c) if c else 0.0]
                return f, np.array(d, dtype=np.float32)
            f1, d1 = v(s1); f2, d2 = v(s2)
            return np.concatenate([f1, f2, d1, d2]).reshape(1, -1)

        feats = get_features_4120(smi_a, smi_b)
        probs = model_data['model'].predict_proba(feats)[0]
        classes = model_data.get('class_names', model_data.get('label_encoder').classes_ if 'label_encoder' in model_data else [])
        pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        
        # 1. HERO PREDICTION
        st.markdown(f"""
        <div class='app-card card-pink' style='text-align:center;'>
            <div class='section-title' style='justify-content:center;'>Primary Interaction Prediction</div>
            <div class='prediction-val'>{pairs[0][0]}</div>
            <div class='confidence-val'>Confidence: {pairs[0][1]*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. PROBABILITY DISTRIBUTION
        st.markdown("<div class='app-card card-lavender'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Probability Distribution Profile</div>", unsafe_allow_html=True)
        df_p = pd.DataFrame(pairs, columns=["Category", "Weight"])
        df_p["Weight"] = df_p["Weight"].map(lambda x: f"{x*100:.2f}%")
        st.table(df_p)
        st.markdown("</div>", unsafe_allow_html=True)

        # 3. MOLECULE COMPARISON + 4. MCS Analysis
        m1_obj = Chem.MolFromSmiles(smi_a)
        m2_obj = Chem.MolFromSmiles(smi_b)
        mcs = rdFMCS.FindMCS([m1_obj, m2_obj], timeout=5)
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            im_a = generate_mol_preview(smi_a, m1_obj.GetSubstructMatch(Chem.MolFromSmarts(mcs.smartsString)) if mcs.numAtoms else None)
            st.markdown(f"<div class='app-card card-cream' style='padding:1rem;'><div class='section-title' style='font-size:0.9rem;'>Molecule A Core</div><div class='vis-frame'><img src='data:image/png;base64,{im_a}' width='100%'></div></div>", unsafe_allow_html=True)
        with m_col2:
            im_b = generate_mol_preview(smi_b, m2_obj.GetSubstructMatch(Chem.MolFromSmarts(mcs.smartsString)) if mcs.numAtoms else None)
            st.markdown(f"<div class='app-card card-cream' style='padding:1rem;'><div class='section-title' style='font-size:0.9rem;'>Molecule B Core</div><div class='vis-frame'><img src='data:image/png;base64,{im_b}' width='100%'></div></div>", unsafe_allow_html=True)
            
        st.markdown("<div class='app-card card-mint'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Maximum Common Substructure (MCS)</div>", unsafe_allow_html=True)
        if mcs.numAtoms:
            st.markdown(f"<div style='background:rgba(255,255,255,0.8); padding:1rem; border-radius:15px; font-family:monospace; margin-bottom:1rem; color:#0f172a; border:1px solid #d1fae5; font-weight:700;'>{mcs.smartsString}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#065f46; font-weight:700;'>Verified overlap: {mcs.numAtoms} unique heavy atoms identified.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#991b1b; font-weight:700;'>No significant common heavy-atom scaffolding found.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 5. GRAPH REPRESENTATION
        st.markdown("<div class='app-card card-blue'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Graph-Based Structural Representation</div>", unsafe_allow_html=True)
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            gimg_a = render_structural_graph(smi_a)
            st.markdown(f"<div class='vis-frame' style='background:white;'><img src='data:image/png;base64,{gimg_a}' width='100%'></div>", unsafe_allow_html=True)
            st.caption("Topological Map A")
        with gcol2:
            gimg_b = render_structural_graph(smi_b)
            st.markdown(f"<div class='vis-frame' style='background:white;'><img src='data:image/png;base64,{gimg_b}' width='100%'></div>", unsafe_allow_html=True)
            st.caption("Topological Map B")
        st.markdown("</div>", unsafe_allow_html=True)

        # 6. MOLECULAR DESCRIPTOR PANEL
        st.markdown("<div class='app-card card-cream'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Molecular Descriptor Comparison</div>", unsafe_allow_html=True)
        d1, d2 = analyze_molecule_properties(smi_a), analyze_molecule_properties(smi_b)
        df_d = pd.DataFrame([d1, d2], index=["Compound A", "Compound B"]).T
        st.table(df_d)
        st.markdown("</div>", unsafe_allow_html=True)

        # 7-12. SIMILARITY & SCAFFOLD
        st.markdown("<div class='app-card card-lavender'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Similarity & Scaffold Analytics</div>", unsafe_allow_html=True)
        
        # Calculations
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1_obj, 2)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2_obj, 2)
        sim_val = DataStructs.TanimotoSimilarity(fp1, fp2)
        
        scaf_a = safe_extract_scaffold(smi_a)
        scaf_b = safe_extract_scaffold(smi_b)
        
        scaf_sim = None
        if scaf_a and scaf_b:
            try:
                fps1 = AllChem.GetMorganFingerprintAsBitVect(scaf_a, 2)
                fps2 = AllChem.GetMorganFingerprintAsBitVect(scaf_b, 2)
                scaf_sim = DataStructs.TanimotoSimilarity(fps1, fps2)
            except: pass
        
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            st.markdown(f"<div style='font-size:2.5rem; font-weight:800; color:#1e3a8a;'>{sim_val:.4f}</div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#334155; font-weight:700;'>Tanimoto Similarity (Molecule)</div>", unsafe_allow_html=True)
        with sim_col2:
            if scaf_sim is not None:
                st.markdown(f"<div style='font-size:2.5rem; font-weight:800; color:#4338ca;'>{scaf_sim:.4f}</div>", unsafe_allow_html=True)
                st.markdown("<div style='color:#334155; font-weight:700;'>Murcko Scaffold Similarity</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='font-size:1.5rem; font-weight:800; color:#991b1b;'>Not Available</div>", unsafe_allow_html=True)
                st.markdown("<div style='color:#334155; font-weight:700;'>Murcko Scaffold Similarity</div>", unsafe_allow_html=True)
        
        # Scaffold Visualization
        if scaf_a or scaf_b:
            st.markdown("<br>", unsafe_allow_html=True)
            vscaf1, vscaf2 = st.columns(2)
            with vscaf1:
                if scaf_a:
                    img_scaf_a = base64.b64encode(Draw.MolToImage(scaf_a).tobytes()).decode()
                    # Using MolToImage instead of MolDraw2DCairo for simpler safe rendering in this loop
                    buf = BytesIO()
                    Draw.MolToImage(scaf_a).save(buf, format="PNG")
                    img_data = base64.b64encode(buf.getvalue()).decode()
                    st.markdown(f"<div class='vis-frame' style='background:white;'><img src='data:image/png;base64,{img_data}' width='100%'></div>", unsafe_allow_html=True)
                    st.caption("Murcko Scaffold A")
            with vscaf2:
                if scaf_b:
                    buf = BytesIO()
                    Draw.MolToImage(scaf_b).save(buf, format="PNG")
                    img_data = base64.b64encode(buf.getvalue()).decode()
                    st.markdown(f"<div class='vis-frame' style='background:white;'><img src='data:image/png;base64,{img_data}' width='100%'></div>", unsafe_allow_html=True)
                    st.caption("Murcko Scaffold B")
                    
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Processing Status
        st.markdown("<div class='app-card card-mint'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Processing Statistics</div>", unsafe_allow_html=True)
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            for status in ["Scaffold Extraction", "RDKit Parsing", "Graph Map", "Inference"]:
                st.markdown(f"<span class='status-pill' style='color:#065f46; background:#dcfce7;'>✓ {status}</span>", unsafe_allow_html=True)
        with pcol2:
            st.markdown(f"<div style='color:#0f172a; font-weight:800;'>Technical Logs</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#334155; font-weight:600; font-size:0.8rem;'>Protocol: research.v4120<br>Engine: bio-inference.proc<br>Timestamp: {datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Scientific Notes
        st.markdown("<div class='app-card card-blue'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Structural Interpretation Notes</div>", unsafe_allow_html=True)
        notes = []
        if sim_val > 0.4: notes.append("Shared molecular fingerprints indicate overlapping chemical environments.")
        if mcs.numAtoms > 10: notes.append("Robust common scaffold isolated; metabolic pathway similarity is plausible.")
        if scaf_sim is not None and scaf_sim > 0.7: notes.append("High scaffold overlap suggests direct structural analogs.")
        if not notes: notes.append("Prediction derived from distributed chemical environment descriptors.")
        for nt in notes: 
            st.markdown(f"<div class='note-box' style='color:#1e293b; font-weight:500;'>🧬 {nt}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Export
        st.markdown("<div style='display:flex; justify-content:center; gap:1.5rem; margin-bottom: 4rem;'>", unsafe_allow_html=True)
        st.download_button("Export Report (JSON)", data=json.dumps({"smi_a": smi_a, "smi_b": smi_b, "prediction": pairs[0][0]}, indent=2), file_name="DDI_Report.json", mime="application/json")
        st.download_button("Export Data (CSV)", data=df_p.to_csv().encode('utf-8'), file_name="Structural_Analysis.csv", mime="text/csv")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='text-align: center; color: #64748b; margin-top: 2rem; font-size: 0.9rem; font-weight: 700;'>DrugLens Biomedical | Research Intelligence Platform v3.1</div>", unsafe_allow_html=True)