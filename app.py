import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="AMP AI | Shivansh Sahu", layout="wide", page_icon="🧬")

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #1e3d59;}
    .stAlert {border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD AI MODELS ---
@st.cache_resource
def load_models():
    model_path = os.path.join(os.path.dirname(__file__), "V2_Preprocessor.pkl")
    preprocessor = joblib.load(model_path)
    model = joblib.load("AMP_Stacking_Model.pkl")
    return preprocessor, model

preprocessor, model = load_models()

# --- 3. BIOLOGICAL EXTRACTION ENGINE ---
def extract_biological_features(sequence):
    clean_seq = sequence.replace('X', '').replace('B', '').replace('Z', '').upper()
    analysis = ProteinAnalysis(clean_seq)
    
    try:
        mol_weight = analysis.molecular_weight()
        iso_point = analysis.isoelectric_point()
        aromaticity = analysis.aromaticity()
        instability = analysis.instability_index()
    except:
        mol_weight, iso_point, aromaticity, instability = 0, 0, 0, 0

    kmers = [clean_seq[i:i+4] for i in range(len(clean_seq) - 3)]
    kmer_sentence = " ".join(kmers) if kmers else ""
    
    return {
        'Molecular_Weight': mol_weight,
        'Isoelectric_Point': iso_point,
        'Aromaticity': aromaticity,
        'Instability_Index': instability,
        'Kmer_Sentence': kmer_sentence,
        'Clean_Seq': clean_seq
    }

# --- 4. SIDEBAR DASHBOARD ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2022/2022294.png", width=80)
st.sidebar.title("Configuration")
st.sidebar.markdown("Configure the AI screening parameters below.")

target_bug = st.sidebar.selectbox(
    "🦠 Select Target Pathogen:",
    ('E. coli', 'P. aeruginosa', 'A. baumannii', 'K. pneumoniae', 'S. typhimurium')
)

st.sidebar.markdown("---")
# Quick Load Example Sequences
example_seq = st.sidebar.selectbox("Load Example Sequence:", ["Custom", "GIGKFLHSAKK (Known AMP)", "AAAAA (Harmless Peptide)"])

default_text = ""
if example_seq == "GIGKFLHSAKK (Known AMP)": default_text = "GIGKFLHSAKK"
elif example_seq == "AAAAA (Harmless Peptide)": default_text = "AAAAA"

user_sequence = st.sidebar.text_area("🧬 Enter Amino Acid Sequence:", value=default_text, height=150).upper()

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Lead Engineer: Shivansh Sahu**")
st.sidebar.markdown("🔬 *Gram-Negative Drug Discovery Pipeline*")

# --- 5. MAIN DASHBOARD ---
st.title("🧬 AI-Driven Antimicrobial Peptide Discoverer")
st.markdown(f"**Targeting:** `{target_bug}` | **Architecture:** Triple-Branch Stacking Ensemble (XGB, RF, SVM)")

# Create Tabs for a cleaner layout
tab1, tab2, tab3 = st.tabs(["🔬 Clinical Screening", "📊 Biochemical Analytics", "🧠 About the AI"])

with tab1:
    if st.sidebar.button("🚀 Run AI Screening", use_container_width=True):
        if len(user_sequence) < 4:
            st.warning("⚠️ Sequence must be at least 4 amino acids long.")
        else:
            with st.spinner("Extracting biology & querying Stacking Ensemble..."):
                
                # Math & Prediction
                features = extract_biological_features(user_sequence)
                input_data = pd.DataFrame([{
                    'bacterium': target_bug,
                    'Molecular_Weight': features['Molecular_Weight'],
                    'Isoelectric_Point': features['Isoelectric_Point'],
                    'Aromaticity': features['Aromaticity'],
                    'Instability_Index': features['Instability_Index'],
                    'Kmer_Sentence': features['Kmer_Sentence']
                }])
                processed_data = preprocessor.transform(input_data)
                probability = model.predict_proba(processed_data)[0][1]
                
                # Layout for Results
                col1, col2 = st.columns([1.5, 1])
                
                with col1:
                    st.subheader("Clinical Verdict")
                    if probability >= 0.75:
                        st.success("✅ **HIGH POTENTIAL AMP DETECTED!**\n\nThe AI predicts with high confidence that this sequence will penetrate the outer membrane of the target pathogen. Recommended for clinical lab synthesis.")
                        st.balloons()
                    elif probability >= 0.50:
                        st.info("⚠️ **MODERATE POTENTIAL**\n\nShows antimicrobial properties, but clinical efficacy against this specific pathogen may be too low for pharmaceutical use.")
                    else:
                        st.error("❌ **NON-AMP / HARMLESS**\n\nDo not synthesize. The AI predicts this protein will not successfully inhibit the target pathogen.")
                
                with col2:
                    # Beautiful Plotly Gauge Chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability * 100,
                        title = {'text': "AI Confidence (%)", 'font': {'size': 20}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "#1e3d59"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 50], 'color': "#ffb3b3"}, # Red zone
                                {'range': [50, 75], 'color': "#ffe6cc"}, # Yellow zone
                                {'range': [75, 100], 'color': "#ccebff"}], # Green zone
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 75} # Clinical strict threshold
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Physical Chemistry Profile")
    if len(user_sequence) >= 4:
        features = extract_biological_features(user_sequence)
        
        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Molecular Weight", f"{features['Molecular_Weight']:.2f} Da")
        col2.metric("Isoelectric Point", f"{features['Isoelectric_Point']:.2f} pH")
        col3.metric("Aromaticity", f"{features['Aromaticity']*100:.1f}%")
        col4.metric("Instability Index", f"{features['Instability_Index']:.2f}")
        
        st.markdown("---")
        
        # Amino Acid Composition Graph
        st.subheader("Amino Acid Composition")
        aa_counts = Counter(features['Clean_Seq'])
        df_aa = pd.DataFrame.from_dict(aa_counts, orient='index').reset_index()
        df_aa.columns = ['Amino Acid', 'Count']
        df_aa = df_aa.sort_values(by='Count', ascending=False)
        
        fig_bar = px.bar(df_aa, x='Amino Acid', y='Count', 
                         title="Frequency of Amino Acids in Sequence",
                         color='Count', color_continuous_scale='Blues')
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.write(f"**Generated NLP K-mer Sentence:** `{features['Kmer_Sentence']}`")
    else:
        st.info("👈 Run an AI Screening first to view the biochemical breakdown.")

with tab3:
    st.subheader("About the Architecture")
    st.write("""
    This diagnostic tool is powered by a **Triple-Branch Machine Learning Pipeline** designed by Shivansh Sahu.
    
    1. **The Biological Engine:** Translates raw 1D protein strings into 4 distinct physical chemistry metrics using `Biopython`.
    2. **The NLP Engine:** Uses a sliding-window algorithm to tokenize the sequence into overlapping 4-mers, vectorized via `TF-IDF`.
    3. **The Stacking Ensemble:** A highly advanced Meta-Learner combining the predictive power of:
        * **Random Forest:** To extract non-linear chemical thresholds.
        * **Support Vector Machine (SVM):** To slice through the high-dimensional TF-IDF text matrix.
        * **XGBoost:** To capture complex biological interactions.
        * **Logistic Regression:** Acts as the final Meta-Judge to output the ultimate clinical probability.
    """)
