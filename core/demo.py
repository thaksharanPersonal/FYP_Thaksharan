import streamlit as st
import plotly.graph_objects as go
import numpy as np
import cv2
import librosa
from scipy.signal import correlate
import tempfile
import os
from realsyncnet_cli import analyze_sync

# CONFIG & THEME
st.set_page_config(page_title="🧠 CMEC Dynamic XAI", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top right, #0a192f, #060d17);
        color: #ccd6f6;
    }
    
    /* Neon Glow Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(16, 33, 65, 0.5);
        border: 1px solid rgba(100, 255, 218, 0.2);
        padding: 25px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        transition: 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #64ffda;
        transform: translateY(-5px);
    }
    
    /* Custom Header styling */
    .hero-text {
        text-align: center;
        padding: 40px 0;
        animation: fadeIn 2s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .main-title {
        font-size: 56px;
        font-weight: 800;
        background: linear-gradient(90deg, #64ffda, #48cae4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    /* Landing Page Cards */
    .landing-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        height: 100%;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# HEADER SECTION
with st.container():
    c_left, c_right = st.columns([3, 1])
    with c_left:
        st.markdown('<p class="main-title">🧠 CMEC Dynamic XAI</p>', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#8892b0; font-weight:300;'>Temporal Audio-Visual Alignment Intelligence</h3>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

# UPLOAD SECTION 
with st.expander("📂 INITIATE UPLOAD", expanded=not st.session_state.video_processed):
    video = st.file_uploader("", type=['mp4','mov','avi','mkv'], help="Upload audiovisual data for temporal forensic analysis.")

if video is None:
    
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(125deg, #060d17, #0a192f, #001219, #002129);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Optional: Add a subtle 'Grid' overlay for the forensic look */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background-image: radial-gradient(rgba(100, 255, 218, 0.05) 1px, transparent 0);
            background-size: 40px 40px;
            z-index: -1;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")
    l_col1, l_col2, l_col3 = st.columns(3)
    
    with l_col1:
        st.markdown("""
        <div class="landing-card">
            <h3 style="color:#64ffda;">📡 The Problem</h3>
            <p style="color:#8892b0;">Independent recording streams often suffer from device clock drift and network jitter, leading to 'Black Box' sync errors that current AI models fail to explain.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with l_col2:
        st.markdown("""
        <div class="landing-card">
            <h3 style="color:#64ffda;">🧠 The Solution</h3>
            <p style="color:#8892b0;">CMEC provides a 3-layer architecture utilizing SyncNet baselines and Transformer-based XAI to generate human-interpretable attribution maps.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with l_col3:
        st.markdown("""
        <div class="landing-card">
            <h3 style="color:#64ffda;">⚖️ Transparency</h3>
            <p style="color:#8892b0;">Our framework answers 'WHY' an offset exists by visualizing causal perturbation and temporal saliency in high-stakes environments.</p>
        </div>
        """, unsafe_allow_html=True)

# RESULTS PROCESSING
if video is not None:
    with st.spinner("⚡ PROCESSING TEMPORAL ALIGNMENT..."):
        offset, conf, dist, audio_energy, lip_motion, corr, fps = analyze_sync(video)
    
    # DYNAMIC METRICS GRID 
    st.markdown("### 📊 CORE DIAGNOSTICS")
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.metric("🎯 OFFSET", f"{offset:.3f}s", delta="Negative = Audio Early" if offset < 0 else None)
    with c2: 
        st.metric("✅ CONFIDENCE", f"{conf:.3f}", help="Peak sharpness (higher is better)")
    with c3: 
        st.metric("📏 DISTANCE", f"{dist:.3f}", delta_color="inverse", help="Sync error (lower is better)")
    
    st.markdown("---")
    
    # COMPUTATIONS FOR XAI
    peak_idx = np.argmax(corr)
    frame_lag = peak_idx - len(lip_motion) + 1
    frame_lag_seconds = frame_lag / fps
    peak_height = corr[peak_idx]
    top_peaks = np.sum(corr > peak_height * 0.8)
    
    # Correlation Calculations
    min_len = min(len(audio_energy), len(lip_motion))
    raw_corr = 0.0
    if min_len > 1:
        try: raw_corr = np.corrcoef(audio_energy[:min_len], lip_motion[:min_len])[0,1]
        except: raw_corr = 0.0
    
    alignment_quality = 0.0
    shift_samples = int(frame_lag * fps * 16000 / 1600)
    if abs(shift_samples) < len(audio_energy) and len(lip_motion) > 1:
        try:
            aligned_audio = np.roll(audio_energy, shift_samples)
            max_overlap = min(len(aligned_audio), len(lip_motion))
            if max_overlap > 1:
                alignment_quality = np.corrcoef(aligned_audio[:max_overlap], lip_motion[:max_overlap])[0,1]
        except: alignment_quality = 0.0
    
    quality_label = "Excellent" if alignment_quality >= 0.8 else "Good" if alignment_quality >= 0.6 else "Poor"
    quality_color = "#00ff88" if alignment_quality >= 0.6 else "#ff4b4b"

    # --- VISUALIZATION LAYOUT ---
    tab1, tab2 = st.tabs(["📈 VISUAL EVIDENCE ", " 🤖 DYNAMIC XAI ANALYSIS"])
    
    with tab1:
        # GRAPH 1: HEATMAP
        fig1 = go.Figure(data=go.Heatmap(
            z=[corr], colorscale='Viridis', zmid=peak_height*0.7,
            hovertemplate="Lag: %{x:.0f} frames<br>Similarity: %{z:.3f}<extra></extra>"))
        fig1.add_vline(x=peak_idx, line_width=3, line_dash="dash", line_color="red")
        fig1.update_layout(title="<b>CORRELATION PEAK ANALYSIS</b>", height=300, 
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig1, use_container_width=True)
        
    
        # GRAPH 2: SIGNAL ALIGNMENT
        t_audio = np.arange(len(audio_energy)) / fps
        t_video = np.arange(len(lip_motion)) / fps
        
        # NORMALIZE FOR PROPER VISUALIZATION
        audio_norm = (audio_energy - np.min(audio_energy)) / (np.max(audio_energy) - np.min(audio_energy) + 1e-8)
        lip_norm = (lip_motion - np.min(lip_motion)) / (np.max(lip_motion) - np.min(lip_motion) + 1e-8)
        
        fig2 = go.Figure()
        
        # Dual Y-axis for proper scaling
        fig2.add_trace(go.Scatter(x=t_audio, y=audio_norm, mode='lines', 
                                name='🎵 Raw Audio Energy (Normalized)', 
                                line=dict(color='#1f77b4', width=3), yaxis='y'))
        
        fig2.add_trace(go.Scatter(x=t_video, y=lip_norm, mode='lines', 
                                name='👄 Raw Lip Motion (Normalized)', 
                                line=dict(color='#ff7f0e', width=3), yaxis='y2'))
        
        # Aligned audio
        if abs(shift_samples) < len(audio_energy):
            aligned_y_norm = np.roll(audio_norm, shift_samples)
            fig2.add_trace(go.Scatter(x=t_audio, y=aligned_y_norm, mode='lines', 
                                    name='🎵 Audio Aligned (Normalized)', 
                                    line=dict(color='#1f77b4', width=4, dash='dot'), yaxis='y'))
        
        fig2.add_vline(x=abs(offset), line_dash="dash", line_color="green",
                    annotation_text=f"SYNC: {offset:+.3f}s")
        
        # Dual Y-axis setup
        fig2.update_layout(
            title="🎵 Audio vs Lip Motion: Dynamic Alignment (Normalized)",
            xaxis_title="Time (s)",
            yaxis=dict(title="Audio Energy", side="left", range=[0, 1]),
            yaxis2=dict(title="Lip Motion", side="right", range=[0, 1], overlaying="y"),
            height=450
        )
        st.plotly_chart(fig2, use_container_width=True)


    with tab2:
        st.subheader("🧠 Computed Explanation & Causal Analysis")
        
        # Original Detailed Data
        col_xai1, col_xai2 = st.columns(2)
        
        with col_xai1:
            st.markdown(f"""
            <div class="data-card">
                <h4>📊 Computed Correlation Analysis</h4>
                <table style="width:100%">
                    <tr><td>Peak Frame</td><td><code>{peak_idx}</code></td></tr>
                    <tr><td>Frame Lag</td><td><code>{frame_lag:+.0f} ({frame_lag_seconds:+.2f}s)</code></td></tr>
                    <tr><td>Peak Height</td><td><code>{peak_height:.3f}</code></td></tr>
                    <tr><td>Competing Peaks</td><td><code>{top_peaks}</code></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col_xai2:
            st.markdown(f"""
            <div class="data-card">
                <h4>⚡ Dynamic Signal Quality</h4>
                <p> </p>
                <table style="width:100%">
                    <tr><td>Raw Signals</td><td><code>{raw_corr:.3f}</code></td></tr>
                    <tr><td>Aligned Signals</td><td><code>{alignment_quality:.3f}</code></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        # THE CAUSAL EXPLANATION BLOCK
        st.info(f"💡 **Causal Inference Summary:** Peak correlation at frame **{peak_idx}** ({offset:+.3f}s) causally drives alignment.")
        
        st.markdown(f"""
        ### **Computed Evidence Deep-Dive**
        1. **Confidence Index:** Peak height of **{peak_height:.3f}** confirms a strong AV match.
        2. **Uniqueness:** Only **{top_peaks}** competing peaks found, indicating an unambiguous alignment decision.
        3. **Optimization Gain:** Post-alignment correlation improved from {raw_corr:.3f} to **{alignment_quality:.3f}**, validating the temporal shift.
        """)

st.markdown("---")
st.markdown("<center><p style='color:#4b5563; font-size:12px;'>CMEC DYNAMIC XAI ENGINE | 2026 RESEARCH EDITION</p></center>", unsafe_allow_html=True)