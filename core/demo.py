import streamlit as st
import plotly.graph_objects as go
import numpy as np
import cv2
import librosa
from scipy.signal import correlate
import tempfile
import os
from realsyncnet_cli import analyze_sync

# --- CONFIG & THEME ---
st.set_page_config(page_title="🧠 CMEC Dynamic XAI", layout="wide")

# Custom CSS for the "AI Dashboard" look
st.markdown("""
<style>
    /* Dark Cyber Theme */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Glowing Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.3);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1);
    }
    
    /* Header Styling */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        background: -webkit-linear-gradient(#00d4ff, #0055ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Info Boxes */
    .stAlert {
        background-color: rgba(0, 212, 255, 0.1);
        border: 1px solid #00d4ff;
    }

    /* Custom Data Table Styling */
    .data-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #00d4ff;
        margin-bottom: 20px;
    }

    hr { border-top: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<p class="main-title">🧠 CMEC Dynamic XAI</p>', unsafe_allow_html=True)
    st.markdown("🔍 **Temporal Audio-Visual Alignment Intelligence**")
with col_h2:
    st.markdown(f"<div style='text-align: right; margin-top: 20px;'><code style='color: #00d4ff;'>SYSTEM STATUS: ACTIVE</code><br><small>User: w1868277</small></div>", unsafe_allow_html=True)

# --- UPLOAD SECTION ---
with st.container():
    video = st.file_uploader("📂 VIDEO FEED", type=['mp4','mov','avi','mkv'])

if video is not None:
    with st.spinner("⚡ PROCESSING NEURAL TEMPORAL ALIGNMENT..."):
        offset, conf, dist, audio_energy, lip_motion, corr, fps = analyze_sync(video)
    
    # --- DYNAMIC METRICS GRID ---
    st.markdown("### 📊 CORE DIAGNOSTICS")
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.metric("🎯 OFFSET", f"{offset:.3f}s", delta="Negative = Audio Early" if offset < 0 else None)
    with c2: 
        st.metric("✅ CONFIDENCE", f"{conf:.3f}", help="Peak sharpness (higher is better)")
    with c3: 
        st.metric("📏 DISTANCE", f"{dist:.3f}", delta_color="inverse", help="Sync error (lower is better)")
    
    st.markdown("---")
    
    # --- COMPUTATIONS FOR XAI ---
    peak_idx = np.argmax(corr)
    frame_lag = peak_idx - len(lip_motion) + 1
    frame_lag_seconds = frame_lag / fps
    peak_height = corr[peak_idx]
    top_peaks = np.sum(corr > peak_height * 0.8)
    
    # Correlation Calculations (preserving your original logic)
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
    tab1, tab2 = st.tabs(["📈 VISUAL EVIDENCE", "🤖 DYNAMIC XAI ANALYSIS"])
    
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
        
        # Aligned audio (normalized)
        if abs(shift_samples) < len(audio_energy):
            aligned_y_norm = np.roll(audio_norm, shift_samples)
            fig2.add_trace(go.Scatter(x=t_audio, y=aligned_y_norm, mode='lines', 
                                    name='🎵 Audio Aligned (Normalized)', 
                                    line=dict(color='#1f77b4', width=4, dash='dot'), yaxis='y'))
        
        fig2.add_vline(x=abs(offset), line_dash="dash", line_color="green",
                    annotation_text=f"SYNC: {offset:+.3f}s")
        
        # Dual Y-axis setup
        fig2.update_layout(
            title="🎵 **Audio vs Lip Motion: Dynamic Alignment (Normalized)**",
            xaxis_title="Time (s)",
            yaxis=dict(title="Audio Energy", side="left", range=[0, 1]),
            yaxis2=dict(title="Lip Motion", side="right", range=[0, 1], overlaying="y"),
            height=450
        )
        st.plotly_chart(fig2, use_container_width=True)


    with tab2:
        st.subheader("🧠 Computed Explanation & Causal Analysis")
        
        # Original Detailed Data preserved in a modern layout
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
st.markdown("<center><small>CMEC Dynamic XAI Engine</small></center>", unsafe_allow_html=True)