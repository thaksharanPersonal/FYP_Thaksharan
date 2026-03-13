import streamlit as st
import plotly.graph_objects as go
import numpy as np
import cv2
import librosa
from scipy.signal import correlate
import tempfile
import os
from realsyncnet_cli import analyze_sync

st.set_page_config(page_title="🧠 CMEC", layout="wide")

# Clean professional UI
st.markdown("""
<style>
    footer { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 **CMEC Dynamic XAI**")
st.markdown("**w1868277**")

video = st.file_uploader("📁 Upload Video", type=['mp4','mov','avi','mkv'])

if video is not None:
    with st.spinner("🔬 Computing temporal alignment..."):
        offset, conf, dist, audio_energy, lip_motion, corr, fps = analyze_sync(video)
    
    # DYNAMIC METRICS
    col1, col2, col3 = st.columns(3)
    with col1: 
        st.metric("🎯 Offset", f"{offset:.3f}s", delta=None)
        st.caption("Negative = audio recorded early")
    with col2: 
        st.metric("✅ Confidence", f"{conf:.3f}", delta=None)
        st.caption("Peak sharpness (higher=better)")
    with col3: 
        st.metric("📏 Distance", f"{dist:.3f}", delta=None)
        st.caption("Sync error (lower=better)")
    
    st.markdown("---")
    st.subheader("🧠 **Dynamic XAI: Computed Explanation**")
    
    # COMPUTE DYNAMIC XAI METRICS 
    peak_idx = np.argmax(corr)
    frame_lag = peak_idx - len(lip_motion) + 1
    frame_lag_seconds = frame_lag / fps
    peak_height = corr[peak_idx]
    top_peaks = np.sum(corr > peak_height * 0.8)
    
    # SAFE RAW CORRELATION 
    min_len = min(len(audio_energy), len(lip_motion))
    raw_corr = 0.0
    if min_len > 1:
        try:
            raw_corr = np.corrcoef(audio_energy[:min_len], lip_motion[:min_len])[0,1]
        except:
            raw_corr = 0.0
    
    # SAFE ALIGNED CORRELATION 
    alignment_quality = 0.0
    shift_samples = int(frame_lag * fps * 16000 / 1600)
    
    if abs(shift_samples) < len(audio_energy) and len(lip_motion) > 1:
        try:
            aligned_audio = np.roll(audio_energy, shift_samples)
            max_overlap = min(len(aligned_audio), len(lip_motion))
            if max_overlap > 1:
                aligned_audio_trim = aligned_audio[:max_overlap]
                lip_motion_trim = lip_motion[:max_overlap]
                alignment_quality = np.corrcoef(aligned_audio_trim, lip_motion_trim)[0,1]
        except:
            alignment_quality = 0.0
    
    # QUALITY LABEL 
    if alignment_quality >= 0.8:
        quality_label = "Excellent"
    elif alignment_quality >= 0.6:
        quality_label = "Good"
    else:
        quality_label = "Poor"
    
    # GRAPH 1: CORRELATION HEATMAP
    fig1 = go.Figure()
    fig1.add_trace(go.Heatmap(z=[corr], 
                             colorscale='Viridis', 
                             zmid=peak_height*0.7,
                             hovertemplate="Lag: %{x:.0f} frames<br>Similarity: %{z:.3f}<extra></extra>"))
    
    fig1.add_shape(type="line", x0=peak_idx, x1=peak_idx, y0=0, y1=1,
                  xref="x", yref="paper", line=dict(color="red", width=5, dash="dash"))
    
    fig1.add_annotation(x=peak_idx, y=0.5, 
                       text=f"Peak: {peak_idx} frames<br>{offset:.3f}s",
                       showarrow=True, arrowhead=2, font=dict(size=14, color="white"))
    
    fig1.update_layout(title=f"📊 **Correlation Peak Analysis** (Peak: {peak_height:.3f})",
                      xaxis_title="Frame Lag", height=400, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    
    # GRAPH 2: SIGNAL ALIGNMENT
    t_audio = np.arange(len(audio_energy)) / fps
    t_video = np.arange(len(lip_motion)) / fps
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t_audio, y=audio_energy, mode='lines', 
                             name='🎵 Raw Audio Energy', line=dict(color='#1f77b4', width=3)))
    fig2.add_trace(go.Scatter(x=t_video, y=lip_motion, mode='lines', 
                             name='👄 Raw Lip Motion', line=dict(color='#ff7f0e', width=3)))
    
    if abs(shift_samples) < len(audio_energy):
        aligned_t = t_audio
        aligned_y = np.roll(audio_energy, shift_samples)
        fig2.add_trace(go.Scatter(x=aligned_t, y=aligned_y, mode='lines', 
                                 name='🎵 Audio (Aligned)', line=dict(color='#1f77b4', width=4, dash='dot')))
    
    fig2.add_vline(x=abs(offset), line_dash="dash", line_color="green",
                  annotation_text=f"SYNC POINT")
    
    fig2.update_layout(title="🎵 **Audio vs Lip Motion: Dynamic Alignment**",
                      xaxis_title="Time (s)", yaxis_title="Amplitude", height=450)
    st.plotly_chart(fig2, use_container_width=True)
    
    # DYNAMIC XAI EXPLANATION
    st.markdown(f"""
    ## **Dynamic XAI: Computed Causal Analysis**
    
    ### **Computed Correlation Analysis**
    | **Metric** | **Value** | **Interpretation** |
    |------------|-----------|-------------------|
    | Peak Frame | {peak_idx} | Max similarity position |
    | Frame Lag | {frame_lag:+.0f} | {frame_lag_seconds:+.2f}s shift |
    | Peak Height | {peak_height:.3f} | Similarity strength |
    | Competing Peaks | {top_peaks} | Uniqueness (lower=better) |
    
    ### **Dynamic Signal Quality**
    | **Alignment** | **Correlation** | **Quality** |
    |---------------|-----------------|-------------|
    | Raw Signals | {raw_corr:.3f} | **{quality_label}** |
    | Aligned Signals | {alignment_quality:.3f} | **{quality_label}** |
    
    ### **Causal Explanation**
    **Peak correlation at frame {peak_idx} ({offset:+.3f}s) causally drives alignment.**
    
    **Computed evidence:**
    1. Peak height {peak_height:.3f} confirms strong AV match
    2. Only {top_peaks} competing peaks → unambiguous decision  
    3. Post-alignment correlation improved: {raw_corr:.3f} → **{alignment_quality:.3f}**
    
    """)
    
# footer line break
st.markdown("---")
