import sys
import librosa
import cv2
import numpy as np
from scipy.signal import correlate
from pathlib import Path
import tempfile
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def analyze_sync(video_path):
    # Handle both string path AND Streamlit file objects
    if hasattr(video_path, 'read'):  # Streamlit UploadedFile
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_path.read())
            tmp_path = tmp.name
    else:  # Regular file path
        tmp_path = str(video_path)
    
    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Audio: Speech energy (RMS)
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        audio_energy = np.array([librosa.feature.rms(y=y[i:i+1600])[0][0] 
                                for i in range(0, len(y), 1600)][:300])
        
        # Video: Mouth motion 
        lip_motion = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(min(300, frame_count)):
            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            roi = cv2.cvtColor(frame[int(h*0.4):int(h*0.6), int(w*0.3):int(w*0.7)], cv2.COLOR_BGR2GRAY)
            lip_motion.append(np.std(roi))
        
        cap.release()
        
        # Cross-correlation (mathematically perfect)
        corr = correlate(audio_energy[:len(lip_motion)], np.array(lip_motion), mode='full')
        offset_frames = np.argmax(corr) - len(lip_motion) + 1
        offset_seconds = offset_frames / fps
        
        # Confidence: Peak-to-noise ratio
        peak_strength = corr.max() / (np.std(audio_energy) * np.std(lip_motion) + 1e-8)
        confidence = min(1.0, peak_strength / 3.0)
        
        return offset_seconds, confidence, 1-confidence, audio_energy, lip_motion, corr, fps
        
    finally:
        # Clean up temp file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not video_path:
        print("0.0000"); print("0.0000"); print("1.0000"); print("ERROR")
        return
    
    offset, conf, dist, *_ = analyze_sync(video_path)
    print(f"{offset:.4f}")
    print(f"{conf:.4f}")
    print(f"{dist:.4f}")
    print("OK")

if __name__ == '__main__':
    main()
