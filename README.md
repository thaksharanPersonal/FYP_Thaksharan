AI system for detecting and explaining temporal misalignment between independent audio and video streams.

This project explores how Explainable AI (XAI) can improve trust and transparency in multimedia synchronization systems.

🔗 Project Microsite:
https://sites.google.com/iit.ac.lk/explainable-av-synchronization/home

**Project Overview**

Audio-video synchronization errors can occur in many real-world applications, such as:

1. Video conferencing
2. Surveillance footage
3. Telehealth consultations
4. Media production

While modern AI models can detect synchronization errors, many operate as black-box systems, providing little insight into how predictions are made.

This project aims to develop a system that can:

1. Detect temporal misalignment between audio and video streams
2. Provide interpretable outputs explaining the detected synchronization errors
3. Improve trust in AI-driven multimedia analysis.

**Key Features**

1. Upload audio and video streams for analysis
2. Detect temporal offset between streams
3. AI-based synchronization detection using SyncNet

Dashboard displaying:
1. Offset prediction
2. Confidence score
3. Distance metrics

Web interface built using Streamlit

**System Architecture**

The system consists of three major components:

**1. User Interface**
Built with Streamlit
Allows users to upload video files
Displays synchronization analysis results

**2. Processing Pipeline**
Extracts and preprocesses audio and video frames
Prepares input data for the AI model

**3. SyncNet Model**
Pre-trained deep learning model
Predicts synchronization offset between audio and video streams

**Technologies Used:**
Python
Streamlit
SyncNet (ECCV 2016)
FFmpeg
Deep Learning for Audio-Visual Analysis
