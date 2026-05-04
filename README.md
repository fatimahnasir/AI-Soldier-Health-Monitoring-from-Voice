# AI Soldier Health Monitoring from Voice
AI-Based Soldier Battlefield Voice Condition Detection
# AI Soldier Voice Situation Detection

An intelligent voice analysis system that detects a soldier’s battlefield condition through speech patterns and vocal expressions using Machine Learning and audio signal processing.

---

## Project Overview
In high-pressure combat situations, soldiers may not always be able to verbally communicate their physical or mental condition. This system uses audio feature extraction and unsupervised machine learning clustering to analyze voice signals and detect hidden emotional or physical stress indicators based on:

*   Speech tempo
*   Voice energy
*   Frequency patterns
*   Vocal tension
*   Breathing rhythm

Unlike traditional monitoring systems that rely on physical sensors, this project uses voice-based AI detection, making it a non-invasive and practical solution for battlefield environments where wearable health devices may not be feasible.

## Key Features
*   **Voice-based battlefield condition detection:** Identifies operational states such as Normal, Stressed, and Fatigued.
*   **Automatic audio feature extraction:** Utilizes Librosa and NumPy for signal analysis.
*   **Unsupervised AI clustering:** Employs K-Means to group voice patterns into behavioral clusters.
*   **Streamlit Dashboard:** Interactive interface for real-time audio upload and analysis.
*   **Non-invasive monitoring:** Ideal for environments where wearable sensors are impractical.

## Technologies Used
*   **Programming Language:** Python
*   **Machine Learning:** Scikit-learn (K-Means Clustering)
*   **Audio Processing:** Librosa, NumPy
*   **Frontend / Dashboard:** Streamlit
*   **Model Storage:** Pickle

## Project Structure
```text
AI-Soldier-Voice-Detection/
│
├── data/                       # Sample voice recordings (.wav)
├── extract_features.py         # Script for MFCC and Chroma extraction
├── train_cluster.py            # Script to train the K-Means model
├── app.py                      # Main Streamlit application
├── kmeans_model.pkl            # Saved ML model
├── file_cluster_mapping.pkl    # Metadata for clusters
└── README.md                   # Project documentation
