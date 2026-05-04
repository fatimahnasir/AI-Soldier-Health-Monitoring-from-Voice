import streamlit as st
import numpy as np
import pickle
import base64
from extract_features import extract_features
import os 

def get_base64_of_bin_file(bin_file):
    """
    Convert local file to Base64 string for CSS embedding.
    """
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
    
        return None


IMAGE_FILE_PATH = "image.png"
base64_image = get_base64_of_bin_file(IMAGE_FILE_PATH)


kmeans = None
file_cluster_mapping = None

try:
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open("file_cluster_mapping.pkl", "rb") as f:
        file_cluster_mapping = pickle.load(f)
except FileNotFoundError:
    st.warning("Model files not found! Prediction logic will be skipped.")
except Exception as e:
    st.error(f"Error loading models: {e}")


st.set_page_config(
    page_title="AI Soldier Voice Detection",
    page_icon="🎖️",
    layout="wide" 
)

if base64_image:
    BACKGROUND_CSS = f"""
    <style>
    /* 1. Main App Background: Image with Overlay */
    .stApp {{
        background-image: url("data:image/png;base64,{base64_image}"); 
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white; 
    }}

    /* Add a dark overlay to make text more readable */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.6); /* Dark semi-transparent overlay */
        z-index: -1; 
    }}

    /* 2. Sidebar Background Color: Black (Kept) */
    .stSidebar {{
        background-color: #000000; 
        color: #ffffff; 
        z-index: 1; 
    }}

    /* 3. Fixed Screen Size / Max Width Control for Main Content */
    .main .block-container {{
        max-width: 1100px;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 2rem; 
    }}

    /* 4. File Uploader Button (Browse files) Customization (In Sidebar) */
    .stSidebar .stFileUploader button {{
        background-color: #000000 !important; 
        color: #f0f0f0 !important; 
        border: 1px solid #f0f0f0 !important; 
    }}

    /* 5. Result Card Styling (Using Maroon for the card background) */
    .card {{
        background-color: #800000; /* Deep Maroon for the Result Card */
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 8px 16px rgba(0,0,0,0.4);
        margin-top: 20px;
        height: 100%; 
    }}

    /* Title styling - ensuring it's clearly visible */
    h1 {{
        text-align: center;
        font-size: 3rem;
        color: #F0F0F0;
        margin-bottom: 0.5rem;
    }}
    p {{
        text-align: center;
        color: #E0E0E0;
    }}
    </style>
    """
    st.markdown(BACKGROUND_CSS, unsafe_allow_html=True)
else:
    # Fallback CSS if image is not found (using the previous dark gradient)
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(120deg, #181A24, #32374A);
        color: white;
    }
    .stSidebar { background-color: #000000; color: #ffffff; }
    .main .block-container { max-width: 1100px; padding-left: 2rem; padding-right: 2rem; padding-top: 2rem; }
    .stSidebar .stFileUploader button { background-color: #000000 !important; color: #f0f0f0 !important; border: 1px solid #f0f0f0 !important; }
    .card { background-color: #800000; border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0px 8px 16px rgba(0,0,0,0.4); margin-top: 20px; height: 100%; }
    h1 { text-align: center; font-size: 3rem; color: #F0F0F0; margin-bottom: 0.5rem; }
    p { text-align: center; color: #E0E0E0; }
    </style>
    """, unsafe_allow_html=True)
    st.warning(f"Using default background. Please ensure '{IMAGE_FILE_PATH}' is in the current directory.")


st.markdown("<h1>🎖️ AI Soldier Voice Situation Detection</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload a soldier's voice recording to detect battlefield situation.</p>", unsafe_allow_html=True)

with st.sidebar:
    
    
    st.header("Upload Audio")
    uploaded_file = st.file_uploader("Choose a .wav audio file", type=["wav"])


if uploaded_file is not None:
    
    col1, col2 = st.columns([1, 2]) 

    with col1:
        st.subheader("Uploaded Audio")
        st.audio(uploaded_file, format="audio/wav")

    with col2:
        if kmeans is not None:
            st.subheader("Detection Result")
            
            try:
                features = extract_features(uploaded_file)
                features = features.reshape(1, -1)
                
                cluster = kmeans.predict(features)[0]
                situation_map = {0: "Normal", 1: "Stressed", 2: "Fatigued"}
                situation = situation_map.get(cluster, "Unknown")
                
                # --- Display results in the Maroon Card ---
                if situation == "Normal":
                    st.markdown(f"""
                    <div class="card" style="border-left: 5px solid #28a745;">
                        <h2>✅ Situation Detected: {situation}</h2>
                        <p>Calm, mission under control. All vitals stable.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif situation == "Stressed":
                    st.markdown(f"""
                    <div class="card" style="border-left: 5px solid #dc3545;">
                        <h2>⚠️ Situation Detected: {situation}</h2>
                        <p>Soldier may be under threat! Immediate attention required!</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif situation == "Fatigued":
                    st.markdown(f"""
                    <div class="card" style="border-left: 5px solid #ffc107;">
                        <h2>💤 Situation Detected: {situation}</h2>
                        <p>Soldier may need rest. Monitor operational readiness.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="card" style="border-left: 5px solid #6c757d;">
                        <h2>❌ Situation Detected: {situation}</h2>
                        <p>Unknown voice signature.</p>
                    </div>
                    """, unsafe_allow_html=True)
            except NameError:
                st.error("The `extract_features` function is missing.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
        else:
            st.info("Prediction is awaiting model loading.")