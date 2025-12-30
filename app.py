import streamlit as st
import pandas as pd
import os
from data_loader import DataLoader
from classifier import TicketClassifier

st.set_page_config(page_title="LLM Ticket Tagger", layout="wide")
st.title("🎯 Support Ticket Auto-Tagging")

# --- Configuration & API Logic ---

with st.sidebar:
    st.header(" Settings")
    input_key = st.text_input("Groq API Key (Optional if set in Env)", type="password")
    mode = st.radio("Select Learning Mode", ["Zero-Shot", "Few-Shot"])
    
    # Use the input key if provided, otherwise check the environment
    api_key = input_key if input_key else os.environ.get("GROQ_API_KEY")

if not api_key:
    st.warning("🔑 Please enter a Groq API Key or set GROQ_API_KEY in your environment.")
    st.stop()

# Set environment variable so the classifier can find it
os.environ["GROQ_API_KEY"] = api_key
classifier = TicketClassifier()

# --- Upload and Load ---
uploaded_file = st.file_uploader("Upload Ticket CSV", type=["csv"])

if uploaded_file:
    df = DataLoader(path=uploaded_file).load_data()
    st.info(f"📁 Loaded {len(df)} tickets.")

    # 🎚️ ADDED: Limit slider to prevent long wait times
    max_batch = min(len(df), 50) 
    num_to_process = st.slider("Number of tickets to classify", 1, max_batch, 5)

    if st.button(f"🚀 Run {mode} Classification"):
        # Subset the data based on slider
        subset = df.head(num_to_process).copy()
        
        results = []
        progress_bar = st.progress(0)
        
        for i, text in enumerate(subset['ticket_text']):
            res = classifier.classify(text, mode=mode.lower())
            
            tags = res.get("tags", ["", "", ""])
            results.append({
                "Primary Tag": tags[0] if len(tags) > 0 else "N/A",
                "Secondary Tag": tags[1] if len(tags) > 1 else "N/A",
                "Tertiary Tag": tags[2] if len(tags) > 2 else "N/A",
                "Justification": res.get("justification", "No justification provided")
            })
            progress_bar.progress((i + 1) / num_to_process)
        
        # Combine results with the subset
        res_df = pd.concat([subset.reset_index(drop=True), pd.DataFrame(results)], axis=1)
        st.success(f"✅ Finished processing {num_to_process} tickets in {mode} mode!")
        st.dataframe(res_df)

# --- Single Ticket Comparison ---
st.divider()
st.subheader("🔍 Compare Methods (Single Ticket)")
test_text = st.text_area("Enter a sample ticket to see the difference:")

if test_text and st.button(" Compare Both"):
    col1, col2 = st.columns(2)
    with col1:
        st.info("Zero-Shot (Generic)")
        st.json(classifier.classify(test_text, mode="zero-shot"))
    with col2:
        st.success("Few-Shot (With Examples)")
        st.json(classifier.classify(test_text, mode="few-shot"))