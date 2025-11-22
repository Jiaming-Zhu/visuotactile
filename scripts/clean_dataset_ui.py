import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="WoodBlock Dataset Cleaner", layout="wide")

# Sidebar: Configuration
st.sidebar.title("Dataset Configuration")
# Default path based on user's context
default_path = "Plaintextdataset/WoodBlock_Native"
dataset_root_str = st.sidebar.text_input("Dataset Root", default_path)
dataset_root = Path(dataset_root_str)

if not dataset_root.exists():
    st.error(f"Path does not exist: {dataset_root}")
    st.stop()

# Helper functions
def load_metadata(episode_dir):
    meta_path = episode_dir / "metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except:
            return {"error": "Failed to load metadata"}
    return {}

def load_tactile(episode_dir):
    pkl_path = episode_dir / "tactile_data.pkl"
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except:
            return None
    return None

def get_episodes(root):
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_")], key=lambda x: x.name)

# Load episodes
episodes = get_episodes(dataset_root)
st.sidebar.write(f"Found {len(episodes)} episodes.")

if len(episodes) == 0:
    st.warning("No episodes found in this directory.")
    st.stop()

# Selection
episode_options = [ep.name for ep in episodes]
# Use session state to keep track of index if needed, but simple selectbox is usually enough
# We try to keep the index stable if possible
index = 0
if "last_index" in st.session_state:
    if st.session_state.last_index < len(episode_options):
        index = st.session_state.last_index

selected_ep_name = st.sidebar.selectbox("Select Episode", episode_options, index=index)
selected_ep_path = dataset_root / selected_ep_name

# Update session state for next time (if we don't delete)
current_index = episode_options.index(selected_ep_name)
st.session_state.last_index = current_index

# Navigation Buttons
col_nav1, col_nav2, col_nav3 = st.sidebar.columns([1, 1, 1])

def prev_episode():
    new_index = max(0, current_index - 1)
    st.session_state.last_index = new_index

def next_episode():
    new_index = min(len(episodes) - 1, current_index + 1)
    st.session_state.last_index = new_index

with col_nav1:
    if st.button("⬅️ Prev", use_container_width=True):
        prev_episode()
        st.rerun()
        
with col_nav3:
    if st.button("Next ➡️", use_container_width=True):
        next_episode()
        st.rerun()

st.title(f"🔍 {selected_ep_name}")
st.caption(f"Path: {selected_ep_path} | Index: {current_index + 1}/{len(episodes)}")

# Load Data
metadata = load_metadata(selected_ep_path)
tactile = load_tactile(selected_ep_path)
anchor_path = selected_ep_path / "visual_anchor.jpg"

# Layout: Top Row with Anchor and Metadata/Actions
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Visual Anchor")
    if anchor_path.exists():
        st.image(str(anchor_path), use_container_width=True)
    else:
        st.info("No visual anchor image.")

with col_right:
    st.subheader("Metadata & Actions")
    
    # Delete Button with double confirmation to be safe or just simple button?
    # User asked for "one-click delete", but usually that implies "easy delete".
    # I will make it a primary button.
    if st.button("🗑️ DELETE THIS EPISODE", type="primary", use_container_width=True):
        try:
            shutil.rmtree(selected_ep_path)
            st.toast(f"Deleted {selected_ep_name}")
            # Adjust index to stay around the same spot
            if current_index >= len(episodes) - 1:
                 st.session_state.last_index = max(0, len(episodes) - 2)
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting: {e}")

    with st.expander("Show Metadata JSON", expanded=True):
        st.json(metadata)

st.divider()

# Gripper Waveforms
st.subheader("📈 Gripper Waveforms")

if tactile:
    timestamps = np.array(tactile.get("timestamps", []))
    if len(timestamps) > 0:
        # Normalize time
        timestamps = timestamps - timestamps[0]
        
        # 1. Gripper Width
        st.write("**Gripper Width (Position)**")
        width_data = pd.DataFrame({
            "Time (s)": timestamps,
            "Width": tactile.get("gripper_width_profile", [0]*len(timestamps))
        })
        st.line_chart(width_data, x="Time (s)", y="Width", height=200, color="#29b5e8")

        col_charts_1, col_charts_2 = st.columns(2)
        
        # 2. Gripper Load
        with col_charts_1:
            st.write("**Gripper Load**")
            load_data = pd.DataFrame({
                "Time (s)": timestamps,
                "Load": tactile.get("load_profile", [0]*len(timestamps))
            })
            st.line_chart(load_data, x="Time (s)", y="Load", height=200, color="#ff8c00")
            
        # 3. Gripper Velocity
        with col_charts_2:
            st.write("**Gripper Velocity**")
            vel_data = pd.DataFrame({
                "Time (s)": timestamps,
                "Velocity": tactile.get("gripper_velocity_profile", [0]*len(timestamps))
            })
            st.line_chart(vel_data, x="Time (s)", y="Velocity", height=200, color="#4caf50")

    else:
        st.warning("Timestamps are empty.")
else:
    st.warning("Could not load tactile data (pkl file missing or corrupt).")

