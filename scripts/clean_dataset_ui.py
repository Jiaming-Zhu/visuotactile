import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dataset Cleaner", layout="wide")

# Sidebar: Configuration
st.sidebar.title("Dataset Configuration")

# --- Dataset Root Selection ---
base_root = Path("Plaintextdataset")
# Check if Plaintextdataset exists, otherwise try to find it relative to CWD or fallback
if not base_root.exists():
    # Try one level up if current dir is inside learn_PyBullet
    if Path("../Plaintextdataset").exists():
        base_root = Path("../Plaintextdataset")
    elif Path(".").exists():
        base_root = Path(".")

# List all directories in base_root
available_dirs = []
if base_root.exists():
    available_dirs = sorted([d.name for d in base_root.iterdir() if d.is_dir()])

# Default selection logic
default_idx = 0
if "WoodBlock_Native" in available_dirs:
    default_idx = available_dirs.index("WoodBlock_Native")

if not available_dirs:
    st.error(f"No subdirectories found in {base_root.resolve()}")
    st.stop()

selected_dir_name = st.sidebar.selectbox("Select Dataset Directory", available_dirs, index=default_idx)
dataset_root = base_root / selected_dir_name

if not dataset_root.exists():
    st.error(f"Path does not exist: {dataset_root}")
    st.stop()

import concurrent.futures

# Helper functions
MAX_POINTS = 1000

def downsample(df, target_points):
    if len(df) > target_points:
        # Simple decimation
        factor = len(df) // target_points
        return df.iloc[::factor, :]
    return df

@st.cache_data(ttl=3600)
def load_metadata(episode_dir_str):
    meta_path = Path(episode_dir_str) / "metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except:
            return {"error": "Failed to load metadata"}
    return {}

@st.cache_data(ttl=3600)
def load_tactile(episode_dir_str):
    pkl_path = Path(episode_dir_str) / "tactile_data.pkl"
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                if data:
                    for key in ["timestamps", "joint_position_profile", "joint_load_profile", "joint_velocity_profile", "gripper_width_profile", "load_profile", "gripper_velocity_profile", "joint_current_profile"]:
                        if key in data:
                            data[key] = np.array(data[key])
                return data
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

# Load Data (Single thread for stability and speed in simple local I/O)
metadata = load_metadata(str(selected_ep_path))
tactile = load_tactile(str(selected_ep_path))

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
            # Clear cache for THIS specific episode only, if possible, or just rely on reload
            # Clearing all cache is too heavy and causes lag
            # st.cache_data.clear() 
            
            # Adjust index to stay around the same spot
            if current_index >= len(episodes) - 1:
                 st.session_state.last_index = max(0, len(episodes) - 2)
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting: {e}")

    with st.expander("Show Metadata JSON", expanded=True):
        st.json(metadata)

st.divider()

# Waveforms
st.subheader("📈 Sensor Waveforms")

if tactile:
    timestamps = np.array(tactile.get("timestamps", []))
    if len(timestamps) > 0:
        timestamps = timestamps - timestamps[0]  # Normalize time

        # --- Joint Selection Logic ---
        # 1. Get joint names from metadata or fallback
        joint_names = metadata.get("joint_names", [])
        
        # If metadata doesn't have joint names, try to infer from data shape
        joint_pos_profile = tactile.get("joint_position_profile", [])
        if not joint_names and len(joint_pos_profile) > 0:
            num_joints = len(joint_pos_profile[0])
            joint_names = [f"Joint_{i}" for i in range(num_joints)]
        
        # Append 'gripper' if it's treated separately in tactile data (often it is)
        # But usually "joint_names" includes all motors.
        # Let's check if we have specific gripper data keys to offer as "virtual" joints if they aren't in the main list
        # Actually, the user wants to select multiple joints.
        
        # Let's standardize: we have "joint_names" which corresponds to columns in `joint_position_profile`.
        # We might also have "gripper_width_profile" which is a separate 1D array.
        
        available_signals = list(joint_names)
        # Check if gripper width is separate
        if "gripper_width_profile" in tactile and "gripper" not in available_signals:
             available_signals.append("gripper_width (separate)")

        # Default selection: Try to find 'gripper' or the last joint
        # We use session state to persist selection across episodes if possible,
        # otherwise default to gripper.
        
        default_selection = []
        
        # Restore previous selection if valid for this episode
        if "last_selected_joints" in st.session_state:
            prev_selected = st.session_state.last_selected_joints
            # Filter to keep only those that exist in current episode
            valid_prev = [j for j in prev_selected if j in available_signals]
            if valid_prev:
                default_selection = valid_prev
        
        # Fallback if no valid previous selection
        if not default_selection:
            for name in available_signals:
                if "gripper" in name.lower():
                    default_selection.append(name)
            if not default_selection and available_signals:
                default_selection = [available_signals[-1]]

        selected_joints = st.multiselect(
            "Select Joints to Visualize", 
            available_signals, 
            default=default_selection,
            key="joint_multiselect"
        )
        
        # Update session state with current selection
        st.session_state.last_selected_joints = selected_joints

        if not selected_joints:
            st.info("Please select at least one joint to view waveforms.")
        else:
            # Optimized data preparation: Slicing numpy arrays directly BEFORE creating DataFrames
            # Calculate step for downsampling
            step = max(1, len(timestamps) // MAX_POINTS)
            
            # Downsample timestamps
            small_timestamps = timestamps[::step]
            
            # Helper to extract and downsample data column
            def get_data_col(name, profile_key, separate_key=None):
                # If it's the separate gripper key
                if separate_key and name == "gripper_width (separate)":
                    raw_data = tactile.get(separate_key, [])
                    if len(raw_data) == len(timestamps):
                        return raw_data[::step]
                    return np.zeros(len(small_timestamps))
                
                # Otherwise find index in joint_names
                if name in joint_names:
                    idx = joint_names.index(name)
                    data = tactile.get(profile_key, [])
                    if len(data) > 0 and len(data) == len(timestamps):
                        # data is already numpy array thanks to load_tactile optimization
                        # But it might be 2D (N, joints) or list of lists if load failed to convert
                        # We ensure it's numpy in load_tactile, but safe check:
                        if isinstance(data, np.ndarray) and data.ndim == 2:
                             if idx < data.shape[1]:
                                 return data[::step, idx]
                        # Fallback for list of lists (shouldn't happen with optimized load_tactile)
                        else:
                            return [data[i][idx] for i in range(0, len(data), step)]
                            
                return np.zeros(len(small_timestamps))

            # 1. Position Plot
            st.write("**Position (Angle/Width)**")
            pos_df = pd.DataFrame({"Time (s)": small_timestamps})
            for name in selected_joints:
                col_data = get_data_col(name, "joint_position_profile", "gripper_width_profile")
                pos_df[name] = col_data
            
            st.line_chart(pos_df, x="Time (s)", y=selected_joints, height=300)

            col_charts_1, col_charts_2 = st.columns(2)

            # 2. Load Plot
            with col_charts_1:
                st.write("**Load**")
                load_df = pd.DataFrame({"Time (s)": small_timestamps})
                for name in selected_joints:
                    col_data = get_data_col(name, "joint_load_profile", "load_profile")
                    load_df[name] = col_data
                st.line_chart(load_df, x="Time (s)", y=selected_joints, height=250)

            # 3. Velocity Plot
            with col_charts_2:
                st.write("**Velocity**")
                vel_df = pd.DataFrame({"Time (s)": small_timestamps})
                for name in selected_joints:
                    col_data = get_data_col(name, "joint_velocity_profile", "gripper_velocity_profile")
                    vel_df[name] = col_data
                st.line_chart(vel_df, x="Time (s)", y=selected_joints, height=250)

            # 4. Current Plot (if available)
            if "joint_current_profile" in tactile:
                 with st.expander("Show Current (Amps)"):
                    curr_df = pd.DataFrame({"Time (s)": small_timestamps})
                    for name in selected_joints:
                        col_data = get_data_col(name, "joint_current_profile", None)
                        curr_df[name] = col_data
                    st.line_chart(curr_df, x="Time (s)", y=selected_joints, height=250)

    else:
        st.warning("Timestamps are empty.")
else:
    st.warning("Could not load tactile data (pkl file missing or corrupt).")
