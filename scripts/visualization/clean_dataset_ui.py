import streamlit as st
import numpy as np
import pandas as pd
import json
import pickle
import shutil
import io
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dataset Cleaner", layout="wide")

# Sidebar: Configuration
st.sidebar.title("Dataset Configuration")

# --- Dataset Root Selection ---
import sys
import os

# Allow specifying root via CLI: streamlit run clean_dataset_ui.py -- --data_root /path/to/dataset
def get_cli_data_root():
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--data_root" and i + 1 < len(args):
            return Path(args[i + 1])
    env_root = os.environ.get("DATASET_ROOT")
    if env_root:
        return Path(env_root)
    return None

cli_root = get_cli_data_root()

if cli_root and cli_root.exists():
    base_root = cli_root
else:
    # Candidate roots to search
    candidates = [
        Path("Plaintextdataset"),
        Path("../Plaintextdataset"),
        Path("/home/martina/Y3_Project/Plaintextdataset"),
        Path("/home/martina/Y3_Project/TeleopDataset"),
        Path("."),
    ]
    base_root = next((p for p in candidates if p.exists()), Path("."))

# Allow user to override root in the sidebar
custom_root_input = st.sidebar.text_input("Dataset Root Path", value=str(base_root.resolve()))
if custom_root_input:
    candidate = Path(custom_root_input)
    if candidate.exists():
        base_root = candidate
    else:
        st.sidebar.warning(f"路径不存在: {custom_root_input}")

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


def build_signal_dataframe(
    time_values,
    selected_joints,
    data_getter,
    profile_key,
    separate_key=None,
    sample_step=1,
):
    df = pd.DataFrame({"Time (s)": time_values})
    for name in selected_joints:
        df[name] = data_getter(
            name,
            profile_key,
            separate_key=separate_key,
            sample_step=sample_step,
        )
    return df


def render_signal_chart(title, df, selected_joints, height=260):
    st.write(f"**{title}**")
    st.line_chart(df, x="Time (s)", y=selected_joints, height=height)


def generate_combined_waveform_png(episode_name, selected_joints, signal_specs):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=300, constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(selected_joints), 1)))

    for ax, (title, df, y_label) in zip(axes.flat, signal_specs):
        if df is None or df.empty or len(df.columns) <= 1:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center", fontsize=12)
            ax.set_title(title)
            ax.set_axis_off()
            continue

        x = df["Time (s)"].to_numpy()
        for idx, joint_name in enumerate(selected_joints):
            ax.plot(
                x,
                df[joint_name].to_numpy(),
                linewidth=1.5,
                color=colors[idx],
                label=joint_name,
            )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

    legend_handles = []
    legend_labels = []
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend_handles = handles
            legend_labels = labels
            break

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_labels), 4),
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
        )

    fig.suptitle(f"{episode_name} Sensor Waveforms", fontsize=18)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()

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
    if st.button("⬅️ Prev", width="stretch"):
        prev_episode()
        st.rerun()
        
with col_nav3:
    if st.button("Next ➡️", width="stretch"):
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
        st.image(str(anchor_path), width="stretch")
    else:
        st.info("No visual anchor image.")

with col_right:
    st.subheader("Metadata & Actions")
    
    # Delete Button with double confirmation to be safe or just simple button?
    # User asked for "one-click delete", but usually that implies "easy delete".
    # I will make it a primary button.
    if st.button("🗑️ DELETE THIS EPISODE", type="primary", width="stretch"):
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
st.subheader("� Sensor Waveforms")

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

            joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}

            def get_data_col(name, profile_key, separate_key=None, sample_step=1):
                sampled_timestamps = timestamps[::sample_step]
                if separate_key and name == "gripper_width (separate)":
                    raw_data = np.asarray(tactile.get(separate_key, []))
                    if raw_data.ndim == 1 and len(raw_data) == len(timestamps):
                        return raw_data[::sample_step]
                    return np.zeros(len(sampled_timestamps), dtype=float)

                idx = joint_name_to_idx.get(name)
                if idx is None:
                    return np.zeros(len(sampled_timestamps), dtype=float)

                data = np.asarray(tactile.get(profile_key, []))
                if data.ndim == 2 and len(data) == len(timestamps) and idx < data.shape[1]:
                    return data[::sample_step, idx]

                return np.zeros(len(sampled_timestamps), dtype=float)

            display_timestamps = timestamps[::step]
            export_timestamps = timestamps

            pos_df = build_signal_dataframe(
                display_timestamps,
                selected_joints,
                get_data_col,
                "joint_position_profile",
                separate_key="gripper_width_profile",
                sample_step=step,
            )
            load_df = build_signal_dataframe(
                display_timestamps,
                selected_joints,
                get_data_col,
                "joint_load_profile",
                separate_key="load_profile",
                sample_step=step,
            )
            vel_df = build_signal_dataframe(
                display_timestamps,
                selected_joints,
                get_data_col,
                "joint_velocity_profile",
                separate_key="gripper_velocity_profile",
                sample_step=step,
            )

            curr_df = None
            if "joint_current_profile" in tactile:
                curr_df = build_signal_dataframe(
                    display_timestamps,
                    selected_joints,
                    get_data_col,
                    "joint_current_profile",
                    sample_step=step,
                )

            row_1_col_1, row_1_col_2 = st.columns(2)
            row_2_col_1, row_2_col_2 = st.columns(2)

            with row_1_col_1:
                render_signal_chart("Position (Angle/Width)", pos_df, selected_joints)
            with row_1_col_2:
                render_signal_chart("Load", load_df, selected_joints)
            with row_2_col_1:
                render_signal_chart("Velocity", vel_df, selected_joints)
            with row_2_col_2:
                if curr_df is not None:
                    render_signal_chart("Current", curr_df, selected_joints)
                else:
                    st.write("**Current**")
                    st.info("Current data is not available for this episode.")

            export_signal_specs = [
                (
                    "Position (Angle/Width)",
                    build_signal_dataframe(
                        export_timestamps,
                        selected_joints,
                        get_data_col,
                        "joint_position_profile",
                        separate_key="gripper_width_profile",
                        sample_step=1,
                    ),
                    "Position / Width",
                ),
                (
                    "Load",
                    build_signal_dataframe(
                        export_timestamps,
                        selected_joints,
                        get_data_col,
                        "joint_load_profile",
                        separate_key="load_profile",
                        sample_step=1,
                    ),
                    "Load",
                ),
                (
                    "Velocity",
                    build_signal_dataframe(
                        export_timestamps,
                        selected_joints,
                        get_data_col,
                        "joint_velocity_profile",
                        separate_key="gripper_velocity_profile",
                        sample_step=1,
                    ),
                    "Velocity",
                ),
                (
                    "Current",
                    build_signal_dataframe(
                        export_timestamps,
                        selected_joints,
                        get_data_col,
                        "joint_current_profile",
                        sample_step=1,
                    ) if "joint_current_profile" in tactile else None,
                    "Current",
                ),
            ]

            export_png = generate_combined_waveform_png(
                selected_ep_name,
                selected_joints,
                export_signal_specs,
            )
            st.download_button(
                "Export 4-in-1 HD PNG",
                data=export_png,
                file_name=f"{selected_ep_name}_sensor_waveforms_hd.png",
                mime="image/png",
                width="stretch",
            )

    else:
        st.warning("Timestamps are empty.")
else:
    st.warning("Could not load tactile data (pkl file missing or corrupt).")
