import os
import argparse
import numpy as np
import pandas as pd
import scipy.signal as signal
import cv2
import shutil
from tqdm import tqdm  # <-- Added this import

# --- Preprocessing functions ---
def compute_magnitude(x, y, z):
    """Computes the magnitude of a 3D vector."""
    return np.sqrt(x**2 + y**2 + z**2)

def low_pass_filter(data, cutoff_freq, fs, order=4):
    """Applies a low-pass filter to the data."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)

def compute_derivative(data, timestamps):
    """Computes the derivative of the data with respect to timestamps."""
    return np.diff(data) / np.diff(timestamps)

def detect_peaks(data, height=None, distance=None):
    """Detects positive and negative peaks in the data."""
    pos, _ = signal.find_peaks(data, height=height, distance=distance)
    neg, _ = signal.find_peaks(-data, height=height, distance=distance)
    return pos, neg

def find_nearest_idx(arr, val):
    """Finds the index of the nearest value in an array."""
    return np.abs(arr - val).argmin()

def main(video_file, accl_csv, gps_csv, output_dir, trail_name):
    """
    Main function to process accelerometer and GPS data, detect events,
    and extract corresponding video frames.
    """
    # --- Create a descriptive output filename ---
    output_filename = f"{trail_name}_frames_accl_events_with_gps.csv"
    output_mapping_csv = os.path.join(output_dir, output_filename)

    # --- Accelerometer CSV Column Names ---
    accl_timestamp_col_orig = 'cts'
    accl_x_col_index = 2
    accl_y_col_index = 3
    accl_z_col_index = 4

    # --- GPS CSV Column Names ---
    gps_timestamp_col = 'cts'
    latitude_col = 'GPS (Lat.) [deg]'
    longitude_col = 'GPS (Long.) [deg]'

    # --- Processing Parameters ---
    default_fs = 50.0
    cutoff_freq = 0.5
    filter_order = 4
    peak_height = 2.5
    peak_distance = 20

    # 1. Load and process accelerometer data
    print("Processing accelerometer data...")
    df_accl = pd.read_csv(accl_csv)
    x = pd.to_numeric(df_accl.iloc[:, accl_x_col_index], errors='coerce').to_numpy()
    y = pd.to_numeric(df_accl.iloc[:, accl_y_col_index], errors='coerce').to_numpy()
    z = pd.to_numeric(df_accl.iloc[:, accl_z_col_index], errors='coerce').to_numpy()
    ts_s = pd.to_numeric(df_accl[accl_timestamp_col_orig], errors='coerce').to_numpy() / 1000.0

    mask = (~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z) & ~np.isnan(ts_s))
    x, y, z, ts_s = x[mask], y[mask], z[mask], ts_s[mask]

    diffs = np.diff(ts_s)
    fs_est = 1.0 / diffs[diffs > 0].mean() if np.count_nonzero(diffs > 0) else default_fs
    print(f"Estimated sampling frequency: {fs_est:.2f} Hz")

    mag = compute_magnitude(x, y, z)
    fmag = low_pass_filter(mag, cutoff_freq, fs_est, order=filter_order)
    jerk = compute_derivative(fmag, ts_s)
    jerk_ts = (ts_s[:-1] + ts_s[1:]) / 2

    # 2. Detect peaks
    pos_peaks, _ = detect_peaks(jerk, height=peak_height, distance=peak_distance)
    print(f"Detected {len(pos_peaks)} positive jerk events.")

    # 3. Load GPS data and map events
    print("Mapping accelerometer events to GPS data...")
    required_gps_cols = [gps_timestamp_col, latitude_col, longitude_col]
    df_gps = (pd.read_csv(gps_csv, usecols=required_gps_cols, engine='python')
              .dropna()
              .sort_values(gps_timestamp_col)
              .reset_index(drop=True))
    
    gps_times = pd.to_numeric(df_gps[gps_timestamp_col], errors='coerce').to_numpy() / 1000.0

    events = []
    for idx in pos_peaks:
        t_evt = jerk_ts[idx]
        gps_i = find_nearest_idx(gps_times, t_evt)
        events.append({
            'jerk_time_s': t_evt,
            'jerk_value': jerk[idx],
            'gps_time_s': gps_times[gps_i],
            'latitude': df_gps.loc[gps_i, latitude_col],
            'longitude': df_gps.loc[gps_i, longitude_col]
        })

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(events).to_csv(output_mapping_csv, index=False)
    print(f"Saved {len(events)} events to {output_mapping_csv}")

    # 4. Extract frames from video
    print(f"\nExtracting frames from {os.path.basename(video_file)}...")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps:.2f}")
    offsets = [0.25, 0.5, 0.75]

    frame_folder = os.path.join(output_dir, f"{trail_name}_event_frames")
    os.makedirs(frame_folder, exist_ok=True)

    # --- MODIFICATION HERE: Wrap the main loop with tqdm for a progress bar ---
    for i, evt in tqdm(enumerate(events, 1), total=len(events), desc="Extracting Frames"):
        t_evt = evt['jerk_time_s']
        for off in offsets:
            t_request = max(0.0, t_evt - off)
            cap.set(cv2.CAP_PROP_POS_MSEC, t_request * 1000.0)
            ret, frame = cap.read()
            if not ret:
                # This warning can be noisy, so it's commented out by default
                # print(f"Warning: could not read frame at {t_request:.3f}s")
                continue

            fn = f"event_{i:03d}_at_{t_evt:.2f}s_minus_{off:.2f}s.jpg"
            out_path = os.path.join(frame_folder, fn)
            cv2.imwrite(out_path, frame)
            # --- REMOVED: Repetitive print statement to keep the progress bar clean ---

    print(f"\nSuccessfully extracted frames for {len(events)} events to '{frame_folder}'")

    # 5. Copy the original GPS file to the output directory
    print("\nCopying original GPS file...")
    shutil.copy(gps_csv, output_dir)
    gps_filename = os.path.basename(gps_csv)
    print(f"Successfully copied {gps_filename} to '{output_dir}'")

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video frames based on accelerometer data.")
    parser.add_argument("--video_file", required=True, help="Path to the video file.")
    parser.add_argument("--accl_csv", required=True, help="Path to the accelerometer CSV file.")
    parser.add_argument("--gps_csv", required=True, help="Path to the GPS CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output files.")
    parser.add_argument("--trail_name", required=True, help="Name of the trail for the output filename.")

    args = parser.parse_args()
    main(args.video_file, args.accl_csv, args.gps_csv, args.output_dir, args.trail_name)