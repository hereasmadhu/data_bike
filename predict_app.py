#!/usr/bin/env python
import os
import glob
import pandas as pd
from ultralytics import YOLO
import argparse
from tqdm import tqdm

def main(frames_dir, events_csv, model_path, output_csv, conf_threshold):
    """
    Main function to run YOLO object detection on extracted frames and
    correlate detections with GPS data.
    """
    # 1. Load YOLO model
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    # 2. Load and prepare events data with GPS
    print(f"Loading events data from: {events_csv}")
    try:
        df_evt = pd.read_csv(events_csv)
        # Create a 1-based event_index to match filenames
        df_evt = df_evt.reset_index().rename(columns={'index': 'event_idx0'})
        df_evt['event_index'] = df_evt['event_idx0'] + 1
        df_evt.set_index('event_index', inplace=True)
    except FileNotFoundError:
        print(f"Error: Events CSV file not found at {events_csv}")
        return

    # 3. Predict on each frame
    image_paths = glob.glob(os.path.join(frames_dir, '*.jpg'))
    if not image_paths:
        print(f"Warning: No .jpg frames found in the specified directory: {frames_dir}")
        return

    print(f"\nFound {len(image_paths)} frames. Starting object detection...")
    records = []
    
    # Wrap the loop with tqdm for a progress bar
    for img_path in tqdm(image_paths, desc="Processing Frames"):
        fname = os.path.basename(img_path)
        
        # Parse event index from filename (assumes "event_XXX_..." pattern)
        try:
            evt_idx = int(fname.split('_')[1])
        except (IndexError, ValueError):
            print(f"⚠️ Skipping unrecognized filename format: {fname}")
            continue

        if evt_idx not in df_evt.index:
            print(f"⚠️ No GPS data for event {evt_idx}, skipping frame {fname}")
            continue

        row = df_evt.loc[evt_idx]
        lat = row['latitude']
        lon = row['longitude']
        jerk_time = row['jerk_time_s']
        gps_time = row['gps_time_s']

        # Run YOLO prediction
        results = model.predict(source=img_path,
                                conf=conf_threshold,
                                verbose=False)
        
        # Process and record detections
        for res in results:
            for box in res.boxes:
                records.append({
                    'frame_filename': fname,
                    'event_index':    evt_idx,
                    'jerk_time_s':    jerk_time,
                    'gps_time_s':     gps_time,
                    'latitude':       lat,
                    'longitude':      lon,
                    'class_id':       int(box.cls[0]),
                    'confidence':     float(box.conf[0]),
                    'x1':             float(box.xyxy[0][0]),
                    'y1':             float(box.xyxy[0][1]),
                    'x2':             float(box.xyxy[0][2]),
                    'y2':             float(box.xyxy[0][3]),
                })

    # 4. Save all detections to the output CSV
    if not records:
        print("\nNo objects detected in any of the frames.")
        return
        
    df_pred = pd.DataFrame.from_records(records)
    df_pred.to_csv(output_csv, index=False)
    print(f"\n✅ Success! Saved {len(df_pred)} detections to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO object detection on video frames and save results with GPS data.")
    
    parser.add_argument("--frames_dir", required=True, help="Full path to the directory containing extracted .jpg frames.")
    parser.add_argument("--events_csv", required=True, help="Full path to the CSV file mapping events to GPS coordinates.")
    parser.add_argument("--model_path", required=True, help="Full path to the trained YOLO model file (e.g., best.pt).")
    parser.add_argument("--output_csv", required=True, help="Full path for the output CSV file where predictions will be saved.")
    parser.add_argument("--conf_threshold", type=float, default=0.25, help="Confidence threshold for object detection (default: 0.25).")
    
    args = parser.parse_args()
    
    main(args.frames_dir, args.events_csv, args.model_path, args.output_csv, args.conf_threshold)