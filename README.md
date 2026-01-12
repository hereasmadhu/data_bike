# 🚴‍♂️ Trail Surface Defect Dashboard

This Streamlit application enables visualization, comparison, and analysis of **trail surface defects before and after maintenance** using object detection outputs and GPS traces.

---

## 📂 Project Structure

```
├── app_dashboard.py                            # Streamlit dashboard
├── extract_app.py                              # Script to extract frames using accelerometer and GPS
├── predict_app.py                              # YOLOv8 prediction script for extracted frames
├── requirements.txt                            # Required Python packages
├── best_Yolo8.pt                               # Trained YOLOv8 model
├── Jordan_Holm/
│   ├── Jordan_front_HERO13 Black-GPS9.csv      # GPS trace file (demo)
│   ├── Jordan_front_HERO13 Black-ACCL.csv      # Accelerometer file (demo)
│   ├── demo_video.mp4                          # Placeholder video file (optional)
│   ├── Jordan_Holm_event_frames/               # Directory for extracted images
│   └── Jordan_Holm_predictions.csv             # Detection results (CSV)
```

---

## 🚀 Getting Started

Find the visual demonstration of using this pipeline here: https://youtu.be/A8lZ2UlACw4

Try the live dashboard at:  
🔗 [https://databike-dashboard-demo.streamlit.app/](https://databike-dashboard-demo.streamlit.app/)

Or follow the instructions below to run locally.

---

## 🧰 Step-by-Step Workflow

### 📁 Step 0: Clone the Repository and Install Dependencies

```bash
git clone https://github.com/hereasmadhu/data_bike.git
cd data_bike
pip install -r requirements.txt
```

This repository includes demo files (GPS, accelerometer, optional video) for testing the full pipeline.

---

### 🎞️ Step 1: Extract Frames from Demo Video

Use `extract_app.py` to extract frames based on vibration events:

```bash
python extract_app.py ^
  --video_file "D:\DataBikeProject\Videos Collected\Jordan_Holm\Jordan_front.MP4" ^
  --gps_csv "Jordan_Holm/Jordan_front_HERO13 Black-GPS9.csv" ^
  --accl_csv "Jordan_Holm/Jordan_front_HERO13 Black-ACCL.csv" ^
  --output_dir "Jordan_Holm/Jordan_Holm_event_frames" ^
  --trail_name "Jordan_Holm"
```
> 📌 For powershell, you should use ` instead of ^., The video file location here provided is local one, so it is just for demo, for now extraction process is already completed, and results are in Jordan_Holm folder.

This creates frame images at locations with notable vibration, based on accelerometer and GPS data.

> ⚠️ If you don’t have the video file, you can skip this step and use pre-generated images in `Jordan_Holm_event_frames`.

---

### 🧠 Step 2: Run Object Detection on Extracted Frames

Apply the pretrained YOLOv8 model to detect surface defects:

```bash
python predict_app.py ^
  --frames_dir "Jordan_Holm/Jordan_Holm_event_frames" ^
  --events_csv "Jordan_Holm/Jordan_Holm_frames_accl_events_with_gps.csv" ^
  --model_path "./best_Yolov8.pt" ^
  --output_csv "Jordan_Holm/Jordan_Holm_predictions.csv"
```

This generates a CSV of defects with bounding boxes, classes, and GPS positions.

---

### 📊 Step 3: Launch Streamlit Dashboard

Run the Streamlit dashboard locally:

```bash
streamlit run app_dashboard.py
```

Then open the interface in your browser to interact with the dashboard.

---

## 📘 How to Use the Dashboard

### ⬅️ Upload Files in Sidebar

Use the **left sidebar** to upload the following four CSV files:

- **Route CSV (Before)** – e.g., `Jordan_Holm/Jordan_front_HERO13 Black-GPS9.csv`
- **Issues CSV (Before)** – e.g., `Jordan_Holm/Jordan_Holm_predictions.csv`
- **Route CSV (After)** – (use same file for demo)
- **Issues CSV (After)** – (use same file for demo)

> 📌 For demo, you may upload the same GPS and prediction files for both "before" and "after" inputs.

---

### ⚙️ Select Column Names

After uploading, select the appropriate column names such as `latitude`, `longitude`, `confidence`.

The app automatically detects common column names like `lat`, `lon`, etc.

---

### 🖼️ Specify Image Folder

Provide the relative path to the folder with extracted frames, such as:

```
Jordan_Holm/Jordan_Holm_event_frames
```

Do **not** use Windows-style paths (like `D:\...`) or relative symbols like `./`.

---

### 🚀 Run the Comparison

Click the **"Run Comparison"** button to visualize:

- Side-by-side trail maps
- Annotated defect images
- Location-aware defect details

---

## 🗺️ Dashboard Features

- **Comparison View**: Map view of defects (before vs. after)
- **Image Gallery**: Clickable annotated image scroll
- **Defect Details**: Inspect location, confidence, and class metadata

All maps are rendered using `folium` and support interactive exploration.

---

## 📬 Contact

For questions or collaboration, please contact:

- [Madhu.M.Thapa@utah.edu](mailto:Madhu.M.Thapa@utah.edu)  
- [sanjay.luitel@utah.edu](mailto:sanjay.luitel@utah.edu)  
- [abbas.rashidi@utah.edu](mailto:abbas.rashidi@utah.edu)  
- [nikola.markovic@utah.edu](mailto:nikola.markovic@utah.edu)
