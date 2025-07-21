# 🚴‍♂️ Trail Surface Defect Dashboard

This Streamlit application allows you to visualize, compare, and analyze surface defects along trails **before and after maintenance**, using object detection results and GPS traces.

## 📂 Project Structure

```
├── app_dashboard.py                            # Streamlit dashboard
├── extract_app.py                              # (Optional) Script to extract frames from videos using accelerometer and GPS
├── predict_app.py                              # (Optional) YOLO-based prediction script
├── requirements.txt                            # Required Python packages
├── best_Yolo8.pt                               # Trained Yolo model
├── Jordan_Holm/
│   ├── Jordan_front_HERO13 Black-GPS9.csv      # GPS trace for Joran to Holm section
│   ├── Jordan_Holm_predictions.csv             # Detection results
│   └── Jordan_Holm_event_frames/               # Directory containing extracted images
```

---

## 🚀 Getting Started

Access the live dashboard at: https://databike-dashboard-demo.streamlit.app/

This interactive Streamlit dashboard allows users to visualize, compare, and analyze surface defects along trails before and after maintenance, using object detection results and GPS traces.

---

## 📘 How to Use the Dashboard

### ⬅️ Step 1: Upload Required Files
Use the **left sidebar** to upload the following four CSV files:

- **Route CSV (Before)** – GPS trace before maintenance
- **Issues CSV (Before)** – Detection results before maintenance
- **Route CSV (After)** – GPS trace after maintenance
- **Issues CSV (After)** – Detection results after maintenance

***For now please use same sets of GPS trace, and predictions fr both before and after case.***

Make sure your CSVs contain the necessary columns:
- For route files: columns like `latitude`, `longitude`, and `timestamp`
- For issues files: `frame_filename`, `class_id`, `confidence`, `latitude`, `longitude`, and either `bbox` or individual box columns `x1`, `y1`, `x2`, `y2`

---

### 🗂️ Step 2: Configure Column Names
After uploading, select the correct **latitude**, **longitude** and **confidence** column names for both before and after route files. The dashboard will try to guess common names like `lat`, `lon`, `latitude`, `longitude`, and `confidence`.

---

### 🖼️ Step 3: Specify Image Directory Path
Provide the relative path to the directory that contains the images referenced in the `frame_filename` column.

For example, if your image folder is named `Jordan_Holm_event_frames` and it is part of your uploaded or deployed repository, you should type:
```
\Jordan_Holm\Jordan_Holm_event_frames
```

Avoid using Windows-style paths like `D:\...` or local paths like `./...`.

---

### 🚀 Step 4: Run the Analysis
Click the **"Run Comparison"** button in the sidebar to launch the dashboard view.

---

## 🗺️ Dashboard Features

- **Comparison View**: Side-by-side maps comparing defect occurrences before and after maintenance.
- **Image Gallery**: Scrollable view of annotated images.
- **Detailed View**: Click a marker or image to inspect detections, metadata, and location.

---

For questions or feedback, please contact the us at:
Madhu.M.Thapa@utah.edu
sanjay.luitel@utah.edu
abbas.rashidi@utah.edu
nikola.markovic@utah.edu
