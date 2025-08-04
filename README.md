# 🚴‍♂️ Trail Surface Defect Dashboard

This Streamlit application enables visualization, comparison, and analysis of **trail surface defects before and after maintenance** using object detection outputs and GPS traces.

---

## 📂 Project Structure

```
├── app_dashboard.py                            # Streamlit dashboard
├── extract_app.py                              # Script to extract frames from videos using accelerometer and GPS
├── predict_app.py                              # YOLO-based prediction script for extracted frames
├── requirements.txt                            # Required Python packages
├── best_Yolo8.pt                               # Trained YOLOv8 model
├── Jordan_Holm/
│   ├── Jordan_front_HERO13 Black-GPS9.csv      # GPS trace for Jordan to Holm section
│   ├── Jordan_Holm_predictions.csv             # Detection results
│   └── Jordan_Holm_event_frames/               # Directory containing extracted images
```

---

## 🚀 Getting Started

A live demo of the dashboard is available at:  
🔗 [https://databike-dashboard-demo.streamlit.app/](https://databike-dashboard-demo.streamlit.app/)

This interactive platform enables users to compare surface condition **before and after maintenance**, leveraging object detection and GPS trace alignment.

---

## 🧰 Step-by-Step Workflow

### 📁 Step 0: Clone Repository and Install Dependencies

Choose a directory on your local machine and clone the repository:

```bash
git clone https://github.com/hereasmadhu/data_bike.git
cd data_bike
```

This repository contains **demo data and videos** for testing the full pipeline.

Install the required packages:

```bash
pip install -r requirements.txt
```

---

### 🎞️ Step 1: Extract Frames from Video

Use the `extract_app.py` script to extract event-based frames from a video file using accelerometer and GPS metadata:

```bash
python extract_app.py --video path/to/video.mp4 --gps path/to/gps.csv --accel path/to/accel.csv --outdir output_folder
```

This will generate a directory of frames corresponding to significant trail events.

---

### 🧠 Step 2: Run Object Detection on Extracted Frames

Apply the YOLOv8 model (`best_Yolo8.pt`) to the extracted frames using `predict_app.py`:

```bash
python predict_app.py --model best_Yolo8.pt --imgdir output_folder --gps path/to/gps.csv --out path/to/output_predictions.csv
```

This step will create a CSV file with bounding box coordinates, class predictions, and geolocations.

---

### 📊 Step 3: Launch Dashboard

With the predicted CSV and GPS trace, launch the Streamlit dashboard:

```bash
streamlit run app_dashboard.py
```

Or visit the hosted dashboard (if available).

---

## 📘 How to Use the Dashboard

### ⬅️ Upload Required Files

In the **left sidebar**, upload the following four files:

- **Route CSV (Before)** – GPS trace before maintenance  
- **Issues CSV (Before)** – Detection results before maintenance  
- **Route CSV (After)** – GPS trace after maintenance  
- **Issues CSV (After)** – Detection results after maintenance  

> 📌 *Note: For demo purposes, you may use the same GPS and detection files for both before and after cases.*

---

### ⚙️ Configure Column Names

After upload, select appropriate column names (e.g., `latitude`, `longitude`, `confidence`) for both route and detection files.

---

### 🖼️ Specify Image Directory Path

Enter the relative path to the directory containing the images listed in the `frame_filename` column. Example:

```
Jordan_Holm/Jordan_Holm_event_frames
```

Avoid using Windows-style paths like `D:\...` or local paths like `./...`.

---

### 🚀 Run Analysis

Click **“Run Comparison”** to load visualizations.

---

## 🗺️ Dashboard Features

- **Comparison View**: Side-by-side maps showing defect distribution before vs after maintenance  
- **Image Gallery**: Scrollable and clickable annotated images  
- **Detailed View**: Marker-level metadata and image inspection

---

## 📬 Contact

For questions, suggestions, or feedback, please contact:

- [Madhu.M.Thapa@utah.edu](mailto:Madhu.M.Thapa@utah.edu)  
- [sanjay.luitel@utah.edu](mailto:sanjay.luitel@utah.edu)  
- [abbas.rashidi@utah.edu](mailto:abbas.rashidi@utah.edu)  
- [nikola.markovic@utah.edu](mailto:nikola.markovic@utah.edu)
