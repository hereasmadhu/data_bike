# app_dashboard.py
# ENHANCED & FIXED VERSION: Corrected session state, function calls, and data passing.

import os
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from PIL import Image, ImageDraw, ImageFont
import math
import io
import json

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Trail Comparison Dashboard")

# Class names and default colors
LABEL_NAMES = [
    "pothole", "longitudinal_crack", "transverse_crack", "alligator_crack",
    "crack_other", "misalignment", "vegetation_obstacle", "pole_obstacle",
    "debris_obstacle", "tactile_paving", "rail_crossing"
]
DEFAULT_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080"
]
IMAGES_PER_PAGE = 9
THUMBNAILS_PER_PAGE = 7


# --- Data Loading ---
@st.cache_data
def load_data(route_file_content, issues_file_content, confidence_col_name="confidence"):
    """
    Loads route and issues data from in-memory file content.
    """
    if not route_file_content or not issues_file_content:
        return pd.DataFrame(), pd.DataFrame()

    try:
        # IMPROVED: Use io.BytesIO to read content from memory
        df_route = pd.read_csv(io.BytesIO(route_file_content), engine='python', sep=',', header=0, on_bad_lines='skip')
        df_issues = pd.read_csv(io.BytesIO(issues_file_content), engine='python', sep=',', header=0, on_bad_lines='skip')

        df_route.columns = df_route.columns.str.strip()
        df_issues.columns = df_issues.columns.str.strip()

        # IMPROVED: More robust column validation
        required_issue_cols = {confidence_col_name, "class_id", "frame_filename", "latitude", "longitude"}
        if not required_issue_cols.issubset(df_issues.columns):
            st.error(f"Fatal: An issues CSV is missing one or more required columns. Needed: {required_issue_cols}. Found: {df_issues.columns.tolist()}")
            return pd.DataFrame(), pd.DataFrame()

        df_issues.rename(columns={confidence_col_name: 'confidence'}, inplace=True)
        df_issues["label"] = df_issues["class_id"].astype(int).map(lambda i: LABEL_NAMES[i] if 0 <= i < len(LABEL_NAMES) else "unknown")
        
        return df_route, df_issues
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()


# --- Image Processing Functions ---
def parse_bounding_box(bbox_str):
    """Parse bounding box string into coordinates."""
    try:
        if pd.isna(bbox_str) or bbox_str == "":
            return None
        # Handle different formats: "x1,y1,x2,y2" or "[x1,y1,x2,y2]" or JSON
        bbox_str = str(bbox_str).strip()
        if bbox_str.startswith('[') and bbox_str.endswith(']'):
            bbox = json.loads(bbox_str)
        else:
            bbox = [float(x.strip()) for x in bbox_str.split(',')]

        if len(bbox) == 4:
            return bbox  # [x1, y1, x2, y2]
        return None
    except:
        return None

def draw_bounding_box(image_path, bbox, label, confidence, color):
    """Draw bounding box on image and return the modified image."""
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, min(x1, img.width)), max(0, min(y1, img.height))
            x2, y2 = max(0, min(x2, img.width)), max(0, min(y2, img.height))

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=20)

            # Draw label background
            try:
                font = ImageFont.load_default(size=48)
            except:
                font = ImageFont.load_default()
            
            text = f"{label} ({confidence:.2f})"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_height = text_bbox[3] - text_bbox[1]
            text_width = text_bbox[2] - text_bbox[0]

            if y1 - text_height - 5 > 0:
                draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 5, y1], fill=color)
                draw.text((x1 + 2, y1 - text_height - 3), text, fill="white", font=font)
            else:
                draw.rectangle([x1, y1, x1 + text_width + 5, y1 + text_height + 5], fill=color)
                draw.text((x1 + 2, y1 + 2), text, fill="white", font=font)


        return img
    except Exception as e:
        st.error(f"Error drawing bounding box: {str(e)}")
        # Return the original image if drawing fails
        return Image.open(image_path)


def get_image_info(row):
    """Extract detailed information about an image and detection."""
    info = {
        "Basic Info": {
            "Filename": row.get('frame_filename', 'N/A'),
            "Issue Type": row.get('label', 'N/A'),
            "Confidence": f"{row.get('confidence', 0):.3f}",
            "Class ID": row.get('class_id', 'N/A')
        },
        "Location": {
            "Latitude": f"{row.get('latitude', 0):.6f}",
            "Longitude": f"{row.get('longitude', 0):.6f}",
            "Timestamp": row.get('timestamp', 'N/A')
        },
        "Detection Details": {}
    }

    # Add bounding box info if available
    bbox_data = None
    if 'bbox' in row.index and pd.notna(row['bbox']):
        bbox_data = parse_bounding_box(row['bbox'])
    elif all(c in row.index for c in ['x1', 'y1', 'x2', 'y2']):
         if all(pd.notna(row[c]) for c in ['x1', 'y1', 'x2', 'y2']):
            bbox_data = [row['x1'], row['y1'], row['x2'], row['y2']]

    if bbox_data:
        info["Detection Details"]["Bounding Box"] = f"[{', '.join(map(str, [round(x, 2) for x in bbox_data]))}]"
        info["Detection Details"]["Box Area"] = f"{abs((bbox_data[2] - bbox_data[0]) * (bbox_data[3] - bbox_data[1])):.0f} px²"
    else:
        info["Detection Details"]["Bounding Box"] = "Not available"

    return info

# --- Helper Functions ---
def find_best_match_index(columns, keywords):
    for i, col in enumerate(columns):
        if any(keyword in col.lower() for keyword in keywords):
            return i
    return 0

# --- UI Functions ---
def build_filter_sidebar(df_issues_before, df_issues_after):
    """Builds the sidebar for filtering data and returns filtered dataframes."""
    st.sidebar.header("Global Filters")
    st.sidebar.checkbox("Show Bounding Boxes", value=st.session_state.show_bbox, key="show_bbox")

    issue_colors = {}
    with st.sidebar.expander("Issue Colors", expanded=False):
        for i, lbl in enumerate(LABEL_NAMES):
            issue_colors[lbl] = st.color_picker(lbl, DEFAULT_COLORS[i], key=f"color_{lbl}")
    st.session_state.issue_colors = issue_colors

    sel_label = st.sidebar.selectbox("Filter by Problem Type", ["All"] + LABEL_NAMES, key="filter_label")
    min_confidence = st.sidebar.slider("Minimum Confidence Score", 0.0, 1.0, 0.3, 0.05, key="filter_confidence")

    df_filtered_before = df_issues_before[df_issues_before["confidence"] >= min_confidence]
    df_filtered_after = df_issues_after[df_issues_after["confidence"] >= min_confidence]
    if sel_label != "All":
        df_filtered_before = df_filtered_before[df_filtered_before["label"] == sel_label]
        df_filtered_after = df_filtered_after[df_filtered_after["label"] == sel_label]
        
    return df_filtered_before.copy(), df_filtered_after.copy()

def build_map(df_route, df_disp, issue_colors, lat_col, lon_col):
    if df_route.empty or lat_col not in df_route.columns or lon_col not in df_route.columns:
        st.error(f"Selected columns '{lat_col}' or '{lon_col}' not found in route data.")
        return folium.Map(location=[40.7608, -111.8910], zoom_start=10)
    
    # Drop rows with missing lat/lon data
    df_route.dropna(subset=[lat_col, lon_col], inplace=True)
    if df_route.empty:
        st.error("No valid GPS data to draw route.")
        return folium.Map(location=[40.7608, -111.8910], zoom_start=10)

    initial_location = (df_route.iloc[0][lat_col], df_route.iloc[0][lon_col])
    m = folium.Map(location=initial_location, zoom_start=16)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.PolyLine(list(zip(df_route[lat_col], df_route[lon_col])), color="blue", weight=5, opacity=0.7).add_to(m)
    
    marker_cluster = MarkerCluster(name="Issues").add_to(m)
    for _, row in df_disp.iterrows():
        popup_text = f"<b>{row['label']}</b><br>Confidence: {row['confidence']:.2f}<br><i>Click marker to view image</i>"
        popup = folium.Popup(popup_text, max_width=250)
        folium.CircleMarker(location=(row["latitude"], row["longitude"]), radius=6, color=issue_colors.get(row['label'], "#FF0000"), fill=True, fill_opacity=0.8, popup=popup).add_to(marker_cluster)
    
    folium.LayerControl().add_to(m)
    return m

def build_detail_map(df_route, lat_col, lon_col, issue_location):
    m = folium.Map(location=issue_location, zoom_start=18, scrollWheelZoom=False, zoom_control=False)
    folium.TileLayer("OpenStreetMap").add_to(m)
    
    df_route.dropna(subset=[lat_col, lon_col], inplace=True)
    if not df_route.empty:
        folium.PolyLine(list(zip(df_route[lat_col], df_route[lon_col])), color="blue", weight=5, opacity=0.7).add_to(m)
    
    folium.Marker(location=issue_location, popup="Current Issue", icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
    return m

def display_horizontal_legend(issue_colors):
    """Displays a single, compact, horizontal legend for all issue types."""
    st.subheader("Legend")
    num_columns = 4
    cols = st.columns(num_columns)
    for i, label in enumerate(LABEL_NAMES):
        color = issue_colors.get(label, "#888")
        with cols[i % num_columns]:
            st.markdown(
                f'<span style="display:inline-block; vertical-align:middle; margin-right:5px; width:12px; height:12px; background-color:{color}; border-radius:3px;"></span> {label}',
                unsafe_allow_html=True
            )

def display_image_with_bbox(image_path, row, issue_colors):
    """Display image with optional bounding box overlay."""
    if not os.path.exists(image_path):
        st.error(f"Image not found: {image_path}")
        return

    if st.session_state.get("show_bbox", True):
        bbox_data = None
        if 'bbox' in row.index and pd.notna(row['bbox']):
            bbox_data = parse_bounding_box(row['bbox'])
        elif all(c in row.index for c in ['x1', 'y1', 'x2', 'y2']):
            if all(pd.notna(row[c]) for c in ['x1', 'y1', 'x2', 'y2']):
                bbox_data = [row['x1'], row['y1'], row['x2'], row['y2']]

        if bbox_data:
            color = issue_colors.get(row['label'], "#FF0000")
            img_with_bbox = draw_bounding_box(image_path, bbox_data, row['label'], row['confidence'], color)
            st.image(img_with_bbox, use_container_width=True)
        else:
            st.image(image_path, use_container_width=True)
            st.info("No bounding box data available for this image.")
    else:
        st.image(image_path, use_container_width=True)


def display_image_info_panel(row):
    """Display detailed information about the selected image."""
    info = get_image_info(row)

    with st.expander("📊 Issue Information", expanded=True):
        for section, data in info.items():
            st.write(f"**{section}:**")
            for key, value in data.items():
                st.write(f"• **{key}:** {value}")
            st.write("")

def display_single_dashboard(df_route, df_disp, issue_colors, map_key, lat_col, lon_col, image_dir):
    st.markdown(f"**{len(df_disp)}** issues found with current filters.")
    with st.expander("📊 Analytics", expanded=True):
        if not df_disp.empty:
            counts_df = df_disp['label'].value_counts().reset_index()
            st.bar_chart(counts_df, x='label', y='count')
        else:
            st.write("No issues to analyze.")
    st.subheader("🗺️ Map")
    folium_map = build_map(df_route, df_disp, issue_colors, lat_col, lon_col)
    map_data = st_folium(folium_map, width=None, height=500, key=map_key)
    if map_data and map_data.get("last_object_clicked"):
        click_lat, click_lng = map_data["last_object_clicked"]["lat"], map_data["last_object_clicked"]["lng"]
        df_disp["_dist"] = ((df_disp["latitude"] - click_lat)**2 + (df_disp["longitude"] - click_lng)**2)
        selected_issue = df_disp.loc[df_disp["_dist"].idxmin()]
        image_path = os.path.join(image_dir, selected_issue["frame_filename"])
        
        # When an issue is clicked, store all necessary info in session state to switch to the fullscreen view
        st.session_state.image_to_show = image_path
        st.session_state.active_fullscreen_df = df_disp.copy()
        st.session_state.active_image_dir = image_dir
        st.session_state.active_route_df = df_route.copy()
        st.session_state.active_lat_col = lat_col
        st.session_state.active_lon_col = lon_col
        st.session_state.issue_colors = issue_colors # Pass colors
        st.rerun()

def display_gallery_for_version(df_disp, image_dir, version_key, df_route, lat_col, lon_col, issue_colors):
    if not image_dir or not os.path.isdir(image_dir):
        st.error(f"Image directory not found: `{image_dir}`.")
        return
    if df_disp.empty:
        st.info("No issues found for the current filter settings.")
        return

    def show_fullscreen_image(path, df_issues, df_route_data, lat_col_name, lon_col_name):
        st.session_state.image_to_show = path
        st.session_state.active_fullscreen_df = df_issues.copy()
        st.session_state.active_image_dir = image_dir
        st.session_state.active_route_df = df_route_data.copy()
        st.session_state.active_lat_col = lat_col_name
        st.session_state.active_lon_col = lon_col_name
        st.session_state.issue_colors = issue_colors

    page_key = f"gallery_page_{version_key}"
    if page_key not in st.session_state: st.session_state[page_key] = 0
    total_items = len(df_disp)
    start_index = st.session_state[page_key] * IMAGES_PER_PAGE
    end_index = min(start_index + IMAGES_PER_PAGE, total_items)
    df_page = df_disp.iloc[start_index:end_index]

    for i in range(0, len(df_page), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(df_page):
                row = df_page.iloc[i+j]
                img_path = os.path.join(image_dir, row.frame_filename)
                with cols[j]:
                    with st.container(border=True):
                        if os.path.exists(img_path):
                            display_image_with_bbox(img_path, row, issue_colors)
                            st.caption(f"{row.label} (Conf: {row.confidence:.2f})")
                            st.button("View Fullscreen", key=f"fs_{version_key}_{row.name}", on_click=show_fullscreen_image, args=(img_path, df_disp, df_route, lat_col, lon_col), use_container_width=True)
                        else:
                            st.warning(f"Not found: {row.frame_filename}")

    st.markdown("---")
    total_pages = math.ceil(total_items / IMAGES_PER_PAGE)
    if total_pages > 1:
        prev_col, mid_col, next_col = st.columns([2, 3, 2])
        if prev_col.button("⬅️ Previous", key=f"prev_{version_key}", disabled=(st.session_state[page_key] == 0), use_container_width=True):
            st.session_state[page_key] -= 1
            st.rerun()
        mid_col.write(f"Page {st.session_state[page_key] + 1} of {total_pages}")
        if next_col.button("Next ➡️", key=f"next_{version_key}", disabled=(st.session_state[page_key] >= total_pages - 1), use_container_width=True):
            st.session_state[page_key] += 1
            st.rerun()

# --- Main Application Logic ---

def run_configuration_view():
    """Displays the initial sidebar for uploading files and configuring the app."""
    st.sidebar.title("Data Upload & Config")
    st.sidebar.info("Upload data, select columns, specify paths, and then click 'Run Comparison'.")

    # Step 1: File Uploads
    st.sidebar.header("1. Before Maintenance Data")
    route_file_before = st.sidebar.file_uploader("Upload Route CSV (Before)", type="csv", key="upload_route_before")
    issues_file_before = st.sidebar.file_uploader("Upload Predictions CSV (Before)", type="csv", key="upload_issues_before")

    st.sidebar.header("2. After Maintenance Data")
    route_file_after = st.sidebar.file_uploader("Upload Route CSV (After)", type="csv", key="upload_route_after")
    issues_file_after = st.sidebar.file_uploader("Upload Predictions CSV (After)", type="csv", key="upload_issues_after")

    if not all([route_file_before, issues_file_before, route_file_after, issues_file_after]):
        st.warning("⬅️ Please upload all four CSV files in the sidebar to continue.")
        st.stop()

    # IMPROVED: Read file content into memory immediately to avoid state issues
    route_content_before = route_file_before.getvalue()
    issues_content_before = issues_file_before.getvalue()
    route_content_after = route_file_after.getvalue()
    issues_content_after = issues_file_after.getvalue()
    
    st.sidebar.markdown("---")
    st.sidebar.header("3. Column & Path Configuration")

    try:
        # IMPROVED: Create temporary dataframes from in-memory content just for column selection
        df_route_b_cols = pd.read_csv(io.BytesIO(route_content_before)).columns.str.strip()
        df_issues_b_cols = pd.read_csv(io.BytesIO(issues_content_before)).columns.str.strip()
        df_route_a_cols = pd.read_csv(io.BytesIO(route_content_after)).columns.str.strip()
        df_issues_a_cols = pd.read_csv(io.BytesIO(issues_content_after)).columns.str.strip()
        
        st.sidebar.subheader("'Before' Settings")
        lat_col_before = st.sidebar.selectbox("Latitude Column (Before)", df_route_b_cols, index=find_best_match_index(df_route_b_cols, ['lat']))
        lon_col_before = st.sidebar.selectbox("Longitude Column (Before)", df_route_b_cols, index=find_best_match_index(df_route_b_cols, ['lon', 'long']))
        conf_col_before = st.sidebar.selectbox("Confidence Column (Before)", df_issues_b_cols, index=find_best_match_index(df_issues_b_cols, ['conf', 'score']))
        img_dir_before = st.sidebar.text_input("Image Directory Path (Before)", placeholder="e.g., C:/Users/YourName/Photos/Before")

        st.sidebar.subheader("'After' Settings")
        lat_col_after = st.sidebar.selectbox("Latitude Column (After)", df_route_a_cols, index=find_best_match_index(df_route_a_cols, ['lat']))
        lon_col_after = st.sidebar.selectbox("Longitude Column (After)", df_route_a_cols, index=find_best_match_index(df_route_a_cols, ['lon', 'long']))
        conf_col_after = st.sidebar.selectbox("Confidence Column (After)", df_issues_a_cols, index=find_best_match_index(df_issues_a_cols, ['conf', 'score']))
        img_dir_after = st.sidebar.text_input("Image Directory Path (After)", placeholder="e.g., C:/Users/YourName/Photos/After")

        if st.sidebar.button("🚀 Run Comparison", type="primary", use_container_width=True):
            st.session_state.config = {
                "route_content_before": route_content_before, "issues_content_before": issues_content_before,
                "lat_col_before": lat_col_before, "lon_col_before": lon_col_before,
                "conf_col_before": conf_col_before, "img_dir_before": img_dir_before,
                "route_content_after": route_content_after, "issues_content_after": issues_content_after,
                "lat_col_after": lat_col_after, "lon_col_after": lon_col_after,
                "conf_col_after": conf_col_after, "img_dir_after": img_dir_after,
            }
            st.session_state.run_app = True
            st.rerun()
    except Exception as e:
        st.error(f"Failed to process CSV files for configuration: {e}")
        st.stop()

def run_dashboard_view():
    """Displays the main dashboard with maps and galleries after configuration."""
    st.sidebar.title("Dashboard Controls")
    if st.sidebar.button("🔄 Start Over with New Data"):
        # IMPROVED: Safely clear the entire session state
        st.session_state.clear()
        st.rerun()
    st.sidebar.markdown("---")
    
    cfg = st.session_state.config
    df_route_before, df_issues_before = load_data(cfg['route_content_before'], cfg['issues_content_before'], cfg['conf_col_before'])
    df_route_after, df_issues_after = load_data(cfg['route_content_after'], cfg['issues_content_after'], cfg['conf_col_after'])

    if df_issues_before.empty or df_issues_after.empty:
        st.error("Data loading failed. Please check your CSV files and column selections, then 'Start Over'.")
        st.stop()
    
    df_disp_before, df_disp_after = build_filter_sidebar(df_issues_before, df_issues_after)
    
    # FIX: Retrieve issue_colors from session_state to be used in this scope.
    issue_colors = st.session_state.issue_colors

    tab1, tab2 = st.tabs(["🗺️ Dashboard Comparison", "🖼️ Image Gallery"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.header("Before Maintenance")
            st.markdown(f"**{len(df_disp_before)}** issues found with current filters.")
            # --- ANALYTICS ---
            with st.expander("📊 Analytics", expanded=True):
                if not df_disp_before.empty:
                    counts_df = df_disp_before['label'].value_counts().reset_index()
                    st.bar_chart(counts_df, x='label', y='count')
                else:
                    st.write("No issues to analyze.")
            # FIX: Correctly call build_map with all arguments.
            folium_map = build_map(df_route_before, df_disp_before, issue_colors, cfg['lat_col_before'], cfg['lon_col_before'])
            map_data = st_folium(folium_map, width=None, height=500, key="before_map")
            if map_data and map_data.get("last_object_clicked"):
                lat, lng = map_data["last_object_clicked"]["lat"], map_data["last_object_clicked"]["lng"]
                df_disp_before["_dist"] = ((df_disp_before["latitude"] - lat)**2 + (df_disp_before["longitude"] - lng)**2)
                selected_issue = df_disp_before.loc[df_disp_before["_dist"].idxmin()]
                st.session_state.image_to_show = os.path.join(cfg['img_dir_before'], selected_issue["frame_filename"])
                
                # FIX: Pass all required data to session_state for the fullscreen view.
                st.session_state.active_fullscreen_df = df_disp_before.copy()
                st.session_state.active_image_dir = cfg['img_dir_before']
                st.session_state.active_route_df = df_route_before.copy()
                st.session_state.active_lat_col = cfg['lat_col_before']
                st.session_state.active_lon_col = cfg['lon_col_before']
                st.rerun()
        with col2:
            st.header("After Maintenance")
            st.markdown(f"**{len(df_disp_after)}** issues found with current filters.")
            # --- ANALYTICS ---
            with st.expander("📊 Analytics", expanded=True):
                if not df_disp_after.empty:
                    counts_df = df_disp_after['label'].value_counts().reset_index()
                    st.bar_chart(counts_df, x='label', y='count')
                else:
                    st.write("No issues to analyze.")
            # FIX: Correctly call build_map with all arguments.
            folium_map = build_map(df_route_after, df_disp_after, issue_colors, cfg['lat_col_after'], cfg['lon_col_after'])
            map_data = st_folium(folium_map, width=None, height=500, key="after_map")
            if map_data and map_data.get("last_object_clicked"):
                lat, lng = map_data["last_object_clicked"]["lat"], map_data["last_object_clicked"]["lng"]
                df_disp_after["_dist"] = ((df_disp_after["latitude"] - lat)**2 + (df_disp_after["longitude"] - lng)**2)
                selected_issue = df_disp_after.loc[df_disp_after["_dist"].idxmin()]
                st.session_state.image_to_show = os.path.join(cfg['img_dir_after'], selected_issue["frame_filename"])
                
                # FIX: Pass all required data to session_state for the fullscreen view.
                st.session_state.active_fullscreen_df = df_disp_after.copy()
                st.session_state.active_image_dir = cfg['img_dir_after']
                st.session_state.active_route_df = df_route_after.copy()
                st.session_state.active_lat_col = cfg['lat_col_after']
                st.session_state.active_lon_col = cfg['lon_col_after']
                st.rerun()
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Before Maintenance Gallery")
            # FIX: Correctly call display_gallery_for_version with all arguments.
            display_gallery_for_version(df_disp_before, cfg['img_dir_before'], "before", df_route_before, cfg['lat_col_before'], cfg['lon_col_before'], issue_colors)
        with col2:
            st.subheader("After Maintenance Gallery")
            # FIX: Correctly call display_gallery_for_version with all arguments.
            display_gallery_for_version(df_disp_after, cfg['img_dir_after'], "after", df_route_after, cfg['lat_col_after'], cfg['lon_col_after'], issue_colors)
    st.markdown("---")
    # FIX: Use the 'issue_colors' variable that is now correctly defined in this scope.
    display_horizontal_legend(issue_colors)


def run_fullscreen_view():
    """Displays the detailed view for a single selected image."""
    st.header("Detailed Issue View")
    if st.button("⬅️ Back to Dashboard"):
        st.session_state.image_to_show = None
        st.session_state.filmstrip_page = 0
        st.rerun()

    image_filename = os.path.basename(st.session_state.image_to_show)
    df_issues = st.session_state.active_fullscreen_df
    issue_row = df_issues[df_issues['frame_filename'] == image_filename].iloc[0]
    issue_location = (issue_row['latitude'], issue_row['longitude'])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📸 Image with Detection")
        display_image_with_bbox(st.session_state.image_to_show, issue_row, st.session_state.issue_colors)

    with col2:
        st.subheader("🗺️ Location")
        detail_map = build_detail_map(st.session_state.active_route_df, st.session_state.active_lat_col, st.session_state.active_lon_col, issue_location)
        st_folium(detail_map, height=300)
        display_image_info_panel(issue_row)

    st.markdown("---")
    st.subheader("Browse Other Issues in this Dataset")

    df_filmstrip = st.session_state.active_fullscreen_df
    image_dir = st.session_state.active_image_dir
    total_items = len(df_filmstrip)
    total_pages = math.ceil(total_items / THUMBNAILS_PER_PAGE)
    
    if 'filmstrip_page' not in st.session_state or st.session_state.filmstrip_page >= total_pages:
        st.session_state.filmstrip_page = 0

    start_index = st.session_state.filmstrip_page * THUMBNAILS_PER_PAGE
    end_index = min(start_index + THUMBNAILS_PER_PAGE, total_items)
    df_page = df_filmstrip.iloc[start_index:end_index]

    if total_pages > 1:
        prev_col, mid_col, next_col = st.columns([1, 5, 1])
        if prev_col.button("⬅️", key="filmstrip_prev", disabled=(st.session_state.filmstrip_page == 0), use_container_width=True):
            st.session_state.filmstrip_page -= 1
            st.rerun()
        mid_col.write(f"Page {st.session_state.filmstrip_page + 1} of {total_pages}")
        if next_col.button("➡️", key="filmstrip_next", disabled=(st.session_state.filmstrip_page >= total_pages - 1), use_container_width=True):
            st.session_state.filmstrip_page += 1
            st.rerun()

    cols = st.columns(len(df_page))
    for i, col in enumerate(cols):
        row = df_page.iloc[i]
        thumb_path = os.path.join(image_dir, row.frame_filename)
        if os.path.exists(thumb_path):
            with col:
                is_selected = (thumb_path == st.session_state.image_to_show)
                with st.container(border=is_selected):
                    display_image_with_bbox(thumb_path, row, st.session_state.issue_colors)
                    if st.button(f"View", key=f"thumb_{row.name}", use_container_width=True, disabled=is_selected):
                        st.session_state.image_to_show = thumb_path
                        st.rerun()

def main():
    """Main application router."""
    if "filmstrip_page" not in st.session_state:
        st.session_state.filmstrip_page = 0
    if "show_bbox" not in st.session_state:
        st.session_state.show_bbox = True
    if "issue_colors" not in st.session_state:
        st.session_state.issue_colors = {lbl: DEFAULT_COLORS[i] for i, lbl in enumerate(LABEL_NAMES)}

    if st.session_state.get("image_to_show"):
        run_fullscreen_view()
    elif st.session_state.get("run_app"):
        run_dashboard_view()
    else:
        run_configuration_view()

if __name__ == "__main__":
    main()