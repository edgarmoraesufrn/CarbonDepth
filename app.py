# Carbon Depth – Cement Carbonation Analysis
# pip install streamlit opencv-python numpy matplotlib reportlab pillow pandas scipy
# cd C:\Moraes\CarbonDepth
# streamlit run app.py

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import io
import tempfile
from pathlib import Path

st.set_page_config(layout="wide")
# ---------------------------
# Paths (cross-platform safe)
# ---------------------------
# Streamlit Community Cloud runs on Linux (case-sensitive filesystem).
# Use BASE_DIR + a small resolver to locate assets/docs whether they live in
# the project root or in subfolders (assets/, docs/).
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

def resolve_resource(*candidates: str) -> Path | None:
    """Return the first existing path among candidates (relative to BASE_DIR)."""
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if not p.is_absolute():
            p = BASE_DIR / p
        if p.exists():
            return p
    return None

from math import pi
import statistics
import pandas as pd
from scipy.stats import linregress

# Optional dependency for v11 calibration (2-click scale).
try:
    from streamlit_drawable_canvas import st_canvas  # pip install streamlit-drawable-canvas
except Exception:
    st_canvas = None
from scipy.optimize import curve_fit

# Optional: use sklearn metrics if available (recommended)
try:
    from sklearn.metrics import r2_score, mean_squared_error
except Exception:
    def mean_squared_error(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean((y_true - y_pred) ** 2))

    def r2_score(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def model_fick_saturating(t_years, a, b):
    """Saturating model: z(t)=a√t + b·√t/(√t+1), with t in years."""
    t_years = np.asarray(t_years, dtype=float)
    u = np.sqrt(np.clip(t_years, 0.0, None))
    return a * u + b * (u / (u + 1.0))


def compute_metrics(y_true, y_pred):
    """Return (R², MSE, RMSE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return max(r2, 0.0), mse, rmse



# =========================
# 🧰 Quality Check Helpers
# =========================
def _mask_border_touch_ratio(mask: np.ndarray, border_px: int = 2) -> float:
    """Return fraction of mask pixels that lie on an image border band (0..1)."""
    if mask is None or mask.size == 0:
        return 0.0
    h, w = mask.shape[:2]
    if h <= 2*border_px or w <= 2*border_px:
        return 1.0
    m = (mask > 0).astype(np.uint8)
    total = int(m.sum())
    if total == 0:
        return 0.0
    band = np.zeros((h, w), dtype=np.uint8)
    band[:border_px, :] = 1
    band[-border_px:, :] = 1
    band[:, :border_px] = 1
    band[:, -border_px:] = 1
    touch = int((m & band).sum())
    return touch / total


def _largest_component_stats(binary_mask: np.ndarray):
    """Connected-component stats for largest component in binary mask."""
    m = (binary_mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(m) == 0:
        return {
            "n_components": 0,
            "largest_area": 0,
            "largest_ratio": 0.0,
            "solidity": 0.0,
            "circularity": 0.0,
            "perimeter_area_ratio": 0.0,
        }

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    # stats rows: [label, x, y, w, h, area] but actually stats is (num_labels, 5) => x,y,w,h,area
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([0])
    n_components = int(num_labels - 1)
    total_area = float(areas.sum()) if areas.size else 0.0
    largest_area = float(areas.max()) if areas.size else 0.0
    largest_ratio = (largest_area / total_area) if total_area > 0 else 0.0

    # Extract contour of largest component for shape metrics
    largest_label = 1 + int(np.argmax(areas)) if areas.size else 0
    comp_mask = (labels == largest_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 0.0
    circularity = 0.0
    perimeter_area_ratio = 0.0

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        perim = float(cv2.arcLength(cnt, True))
        if area > 0:
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
            if hull_area > 0:
                solidity = area / hull_area
            if perim > 0:
                circularity = (4.0 * pi * area) / (perim ** 2)  # 1.0 is perfect circle
                perimeter_area_ratio = perim / area

    return {
        "n_components": n_components,
        "largest_area": largest_area,
        "largest_ratio": float(largest_ratio),
        "solidity": float(solidity),
        "circularity": float(circularity),
        "perimeter_area_ratio": float(perimeter_area_ratio),
    }


def evaluate_quality_checks(
    image_bgr: np.ndarray,
    mask_cement: np.ndarray,
    mask_purple_on_cement: np.ndarray,
    percent_alkaline: float,
):
    """
    Heuristic QA flags that DO NOT change outputs.

    Design goals:
      • Conservative (avoid false alarms).
      • Explanatory (helps user decide if the result needs review).
      • Non-blocking (never changes the core segmentation / measurements).

    Returns: status_str, severity ('ok'|'caution'|'review'), messages(list[str]), metrics(dict)
    """
    flags: list[str] = []
    cautions = 0
    reviews = 0
    metrics: dict = {}

    # -------------------------
    # 1) Alkaline mask size
    # -------------------------
    purple_area = int(np.count_nonzero(mask_purple_on_cement))
    cement_area = int(np.count_nonzero(mask_cement))
    alkaline_ratio = (purple_area / cement_area) if cement_area > 0 else 0.0
    metrics["alkaline_area_px"] = purple_area
    metrics["cement_area_px"] = cement_area
    metrics["alkaline_percent"] = float(percent_alkaline)

    # Very small alkaline region can be real (advanced carbonation), so keep thresholds gentle.
    if cement_area > 0:
        if alkaline_ratio < 0.001:  # <0.1%
            reviews += 1
            flags.append("❌ Alkaline region is extremely small (<0.1% of cement area). Verify staining/segmentation.")
        elif alkaline_ratio < 0.005:  # <0.5%
            cautions += 1
            flags.append("⚠️ Alkaline region is small (<0.5% of cement area). Result may be sensitive to noise.")
    else:
        reviews += 1
        flags.append("❌ Cement mask is empty or invalid. Please re-check image/background and try again.")

    # -------------------------
    # 2) Fragmentation / morphology of alkaline mask
    # -------------------------
    n_components = 0
    largest_ratio = 0.0
    solidity = 1.0
    per_area_ratio = 0.0

    if purple_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask_purple_on_cement > 0).astype(np.uint8), connectivity=8
        )
        n_components = max(0, int(num_labels) - 1)
        metrics["purple_n_components"] = n_components

        if n_components > 0:
            areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
            largest_area = float(np.max(areas)) if areas.size else 0.0
            largest_ratio = (largest_area / float(purple_area)) if purple_area > 0 else 0.0
            metrics["purple_largest_area"] = float(largest_area)
            metrics["purple_largest_ratio"] = float(largest_ratio)

        # contour-based metrics (use the largest component contour)
        mask_u8 = (mask_purple_on_cement > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(cnt))
            peri = float(cv2.arcLength(cnt, True))
            per_area_ratio = (peri / area) if area > 0 else 0.0
            metrics["purple_perimeter_area_ratio"] = float(per_area_ratio)

            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = (area / hull_area) if hull_area > 0 else 1.0
            metrics["purple_solidity"] = float(solidity)

        # Gentle fragmentation rule:
        #  - it's common to have small speckles; flag only if fragmentation is high AND largest component is not dominant.
        if n_components >= 10 and largest_ratio < 0.85:
            cautions += 1
            flags.append("⚠️ Alkaline mask appears fragmented (many disconnected components). Consider increasing cleanup kernel.")
        if n_components >= 18 and largest_ratio < 0.75:
            reviews += 1
            flags.append("❌ Alkaline mask is highly fragmented. Review image quality and segmentation settings.")

        # Strongly jagged boundary / low solidity (only as a mild signal)
        if per_area_ratio > 0.020 and solidity < 0.85:
            cautions += 1
            flags.append("⚠️ Alkaline boundary looks irregular (high perimeter/area). Check focus/lighting and cleanup kernel.")

    else:
        metrics["purple_n_components"] = 0
        metrics["purple_largest_ratio"] = 0.0

    # -------------------------
    # 3) Lighting / exposure (HSV Value channel)
    # -------------------------
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    v_mean = float(np.mean(v))
    v_std = float(np.std(v))
    metrics["v_mean"] = v_mean
    metrics["v_std"] = v_std

    # Keep conservative: only warn on very dark/flat images
    if v_mean < 35 or v_std < 8:
        reviews += 1
        flags.append("❌ Image appears underexposed or low-contrast. Improve lighting and avoid shadows.")
    elif v_mean < 50 or v_std < 15:
        cautions += 1
        flags.append("⚠️ Image lighting/contrast is suboptimal. Results may be sensitive; consider re-taking the photo.")

    # -------------------------
    # 4) Mask touching borders (cropping risk)
    # -------------------------
    border_touch_ratio = 0.0
    if mask_cement is not None and mask_cement.size > 0:
        m = (mask_cement > 0).astype(np.uint8)
        h, w = m.shape[:2]
        border = np.zeros_like(m)
        border[0, :] = 1
        border[h - 1, :] = 1
        border[:, 0] = 1
        border[:, w - 1] = 1
        touches = int(np.count_nonzero(m & border))
        total_border = int(np.count_nonzero(border))
        border_touch_ratio = (touches / total_border) if total_border > 0 else 0.0
        metrics["cement_border_touch_ratio"] = float(border_touch_ratio)

        if border_touch_ratio > 0.08:
            reviews += 1
            flags.append("❌ Specimen mask touches the image border (likely cropped). Reframe the specimen fully within the photo.")
        elif border_touch_ratio > 0.03:
            cautions += 1
            flags.append("⚠️ Specimen mask is close to the image border. Ensure full specimen is visible (avoid cropping).")

    # -------------------------
    # Final severity
    # -------------------------
    if reviews > 0:
        severity = "review"
        status = "❌ Review recommended"
    elif cautions > 0:
        severity = "caution"
        status = "⚠️ Caution"
    else:
        severity = "ok"
        status = "✅ OK"

    return status, severity, flags, metrics
# --- 🎨 Custom CSS: Light blue background ---
st.markdown(
    """
    <style>
/* ===== Carbon Depth | Professional theme ===== */
:root{
  --cd-bg: #f6f8ff;
  --cd-card: #ffffff;
  --cd-text: #111827;
  --cd-muted: #4b5563;
  --cd-accent: #1f4e79;
  --cd-accent-2: #2f6fb3;
  --cd-border: rgba(17,24,39,0.12);
}

/* App background + default text */
.stApp{
  background: var(--cd-bg);
  color: var(--cd-text);
}

/* Make Streamlit labels and markdown readable everywhere */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
label, .stTextInput label, .stSelectbox label, .stRadio label,
.stNumberInput label, .stSlider label, .stDataFrame label{
  color: var(--cd-text) !important;
}

/* Headings */
h1, h2, h3, h4{
  color: var(--cd-accent) !important;
  letter-spacing: 0.2px;
}

/* Cards / containers (Streamlit uses 'block-container' for main content) */
.block-container{
  padding-top: 1.6rem;
  padding-bottom: 2.0rem;
  max-width: 1200px;
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput input,
.stSelectbox > div > div,
.stMultiSelect > div > div{
  border: 1px solid var(--cd-border) !important;
  background: var(--cd-card) !important;
}

/* Data editor / tables */
[data-testid="stDataFrame"]{
  border: 1px solid var(--cd-border);
  border-radius: 12px;
  overflow: hidden;
  background: var(--cd-card);
}

/* Buttons */
.stButton > button{
  border-radius: 12px;
  border: 1px solid var(--cd-border);
  background: linear-gradient(180deg, var(--cd-accent), var(--cd-accent-2));
  color: #ffffff;
  font-weight: 600;
  padding: 0.55rem 1.0rem;
}
.stButton > button:hover{ filter: brightness(1.05); }

/* Download buttons (Streamlit renders these separately from .stButton) */
.stDownloadButton > button{
  border-radius: 12px;
  border: 1px solid var(--cd-border) !important;
  background: linear-gradient(180deg, var(--cd-accent), var(--cd-accent-2)) !important;
  color: #ffffff !important;
  font-weight: 600;
  padding: 0.55rem 1.0rem;
}
.stDownloadButton > button:hover{ filter: brightness(1.05); }

/* Download buttons */
.stDownloadButton > button{
  border-radius: 12px;
  border: 1px solid var(--cd-border) !important;
  background: linear-gradient(180deg, var(--cd-accent), var(--cd-accent-2)) !important;
  color: #ffffff !important;
  font-weight: 600;
  padding: 0.55rem 1.0rem;
}
.stDownloadButton > button:hover{ filter: brightness(1.05); }

/* Download buttons (Streamlit renders these separately from .stButton) */
.stDownloadButton > button{
  border-radius: 12px;
  border: 1px solid var(--cd-border);
  background: linear-gradient(180deg, var(--cd-accent), var(--cd-accent-2));
  color: #ffffff !important;
  font-weight: 600;
  padding: 0.55rem 1.0rem;
}
.stDownloadButton > button:hover{ filter: brightness(1.05); }

/* Metrics */
[data-testid="stMetric"]{
  background: var(--cd-card);
  border: 1px solid var(--cd-border);
  border-radius: 14px;
  padding: 12px 14px;
}
[data-testid="stMetricLabel"]{
  color: var(--cd-muted) !important;
}
[data-testid="stMetricValue"]{
  color: var(--cd-text) !important;
}

/* Expanders */
.streamlit-expanderHeader{
  border-radius: 12px;
  border: 1px solid var(--cd-border);
  background: var(--cd-card);
}
.streamlit-expanderContent{
  border: 1px solid var(--cd-border);
  border-top: none;
  border-radius: 0 0 12px 12px;
  background: var(--cd-card);
}

/* Footer */
.footer{
  margin-top: 2rem;
  padding: 1.0rem 1.0rem 0.6rem 1.0rem;
  border-top: 1px solid var(--cd-border);
  color: var(--cd-muted);
  font-size: 0.95rem;
}

/* Ensure radio option text stays visible on light background */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] label p{
  color: var(--cd-text) !important;
}
div[data-testid="stRadio"] > label,
div[data-testid="stRadio"] > label span{
  color: var(--cd-text) !important;
}


/* Force input text color (Streamlit sometimes uses -webkit-text-fill-color) */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
div[data-baseweb="select"] *{
  color: var(--cd-text) !important;
  -webkit-text-fill-color: var(--cd-text) !important;
}

/* Placeholder / disabled text */
.stTextInput input::placeholder, .stNumberInput input::placeholder{
  color: rgba(17,24,39,0.55) !important;
  -webkit-text-fill-color: rgba(17,24,39,0.55) !important;
}

/* ===== Inputs text + placeholders (robust readability) ===== */
.stTextInput input, .stNumberInput input{
  color: var(--cd-text) !important;
}
.stTextInput input::placeholder, .stNumberInput input::placeholder, textarea::placeholder{
  color: rgba(75,85,99,0.9) !important;
  opacity: 1 !important;
}
/* Selectbox value text */
div[data-baseweb="select"] *{
  color: var(--cd-text) !important;
}

/* File uploader readability + consistent button styling */
[data-testid="stFileUploader"] section{
  background: var(--cd-card) !important;
  border: 1px solid var(--cd-border) !important;
  border-radius: 12px !important;
}
[data-testid="stFileUploader"] *{
  color: var(--cd-text) !important;
}
[data-testid="stFileUploader"] button{
  border-radius: 12px !important;
  border: 1px solid var(--cd-border) !important;
  background: linear-gradient(180deg, var(--cd-accent), var(--cd-accent-2)) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
}

</style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
/* ===== Carbon Depth | Professional theme ===== */
:root{
  --cd-bg: #f6f8ff;
  --cd-card: #ffffff;
  --cd-text: #111827;
  --cd-muted: #4b5563;
  --cd-accent: #1f4e79;
  --cd-accent-2: #2f6fb3;
  --cd-border: rgba(17,24,39,0.12);
}

/* App background + default text */
.stApp{
  background: var(--cd-bg);
  color: var(--cd-text);
}

/* Make Streamlit labels and markdown readable everywhere */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
label, .stTextInput label, .stSelectbox label, .stRadio label,
.stNumberInput label, .stSlider label, .stDataFrame label{
  color: var(--cd-text) !important;
}

/* Headings */
h1, h2, h3, h4{
  color: var(--cd-accent) !important;
  letter-spacing: 0.2px;
}

/* Cards / containers (Streamlit uses 'block-container' for main content) */
.block-container{
  padding-top: 1.6rem;
  padding-bottom: 2.0rem;
  max-width: 1200px;
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput input,
.stSelectbox > div > div,
.stMultiSelect > div > div{
  border: 1px solid var(--cd-border) !important;
  background: var(--cd-card) !important;
}

/* Data editor / tables */
[data-testid="stDataFrame"]{
  border: 1px solid var(--cd-border);
  border-radius: 12px;
  overflow: hidden;
  background: var(--cd-card);
}

/* Buttons */
.stButton > button{
  border-radius: 12px;
  border: 1px solid var(--cd-border);
  background: linear-gradient(180deg, var(--cd-accent), var(--cd-accent-2));
  color: #ffffff;
  font-weight: 600;
  padding: 0.55rem 1.0rem;
}
.stButton > button:hover{ filter: brightness(1.05); }

/* Metrics */
[data-testid="stMetric"]{
  background: var(--cd-card);
  border: 1px solid var(--cd-border);
  border-radius: 14px;
  padding: 12px 14px;
}
[data-testid="stMetricLabel"]{
  color: var(--cd-muted) !important;
}
[data-testid="stMetricValue"]{
  color: var(--cd-text) !important;
}

/* Expanders */
.streamlit-expanderHeader{
  border-radius: 12px;
  border: 1px solid var(--cd-border);
  background: var(--cd-card);
}
.streamlit-expanderContent{
  border: 1px solid var(--cd-border);
  border-top: none;
  border-radius: 0 0 12px 12px;
  background: var(--cd-card);
}

/* Footer */
.footer{
  margin-top: 2rem;
  padding: 1.0rem 1.0rem 0.6rem 1.0rem;
  border-top: 1px solid var(--cd-border);
  color: var(--cd-muted);
  font-size: 0.95rem;
}

/* Ensure radio option text stays visible on light background */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] label p{
  color: var(--cd-text) !important;
}
div[data-testid="stRadio"] > label,
div[data-testid="stRadio"] > label span{
  color: var(--cd-text) !important;
}


/* Force input text color (Streamlit sometimes uses -webkit-text-fill-color) */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
div[data-baseweb="select"] *{
  color: var(--cd-text) !important;
  -webkit-text-fill-color: var(--cd-text) !important;
}

/* Placeholder / disabled text */
.stTextInput input::placeholder, .stNumberInput input::placeholder{
  color: rgba(17,24,39,0.55) !important;
  -webkit-text-fill-color: rgba(17,24,39,0.55) !important;
}

/* ===== Inputs text + placeholders (robust readability) ===== */
.stTextInput input, .stNumberInput input{
  color: var(--cd-text) !important;
}
.stTextInput input::placeholder, .stNumberInput input::placeholder, textarea::placeholder{
  color: rgba(75,85,99,0.9) !important;
  opacity: 1 !important;
}
/* Selectbox value text */
div[data-baseweb="select"] *{
  color: var(--cd-text) !important;
}

/* File uploader readability + consistent button styling */
[data-testid="stFileUploader"] section{
  background: var(--cd-card) !important;
  border: 1px solid var(--cd-border) !important;
  border-radius: 12px !important;
}
[data-testid="stFileUploader"] *{
  color: var(--cd-text) !important;
}
[data-testid="stFileUploader"] button{
  border-radius: 12px !important;
  border: 1px solid var(--cd-border) !important;
  background: linear-gradient(180deg, var(--cd-accent), var(--cd-accent-2)) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
}

</style>
    """,
    unsafe_allow_html=True
)

# --- 🏢 Logo and Title ---
try:
    logo = Image.open(resolve_resource("assets/Logo.png", "Logo.png") or "Logo.png")
    st.image(logo, width=250)
except Exception:
    st.warning("Main logo (Logo.png) not found.")

st.markdown(
    "<h1 style='color: #2E418E; font-size: 1.8em;'>🧪 Carbon Depth – Cement Carbonation Analysis</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='color: #5593CA; font-size: 1.1em;'>Quantitative phenolphthalein test for cement and concrete durability</p>",
    unsafe_allow_html=True
)
# --- 📘 Procedure / SOP ---
st.subheader("📘 Procedure / SOP")
st.markdown(
    "Access the LabCim procedure for the phenolphthalein-based determination of carbonation depth "
    "in cementitious materials. This is independent from the image-processing pipeline."
)

sop_path = resolve_resource("docs/Carbon Depth Method.pdf", "Carbon Depth Method.pdf",
                           "docs/Carbon_Depth_Method.pdf", "Carbon_Depth_Method.pdf")
if sop_path is None:
    st.info("Procedure PDF not found. Place it in **docs/** (recommended) or in the project root.")
else:
    try:
        sop_bytes = sop_path.read_bytes()
        st.download_button(
            "⬇️ Download Procedure (PDF)",
            data=sop_bytes,
            file_name="Carbon_Depth_Method.pdf",
            mime="application/pdf",
            use_container_width=False,
        )
    except Exception as e:
        st.error(f"Unable to read the procedure PDF: {e}")

# --- 📤 Upload Image ---
uploaded_file = st.file_uploader(
    "📷 Upload the original or processed image (.png, .jpg, .jpeg)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # ✅ Read file only once
    try:
        file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Failed to decode image. The file may be corrupted or not a valid image.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading image: {str(e)}")
        st.stop()

    # --- 📐 Specimen Configuration ---
    st.subheader("🎯 Specimen Configuration")
    col_shape, col_dims = st.columns(2)
    with col_shape:
        specimen_shape = st.selectbox("Specimen Shape", ["Cylindrical", "Rectangular"])
    with col_dims:
        if specimen_shape == "Cylindrical":
            d_mm = st.number_input("Diameter (mm)", min_value=1.0, value=50.00, format="%.2f")
            h_mm = st.number_input("Height (mm)", min_value=1.0, value=100.00, format="%.2f")
        else:
            base_mm = st.number_input("Base (mm)", min_value=1.0, value=50.00, format="%.2f")
            h_mm = st.number_input("Height (mm)", min_value=1.0, value=100.00, format="%.2f")
            depth_mm = st.number_input("Depth (mm)", min_value=1.0, value=50.00, format="%.2f")

    st.subheader("🎛️ Segmentation Settings")
    # NOTE: The former "gray regions" control became a derived quantity (total − purple),
    # so the slider was removed to avoid a placebo control. We keep the same fixed value
    # here to preserve the v6/v7 image-processing behavior.
    kernel_size_gray = 5

    # Single, meaningful control: alkaline/purple cleanup
    kernel_size_purple = st.slider("Cleanup kernel (alkaline/purple)", 1, 15, 5, step=2)
    st.markdown(
        "<small>💡 Tip: Refines the alkaline (purple) region. Increase to remove noise; decrease to preserve fine details.</small>",
        unsafe_allow_html=True,
    )

    # Optional (experimental): enhanced segmentation for difficult tone-on-tone images.
    hard_mode = st.toggle(
        "Hard image mode (tone-on-tone) [experimental]",
        value=False,
        help=(
            "Runs an alternative segmentation pipeline with stronger contrast enhancement and illumination normalization. "
            "If the result looks worse, turn this OFF to use the default (stable) mode."
        ),
    )


    # --- 🖼️ Load and Enhance Image ---
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    image_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)

    # --- 🎨 Color Segmentation ---
    lower_purple = np.array([115, 20, 20])
    upper_purple = np.array([179, 255, 255])
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_gray = np.array([0, 10, 50])
    upper_gray = np.array([180, 90, 200])
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

    mask_gray = cv2.bitwise_and(mask_gray, cv2.bitwise_not(mask_white))
    mask_purple = cv2.bitwise_and(mask_purple, cv2.bitwise_not(mask_white))

    # --- 🔧 Morphology for purple ---
    merge_kernel = np.ones((kernel_size_purple * 3, kernel_size_purple * 3), dtype=np.uint8)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, merge_kernel)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, np.ones((kernel_size_purple, kernel_size_purple), np.uint8))

    # --- 🛠️ Shape-Based Mask (bright regions) ---
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    white_rgb = cv2.inRange(image, (250, 250, 255), (255, 255, 255))
    _, mask_bright = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    mask_bright_no_white = cv2.bitwise_and(mask_bright, cv2.bitwise_not(white_rgb))
    kernel = np.ones((15, 15), np.uint8)
    mask_shape = cv2.morphologyEx(mask_bright_no_white, cv2.MORPH_CLOSE, kernel)
    mask_shape = cv2.morphologyEx(mask_shape, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_shape_clean = np.zeros_like(mask_shape)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:
            cv2.drawContours(mask_shape_clean, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # --- 🎯 Combine color and shape to form initial mask_cement ---
    mask_cement_color = cv2.bitwise_or(mask_gray, mask_purple)
    mask_cement = cv2.bitwise_or(mask_cement_color, cv2.bitwise_and(mask_shape_clean, cv2.bitwise_not(mask_cement_color)))

    # ✅ APLICAR kernel_size_gray NO mask_cement (controle do contorno)
    kernel_gray = np.ones((kernel_size_gray, kernel_size_gray), np.uint8)
    mask_cement = cv2.morphologyEx(mask_cement, cv2.MORPH_CLOSE, kernel_gray)  # Fecha buracos
    mask_cement = cv2.morphologyEx(mask_cement, cv2.MORPH_OPEN, kernel_gray)   # Remove ruído

    # ✅ Default alkaline mask (stable core)
    mask_purple_default = cv2.bitwise_and(mask_purple, mask_cement)

    # --- 🧪 Optional: Enhanced segmentation for hard images (tone-on-tone) ---
    used_hard_mode = False
    if hard_mode:
        try:
            # Illumination normalization on V channel (reduces shading / uneven lighting)
            hsv0 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v = hsv0[:, :, 2].astype(np.float32)
            v_blur = cv2.GaussianBlur(v, (0, 0), sigmaX=35, sigmaY=35)
            v_corr = cv2.normalize(v - v_blur + 128.0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            hsv_corr = hsv0.copy()
            hsv_corr[:, :, 2] = v_corr
            bgr_corr = cv2.cvtColor(hsv_corr, cv2.COLOR_HSV2BGR)

            # Stronger CLAHE on L* (boosts local contrast without changing the default pipeline)
            lab2 = cv2.cvtColor(bgr_corr, cv2.COLOR_BGR2LAB)
            l2, a2, b2 = cv2.split(lab2)
            clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l2 = clahe2.apply(l2)
            lab2 = cv2.merge([l2, a2, b2])
            bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

            hsv2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)
            # Slightly relaxed thresholds for low-saturation purple (tone-on-tone cases)
            lower_p = np.array([110, 10, 20])
            upper_p = np.array([179, 255, 255])
            mask_hsv = cv2.inRange(hsv2, lower_p, upper_p)

            # LAB-based magenta detector (helps when hue/saturation is weak)
            lab3 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2LAB)
            L3, A3, B3 = cv2.split(lab3)
            mask_lab_a = cv2.inRange(A3, 135, 255)
            mask_lab_b = cv2.inRange(B3, 0, 170)
            mask_lab = cv2.bitwise_and(mask_lab_a, mask_lab_b)

            mask_purple_hard = cv2.bitwise_or(mask_hsv, mask_lab)

            # Cleanup (reuse user kernel slider)
            merge_kernel2 = np.ones((kernel_size_purple * 3, kernel_size_purple * 3), dtype=np.uint8)
            mask_purple_hard = cv2.morphologyEx(mask_purple_hard, cv2.MORPH_CLOSE, merge_kernel2)
            mask_purple_hard = cv2.morphologyEx(
                mask_purple_hard, cv2.MORPH_OPEN, np.ones((kernel_size_purple, kernel_size_purple), np.uint8)
            )

            # Restrict to specimen area
            mask_purple_hard = cv2.bitwise_and(mask_purple_hard, mask_cement)

            # Sanity check (avoid catastrophic masks); fallback to default if suspicious
            area_total_tmp = int(np.count_nonzero(mask_cement))
            area_hard_tmp = int(np.count_nonzero(mask_purple_hard))
            frac = (area_hard_tmp / area_total_tmp) if area_total_tmp > 0 else 0.0

            if 0.001 <= frac <= 0.995:
                mask_purple = mask_purple_hard
                used_hard_mode = True
            else:
                mask_purple = mask_purple_default
        except Exception:
            mask_purple = mask_purple_default
    else:
        mask_purple = mask_purple_default

    # ✅ mask_gray = mask_cement - mask_purple (definition)
    mask_gray = cv2.bitwise_and(mask_cement, cv2.bitwise_not(mask_purple))


    # Final mask for purple on cement
    mask_red_on_cement = cv2.bitwise_and(mask_purple, mask_cement)

    # --- 📏 Optional calibration (2-click) ---
    with st.expander("📏 Optional Calibration (2 clicks) — px→mm (add-on)", expanded=False):
        st.markdown(
            "Use this only if you want to **override the pixel scale (mm/pixel)** using a known distance. "
            "If you do not use calibration, the app behaves exactly as before."
        )

        if st_canvas is None:
            st.warning(
                "Calibration canvas is not available in your environment. "
                "To enable the 2-click tool, install the optional dependency: "
                "`pip install streamlit-drawable-canvas` and restart the app."
            )
            with st.expander("Manual override (no extra install)", expanded=False):
                st.caption("Optional fallback: manually set the pixel scale (mm/pixel). This does not change the core algorithm—only the px→mm conversion.")
                if "calib_mm_per_px" not in st.session_state:
                    st.session_state.calib_mm_per_px = None
                mm_per_px_manual = st.number_input(
                    "Override scale (mm/pixel)",
                    min_value=0.0,
                    value=float(st.session_state.calib_mm_per_px) if st.session_state.calib_mm_per_px else 0.0,
                    step=0.0001,
                    format="%.6f",
                    help="If you enter a value here, it will override the specimen dimension-based scale."
                )
                colA, colB = st.columns([1,1])
                with colA:
                    if st.button("Apply manual scale", key="apply_manual_scale"):
                        if mm_per_px_manual and mm_per_px_manual > 0:
                            st.session_state.calib_mm_per_px = float(mm_per_px_manual)
                            st.success(f"Manual calibration applied: {st.session_state.calib_mm_per_px:.6f} mm/pixel")
                        else:
                            st.error("Please enter a value > 0.")
                with colB:
                    if st.button("Reset calibration", key="reset_manual_scale"):
                        st.session_state.calib_mm_per_px = None
                        st.session_state.calib_px_dist = None
                        st.session_state.calib_known_mm = None
                        st.success("Calibration cleared.")
        else:
            # Persist calibration in session state
            if "calib_mm_per_px" not in st.session_state:
                st.session_state.calib_mm_per_px = None
            if "calib_px_dist" not in st.session_state:
                st.session_state.calib_px_dist = None
            if "calib_known_mm" not in st.session_state:
                st.session_state.calib_known_mm = None
            if "use_calibration" not in st.session_state:
                st.session_state.use_calibration = False
            if "override_width_with_calibration" not in st.session_state:
                st.session_state.override_width_with_calibration = False

            use_calib = st.toggle(
                "Use calibration (override px→mm scale)",
                value=bool(st.session_state.use_calibration),
                help="When enabled, the carbonation depth scale will use your 2-point calibration instead of diameter/base ÷ pixel width."
            )
            st.session_state.use_calibration = use_calib

            col_c1, col_c2 = st.columns([1, 1])
            with col_c1:
                known_mm = st.number_input(
                    "Known distance (mm)",
                    min_value=0.1,
                    value=float(st.session_state.calib_known_mm) if st.session_state.calib_known_mm else 50.0,
                    format="%.2f",
                    help="Enter the real-world distance between the two points you will mark."
                )
            with col_c2:
                st.session_state.override_width_with_calibration = st.checkbox(
                    "Also override specimen width for volume (optional)",
                    value=bool(st.session_state.override_width_with_calibration),
                    help="If checked, the program will estimate specimen width (diameter/base) from the mask width using the calibrated mm/pixel. Height remains the value you entered."
                )

            # Prepare canvas size while preserving aspect ratio
            pil_bg = Image.fromarray(image_rgb)
            img_w, img_h = pil_bg.size
            canvas_w = min(720, img_w)
            scale = img_w / canvas_w
            canvas_h = int(img_h / scale)

            st.caption("Draw **one line** over a feature with known length (e.g., specimen diameter/base).")
            try:
                canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=3,
                stroke_color="#1f4e79",
                background_image=pil_bg.resize((canvas_w, canvas_h)),
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode="line",
                key="calibration_canvas",
            )
            except Exception:
                canvas_result = None
                st.info("Calibration canvas is not available due to a Streamlit compatibility issue. Your results are unaffected — you can keep using manual specimen dimensions, or install compatible versions of Streamlit and streamlit-drawable-canvas.")


            col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
            with col_b1:
                calibrate_now = st.button("📏 Calibrate", use_container_width=True)
            with col_b2:
                reset_calib = st.button("↩️ Reset calibration", use_container_width=True)
            with col_b3:
                st.caption("Tip: if you drew multiple lines, only the **last** one is used.")

            if reset_calib:
                st.session_state.calib_mm_per_px = None
                st.session_state.calib_px_dist = None
                st.session_state.calib_known_mm = None
                st.session_state.use_calibration = False
                st.session_state.override_width_with_calibration = False
                st.success("Calibration reset.")

            # Compute calibration on demand
            if calibrate_now and (canvas_result is not None):
                try:
                    data = canvas_result.json_data
                    objs = (data.get("objects") if isinstance(data, dict) else None) or []
                    if not objs:
                        st.warning("Draw a line on the image before calibrating.")
                    else:
                        obj = objs[-1]
                        x1, y1, x2, y2 = obj.get("x1"), obj.get("y1"), obj.get("x2"), obj.get("y2")
                        if None in (x1, y1, x2, y2):
                            st.warning("Could not read the line coordinates. Please draw the line again.")
                        else:
                            px_dist_canvas = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
                            px_dist_img = px_dist_canvas * scale  # convert to original image pixel scale
                            if px_dist_img <= 1.0:
                                st.warning("The selected distance is too small. Please draw a longer line.")
                            else:
                                mm_per_px = float(known_mm) / px_dist_img
                                st.session_state.calib_mm_per_px = mm_per_px
                                st.session_state.calib_px_dist = px_dist_img
                                st.session_state.calib_known_mm = float(known_mm)
                                st.session_state.use_calibration = True
                                st.success(f"Calibration set: {px_dist_img:.1f} px = {known_mm:.2f} mm → {mm_per_px:.6f} mm/px")
                except Exception as e:
                    st.warning(f"Calibration failed: {e}")

            # Display current calibration
            if st.session_state.calib_mm_per_px and st.session_state.calib_px_dist and st.session_state.calib_known_mm:
                st.info(
                    f"Current calibration: **{st.session_state.calib_px_dist:.1f} px = {st.session_state.calib_known_mm:.2f} mm** "
                    f"→ **{st.session_state.calib_mm_per_px:.6f} mm/px**"
                )
    # --- 📊 Quantitative Results ---
    area_total = cv2.countNonZero(mask_cement)
    area_red = cv2.countNonZero(mask_purple)
    area_gray = cv2.countNonZero(mask_gray)
    percent_red = (area_red / area_total) * 100 if area_total > 0 else 0
    percent_carbonated = 100 - percent_red

    # UI note (does not affect results)
    if hard_mode:
        if used_hard_mode:
            st.caption("🧪 Hard image mode: enhanced segmentation applied (experimental).")
        else:
            st.caption("🧪 Hard image mode enabled, but enhanced segmentation was not reliable; default segmentation was used.")


    # --- 📏 Carbonation Depth (Final Version) ---
    x_indices = np.where(np.any(mask_cement > 0, axis=0))[0]
    if len(x_indices) == 0:
        # Fallback: keep manual dimensions (cannot infer width from mask)
        st.warning("No cement region detected. Depth calculation and calibration-based scaling are disabled for this image.")
        width_for_scale_mm = d_mm if specimen_shape == "Cylindrical" else base_mm
        if specimen_shape == "Cylindrical":
            r = width_for_scale_mm / 2
            volume_total = pi * (r ** 2) * h_mm
        else:
            volume_total = base_mm * h_mm * depth_mm
        volume_preserved = volume_total * (percent_red / 100)
        volume_carbonated = volume_total * (percent_carbonated / 100)
    else:
        x_left, x_right = x_indices[0], x_indices[-1]
        width_cement_px = x_right - x_left + 1

        # Default pixel scale from manual dimension
        width_manual_mm = d_mm if specimen_shape == "Cylindrical" else base_mm
        mm_per_pixel_default = width_manual_mm / width_cement_px

        # Optional override from 2-click calibration (v11 add-on)
        use_calib = bool(st.session_state.get("use_calibration", False))
        calib_mm_per_px = st.session_state.get("calib_mm_per_px", None)
        if use_calib and calib_mm_per_px:
            mm_per_pixel = float(calib_mm_per_px)
            scale_source = "calibration"
        else:
            mm_per_pixel = float(mm_per_pixel_default)
            scale_source = "manual dimension"

        # Optional: override specimen width used for volume calculations (diameter/base) using mask width + calibrated mm/px
        width_for_calc_mm = width_manual_mm
        if (
            scale_source == "calibration"
            and bool(st.session_state.get("override_width_with_calibration", False))
        ):
            width_for_calc_mm = width_cement_px * mm_per_pixel
            st.caption(f"📏 Using calibration to estimate specimen width: {width_for_calc_mm:.2f} mm (from mask width).")

        # Volume estimation (uses width_for_calc_mm; height/depth from user input)
        if specimen_shape == "Cylindrical":
            r = width_for_calc_mm / 2
            volume_total = pi * (r ** 2) * h_mm
        else:
            volume_total = width_for_calc_mm * h_mm * depth_mm
        volume_preserved = volume_total * (percent_red / 100)
        volume_carbonated = volume_total * (percent_carbonated / 100)
    # Depth sampling across the detected cement width (requires a valid cement mask)
    if len(x_indices) == 0:
        depths = []
        avg_depth, std_depth = 0.0, 0.0
    else:
        x_left, x_right = x_indices[0], x_indices[-1]
        width_cement_px = x_right - x_left + 1
        height = mask_cement.shape[0]
        center_y = height // 2
        offset = int(0.05 * height)
        y_lines = [center_y - offset, center_y, center_y + offset]  # top, center, bottom

        # Fill small holes in purple mask
        kernel = np.ones((5,5), np.uint8)
        mask_purple_filled = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel)

        depths = []
        for y in y_lines:
            row_purple = mask_purple_filled[y, :]
            L_purple = np.sum(row_purple > 0)

            depth = (width_cement_px - L_purple) * mm_per_pixel / 2
            depths.append(depth)

        # --- Média e desvio padrão das 3 medidas ---
        avg_depth = statistics.mean(depths)
        std_depth = statistics.stdev(depths) if len(depths) > 1 else 0.0

    # --- 📦 Pack variables for Image Analysis PDF report ---
    def _to_u8_mask(m):
        """Ensure a mask is uint8 in 0..255 (works for 0/1 or 0/255 masks)."""
        m = m.astype(np.uint8)
        if m.size and m.max() <= 1:
            m = m * 255
        return m

    _mask_cement_u8 = _to_u8_mask(mask_cement)
    _mask_alkaline_u8 = _to_u8_mask(mask_purple)

    st.session_state["img_report_data"] = {
        "sample_name_default": uploaded_file.name.split('.')[0],
        "area_total": int(area_total),
        "area_red": int(area_red),
        "percent_red": float(percent_red),
        "percent_carbonated": float(percent_carbonated),
        "original_img": image_rgb.copy(),
        "mask_cement_img": _mask_cement_u8.copy(),
        "mask_alkaline_img": _mask_alkaline_u8.copy(),
        "dimension": float(d_mm if specimen_shape == "Cylindrical" else base_mm),
        "height_mm": float(h_mm),
        "volume_total": float(volume_total),
        "volume_preserved": float(volume_preserved),
        "volume_carbonated": float(volume_carbonated),
        "avg_depth": float(avg_depth),
        "std_depth": float(std_depth),
        "depth_list": [float(x) for x in depths],
    }

# --- 🖼️ Display Results ---
    st.subheader("📸 Results Overview")
    col_img, col_res = st.columns([2, 3])
    with col_img:
        st.image(image_rgb, caption="Original Image", use_container_width=True)
    with col_res:
        st.markdown(f"""
        **📊 Area Analysis**  
        - Total cement area: {area_total:,} px  
        - Alkaline area (red): {area_red:,} px  
        - Preserved alkalinity: **{percent_red:.2f}%**  
        - Estimated carbonation: **{percent_carbonated:.2f}%**  

        **📦 Volume Estimation**  
        - Total volume: {volume_total:.2f} mm³  
        - Preserved alkaline volume: {volume_preserved:.2f} mm³  
        - Carbonated volume: {volume_carbonated:.2f} mm³  

        **📐 Depth Measurement**  
        - Average depth: **{avg_depth:.2f} mm**  
        - Std dev: {std_depth:.2f} mm
        """)

    # --- 📄 1. PDF Report (Image Analysis Only) ---
    
    # --- ✅ Quality Check (QA Flags) ---
    st.subheader("✅ Quality Check")
    try:
        qa_status, qa_severity, qa_flags, qa_metrics = evaluate_quality_checks(
            enhanced_bgr, mask_cement, mask_red_on_cement, percent_red
        )

        if qa_severity == "ok":
            st.success(f"**{qa_status}** — The image and masks look consistent.")
        elif qa_severity == "caution":
            st.warning(f"**{qa_status}** — Results may be sensitive to lighting/segmentation. Please review.")
        else:
            st.error(f"**{qa_status}** — Please review the image acquisition and segmentation settings before reporting.")

        if qa_flags:
            st.markdown("**Flags:**")
            for f in qa_flags:
                st.markdown(f"- {f}")
        show_qa_metrics = st.checkbox("Show QA metrics (advanced)", value=False)
        if show_qa_metrics:
            import pandas as pd
            qa_metrics_pretty = {}
            for k, v in qa_metrics.items():
                qa_metrics_pretty[k] = round(v, 4) if isinstance(v, float) else v
            qa_df = pd.DataFrame({"Metric": list(qa_metrics_pretty.keys()), "Value": list(qa_metrics_pretty.values())})
            st.dataframe(qa_df, use_container_width=True, hide_index=True)

    except Exception as _qa_err:
        st.info("Quality Check is temporarily unavailable for this image.")

    st.subheader("📄 Generate Reports")
    sample_name = st.text_input("🧾 Sample Name", value=uploaded_file.name.split('.')[0] if uploaded_file else "sample")

    def generate_image_pdf(sample_name, area_total, area_red, percent_red, percent_carbonated,
                        original_img, mask_cement_img, mask_alkaline_img,
                        dimension, height_mm, volume_total, volume_preserved, volume_carbonated,
                        avg_depth, std_depth, depth_list):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width_a4, height_a4 = A4

        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width_a4 / 2, height_a4 - 110, "Image Analysis Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, height_a4 - 140, f"Sample: {sample_name}")
        c.drawString(50, height_a4 - 160, f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

        # --- Quantitative Results ---
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height_a4 - 190, "Quantitative Results:")
        c.setFont("Helvetica", 11)
        c.drawString(60, height_a4 - 210, f"Total cement area: {area_total} pixels")
        c.drawString(60, height_a4 - 230, f"Alkaline area: {area_red} pixels")
        c.drawString(60, height_a4 - 250, f"Preserved alkalinity: {percent_red:.2f}%")
        c.drawString(60, height_a4 - 270, f"Estimated carbonation: {percent_carbonated:.2f}%")
        c.drawString(60, height_a4 - 290, f"Diameter/Base: {dimension:.2f} mm")
        c.drawString(60, height_a4 - 310, f"Height: {height_mm:.2f} mm")
        c.drawString(60, height_a4 - 330, f"Total volume: {volume_total:.2f} mm³")
        c.drawString(60, height_a4 - 350, f"Preserved alkaline volume: {volume_preserved:.2f} mm³")
        c.drawString(60, height_a4 - 370, f"Estimated carbonated volume: {volume_carbonated:.2f} mm³")
        c.drawString(60, height_a4 - 390, f"Average carbonation depth: {avg_depth:.2f} mm")
        c.drawString(60, height_a4 - 410, f"Standard deviation: {std_depth:.2f} mm")

        # depth list (evita estourar linha)
        depth_str = ", ".join([f"{d:.2f}" for d in depth_list]) + " mm"
        c.drawString(60, height_a4 - 430, "Depth measurements:")
        c.setFont("Helvetica", 10)
        c.drawString(60, height_a4 - 445, depth_str[:120])
        if len(depth_str) > 120:
            c.drawString(60, height_a4 - 458, depth_str[120:240])

        # --- Images ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img1, \
            tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img2, \
            tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img3:

            Image.fromarray(original_img).save(img1.name)
            Image.fromarray(mask_cement_img, "L").save(img2.name)
            Image.fromarray(mask_alkaline_img, "L").save(img3.name)

            y_img = 180
            c.drawImage(ImageReader(img1.name), 50,  y_img, width=150, height=150, preserveAspectRatio=True, anchor='c')
            c.drawImage(ImageReader(img2.name), 220, y_img, width=150, height=150, preserveAspectRatio=True, anchor='c')
            c.drawImage(ImageReader(img3.name), 390, y_img, width=150, height=150, preserveAspectRatio=True, anchor='c')

        c.setFont("Helvetica-Oblique", 9)
        c.drawString(50,  165, "Original Image")
        c.drawString(220, 165, "Cement Mask")
        c.drawString(390, 165, "Alkaline Area")

        # --- Footer ---
        footer_y = 50
        c.setFont("Helvetica-Bold", 10)
        c.setFillColorRGB(0.2, 0.2, 0.5)
        c.drawString(50, footer_y + 30, "Developed by:")

        try:
            lab_logo_path = resolve_resource("assets/logola.png", "assets/logola.jpg", "logola.png", "logola.jpg", "assets/Logo.png", "Logo.png")
            if lab_logo_path is not None:
                lab_logo = ImageReader(str(lab_logo_path))
            c.drawImage(lab_logo, 50, footer_y + 5, width=80, height=20, mask='auto')
        except Exception:
            c.setFillColorRGB(0, 0, 0)
            c.drawString(50, footer_y + 20, "Lab Logo Not Found")

        c.setFont("Helvetica", 9)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(50, footer_y - 10, "Technological Center for Oil Well Cementing (NTCPP)")
        c.drawString(50, footer_y - 20, "Federal University of Rio Grande do Norte (UFRN)")
        c.drawString(50, footer_y - 30, "Campus Universitário - Lagoa Nova, ZIP 59078-900, Natal/RN - Brazil")
        c.drawString(50, footer_y - 40, "ntcpp.ufrn@gmail.com | www.labcim.com.br")

        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer


    # ✅ condição mínima: só habilita se a análise de imagem já rodou e você tem os outputs
    can_make_img_report = (
        "area_total" in globals() or "area_total" in st.session_state
    )

    # Melhor: use as suas variáveis reais aqui:
    
    if st.button("📥 Generate Image Analysis Report", key="btn_img_report"):
            try:
                if "img_report_data" not in st.session_state:
                    st.error("No image analysis results available yet. Please upload and process an image first.")
                else:
                    D = st.session_state["img_report_data"]

                    pdf_img = generate_image_pdf(
                        sample_name=sample_name,
                        area_total=D["area_total"],
                        area_red=D["area_red"],
                        percent_red=D["percent_red"],
                        percent_carbonated=D["percent_carbonated"],
                        original_img=D["original_img"],
                        mask_cement_img=D["mask_cement_img"],
                        mask_alkaline_img=D["mask_alkaline_img"],
                        dimension=D["dimension"],
                        height_mm=D["height_mm"],
                        volume_total=D["volume_total"],
                        volume_preserved=D["volume_preserved"],
                        volume_carbonated=D["volume_carbonated"],
                        avg_depth=D["avg_depth"],
                        std_depth=D["std_depth"],
                        depth_list=D["depth_list"],
                    )

                    st.session_state["pdf_image_bytes"] = pdf_img.getvalue()
                    st.success("✅ Image Analysis PDF generated. Use the download button below.")
            except Exception as e:
                st.error(f"Failed to generate Image Analysis PDF: {str(e)}")

    # Download button (visible if PDF exists)
    if "pdf_image_bytes" in st.session_state and st.session_state["pdf_image_bytes"]:
        st.download_button(
            "📄 Download Image Analysis Report",
            data=st.session_state["pdf_image_bytes"],
            file_name=f"ImageReport_{sample_name}.pdf",
            mime="application/pdf",
        )
# --- 📈 Carbonation Kinetics Module ---
    st.subheader("📈 Carbonation Kinetics Analysis")
    st.markdown("Enter depth vs. time data to fit carbonation kinetics models (e.g., $X = k \sqrt{t}$).")

    # --- Assay type (controls default model suggestion)
    if "assay_type" not in st.session_state:
        st.session_state.assay_type = "Natural carbonation (start at t=0)"

    assay_type = st.radio(
        "🧪 Assay Type",
        ["Natural carbonation (start at t=0)", "Accelerated carbonation"],
        help="Choose 'Accelerated' if curing/pre-exposure may have caused initial carbonation (non-zero intercept).",
        key="assay_type_radio",
        horizontal=False,
    )
    st.session_state.assay_type = assay_type

    # --- Initial kinetics table
    if "kinetics_data" not in st.session_state:
        st.session_state.kinetics_data = [
            {"time_days": 30.0, "depth_mm": 0.91},
            {"time_days": 60.0, "depth_mm": 2.85},
            {"time_days": 90.0, "depth_mm": 4.11},
        ]

    df_kinetics = pd.DataFrame(st.session_state.kinetics_data)
    df_edited = st.data_editor(
        df_kinetics,
        key="kinetics_editor",
        column_config={
            "time_days": st.column_config.NumberColumn("Time (days)", min_value=0.1, format="%.2f"),
            "depth_mm": st.column_config.NumberColumn("Depth (mm)", min_value=0.0, format="%.2f"),
        },
        num_rows="dynamic",
        use_container_width=True,
    )
    st.session_state.kinetics_data = df_edited.to_dict("records")

    # --- Model definitions (t in years)
    def model_fick_forced(t_years, k):
        return k * np.sqrt(t_years)

    def model_fick_intercept(t_years, k, b):
        return k * np.sqrt(t_years) + b

    def model_saturating_intercept(t_years, c, a, b):
        # z(t) = c + a*sqrt(t) + b*sqrt(t)/(sqrt(t)+1)
        s = np.sqrt(t_years)
        return c + a * s + b * s / (s + 1.0)

    # --- Model choice
    model_options = [
        "Fick (X = k√t) – forced b=0",
        "Fick (X = k√t + b) – with intercept",
        "Saturating model: z(t) = c + a√t + b·√t/(√t+1)",
    ]
    default_model = (
        "Fick (X = k√t) – forced b=0"
        if assay_type == "Natural carbonation (start at t=0)"
        else "Fick (X = k√t + b) – with intercept"
    )

    model_choice = st.selectbox(
        "📌 Kinetics Model",
        model_options,
        index=model_options.index(default_model),
        key="model_choice_select",
        help="For natural exposure, a forced origin model is often adequate. For accelerated exposure, allow an intercept.",
    )

    # --- Extract and validate data
    times_days = df_edited.get("time_days", pd.Series(dtype=float)).to_numpy(dtype=float)
    depths_mm = df_edited.get("depth_mm", pd.Series(dtype=float)).to_numpy(dtype=float)

    fig = None
    params = {}
    r2 = None
    mse = None
    rmse = None
    pred_10 = None

    if len(times_days) >= 3:
        valid = [(t, d) for t, d in zip(times_days, depths_mm) if np.isfinite(t) and np.isfinite(d) and t > 0 and d >= 0]
        if len(valid) < 3:
            st.warning("Need at least 3 valid points (time > 0, depth ≥ 0).")
        else:
            t_days, y = map(np.array, zip(*valid))
            t_years = t_days / 365.25

            # Guard against zero/negative years
            if np.any(t_years <= 0):
                st.warning("Time values must be positive.")
            else:
                try:
                    # --- Fit according to selected model
                    if model_choice.startswith("Fick (X = k√t) – forced"):
                        s = np.sqrt(t_years)
                        k_hat = float(np.sum(y * s) / np.sum(s ** 2))
                        y_hat = model_fick_forced(t_years, k_hat)

                        params = {"k": k_hat, "b": 0.0, "model": "fick_forced"}
                        eq_label = rf"$X = {k_hat:.2f}\,\sqrt{{t}}$"

                    elif model_choice.startswith("Fick (X = k√t + b)"):
                        # Linear regression on sqrt(t)
                        s = np.sqrt(t_years)
                        slope, intercept, r_val, p_val, std_err = linregress(s, y)
                        k_hat = float(slope)
                        b_hat = float(intercept)
                        y_hat = model_fick_intercept(t_years, k_hat, b_hat)

                        params = {"k": k_hat, "b": b_hat, "model": "fick_intercept"}
                        eq_label = rf"$X = {k_hat:.2f}\,\sqrt{{t}} {b_hat:+.2f}$"

                    else:
                        # Saturating with intercept c (NOT forced through 0)
                        # initial guesses: c ~ y at earliest time, a ~ k from forced, b ~ 0
                        s = np.sqrt(t_years)
                        k0 = float(np.sum(y * s) / np.sum(s ** 2))
                        c0 = float(y[np.argmin(t_years)])
                        p0 = [c0, k0, 0.0]
                        popt, pcov = curve_fit(
                            model_saturating_intercept,
                            t_years,
                            y,
                            p0=p0,
                            maxfev=20000,
                        )
                        c_hat, a_hat, b_hat = map(float, popt)
                        y_hat = model_saturating_intercept(t_years, c_hat, a_hat, b_hat)

                        params = {"c": c_hat, "a": a_hat, "b": b_hat, "model": "saturating_intercept"}
                        eq_label = rf"$z(t)= {c_hat:.2f} + {a_hat:.2f}\sqrt{{t}} {b_hat:+.2f}\frac{{\sqrt{{t}}}}{{\sqrt{{t}}+1}}$"

                    # --- Metrics
                    resid = y - y_hat
                    mse = float(np.mean(resid ** 2))
                    rmse = float(np.sqrt(mse))
                    ss_res = float(np.sum(resid ** 2))
                    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                    r2 = max(min(r2, 1.0), 0.0)

                    # --- Prediction at 10 years
                    t10 = 10.0
                    if params.get("model") == "fick_forced":
                        pred_10 = float(model_fick_forced(np.array([t10]), params["k"])[0])
                    elif params.get("model") == "fick_intercept":
                        pred_10 = float(model_fick_intercept(np.array([t10]), params["k"], params["b"])[0])
                    else:
                        pred_10 = float(model_saturating_intercept(np.array([t10]), params["c"], params["a"], params["b"])[0])

                    # --- Plot
                    t_smooth = np.linspace(0.001, max(1.0, float(np.max(t_years)) * 1.05), 250)
                    if params.get("model") == "fick_forced":
                        y_smooth = model_fick_forced(t_smooth, params["k"])
                    elif params.get("model") == "fick_intercept":
                        y_smooth = model_fick_intercept(t_smooth, params["k"], params["b"])
                    else:
                        y_smooth = model_saturating_intercept(t_smooth, params["c"], params["a"], params["b"])

                    y_max = max(float(np.max(y_smooth)), float(np.max(y))) * 1.10

                    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=220)

                    ax.scatter(

                        t_years, y, s=46, marker='o',

                        facecolors='white', edgecolors='black', linewidths=0.9,

                        label='Measured data', zorder=3

                    )

                    ax.plot(t_smooth, y_smooth, linewidth=2.2, label=eq_label, zorder=2)

                    ax.set_xlabel('Exposure time (years)')

                    ax.set_ylabel('Carbonation depth (mm)')

                    ax.set_title('Carbonation kinetics')

                    ax.set_xlim(0, max(1.0, float(np.max(t_years)) * 1.05))

                    ax.set_ylim(0, y_max)

                    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.35)

                    ax.tick_params(direction='out', length=4, width=1)

                    for spine in ['top', 'right']:

                        ax.spines[spine].set_visible(False)

                    ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='0.85')

                    fig.tight_layout()

                    st.pyplot(fig, use_container_width=True)

                    # --- KPIs
                    c1, c2, c3, c4 = st.columns(4)

                    if params.get("model") in ["fick_forced", "fick_intercept"]:
                        c1.metric("k", f"{params['k']:.2f}", "mm/√year")
                        c2.metric("b", f"{params.get('b', 0.0):.2f}", "mm")
                    else:
                        c1.metric("a", f"{params['a']:.2f}", "mm/√year")
                        c2.metric("c", f"{params['c']:.2f}", "mm")

                    c3.metric("R²", f"{r2:.8f}")
                    c4.metric("RMSE", f"{rmse:.5f}", "mm")

                    if r2 is not None and r2 > 0.9999:
                        st.caption(f"Precision note: 1 − R² = {(1 - r2):.2e}")

                    st.caption(f"MSE: {mse:.4f} mm² • Prediction at 10 years: {pred_10:.2f} mm")

                    # --- Model comparison (reference)
                    with st.expander("🔍 Model Comparison (reference)"):
                        s = np.sqrt(t_years)
                        # forced
                        k_forced = float(np.sum(y * s) / np.sum(s ** 2))
                        y_forced = model_fick_forced(t_years, k_forced)
                        mse_forced = float(np.mean((y - y_forced) ** 2))
                        rmse_forced = float(np.sqrt(mse_forced))
                        ss_res = float(np.sum((y - y_forced) ** 2))
                        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                        r2_forced = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                        r2_forced = max(min(r2_forced, 1.0), 0.0)

                        # intercept
                        slope, intercept, r_val, p_val, std_err = linregress(s, y)
                        y_int = model_fick_intercept(t_years, float(slope), float(intercept))
                        mse_int = float(np.mean((y - y_int) ** 2))
                        rmse_int = float(np.sqrt(mse_int))
                        ss_res = float(np.sum((y - y_int) ** 2))
                        r2_int = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                        r2_int = max(min(r2_int, 1.0), 0.0)

                        st.markdown(
                            f"""
| Model | Parameters | R² | RMSE (mm) | MSE (mm²) |
|---|---:|---:|---:|---:|
| Fick forced | k={k_forced:.2f}, b=0 | {r2_forced:.5f} | {rmse_forced:.5f} | {mse_forced:.4f} |
| Fick + intercept | k={float(slope):.2f}, b={float(intercept):+.2f} | {r2_int:.3f} | {rmse_int:.3f} | {mse_int:.4f} |
"""
                        )

                except Exception as e:
                    st.error(f"Model fitting failed: {e}")
    else:
        st.info("Enter at least 3 data points to enable kinetic analysis.")

    # --- 📄 Kinetics PDF Report ---
    if fig is not None and r2 is not None and mse is not None and rmse is not None:
        if st.button("📥 Generate Kinetics Analysis Report", key="btn_kin_report"):
            def generate_kinetics_pdf(sample_name, assay_type, model_choice, df_kinetics, fig, params, r2, mse, rmse, pred_10):
                buffer = io.BytesIO()
                c = canvas.Canvas(buffer, pagesize=A4)
                width_a4, height_a4 = A4

                # Header
                c.setFont("Helvetica-Bold", 16)
                c.drawCentredString(width_a4 / 2, height_a4 - 70, "Kinetics Analysis Report")

                c.setFont("Helvetica", 11)
                c.drawString(50, height_a4 - 100, f"Sample: {sample_name}")
                c.drawString(50, height_a4 - 118, f"Assay Type: {assay_type}")
                c.drawString(50, height_a4 - 136, f"Model: {model_choice}")
                c.drawString(50, height_a4 - 154, f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

                # Table
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, height_a4 - 185, "Time vs Depth Data:")
                c.setFont("Helvetica", 10)
                y_table = height_a4 - 205
                c.drawString(60, y_table, "Time (days)")
                c.drawString(160, y_table, "Depth (mm)")
                y_table -= 12

                for _, row in df_kinetics.iterrows():
                    y_table -= 14
                    c.drawString(60, y_table, f"{float(row['time_days']):.2f}")
                    c.drawString(160, y_table, f"{float(row['depth_mm']):.2f}")

                # Metrics
                y_metrics = y_table - 30
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y_metrics, "Fitted Parameters & Metrics:")
                c.setFont("Helvetica", 11)

                y_line = y_metrics - 18
                if params.get("model") in ["fick_forced", "fick_intercept"]:
                    c.drawString(60, y_line, f"k: {params['k']:.3f} mm/√year")
                    y_line -= 16
                    c.drawString(60, y_line, f"b: {params.get('b', 0.0):+.3f} mm")
                    y_line -= 16
                else:
                    c.drawString(60, y_line, f"c: {params['c']:.3f} mm")
                    y_line -= 16
                    c.drawString(60, y_line, f"a: {params['a']:.3f} mm/√year")
                    y_line -= 16
                    c.drawString(60, y_line, f"b: {params['b']:+.3f} (mm)")
                    y_line -= 16

                c.drawString(60, y_line, f"R²: {r2:.8f}")
                y_line -= 16
                c.drawString(60, y_line, f"RMSE: {rmse:.5f} mm")
                y_line -= 16
                c.drawString(60, y_line, f"MSE: {mse:.4f} mm²")
                y_line -= 16
                c.drawString(60, y_line, f"Prediction at 10 years: {pred_10:.2f} mm")

                # Plot
                plot_y = 220
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name, dpi=170, bbox_inches="tight")
                    c.drawImage(ImageReader(tmpfile.name), 50, plot_y, width=500, height=280)

                # Footer
                footer_y = 60
                c.setFont("Helvetica-Bold", 10)
                c.setFillColorRGB(0.2, 0.2, 0.5)
                c.drawString(50, footer_y + 30, "Developed by:")

                try:
                    lab_logo_path = resolve_resource("assets/logola.png", "assets/logola.jpg", "logola.png", "logola.jpg", "assets/Logo.png", "Logo.png")
                    if lab_logo_path is not None:
                        lab_logo = ImageReader(str(lab_logo_path))
                    c.drawImage(lab_logo, 50, footer_y + 5, width=90, height=22, mask="auto")
                except Exception:
                    c.setFillColorRGB(0, 0, 0)
                    c.drawString(50, footer_y + 12, "Lab logo not found (logola.png)")

                c.setFont("Helvetica", 9)
                c.setFillColorRGB(0, 0, 0)
                c.drawString(50, footer_y - 2, "Technological Center for Oil Well Cementing (NTCPP)")
                c.drawString(50, footer_y - 14, "Federal University of Rio Grande do Norte (UFRN)")
                c.drawString(50, footer_y - 26, "Campus Universitário - Lagoa Nova, ZIP 59078-900, Natal/RN - Brazil")
                c.drawString(50, footer_y - 38, "ntcpp.ufrn@gmail.com | www.labcim.com.br")

                c.showPage()
                c.save()
                buffer.seek(0)
                return buffer

            pdf_kin = generate_kinetics_pdf(
                sample_name=sample_name,
                assay_type=assay_type,
                model_choice=model_choice,
                df_kinetics=df_edited,
                fig=fig,
                params=params,
                r2=r2,
                mse=mse,
                rmse=rmse,
                pred_10=pred_10,
            )

            st.download_button(
                "📄 Download Kinetics Analysis Report",
                data=pdf_kin,
                file_name=f"Kinetics_{sample_name}.pdf",
                mime="application/pdf",
                key="dl_kin_pdf",
            )


# --- 🏁 Footer with lab logo ---
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("**Developed by:**", unsafe_allow_html=True)

try:
    lab_logo_path = resolve_resource("assets/logola.png", "assets/logola.jpg", "logola.png", "logola.jpg")
    if lab_logo_path is not None:
        lab_logo = Image.open(lab_logo_path)
        st.image(lab_logo, width=150)
except Exception:
    st.markdown("Lab logo (logola.png) not found in directory.", unsafe_allow_html=True)

st.markdown("""
Technological Center for Oil Well Cementing (NTCPP)<br>
Federal University of Rio Grande do Norte (UFRN)<br>
Campus Universitário - Lagoa Nova, ZIP 59078-900, Natal/RN - Brazil<br>
Email: ntcpp.ufrn@gmail.com | Website: <a href="http://www.labcim.com.br" target="_blank">www.labcim.com.br</a>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
