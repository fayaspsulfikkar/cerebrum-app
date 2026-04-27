"""
pipeline_utils.py - Helper functions for the TRIBE v2 brain analysis pipeline.

ROI mapping uses nilearn's fetch_atlas_surf_destrieux() which provides the
aparc.a2009s surface annotation files directly aligned to fsaverage5.
Each vertex is assigned a label index without any volumetric projection.
Vertex indices are built directly from surface labels - no approximation.
"""

import logging
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu

logger = logging.getLogger("tribe_pipeline")

# ---- TR Duration -------------------------------------------------------------
# TRIBE v2 fMRI sampling rate: one TR every 2.0 seconds (0.5 Hz).
TR_SECONDS = 2.0

# ---- Surface Destrieux label substrings for each ROI -------------------------
# fetch_atlas_surf_destrieux() labels are pure region names WITHOUT hemisphere
# prefix (e.g. "G_front_inf-Opercular", not "L G_front_inf-Opercular").
# We match both LH and RH vertices using the same name patterns.
DESTRIEUX_ROI_PATTERNS: Dict[str, List[str]] = {
    "broca": [
        "G_front_inf-Opercular",     # pars opercularis  (Brodmann area 44)
        "G_front_inf-Triangul",      # pars triangularis (Brodmann area 45)
        "G_and_S_frontomargin",      # frontal operculum adjacent area
    ],
    "amygdala": [
        # Cortical surface proxy: amygdala strongly co-activates with temporal pole & MTG
        "Pole_temporal",
        "G_temporal_middle",
    ],
    "nacc": [
        # Cortical surface proxy: nucleus accumbens circuitry via OFC and ACC
        "G_orbital",
        "G_and_S_cingul-Ant",
    ],
    "hippocampus": [
        # Parahippocampal gyrus is the correct cortical surface label
        "G_oc-temp_med-Parahip",
        "G_oc-temp_med-Lingual",
    ],
    "parietal": [
        "G_parietal_sup",
        "S_intrapariet_and_P_trans",
    ],
    "tpj": [
        "G_pariet_inf-Angular",
        "G_pariet_inf-Supramar",
        "G_temp_sup-Plan_tempo",
    ],
}

REGION_META: Dict[str, Dict] = {
    "broca":       {"label": "Broca Area",        "sub": "Language"},
    "amygdala":    {"label": "Amygdala",           "sub": "Emotion"},
    "nacc":        {"label": "Nucleus Accumbens",  "sub": "Reward"},
    "hippocampus": {"label": "Hippocampus",        "sub": "Memory"},
    "parietal":    {"label": "Superior Parietal",  "sub": "Attention"},
    "tpj":         {"label": "TPJ",                "sub": "Social"},
}

REGION_ORDER = ["broca", "amygdala", "nacc", "hippocampus", "parietal", "tpj"]

# Module-level cache - atlas is built once per process
_roi_indices: Dict[str, np.ndarray] = {}


def build_roi_indices() -> Dict[str, np.ndarray]:
    """
    Build vertex index arrays for each ROI directly from the Destrieux surface atlas.

    Uses nilearn.datasets.fetch_atlas_surf_destrieux() which provides the
    FreeSurfer aparc.a2009s annotation natively aligned to fsaverage5.

    - map_left  : int array (10242,) - Destrieux label per LH vertex
    - map_right : int array (10242,) - Destrieux label per RH vertex
    - labels    : list of (index, name) tuples

    No volumetric projection, no vol_to_surf, no interpolation.
    TRIBE output layout: vertices 0..10241 = LH, 10242..20483 = RH.

    Returns:
        Dict {roi_name: int array of vertex indices in [0, 20483]}
    """
    global _roi_indices
    if _roi_indices:
        return _roi_indices

    logger.info("Loading Destrieux surface atlas (aparc.a2009s) on fsaverage5 ...")
    from nilearn import datasets, surface

    atlas = datasets.fetch_atlas_surf_destrieux()

    # Load per-vertex integer label arrays directly from the annotation files.
    # surface.load_surf_data on an .annot file returns shape (N_vertices,) of label indices.
    lh_labels = surface.load_surf_data(atlas["map_left"]).astype(int)   # (10242,)
    rh_labels = surface.load_surf_data(atlas["map_right"]).astype(int)  # (10242,)
    assert len(lh_labels) == 10242, f"LH expected 10242 vertices, got {len(lh_labels)}"
    assert len(rh_labels) == 10242, f"RH expected 10242 vertices, got {len(rh_labels)}"

    # atlas["labels"] is a plain list of strings: ['Unknown', 'G_and_S_frontomargin', ...]
    # The list index IS the label index stored in the vertex arrays.
    label_map: Dict[int, str] = {i: name for i, name in enumerate(atlas["labels"])}

    logger.info(
        "Surface atlas loaded: %d labels, %d LH vertices, %d RH vertices",
        len(label_map), len(lh_labels), len(rh_labels),
    )
    logger.info("Sample labels: %s", list(label_map.values())[:6])

    def find_label_ids(patterns: List[str]) -> List[int]:
        """Return all label indices whose name contains any pattern (case-insensitive)."""
        matched = []
        for lid, lname in label_map.items():
            for pat in patterns:
                if pat.lower() in lname.lower():
                    matched.append(lid)
                    break
        return matched

    # Track all used vertex indices to verify distinctness
    used_lh: np.ndarray = np.zeros(10242, dtype=bool)
    used_rh: np.ndarray = np.zeros(10242, dtype=bool)

    for roi in REGION_ORDER:
        patterns    = DESTRIEUX_ROI_PATTERNS[roi]
        matched_ids = find_label_ids(patterns)

        if not matched_ids:
            raise RuntimeError(
                f"ROI '{roi}': no surface labels matched patterns {patterns}. "
                f"Available labels: {list(label_map.values())[:10]}"
            )

        # Get the binary masks in each hemisphere
        lh_mask = np.isin(lh_labels, matched_ids)
        rh_mask = np.isin(rh_labels, matched_ids)

        # Check overlap with already-allocated vertices
        lh_overlap = int((used_lh & lh_mask).sum())
        rh_overlap = int((used_rh & rh_mask).sum())
        if lh_overlap > 0 or rh_overlap > 0:
            logger.warning(
                "ROI '%s': %d LH + %d RH vertices overlap with a prior ROI "
                "(Destrieux regions can be shared across ROI definitions).",
                roi, lh_overlap, rh_overlap,
            )

        # Vertex indices: LH = 0..10241, RH = 10242..20483
        lh_idx = np.where(lh_mask)[0]
        rh_idx = np.where(rh_mask)[0] + 10242
        combined = np.concatenate([lh_idx, rh_idx])

        if len(combined) == 0:
            raise RuntimeError(
                f"ROI '{roi}' resolved to 0 vertices even though label IDs were found. "
                f"Matched IDs: {matched_ids}"
            )

        _roi_indices[roi] = combined
        used_lh |= lh_mask
        used_rh |= rh_mask

        matched_names = [label_map[i] for i in matched_ids]
        logger.info(
            "ROI %-14s: %4d LH + %4d RH = %5d vertices | labels=%s",
            roi, len(lh_idx), len(rh_idx), len(combined), matched_names,
        )

    # Cross-ROI distinctness report
    distinct = True
    all_used = []
    for roi, idx in _roi_indices.items():
        all_used.append(set(idx.tolist()))
    pairs = [(REGION_ORDER[i], REGION_ORDER[j])
             for i in range(len(REGION_ORDER))
             for j in range(i + 1, len(REGION_ORDER))]
    for r1, r2 in pairs:
        overlap = len(set(_roi_indices[r1].tolist()) & set(_roi_indices[r2].tolist()))
        if overlap > 0:
            logger.warning("Overlap between '%s' and '%s': %d shared vertices", r1, r2, overlap)
            distinct = False
    if distinct:
        logger.info("ROI distinctness check: all ROIs are non-overlapping.")
    else:
        logger.warning("Some ROIs share vertices (expected for adjacent Destrieux regions).")

    return _roi_indices


# ---- Core analysis helpers ---------------------------------------------------

def compute_roi_timeseries(preds: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Map TRIBE predictions (T, 20484) to per-ROI mean time-series.
    Computes the arithmetic mean over the selected vertex set for each ROI.
    """
    indices = build_roi_indices()
    return {roi: preds[:, indices[roi]].mean(axis=1) for roi in REGION_ORDER}


def smooth_timeseries(ts: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian-smooth a 1-D time series (sigma in TR units)."""
    return gaussian_filter1d(ts.astype(float), sigma=sigma)


def compute_hook_strength(roi_ts: Dict[str, np.ndarray], n_trs: int = 3) -> float:
    """
    Hook strength = max activation across all ROIs within the first n_trs TRs.
    With TR=2.0s, n_trs=3 covers the first 6.0 seconds of the video.
    """
    values = [float(ts[:n_trs].max()) for ts in roi_ts.values() if len(ts) >= 1]
    return float(np.max(values)) if values else 0.0


def compute_peak(roi_ts: Dict[str, np.ndarray]) -> Tuple[str, float, float]:
    """
    Find global peak activation across all ROIs.
    Returns: (peak_region, peak_time_seconds, peak_value)
    """
    best_region, best_val, best_tr = "broca", -np.inf, 0
    for roi, ts in roi_ts.items():
        if len(ts) == 0:
            continue
        idx = int(np.argmax(ts))
        val = float(ts[idx])
        if val > best_val:
            best_val, best_region, best_tr = val, roi, idx
    return best_region, float(best_tr * TR_SECONDS), best_val


def compute_retention_slope(ts: np.ndarray) -> float:
    """
    Linear slope over the last 25% of the time-series.
    Negative = activation drops quickly; Positive = sustained engagement.
    """
    if len(ts) < 4:
        return 0.0
    tail = ts[int(len(ts) * 0.75):]
    x = np.arange(len(tail), dtype=float)
    return float(np.polyfit(x, tail, 1)[0]) if x.std() > 0 else 0.0


def top5_tr_indices(roi_ts: Dict[str, np.ndarray]) -> List[int]:
    """Return top-5 TR indices by cross-ROI mean activation, in chronological order."""
    combined = np.stack(list(roi_ts.values()), axis=0).mean(axis=0)
    n = min(5, len(combined))
    return sorted(int(i) for i in np.argsort(combined)[::-1][:n])


def extract_top_frames(
    video_path: str,
    peak_trs: List[int],
    out_dir: str,
    video_id: int,
) -> List[Dict]:
    """Extract video frames at specified TR indices and save as JPEG."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        logger.warning("ID %d: FPS unreadable, defaulting to %.1f", video_id, fps)

    results = []
    for rank, tr_idx in enumerate(peak_trs):
        time_s = tr_idx * TR_SECONDS
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_s * fps))
        ret, frame = cap.read()
        if not ret:
            logger.warning("ID %d: no frame at %.2fs (TR=%d)", video_id, time_s, tr_idx)
            continue
        frame_path = os.path.join(out_dir, f"frame_{rank + 1}.jpg")
        cv2.imwrite(frame_path, frame)
        results.append({
            "rank": rank + 1,
            "tr": tr_idx,
            "time_s": round(time_s, 3),
            "frame_path": frame_path,
        })
    cap.release()
    return results


# ---- Statistics --------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two groups."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def run_mann_whitney(viral: np.ndarray, non_viral: np.ndarray) -> Dict:
    """Mann-Whitney U test + Cohen's d."""
    if len(viral) < 2 or len(non_viral) < 2:
        return {"u_stat": None, "p_value": None, "cohens_d": None}
    stat, p = mannwhitneyu(viral, non_viral, alternative="two-sided")
    return {
        "u_stat": float(stat),
        "p_value": float(p),
        "cohens_d": round(cohens_d(viral, non_viral), 4),
    }


# ---- Logging -----------------------------------------------------------------

def setup_logger(log_file: str = "logs/pipeline.log") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log = logging.getLogger("tribe_pipeline")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fmt = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", "%H:%M:%S")
        for h in [logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")]:
            h.setFormatter(fmt)
            log.addHandler(h)
    return log


def eta_string(elapsed_s: float, done: int, total: int) -> str:
    """Human-readable ETA string with per-video throughput."""
    if done == 0:
        return "ETA: unknown"
    per_item = elapsed_s / done
    h, rem = divmod(int(per_item * (total - done)), 3600)
    m, s = divmod(rem, 60)
    return f"ETA: {h:02d}h {m:02d}m {s:02d}s ({per_item:.1f}s/video)"
