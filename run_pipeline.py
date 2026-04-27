"""
run_pipeline.py -- TRIBE v2 Brain Analysis Pipeline Orchestrator
================================================================
Steps:
  0  Setup: directories, model load, clean old .npz
  1+2 Inference: per-video TRIBE prediction --> uploads/tribe_outputs/{id}.npz
  3  ROI summary CSV
  4  Temporal analysis (hook, peak, engagement curves)
  5  Frame-level key moments
  6  Audio vs Visual ablation (sample of 20 videos)
  7  Viral vs Non-Viral comparison
  8  Final outputs

Usage:
  python run_pipeline.py               # full run
  python run_pipeline.py --dry-run 3   # first 3 videos only
  python run_pipeline.py --step 3      # resume from step 3 onward
  python run_pipeline.py --no-ablation # skip step 6

IMPORTANT:
  This pipeline requires the real TRIBE v2 model weights.
  Run 'python download_model.py' before executing this script.
  No simulation or fallback will be used under any condition.
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ── CRITICAL: Force CPU-only mode before any torch/neuralset imports ──────────
# neuralset's audio and video extractors call torch.cuda.is_available()
# independently at runtime. On this machine torch is not compiled with CUDA,
# so any attempt to move tensors to "cuda" raises AssertionError.
# Monkeypatching is_available() to always return False prevents any extractor
# from choosing CUDA regardless of what device string is passed.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.cuda.is_available = lambda: False  # type: ignore[assignment]

# ── CRITICAL: Patch Windows os.kill() for neuralset/exca ──────────
# exca.cachedict.inflight uses os.kill(pid, 0) to check if a worker is alive.
# On Windows, if the process is dead, this throws OSError[WinError 87] instead
# of ProcessLookupError, crashing the pipeline. We intercept it here.
_orig_os_kill = os.kill
def patched_os_kill(pid, sig, *args, **kwargs):
    try:
        return _orig_os_kill(pid, sig, *args, **kwargs)
    except OSError as e:
        if sig == 0 and getattr(e, "winerror", None) == 87:
            raise ProcessLookupError("Windows os.kill ProcessLookupError") from e
        raise
os.kill = patched_os_kill
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pipeline_utils as pu
from pipeline_utils import (
    REGION_ORDER, REGION_META, TR_SECONDS,
    compute_roi_timeseries, compute_hook_strength, compute_peak,
    compute_retention_slope, extract_top_frames, top5_tr_indices,
    smooth_timeseries, run_mann_whitney, cohens_d, eta_string,
)

# ---- Directory layout --------------------------------------------------------
TRIBE_OUT_DIR   = Path("uploads/tribe_outputs")
KEY_FRAMES_DIR  = Path("uploads/key_frames")
TIME_SERIES_DIR = Path("research_plots/time_series")
RESULTS_DIR     = Path("results")
ROI_CSV         = Path("uploads/roi_summary.csv")
FINAL_CSV       = RESULTS_DIR / "final_dataset.csv"
INSIGHTS_JSON   = RESULTS_DIR / "video_deep_insights.json"
SUMMARY_TXT     = RESULTS_DIR / "research_summary.txt"
METADATA_CSV    = Path("uploads/metadata.csv")
VIDEO_BASE      = Path("uploads/videos")
MODELS_DIR      = Path("models")


# ---- Helpers -----------------------------------------------------------------

def load_metadata():
    """
    Load metadata.csv -> list of dicts: {id, platform, label, views}.
    Views read exclusively from metadata.csv. Strict file order preserved.
    """
    rows = []
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            views_raw = r.get("views", "").strip()
            views = None
            if views_raw and views_raw.lower() not in ("null", "none", ""):
                try:
                    views = float(views_raw)
                except ValueError:
                    pass
            rows.append({
                "id":       int(r["id"]),
                "platform": r["platform"].strip(),
                "label":    r["label"].strip(),
                "views":    views,
            })
    return rows


def video_path_for(video_id: int, label: str) -> Path:
    folder = "viral" if label == "viral" else "non_viral"
    return VIDEO_BASE / folder / f"{video_id}.mp4"


def clean_old_npz(log):
    """Remove stale .npz files from previous pipeline runs. DISABLED by user."""
    log.info("clean_old_npz DISABLED: Retaining existing .npz files to allow resuming.")
    return


def make_dirs():
    for d in [TRIBE_OUT_DIR, KEY_FRAMES_DIR, TIME_SERIES_DIR, RESULTS_DIR, Path("logs")]:
        d.mkdir(parents=True, exist_ok=True)


# ---- Model loading -----------------------------------------------------------

def load_tribe_model(log):
    """
    Load the real TRIBE v2 model from the local models/ directory.

    Expects weights downloaded by download_model.py (repo: pbhatt17/TRIBE-2).
    If the model is missing or fails to load, execution STOPS immediately
    with a RuntimeError. No simulation or fallback is used under any condition.
    """
    # Force CPU-only mode globally: some internal neuralset extractors call
    # torch.cuda.is_available() independently and crash when CUDA is not
    # compiled into torch (which is the case on this machine).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ckpt = MODELS_DIR / "best.ckpt"
    cfg  = MODELS_DIR / "config.yaml"

    if not MODELS_DIR.exists():
        raise RuntimeError(
            "models/ directory not found.\n"
            "Run: python download_model.py"
        )
    if not ckpt.exists():
        raise RuntimeError(
            f"Checkpoint not found: {ckpt}\n"
            "Run: python download_model.py"
        )
    if not cfg.exists():
        raise RuntimeError(
            f"Config not found: {cfg}\n"
            "Run: python download_model.py"
        )

    log.info("Loading TribeModel from %s ...", MODELS_DIR)
    # Patch PosixPath -> WindowsPath: checkpoint was saved on Linux;
    # torch.load fails to unpickle PosixPath objects on Windows without this.
    from ckpt_patch import apply_posixpath_patch
    apply_posixpath_patch()
    from tribev2.demo_utils import TribeModel
    try:
        model = TribeModel.from_pretrained(
            str(MODELS_DIR),          # checkpoint_dir: local path with best.ckpt + config.yaml
            checkpoint_name="best.ckpt",
            cache_folder="cache",     # feature extraction cache (separate from model weights)
            device="cpu",
            config_update={
                "data.text_feature.device": "cpu",
                "data.image_feature.image.device": "cpu",
                "data.audio_feature.device": "cpu",
                "data.video_feature.image.device": "cpu",
            }
        )
    except Exception as e:
        raise RuntimeError(
            f"TribeModel failed to load: {type(e).__name__}: {e}\n"
            "Ensure download_model.py completed successfully and models/best.ckpt exists."
        ) from e

    log.info("TribeModel loaded from %s", MODELS_DIR)
    return model


# ---- Step 1+2: Inference -----------------------------------------------------

def run_inference(model, meta_rows, log, dry_run_n=None):
    """
    Run real TRIBE v2 inference on each video (strict metadata.csv order).

    - Calls model.get_events_dataframe() + model.predict() per video
    - Maps (T, 20484) predictions to (T, 6) ROI time-series
    - Saves uploads/tribe_outputs/{id}.npz

    REQUIRES a live TribeModel. Raises RuntimeError if model is None.
    No simulation or fake outputs are generated under any condition.

    .npz keys:
      time_series    : (T, 6) float32 -- REGION_ORDER columns
      regions        : ['broca', 'amygdala', 'nacc', 'hippocampus', 'parietal', 'tpj']
      mean_activation: float
    """
    if model is None:
        raise RuntimeError(
            "run_inference called without a TribeModel.\n"
            "Load the model first via load_tribe_model()."
        )

    total = len(meta_rows) if dry_run_n is None else min(dry_run_n, len(meta_rows))
    log.info("=== STEP 1+2: Inference on %d videos (TR=%.1fs) ===", total, TR_SECONDS)

    pu.build_roi_indices()   # build and cache Destrieux atlas ROI indices once

    t_start = time.time()
    done    = 0
    errors  = []

    for i, row in enumerate(meta_rows[:total]):
        vid_id   = row["id"]
        label    = row["label"]
        vpath    = video_path_for(vid_id, label)
        out_path = TRIBE_OUT_DIR / f"{vid_id}.npz"

        log.info(
            "[%d/%d] ID=%d  label=%-10s  video=%s  %s",
            i + 1, total, vid_id, label, vpath.name,
            eta_string(time.time() - t_start, done, total),
        )

        if not vpath.exists():
            msg = f"ID={vid_id}: video not found at {vpath}"
            log.error("  SKIP -- %s", msg)
            errors.append({"id": vid_id, "error": msg})
            done += 1
            continue

        if out_path.exists():
            log.info("  SKIP -- ID=%d: %s already exists", vid_id, out_path.name)
            done += 1
            continue

        try:
            t0 = time.time()
            start_str = time.strftime('%H:%M:%S', time.localtime(t0))
            
            import cv2
            cap = cv2.VideoCapture(str(vpath))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            vid_duration = total_frames / fps if fps > 0 else 0.0

            log.info("  START: %s | ID=%d frames=%d duration=%.2fs", start_str, vid_id, total_frames, vid_duration)

            # Build event DataFrame directly — use audio_only=True to skip
            # whisperx transcription (unavailable on Python 3.14 / Windows:
            # ctranslate2==4.4.0 is not published for this platform).
            # The model still receives full audio + video features; only the
            # word/sentence text stream is omitted.
            import pandas as pd
            from tribev2.demo_utils import get_audio_and_text_events
            event = {
                "type":     "Video",
                "filepath": str(vpath),
                "start":    0,
                "timeline": "default",
                "subject":  "default",
            }
            events = get_audio_and_text_events(
                pd.DataFrame([event]),
                audio_only=True,   # skip whisperx / ctranslate2 dependency
            )

            preds, segments = model.predict(events, verbose=False)
            # preds: (T, 20484) -- real fMRI-like brain responses

            log.info(
                "  preds shape: %s  (T=%d TRs = %.1fs)",
                preds.shape, preds.shape[0], preds.shape[0] * TR_SECONDS,
            )

            roi_ts   = compute_roi_timeseries(preds)
            matrix   = np.stack([roi_ts[r] for r in REGION_ORDER], axis=1).astype(np.float32)
            mean_act = float(matrix.mean())

            np.savez_compressed(
                out_path,
                time_series=matrix,
                regions=np.array(REGION_ORDER),
                mean_activation=np.float32(mean_act),
            )

            t1 = time.time()
            end_str = time.strftime('%H:%M:%S', time.localtime(t1))
            took = t1 - t0
            
            log.info(
                "  END: %s | ID=%d  shape=(%d, %d)  mean_act=%.4f  took=%.1fs",
                end_str, vid_id, matrix.shape[0], matrix.shape[1], mean_act, took
            )
            log.info("  SUMMARY => ID=%d | shape=%s | duration=%.2fs | TRs=%d | processed_in=%.2fs", 
                     vid_id, matrix.shape, vid_duration, matrix.shape[0], took)
            log.info("  SUCCESS: saved %s", out_path.name)

        except Exception as e:
            msg = f"ID={vid_id}: {type(e).__name__}: {e}"
            log.error("  ERROR -- %s\n%s", msg, traceback.format_exc())
            errors.append({"id": vid_id, "error": msg})

        done += 1

    log.info("Inference complete. %d/%d succeeded, %d errors.",
             total - len(errors), total, len(errors))
    if errors:
        err_path = RESULTS_DIR / "inference_errors.json"
        err_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        log.warning("Errors saved to %s", err_path)
    return errors


# ---- Step 3: ROI Summary CSV -------------------------------------------------

def build_roi_summary(meta_rows, log):
    log.info("=== STEP 3: ROI Summary CSV ===")
    rows_out = []
    missing  = []

    for row in meta_rows:
        vid_id = row["id"]
        npz    = TRIBE_OUT_DIR / f"{vid_id}.npz"
        if not npz.exists():
            missing.append(vid_id)
            continue

        data         = np.load(npz, allow_pickle=True)
        matrix       = data["time_series"]          # (T, 6)
        per_roi_mean = matrix.mean(axis=0)           # (6,)

        out = {
            "id":              vid_id,
            "platform":        row["platform"],
            "label":           row["label"],
            "views":           row["views"] if row["views"] is not None else "",
            "mean_activation": round(float(matrix.mean()), 4),
        }
        for j, roi in enumerate(REGION_ORDER):
            out[roi] = round(float(per_roi_mean[j]), 4)
        rows_out.append(out)

    if missing:
        log.warning("Step 3: %d IDs missing .npz (skipped): %s", len(missing), missing[:10])

    fieldnames = ["id", "platform", "label", "views", "mean_activation"] + REGION_ORDER
    with open(ROI_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    log.info("Step 3 done -> %s  (%d rows)", ROI_CSV, len(rows_out))


# ---- Step 4: Temporal Analysis -----------------------------------------------

def run_temporal_analysis(meta_rows, log):
    log.info("=== STEP 4: Temporal Analysis ===")
    results = {}

    for row in meta_rows:
        vid_id = row["id"]
        npz    = TRIBE_OUT_DIR / f"{vid_id}.npz"
        if not npz.exists():
            continue

        data   = np.load(npz, allow_pickle=True)
        matrix = data["time_series"]    # (T, 6)
        T      = matrix.shape[0]
        roi_ts = {REGION_ORDER[j]: matrix[:, j] for j in range(6)}

        hook              = compute_hook_strength(roi_ts, n_trs=3)
        peak_region, peak_time, peak_val = compute_peak(roi_ts)
        retention         = compute_retention_slope(matrix.mean(axis=1))

        # Engagement curve plot
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor("#0d0d0d")
        ax.set_facecolor("#0d0d0d")
        colors = ["#00e5ff", "#ff3d71", "#ffaa00", "#00e096", "#6366f1", "#a855f7"]
        x_sec  = np.arange(T) * TR_SECONDS

        for j, roi in enumerate(REGION_ORDER):
            smoothed = smooth_timeseries(matrix[:, j], sigma=1.5)
            ax.plot(x_sec, smoothed, color=colors[j], linewidth=1.3,
                    alpha=0.85, label=REGION_META[roi]["sub"])

        ax.axvline(x=3 * TR_SECONDS, color="#ffffff", linestyle="--",
                   linewidth=0.7, alpha=0.4, label="Hook boundary")
        ax.set_xlabel("Time (seconds)", color="#888", fontsize=8)
        ax.set_ylabel("Activation",     color="#888", fontsize=8)
        ax.set_title(f"ID {vid_id}  |  {row['label']}  |  hook={hook:.3f}",
                     color="#ccc", fontsize=9)
        ax.legend(fontsize=6, loc="upper right", facecolor="#111",
                  edgecolor="#333", labelcolor="#aaa")
        ax.tick_params(colors="#444", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#1a1a1a")
        plt.tight_layout()
        plt.savefig(TIME_SERIES_DIR / f"{vid_id}.png", facecolor="#0d0d0d", dpi=130)
        plt.close()

        results[vid_id] = {
            "hook_strength":   round(hook, 4),
            "peak_region":     peak_region,
            "peak_time_s":     round(peak_time, 2),
            "peak_val":        round(peak_val, 4),
            "retention_slope": round(retention, 6),
        }

    log.info("Step 4 done -> %d engagement curve PNGs saved", len(results))
    return results


# ---- Step 5: Frame-Level Insights --------------------------------------------

def run_frame_insights(meta_rows, log):
    log.info("=== STEP 5: Frame-Level Insights ===")
    frame_results = {}

    for row in meta_rows:
        vid_id = row["id"]
        npz    = TRIBE_OUT_DIR / f"{vid_id}.npz"
        vpath  = video_path_for(vid_id, row["label"])
        if not npz.exists() or not vpath.exists():
            continue

        data   = np.load(npz, allow_pickle=True)
        matrix = data["time_series"]
        roi_ts = {REGION_ORDER[j]: matrix[:, j] for j in range(6)}

        top_trs    = top5_tr_indices(roi_ts)
        out_dir    = str(KEY_FRAMES_DIR / str(vid_id))
        frame_info = extract_top_frames(str(vpath), top_trs, out_dir, vid_id)

        peak_moments = []
        for fi in frame_info:
            tr_idx  = fi["tr"]
            vals    = {r: float(matrix[tr_idx, j]) for j, r in enumerate(REGION_ORDER)}
            dom_roi = max(vals, key=vals.get)
            peak_moments.append({
                "rank":   fi["rank"],
                "time":   fi["time_s"],
                "region": dom_roi,
                "value":  round(vals[dom_roi], 4),
            })

        json_path = KEY_FRAMES_DIR / str(vid_id) / "peak_moments.json"
        json_path.write_text(json.dumps(peak_moments, indent=2), encoding="utf-8")
        frame_results[vid_id] = peak_moments

    log.info("Step 5 done -> frames extracted for %d videos", len(frame_results))
    return frame_results


# ---- Step 6: Audio vs Visual Ablation ----------------------------------------

def run_ablation(model, meta_rows, log, n_sample=20):
    """
    Ablation on a sample (10 viral + 10 non-viral).
    Requires a live TribeModel. Raises RuntimeError if model is None.
    """
    if model is None:
        raise RuntimeError(
            "run_ablation called without a TribeModel.\n"
            "Load the model first via load_tribe_model()."
        )

    log.info("=== STEP 6: Audio vs Visual Ablation (sample=%d) ===", n_sample)
    viral_rows     = [r for r in meta_rows if r["label"] == "viral"]
    non_viral_rows = [r for r in meta_rows if r["label"] != "viral"]
    sample_rows    = viral_rows[:n_sample // 2] + non_viral_rows[:n_sample // 2]

    ablation_results = {}

    for i, row in enumerate(sample_rows):
        vid_id = row["id"]
        vpath  = video_path_for(vid_id, row["label"])
        if not vpath.exists():
            continue

        log.info("[Ablation %d/%d] ID=%d", i + 1, len(sample_rows), vid_id)

        def run_mode(mode):
            try:
                ev = model.get_events_dataframe(video_path=str(vpath))
                if mode == "video_only":
                    ev = ev[ev["type"] != "Audio"].copy()
                elif mode == "audio_only":
                    ev = ev[ev["type"] != "Video"].copy()
                p, _ = model.predict(ev, verbose=False)
                if p is not None and p.shape[0] > 0:
                    roi_ts = compute_roi_timeseries(p)
                    return float(np.mean([ts.mean() for ts in roi_ts.values()]))
            except Exception as e:
                log.error("  Ablation mode=%s ID=%d: %s", mode, vid_id, e)
            return None

        full_act  = run_mode("full")
        video_act = run_mode("video_only")
        audio_act = run_mode("audio_only")

        denom = (video_act or 0.0) + (audio_act or 0.0)
        if denom > 0 and video_act is not None and audio_act is not None:
            visual_pct = round(100 * video_act / denom, 1)
            audio_pct  = round(100 * audio_act / denom, 1)
        else:
            visual_pct = audio_pct = None

        ablation_results[vid_id] = {
            "full_activation":  round(full_act,  4) if full_act  is not None else None,
            "video_activation": round(video_act, 4) if video_act is not None else None,
            "audio_activation": round(audio_act, 4) if audio_act is not None else None,
            "visual_pct":       visual_pct,
            "audio_pct":        audio_pct,
        }
        log.info("  ID=%d  visual=%.1f%%  audio=%.1f%%",
                 vid_id, visual_pct or 0, audio_pct or 0)

    abl_path = RESULTS_DIR / "ablation_results.json"
    abl_path.write_text(json.dumps(ablation_results, indent=2), encoding="utf-8")
    log.info("Step 6 done -> %s", abl_path)
    return ablation_results


# ---- Step 7: Viral vs Non-Viral ----------------------------------------------

def run_viral_comparison(meta_rows, log):
    log.info("=== STEP 7: Viral vs Non-Viral Analysis ===")

    if not ROI_CSV.exists():
        log.error("roi_summary.csv not found -- run Step 3 first")
        return {}

    rows_by_id = {}
    with open(ROI_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows_by_id[int(r["id"])] = r

    viral_rows    = [rows_by_id[r["id"]] for r in meta_rows
                     if r["id"] in rows_by_id and r["label"] == "viral"]
    nonviral_rows = [rows_by_id[r["id"]] for r in meta_rows
                     if r["id"] in rows_by_id and r["label"] != "viral"]

    stats_out = {}
    for roi in REGION_ORDER + ["mean_activation"]:
        v_vals  = np.array([float(r[roi]) for r in viral_rows    if r.get(roi)])
        nv_vals = np.array([float(r[roi]) for r in nonviral_rows if r.get(roi)])
        mw      = run_mann_whitney(v_vals, nv_vals)
        stats_out[roi] = {
            "viral_mean":    round(float(v_vals.mean()),  4) if len(v_vals)  > 0 else None,
            "nonviral_mean": round(float(nv_vals.mean()), 4) if len(nv_vals) > 0 else None,
            "mann_whitney":  mw,
        }
        log.info("  %-16s  viral=%.4f  non_viral=%.4f  p=%.4f  d=%.3f",
                 roi,
                 stats_out[roi]["viral_mean"]    or 0,
                 stats_out[roi]["nonviral_mean"] or 0,
                 mw["p_value"]  or 1.0,
                 mw["cohens_d"] or 0)

    # KDE plots
    from scipy.stats import gaussian_kde
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), facecolor="#0d0d0d")
    axes       = axes.flatten()
    colors_v, colors_nv = "#00e5ff", "#ff3d71"

    for idx, roi in enumerate(REGION_ORDER):
        ax      = axes[idx]
        ax.set_facecolor("#111")
        v_vals  = np.array([float(r[roi]) for r in viral_rows    if r.get(roi)])
        nv_vals = np.array([float(r[roi]) for r in nonviral_rows if r.get(roi)])
        if len(v_vals) == 0 and len(nv_vals) == 0:
            continue
        all_vals = np.concatenate([v_vals, nv_vals]) \
                   if len(v_vals) > 0 and len(nv_vals) > 0 \
                   else (v_vals if len(v_vals) > 0 else nv_vals)
        xs = np.linspace(float(all_vals.min()) - 0.01, float(all_vals.max()) + 0.01, 200)

        for vals, color, lbl in [(v_vals, colors_v, "Viral"), (nv_vals, colors_nv, "Non-Viral")]:
            if len(vals) > 1:
                kde = gaussian_kde(vals)
                ax.plot(xs, kde(xs), color=color, linewidth=2, label=lbl)
                ax.fill_between(xs, kde(xs), alpha=0.12, color=color)

        ax.set_title(REGION_META[roi]["label"], color="#ccc", fontsize=9)
        ax.set_xlabel("Mean Activation", color="#666", fontsize=7)
        ax.tick_params(colors="#444", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#222")
        ax.legend(fontsize=7, facecolor="#111", edgecolor="#333", labelcolor="#aaa")

    plt.suptitle("Viral vs Non-Viral Brain Activation -- KDE", color="#eee", fontsize=11)
    plt.tight_layout()
    kde_path = RESULTS_DIR / "viral_vs_nonviral_kde.png"
    plt.savefig(kde_path, facecolor="#0d0d0d", dpi=130)
    plt.close()
    log.info("  KDE plot -> %s", kde_path)

    # Views correlation
    view_vals, act_vals = [], []
    for r in meta_rows:
        roi_r = rows_by_id.get(r["id"])
        if roi_r and r["views"] is not None:
            view_vals.append(float(r["views"]))
            act_vals.append(float(roi_r.get("mean_activation", 0)))

    if len(view_vals) > 2:
        from scipy.stats import pearsonr, spearmanr
        pr, pp = pearsonr(view_vals, act_vals)
        sr, sp = spearmanr(view_vals, act_vals)
        stats_out["views_correlation"] = {
            "n": len(view_vals),
            "pearson_r":  round(pr, 4), "pearson_p":  round(pp, 4),
            "spearman_r": round(sr, 4), "spearman_p": round(sp, 4),
        }
    else:
        stats_out["views_correlation"] = {"n": len(view_vals), "note": "insufficient data"}

    stats_path = RESULTS_DIR / "viral_stats.json"
    stats_path.write_text(json.dumps(stats_out, indent=2), encoding="utf-8")
    log.info("Step 7 done -> %s", stats_path)
    return stats_out


# ---- Step 8: Final Outputs ---------------------------------------------------

def build_final_outputs(meta_rows, temporal_results, frame_results,
                        ablation_results, viral_stats, log):
    log.info("=== STEP 8: Final Outputs ===")

    roi_by_id = {}
    if ROI_CSV.exists():
        with open(ROI_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                roi_by_id[int(r["id"])] = r

    # final_dataset.csv
    fieldnames = (
        ["id", "platform", "label", "views", "mean_activation"]
        + REGION_ORDER
        + ["hook_strength", "peak_region", "peak_time_s", "retention_slope"]
    )
    with open(FINAL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in meta_rows:
            vid_id = row["id"]
            out    = {
                "id":       vid_id,
                "platform": row["platform"],
                "label":    row["label"],
                "views":    row["views"] if row["views"] is not None else "",
            }
            if vid_id in roi_by_id:
                r = roi_by_id[vid_id]
                out["mean_activation"] = r.get("mean_activation", "")
                for roi in REGION_ORDER:
                    out[roi] = r.get(roi, "")
            if vid_id in temporal_results:
                t = temporal_results[vid_id]
                out["hook_strength"]   = t["hook_strength"]
                out["peak_region"]     = t["peak_region"]
                out["peak_time_s"]     = t["peak_time_s"]
                out["retention_slope"] = t["retention_slope"]
            w.writerow(out)
    log.info("  -> %s", FINAL_CSV)

    # video_deep_insights.json
    insights = {}
    for row in meta_rows:
        vid_id = row["id"]
        entry  = {"id": vid_id, "platform": row["platform"], "label": row["label"]}
        if vid_id in temporal_results:
            entry.update(temporal_results[vid_id])
        if vid_id in frame_results:
            entry["peak_moments"]     = frame_results[vid_id]
        if vid_id in ablation_results:
            abl = ablation_results[vid_id]
            entry["audio_pct"]    = abl.get("audio_pct")
            entry["visual_pct"]   = abl.get("visual_pct")
        if vid_id in roi_by_id:
            entry["engagement_curve"] = f"research_plots/time_series/{vid_id}.png"
        insights[str(vid_id)] = entry
    INSIGHTS_JSON.write_text(
        json.dumps(insights, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("  -> %s", INSIGHTS_JSON)

    # research_summary.txt
    lines = ["TRIBE v2 Brain Analysis -- Research Summary", "=" * 60, ""]

    if viral_stats and "mean_activation" in viral_stats:
        s   = viral_stats["mean_activation"]
        mw  = s.get("mann_whitney", {})
        p   = mw.get("p_value")
        d   = mw.get("cohens_d")
        sig = "YES (p < 0.05)" if (p is not None and p < 0.05) else "NO (p >= 0.05)"
        lines += [
            "Q1: Does brain activation predict virality?",
            f"  Viral mean activation    : {s.get('viral_mean', 'N/A')}",
            f"  Non-viral mean activation: {s.get('nonviral_mean', 'N/A')}",
            f"  Mann-Whitney p           : {p:.4f}" if p is not None else "  p-value: N/A",
            f"  Cohen's d                : {d}"     if d is not None else "",
            f"  Answer: {sig}",
            "",
        ]

    if temporal_results:
        viral_hooks  = [temporal_results[r["id"]]["hook_strength"]
                        for r in meta_rows if r["id"] in temporal_results and r["label"] == "viral"]
        nv_hooks     = [temporal_results[r["id"]]["hook_strength"]
                        for r in meta_rows if r["id"] in temporal_results and r["label"] != "viral"]
        v_h  = float(np.mean(viral_hooks)) if viral_hooks else None
        nv_h = float(np.mean(nv_hooks))    if nv_hooks    else None
        lines += [
            "Q2: Do viral videos have stronger hooks (first 6 seconds)?",
            f"  Viral hook mean    : {v_h:.4f}"  if v_h  is not None else "  Viral hook: N/A",
            f"  Non-viral hook mean: {nv_h:.4f}" if nv_h is not None else "  Non-viral hook: N/A",
            f"  Viral > Non-viral  : {'YES' if v_h and nv_h and v_h > nv_h else 'NO'}",
            "",
        ]

    if viral_stats:
        region_diffs = {roi: abs((viral_stats.get(roi, {}).get("viral_mean") or 0) -
                                 (viral_stats.get(roi, {}).get("nonviral_mean") or 0))
                        for roi in REGION_ORDER}
        top_roi = max(region_diffs, key=region_diffs.get)
        s       = viral_stats.get(top_roi, {})
        lines  += [
            "Q3: Which brain region differs most between viral vs non-viral?",
            f"  Region: {REGION_META[top_roi]['label']} ({top_roi})",
            f"  Viral mean    : {s.get('viral_mean')}",
            f"  Non-viral mean: {s.get('nonviral_mean')}",
            f"  Absolute diff : {region_diffs[top_roi]:.4f}",
            f"  Cohen's d     : {s.get('mann_whitney', {}).get('cohens_d')}",
            "",
        ]

    if ablation_results:
        vis_pcts = [v["visual_pct"] for v in ablation_results.values() if v.get("visual_pct") is not None]
        aud_pcts = [v["audio_pct"]  for v in ablation_results.values() if v.get("audio_pct")  is not None]
        lines += [
            "Q4: Audio vs Visual contribution?",
            f"  Sample size    : {len(ablation_results)} videos",
            f"  Visual mean %  : {np.mean(vis_pcts):.1f}%" if vis_pcts else "  Visual: N/A",
            f"  Audio  mean %  : {np.mean(aud_pcts):.1f}%" if aud_pcts else "  Audio: N/A",
            f"  Dominant       : {'Visual' if (vis_pcts and aud_pcts and np.mean(vis_pcts) > np.mean(aud_pcts)) else 'Audio'}",
            "",
        ]

    views_corr = viral_stats.get("views_correlation", {}) if viral_stats else {}
    if views_corr.get("pearson_r") is not None:
        lines += [
            "Views vs Activation Correlation:",
            f"  Pearson  r={views_corr['pearson_r']:.4f}  p={views_corr['pearson_p']:.4f}",
            f"  Spearman r={views_corr['spearman_r']:.4f}  p={views_corr['spearman_p']:.4f}",
            f"  N videos with views data: {views_corr['n']}",
            "",
        ]

    lines.append("ROI Mapping: nilearn Destrieux surface atlas on fsaverage5 (aparc.a2009s)")
    lines.append("TR duration: 2.0 seconds")
    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")
    log.info("  -> %s", SUMMARY_TXT)
    log.info("Step 8 done.")


# ---- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TRIBE v2 Brain Analysis Pipeline")
    parser.add_argument("--dry-run",     type=int, default=None, metavar="N",
                        help="Process only the first N videos")
    parser.add_argument("--step",        type=int, default=0,    metavar="STEP",
                        help="Start from this step number (0=all)")
    parser.add_argument("--no-ablation", action="store_true",
                        help="Skip Step 6 (audio vs visual ablation)")
    args = parser.parse_args()

    log = pu.setup_logger("logs/pipeline.log")
    log.info("=" * 60)
    log.info("TRIBE v2 Brain Analysis Pipeline  |  TR=%.1fs", TR_SECONDS)
    log.info("dry-run=%s  start-step=%d", args.dry_run, args.step)
    log.info("=" * 60)

    make_dirs()
    meta_rows = load_metadata()
    log.info("Loaded %d videos from %s", len(meta_rows), METADATA_CSV)

    temporal_results = {}
    frame_results    = {}
    ablation_results = {}
    viral_stats      = {}
    model            = None

    if args.step <= 1:
        clean_old_npz(log)
        model = load_tribe_model(log)   # hard stop if model unavailable
        run_inference(model, meta_rows, log, dry_run_n=args.dry_run)

    if args.step <= 3:
        build_roi_summary(meta_rows, log)

    if args.step <= 4:
        temporal_results = run_temporal_analysis(meta_rows, log)

    if args.step <= 5:
        frame_results = run_frame_insights(meta_rows, log)

    if args.step <= 6 and not args.no_ablation:
        if model is None:
            model = load_tribe_model(log)
        ablation_results = run_ablation(model, meta_rows, log, n_sample=20)

    if args.step <= 7:
        viral_stats = run_viral_comparison(meta_rows, log)

    if args.step <= 8:
        build_final_outputs(
            meta_rows, temporal_results, frame_results, ablation_results, viral_stats, log
        )

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
