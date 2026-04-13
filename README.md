# CEREBRUM

> Neural Engagement Analysis Platform — Predict how the brain responds to video content.

![Python](https://img.shields.io/badge/Python-3.12-00e5ff?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000?style=flat-square&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📖 About

**Cerebrum** is an advanced web-based platform that analyzes human cognitive and emotional responses to video content. By leveraging Meta's state-of-the-art **TRIBE v2** multimodal brain encoding foundation model, Cerebrum predicts localized fMRI-like brain activity directly from standard `.mp4` video files. 

This platform bridges the gap between complex neuroscience research and practical content analysis. Content creators, marketers, and researchers can upload a video and instantly see how it will engage specific functional regions of the human brain, predicting metrics like memorability, emotional resonance, and viral potential.

## 🧠 Brain Regions Analyzed

Cerebrum extracts spatial temporal features and maps them to 6 specific functional brain regions:

| Region | Cognitive Function | Insight / Interpretation |
|--------|---------------------|--------------------------|
| **Broca Area** | Language Processing | Predicts verbal complexity, speech engagement, and linguistic depth. |
| **Amygdala** | Emotion & Arousal | Predicts emotional resonance, intensity, and visceral reactions. |
| **Nucleus Accumbens** | Reward & Motivation | Predicts dopamine response, shareability, and viral engagement potential. |
| **Hippocampus** | Memory Formation | Predicts how memorable the content is over the long term. |
| **Superior Parietal** | Visual Attention | Predicts spatial awareness and visual attention capture. |
| **Temporo-Parietal Junction** | Social Cognition | Predicts relatability, theory of mind, and understanding social dynamics. |

## 🚀 Features

- **Multi-Region Activation Scores:** Generates a 0–1 engagement score and percentile ranking for each of the 6 core regions.
- **Timeline Heatmaps:** Visualizes which specific brain regions activate at exactly which second during the video's playback.
- **Radar Profiles:** A 6-point spider chart that gives an instant visual fingerprint of the video's cognitive impact.
- **Category Benchmarking:** Compare a video's neural response against baselines for Music, Comedy, Education, Vlog, and Fitness content.
- **Side-by-Side Comparison:** Upload two different cuts of a video and compare their neural deltas.
- **Automated Reporting:** Export your complete analysis history as a structured CSV dataset, or generate polished PDF reports for individual videos.
- **Secure Architecture:** JWT-based session management backed by Google OAuth 2.0 authentication.

## 📦 Packages & Architecture

Cerebrum is built as a monolithic web service integrating a deep learning inference pipeline with a modern web stack.

### Core Stack
* **Web Framework:** [Flask 3.1](https://flask.palletsprojects.com/)
* **Database / ORM:** SQLite (Dev) / PostgreSQL (Prod) via `Flask-SQLAlchemy`
* **Authentication:** `Authlib` (Google OAuth 2.0) + `PyJWT`
* **Machine Learning:** `PyTorch 2.11`, `Transformers`, `einops`
* **Brain Encoding Model:** [TRIBE v2](https://github.com/facebookresearch/tribev2) (Meta Research)
* **Visualizations:** `Matplotlib` (Heatmaps, Radars, Waveforms)
* **Reporting:** `fpdf2` (PDF generation), `pandas` (CSV extraction)

### UI Design System
The frontend uses a custom *Monospace Minimal* design system built entirely from scratch with vanilla HTML/CSS. It utilizes `IBM Plex Mono`, pure blacks, and neon cyan accents to create a technical, "data-first" aesthetic without relying on massive framework bundles like React or Tailwind.

## ⚙️ Quick Start

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/fayaspsulfikkar/cerebrum-app.git
   cd cerebrum-app
   ```

2. **Install core dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model Weights (*Required*)**
   The TRIBE v2 foundation model requires ~676 MB of weights. Run the downloader script:
   ```bash
   python download_model.py
   ```

4. **Configure Environment**
   Copy the example environment file and fill in your details:
   ```bash
   cp .env.example .env
   ```
   *You will need to create OAuth credentials in the Google Cloud Console and set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`.*

5. **Start the server**
   ```bash
   python app.py
   ```
   Navigate to `http://localhost:5000` in your browser.

## 🌐 Production Deployment (Render.com)

This application is configured for seamless deployment to Render's Web Services.

1. Push this repository to your GitHub account.
2. In Render, create a new **Web Service** hooked to this repository.
3. Configure the build:
   - **Build Command:** `pip install -r requirements.txt && python download_model.py`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
4. Attach a Render **PostgreSQL** database and pass the `DATABASE_URL` to your Web Service environment variables.
5. In your Web Service settings, add your Google OAuth keys and set `FLASK_ENV=production`.
6. Once deployed, open the Render console/shell and run:
   ```python
   from app import db, create_app
   app = create_app()
   with app.app_context(): db.create_all()
   ```

## 🏷️ Release History

* **v1.0.0 (Current)** — Initial Release. Complete web platform featuring Google OAuth, SQLAlchemy PostgreSQL integration, timeline heatmaps, compare feature, and professional export tools.
* **v0.1.0-beta** — Initial MVP. Proof-of-concept inference script linking the CLI test utility to a basic Flask web interface.

## 🎓 Credits & Attribution

The core inference engine of Cerebrum is built on top of **Meta's TRIBE v2** multimodal brain encoding model. We extend our deep gratitude to the researchers for open-sourcing this incredible foundation model.

> **Paper:** Wen, H., Shi, J., Zhang, Y., Lu, K., & Liu, Z. (2023). *TRIBE v2: Multimodal Brain Encoding Model.*  
> **Original Codebase:** [facebookresearch/tribev2](https://github.com/facebookresearch/tribev2)

## 📄 License

The web application wrapper (UI, standard APIs, database integration) is provided under the **MIT License**.

***Note:** The underlying Meta TRIBE v2 model weights and core `tribev2/` package code are subject to Meta's Open Research License. Please refer to the original repository for usage restrictions related to the ML models.*
