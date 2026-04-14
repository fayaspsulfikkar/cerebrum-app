# CEREBRUM: Brain Engagement Analysis Platform

Neural engagement analysis using Meta's TRIBE v2 model to measure how videos activate different brain regions.

## Research Finding

**Brain Region Activation Does Not Predict Social Media Virality**

We analyzed 20 videos (10 viral, 10 non-viral) using TRIBE v2 and found NO correlation between brain activation and social media virality. Virality is driven by algorithmic factors, not neural engagement alone.

### Results Summary
- Viral videos: Average brain activation 0.38-0.40
- Non-viral videos: Average brain activation 0.37-0.40
- **Difference: Statistically insignificant**

### Implications
TRIBE v2 is valuable for understanding content quality and neural engagement, but cannot predict virality.

## Features

- Google OAuth login
- Video upload and analysis
- 6 brain region activation analysis
- Timeline heatmaps
- Radar charts
- Side-by-side comparison
- CSV & PDF export

## Usage

Local deployment only (requires TRIBE v2 download):

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Dataset

20 videos analyzed:
- 10 viral YouTube Shorts (100K+ views)
- 10 non-viral videos (<10K views)

Results in `reels_data.csv`

## Paper

Read the full research: [ArXiv link]

## Credits

Built on Meta's TRIBE v2:
- Paper: https://arxiv.org/abs/2403.xxxxx
- Code: https://github.com/facebookresearch/tribev2
