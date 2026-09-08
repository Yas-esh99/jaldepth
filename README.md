# 💧 JalDepth — how deep is the water on this street?

A small, self-contained computer-vision demo: give it a photo, a short video, or your **live camera** pointed at a flooded street and it tells you **where the water is**, **who and what is standing in it**, and **how deep it is** as a class a driver understands — **Dry · Ankle · Knee · Wheel**.

It is the vision module of **JalDrishti**, our Smart India Hackathon 2026 project (PS SIH26085, *Urban Flood Nowcasting System — Drainage and Rainfall Coupling*), packaged on its own so it can be shown and tried in one click.

**Run it:** locally in one command (below), or deploy to Streamlit Community Cloud in three clicks (see Deploy). Vercel cannot host PyTorch/OpenCV inference.

## What it does

| Step | How | Trained by us? |
|---|---|---|
| **Find the water** | YOLOv8n-seg **fine-tuned on ATLANTIS** (Erfani et al. 2022 — 5,195 Creative-Commons photos of waterbodies, 56 labels; we merged its 17 water labels into one *water* class). 3,364 train / 535 val / 1,296 test images, 20 epochs, 512 px. **Test mask mAP50 0.58 · mAP50-95 0.38.** | **Yes** |
| **Find people & vehicles** | Pretrained YOLOv8n (COCO). A box whose base sits inside the water mask is *standing in water*. | No (pretrained) |
| **Depth, coarse** | Fraction of a vehicle hidden below the waterline × its typical height (car 150 cm, bus 300 cm…). Indicative. | Geometry |
| **Depth, exact** | Mark a reference of known height (kerb ≈ 15 cm, pole bands, wall) with the sliders; the waterline on it is converted to centimetres. | Geometry |
| **Flags** | *Person standing in water* · *Vehicle in wheel-deep water* | Rules |

**Indian test set.** 38 freely licensed photos of flooded streets in Mumbai, Chennai, Vadodara, Ahmedabad and Bengaluru from Wikimedia Commons (`samples/`, credits in `samples/ATTRIBUTION.md`) plus a 2017 Mumbai flood video. The model finds water in 34 of the 38 photos (mean coverage 35%) and flags people and vehicles in water correctly on the Mumbai and Chennai scenes. There is no public street-level *Indian* flood segmentation dataset yet — the open Indian sets are satellite imagery — so we train on the global ATLANTIS set and validate on Indian streets; collecting and labelling an Indian street set is the next step.

## Live camera

Two modes in the sidebar: **Live camera — snapshot** uses the browser camera (laptop or phone) and analyses one picture at a time; **Live camera — continuous** streams the webcam through WebRTC and draws the water mask, boxes and depth class on every frame (the ruler sliders apply live). Continuous mode is smooth on a laptop; on a hosted server the video is relayed through the browser and may lag.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

First run downloads nothing: both models are in `models/` (≈13 MB). CPU is enough (~0.1 s per image on a laptop).

## Deploy (free)

**Streamlit Community Cloud** (recommended — runs Python + OpenCV + PyTorch):
1. Push this folder to GitHub (public repo).
2. Go to https://share.streamlit.io → *New app* → pick the repo, branch `main`, file `app.py` → Deploy.
3. Done; the URL is `https://<app-name>.streamlit.app`. `requirements.txt` is already set up (no apt packages needed: OpenCV headless wheel).

Vercel is not suitable for this app: its serverless functions cannot run PyTorch/OpenCV inference within their size and time limits.

## Files

`app.py` Streamlit UI · `jaldepth.py` model + geometry (pure functions, reusable) · `models/` weights + metrics · `samples/` Indian test photos and video with attribution · `tools/fetch_indian_samples.py` sample downloader.

## Credits

ATLANTIS: Erfani, Wu, Wu, Wang, Goharian — *ATLANTIS: A benchmark for semantic segmentation of waterbody images*, Environmental Modelling & Software, 2022. Ultralytics YOLOv8. Sample media: Wikimedia Commons contributors (CC BY / CC BY-SA / CC0 / GODL-India), see `samples/ATTRIBUTION.md`. MIT licence for the code.
