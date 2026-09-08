import os, json, io, glob, tempfile
import numpy as np, cv2, streamlit as st
from PIL import Image
import jaldepth as jd

st.set_page_config(page_title="JalDepth — flood water depth from a camera", page_icon="💧", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__)); SAMPLES = os.path.join(HERE, "samples")
metrics = json.load(open(os.path.join(HERE, "models", "water_seg_metrics.json")))
attrib = {r["file"]: r for r in json.load(open(os.path.join(SAMPLES, "attribution.json")))} if os.path.exists(os.path.join(SAMPLES, "attribution.json")) else {}

st.markdown("""<style>
.big{font-size:44px;font-weight:800;line-height:1;margin:0}.lbl{color:#5b6b7c;font-size:12px;text-transform:uppercase;letter-spacing:1px}
.card{background:#fff;border:1px solid #dfe6ee;border-radius:12px;padding:14px 16px}.flag{background:#fdecea;border:1px solid #f5c2c0;border-radius:8px;padding:6px 10px;margin:4px 0;font-weight:600}
</style>""", unsafe_allow_html=True)
st.title("💧 JalDepth — how deep is the water on this street?")
st.caption("Computer-vision module of **JalDrishti** (SIH 2026, PS SIH26085). A trained water-segmentation model finds the water, YOLOv8 finds people and vehicles, and known-height references turn pixels into a depth class a driver understands: **Dry · Ankle · Knee · Wheel**.")

with st.sidebar:
    st.header("Input")
    src = st.radio("Image source", ["Indian sample photos (Wikimedia Commons)", "Upload a photo", "Upload a short video"], index=0)
    conf_seg = st.slider("Water model confidence", 0.05, 0.8, 0.25, 0.05); conf_det = st.slider("Object detector confidence", 0.1, 0.8, 0.3, 0.05)
    st.divider(); st.header("Ruler (optional)")
    st.caption("If a kerb, pole or wall of known height is visible, mark it to read an exact depth.")
    use_ruler = st.checkbox("Use a reference ruler", value=False)
    rx = st.slider("Ruler x (px)", 0, 1280, 640); ryb = st.slider("Ruler base y (px)", 0, 720, 600); ryt = st.slider("Ruler top y (px)", 0, 720, 400); rh = st.number_input("Ruler height (cm)", 5.0, 500.0, 100.0, 5.0)
    st.divider(); st.markdown(f"**Model card** · YOLOv8n-seg fine-tuned on ATLANTIS ({metrics['train_images']} train imgs) · test mask mAP50 **{metrics['test_mask_mAP50']:.2f}** · YOLOv8n COCO for objects")

def to_bgr(pil): return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
def fit(img, w=1280):
    if img.shape[1] > w: img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])))
    return img

img = None; caption = ""
if src.startswith("Indian"):
    files = sorted([f for f in os.listdir(SAMPLES) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not files: st.warning("No samples found — run tools/fetch_indian_samples.py")
    else:
        pick = st.selectbox("Choose a sample", files, format_func=lambda f: attrib.get(f, {}).get("title", f).replace("File:", ""))
        img = fit(cv2.imread(os.path.join(SAMPLES, pick))); a = attrib.get(pick, {}); caption = f"{a.get('title','').replace('File:','')} — {a.get('author','')} · {a.get('license','')} · Wikimedia Commons"
elif src == "Upload a photo":
    up = st.file_uploader("Photo of a flooded street", type=["jpg", "jpeg", "png"])
    if up: img = fit(to_bgr(Image.open(up))); caption = up.name
else:
    up = st.file_uploader("Short video (mp4/mov, ≤200 MB)", type=["mp4", "mov", "m4v"])
    if up:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmp.write(up.read()); tmp.close()
        cap = cv2.VideoCapture(tmp.name); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
        t = st.slider("Frame (seconds)", 0.0, max(0.1, n / fps), 0.0, 0.5); cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, fr = cap.read()
        if ok: img = fit(fr); caption = f"{up.name} @ {t:.1f}s"
        if st.button("Analyse whole video (1 frame / s)"):
            rows = []; prog = st.progress(0.0)
            for k in range(0, n, int(fps)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, fr = cap.read()
                if not ok: break
                res, m = jd.analyze_image(fit(fr), None, conf_seg, conf_det)
                rows.append({"t (s)": round(k / fps, 1), "water %": round(res["water_fraction"] * 100), "people": sum(d["name"] == "person" for d in res["detections"]), "vehicles": sum(d["name"] in jd.VEHICLES for d in res["detections"]), "vehicle depth (cm)": res.get("vehicle_depth_cm"), "flags": ", ".join(res["flags"])})
                prog.progress(min(1.0, (k + fps) / n))
            st.dataframe(rows, width='stretch')

if img is not None:
    ruler = {"x": int(rx * img.shape[1] / 1280), "y_bottom": int(ryb * img.shape[0] / 720), "y_top": int(ryt * img.shape[0] / 720), "height_cm": float(rh)} if use_ruler else None
    with st.spinner("Running water segmentation + detection…"):
        res, mask = jd.analyze_image(img, ruler, conf_seg, conf_det); vis = jd.draw(img, mask, res, ruler)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=caption, width='stretch')
    with c2:
        wf = res["water_fraction"] * 100
        st.markdown(f'<div class="card"><div class="lbl">Water coverage (segmentation model)</div><p class="big">{wf:.0f}%</p></div>', unsafe_allow_html=True); st.write("")
        if res.get("vehicle_depth_cm") is not None:
            name, col = jd.depth_class(res["vehicle_depth_cm"]); hexc = "#%02x%02x%02x" % (col[2], col[1], col[0])
            st.markdown(f'<div class="card"><div class="lbl">Estimated depth from submerged vehicles (coarse)</div><p class="big" style="color:{hexc}">{res["vehicle_depth_cm"]:.0f} cm · {name}</p><div class="lbl" style="text-transform:none">uses typical vehicle heights (car 150 cm) — indicative only</div></div>', unsafe_allow_html=True); st.write("")
        if "ruler" in res:
            r = res["ruler"]; name, col = jd.depth_class(r["depth_cm"]); hexc = "#%02x%02x%02x" % (col[2], col[1], col[0])
            st.markdown(f'<div class="card"><div class="lbl">Depth at the ruler</div><p class="big" style="color:{hexc}">{r["depth_cm"]:.0f} cm · {name}</p><div class="lbl" style="text-transform:none">waterline solidity {r["solidity"]}</div></div>', unsafe_allow_html=True); st.write("")
        n_p = sum(d["name"] == "person" for d in res["detections"]); n_v = sum(d["name"] in jd.VEHICLES for d in res["detections"]); n_w = sum(d["in_water"] for d in res["detections"])
        st.markdown(f'<div class="card"><div class="lbl">Objects</div><b>{n_p}</b> people · <b>{n_v}</b> vehicles · <b>{n_w}</b> standing in water</div>', unsafe_allow_html=True); st.write("")
        for f in res["flags"]: st.markdown(f'<div class="flag">⚠ {f}</div>', unsafe_allow_html=True)
        if not res["flags"]: st.caption("No emergency flags.")
    with st.expander("How it works"):
        st.markdown(f"""
1. **Water segmentation** — a YOLOv8n-seg network we fine-tuned on **ATLANTIS** (Erfani et al. 2022; 5,195 Creative-Commons photos, 17 waterbody labels merged into *water*). Test mask mAP50 = {metrics['test_mask_mAP50']:.2f}, mAP50-95 = {metrics['test_mask_mAP50_95']:.2f}. Blue overlay = water.
2. **Object detection** — pretrained YOLOv8n (COCO) finds people, cars, buses, trucks, motorcycles. A box whose base lies inside the water mask is *standing in water*.
3. **Depth** — two ways. *Coarse:* the fraction of a vehicle hidden below the waterline × its typical height. *Exact:* mark a reference of known height (kerb ≈ 15 cm, pole bands, wall) and the waterline on it is converted to centimetres.
4. **Classes** — Dry < 5 cm · Ankle 5–20 · Knee 20–50 · Wheel > 50 cm. Classes, not false precision: that is what a driver or a control room needs.

Limits: night, glare and muddy reflections lower the mask quality; depth without a ruler is indicative. In JalDrishti this module runs on existing CCTV and feeds a self-correcting flood nowcast.
""")
else:
    st.info("Pick a sample on the left, or upload a photo / video.")
st.caption("Sample photos: Wikimedia Commons, CC BY-SA / public domain — see samples/ATTRIBUTION.md. Dataset: ATLANTIS (Erfani et al., Environmental Modelling & Software, 2022).")
