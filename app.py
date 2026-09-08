import os, json, io, glob, tempfile
import numpy as np, cv2, streamlit as st
from PIL import Image
import jaldepth as jd
import tankrig as tr

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
    src = st.radio("Image source", ["Indian sample photos (Wikimedia Commons)", "Upload a photo", "Upload a short video", "Live camera — snapshot", "Live camera — continuous"], index=0)
    scene_lbl = st.radio("Scene", ["Real street / CCTV (India)", "Demo tray — kerb + toy car (our rig)", "Glass tank — side view (model car)", "Table-top tray (top-down, toy car)"], index=0,
                         help="Demo tray: the striped kerb stands for 15 cm and the toy tyre for 60 cm — no ruler. Glass tank: the model car is the ruler. Tray top-down: sizes scaled to the toy car.")
    scene = "street" if scene_lbl.startswith("Real") else "box" if scene_lbl.startswith("Demo") else "tank" if scene_lbl.startswith("Glass") else "rig"
    if scene == "box":
        st.markdown("**Calibrate once** (camera fixed): drag the two green lines onto the road/floor and the top of the kerb. Tint the water blue for a crisp waterline.")
        b_floor = st.slider("Road / floor line — % from top", 30, 100, 72); b_kerb = st.slider("Kerb top line — % from top", 10, 95, 62)
        b_x0, b_x1 = st.slider("Measure across x-range (%)", 0, 100, (10, 90), help="Front wall / kerb face where the waterline is visible; avoid the pole and clutter.")
        b_kerb_cm = st.number_input("Kerb represents (cm)", 5.0, 50.0, 15.0, 1.0); b_tyre_cm = st.number_input("Tyre represents (cm)", 20.0, 120.0, 60.0, 5.0)
        b_mode = st.radio("Water detection", ["Blue-tinted water", "Compare with empty-box reference"], index=0); b_sens = st.slider("Sensitivity (fraction of row that must be water)", 0.2, 0.8, 0.45, 0.05)
        if b_mode.startswith("Compare"):
            if st.button("📷 Capture empty-box reference now"): tr._ref["capture"] = True; st.success("Next frame/image will be stored as the empty reference.")
            st.caption(f"Reference stored: {'yes' if tr._ref.get('frame') is not None else 'no'}")
    toy_len = st.number_input("Toy car length (cm)", 2.0, 30.0, 7.0, 0.5) if scene == "rig" else 7.0
    if scene == "tank":
        st.markdown("**The model car is the ruler** — no printed scale needed. Its length sets the pixel scale, its tyres set the floor.")
        t_car_len = st.number_input("Model car length (cm)", 3.0, 60.0, 24.0, 0.5, help="Measure the die-cast car nose to tail. A 1:18 Golf is ~24 cm.")
        t_real_len = st.number_input("Real car length (cm)", 200.0, 1500.0, 430.0, 10.0, help="The real car the model represents (Golf ≈ 430 cm). Scale = real / model.")
        with st.expander("Manual fallback (only if the car is never detected)"):
            t_manual = st.checkbox("Use manual lines", value=False); t_floor = st.slider("Floor y (%)", 30, 100, 86); t_mark = st.slider("Mark y (%)", 5, 95, 58); t_mark_cm = st.number_input("Mark height (cm)", 1.0, 100.0, 10.0, 0.5)
        t_x0, t_x1 = st.slider("Tank interior x-range (%)", 0, 100, (4, 96)); t_top = st.slider("Tank top y (%)", 0, 60, 4)
    conf_seg = st.slider("Water model confidence", 0.05, 0.8, 0.25, 0.05); conf_det = st.slider("Object detector confidence", 0.1, 0.8, 0.3, 0.05)
    st.divider(); st.header("Ruler (optional)")
    st.caption("If a kerb, pole or wall of known height is visible, mark it to read an exact depth. In rig mode enter the REAL height it stands for (e.g. a 3 cm block = 15 cm kerb).")
    use_ruler = st.checkbox("Use a reference ruler", value=False)
    rx = st.slider("Ruler x (px)", 0, 1280, 640); ryb = st.slider("Ruler base y (px)", 0, 720, 600); ryt = st.slider("Ruler top y (px)", 0, 720, 400); rh = st.number_input("Ruler height (cm)", 5.0, 500.0, 100.0, 5.0)
    st.divider(); st.markdown("**Depth is always estimated**, in this order of trust: ruler → known-size objects standing in water (people, cars, bikes, hydrants…) → scene estimate from how much of the frame is water. The card shows which method and how confident.")
    st.markdown(f"**Model card** · YOLOv8n-seg fine-tuned on ATLANTIS ({metrics['train_images']} train imgs) · test mask mAP50 **{metrics['test_mask_mAP50']:.2f}** · YOLOv8n COCO for objects")

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
elif src == "Live camera — snapshot":
    st.caption("Uses your browser camera (works on phones too). Point it at water with a kerb or a person/vehicle in view and take a picture.")
    shot = st.camera_input("Take a picture")
    if shot: img = fit(to_bgr(Image.open(shot))); caption = "live camera snapshot"
elif src == "Live camera — continuous":
    st.caption("Continuous analysis of your webcam: the water mask, boxes and depth class are drawn on every frame. Best on a laptop; on hosted servers the video is relayed through the browser and may be slower.")
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        import av as _av
        live_ruler = {"x": int(rx), "y_bottom": int(ryb), "y_top": int(ryt), "height_cm": float(rh)} if use_ruler else None
        tank_cfg = (t_x0, t_x1, t_top, t_car_len, t_real_len, ({"y_floor": t_floor / 100, "y_mark": t_mark / 100, "mark_cm": t_mark_cm} if t_manual else None)) if scene == "tank" else None
        box_cfg = (b_floor, b_kerb, b_x0, b_x1, b_kerb_cm, b_tyre_cm, "ref" if b_mode.startswith("Compare") else "tint", b_sens) if scene == "box" else None
        def _cb(frame):
            im = frame.to_ndarray(format="bgr24"); im = fit(im)
            if box_cfg:
                fl, kb, x0p, x1p, kcm, tcm, md, sens = box_cfg; H_, W_ = im.shape[:2]; ref = tr.maybe_capture_ref(im)
                bres = tr.analyze_box(im, (x0p / 100 * W_, x1p / 100 * W_), fl / 100 * H_, kb / 100 * H_, kcm, md, ref, tcm, min_frac=sens); vis = tr.draw_box(im, bres, kcm)
            elif tank_cfg:
                x0p, x1p, topp, clen, rlen, man = tank_cfg; H_, W_ = im.shape[:2]; roi = (int(x0p / 100 * W_), int(topp / 100 * H_), int(x1p / 100 * W_), int(0.95 * H_))
                manual = {"y_floor": man["y_floor"] * H_, "y_mark": man["y_mark"] * H_, "mark_cm": man["mark_cm"]} if man else None
                tres, m = tr.analyze_tank(im, roi, clen, rlen, manual, conf_seg, conf_det); vis = tr.draw_tank(im, tres)
            else:
                res, m = jd.analyze_image(im, live_ruler, conf_seg, conf_det, scene, toy_len); vis = jd.draw(im, m, res, live_ruler)
            return _av.VideoFrame.from_ndarray(vis, format="bgr24")
        webrtc_streamer(key="jaldepth-live", mode=WebRtcMode.SENDRECV, video_frame_callback=_cb, media_stream_constraints={"video": {"width": {"ideal": 1280}}, "audio": False},
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, async_processing=True)
        st.caption("Tip: the ruler sliders in the sidebar apply live — line the white box up with a kerb or pole and set its height.")
    except Exception as e:
        st.error(f"Continuous mode needs the optional packages streamlit-webrtc and av ({e}). Use the snapshot mode instead.")
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
                if scene == "box":
                    fr2 = fit(fr); H_, W_ = fr2.shape[:2]; bres = tr.analyze_box(fr2, (b_x0 / 100 * W_, b_x1 / 100 * W_), b_floor / 100 * H_, b_kerb / 100 * H_, b_kerb_cm, "ref" if b_mode.startswith("Compare") else "tint", tr._ref.get("frame"), b_tyre_cm, min_frac=b_sens, smooth=False)
                    rows.append({"t (s)": round(k / fps, 1), "level": bres["class"], "real depth (cm)": bres["real_cm"], "kerb submerged": bres["kerb_submerged"], "tyre submerged": bres["tyre_submerged"], "action": bres["action"]}); prog.progress(min(1.0, (k + fps) / n)); continue
                if scene == "tank":
                    fr2 = fit(fr); H_, W_ = fr2.shape[:2]; roi = (int(t_x0 / 100 * W_), int(t_top / 100 * H_), int(t_x1 / 100 * W_), int(0.95 * H_))
                    manual = {"y_floor": t_floor / 100 * H_, "y_mark": t_mark / 100 * H_, "mark_cm": t_mark_cm} if t_manual else None
                    tres, m = tr.analyze_tank(fr2, roi, t_car_len, t_real_len, manual, conf_seg, conf_det)
                    rows.append({"t (s)": round(k / fps, 1), "tank depth (cm)": tres["depth_cm"], "real (cm)": tres["real_cm"], "class": tres["class"], "car submerged": (tres["car"] or {}).get("submerged_fraction")}); prog.progress(min(1.0, (k + fps) / n)); continue
                res, m = jd.analyze_image(fit(fr), None, conf_seg, conf_det, scene, toy_len)
                rows.append({"t (s)": round(k / fps, 1), "water %": round(res["water_fraction"] * 100), "people": sum(d["name"] == "person" for d in res["detections"]), "vehicles": sum(d["name"] in jd.VEHICLES for d in res["detections"]), "depth (cm)": res["depth"]["real_cm"], "class": res["depth"]["class"], "confidence": res["depth"]["confidence"], "flags": ", ".join(res["flags"])})
                prog.progress(min(1.0, (k + fps) / n))
            st.dataframe(rows, width='stretch')

if img is not None:
    ruler = {"x": int(rx * img.shape[1] / 1280), "y_bottom": int(ryb * img.shape[0] / 720), "y_top": int(ryt * img.shape[0] / 720), "height_cm": float(rh)} if use_ruler else None
    with st.spinner("Running water segmentation + detection…"):
        if scene == "box":
            H_, W_ = img.shape[:2]; ref = tr.maybe_capture_ref(img); mode = "ref" if b_mode.startswith("Compare") else "tint"
            bres = tr.analyze_box(img, (b_x0 / 100 * W_, b_x1 / 100 * W_), b_floor / 100 * H_, b_kerb / 100 * H_, b_kerb_cm, mode, ref, b_tyre_cm, min_frac=b_sens, smooth=False); vis = tr.draw_box(img, bres, b_kerb_cm); mask = None
            res = {"water_fraction": 0.0, "detections": [], "flags": ([bres["action"]] if bres["class"] in ("Knee-deep", "Wheel-deep") else []), "depth": {"cm": bres["real_cm"], "real_cm": bres["real_cm"], "class": bres["class"], "confidence": bres["confidence"], "method": bres["method"]}, "box": bres}
        elif scene == "tank":
            H_, W_ = img.shape[:2]; roi = (int(t_x0 / 100 * W_), int(t_top / 100 * H_), int(t_x1 / 100 * W_), int(0.95 * H_))
            manual = {"y_floor": t_floor / 100 * H_, "y_mark": t_mark / 100 * H_, "mark_cm": t_mark_cm} if t_manual else None
            tres, mask = tr.analyze_tank(img, roi, t_car_len, t_real_len, manual, conf_seg, conf_det); vis = tr.draw_tank(img, tres)
            res = {"water_fraction": float((mask > 0).mean()) if mask is not None else 0.0, "detections": [], "flags": [], "depth": {"cm": tres["depth_cm"], "real_cm": tres["real_cm"], "class": tres["class"], "confidence": tres["confidence"], "method": tres["method"]}, "tank": tres}
        else:
            res, mask = jd.analyze_image(img, ruler, conf_seg, conf_det, scene, toy_len); vis = jd.draw(img, mask, res, ruler)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=caption, width='stretch')
    with c2:
        dp = res["depth"]; name, col = jd.depth_class(dp["real_cm"]); hexc = "#%02x%02x%02x" % (col[2], col[1], col[0])
        if "box" in res:
            b = res["box"]; nm, dsc, act, c_ = tr.box_class(b["real_cm"]); hexb = "#%02x%02x%02x" % (c_[2], c_[1], c_[0])
            st.markdown(f'<div class="card"><div class="lbl">Water level</div><p class="big" style="color:{hexb}">{nm.upper()}</p><div style="font-size:15px;margin-top:4px">{dsc}</div><div class="lbl" style="text-transform:none;margin-top:6px">Action: <b>{act}</b></div></div>', unsafe_allow_html=True); st.write("")
            st.markdown(f'<div class="card"><div class="lbl">Estimated real-world depth</div><p class="big">{b["real_cm"]:.0f} cm</p><div class="lbl" style="text-transform:none">kerb {b["kerb_submerged"]*100:.0f}% submerged · tyre {b["tyre_submerged"]*100:.0f}% submerged · confidence {b["confidence"]:.2f}</div><div class="lbl" style="text-transform:none">{b["method"]}</div></div>', unsafe_allow_html=True); st.write("")
        elif "tank" in res:
            t = res["tank"]
            st.markdown(f'<div class="card"><div class="lbl">Water depth in the tank</div><p class="big">{t["depth_cm"]:.1f} cm</p><div class="lbl" style="text-transform:none">{t["method"]} · {t["px_per_cm"]} px/cm · confidence {t["confidence"]:.2f}</div></div>', unsafe_allow_html=True); st.write("")
            st.markdown(f'<div class="card"><div class="lbl">Real-world equivalent at 1:{t["model_scale"]:g}</div><p class="big" style="color:{hexc}">{t["real_cm"]:.0f} cm · {name}</p></div>', unsafe_allow_html=True); st.write("")
            if t.get("car"):
                c = t["car"]; st.markdown(f'<div class="card"><div class="lbl">Model car</div><b>{c["submerged_fraction"]*100:.0f}%</b> of the car is under water · water reaches the <b>{c["part"]}</b> · detector confidence {c["conf"]:.2f}</div>', unsafe_allow_html=True); st.write("")
            else:
                st.caption("No car detected yet — the scale is assumed. Lower the detector confidence, or open the manual fallback.")
        else:
            st.markdown(f'<div class="card"><div class="lbl">Estimated water depth</div><p class="big" style="color:{hexc}">{dp["real_cm"]:.0f} cm · {name}</p><div class="lbl" style="text-transform:none">method: {dp["method"]} · confidence {dp["confidence"]:.2f}</div></div>', unsafe_allow_html=True); st.write("")
        if "box" not in res:
            wf = res["water_fraction"] * 100
            st.markdown(f'<div class="card"><div class="lbl">Water coverage (segmentation model)</div><p class="big">{wf:.0f}%</p></div>', unsafe_allow_html=True); st.write("")
        n_p = sum(d["name"] == "person" for d in res["detections"]); n_v = sum(d["name"] in jd.VEHICLES for d in res["detections"]); n_w = sum(d["in_water"] for d in res["detections"])
        if "box" not in res and "tank" not in res: st.markdown(f'<div class="card"><div class="lbl">Reference objects</div><b>{n_p}</b> people · <b>{n_v}</b> vehicles · <b>{n_w}</b> standing in water (used for depth)</div>', unsafe_allow_html=True); st.write("")
        for f in res["flags"]: st.markdown(f'<div class="flag">⚠ {f}</div>', unsafe_allow_html=True)
        if not res["flags"]: st.caption("No emergency flags.")
    with st.expander("How it works"):
        st.markdown(f"""
1. **Water segmentation** — a YOLOv8n-seg network we fine-tuned on **ATLANTIS** (Erfani et al. 2022; 5,195 Creative-Commons photos, 17 waterbody labels merged into *water*). Test mask mAP50 = {metrics['test_mask_mAP50']:.2f}, mAP50-95 = {metrics['test_mask_mAP50_95']:.2f}. Blue overlay = water.
2. **Object detection** — pretrained YOLOv8n (COCO) finds people, cars, buses, trucks, motorcycles. A box whose base lies inside the water mask is *standing in water*.
2b. **Learned depth classifier** — when no reference object is in view, a YOLOv8n-cls model we trained on ~300 flood photos we labelled by eye (Dry / Ankle / Knee / Wheel) gives the scene class; on 59 held-out photos the full estimator is right 68% exactly and 90% within one class.
3. **Depth — always estimated**, most trusted first: (a) a **ruler** you mark (kerb ≈ 15 cm, pole bands, wall) → exact centimetres; (b) **known-size objects standing in water** — a person is ~170 cm, a car ~150 cm tall: the part hidden below the waterline (expected height − visible height, scale from the object's width) gives the depth, combined across objects; (c) a **scene estimate** from how much of the lower frame is water when nothing else is in view (low confidence). In **rig mode** every size is scaled to the toy car you enter, so a tray with a 7 cm car behaves like a street.
4. **Demo tray (kerb + toy car)** — the striped kerb stands for a 15 cm kerb and the toy tyre for 60 cm; you drag two lines (floor, kerb top) once, and the waterline is read on the wall/kerb face from blue-tinted water (or by comparing with an empty-box reference). Classes follow the build guide: Dry (kerb visible) · Ankle (< 15 cm, kerb partly covered) · Knee (15–60 cm, tyre partly submerged) · Wheel (> 60 cm, tyre fully submerged), each with the action to take.
4b. **Glass tank (side view)** — the water surface is a long horizontal edge found with the segmentation mask and an edge detector. **The model car is the ruler**: its detected length gives pixels per centimetre and its tyres give the floor, so no printed scale is needed. Depth is read to the pixel, converted to real-world centimetres by the model scale, and the car's submerged fraction is reported.
5. **Classes** — Dry < 5 cm · Ankle 5–20 · Knee 20–50 · Wheel > 50 cm. Classes, not false precision: that is what a driver or a control room needs.

Limits: night, glare and muddy reflections lower the mask quality; depth without a ruler is indicative. In JalDrishti this module runs on existing CCTV and feeds a self-correcting flood nowcast.
""")
elif src != "Live camera — continuous":
    st.info("Pick a sample on the left, upload a photo / video, or use your camera.")
st.caption("Sample photos: Wikimedia Commons, CC BY-SA / public domain — see samples/ATTRIBUTION.md. Dataset: ATLANTIS (Erfani et al., Environmental Modelling & Software, 2022).")
