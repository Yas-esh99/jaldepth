"""Tank rig (side view): a glass tank with a printed cm scale, a road bed and a model car, camera looking from the side.
Depth = position of the horizontal water surface against the calibrated scale. Robust and precise: the surface is a long
horizontal edge, so we combine the water-segmentation mask (top edge per column) with a row-wise horizontal-edge detector."""
import numpy as np, cv2
import jaldepth as jd

def surface_line(img, roi, mask=None):
    """Return (y_surface, confidence, method). roi = (x0, y0, x1, y1) region inside the tank glass."""
    x0, y0, x1, y1 = [int(v) for v in roi]; crop = img[y0:y1, x0:x1]
    h = y1 - y0
    # 1) row-wise horizontal edge energy (water surface = long horizontal transition), smoothed
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32); hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    dy = np.abs(np.diff(cv2.GaussianBlur(g, (5, 5), 0), axis=0)).mean(axis=1); ds = np.abs(np.diff(cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0), axis=0)).mean(axis=1)
    e = cv2.GaussianBlur((dy / (dy.max() + 1e-6) + ds / (ds.max() + 1e-6)).reshape(-1, 1), (1, 9), 0).ravel()
    # the water surface also separates a more saturated/darker lower region from the upper region: reward rows where below is more saturated than above
    sat = hsv[:, :, 1].mean(axis=1); cum = np.cumsum(sat); n = np.arange(1, h + 1)
    above = cum / n; below = (cum[-1] - cum) / np.maximum(1, h - n)
    contrast = np.clip((below - above)[:-1] / 60.0, 0, 1)
    score = e * (0.5 + contrast); score[: int(0.05 * h)] = 0; score[int(0.95 * h):] = 0
    y_edge = int(np.argmax(score)); conf_edge = float(score[y_edge] / (score.mean() + 1e-6)) / 20.0
    # 2) segmentation mask top edge (median over columns)
    y_mask = None
    if mask is not None:
        m = mask[y0:y1, x0:x1] > 0; cols = m.any(axis=0)
        if cols.mean() > 0.3:
            tops = np.array([np.argmax(m[:, c]) for c in np.where(cols)[0]]); y_mask = int(np.median(tops))
    if y_mask is not None and abs(y_mask - y_edge) < 0.06 * h:
        return y0 + int(round(0.5 * (y_mask + y_edge))), min(0.98, 0.6 + conf_edge), "surface: segmentation + edge agree"
    if y_mask is not None and conf_edge < 0.3: return y0 + y_mask, 0.6, "surface: segmentation mask"
    return y0 + y_edge, float(np.clip(0.4 + conf_edge, 0.4, 0.9)), "surface: horizontal edge"

def depth_from_scale(y_surface, y_floor, y_mark, mark_cm):
    px_per_cm = (y_floor - y_mark) / max(1e-6, mark_cm)
    return max(0.0, (y_floor - y_surface) / px_per_cm), px_per_cm

def car_submersion(img, y_surface, y_floor, conf_det=0.25):
    dets = [d for d in jd.detect(img, conf_det) if d["name"] in ("car", "truck", "bus")]
    if not dets: return None
    d = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
    x0, y0, x1, y1 = d["bbox"]; top = y0; bottom = min(y1, y_floor)
    frac = float(np.clip((bottom - y_surface) / max(1.0, bottom - top), 0, 1)) if y_surface < bottom else 0.0
    part = "clear" if frac < 0.05 else "wheels" if frac < 0.3 else "doors / body" if frac < 0.65 else "windows" if frac < 0.9 else "roof"
    return {"bbox": d["bbox"], "conf": d["conf"], "submerged_fraction": round(frac, 2), "part": part}

_last_cal = {}   # remembers the last confident car-based calibration (per size key) so live video survives frames where the car is not detected

def calibrate_from_car(img, car_len_cm, conf_det=0.25):
    """The model car is the ruler: its length in pixels gives px/cm, its tyre bottoms give the floor. Returns dict or None."""
    dets = [d for d in jd.detect(img, conf_det) if d["name"] in ("car", "truck", "bus")]
    if not dets: return None
    d = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
    x0, y0, x1, y1 = d["bbox"]; w = x1 - x0
    if w < 0.15 * img.shape[1]: return None                      # too small to be the model car in a side view
    return {"px_per_cm": w / car_len_cm, "y_floor": y1, "bbox": d["bbox"], "conf": d["conf"]}

def analyze_tank(img, roi=None, car_len_cm=24.0, real_len_cm=430.0, manual=None, conf_seg=0.25, conf_det=0.25, use_mask=True):
    """Ruler-free: calibration from the model car (fallback: cached calibration, then manual lines {y_floor, y_mark, mark_cm})."""
    H, W = img.shape[:2]; roi = roi or (int(0.04 * W), int(0.04 * H), int(0.96 * W), int(0.9 * H))
    key = (W, H); cal = calibrate_from_car(img, car_len_cm, conf_det); src = "car"
    if cal and cal["conf"] >= 0.35: _last_cal[key] = cal
    if not cal:
        if key in _last_cal: cal = _last_cal[key]; src = "car (remembered)"
        elif manual: ppc = (manual["y_floor"] - manual["y_mark"]) / max(1e-6, manual["mark_cm"]); cal = {"px_per_cm": ppc, "y_floor": manual["y_floor"], "bbox": None, "conf": 1.0}; src = "manual lines"
        else: cal = {"px_per_cm": W / (car_len_cm * 3.0), "y_floor": roi[3], "bbox": None, "conf": 0.2}; src = "assumed (no car found)"
    roi = (roi[0], roi[1], roi[2], int(min(roi[3], cal["y_floor"])))
    mask = jd.water_mask(img, conf_seg, tinted=True) if use_mask else None
    ys, conf, method = surface_line(img, roi, mask)
    depth_cm = max(0.0, (cal["y_floor"] - ys) / cal["px_per_cm"]); model_scale = real_len_cm / car_len_cm
    real_cm = depth_cm * model_scale; cls = jd.depth_class(real_cm)[0]
    car = None
    if cal.get("bbox"):
        x0, y0, x1, y1 = cal["bbox"]; frac = float(np.clip((y1 - ys) / max(1.0, y1 - y0), 0, 1)) if ys < y1 else 0.0
        part = "clear" if frac < 0.05 else "wheels" if frac < 0.3 else "doors / body" if frac < 0.65 else "windows" if frac < 0.9 else "roof"
        car = {"bbox": cal["bbox"], "conf": cal["conf"], "submerged_fraction": round(frac, 2), "part": part}
    conf_total = round(conf * (0.95 if src.startswith("car") else 0.6 if src == "manual lines" else 0.3), 2)
    return {"y_surface": int(ys), "depth_cm": round(depth_cm, 1), "real_cm": round(real_cm, 0), "class": cls, "confidence": conf_total, "method": f"{method} · scale from {src}",
            "px_per_cm": round(cal["px_per_cm"], 2), "y_floor": int(cal["y_floor"]), "car": car, "model_scale": round(model_scale, 1), "roi": roi}, mask

def draw_tank(img, res):
    vis = img.copy(); x0, y0, x1, y1 = [int(v) for v in res["roi"]]; y_floor = res["y_floor"]
    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 1)
    cv2.line(vis, (x0, int(y_floor)), (x1, int(y_floor)), (80, 220, 80), 2); cv2.putText(vis, "floor (tyre bottoms) = 0 cm", (x0 + 8, int(y_floor) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2)
    ys = res["y_surface"]; cv2.line(vis, (x0, ys), (x1, ys), (0, 140, 255), 3)
    # depth ticks derived from the car scale, every 2 cm
    ppc = res["px_per_cm"]; cm = 2
    while y_floor - cm * ppc > y0:
        yy = int(y_floor - cm * ppc); cv2.line(vis, (x0, yy), (x0 + (18 if cm % 10 else 30), yy), (80, 220, 80), 1)
        if cm % 10 == 0: cv2.putText(vis, f"{cm} cm", (x0 + 34, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1)
        cm += 2
    name, col = jd.depth_class(res["real_cm"])
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (27, 58, 107), -1)
    cv2.putText(vis, f'WATER DEPTH {res["depth_cm"]:.1f} cm in tank  ->  {res["real_cm"]:.0f} cm real at 1:{res["model_scale"]:g}  ·  {name.upper()}  ·  conf {res["confidence"]:.2f}', (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if res.get("car"):
        c = res["car"]; bx0, by0, bx1, by1 = [int(v) for v in c["bbox"]]; cv2.rectangle(vis, (bx0, by0), (bx1, by1), (60, 220, 255), 2)
        cv2.putText(vis, f'car {c["conf"]:.2f} · {c["submerged_fraction"]*100:.0f}% submerged · water at {c["part"]}', (bx0, max(50, by0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 255), 2)
    return vis
