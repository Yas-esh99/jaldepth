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


# ───────────────────────── Plastic box rig (translucent box, tiny toy car, no ruler) ─────────────────────────
from collections import deque
_box_hist = deque(maxlen=7)

def box_waterline(img, strip, y_floor, y_rim, mode="tint", ref=None, hsv_lo=(85, 40, 40), hsv_hi=(135, 255, 255), min_frac=0.45):
    """Find the water surface on the front wall inside a horizontal strip (x0..x1) between rim and floor.
    'tint': rows are water when most columns fall in the blue-tint HSV range (food colouring).
    'ref' : rows are water when they differ from an empty-box reference frame. Returns (y_water, row_profile, solidity)."""
    x0, x1 = int(strip[0]), int(strip[1]); y_top, y_bot = int(min(y_rim, y_floor)), int(max(y_rim, y_floor))
    band = img[y_top:y_bot, x0:x1]
    if mode == "ref" and ref is not None:
        rb = ref[y_top:y_bot, x0:x1]; diff = np.abs(band.astype(np.int16) - rb.astype(np.int16)).sum(axis=2); wet = diff > 60
    else:
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV); wet = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8)) > 0
    prof = wet.mean(axis=1)                             # fraction of 'water' columns per row, top->bottom
    prof_s = cv2.GaussianBlur(prof.reshape(-1, 1).astype(np.float32), (1, 7), 0).ravel()
    # walk up from the floor while rows are wet (allow small gaps: ripples, car body)
    rows = prof_s[::-1]; y = 0; gap = 0
    for i, fr in enumerate(rows):
        if fr >= min_frac: y = i + 1; gap = 0
        else:
            gap += 1
            if gap > 6: break
    solidity = float(rows[:y].mean()) if y else 0.0
    return y_bot - y, prof_s, solidity

# JalDrishti demo-model classes (kerb = 15 cm real, tyre = 60 cm real), from the team's build guide
BOX_CLASSES = [("Dry", 2, "Kerb fully visible, road is dry", "No action — monitor only", (46, 139, 87)),
               ("Ankle-deep", 15, "Water covers part of the kerb (< 15 cm)", "Pre-alert, monitor closely", (48, 194, 242)),
               ("Knee-deep", 60, "Kerb not visible, tyre partly submerged (15–60 cm)", "Alert · deploy pump · advise caution", (31, 123, 224)),
               ("Wheel-deep", 1e9, "Tyre fully submerged (> 60 cm)", "Close road · diversions · strong alert", (43, 57, 192))]
def box_class(real_cm):
    for name, upper, desc, action, col in BOX_CLASSES:
        if real_cm < upper: return name, desc, action, col
    return BOX_CLASSES[-1][0], BOX_CLASSES[-1][2], BOX_CLASSES[-1][3], BOX_CLASSES[-1][4]

def analyze_box(img, strip, y_floor, y_kerb_top, kerb_real_cm=15.0, mode="tint", ref=None, tyre_real_cm=60.0, kerb_model_cm=None, hsv_lo=(85, 40, 40), hsv_hi=(135, 255, 255), min_frac=0.45, smooth=True, y_top=None):
    """Depth from the waterline on the wall/kerb face between the floor line and the top of the search band.
    Scale: the kerb's pixel height stands for kerb_real_cm (15 cm) — no ruler needed. Real depth -> demo classes (kerb / tyre)."""
    y_search_top = int(y_top) if y_top is not None else int(y_kerb_top - 3.0 * (y_floor - y_kerb_top))   # allow water well above the kerb (up to ~60 cm real)
    yw, prof, sol = box_waterline(img, strip, y_floor, max(0, y_search_top), mode, ref, hsv_lo, hsv_hi, min_frac)
    ppc_real = (y_floor - y_kerb_top) / max(1e-6, kerb_real_cm); real = max(0.0, (y_floor - yw) / ppc_real)
    if smooth:
        _box_hist.append(real); real = float(np.median(_box_hist))
    name, desc, action, col = box_class(real)
    model_cm = round(real * kerb_model_cm / kerb_real_cm, 1) if kerb_model_cm else None
    kerb_frac = float(np.clip(real / kerb_real_cm, 0, 1)); tyre_frac = float(np.clip(real / tyre_real_cm, 0, 1))
    conf = round(float(np.clip(0.35 + 0.6 * sol, 0.35, 0.95)) if real > 1 else 0.9, 2)
    return {"y_surface": int(yw), "real_cm": round(real, 0), "model_cm": model_cm, "class": name, "description": desc, "action": action, "confidence": conf,
            "method": f"waterline on the kerb/wall ({'blue tint' if mode == 'tint' else 'reference frame'}) · scale: kerb = {kerb_real_cm:g} cm real",
            "px_per_real_cm": round(ppc_real, 3), "y_floor": int(y_floor), "y_kerb_top": int(y_kerb_top), "y_search_top": int(max(0, y_search_top)), "strip": [int(strip[0]), int(strip[1])],
            "kerb_submerged": round(kerb_frac, 2), "tyre_submerged": round(tyre_frac, 2), "depth_cm": round(real, 0), "profile": prof.tolist()}

def draw_box(img, res, kerb_real_cm=15.0):
    vis = img.copy(); x0, x1 = res["strip"]; yf, yk, yt, yw = res["y_floor"], res["y_kerb_top"], res["y_search_top"], res["y_surface"]
    cv2.rectangle(vis, (x0, yt), (x1, yf), (255, 255, 255), 1)
    cv2.line(vis, (x0, yf), (x1, yf), (80, 220, 80), 2); cv2.putText(vis, "road / floor = 0", (x0 + 6, yf + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 80), 2)
    cv2.line(vis, (x0, yk), (x1, yk), (80, 220, 80), 2); cv2.putText(vis, f"kerb top = {kerb_real_cm:g} cm", (x0 + 6, yk - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 80), 2)
    ppc = res["px_per_real_cm"]
    for cm in (15, 30, 45, 60):
        yy = int(yf - cm * ppc)
        if yy > yt: cv2.line(vis, (x1 - 30, yy), (x1, yy), (80, 220, 80), 1); cv2.putText(vis, f"{cm}", (x1 - 60, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 80), 1)
    name, desc, action, col = box_class(res["real_cm"])
    if res["real_cm"] > 1:
        cv2.line(vis, (x0, yw), (x1, yw), (0, 140, 255), 3); cv2.putText(vis, f'{res["real_cm"]:.0f} cm', (x0 + 6, yw - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 62), (27, 58, 107), -1)
    cv2.putText(vis, f'WATER LEVEL: {name.upper()}  ·  {res["real_cm"]:.0f} cm real  ·  conf {res["confidence"]:.2f}', (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
    cv2.putText(vis, f'{desc}  →  {action}', (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis

_ref = {"frame": None, "capture": False}
def maybe_capture_ref(img):
    if _ref.get("capture"): _ref["frame"] = img.copy(); _ref["capture"] = False
    return _ref.get("frame")
