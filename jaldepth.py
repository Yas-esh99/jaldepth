"""JalDepth core: water segmentation (YOLOv8n-seg fine-tuned on ATLANTIS + Indian footage), object detection (YOLOv8n COCO),
and a depth estimator that ALWAYS returns a number: ruler > known-size objects in water > scene heuristic.
Scenes: 'street' (real CCTV / photos) or 'rig' (table-top tray with a toy car, tinted water)."""
import os, json
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
WATER_W = os.path.join(HERE, "models", "yolov8n-seg-water.pt"); WATER_IN_W = os.path.join(HERE, "models", "yolov8n-seg-water-in.pt"); DET_W = os.path.join(HERE, "models", "yolov8n.pt")
CLASSES = [("Dry", 5, (46, 139, 87)), ("Ankle-deep", 20, (48, 194, 242)), ("Knee-deep", 50, (31, 123, 224)), ("Wheel-deep", 1e9, (43, 57, 192))]
# typical real-world sizes in cm: (height, width when seen frontally, length when seen from the side)
REF = {"person": (170, 45, 45), "car": (150, 180, 450), "truck": (300, 250, 800), "bus": (320, 250, 1200), "motorcycle": (110, 60, 210), "bicycle": (100, 45, 175),
       "fire hydrant": (75, 30, 30), "bench": (45, 150, 150), "stop sign": (210, 75, 75), "dog": (55, 25, 80), "cow": (140, 60, 220)}
VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle"}

def depth_class(cm: float):
    for name, upper, col in CLASSES:
        if cm < upper: return name, col
    return CLASSES[-1][0], CLASSES[-1][2]

_models = {}
def load_models():
    if not _models:
        from ultralytics import YOLO
        _models["water"] = YOLO(WATER_W); _models["det"] = YOLO(DET_W)
        _models["water_in"] = YOLO(WATER_IN_W) if os.path.exists(WATER_IN_W) else None   # stage 2: self-trained on Indian footage
    return _models

def _mask_from(model, img, conf):
    res = model.predict(img, conf=conf, verbose=False, imgsz=640)[0]; m = np.zeros(img.shape[:2], np.uint8)
    if res.masks is not None:
        for poly in res.masks.xy:
            if len(poly) >= 3: cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    return m

def water_mask(img, conf=0.25, tinted=False):
    """Indian-adapted model first; if it finds almost nothing, fall back to the ATLANTIS model (they miss different scenes)."""
    mods = load_models()
    m = _mask_from(mods["water_in"], img, conf) if mods.get("water_in") else _mask_from(mods["water"], img, conf)
    if (m > 0).mean() < 0.03:
        cands = []
        if mods.get("water_in"): cands.append(_mask_from(mods["water"], img, conf))
        small = cv2.resize(img, (img.shape[1] * 3 // 4, img.shape[0] * 3 // 4))          # second pass at 75% scale: the models are scale-sensitive on some scenes
        for mod in ([mods["water_in"]] if mods.get("water_in") else []) + [mods["water"]]:
            cands.append(cv2.resize(_mask_from(mod, small, conf), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST))
        best = max(cands, key=lambda c: (c > 0).mean()) if cands else m
        if (best > 0).mean() > (m > 0).mean(): m = best
    if tinted:   # table-top rig: union with a blue-tint mask (food colouring in the tray)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); hm = cv2.inRange(hsv, np.array([85, 50, 40]), np.array([135, 255, 255]))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)); hm = cv2.morphologyEx(cv2.morphologyEx(hm, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k); m = cv2.bitwise_or(m, hm)
    return m

def detect(img, conf=0.3):
    res = load_models()["det"].predict(img, conf=conf, verbose=False, imgsz=640)[0]; names = res.names; out = []
    for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
        n = names[int(c)]
        if n in REF: out.append({"name": n, "bbox": [float(x) for x in b], "conf": float(s)})
    return out

def in_water(bbox, mask):
    x0, y0, x1, y1 = [int(v) for v in bbox]; h = mask.shape[0]; yb = min(h - 1, y1); cx = (x0 + x1) // 2
    patch = mask[max(0, yb - 14):min(h, yb + 6), max(0, cx - max(20, (x1 - x0) // 3)):min(mask.shape[1], cx + max(20, (x1 - x0) // 3))]
    return patch.size > 0 and (patch > 0).mean() > 0.25

def object_depth(d, mask, scale=1.0):
    """Submerged depth of a known-size object = expected height − visible height, with pixel scale taken from the box width
    (width is not affected by submersion). Also uses how far the water mask reaches up the box. Returns cm or None."""
    x0, y0, x1, y1 = d["bbox"]; w_px, h_px = max(1.0, x1 - x0), max(1.0, y1 - y0); H, Wf, L = REF[d["name"]]; H, Wf, L = H * scale, Wf * scale, L * scale
    side = (w_px / h_px) > (1.6 if d["name"] in VEHICLES else 0.9)
    px_per_cm = w_px / (L if side else Wf)
    visible_cm = h_px / px_per_cm; sub = H - visible_cm                      # geometric estimate
    m = mask[int(y0):int(y1), int(max(0, (x0 + x1) / 2 - 4)):int((x0 + x1) / 2 + 5)]
    frac = 0.0
    if m.size:
        col = m.max(axis=1) > 0
        if col.any(): frac = (len(col) - int(np.argmax(col))) / len(col)     # part of the box below the water top
    overlap = frac * H
    est = max(sub, overlap * 0.6) if sub > 0 else overlap
    est = float(np.clip(est, 0, 0.8 * H))          # a box rarely shows the full object; cap to avoid over-estimates
    return est if (est > 0 or frac > 0) else 0.0

def ruler_depth(mask, x, y_bottom, y_top, height_cm, band=10):
    x0, x1 = max(0, x - band), min(mask.shape[1], x + band + 1)
    col = mask[y_top:y_bottom + 1, x0:x1] > 0; frac = col.mean(axis=1)[::-1] if col.size else np.zeros(1)
    y = 0; gap = 0
    for i, fr in enumerate(frac):
        if fr >= 0.5: y = i + 1; gap = 0
        else:
            gap += 1
            if gap > 4: break
    cm = y / max(1, (y_bottom - y_top)) * height_cm; solidity = float(frac[:y].mean()) if y else 1.0
    return {"depth_cm": round(cm, 1), "waterline_y": int(y_bottom - y), "solidity": round(solidity, 2), "class": depth_class(cm)[0]}

def scene_depth(mask, scale=1.0):
    """No reference in view: estimate from how much of the lower half of the frame is water and how high the water reaches.
    Calibrated to the class thresholds; low confidence by design."""
    h = mask.shape[0]; lower = (mask[h // 2:] > 0).mean(); rows = np.where((mask > 0).mean(axis=1) > 0.3)[0]
    reach = 1 - rows.min() / h if rows.size else 0.0                          # 0 = no water, 1 = water to the top
    score = 0.65 * lower + 0.35 * reach
    cm = float(np.interp(score, [0, 0.08, 0.3, 0.55, 0.8], [0, 5, 15, 28, 40])) * scale   # no reference in view: never claim more than knee-deep
    return cm

def analyze_image(img, ruler=None, conf_seg=0.25, conf_det=0.3, scene="street", toy_len_cm=7.0):
    scale = 1.0 if scene == "street" else toy_len_cm / REF["car"][2]         # rig: a 7 cm toy car stands for a 450 cm car
    mask = water_mask(img, conf_seg, tinted=(scene == "rig")); dets = detect(img, conf_det if scene == "street" else max(0.15, conf_det - 0.1))
    for d in dets:
        d["in_water"] = bool(in_water(d["bbox"], mask)); d["depth_cm"] = object_depth(d, mask, scale) if d["in_water"] else None
    out = {"water_fraction": float((mask > 0).mean()), "detections": dets, "scene": scene, "scale": scale}
    # ---- depth: always produce one number ----
    if ruler:
        r = ruler_depth(mask, **ruler); out["ruler"] = r; depth, conf, method = r["depth_cm"], min(0.95, 0.6 + 0.35 * r["solidity"]), "ruler (known-height reference)"
    else:
        objs = [d["depth_cm"] for d in dets if d.get("depth_cm") is not None]
        if objs:
            depth = float(np.percentile(objs, 35)); conf = 0.55 + 0.1 * min(3, len(objs) - 1); method = f"{len(objs)} known-size object(s) in water"
        else:
            depth = scene_depth(mask, scale); conf = 0.3; method = "scene estimate (no reference in view)"
    real_cm = depth if ruler else (depth / scale if scale != 1.0 else depth)   # ruler height is entered in real-world cm
    out["depth"] = {"cm": round(depth, 1), "real_cm": round(real_cm, 1), "class": depth_class(real_cm)[0], "confidence": round(conf, 2), "method": method}
    flags = []
    if any(d["name"] == "person" and d["in_water"] for d in dets): flags.append("Person standing in water")
    if any(d["name"] in VEHICLES and d["in_water"] and ((d.get("depth_cm") or 0) / scale) >= 50 for d in dets): flags.append("Vehicle in wheel-deep water")
    if out["depth"]["class"] == "Wheel-deep": flags.append("Road likely impassable")
    out["flags"] = list(dict.fromkeys(flags))
    return out, mask

def draw(img, mask, result, ruler=None):
    vis = img.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ov = vis.copy(); cv2.drawContours(ov, cnts, -1, (255, 160, 30), -1); vis = cv2.addWeighted(vis, 0.65, ov, 0.35, 0); cv2.drawContours(vis, cnts, -1, (255, 200, 60), 2)
    sc = result.get("scale", 1.0)
    for d in result["detections"]:
        x0, y0, x1, y1 = [int(v) for v in d["bbox"]]; col = (60, 220, 255) if d["in_water"] else (200, 200, 200)
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        lab = f'{d["name"]} {d["conf"]:.2f}' + (f' · ~{d["depth_cm"] / sc:.0f} cm' if d.get("depth_cm") is not None else "") + (" · in water" if d["in_water"] else "")
        cv2.rectangle(vis, (x0, max(0, y0 - 22)), (x0 + 8 * len(lab) + 6, y0), col, -1); cv2.putText(vis, lab, (x0 + 3, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    if ruler and "ruler" in result:
        r = result["ruler"]; x, yb, yt = ruler["x"], ruler["y_bottom"], ruler["y_top"]
        cv2.rectangle(vis, (x - 12, yt), (x + 12, yb), (255, 255, 255), 2); cv2.line(vis, (x - 60, r["waterline_y"]), (x + 60, r["waterline_y"]), (0, 140, 255), 3)
    dp = result["depth"]; name, col = depth_class(dp["real_cm"])
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (27, 58, 107), -1)
    cv2.putText(vis, f'DEPTH ~{dp["real_cm"]:.0f} cm  {name.upper()}  ·  conf {dp["confidence"]:.2f}  ·  {dp["method"]}  ·  water {result["water_fraction"] * 100:.0f}%', (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis
