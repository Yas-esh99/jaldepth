"""JalDepth core: water segmentation (YOLOv8n-seg fine-tuned on ATLANTIS), object detection (YOLOv8n COCO),
ruler-based depth estimation and coarse vehicle-based depth. Pure functions on numpy images (BGR)."""
import os, json
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
WATER_W = os.path.join(HERE, "models", "yolov8n-seg-water.pt"); DET_W = os.path.join(HERE, "models", "yolov8n.pt")
CLASSES = [("Dry", 5, (46, 139, 87)), ("Ankle-deep", 20, (48, 194, 242)), ("Knee-deep", 50, (31, 123, 224)), ("Wheel-deep", 1e9, (43, 57, 192))]
VEHICLES = {"car": 150.0, "truck": 250.0, "bus": 300.0, "motorcycle": 110.0}   # typical heights (cm) used for coarse depth

def depth_class(cm: float):
    for name, upper, col in CLASSES:
        if cm < upper: return name, col
    return CLASSES[-1][0], CLASSES[-1][2]

_models = {}
def load_models():
    if not _models:
        from ultralytics import YOLO
        _models["water"] = YOLO(WATER_W); _models["det"] = YOLO(DET_W)
    return _models

def water_mask(img, conf=0.25):
    res = load_models()["water"].predict(img, conf=conf, verbose=False, imgsz=640)[0]
    m = np.zeros(img.shape[:2], np.uint8)
    if res.masks is not None:
        for poly in res.masks.xy:
            if len(poly) >= 3: cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    return m

def detect(img, conf=0.3):
    res = load_models()["det"].predict(img, conf=conf, verbose=False, imgsz=640)[0]; names = res.names; out = []
    for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
        n = names[int(c)]
        if n in VEHICLES or n == "person": out.append({"name": n, "bbox": [float(x) for x in b], "conf": float(s)})
    return out

def in_water(bbox, mask):
    x0, y0, x1, y1 = [int(v) for v in bbox]; cx = (x0 + x1) // 2; yb = min(mask.shape[0] - 1, y1)
    patch = mask[max(0, yb - 12):yb + 1, max(0, cx - 20):cx + 21]
    return patch.size > 0 and (patch > 0).mean() > 0.3

def vehicle_depth(bbox, mask, height_cm):
    """Fraction of the vehicle box below the water top at its centre column × typical height."""
    x0, y0, x1, y1 = [int(v) for v in bbox]; cx = (x0 + x1) // 2
    col = mask[y0:y1, max(0, cx - 3):cx + 4].max(axis=1) > 0
    if not col.any(): return None
    top = int(np.argmax(col)); frac = (y1 - y0 - top) / max(1, (y1 - y0))
    return float(np.clip(frac * height_cm, 0, 150))

def ruler_depth(mask, x, y_bottom, y_top, height_cm, band=10):
    """Walk up the ruler column from its base while the mask says water; convert pixels to cm."""
    x0, x1 = max(0, x - band), min(mask.shape[1], x + band + 1)
    col = mask[y_top:y_bottom + 1, x0:x1] > 0; frac = col.mean(axis=1)[::-1]
    y = 0; gap = 0
    for i, fr in enumerate(frac):
        if fr >= 0.5: y = i + 1; gap = 0
        else:
            gap += 1
            if gap > 4: break
    px = y; cm = px / max(1, (y_bottom - y_top)) * height_cm
    solidity = float(frac[:y].mean()) if y else 1.0
    return {"depth_cm": round(cm, 1), "waterline_y": int(y_bottom - px), "solidity": round(solidity, 2), "class": depth_class(cm)[0]}

def analyze_image(img, ruler=None, conf_seg=0.25, conf_det=0.3):
    mask = water_mask(img, conf_seg); dets = detect(img, conf_det)
    for d in dets:
        d["in_water"] = bool(in_water(d["bbox"], mask))
        d["depth_cm"] = vehicle_depth(d["bbox"], mask, VEHICLES[d["name"]]) if (d["name"] in VEHICLES and d["in_water"]) else None
    depths = [d["depth_cm"] for d in dets if d.get("depth_cm")]
    out = {"water_fraction": float((mask > 0).mean()), "detections": dets, "vehicle_depth_cm": float(np.median(depths)) if depths else None}
    if out["vehicle_depth_cm"] is not None: out["vehicle_class"] = depth_class(out["vehicle_depth_cm"])[0]
    if ruler: out["ruler"] = ruler_depth(mask, **ruler)
    flags = []
    if any(d["name"] == "person" and d["in_water"] for d in dets): flags.append("Person standing in water")
    if any(d["name"] in VEHICLES and d["in_water"] and (d.get("depth_cm") or 0) >= 50 for d in dets): flags.append("Vehicle in wheel-deep water")
    out["flags"] = flags
    return out, mask

def draw(img, mask, result, ruler=None):
    vis = img.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ov = vis.copy(); cv2.drawContours(ov, cnts, -1, (255, 160, 30), -1); vis = cv2.addWeighted(vis, 0.65, ov, 0.35, 0); cv2.drawContours(vis, cnts, -1, (255, 200, 60), 2)
    for d in result["detections"]:
        x0, y0, x1, y1 = [int(v) for v in d["bbox"]]; col = (60, 220, 255) if d["in_water"] else (200, 200, 200)
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        lab = f'{d["name"]} {d["conf"]:.2f}' + (f' · ~{d["depth_cm"]:.0f} cm' if d.get("depth_cm") else "") + (" · in water" if d["in_water"] else "")
        cv2.rectangle(vis, (x0, max(0, y0 - 22)), (x0 + 8 * len(lab) + 6, y0), col, -1); cv2.putText(vis, lab, (x0 + 3, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    if ruler and "ruler" in result:
        r = result["ruler"]; x, yb, yt = ruler["x"], ruler["y_bottom"], ruler["y_top"]
        cv2.rectangle(vis, (x - 12, yt), (x + 12, yb), (255, 255, 255), 2); cv2.line(vis, (x - 60, r["waterline_y"]), (x + 60, r["waterline_y"]), (0, 140, 255), 3)
        cv2.putText(vis, f'{r["depth_cm"]:.0f} cm · {r["class"]}', (x + 18, r["waterline_y"] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
    return vis
