"""Download freely licensed Indian street-flood photos (and one video) from Wikimedia Commons into samples/, with attribution."""
import os, re, json, sys, requests, urllib.parse
UA = {"User-Agent": "JalDepth-demo/0.1 (SIH2026 student project; aryanpatel142006@gmail.com)"}; API = "https://commons.wikimedia.org/w/api.php"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(HERE, "samples"); os.makedirs(OUT, exist_ok=True)
CATS = ["Category:Floods in Chennai", "Category:2017 Mumbai flood", "Category:Floods in Gujarat"]
EXTRA = ["File:Bombay flooded street.jpg", "File:Bombay flooded street2.jpg", "File:Mumbai, India ⥁ (27674519391).jpg", "File:Flooded Marina beach Near Service Road ( 2015 ).jpg",
         "File:Aadhaar Bangalore flood.jpg", "File:Bangalore monsoon.jpg", "File:A Santro that went \"off road\" (35520266).jpg", "File:Monsoon (36877269620).jpg", "File:Aug 29 2017 Mumbai Floods.webm"]
SKIP = re.compile(r"aerial|MODIS|railway|Navy|map|satellite|Shedhi", re.I)
titles = list(EXTRA)
for c in CATS:
    r = requests.get(API, params={"action": "query", "list": "categorymembers", "cmtitle": c, "cmtype": "file", "cmlimit": 200, "format": "json"}, headers=UA, timeout=30).json()
    titles += [m["title"] for m in r["query"]["categorymembers"] if not SKIP.search(m["title"])]
titles = list(dict.fromkeys(titles)); rows = []
for i in range(0, len(titles), 20):
    chunk = titles[i:i + 20]
    r = requests.get(API, params={"action": "query", "titles": "|".join(chunk), "prop": "imageinfo", "iiprop": "url|size|extmetadata|mime", "iiurlwidth": 1280, "format": "json"}, headers=UA, timeout=60).json()
    for p in r["query"]["pages"].values():
        ii = (p.get("imageinfo") or [{}])[0]; em = ii.get("extmetadata", {}); mime = ii.get("mime", "")
        lic = em.get("LicenseShortName", {}).get("value", ""); author = re.sub("<[^>]+>", "", em.get("Artist", {}).get("value", "")).strip()[:80]
        if not ii.get("url") or ("Public domain" not in lic and "CC" not in lic and "GODL" not in lic): continue
        url = ii.get("thumburl") if mime.startswith("image") else ii["url"]
        safe = re.sub(r"[^A-Za-z0-9]+", "_", p["title"].replace("File:", ""))[:60].strip("_"); ext = ".jpg" if mime.startswith("image") else os.path.splitext(ii["url"])[1]
        fn = safe + ext; data = requests.get(url, headers=UA, timeout=120).content
        if len(data) < 20000: continue
        open(os.path.join(OUT, fn), "wb").write(data); rows.append({"file": fn, "title": p["title"], "author": author, "license": lic, "source": ii.get("descriptionurl")}); print("saved", fn, lic, len(data) // 1024, "KB")
json.dump(rows, open(os.path.join(OUT, "attribution.json"), "w"), indent=1, ensure_ascii=False)
with open(os.path.join(OUT, "ATTRIBUTION.md"), "w") as f:
    f.write("# Sample images — Wikimedia Commons\n\nAll samples are freely licensed; credit and licence per file:\n\n| File | Title | Author | Licence | Source |\n|---|---|---|---|---|\n")
    for r_ in rows: f.write(f"| {r_['file']} | {r_['title']} | {r_['author']} | {r_['license']} | {r_['source']} |\n")
print("total", len(rows))
