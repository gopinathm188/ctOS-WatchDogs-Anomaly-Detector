#!/usr/bin/env python3
"""
ctOS Anomaly Detection Backend — Jetson Orin Edition
Watch Dogs Theme | YOLOv8 | WebSocket + MJPEG Stream | CSV Logging

Streams video to browser via MJPEG on port 8080
Sends detection data via WebSocket on port 8765

Usage:
    conda activate dev_38
    pip install aiohttp --index-url https://pypi.org/simple/
    python3 anomaly_detector.py --camera 0

Then open watchdogs_ui.html in browser
"""

import argparse
import asyncio
import csv
import json
import math
import os
import time
import random
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ultralytics import YOLO

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    print("[WARNING] pip install websockets --index-url https://pypi.org/simple/")

try:
    from aiohttp import web
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
    print("[WARNING] pip install aiohttp --index-url https://pypi.org/simple/")

# ════════════════════════════════════════════
#   RULES
# ════════════════════════════════════════════
RULES = {
    "forbidden_classes":  ["knife", "scissors", "baseball bat"],
    "normal_classes":     ["person", "car", "truck", "bus", "bicycle",
                           "motorcycle", "chair", "bottle", "cup",
                           "backpack", "laptop", "book", "dog", "cat"],
    "max_persons":        3,
    "proximity_thresh":   0.25,
    "min_confidence":     0.45,
    "img_size":           640,
    "cooldown_seconds":   2.0,
}

OUTPUT_DIR = Path("output")
LOG_FILE   = OUTPUT_DIR / "anomaly_log.csv"
IMG_DIR    = OUTPUT_DIR / "anomaly_images"

# Watch Dogs BGR palette
C_GREEN  = (0,   255, 159)
C_BLUE   = (255, 207,   0)
C_RED    = ( 68,  34, 255)
C_AMBER  = (  0, 170, 255)
C_DARK   = ( 16,  12,   8)
C_PANEL  = ( 32,  21,  13)
C_TEXT   = (150, 190, 220)
C_DIM    = ( 60,  90, 100)
FONT     = cv2.FONT_HERSHEY_SIMPLEX
FONT_MON = cv2.FONT_HERSHEY_PLAIN

# ════════════════════════════════════════════
#   STATE
# ════════════════════════════════════════════
last_log_time = defaultdict(float)
ws_clients    = set()
object_ids    = {}
latest_frame  = None          # shared MJPEG frame (JPEG bytes)
stats = {
    "total_frames":    0,
    "total_anomalies": 0,
    "last_anomaly":    "None",
    "last_anomaly_ts": None,
    "persons":         0,
    "objects":         0,
    "start_time":      time.time(),
}

# ════════════════════════════════════════════
#   SETUP
# ════════════════════════════════════════════
def setup_output():
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp","frame","rule","severity","detail","objects_in_frame"
            ])
    print(f"[ctOS] Log : {LOG_FILE.resolve()}")
    print(f"[ctOS] Imgs: {IMG_DIR.resolve()}")

def make_id():
    return "OBJ-" + "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789", k=4))

def centroid_dist(a, b):
    return math.sqrt((a["cx"]-b["cx"])**2 + (a["cy"]-b["cy"])**2)

def pulse(t, speed=3.0):
    return 0.5 + 0.5 * math.sin(t * speed)

def secs_since(ts):
    return round(time.time()-ts, 1) if ts else None

# ════════════════════════════════════════════
#   ANOMALY RULES
# ════════════════════════════════════════════
def evaluate_rules(dets, frame_idx):
    violations = []
    now = time.time()
    persons = [d for d in dets if d["label"].lower() == "person"]

    # Rule 1 — Forbidden weapon
    for d in dets:
        for f in RULES["forbidden_classes"]:
            if f.lower() in d["label"].lower():
                key = f"weapon_{d['label'].lower().replace(' ','_')}"
                if now - last_log_time[key] >= RULES["cooldown_seconds"]:
                    violations.append({
                        "key": key, "rule": "forbidden_object",
                        "severity": "critical",
                        "detail": f"WEAPON DETECTED: '{d['label']}' ({d['conf']:.0%} conf)",
                        "label": d["label"],
                    })
                    last_log_time[key] = now

    # Rule 2 — Overcrowd
    if len(persons) > RULES["max_persons"]:
        key = "overcrowd"
        if now - last_log_time[key] >= RULES["cooldown_seconds"]:
            violations.append({
                "key": key, "rule": "overcrowd",
                "severity": "warning",
                "detail": f"{len(persons)} persons in frame (limit {RULES['max_persons']})",
                "label": "person",
            })
            last_log_time[key] = now

    # Rule 3 — Proximity
    if len(persons) >= 2:
        for i in range(len(persons)):
            for j in range(i+1, len(persons)):
                dist = centroid_dist(persons[i], persons[j])
                if dist < RULES["proximity_thresh"]:
                    key = "proximity"
                    if now - last_log_time[key] >= RULES["cooldown_seconds"]:
                        violations.append({
                            "key": key, "rule": "proximity",
                            "severity": "warning",
                            "detail": f"Persons too close: {dist*100:.0f}% frame apart",
                            "label": "person",
                        })
                        last_log_time[key] = now

    return violations

# ════════════════════════════════════════════
#   LOGGING
# ════════════════════════════════════════════
def log_anomaly(frame_idx, v, dets, frame=None):
    ts = datetime.now().isoformat(timespec="seconds")
    classes = ", ".join(sorted(set(d["label"] for d in dets)))
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            ts, f"{frame_idx:06d}", v["rule"], v["severity"], v["detail"], classes
        ])
    print(f"[ANOMALY] {ts} | {v['severity'].upper()} | {v['detail']}")
    stats["total_anomalies"] += 1
    stats["last_anomaly"]    = v["detail"]
    stats["last_anomaly_ts"] = time.time()
    if frame is not None:
        fname = IMG_DIR / f"anomaly_{ts.replace(':','-')}_{v['key']}.jpg"
        cv2.imwrite(str(fname), frame)

# ════════════════════════════════════════════
#   DRAW — Watch Dogs overlay
# ════════════════════════════════════════════
def draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2, length=16):
    pts  = [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]
    dirs = [(1,1),(-1,1),(1,-1),(-1,-1)]
    for (cx,cy),(dx,dy) in zip(pts,dirs):
        cv2.line(frame,(cx,cy),(cx+dx*length,cy),color,thickness)
        cv2.line(frame,(cx,cy),(cx,cy+dy*length),color,thickness)

def draw_label(frame, text, x, y, color):
    s,t=0.40,1
    (tw,th),_=cv2.getTextSize(text,FONT,s,t)
    cv2.rectangle(frame,(x,y-th-5),(x+tw+8,y+2),color,-1)
    cv2.putText(frame,text,(x+4,y-1),FONT,s,C_DARK,t,cv2.LINE_AA)

def draw_panel_bg(frame, x, y, w, h, alpha=0.78):
    ov=frame.copy()
    cv2.rectangle(ov,(x,y),(x+w,y+h),C_PANEL,-1)
    cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
    cv2.rectangle(frame,(x,y),(x+w,y+h),C_DIM,1)

def draw_bar(frame, x, y, w, h, pct, color):
    cv2.rectangle(frame,(x,y),(x+w,y+h),C_DIM,1)
    fill=int(w*max(0.0,min(1.0,pct)))
    if fill>0: cv2.rectangle(frame,(x+1,y+1),(x+fill,y+h-1),color,-1)

def draw_grid(frame):
    fh,fw=frame.shape[:2]; ov=frame.copy()
    for x in range(0,fw,60): cv2.line(ov,(x,0),(x,fh),C_BLUE,1)
    for y in range(0,fh,60): cv2.line(ov,(0,y),(fw,y),C_BLUE,1)
    cv2.addWeighted(ov,0.03,frame,0.97,0,frame)

def draw_scanline(frame, fi):
    fh,fw=frame.shape[:2]; y=int((fi*2)%fh)
    ov=frame.copy()
    cv2.rectangle(ov,(0,y),(fw,y+3),C_BLUE,-1)
    cv2.addWeighted(ov,0.04,frame,0.96,0,frame)

def draw_topbar(frame, fps, fi, has_anomaly):
    fh,fw=frame.shape[:2]
    draw_panel_bg(frame,0,0,fw,24,alpha=0.90)
    ts=datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame,
        f"ctOS // ANOMALY DETECTION  |  {ts}  |  FRAME:{fi:05d}  |  FPS:{fps:.1f}",
        (8,16),FONT_MON,0.9,C_BLUE,1,cv2.LINE_AA)
    sc=C_RED if has_anomaly else C_GREEN
    label="!! ANOMALY DETECTED" if has_anomaly else "SYSTEM SECURE"
    cv2.putText(frame,label,(fw-200,16),FONT_MON,0.9,sc,1,cv2.LINE_AA)
    if int(time.time()*2)%2==0:
        cv2.circle(frame,(fw-210,12),4,C_RED,-1)

def draw_bottombar(frame):
    fh,fw=frame.shape[:2]
    draw_panel_bg(frame,0,fh-18,fw,18,alpha=0.90)
    cv2.putText(frame,
        "NORMAL: PERSON (GREEN)  |  ANOMALY: KNIFE/WEAPON (RED)  |  GPU: ORIN  |  CAM-07 // SECTOR-B",
        (8,fh-5),FONT_MON,0.75,C_DIM,1,cv2.LINE_AA)

def draw_alert_banner(frame, v, t):
    if pulse(t,speed=5)<0.35: return
    fh,fw=frame.shape[:2]; bh=36; by=fh//2-bh//2
    color=C_RED if v["severity"]=="critical" else C_AMBER
    ov=frame.copy()
    cv2.rectangle(ov,(0,by),(fw,by+bh),color,-1)
    cv2.addWeighted(ov,0.28,frame,0.72,0,frame)
    cv2.rectangle(frame,(0,by),(fw,by+bh),color,2)
    prefix="!! CRITICAL ALERT" if v["severity"]=="critical" else "!! WARNING"
    msg=f"{prefix}: {v['detail']}"
    (tw,_),_=cv2.getTextSize(msg,FONT,0.6,1)
    cv2.putText(frame,msg,((fw-tw)//2,by+23),FONT,0.6,color,1,cv2.LINE_AA)

def draw_detection(frame, d, t):
    x1,y1,x2,y2=d["box"]
    is_wpn=any(f in d["label"].lower() for f in RULES["forbidden_classes"])
    if is_wpn:
        color=C_RED
        ov=frame.copy()
        cv2.rectangle(ov,(x1,y1),(x2,y2),C_RED,-1)
        cv2.addWeighted(ov,0.15*pulse(t,6),frame,1-0.15*pulse(t,6),0,frame)
    else:
        color=C_GREEN
    draw_corner_box(frame,x1,y1,x2,y2,color)
    tag="KNIFE!!" if is_wpn else d["label"].upper()
    draw_label(frame,f"{tag} {int(d['conf']*100)}%",x1,y1-2,color)
    cv2.putText(frame,d["id_str"],(x1+3,y2-4),FONT_MON,0.75,color,1,cv2.LINE_AA)
    if d["label"].lower()=="person" and not is_wpn:
        mx=(x1+x2)//2
        cv2.circle(frame,(mx,y1+10),5,C_GREEN,1)
        cv2.putText(frame,"ID:VERIFIED",(mx-26,y1+22),FONT_MON,0.7,C_GREEN,1,cv2.LINE_AA)

# ════════════════════════════════════════════
#   MJPEG HTTP SERVER
# ════════════════════════════════════════════
async def mjpeg_handler(request):
    """Serve MJPEG stream at http://localhost:8080/stream"""
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*',
        }
    )
    await response.prepare(request)
    while True:
        if latest_frame is not None:
            try:
                await response.write(
                    b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                    latest_frame + b'\r\n'
                )
            except Exception:
                break
        await asyncio.sleep(0.033)
    return response

async def start_mjpeg_server():
    app = web.Application()
    app.router.add_get('/stream', mjpeg_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    print("[MJPEG] Video stream at http://localhost:8080/stream")

# ════════════════════════════════════════════
#   WEBSOCKET
# ════════════════════════════════════════════
async def ws_handler(websocket, _path=None):
    ws_clients.add(websocket)
    print(f"[WS] Client connected ({len(ws_clients)})")
    try:
        await websocket.wait_closed()
    finally:
        ws_clients.discard(websocket)

async def broadcast(payload):
    if not ws_clients: return
    msg=json.dumps(payload)
    await asyncio.gather(*(ws.send(msg) for ws in list(ws_clients)),return_exceptions=True)

# ════════════════════════════════════════════
#   MAIN DETECTION LOOP
# ════════════════════════════════════════════
async def run_detection(args):
    global latest_frame
    setup_output()

    print(f"\n{'='*55}")
    print("  ctOS ANOMALY DETECTION — ONLINE")
    print(f"  Forbidden : {RULES['forbidden_classes']}")
    print(f"  Max persons: {RULES['max_persons']}")
    print(f"  Video stream: http://localhost:8080/stream")
    print(f"  WebSocket   : ws://localhost:8765")
    print(f"{'='*55}\n")

    model=YOLO("yolov8n.pt",task="detect")
    print("Model loaded OK\n")

    src=int(args.camera) if str(args.camera).isdigit() else args.camera
    cap=cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_FPS,30)
    if not cap.isOpened():
        print("ERROR: Cannot open camera."); return

    frame_idx=0; fps=0.0; fps_t=time.time(); fps_frames=0
    alert_timer=0.0; active_viol=None

    while True:
        cap.grab()
        ret,frame=cap.retrieve()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue

        fh,fw=frame.shape[:2]
        t=time.time()

        # Inference
        try:
            results=model.track(frame,persist=True,conf=RULES["min_confidence"],
                imgsz=RULES["img_size"],verbose=False,device="cuda:0")
        except:
            results=model.track(frame,persist=True,conf=RULES["min_confidence"],
                imgsz=RULES["img_size"],verbose=False,device="cpu")

        # Parse
        dets=[]
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy    =box.xyxy[0].cpu().numpy().astype(int)
                conf_val=float(box.conf[0])
                cls     =int(box.cls[0])
                label   =model.names[cls]
                tid     =int(box.id[0]) if box.id is not None else -1
                x1,y1,x2,y2=xyxy
                if tid not in object_ids: object_ids[tid]=make_id()
                dets.append(dict(label=label,conf=conf_val,box=xyxy,
                    cx=(x1+x2)/(2*fw),cy=(y1+y2)/(2*fh),id_str=object_ids[tid]))

        # Rules
        violations=evaluate_rules(dets,frame_idx)
        for v in violations:
            log_anomaly(frame_idx,v,dets,frame=frame.copy())
            active_viol=v; alert_timer=t+3.5

        # Draw Watch Dogs overlay
        draw_grid(frame)
        draw_scanline(frame,frame_idx)
        for d in dets: draw_detection(frame,d,t)
        if t<alert_timer and active_viol:
            draw_alert_banner(frame,active_viol,t)

        weapons=sum(1 for d in dets if any(f in d["label"].lower() for f in RULES["forbidden_classes"]))
        persons=sum(1 for d in dets if d["label"].lower()=="person")
        threat_pct=min(weapons*0.5+len(violations)*0.2+persons*0.05,1.0)

        draw_topbar(frame,fps,frame_idx,bool(violations))
        draw_bottombar(frame)

        # ── Encode frame as JPEG for MJPEG stream ────────────
        ret_enc, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret_enc:
            latest_frame = jpeg.tobytes()

        # Stats
        stats["total_frames"]=frame_idx
        stats["persons"]=persons
        stats["objects"]=len(dets)

        # WebSocket broadcast
        if WS_AVAILABLE:
            await broadcast({
                "detections":[{"label":d["label"],"conf":round(d["conf"],3),"id":d["id_str"]} for d in dets],
                "violations":[{"rule":v["rule"],"severity":v["severity"],"detail":v["detail"]} for v in violations],
                "frame":frame_idx,"fps":round(fps,1),"threat_pct":round(threat_pct,3),
                "stats":{
                    "total_frames":stats["total_frames"],
                    "total_anomalies":stats["total_anomalies"],
                    "last_anomaly":stats["last_anomaly"],
                    "since_anomaly":secs_since(stats["last_anomaly_ts"]),
                    "persons":persons,"objects":len(dets),
                    "uptime":round(time.time()-stats["start_time"],0),
                }
            })

        fps_frames+=1
        if time.time()-fps_t>=1.0:
            fps=fps_frames/(time.time()-fps_t); fps_frames=0; fps_t=time.time()

        frame_idx+=1
        await asyncio.sleep(0)

    cap.release()
    print(f"\n[ctOS] Done. Anomalies: {stats['total_anomalies']}")

# ════════════════════════════════════════════
#   ENTRY POINT
# ════════════════════════════════════════════
async def main(args):
    tasks=[run_detection(args)]
    if WS_AVAILABLE:
        tasks.append(websockets.serve(ws_handler,"0.0.0.0",args.ws_port))
        print(f"[WS] WebSocket on ws://0.0.0.0:{args.ws_port}")
    if HTTP_AVAILABLE:
        tasks.append(start_mjpeg_server())
    await asyncio.gather(*tasks)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--camera",  default="0")
    parser.add_argument("--ws-port", type=int,default=8765)
    parser.add_argument("--threshold",type=float,default=0.45)
    args=parser.parse_args()
    RULES["min_confidence"]=args.threshold
    asyncio.run(main(args))
