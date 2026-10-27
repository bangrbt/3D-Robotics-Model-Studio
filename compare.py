"""Controlled DIGITAL perturbations of the same labeled image/video.

These tests do not claim to reproduce physical lighting or camera distances.
Only genuine video frames are eligible for temporal command evaluation.
"""
from pathlib import Path
import argparse
import csv
import json
import time

import cv2
import numpy as np

from experiments import CONDITIONS, TrialMetrics
from gestures import GESTURES, GateState, StableGestureGate
from vision import MODEL, ROOT, Observation, create_recognizer, ensure_model, recognize


def transform(frame, condition, reference_landmarks=()):
    h,w=frame.shape[:2]
    if condition=="normal":
        return frame.copy()
    if condition=="low_light":
        return (frame.astype(np.float32)*0.25).astype(np.uint8)
    if condition=="occluded":
        output=frame.copy()
        if reference_landmarks:
            xs=[p.x*w for p in reference_landmarks]
            ys=[p.y*h for p in reference_landmarks]
            x0,x1=max(0,int(min(xs))-8),min(w,int(max(xs))+8)
            y0,y1=max(0,int(min(ys))-8),min(h,int(max(ys))+8)
            cv2.rectangle(output,(x0,y0),(x1,int(y0+0.4*(y1-y0))),(25,25,25),-1)
        return output
    # Scale image size only; names are categories, NOT measured centimeters.
    scale={"near":1.25,"medium":0.8,"far":0.45}[condition]
    resized=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))))
    if scale>1:
        dx,dy=(resized.shape[1]-w)//2,(resized.shape[0]-h)//2
        return resized[dy:dy+h,dx:dx+w].copy()
    output=np.zeros_like(frame)
    sh,sw=resized.shape[:2]
    dx,dy=(w-sw)//2,(h-sh)//2
    output[dy:dy+sh,dx:dx+sw]=resized
    return output


def evaluate(frames,fps,expected,model,confidence=0.65,is_image=False,preview_dir=None):
    reference=[]
    with create_recognizer(model) as recognizer:
        for i,frame in enumerate(frames):
            reference.append(recognize(recognizer,frame,max(i,int(i*1000/fps))))
    rows=[]
    for condition in CONDITIONS:
        metrics=TrialMetrics(expected)
        gate=StableGestureGate(confidence=confidence)
        started=time.perf_counter()
        with create_recognizer(model) as recognizer:
            for i,frame in enumerate(frames):
                changed=transform(frame,condition,reference[i].landmarks)
                observation=recognize(recognizer,changed,max(i,int(i*1000/fps)))
                if is_image:
                    # No frame duplication: a photo cannot establish stability.
                    valid=observation.gesture in GESTURES and observation.confidence>=confidence
                    state=GateState(observation.gesture if valid else None,1 if valid else 0,5,False)
                else:
                    state=gate.update(observation.gesture,observation.confidence,i/fps,observation.handedness)
                metrics.observe(observation,state,i/fps)
                if preview_dir is not None and i==0:
                    preview_dir.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(preview_dir/f"{condition}.jpg"),changed)
        elapsed=time.perf_counter()-started
        row=metrics.row(condition,len(frames)/fps,"digital_image_simulation" if is_image else "digital_video_simulation")
        row["temporal_evaluation"]=not is_image
        row["predictions_json"]=json.dumps(metrics.confusion,sort_keys=True)
        row["processing_fps"]=round(len(frames)/max(0.001,elapsed),2)
        row["note"]={"normal":"original input", "low_light":"pixel intensity x0.25; not a physical low-light capture",
                     "occluded":"mask upper 40% of baseline hand bounding box, if detected",
                     "near":"digital scale x1.25; not measured distance",
                     "medium":"digital scale x0.80; not measured distance",
                     "far":"digital scale x0.45; not measured distance"}[condition]
        if is_image:
            for key in ("correct_events","wrong_events","success","first_correct_seconds","fps","duration_seconds"):
                row[key]=""
        rows.append(row)
        print(f"{condition:12s} frame_correct={row['frame_accuracy_percent']:6.2f}% hand={row['detection_percent']:6.2f}% command={'not tested (photo)' if is_image else row['correct_events']}")
    return rows


def main():
    parser=argparse.ArgumentParser(description="Compare DIGITAL light/occlusion/scale perturbations; no physical measurements")
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image",type=Path)
    source.add_argument("--video",type=Path,help="Clip holding exactly the --expected gesture throughout")
    parser.add_argument("--expected",choices=list(GESTURES),required=True)
    parser.add_argument("--limit",type=int,default=180,help="Video frame limit to control memory use")
    parser.add_argument("--confidence",type=float,default=0.65)
    parser.add_argument("--model",type=Path,default=MODEL)
    parser.add_argument("--output",type=Path,default=ROOT/"results"/"simulation.csv")
    args=parser.parse_args()
    StableGestureGate(confidence=args.confidence)
    if not 5<=args.limit<=600:
        parser.error("--limit phai tu 5 den 600")
    frames=[]
    fps=30.0
    if args.image:
        frame=cv2.imread(str(args.image))
        if frame is None:
            raise RuntimeError(f"Khong doc duoc anh {args.image}")
        frames=[cv2.flip(frame,1)]
    else:
        cap=cv2.VideoCapture(str(args.video))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Khong mo duoc video {args.video}")
            value=cap.get(cv2.CAP_PROP_FPS)
            fps=value if 0<value<240 else fps
            for _ in range(args.limit):
                ok,frame=cap.read()
                if not ok:
                    break
                h,w=frame.shape[:2]
                if w>960:
                    frame=cv2.resize(frame,(960,round(h*960/w)))
                frames.append(cv2.flip(frame,1))
        finally:
            cap.release()
        if len(frames)<5:
            raise RuntimeError("Can video co it nhat 5 khung hinh thuc.")
    rows=evaluate(frames,fps,args.expected,ensure_model(args.model),args.confidence,bool(args.image),
                  args.output.parent/(args.output.stem+"_inputs"))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Saved: {args.output}; DIGITAL simulations only, not real webcam condition trials.")


if __name__=="__main__":
    main()
