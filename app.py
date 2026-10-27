"""Webcam gestures -> five-frame gate -> visible commands and 3D effects."""
from collections import deque
from datetime import datetime
from pathlib import Path
import argparse
import csv
import json
import sys
import time

import cv2

from experiments import CONDITIONS, ExperimentLab, append_csv, summarize
from gestures import GESTURES, StableGestureGate
from scene3d import Scene3D
from palm_ar import PalmAR
from ui import render
from vision import MODEL, ROOT, create_recognizer, ensure_model, open_camera, recognize

WINDOW="HRI - Hand Gesture Lab"


def export_comparison(lab):
    rows=summarize(lab.rows,lab.expected)
    path=lab.path.parent/"comparison.csv"
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=["condition","expected_gesture","trials","frame_accuracy","detection","success","wrong_events","latency"]
    with path.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for condition,metrics in rows.items():
            writer.writerow({"condition":condition,"expected_gesture":lab.expected,**metrics})
    lab.message="Đã lưu bảng so sánh vào results/comparison.csv"
    return path


def main():
    parser=argparse.ArgumentParser(description="Hand gestures, stable >=5 consecutive frames, commands and 3D")
    parser.add_argument("--camera",type=int,default=0)
    parser.add_argument("--model",type=Path,default=MODEL)
    parser.add_argument("--frames",type=int,default=5,help="Minimum 5 consecutive valid frames")
    parser.add_argument("--confidence",type=float,default=0.65)
    parser.add_argument("--download-model",action="store_true")
    parser.add_argument("--check-camera",action="store_true")
    parser.add_argument("--video",type=Path,help="Read each real frame of an existing video instead of webcam")
    parser.add_argument("--headless",action="store_true",help="Process --video without GUI")
    parser.add_argument("--preview",type=Path,help="Render ONE image; never count it as five frames or emit a command")
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts"/"gesture_preview.jpg")
    parser.add_argument("--results",type=Path,default=ROOT/"results")
    args=parser.parse_args()
    if args.headless and not args.video:
        parser.error("--headless can dung voi --video")
    gate=StableGestureGate(args.frames,args.confidence)
    lab=ExperimentLab(args.results/"trials.csv")
    scene=Scene3D()
    ar=PalmAR()
    if args.check_camera:
        cap=open_camera(args.camera)
        try:
            ok,frame=cap.read()
            if not ok:
                raise RuntimeError("Webcam khong tra ve khung hinh.")
            print(f"Camera {args.camera} OK: {frame.shape[1]}x{frame.shape[0]}")
        finally:
            cap.release()
        return
    model=ensure_model(args.model)
    if args.download_model:
        print(f"Model ready: {model}")
        return
    if args.preview:
        frame=cv2.imread(str(args.preview))
        if frame is None:
            raise RuntimeError(f"Khong doc duoc anh: {args.preview}")
        frame=cv2.flip(frame,1)
        with create_recognizer(model) as recognizer:
            observation=recognize(recognizer,frame,0)
        if observation.gesture in GESTURES:
            lab.expected=observation.gesture
        state=gate.update(observation.gesture,observation.confidence,0,observation.handedness)
        ar.update(observation.landmarks,state,1/30,observation.handedness,scene)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        canvas=render(frame,observation,state,scene,lab,0,source="ẢNH TĨNH · CHỈ 1 FRAME · CHƯA PHÁT LỆNH",ar=ar)
        if not cv2.imwrite(str(args.output),canvas):
            raise RuntimeError("Khong ghi duoc anh xem truoc.")
        print(json.dumps({"gesture":observation.gesture,"confidence":observation.confidence,"stable_frames":state.frames,"event":state.event,"output":str(args.output)},ensure_ascii=True))
        return
    cap=cv2.VideoCapture(str(args.video)) if args.video else open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video {args.video}")
    video_fps=cap.get(cv2.CAP_PROP_FPS)
    if not 0<video_fps<240:
        video_fps=30.0
    last_ms=-1
    frame_index=0
    last_command=None
    event_at=-999.0
    show_landmarks=False
    previous=time.monotonic()
    fps_times=deque(maxlen=30)
    last_wrist=None
    try:
        with create_recognizer(model) as recognizer:
            if not args.headless:
                cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW,1200,900)
                def click(event,x,y,flags,param):
                    if event==cv2.EVENT_LBUTTONDOWN and 36<=x<=818 and 741<=y<=890 and lab.active is None:
                        index=(y-741)//25
                        if 0<=index<len(CONDITIONS):
                            lab.condition=list(CONDITIONS)[index]
                cv2.setMouseCallback(WINDOW,click)
            while True:
                ok,frame=cap.read()
                if not ok:
                    if args.video:
                        break
                    raise RuntimeError("Mat ket noi webcam. Kiem tra camera va chay lai.")
                wall_now=time.monotonic()
                now=frame_index/video_fps if args.video else wall_now
                timestamp=max(last_ms+1,int(now*1000))
                last_ms=timestamp
                frame_index+=1
                frame=cv2.flip(frame,1)
                if lab.tick(now):
                    gate.reset()
                observation=recognize(recognizer,frame,timestamp)
                if observation.landmarks:
                    wrist=observation.landmarks[0]
                    position=(wrist.x,wrist.y)
                    if last_wrist and (position[0]-last_wrist[0])**2+(position[1]-last_wrist[1])**2>0.25**2:
                        gate.candidate,gate.count=None,0
                    last_wrist=position
                else:
                    last_wrist=None
                state=gate.update(observation.gesture,observation.confidence,now,observation.handedness)
                lab.observe(observation,state,now)
                if state.event:
                    scene.apply(state.event)
                    last_command=(state.event,GESTURES[state.gesture][2])
                    event_at=now
                    event={"timestamp":datetime.now().astimezone().isoformat(timespec="milliseconds"),
                           "source":"video" if args.video else "webcam","frame":frame_index,
                           "gesture":state.gesture,"command":state.event,
                           "confidence":round(observation.confidence,4),"stable_frames":state.frames}
                    append_csv(args.results/"commands.csv",event)
                    print(json.dumps(event,ensure_ascii=True),flush=True)
                dt=wall_now-previous
                previous=wall_now
                fps_times.append(max(dt,0.001))
                scene.update(1/video_fps if args.video else dt)
                old_bursts=ar.burst_count
                ar.update(observation.landmarks,state,1/video_fps if args.video else dt,observation.handedness,scene)
                if ar.burst_count>old_bursts:
                    last_command=("BURST","BUNG LẤP LÁNH")
                    event_at=now
                    effect={"timestamp":datetime.now().astimezone().isoformat(timespec="milliseconds"),
                            "source":"video" if args.video else "webcam","frame":frame_index,
                            "gesture":state.gesture,"effect":"BURST","stable_frames":state.frames}
                    append_csv(args.results/"effects.csv",effect)
                    print(json.dumps(effect,ensure_ascii=True),flush=True)
                if args.headless:
                    continue
                fps=len(fps_times)/sum(fps_times)
                canvas=render(frame,observation,state,scene,lab,now,fps,last_command,now-event_at,show_landmarks,
                              "VIDEO · ẢNH GƯƠNG" if args.video else "WEBCAM · ẢNH GƯƠNG",ar=ar)
                cv2.imshow(WINDOW,canvas)
                key=cv2.waitKey(max(1,int(1000/video_fps)) if args.video else 1)&0xFF
                if key in (27,ord("q"),ord("Q")) or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:
                    break
                if key in (ord("l"),ord("L")):
                    show_landmarks=not show_landmarks
                elif key in (ord("r"),ord("R")):
                    scene=Scene3D()
                    ar=PalmAR()
                    gate.reset()
                    last_command=None
                elif key in (ord("x"),ord("X")):
                    lab.cancel()
                    gate.reset()
                elif key in (ord("s"),ord("S")):
                    print(f"Saved: {export_comparison(lab)}")
                elif args.video is None and key in (ord("t"),ord("T")):
                    lab.start(now)
                elif lab.active is None and ord("1")<=key<=ord("6"):
                    lab.condition=list(CONDITIONS)[key-ord("1")]
                elif lab.active is None and key in (ord("g"),ord("G")):
                    names=list(GESTURES)
                    lab.expected=names[(names.index(lab.expected)+1)%len(names)]
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if lab.rows:
            export_comparison(lab)


if __name__=="__main__":
    try:
        main()
    except (RuntimeError,OSError,ValueError) as error:
        print(f"LOI: {error}",file=sys.stderr)
        sys.exit(1)
