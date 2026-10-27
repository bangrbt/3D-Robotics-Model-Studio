"""Reliable small sample queue + latest-only camera queue, separate from Tk."""
from dataclasses import dataclass
from queue import Empty,Full,Queue
import math
import threading
import time

import cv2
import numpy as np

from pose_robot.worker import open_camera
from .core import HandObservation
from .vision import create_detector,detect,ensure_model


@dataclass
class Sample:
    observation: HandObservation
    time: float
    captured_at: float
    manual_uv: tuple | None=None
    manual_pressed: bool=False
    discontinuity: bool=False


@dataclass
class Frame:
    image: object
    observation: HandObservation
    fps: float
    captured_at: float


def demo_pointer(now):
    """Synthetic pointer trajectory, explicitly displayed as a demo in the UI."""
    cycle=int(now/3.8)%4
    phase=now%3.8
    t=float(np.clip((phase-.25)/2.8,0,1))
    if cycle==0:
        uv=(.5+.24*math.cos(2*math.pi*t),.5+.24*1.4*math.sin(2*math.pi*t))
    else:
        vertices={1:((.25,.18),(.75,.18),(.75,.88),(.25,.88),(.25,.18)),
                  2:((.5,.16),(.78,.84),(.22,.84),(.5,.16)),
                  3:((.16,.28),(.84,.28),(.84,.72),(.16,.72),(.16,.28))}[cycle]
        position=t*(len(vertices)-1)
        index=min(int(position),len(vertices)-2)
        fraction=position-index
        uv=tuple((1-fraction)*np.array(vertices[index])+fraction*np.array(vertices[index+1]))
    return uv,.05<=phase<3.25


class HandWorker:
    def __init__(self,kind="camera",source=0):
        self.kind,self.source=kind,source
        self.samples=Queue(maxsize=90)
        self.frames=Queue(maxsize=1)
        self.messages=Queue()
        self.stop_event=threading.Event()
        self.pause_event=threading.Event()
        self.thread=threading.Thread(target=self._run,name="hand-tracking",daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def pause(self,value=True):
        self.pause_event.set() if value else self.pause_event.clear()

    def clear_samples(self):
        while True:
            try:
                self.samples.get_nowait()
            except Empty:
                return

    def latest(self):
        try:
            return self.frames.get_nowait()
        except Empty:
            return None

    def _publish(self,sample,frame):
        try:
            self.samples.put_nowait(sample)
        except Full:
            self.clear_samples()
            sample.discontinuity=True
            self.samples.put_nowait(sample)
        try:
            self.frames.put_nowait(frame)
        except Full:
            self.latest()
            self.frames.put_nowait(frame)

    def _run(self):
        cap=model=None
        try:
            if self.kind!="demo":
                self.messages.put(("status","Đang chuẩn bị mô hình bàn tay…"))
                path=ensure_model()
                if self.stop_event.is_set():
                    return
                model=create_detector(path)
                if self.kind=="camera":
                    self.messages.put(("status","Đang kết nối camera…"))
                    cap=open_camera(int(self.source))
                elif self.kind=="video":
                    cap=cv2.VideoCapture(str(self.source))
                    if not cap.isOpened():
                        raise RuntimeError("Không mở được video đã chọn.")
            fps=cap.get(cv2.CAP_PROP_FPS) if cap is not None else 30
            fps=fps if 1<=fps<=240 else 30
            self.messages.put(("ready",self.kind))
            index,last_ms,failures=0,-1,0
            previous=time.monotonic()
            smoothed_fps=0
            resumed=False
            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    resumed=True
                    self.stop_event.wait(.035)
                    continue
                started=time.monotonic()
                if self.kind=="demo":
                    frame=np.full((480,640,3),(250,248,245),np.uint8)
                    observation=HandObservation()
                    now=index/30
                    uv,pressed=demo_pointer(now)
                else:
                    if self.kind=="image":
                        frame=cv2.imread(str(self.source))
                        ok=frame is not None
                    else:
                        ok,frame=cap.read()
                    if not ok:
                        if self.kind=="video":
                            break
                        failures+=1
                        if failures>=8 or self.kind=="image":
                            raise RuntimeError("Không nhận được khung hình. Kiểm tra camera hoặc tệp đã chọn.")
                        self.stop_event.wait(.05)
                        continue
                    failures=0
                    now=index/fps if self.kind in ("video","image") else started
                    timestamp=max(last_ms+1,int(now*1000))
                    last_ms=timestamp
                    observation=detect(model,frame,timestamp)
                    uv,pressed=None,False
                done=time.monotonic()
                measured=1/max(done-previous,.001)
                smoothed_fps=measured if not index else .85*smoothed_fps+.15*measured
                previous=done
                self._publish(Sample(observation,now,done,uv,pressed,resumed),Frame(frame,observation,smoothed_fps,done))
                resumed=False
                index+=1
                if self.kind=="image":
                    break
                if self.kind in ("demo","video"):
                    self.stop_event.wait(max(0,1/(30 if self.kind=="demo" else fps)-(time.monotonic()-started)))
        except Exception as error:
            self.messages.put(("error",str(error)))
        finally:
            try:
                if cap is not None:
                    cap.release()
                if model is not None:
                    model.close()
            finally:
                self.messages.put(("ended",self.kind))
