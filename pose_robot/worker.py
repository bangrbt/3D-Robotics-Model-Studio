"""Camera and inference live off the UI thread; only the latest frame is queued."""
from dataclasses import dataclass
from queue import Empty, Full, Queue
import threading
import time
import sys

import cv2

from .core import Observation, PoseController
from .graphics import demo_frame, demo_observation
from .vision import create_detector, detect, ensure_model


@dataclass
class Packet:
    frame: object
    observation: Observation
    state: object
    fps: float
    captured_at: float
    source: str
    event: str | None = None


def open_camera(index):
    backends=(cv2.CAP_DSHOW,cv2.CAP_ANY) if sys.platform=="win32" else (cv2.CAP_ANY,)
    for backend in backends:
        cap=cv2.VideoCapture(index,backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
            cap.set(cv2.CAP_PROP_FPS,30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            return cap
        cap.release()
    raise RuntimeError(f"Không mở được camera {index}. Đóng Camera/Zoom/Teams, kiểm tra quyền Camera hoặc chọn thiết bị khác.")


class PoseWorker:
    def __init__(self,kind="camera",source=0,preference="auto",mirror=True,model=None):
        self.kind,self.source,self.model=kind,source,model
        self.frames=Queue(maxsize=1)
        self.messages=Queue()
        self.stop_event=threading.Event()
        self.pause_event=threading.Event()
        self.lock=threading.Lock()
        self.options=(preference,mirror)
        self.thread=threading.Thread(target=self._run,name="pose-inference",daemon=True)

    def start(self):
        self.thread.start()

    def configure(self,preference,mirror):
        with self.lock:
            self.options=(preference,mirror)

    def stop(self):
        self.stop_event.set()

    def pause(self,value=True):
        self.pause_event.set() if value else self.pause_event.clear()

    def latest(self):
        try:
            return self.frames.get_nowait()
        except Empty:
            return None

    def _publish(self,packet):
        # Events have their own reliable queue, even if a UI frame is dropped.
        if packet.event:
            self.messages.put(("action",(packet.state.side,packet.event)))
        try:
            self.frames.put_nowait(packet)
        except Full:
            try:
                self.frames.get_nowait()
            except Empty:
                pass
            self.frames.put_nowait(packet)

    def _run(self):
        cap=detector=None
        controller=PoseController()
        try:
            if self.kind!="demo":
                self.messages.put(("status","Đang chuẩn bị mô hình nhận dạng…"))
                model=ensure_model() if self.model is None else ensure_model(self.model)
                if self.stop_event.is_set():
                    return
                detector=create_detector(model)
            if self.kind=="camera":
                self.messages.put(("status","Đang kết nối camera…"))
                cap=open_camera(int(self.source))
            elif self.kind=="video":
                cap=cv2.VideoCapture(str(self.source))
                if not cap.isOpened():
                    raise RuntimeError("Không mở được video. Hãy chọn một tệp video hợp lệ.")
            fps=cap.get(cv2.CAP_PROP_FPS) if cap is not None else 30
            if not 1<=fps<=240:
                fps=30
            self.messages.put(("ready",self.kind))
            index,last_ms,failures=0,-1,0
            previous=time.monotonic()
            demo_time=0.0
            smoothed_fps=0.0
            was_paused=False
            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    was_paused=True
                    self.stop_event.wait(.035)
                    continue
                if was_paused:
                    controller.reset()
                    previous=time.monotonic()
                    was_paused=False
                started=time.monotonic()
                if self.kind=="demo":
                    observation=demo_observation(demo_time)
                    frame=demo_frame(observation)
                    now=demo_time
                    demo_time+=1/30
                elif self.kind=="image":
                    frame=cv2.imread(str(self.source))
                    if frame is None:
                        raise RuntimeError("Không đọc được ảnh đã chọn.")
                    now=0.0
                else:
                    ok,frame=cap.read()
                    if not ok:
                        if self.kind=="video":
                            break
                        failures+=1
                        self.messages.put(("status","Đang chờ camera trả về khung hình…"))
                        if failures>=8:
                            raise RuntimeError("Camera đã mở nhưng không nhận được hình. Kiểm tra kết nối hoặc chọn camera khác.")
                        self.stop_event.wait(.05)
                        continue
                    failures=0
                    now=index/fps if self.kind=="video" else time.monotonic()
                if self.kind!="demo":
                    timestamp=max(last_ms+1,int(now*1000))
                    last_ms=timestamp
                    observation=detect(detector,frame,timestamp)
                with self.lock:
                    preference,mirror=self.options
                state=controller.update(observation,now,preference,mirror)
                done=time.monotonic()
                instantaneous=1/max(done-previous,.001)
                smoothed_fps=instantaneous if not index else .85*smoothed_fps+.15*instantaneous
                previous=done
                self._publish(Packet(frame,observation,state,smoothed_fps,done,self.kind,state.event))
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
                if detector is not None:
                    detector.close()
            finally:
                self.messages.put(("ended",self.kind))
