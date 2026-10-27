"""Low-latency drawing state, pen gating and continuous robot pen following."""
from dataclasses import dataclass,field
from collections import deque
import math

import numpy as np

from .kinematics import map_pointer,solve
from .shapes import Shape,recognize_recent


@dataclass
class HandObservation:
    landmarks: list=field(default_factory=list)
    world_landmarks: list=field(default_factory=list)
    handedness: str | None=None
    width: int=640
    height: int=480


@dataclass
class Stroke:
    points: list=field(default_factory=list)
    color: str="#285CE6"
    width: int=3


class OneEuro:
    def __init__(self):
        self.value=None
        self.derivative=np.zeros(2)
        self.raw=None
        self.time=None

    def update(self,value,now):
        value=np.asarray(value,dtype=float)
        if self.value is None or now<=self.time or now-self.time>.3:
            self.value,self.raw,self.time=value.copy(),value.copy(),now
            self.derivative=np.zeros(2)
            return self.value.copy()
        dt=now-self.time
        velocity=(value-self.raw)/dt
        alpha_d=1/(1+1/(2*math.pi*dt))
        self.derivative+=alpha_d*(velocity-self.derivative)
        cutoff=2.8+5*np.linalg.norm(self.derivative)
        alpha=1/(1+1/(2*math.pi*cutoff*dt))
        self.value+=alpha*(value-self.value)
        self.raw,self.time=value.copy(),now
        return self.value.copy()


def pen_gesture(observation,mode="pinch",was_pressed=False):
    if len(observation.landmarks)<21:
        return None,"Không thấy bàn tay"
    p=np.array([(v.x,v.y,v.z) for v in observation.landmarks],dtype=float)
    if not np.isfinite(p).all() or not (0<=p[8,0]<=1 and 0<=p[8,1]<=1):
        return None,"Điểm đầu ngón tay không hợp lệ"
    palm=np.linalg.norm((p[5,:2]-p[17,:2])*np.array((observation.width,observation.height)))
    if palm<14:
        return None,"Đưa bàn tay gần camera hơn"
    if len(observation.world_landmarks)==21:
        geometry=np.array([(v.x,v.y,v.z) for v in observation.world_landmarks],dtype=float)
        if not np.isfinite(geometry).all():
            return None,"Điểm mốc không ổn định"
    else:
        geometry=p*np.array((observation.width,observation.height,observation.width))
    scale=np.linalg.norm(geometry[5]-geometry[17])
    if scale<1e-8:
        return None,"Không đủ điểm mốc lòng bàn tay"
    if mode=="pinch":
        ratio=np.linalg.norm(geometry[4]-geometry[8])/scale
        return bool(ratio<(.48 if was_pressed else .32)),""
    def extended(mcp,pip,tip):
        a,b=geometry[mcp]-geometry[pip],geometry[tip]-geometry[pip]
        denominator=np.linalg.norm(a)*np.linalg.norm(b)
        return denominator>1e-9 and np.dot(a,b)/denominator<-.82
    return bool(extended(5,6,8) and not extended(9,10,12)),""


class DrawingEngine:
    def __init__(self,required=5):
        if required<5:
            raise ValueError("At least five pen-down frames are required")
        self.required=required
        self.strokes=[]
        self.current=None
        self.version=0
        self.completed_version=0
        self.color="#285CE6"
        self.width=3
        self.shape=Shape("none","Chưa có hình")
        self.robot=solve((.5,.7))
        self.reset_tracking()

    def reset_tracking(self):
        self.lift()
        self.filter=OneEuro()
        self.previous_raw=None
        self.previous_time=None
        self.identity=None
        self.pressed=False
        self.valid=False
        self.status="Chờ bàn tay"
        self.pointer=None
        self.blocked=False

    def lift(self):
        if getattr(self,"current",None) is not None:
            self.strokes.append(self.current)
            self.current=None
            self.completed_version+=1
            self.version+=1
            self.shape=recognize_recent(self.strokes)
        self.pen_down=False
        self.frames=0
        self.pending=deque(maxlen=getattr(self,"required",5))

    def update(self,observation,now,mirror=True,mode="pinch"):
        pressed,reason=pen_gesture(observation,mode,self.pressed)
        if pressed is None:
            self.update_pointer(None,False,now)
            self.status=reason
            return
        point=observation.landmarks[8]
        uv=map_pointer((point.x,point.y),observation.width,observation.height,mirror)
        self.update_pointer(uv,pressed,now,observation.handedness)
        if uv is None:
            self.status="Đưa đầu ngón trỏ vào vùng vẽ"

    def update_pointer(self,uv,pressed,now,identity="manual"):
        if uv is None or not np.isfinite(uv).all():
            self.lift()
            self.valid=self.pressed=False
            self.pointer=None
            self.previous_raw=None
            self.filter=OneEuro()
            self.blocked=False
            self.status="Chờ bàn tay"
            return
        raw=np.asarray(uv,dtype=float)
        if raw.shape!=(2,) or (raw<0).any() or (raw>1).any():
            self.update_pointer(None,False,now)
            return
        gap=self.previous_time is not None and (now<=self.previous_time or now-self.previous_time>.3)
        switched=self.identity is not None and identity is not None and identity!=self.identity
        jump=self.previous_raw is not None and np.linalg.norm(raw-self.previous_raw)>.28
        if gap or switched or jump:
            self.lift()
            self.filter=OneEuro()
        self.previous_raw,self.previous_time,self.identity=raw.copy(),now,identity
        self.pointer=np.clip(self.filter.update(raw,now),0,1)
        self.robot=solve(self.pointer)
        self.valid=True
        self.pressed=bool(pressed)
        if not pressed:
            self.lift()
            self.blocked=False
            self.status="Nhấc bút · Di chuyển tự do"
            return
        if self.blocked:
            self.status="Thả tay để bắt đầu nét mới"
            return
        if not self.pen_down:
            self.frames=min(self.required,self.frames+1)
            self.pending.append(tuple(self.robot.uv))
            self.status=f"Hạ bút · {self.frames}/{self.required} frame"
            if self.frames<self.required:
                return
            self.current=Stroke(list(self.pending),self.color,self.width)
            self.pen_down=True
            self.version+=1
        else:
            tip=np.asarray(self.robot.uv)
            if np.linalg.norm(tip-np.asarray(self.current.points[-1]))>.0008:
                self.current.points.append(tuple(self.robot.uv))
                self.version+=1
        self.status="Đang vẽ · Robot theo nét trực tiếp"
        # Bound memory without connecting an old stroke to a new source.
        if self.current is not None and len(self.current.points)>=12000:
            self.lift()
            self.blocked=True
            self.status="Nét dài đã lưu · Thả tay để vẽ tiếp"

    def clear(self):
        self.current=None
        self.strokes.clear()
        self.lift()
        self.shape=Shape("none","Chưa có hình")
        self.version+=1
        self.completed_version+=1
        self.blocked=self.pressed

    def undo(self):
        if self.current is not None:
            self.current=None
        elif self.strokes:
            self.strokes.pop()
        self.lift()
        self.version+=1
        self.completed_version+=1
        self.shape=recognize_recent(self.strokes) if self.strokes else Shape("none","Chưa có hình")
        self.blocked=self.pressed

    def change_style(self,color=None,width=None):
        self.lift()
        self.blocked=self.pressed
        if color is not None:
            self.color=color
        if width is not None:
            self.width=int(width)

    def all_strokes(self):
        return self.strokes+([self.current] if self.current else [])
