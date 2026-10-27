"""Exact 2-link inverse kinematics over an entirely reachable drawing sheet."""
from dataclasses import dataclass
import math

import numpy as np

PAPER_WIDTH=1.4
PAPER_HEIGHT=1.0
PAPER_LEFT=.20
PAPER_BOTTOM=.25
LINKS=(1.05,1.05)


@dataclass(frozen=True)
class RobotPose:
    shoulder: float
    elbow: float
    joints: np.ndarray
    uv: tuple[float,float]


def forward(shoulder,elbow,lengths=LINKS):
    l1,l2=lengths
    a=np.array((l1*math.cos(shoulder),l1*math.sin(shoulder)))
    b=a+np.array((l2*math.cos(shoulder+elbow),l2*math.sin(shoulder+elbow)))
    return np.array(((0.,0.),a,b))


def solve(uv):
    u,v=map(float,uv)
    if not np.isfinite((u,v)).all() or not (0<=u<=1 and 0<=v<=1):
        raise ValueError("Target must be inside the drawing sheet")
    x=PAPER_LEFT+u*PAPER_WIDTH
    y=PAPER_BOTTOM+(1-v)*PAPER_HEIGHT
    l1,l2=LINKS
    cosine=(x*x+y*y-l1*l1-l2*l2)/(2*l1*l2)
    if not -1-1e-9<=cosine<=1+1e-9:
        raise ValueError("Target is outside the robot workspace")
    elbow=-math.acos(float(np.clip(cosine,-1,1)))
    shoulder=math.atan2(y,x)-math.atan2(l2*math.sin(elbow),l1+l2*math.cos(elbow))
    joints=forward(shoulder,elbow)
    # Ink uses the solved end effector, rather than a separate hand trail.
    tip=joints[-1]
    actual=(float((tip[0]-PAPER_LEFT)/PAPER_WIDTH),float(1-(tip[1]-PAPER_BOTTOM)/PAPER_HEIGHT))
    return RobotPose(shoulder,elbow,joints,actual)


def drawing_roi(width,height):
    """A camera rectangle with the SAME physical aspect ratio as robot paper."""
    if width<=0 or height<=0:
        raise ValueError("Invalid frame dimensions")
    w=min(width*.72,height*.74*PAPER_WIDTH/PAPER_HEIGHT)
    h=w*PAPER_HEIGHT/PAPER_WIDTH
    return ((width-w)/2,(height-h)/2,w,h)


def map_pointer(point,width,height,mirror=True):
    x,y=map(float,point)
    if mirror:
        x=1-x
    rx,ry,rw,rh=drawing_roi(width,height)
    u,v=(x*width-rx)/rw,(y*height-ry)/rh
    if not np.isfinite((u,v)).all() or not (0<=u<=1 and 0<=v<=1):
        return None
    return float(u),float(v)
