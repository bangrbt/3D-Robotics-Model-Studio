"""White, restrained camera/robot visuals; dimensions follow the desktop layout."""
from functools import lru_cache
from pathlib import Path
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import ARM_INDICES, Observation, robot_points

INK = "#17263F"
MUTED = "#748196"
BLUE = "#285CE6"
BORDER = "#E8EDF4"


@lru_cache(maxsize=32)
def font(size, bold=False):
    candidates = [Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts"/("segoeuib.ttf" if bold else "segoeui.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def empty_camera(size, message="Camera chưa bật", detail="Bấm Bật camera để bắt đầu"):
    w, h = size
    image = Image.new("RGB", size, "#F8FAFD")
    d = ImageDraw.Draw(image)
    cx, cy = w//2, h//2-32
    d.rounded_rectangle((cx-32,cy-24,cx+22,cy+20), radius=10, outline="#A9B7CB", width=3)
    d.polygon(((cx+24,cy-10),(cx+40,cy-20),(cx+40,cy+16),(cx+24,cy+7)), outline="#A9B7CB", width=3)
    d.text((w/2,cy+49), message, fill=INK, font=font(20,True), anchor="mt")
    d.text((w/2,cy+83), detail, fill=MUTED, font=font(13), anchor="mt")
    return image


def camera_image(frame, observation, state, size, mirror=True, skeleton=True, demo=False):
    if frame is None:
        return empty_camera(size)
    w,h = size
    source = cv2.flip(frame,1) if mirror else frame
    sh,sw = source.shape[:2]
    scale = min(w/sw,h/sh)
    width,height = max(1,round(sw*scale)),max(1,round(sh*scale))
    ox,oy = (w-width)//2,(h-height)//2
    image = Image.new("RGB",size,"#F8FAFD")
    image.paste(Image.fromarray(cv2.cvtColor(cv2.resize(source,(width,height)),cv2.COLOR_BGR2RGB)),(ox,oy))
    d = ImageDraw.Draw(image)
    if skeleton and observation.landmarks:
        def point(index):
            p=observation.landmarks[index]
            return (ox+(1-p.x if mirror else p.x)*width,oy+p.y*height)
        for side,indices in ARM_INDICES.items():
            if all((observation.landmarks[i].visibility or 0)>=.65 and (observation.landmarks[i].presence or 0)>=.65 for i in indices):
                points=[point(i) for i in indices]
                active=state.valid and state.side==side
                color=BLUE if active else "#A8B6C9"
                d.line(points,fill="white",width=9)
                d.line(points,fill=color,width=5)
                for x,y in points:
                    d.ellipse((x-6,y-6,x+6,y+6),fill="white",outline=color,width=3)
                if active and state.angle is not None:
                    x,y=points[1]
                    choices=[(x+14,y-35),(x+14,y+13),(x-90,y-35),(x-90,y+13)]
                    nose=point(0)
                    x,y=max(choices,key=lambda p:(p[0]+38-nose[0])**2+(p[1]+14-nose[1])**2)
                    x=float(np.clip(x,6,max(6,w-88)))
                    y=float(np.clip(y,5,max(5,h-38)))
                    d.rounded_rectangle((x,y,x+76,y+29),radius=8,fill=BLUE)
                    d.text((x+38,y+4),f"{state.angle:.1f}°",fill="white",font=font(15,True),anchor="mt")
    if demo:
        d.rounded_rectangle((14,14,142,43),radius=8,fill="white",outline=BORDER)
        d.text((78,20),"MINH HỌA",font=font(12,True),fill=BLUE,anchor="mt")
    return image


def robot_image(state, size):
    w,h = size
    if w<280 or h<200:
        return robot_image(state,(max(280,w),max(200,h))).resize(size,Image.Resampling.LANCZOS)
    # Supersampling keeps robot links and lettering sharp at every window size.
    factor = 2
    image = Image.new("RGB",(w*factor,h*factor),"white")
    d = ImageDraw.Draw(image)
    def xy(point):
        return tuple(float(v)*factor for v in point)
    def line(points,fill,width):
        d.line([xy(p) for p in points],fill=fill,width=round(width*factor),joint="curve")
    def circle(point,radius,fill,outline=None,width=1):
        x,y=xy(point)
        r=radius*factor
        d.ellipse((x-r,y-r,x+r,y+r),fill=fill,outline=outline,width=round(width*factor))
    def rect(bounds,fill,radius=8,outline=None):
        d.rounded_rectangle(tuple(v*factor for v in bounds),radius=radius*factor,fill=fill,outline=outline,width=2)
    def text(point,value,color=MUTED,size=12,bold=False,anchor=None):
        d.text(xy(point),value,fill=color,font=font(size*factor,bold),anchor=anchor)
    origin = np.array((w*.5,h*.46))
    length = min(w,h)*.215
    for radius in (length,length*1.9):
        x,y=xy(origin)
        r=radius*factor
        d.ellipse((x-r,y-r,x+r,y+r),outline="#EEF2F7",width=2)
    for x in np.arange(22,w,28):
        for y in np.arange(24,h-10,28):
            circle((x,y),.65,"#E9EEF5")
    base_y = h-32
    # Slim mechanical mounting column and base behind the moving arm.
    rect((origin[0]-15,origin[1]+10,origin[0]+15,base_y-10),"#E4EAF2",radius=7)
    rect((origin[0]-58,base_y-17,origin[0]+58,base_y+2),"#263954",radius=8)
    line(((origin[0]-40,base_y-7),(origin[0]+40,base_y-7)),"#526B8C",2)
    points = robot_points(state.shoulder_heading,state.robot_elbow,state.bend,origin,(length,length*.94))
    color = BLUE if state.valid else "#B5C2D4"
    line(points,"#18325D" if state.valid else "#A8B7CA",19)
    line(points,color,13)
    line((points[0],points[1]),"#7AA1FF" if state.valid else "#D4DDE8",5)
    for index,p in enumerate(points[:2]):
        circle(p,15 if index==0 else 12,"white",color,4)
        circle(p,4,"#1E3861" if state.valid else "#A8B7CA")
    wrist=points[2]
    direction=wrist-points[1]
    direction/=np.linalg.norm(direction)
    normal=np.array((-direction[1],direction[0]))
    tip=wrist+direction*14
    line((wrist,tip),color,7)
    for sign in (-1,1):
        p=tip+normal*sign*8
        line((tip,p,p+direction*13),"#243B5D" if state.valid else "#A8B7CA",4)
    elbow=points[1]
    a=math.degrees(math.atan2(*(points[0]-elbow)[::-1]))
    b=math.degrees(math.atan2(*(points[2]-elbow)[::-1]))
    if (b-a)%360>180:
        a,b=b,a
    ex,ey=xy(elbow)
    radius=27*factor
    d.arc((ex-radius,ey-radius,ex+radius,ey+radius),a,a+(b-a)%360,fill="#80A4FC",width=3)
    label=elbow+np.array((24,-37))
    label=np.clip(label,(10,12),(max(10,w-83),max(12,h-32)))
    text(label,f"θ {state.robot_elbow:.1f}°",color=INK if state.valid else MUTED,size=14,bold=True)
    text((18,h-23),"2 KHÂU  /  2 KHỚP",size=10,bold=True)
    text((w-18,h-23),"THEO GÓC KHUỶU" if state.valid else "ĐANG CHỜ TƯ THẾ",size=10,anchor="rt")
    return image.resize(size,Image.Resampling.LANCZOS)


def demo_observation(now):
    from types import SimpleNamespace
    points=[SimpleNamespace(x=.5,y=.5,z=0,visibility=.99,presence=.99) for _ in range(33)]
    values={0:(.5,.17),11:(.64,.38),12:(.36,.38),13:(.74,.58),15:(.77,.79),
            23:(.59,.72),24:(.41,.72),25:(.6,.87),26:(.4,.87),27:(.6,.98),28:(.4,.98)}
    phase=now%12
    if phase<2:
        values.update({14:(.25,.51),16:(.23,.2)})
    elif phase<4 or phase>=10:
        values.update({14:(.27,.58),16:(.23,.82)})
    elif phase<5:
        values.update({14:(.2,.47),16:(.09,.47)})
    else:
        values.update({14:(.23,.48),16:(.23+.14*math.sin((phase-5)*2*math.pi*1.5),.19)})
    for i,(x,y) in values.items():
        points[i].x,points[i].y=x,y
    return Observation(points,[],640,480)


def demo_frame(observation):
    image=Image.new("RGB",(640,480),"#F5F8FD")
    d=ImageDraw.Draw(image)
    def p(i):
        v=observation.landmarks[i]
        return (v.x*640,v.y*480)
    d.ellipse((264,30,376,146),fill="#E6EDFA",outline="#CBD7EB",width=2)
    d.polygon([p(11),p(12),p(24),p(23)],fill="#DCE7FA",outline="#C9D8F2",width=2)
    for chain in ((11,13,15),(12,14,16),(23,25,27),(24,26,28)):
        d.line([p(i) for i in chain],fill="#B7C9E6",width=17,joint="curve")
        for i in chain:
            x,y=p(i)
            d.ellipse((x-8,y-8,x+8,y+8),fill="#C4D5ED")
    return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
