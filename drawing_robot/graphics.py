"""White paper, genuine robot geometry and a matching camera drawing region."""
import math

import cv2
import numpy as np
from PIL import Image,ImageDraw

from pose_robot.graphics import BLUE,INK,MUTED,font
from .kinematics import PAPER_BOTTOM,PAPER_HEIGHT,PAPER_LEFT,PAPER_WIDTH,drawing_roi

HAND_CONNECTIONS=((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
                  (9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20))


def stroke_draw(draw,stroke,rect,factor=1):
    x,y,w,h=rect
    points=[(x+float(u)*w,y+float(v)*h) for u,v in stroke.points]
    if not points:
        return
    width=max(1,round(stroke.width*w/500))
    if len(points)>1:
        draw.line(points,fill=stroke.color,width=width,joint="curve")
    r=max(.75,width/2)
    for px,py in (points[0],points[-1]):
        draw.ellipse((px-r,py-r,px+r,py+r),fill=stroke.color)


class PaperRenderer:
    def __init__(self):
        self.cache_key=None
        self.cache=None
        self.rect=None
        self.transform=None

    def _layout(self,size,show_robot):
        w,h=size
        if show_robot:
            scale=min(max(1,w-30)/2.6,max(1,h-32)/1.65)
            offset=np.array(((w-2.6*scale)/2+.85*scale,(h-1.65*scale)/2+1.45*scale))
            def point(x,y):
                return offset+np.array((x*scale,-y*scale))
            top=point(PAPER_LEFT,PAPER_BOTTOM+PAPER_HEIGHT)
            rect=(top[0],top[1],PAPER_WIDTH*scale,PAPER_HEIGHT*scale)
            return rect,point
        scale=min(max(1,w-48)/PAPER_WIDTH,max(1,h-48)/PAPER_HEIGHT)
        rect=((w-PAPER_WIDTH*scale)/2,(h-PAPER_HEIGHT*scale)/2,PAPER_WIDTH*scale,PAPER_HEIGHT*scale)
        return rect,None

    def render(self,engine,size,show_robot=True,_high_resolution=False):
        if not _high_resolution:
            image=self.render(engine,(size[0]*2,size[1]*2),show_robot,True)
            return image.resize(size,Image.Resampling.LANCZOS)
        # Render committed strokes once; only the current stroke changes per frame.
        key=(id(engine),engine.completed_version,size,show_robot)
        if self.cache_key!=key:
            self.rect,self.transform=self._layout(size,show_robot)
            image=Image.new("RGB",size,"#F8FAFD")
            d=ImageDraw.Draw(image)
            x,y,w,h=self.rect
            d.rounded_rectangle((x-7,y-7,x+w+7,y+h+7),radius=9,fill="#EEF2F8")
            d.rectangle((x,y,x+w,y+h),fill="white",outline="#DFE6F1",width=1)
            for px in np.arange(x+16,x+w-5,22):
                for py in np.arange(y+16,y+h-5,22):
                    d.ellipse((px,py,px+1,py+1),fill="#E6EDF7")
            for stroke in engine.strokes:
                stroke_draw(d,stroke,self.rect)
            self.cache_key,self.cache=key,image
        image=self.cache.copy()
        d=ImageDraw.Draw(image)
        if engine.current:
            stroke_draw(d,engine.current,self.rect)
        x,y,w,h=self.rect
        if show_robot:
            transform=self.transform
            points=[transform(*p) for p in engine.robot.joints]
            base,elbow,tip=points
            color="#506FAD" if not engine.pen_down else BLUE
            # Links are narrow enough to keep the original ink visible.
            d.line([tuple(p) for p in points],fill="#DDE5F3",width=17,joint="curve")
            d.line([tuple(p) for p in points],fill=color,width=10,joint="curve")
            d.line([tuple(p) for p in points],fill="#AEC3EF",width=3,joint="curve")
            for p,r in ((base,14),(elbow,12)):
                px,py=p
                d.ellipse((px-r,py-r,px+r,py+r),fill="white",outline=color,width=3)
                d.ellipse((px-3,py-3,px+3,py+3),fill=color)
            bx,by=base
            d.rounded_rectangle((bx-29,by+16,bx+29,by+29),radius=5,fill="#263954")
            d.line(((bx,by+11),(bx,by+17)),fill="#263954",width=9)
        else:
            u,v=engine.robot.uv
            tip=np.array((x+u*w,y+v*h))
        px,py=tip
        color=engine.color if engine.pen_down else "#8FA5C9"
        d.ellipse((px-9,py-9,px+9,py+9),outline=color,width=2)
        d.ellipse((px-3,py-3,px+3,py+3),fill=color)
        if engine.pen_down:
            d.ellipse((px-14,py-14,px+14,py+14),outline="#C5D4FC",width=1)
        return image


def camera_image(packet,engine,size,mirror=True,skeleton=False,source="camera"):
    w,h=size
    if packet is None:
        from pose_robot.graphics import empty_camera
        return empty_camera(size,"Vẽ trong không trung","Bật camera hoặc chọn Vẽ bằng chuột để thử")
    original=packet.image
    # Demo/mouse frames contain no actual webcam image.
    show=cv2.flip(original,1) if mirror and source not in ("demo","mouse") else original
    sh,sw=show.shape[:2]
    scale=min(w/sw,h/sh)
    width,height=max(1,round(sw*scale)),max(1,round(sh*scale))
    ox,oy=(w-width)//2,(h-height)//2
    image=Image.new("RGB",size,"#F8FAFD")
    image.paste(Image.fromarray(cv2.cvtColor(cv2.resize(show,(width,height)),cv2.COLOR_BGR2RGB)),(ox,oy))
    d=ImageDraw.Draw(image)
    rx,ry,rw,rh=drawing_roi(sw,sh)
    rect=(ox+rx*scale,oy+ry*scale,rw*scale,rh*scale)
    x,y,pw,ph=rect
    # Registration corners communicate the drawing area without hiding the hand.
    corner=17
    for px,py,sx,sy in ((x,y,1,1),(x+pw,y,-1,1),(x,y+ph,1,-1),(x+pw,y+ph,-1,-1)):
        d.line(((px+sx*corner,py),(px,py),(px,py+sy*corner)),fill=BLUE,width=3)
    if source in ("demo","mouse"):
        for px in np.arange(x+20,x+pw,24):
            for py in np.arange(y+20,y+ph,24):
                d.ellipse((px,py,px+1,py+1),fill="#DFE7F4")
        d.rounded_rectangle((14,14,160,43),radius=7,fill="white",outline="#E2E9F4")
        d.text((87,20),"MINH HỌA" if source=="demo" else "CHUỘT THỬ",font=font(12,True),fill=BLUE,anchor="mt")
    observation=packet.observation
    if skeleton and observation.landmarks:
        def p(i):
            point=observation.landmarks[i]
            return (ox+(1-point.x if mirror else point.x)*width,oy+point.y*height)
        for a,b in HAND_CONNECTIONS:
            d.line((p(a),p(b)),fill="#B0C2E4",width=2)
        for i in range(21):
            px,py=p(i)
            d.ellipse((px-2,py-2,px+2,py+2),fill=BLUE)
    for stroke in engine.all_strokes():
        stroke_draw(d,stroke,rect)
    if engine.pointer is not None and engine.valid:
        u,v=engine.pointer
        px,py=x+u*pw,y+v*ph
        color=engine.color if engine.pen_down else "#8B9CB5"
        d.ellipse((px-11,py-11,px+11,py+11),outline="white",width=4)
        d.ellipse((px-10,py-10,px+10,py+10),outline=color,width=2)
        d.ellipse((px-3,py-3,px+3,py+3),fill=color)
    return image
