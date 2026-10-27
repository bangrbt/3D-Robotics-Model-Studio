"""Conservative basic-shape labels. Recognition never rewrites the live ink."""
from dataclasses import dataclass

import cv2
import numpy as np

from .kinematics import PAPER_WIDTH,PAPER_HEIGHT


@dataclass(frozen=True)
class Shape:
    key: str="freehand"
    label: str="Nét tự do"
    closed: bool=False


def _resample(points,count=100):
    distances=np.linalg.norm(np.diff(points,axis=0),axis=1)
    cumulative=np.r_[0,np.cumsum(distances)]
    unique=np.r_[True,np.diff(cumulative)>1e-8]
    cumulative,points=cumulative[unique],points[unique]
    if cumulative[-1]<1e-8:
        return np.repeat(points[:1],count,axis=0)
    target=np.linspace(0,cumulative[-1],count)
    return np.column_stack([np.interp(target,cumulative,points[:,i]) for i in range(2)])


def recognize_shape(uv):
    p=np.asarray(uv,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<1 or not np.isfinite(p).all():
        return Shape()
    p=p*np.array((PAPER_WIDTH,PAPER_HEIGHT))
    if len(p)==1:
        return Shape("point","Chấm")
    path=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
    span=np.ptp(p,axis=0)
    diagonal=float(np.linalg.norm(span))
    if diagonal<.025:
        return Shape("point","Chấm")
    if path<.06:
        return Shape()
    endpoints=float(np.linalg.norm(p[-1]-p[0]))
    centered=p-p.mean(axis=0)
    _,s,_=np.linalg.svd(centered,full_matrices=False)
    if len(s)>1 and s[1]/max(s[0],1e-9)<.09 and endpoints/max(path,1e-9)>.78:
        return Shape("line","Đường thẳng")
    closed=endpoints<=max(.035,diagonal*.17)
    if not closed or len(p)<10 or diagonal<.12:
        return Shape()
    p=np.vstack((p,p[0]))
    p=_resample(p)
    contour=(p*1000).astype(np.float32).reshape(-1,1,2)
    perimeter=cv2.arcLength(contour,True)
    polygon=cv2.approxPolyDP(contour,.035*perimeter,True).reshape(-1,2)/1000
    if not cv2.isContourConvex((polygon*1000).astype(np.float32).reshape(-1,1,2)):
        return Shape()
    area=abs(cv2.contourArea(contour))/1e6
    if area<diagonal*diagonal*.08:
        return Shape()
    if len(polygon)==3:
        return Shape("triangle","Tam giác",True)
    if len(polygon)==4:
        edges=np.roll(polygon,-1,axis=0)-polygon
        lengths=np.linalg.norm(edges,axis=1)
        if min(lengths)<.02:
            return Shape()
        normalized=edges/lengths[:,None]
        cosines=np.abs(np.sum(normalized*np.roll(normalized,1,axis=0),axis=1))
        if max(cosines)<.35:
            ratio=max(lengths)/min(lengths)
            return Shape("square","Hình vuông",True) if ratio<1.25 else Shape("rectangle","Chữ nhật",True)
        return Shape()
    if len(polygon)>=5:
        # Radial error + aspect ratio keep freehand loops from being called circles.
        matrix=np.column_stack((2*p[:,0],2*p[:,1],np.ones(len(p))))
        fit=np.linalg.lstsq(matrix,np.sum(p*p,axis=1),rcond=None)[0]
        distances=np.linalg.norm(p-fit[:2],axis=1)
        radius=float(distances.mean())
        aspect=max(span)/max(min(span),1e-9)
        if radius>.03 and distances.std()/radius<.13 and aspect<1.3:
            return Shape("circle","Hình tròn",True)
        if len(p)>=5 and aspect<3.5:
            (cx,cy),(a,b),rotation=cv2.fitEllipse(contour)
            if min(a,b)>1:
                theta=np.deg2rad(rotation)
                q=p*1000-np.array((cx,cy))
                x=q[:,0]*np.cos(theta)+q[:,1]*np.sin(theta)
                y=-q[:,0]*np.sin(theta)+q[:,1]*np.cos(theta)
                residual=np.sqrt((x/(a/2))**2+(y/(b/2))**2)
                if np.std(residual)<.12:
                    return Shape("ellipse","Elip",True)
    return Shape()


def recognize_recent(strokes):
    if not strokes:
        return Shape()
    last=recognize_shape(strokes[-1].points)
    if last.closed:
        return last
    # Allow a square/triangle to be drawn as separate, connected edges.
    for count in range(2,min(8,len(strokes))+1):
        group=strokes[-count:]
        if any(recognize_shape(s.points).closed for s in group[:-1]):
            break
        chain=list(group[0].points)
        valid=True
        for stroke in group[1:]:
            points=list(stroke.points)
            if not points:
                continue
            end=np.array(chain[-1])*np.array((PAPER_WIDTH,PAPER_HEIGHT))
            distances=[np.linalg.norm(end-np.array(points[i])*np.array((PAPER_WIDTH,PAPER_HEIGHT))) for i in (0,-1)]
            if min(distances)>.07:
                valid=False
                break
            if distances[1]<distances[0]:
                points.reverse()
            chain.extend(points)
        if valid:
            result=recognize_shape(chain)
            if result.closed:
                return result
    return last
