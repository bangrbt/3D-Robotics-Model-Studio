"""Illustrate AR rendering on one photo with EXPLICIT simulated gesture states.

This is a visual preview, not gesture-recognition or webcam accuracy evidence.
It writes no command/evaluation logs and does not duplicate a photo as a trial.
"""
from pathlib import Path
import argparse

import cv2
from PIL import Image, ImageDraw

from gestures import GateState
from palm_ar import PalmAR
from scene3d import Scene3D
from ui import font
from vision import ROOT, create_recognizer, ensure_model, recognize


def main():
    parser=argparse.ArgumentParser(description="AR animation illustration with simulated controls, NOT live gesture validation")
    parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts"/"palm_ar_demo.gif")
    args=parser.parse_args()
    photo=cv2.imread(str(args.image))
    if photo is None:
        raise RuntimeError("Khong doc duoc anh minh hoa.")
    photo=cv2.flip(photo,1)
    with create_recognizer(ensure_model()) as recognizer:
        observation=recognize(recognizer,photo,0)
    if len(observation.landmarks)<21:
        raise RuntimeError("Anh minh hoa can co 21 diem moc tay.")
    h,w=photo.shape[:2]
    ratio=min(640/w,480/h)
    width,height=round(w*ratio),round(h*ratio)
    ox,oy=(640-width)//2,(480-height)//2
    camera=cv2.copyMakeBorder(cv2.resize(photo,(width,height)),oy,480-height-oy,ox,640-width-ox,
                              cv2.BORDER_CONSTANT,value=(25,19,11))
    ar,scene=PalmAR(),Scene3D()
    frames=[]
    for i in range(90):
        if i<20:
            gesture,command,label="Open_Palm","START","XÒE TAY · HỨNG KHỐI"
        elif i<35:
            gesture,command,label="Closed_Fist","STOP","NẮM TAY · TÍCH NĂNG LƯỢNG"
        else:
            gesture,command,label="Open_Palm","START","XÒE RA · BUNG LẤP LÁNH"
        event=command if i in (0,20,35) else None
        state=GateState(gesture,5,5,True,event)
        if event:
            scene.apply(event)
        scene.update(1/15)
        ar.update(observation.landmarks,state,1/15,observation.handedness,scene)
        overlay=ar.overlay(camera,(ox,oy,width,height),scene)
        canvas=Image.new("RGB",(640,566),"#0b111d")
        canvas.paste(Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)),(0,42))
        draw=ImageDraw.Draw(canvas)
        draw.text((18,10),label,font=font(19,True),fill="#55dfcc")
        draw.text((12,531),"MINH HỌA AR · ẢNH TĨNH / TRẠNG THÁI CỬ CHỈ MÔ PHỎNG",font=font(13,True),fill="#c7d2e4")
        frames.append(canvas)
        if i in (15,28,43):
            args.output.parent.mkdir(parents=True,exist_ok=True)
            canvas.save(args.output.parent/f"palm_ar_{'hold' if i==15 else 'charge' if i==28 else 'burst'}.jpg")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(args.output,save_all=True,append_images=frames[1:],duration=67,loop=0,optimize=True)
    print(f"Illustration saved: {args.output}. Simulated controls, not a webcam measurement.")


if __name__=="__main__":
    main()
