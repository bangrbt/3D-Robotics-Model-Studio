"""Export only the actual robot pen trace; no snapping or ideal-shape replacement."""
from html import escape
from pathlib import Path

from PIL import Image,ImageDraw

from .graphics import stroke_draw


def export_svg(engine,path):
    width,height=1400,1000
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>']
    for stroke in engine.all_strokes():
        if not stroke.points:
            continue
        color=escape(stroke.color,quote=True)
        thick=stroke.width*width/500
        if len(stroke.points)==1 or len(set(stroke.points))==1:
            u,v=stroke.points[0]
            lines.append(f'<circle cx="{u*width:.3f}" cy="{v*height:.3f}" r="{thick/2:.3f}" fill="{color}"/>')
        else:
            points=" ".join(f"{u*width:.3f},{v*height:.3f}" for u,v in stroke.points)
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{thick:.2f}" stroke-linecap="round" stroke-linejoin="round"/>')
    lines.append('</svg>')
    Path(path).write_text("\n".join(lines),encoding="utf-8")


def export_png(engine,path):
    image=Image.new("RGB",(1400,1000),"white")
    d=ImageDraw.Draw(image)
    for stroke in engine.all_strokes():
        stroke_draw(d,stroke,(0,0,1400,1000))
    image.save(path,format="PNG")
