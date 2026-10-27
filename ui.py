"""Vietnamese desktop UI, hand skeleton and live comparison table."""
from functools import lru_cache
from pathlib import Path
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments import CONDITIONS, summarize
from gestures import GESTURES

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
               (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
               (13,17),(0,17),(17,18),(18,19),(19,20)]


@lru_cache(maxsize=32)
def font(size, bold=False):
    candidates = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / ("arialbd.ttf" if bold else "arial.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def render(frame, observation, gate, scene, lab, now, fps=0, last_command=None,
           event_age=999, landmarks=False, source="WEBCAM · ẢNH GƯƠNG", ar=None):
    image = Image.new("RGB", (1280, 960), "#0b111d")
    draw = ImageDraw.Draw(image)
    white, muted, cyan = "#eef4fd", "#95a8c0", "#55dfcc"

    def text(x, y, value, size=17, color=white, bold=False, anchor=None):
        draw.text((x,y), str(value), fill=color, font=font(size,bold), anchor=anchor)

    def box(rect, fill="#152235", outline=None, radius=18):
        draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=2)

    text(24,18,"HRI / MAGIC PALM",14,cyan,True)
    text(24,44,"HỨNG KHỐI 3D TRÊN BÀN TAY",31,bold=True)
    text(24,89,"Xòe tay để hứng · nắm rồi xòe để bung lấp lánh · ổn định ≥ 5 frame",18,muted)
    box((1052,39,1256,85),fill="#153a35")
    text(1154,51,f"ỔN ĐỊNH {gate.frames}/{gate.required}",18,cyan,True,"ma")
    box((24,128,680,652))
    h,w=frame.shape[:2]
    scale=min(640/w,480/h)
    width,height=max(1,round(w*scale)),max(1,round(h*scale))
    ox,oy=(640-width)//2,(480-height)//2
    camera=np.full((480,640,3),(29,17,11),dtype=np.uint8)
    camera[oy:oy+height,ox:ox+width]=cv2.resize(frame,(width,height))
    if landmarks and observation.landmarks:
        points=[(ox+int(p.x*width),oy+int(p.y*height)) for p in observation.landmarks]
        for a,b in CONNECTIONS:
            cv2.line(camera,points[a],points[b],(204,223,85),2,cv2.LINE_AA)
        for index,p in enumerate(points):
            cv2.circle(camera,p,5 if index in (4,8,12,16,20) else 3,(110,185,255),-1,cv2.LINE_AA)
    if ar is not None:
        camera=ar.overlay(camera,(ox,oy,width,height),scene)
    image.paste(Image.fromarray(cv2.cvtColor(camera,cv2.COLOR_BGR2RGB)),(32,136))
    text(42,628,source,13,muted)
    text(663,628,f"{fps:.0f} FPS · sáng {observation.brightness:.0f}/255 · tay {100*observation.area:.1f}%",13,muted,anchor="ra")
    box((704,128,1256,474))
    text(726,146,"PHÉP THUẬT TRONG LÒNG BÀN TAY",15,cyan,True)
    steps=(("XÒE TAY ĐỂ HỨNG","Đưa lòng bàn tay vào khung hình"),
           ("NẮM TAY ĐỂ TÍCH NĂNG LƯỢNG","Giữ ổn định đến khi khối sáng lên"),
           ("XÒE RA ĐỂ BUNG LẤP LÁNH","Các hạt bay ra rồi khối tự tái tạo"))
    active=0 if ar is None or not ar.spawned else (2 if ar.armed or ar.phase=="burst" else 1)
    for i,(title,description) in enumerate(steps):
        y=196+i*73
        color=cyan if i==active else muted
        draw.ellipse((726,y,764,y+38),fill="#21443f" if i==active else "#243348",outline=color,width=2)
        text(745,y+8,str(i+1),18,color,True,"ma")
        text(778,y,title,17,color,True)
        text(778,y+27,description,15,muted)
    hint="Xòe tay ổn định để khối xuất hiện" if ar is None else ar.hint
    text(726,423,hint,16,cyan)
    text(726,450,"Giơ ngón cái + trỏ + út: bung ngay khi đang hứng",13,muted)
    box((704,490,1256,652),outline=cyan if event_age<0.65 else None)
    raw_label="Chưa thấy bàn tay" if not observation.landmarks else GESTURES.get(observation.gesture,("Chưa rõ cử chỉ",))[0]
    text(725,505,f"Nhận dạng: {raw_label}",18,white,True)
    text(1235,536,f"Tin cậy: {observation.confidence:.0%}",14,muted,anchor="ra")
    if gate.required<=12:
        gap=140/gate.required
        for index in range(gate.required):
            x=726+index*gap
            box((x,537,x+gap-5,544),fill=cyan if index<gate.frames else "#30435a",radius=3)
    else:
        box((726,537,866,544),fill="#30435a",radius=3)
        if gate.frames:
            box((726,537,726+140*gate.frames/gate.required,544),fill=cyan,radius=3)
    if last_command:
        command,label=last_command
        text(726,558,label,26,cyan if event_age<2 else white,True)
        text(726,600,f"Lệnh gần nhất · {event_age:.1f}s trước",13,muted)
        text(1236,600,command,13,muted,anchor="ra")
    else:
        text(726,558,"CHỜ CỬ CHỈ ỔN ĐỊNH",23,muted,True)
    state_text="Đã xác nhận · giữ nguyên không phát lặp" if gate.stable else "Chưa đủ chuỗi ổn định · chưa phát lệnh mới"
    text(726,625,state_text,14,muted)
    box((24,668,832,910))
    text(42,684,"SO SÁNH ĐIỀU KIỆN",16,cyan,True)
    text(814,685,f"Đang so: {GESTURES[lab.expected][0]}",15,white,anchor="ra")
    for x,label in ((42,"Điều kiện [1–6]"),(298,"Lượt"),(360,"Frame đúng"),(473,"Lượt đạt"),(580,"Lệnh sai"),(684,"Trễ TB")):
        text(x,716,label,13,muted)
    stats=summarize(lab.rows,lab.expected)
    for index,(condition,label) in enumerate(CONDITIONS.items()):
        y=743+index*25
        if lab.condition==condition:
            box((36,y-2,818,y+22),fill="#24413f",radius=5)
        text(44,y,f"{index+1}. {label}",14,white)
        row=stats.get(condition)
        values=("—","—","—","—","—") if row is None else (row["trials"],f"{row['frame_accuracy']:.0f}%",f"{row['success']:.0f}%",row["wrong_events"],"—" if row["latency"] is None else f"{row['latency']:.2f}s")
        for x,value in zip((298,360,473,580,684),values):
            text(x,y,value,14,cyan if row else muted)
    box((848,668,1256,910))
    text(868,684,"CỬ CHỈ → LỆNH",16,cyan,True)
    for index,(gesture_name,command,command_label) in enumerate(GESTURES.values()):
        y=716+index*25
        text(867,y,gesture_name,14,white)
        short={"BUNG LẤP LÁNH":"BUNG TỎA"}.get(command_label,command_label)
        text(1236,y,short,13,cyan,True,anchor="ra")
    text(26,920,"1–6: điều kiện   G: cử chỉ đo   T: đo 4s   X: hủy   S: lưu bảng   L: điểm tay   R: đặt lại 3D   Q: thoát",14,muted)
    status=lab.status(now)
    if lab.active:
        box((34,568,670,610),fill="#102b2b",radius=8)
        text(48,578,status,19,cyan,True)
    else:
        text(26,942,status,12,muted)
    return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
