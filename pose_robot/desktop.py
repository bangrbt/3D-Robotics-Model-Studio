"""Native, responsive white desktop UI with actual controls and clear states."""
from collections import deque
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from queue import Empty
import time
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageDraw, ImageGrab, ImageTk

from .core import ACTION_NAMES, MotionState
from .graphics import BLUE, BORDER, INK, MUTED, camera_image, empty_camera, robot_image
from .worker import PoseWorker


class PoseDesktop:
    def __init__(self,root,initial=None,source=0):
        self.root=root
        root.title("Motion Studio · Human Pose to Robot")
        root.configure(bg="white")
        root.geometry("1400x860")
        root.minsize(1100,740)
        self.worker=None
        self.packet=None
        self.state=MotionState()
        self.events=deque(maxlen=4)
        self.closing=False
        self.stopping=False
        self.paused=False
        self.dirty=True
        self.last_paint=0.0
        self.images={}
        self.mode="idle"
        self.stale=False
        self.poll_id=None
        self.arm=tk.StringVar(value="Tự động")
        self.camera=tk.StringVar(value="Camera 0")
        self.mirror=tk.BooleanVar(value=True)
        self.skeleton=tk.BooleanVar(value=True)
        self.status=tk.StringVar(value="Sẵn sàng kết nối")
        self.hint=tk.StringVar(value="Đứng lùi vừa đủ để thấy rõ vai, khuỷu tay và cổ tay.")
        self.source_label=tk.StringVar(value="Chưa chọn nguồn hình")
        self.tracking=tk.StringVar(value="Chờ camera")
        self.fps=tk.StringVar(value="— FPS")
        self._style()
        self._build()
        root.protocol("WM_DELETE_WINDOW",self.close)
        root.bind("<Escape>",lambda event:self.stop())
        root.bind("<space>",self._space)
        root.bind("<Configure>",self._resize)
        root.bind("<Destroy>",self._destroyed,add="+")
        self.poll_id=root.after(25,self._poll)
        if initial:
            root.after(200,lambda:self.start(initial,source))

    def _style(self):
        s=ttk.Style(self.root)
        s.theme_use("clam")
        s.configure("TFrame",background="white")
        s.configure("TLabel",background="white",foreground=INK,font=("Segoe UI",10))
        s.configure("Title.TLabel",font=("Segoe UI Semibold",23),foreground=INK)
        s.configure("Muted.TLabel",foreground=MUTED,font=("Segoe UI",10))
        s.configure("Small.TLabel",foreground=MUTED,font=("Segoe UI",9))
        s.configure("CardTitle.TLabel",font=("Segoe UI Semibold",12))
        s.configure("Value.TLabel",font=("Segoe UI Semibold",23),foreground=INK)
        s.configure("TButton",font=("Segoe UI Semibold",10),padding=(14,9),background="white",foreground=INK,
                    borderwidth=1,bordercolor=BORDER,focuscolor="white")
        s.map("TButton",background=[("active","#F0F4FC"),("disabled","#FAFBFD")],foreground=[("disabled","#AFBACB")])
        s.configure("Primary.TButton",background=BLUE,foreground="white",bordercolor=BLUE,focuscolor=BLUE)
        s.map("Primary.TButton",background=[("active","#174ACD"),("disabled","#D6DFF2")],foreground=[("disabled","white")])
        s.configure("TCombobox",font=("Segoe UI",10),padding=7,fieldbackground="white",background="white",bordercolor=BORDER,
                    arrowcolor=MUTED,selectbackground="#EAF0FF",selectforeground=INK)
        s.map("TCombobox",fieldbackground=[("readonly","white")],selectbackground=[("readonly","white")],selectforeground=[("readonly",INK)])
        s.configure("TCheckbutton",font=("Segoe UI",10),background="white",foreground=MUTED,padding=(3,0))
        s.map("TCheckbutton",background=[("active","white")])
        s.configure("Horizontal.TProgressbar",background=BLUE,troughcolor="#EAF0F8",borderwidth=0,lightcolor=BLUE,darkcolor=BLUE)
        self.style_images=[]
        # Rounded, native clickable buttons with restrained hover colors.
        for prefix,colors in (("Clean",("white","#F3F6FC","#FAFBFD")),("Primary",(BLUE,"#174ACD","#DAE3F6"))):
            images=[]
            for color in colors:
                image=Image.new("RGB",(80,80),"white")
                draw=ImageDraw.Draw(image)
                draw.rounded_rectangle((1,1,78,78),radius=16,fill=color,outline=BORDER if prefix=="Clean" else color,width=2)
                photo=ImageTk.PhotoImage(image.resize((40,40),Image.Resampling.LANCZOS),master=self.root)
                images.append(photo)
                self.style_images.append(photo)
            element=prefix+".button"
            s.element_create(element,"image",images[0],("disabled",images[2]),("active",images[1]),border=9,sticky="nsew")
            name="TButton" if prefix=="Clean" else "Primary.TButton"
            s.layout(name,[(element,{"sticky":"nsew","children":[("Button.padding",{"sticky":"nsew","children":[("Button.label",{"sticky":"nsew"})]})]})])
        checks=[]
        for selected in (False,True):
            image=Image.new("RGB",(40,40),"white")
            draw=ImageDraw.Draw(image)
            draw.rounded_rectangle((3,3,36,36),radius=8,fill=BLUE if selected else "white",outline=BLUE if selected else "#B9C5D6",width=2)
            if selected:
                draw.line(((10,20),(17,27),(29,13)),fill="white",width=4,joint="curve")
            photo=ImageTk.PhotoImage(image.resize((18,18),Image.Resampling.LANCZOS),master=self.root)
            checks.append(photo)
            self.style_images.append(photo)
        s.element_create("Clean.indicator","image",checks[0],("selected",checks[1]),sticky="")
        s.layout("TCheckbutton",[("Checkbutton.padding",{"sticky":"nsew","children":[("Clean.indicator",{"side":"left","sticky":""}),("Checkbutton.label",{"side":"left","sticky":"w"})]})])

    def _line(self,parent):
        return tk.Frame(parent,bg=BORDER,height=1)

    def _card(self,parent):
        return tk.Frame(parent,bg="white",highlightthickness=1,highlightbackground=BORDER,bd=0)

    def _build(self):
        root=self.root
        header=ttk.Frame(root,padding=(30,20,30,17))
        header.pack(fill="x")
        logo=tk.Canvas(header,width=42,height=42,bg="white",highlightthickness=0)
        logo.pack(side="left",padx=(0,14))
        logo.create_rectangle(1,1,41,41,fill="#EDF2FF",outline="")
        logo.create_line(11,30,22,15,32,20,fill=BLUE,width=4)
        for x,y in ((11,30),(22,15),(32,20)):
            logo.create_oval(x-3,y-3,x+3,y+3,fill="white",outline=BLUE,width=2)
        titles=ttk.Frame(header)
        titles.pack(side="left")
        ttk.Label(titles,text="Motion Studio",style="Title.TLabel").pack(anchor="w")
        ttk.Label(titles,text="Human Pose to Robot Motion",style="Muted.TLabel").pack(anchor="w",pady=(1,0))
        status_frame=ttk.Frame(header)
        status_frame.pack(side="right",anchor="center")
        self.status_dot=tk.Label(status_frame,text="●",fg="#A8B5C7",bg="white",font=("Segoe UI",12))
        self.status_dot.pack(side="left",padx=6)
        ttk.Label(status_frame,textvariable=self.status,style="Muted.TLabel").pack(side="left")
        self._line(root).pack(fill="x")
        toolbar=ttk.Frame(root,padding=(30,16))
        toolbar.pack(fill="x")
        camera_select=ttk.Combobox(toolbar,textvariable=self.camera,values=[f"Camera {i}" for i in range(4)],state="readonly",width=11)
        camera_select.pack(side="left",padx=(0,10))
        self.start_button=ttk.Button(toolbar,text="Bật camera",style="Primary.TButton",command=self.connect)
        self.start_button.pack(side="left",padx=(0,8))
        self.pause_button=ttk.Button(toolbar,text="Tạm dừng",command=self.pause,state="disabled")
        self.pause_button.pack(side="left",padx=(0,8))
        self.stop_button=ttk.Button(toolbar,text="Ngắt kết nối",command=self.stop,state="disabled")
        self.stop_button.pack(side="left",padx=(0,8))
        ttk.Button(toolbar,text="Mở video",command=self.open_video).pack(side="left",padx=(0,8))
        ttk.Button(toolbar,text="Xem minh họa",command=lambda:self.start("demo",0)).pack(side="left")
        ttk.Checkbutton(toolbar,text="Ảnh gương",variable=self.mirror,command=self.configure).pack(side="right",padx=(10,0))
        ttk.Checkbutton(toolbar,text="Điểm mốc",variable=self.skeleton,command=self.configure).pack(side="right")
        self.notice=tk.Label(root,textvariable=self.hint,bg="#F4F7FD",fg="#566985",font=("Segoe UI",10),anchor="w",padx=14,pady=10)
        self.notice.pack(fill="x",padx=30,pady=(0,18))
        body=ttk.Frame(root,padding=(30,0,30,0))
        body.pack(fill="both",expand=True)
        body.columnconfigure(0,weight=6,uniform="panels")
        body.columnconfigure(1,weight=5,uniform="panels")
        body.rowconfigure(0,weight=1)
        left=self._card(body)
        left.grid(row=0,column=0,sticky="nsew",padx=(0,10))
        right=self._card(body)
        right.grid(row=0,column=1,sticky="nsew",padx=(10,0))
        left_top=ttk.Frame(left,padding=(18,15,18,11))
        left_top.pack(fill="x")
        ttk.Label(left_top,text="Quan sát chuyển động",style="CardTitle.TLabel").pack(side="left")
        ttk.Label(left_top,textvariable=self.fps,style="Small.TLabel").pack(side="right")
        self.camera_canvas=tk.Canvas(left,bg="#F8FAFD",highlightthickness=0,bd=0)
        self.camera_canvas.pack(fill="both",expand=True,padx=12,pady=(0,0))
        left_bottom=ttk.Frame(left,padding=(18,10,18,12))
        left_bottom.pack(fill="x")
        ttk.Label(left_bottom,textvariable=self.source_label,style="Small.TLabel").pack(side="left")
        self.track_label=ttk.Label(left_bottom,textvariable=self.tracking,style="Small.TLabel")
        self.track_label.pack(side="right")
        right_top=ttk.Frame(right,padding=(18,15,18,11))
        right_top.pack(fill="x")
        ttk.Label(right_top,text="Robot mô phỏng",style="CardTitle.TLabel").pack(side="left")
        ttk.Label(right_top,text="2D / Đồng bộ trực tiếp",style="Small.TLabel").pack(side="right")
        self.robot_canvas=tk.Canvas(right,bg="white",highlightthickness=0,bd=0)
        self.robot_canvas.pack(fill="both",expand=True,padx=12)
        right_bottom=ttk.Frame(right,padding=(18,5,18,11))
        right_bottom.pack(fill="x")
        ttk.Label(right_bottom,text="Tay điều khiển",style="Small.TLabel").pack(side="left",padx=(0,12))
        arm_select=ttk.Combobox(right_bottom,textvariable=self.arm,values=("Tự động","Tay trái","Tay phải"),state="readonly",width=12)
        arm_select.pack(side="left")
        arm_select.bind("<<ComboboxSelected>>",lambda e:self.configure())
        ttk.Label(right_bottom,text="Góc theo mặt phẳng ảnh",style="Small.TLabel").pack(side="right")
        metrics=ttk.Frame(root,padding=(30,17,30,0))
        metrics.pack(fill="x")
        for i in range(4):
            metrics.columnconfigure(i,weight=1,uniform="metrics")
        self.metric_values=[]
        for i,(title,value,caption) in enumerate((("GÓC KHUỶU TAY","—","Vai · Khuỷu · Cổ tay"),
                                                 ("GÓC ROBOT","100.0°","Góc bên trong tại khớp J2"),
                                                 ("ĐỘ TIN CẬY ĐIỂM MỐC","—","Thấp nhất của 3 điểm tay"),
                                                 ("HÀNH ĐỘNG","Chờ camera","Xác nhận sau ≥ 5 frame"))):
            card=self._card(metrics)
            card.grid(row=0,column=i,sticky="ew",padx=(0 if i==0 else 8,0 if i==3 else 8))
            ttk.Label(card,text=title,style="Small.TLabel").pack(anchor="w",padx=17,pady=(13,4))
            label=ttk.Label(card,text=value,style="Value.TLabel")
            if i==3:
                label.configure(font=("Segoe UI Semibold",20))
            label.pack(anchor="w",padx=17)
            ttk.Label(card,text=caption,style="Small.TLabel").pack(anchor="w",padx=17,pady=(4,13))
            self.metric_values.append(label)
        bottom=ttk.Frame(root,padding=(30,17,30,17))
        bottom.pack(fill="x")
        left_notes=ttk.Frame(bottom)
        left_notes.pack(side="left",fill="x",expand=True)
        self.history_label=ttk.Label(left_notes,text="Chưa có lệnh mới",style="Small.TLabel")
        self.history_label.pack(anchor="w")
        ttk.Label(left_notes,text="Giơ tay cao hơn vai · Hạ tay xuống · Vẫy cổ tay qua lại",style="Small.TLabel").pack(anchor="w",pady=(5,0))
        confirm=ttk.Frame(bottom)
        confirm.pack(side="right",padx=(22,0))
        self.confirm_label=ttk.Label(confirm,text="Độ ổn định  0 / 5",style="Small.TLabel")
        self.confirm_label.pack(anchor="e",pady=(0,7))
        self.progress=ttk.Progressbar(confirm,maximum=5,length=170)
        self.progress.pack()

    def _resize(self,event):
        if event.widget is self.root:
            self.dirty=True
            self.notice.configure(wraplength=max(200,event.width-100))

    def _destroyed(self,event):
        if event.widget is self.root:
            if self.poll_id is not None:
                try:
                    self.root.after_cancel(self.poll_id)
                except tk.TclError:
                    pass
                self.poll_id=None
            self.images.clear()
            self.style_images.clear()

    def _space(self,event):
        if isinstance(self.root.focus_get(),ttk.Combobox):
            return
        self.pause()

    def preference(self):
        return {"Tự động":"auto","Tay trái":"left","Tay phải":"right"}[self.arm.get()]

    def configure(self):
        if self.worker:
            self.worker.configure(self.preference(),self.mirror.get())
        self.dirty=True

    def connect(self):
        self.start("camera",int(self.camera.get().split()[-1]))

    def open_video(self):
        if self.worker and self.worker.thread.is_alive():
            self.hint.set("Ngắt nguồn hiện tại trước khi mở video khác.")
            return
        path=filedialog.askopenfilename(parent=self.root,title="Chọn video chuyển động",filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("Tất cả tệp","*.*")])
        if path:
            self.start("video",path)

    def start(self,kind,source):
        if self.closing or (self.worker and self.worker.thread.is_alive()):
            self.hint.set("Ngắt nguồn hiện tại trước khi chọn một nguồn mới.")
            return
        self.packet=None
        self.state=MotionState()
        self.stale=False
        self.mode=kind
        self.stopping=self.paused=False
        self.events.clear()
        self.history_label.configure(text="Chưa có lệnh mới")
        self.worker=PoseWorker(kind,source,self.preference(),self.mirror.get())
        self.worker.start()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.pause_button.configure(state="disabled",text="Tạm dừng")
        self.status.set("Đang khởi tạo…")
        self.hint.set("Đang kết nối. Bạn vẫn có thể thao tác giao diện.")
        self.notice.configure(bg="#F4F7FD",fg="#566985")
        self.source_label.set({"demo":"MINH HỌA · Không sử dụng camera","image":"ẢNH KIỂM TRA · Chỉ 1 frame","video":Path(str(source)).name,"camera":f"Camera {source} · Trực tiếp"}[kind])
        self.dirty=True

    def pause(self):
        if not self.worker or not self.worker.thread.is_alive() or self.stopping:
            return
        self.paused=not self.paused
        self.worker.pause(self.paused)
        self.pause_button.configure(text="Tiếp tục" if self.paused else "Tạm dừng")
        self.status.set("Tạm dừng" if self.paused else "Đang nhận dạng")
        self.status_dot.configure(fg="#D49B36" if self.paused else "#169B7C")
        self.hint.set("Đang tạm dừng · Robot giữ tư thế cuối. Bấm Tiếp tục để nhận dạng lại." if self.paused else "Đứng lùi vừa đủ để thấy rõ vai, khuỷu tay và cổ tay.")
        self.dirty=True

    def stop(self):
        if self.worker and self.worker.thread.is_alive():
            self.worker.stop()
            self.stopping=True
            self.pause_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.status.set("Đang ngắt kết nối…")
            self.dirty=True

    def close(self):
        self.closing=True
        self.stop()
        if not self.worker or not self.worker.thread.is_alive():
            self.root.destroy()

    def _put_image(self,canvas,image,key):
        self.images[key]=ImageTk.PhotoImage(image,master=self.root)
        canvas.delete("all")
        canvas.create_image(0,0,image=self.images[key],anchor="nw")

    def _paint(self):
        cw,ch=max(40,self.camera_canvas.winfo_width()),max(40,self.camera_canvas.winfo_height())
        rw,rh=max(40,self.robot_canvas.winfo_width()),max(40,self.robot_canvas.winfo_height())
        if self.packet is None:
            camera=empty_camera((cw,ch))
        else:
            camera=camera_image(self.packet.frame,self.packet.observation,self.state,(cw,ch),self.mirror.get(),self.skeleton.get() and not self.stale,self.packet.source=="demo")
        self._put_image(self.camera_canvas,camera,"camera")
        self._put_image(self.robot_canvas,robot_image(self.state,(rw,rh)),"robot")
        s=self.state
        values=(f"{s.angle:.1f}°" if s.angle is not None else "—",f"{s.robot_elbow:.1f}°",f"{s.confidence:.0%}" if s.valid else "—",ACTION_NAMES[s.action] if s.valid else "Chờ tư thế")
        for label,value in zip(self.metric_values,values):
            label.configure(text=value)
        self.metric_values[3].configure(foreground=BLUE if s.valid and s.action in ("RAISE","LOWER","WAVE") else INK)
        self.progress.configure(value=s.frames,maximum=s.required)
        self.confirm_label.configure(text=f"Độ ổn định  {s.frames} / {s.required}")
        self.tracking.set(f"{'Tay trái' if s.side=='left' else 'Tay phải'} · Đủ 3 điểm" if s.valid else "Cần thấy đủ vai · khuỷu · cổ tay")
        self.dirty=False

    def _poll(self):
        self.poll_id=None
        if self.closing:
            if not self.worker or not self.worker.thread.is_alive():
                self.root.destroy()
                return
            self.poll_id=self.root.after(30,self._poll)
            return
        if self.worker:
            while True:
                try:
                    kind,value=self.worker.messages.get_nowait()
                except Empty:
                    break
                if kind=="status":
                    self.hint.set(value)
                elif kind=="ready":
                    self.status.set("Chế độ minh họa" if value=="demo" else "Ảnh kiểm tra" if value=="image" else "Đang nhận dạng")
                    self.status_dot.configure(fg=BLUE if value=="demo" else "#169B7C")
                    self.pause_button.configure(state="normal" if value!="image" else "disabled")
                    self.hint.set("Chuyển động được tạo sẵn để xem giao diện; không phải kết quả webcam." if value=="demo" else "Đưa cả vai, khuỷu tay và cổ tay vào hình. Vẫy tay cần chuyển động qua lại rõ ràng.")
                elif kind=="action":
                    side,action=value
                    self.events.appendleft(f"{datetime.now():%H:%M:%S}   {ACTION_NAMES[action]}")
                    self.history_label.configure(text="     ·     ".join(list(self.events)[:3]))
                elif kind=="error":
                    self.invalidate()
                    self.status.set("Chưa kết nối được")
                    self.status_dot.configure(fg="#CB5C42")
                    self.hint.set(value)
                    self.notice.configure(bg="#FFF5F1",fg="#A54D35")
                elif kind=="ended":
                    self.pause_button.configure(state="disabled",text="Tạm dừng")
                    self.stop_button.configure(state="disabled")
                    self.start_button.configure(state="normal")
                    if self.stopping:
                        self.packet=None
                        self.state=MotionState()
                        self.mode="idle"
                        self.status.set("Đã ngắt kết nối")
                        self.status_dot.configure(fg="#A8B5C7")
                        self.source_label.set("Chưa chọn nguồn hình")
                        self.fps.set("— FPS")
                        self.hint.set("Camera đã được giải phóng. Chọn nguồn hình để bắt đầu lại.")
                    elif value=="video" and not self.status.get().startswith("Chưa"):
                        self.status.set("Đã phát hết video")
                    self.dirty=True
            packet=self.worker.latest()
            if packet is not None and not self.stopping:
                self.packet=packet
                self.state=packet.state
                if self.stale:
                    self.stale=False
                    self.status.set("Chế độ minh họa" if packet.source=="demo" else "Đang nhận dạng")
                    self.status_dot.configure(fg="#169B7C")
                    self.hint.set("Đã nhận lại khung hình. Giữ vai, khuỷu và cổ tay trong hình.")
                self.fps.set("ẢNH TĨNH" if packet.source=="image" else f"{packet.fps:.0f} FPS")
                self.dirty=True
        now=time.monotonic()
        if (self.worker and self.worker.thread.is_alive() and self.packet and self.mode in ("camera","video")
                and not self.paused and not self.stopping and not self.stale and now-self.packet.captured_at>.9):
            self.invalidate()
            self.status.set("Đang chờ tín hiệu")
            self.status_dot.configure(fg="#D49B36")
            self.hint.set("Khung hình chưa cập nhật. Robot tạm giữ tư thế cuối; không phát lệnh mới.")
        if self.dirty and now-self.last_paint>.035:
            self._paint()
            self.last_paint=now
        self.poll_id=self.root.after(16,self._poll)

    def invalidate(self):
        self.stale=True
        self.state=replace(self.state,valid=False,angle=None,raw_angle=None,confidence=0,
                           action="NO_POSE",candidate="NO_POSE",frames=0,event=None,points=None)
        self.dirty=True

    def snapshot(self,path):
        """Capture this app window only; used for visual verification."""
        self.root.update_idletasks()
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        # The outer application HWND (rather than Tk's child client handle).
        import ctypes
        hwnd=ctypes.windll.user32.GetParent(self.root.winfo_id())
        ImageGrab.grab(window=hwnd).save(path)
