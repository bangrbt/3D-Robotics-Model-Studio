"""White Air Drawing desktop, hand input and robot ink share one drawing engine."""
from pathlib import Path
from queue import Empty
import math
import time
import tkinter as tk
from tkinter import filedialog,ttk

import numpy as np

from pose_robot.desktop import PoseDesktop
from pose_robot.graphics import BLUE,INK,MUTED,BORDER
from .core import DrawingEngine,HandObservation
from .export import export_png,export_svg
from .graphics import PaperRenderer,camera_image
from .kinematics import drawing_roi
from .worker import Frame,HandWorker


class DrawingDesktop(PoseDesktop):
    def __init__(self,root,initial=None,source=0):
        self.root=root
        root.title("Motion Studio · Air Drawing Robot")
        root.configure(bg="white")
        available_width=max(900,root.winfo_screenwidth()-60)
        available_height=max(600,root.winfo_screenheight()-80)
        root.geometry(f"{min(1480,available_width)}x{min(880,available_height)}")
        root.minsize(min(1260,available_width),min(790,available_height))
        self.worker=self.packet=None
        self.user_engine=DrawingEngine()
        self.engine=self.user_engine
        self.renderer=PaperRenderer()
        self.closing=self.stopping=self.paused=self.stale=False
        self.dirty=True
        self.last_paint=0
        self.last_input=0
        self.images={}
        self.poll_id=None
        self.mode="idle"
        self.camera=tk.StringVar(value="Camera 0")
        self.mirror=tk.BooleanVar(value=True)
        self.skeleton=tk.BooleanVar(value=False)
        self.show_robot=tk.BooleanVar(value=True)
        self.pen_mode=tk.StringVar(value="Chụm để vẽ")
        self.pen_size=tk.StringVar(value="3 px")
        self.status=tk.StringVar(value="Sẵn sàng kết nối")
        self.hint=tk.StringVar(value="Chụm ngón cái + ngón trỏ để vẽ · Thả ra để nhấc bút · Giữ đầu ngón trỏ trong vùng xanh.")
        self.source_label=tk.StringVar(value="Chưa chọn nguồn hình")
        self.tracking=tk.StringVar(value="Chờ bàn tay")
        self.fps=tk.StringVar(value="— FPS")
        self.mouse_uv=None
        self.mouse_pressed=False
        self._style()
        self._build()
        root.protocol("WM_DELETE_WINDOW",self.close)
        root.bind("<Configure>",self._resize)
        root.bind("<Destroy>",self._destroyed,add="+")
        root.bind("<Escape>",lambda event:self.stop())
        root.bind("<space>",self._space)
        root.bind("<Control-z>",lambda event:self.undo())
        root.bind("<Delete>",lambda event:self.clear())
        root.bind("<ButtonRelease-1>",self._mouse_release,add="+")
        self.poll_id=root.after(25,self._poll)
        if initial:
            root.after(180,lambda:self.start(initial,source))

    def _build(self):
        root=self.root
        header=ttk.Frame(root,padding=(28,17,28,16))
        header.pack(fill="x")
        logo=tk.Canvas(header,width=42,height=42,bg="white",highlightthickness=0)
        logo.pack(side="left",padx=(0,14))
        logo.create_rectangle(1,1,41,41,fill="#EDF2FF",outline="")
        logo.create_line(10,31,22,14,32,24,fill=BLUE,width=4)
        logo.create_oval(18,10,26,18,fill="white",outline=BLUE,width=2)
        logo.create_line(32,24,30,33,fill=BLUE,width=3)
        titles=ttk.Frame(header)
        titles.pack(side="left")
        ttk.Label(titles,text="Motion Studio",style="Title.TLabel").pack(anchor="w")
        ttk.Label(titles,text="Air Drawing · Robot vẽ theo chuyển động tay",style="Muted.TLabel").pack(anchor="w")
        status=ttk.Frame(header)
        status.pack(side="right")
        self.status_dot=tk.Label(status,text="●",fg="#A8B5C7",bg="white",font=("Segoe UI",12))
        self.status_dot.pack(side="left",padx=7)
        ttk.Label(status,textvariable=self.status,style="Muted.TLabel").pack(side="left")
        self._line(root).pack(fill="x")
        row=ttk.Frame(root,padding=(28,13))
        row.pack(fill="x")
        ttk.Combobox(row,textvariable=self.camera,values=[f"Camera {i}" for i in range(4)],state="readonly",width=10).pack(side="left",padx=(0,8))
        self.start_button=ttk.Button(row,text="Bật camera",style="Primary.TButton",command=self.connect)
        self.start_button.pack(side="left",padx=(0,7))
        self.pause_button=ttk.Button(row,text="Tạm dừng",command=self.pause,state="disabled")
        self.pause_button.pack(side="left",padx=(0,7))
        self.stop_button=ttk.Button(row,text="Ngắt kết nối",command=self.stop,state="disabled")
        self.stop_button.pack(side="left",padx=(0,7))
        ttk.Button(row,text="Mở video",command=self.open_video).pack(side="left",padx=(0,7))
        ttk.Button(row,text="Xem minh họa",command=lambda:self.start("demo",0)).pack(side="left",padx=(0,7))
        ttk.Button(row,text="Vẽ bằng chuột",command=lambda:self.start("mouse",0)).pack(side="left")
        ttk.Checkbutton(row,text="Ảnh gương",variable=self.mirror,command=self.configure).pack(side="right",padx=(9,0))
        ttk.Checkbutton(row,text="Điểm tay",variable=self.skeleton,command=self.redraw).pack(side="right")
        self.notice=tk.Label(root,textvariable=self.hint,bg="#F4F7FD",fg="#566985",font=("Segoe UI",10),anchor="w",padx=14,pady=10)
        self.notice.pack(fill="x",padx=28,pady=(0,10))
        tools=ttk.Frame(root,padding=(28,0,28,12))
        tools.pack(fill="x")
        ttk.Label(tools,text="Bút",style="Small.TLabel").pack(side="left",padx=(0,9))
        mode=ttk.Combobox(tools,textvariable=self.pen_mode,values=("Chụm để vẽ","Ngón trỏ để vẽ"),state="readonly",width=16)
        mode.pack(side="left",padx=(0,10))
        mode.bind("<<ComboboxSelected>>",lambda e:self.configure())
        size=ttk.Combobox(tools,textvariable=self.pen_size,values=("2 px","3 px","5 px","7 px"),state="readonly",width=5)
        size.pack(side="left",padx=(0,10))
        size.bind("<<ComboboxSelected>>",lambda e:self.change_style(width=int(self.pen_size.get().split()[0])))
        self.palette=tk.Canvas(tools,width=150,height=34,bg="white",highlightthickness=0)
        self.palette.pack(side="left",padx=(0,8))
        self._palette()
        ttk.Button(tools,text="Hoàn tác",command=self.undo).pack(side="left",padx=(0,7))
        ttk.Button(tools,text="Xóa bản vẽ",command=self.clear).pack(side="left")
        ttk.Button(tools,text="Lưu SVG",command=lambda:self.save("svg")).pack(side="right")
        ttk.Button(tools,text="Lưu PNG",command=lambda:self.save("png")).pack(side="right",padx=(0,7))
        ttk.Checkbutton(tools,text="Hiện robot",variable=self.show_robot,command=self.redraw).pack(side="right",padx=(0,14))
        body=ttk.Frame(root,padding=(28,0,28,0))
        body.pack(fill="both",expand=True)
        body.columnconfigure(0,weight=1,uniform="drawing")
        body.columnconfigure(1,weight=1,uniform="drawing")
        body.rowconfigure(0,weight=1)
        left=self._card(body)
        left.grid(row=0,column=0,sticky="nsew",padx=(0,10))
        right=self._card(body)
        right.grid(row=0,column=1,sticky="nsew",padx=(10,0))
        for card,title,extra in ((left,"Tay bạn vẽ",self.fps),(right,"Robot vẽ theo",None)):
            top=ttk.Frame(card,padding=(17,13,17,11))
            top.pack(fill="x")
            ttk.Label(top,text=title,style="CardTitle.TLabel").pack(side="left")
            ttk.Label(top,textvariable=extra,text="Đầu bút đồng bộ trực tiếp" if extra is None else "",style="Small.TLabel").pack(side="right")
        self.camera_canvas=tk.Canvas(left,bg="#F8FAFD",highlightthickness=0,bd=0)
        self.camera_canvas.pack(fill="both",expand=True,padx=11)
        self.camera_canvas.bind("<Motion>",self._mouse_move)
        self.camera_canvas.bind("<ButtonPress-1>",self._mouse_down)
        self.camera_canvas.bind("<Leave>",self._mouse_leave)
        footer=ttk.Frame(left,padding=(17,10,17,12))
        footer.pack(fill="x")
        ttk.Label(footer,textvariable=self.source_label,style="Small.TLabel").pack(side="left")
        ttk.Label(footer,textvariable=self.tracking,style="Small.TLabel").pack(side="right")
        self.robot_canvas=tk.Canvas(right,bg="#F8FAFD",highlightthickness=0,bd=0)
        self.robot_canvas.pack(fill="both",expand=True,padx=11)
        bottom=ttk.Frame(right,padding=(17,10,17,12))
        bottom.pack(fill="x")
        ttk.Label(bottom,text="Giữ nguyên nét gốc · Không tự nắn hình",style="Small.TLabel").pack(side="left")
        self.count_label=ttk.Label(bottom,text="0 nét",style="Small.TLabel")
        self.count_label.pack(side="right")
        metrics=ttk.Frame(root,padding=(28,14,28,0))
        metrics.pack(fill="x")
        self.metric_values=[]
        for i,(title,value,caption) in enumerate((("TRẠNG THÁI BÚT","Nhấc bút","Chụm 5 frame để bắt đầu nét"),
                                                 ("KHỚP VAI","—","Góc quay của khâu thứ nhất"),
                                                 ("KHỚP KHUỶU","—","Góc gập của khâu thứ hai"),
                                                 ("HÌNH GẦN NHẤT","Chưa có hình","Nhận dạng khi nhấc bút"))):
            metrics.columnconfigure(i,weight=1,uniform="stats")
            card=self._card(metrics)
            card.grid(row=0,column=i,sticky="ew",padx=(0 if i==0 else 7,0 if i==3 else 7))
            ttk.Label(card,text=title,style="Small.TLabel").pack(anchor="w",padx=16,pady=(11,3))
            label=ttk.Label(card,text=value,font=("Segoe UI Semibold",20),foreground=BLUE if i in (0,3) else INK)
            label.pack(anchor="w",padx=16)
            ttk.Label(card,text=caption,style="Small.TLabel").pack(anchor="w",padx=16,pady=(3,10))
            self.metric_values.append(label)
        end=ttk.Frame(root,padding=(28,13,28,13))
        end.pack(fill="x")
        self.live_label=ttk.Label(end,text="Tròn · Vuông · Chữ nhật · Tam giác · Đường thẳng · Elip",style="Small.TLabel")
        self.live_label.pack(side="left")
        confirm=ttk.Frame(end)
        confirm.pack(side="right")
        self.confirm_label=ttk.Label(confirm,text="Hạ bút  0 / 5",style="Small.TLabel")
        self.confirm_label.pack(side="left",padx=(0,12))
        self.progress=ttk.Progressbar(confirm,maximum=5,length=140)
        self.progress.pack(side="left")

    def _palette(self):
        self.palette.delete("all")
        for i,color in enumerate(("#285CE6","#17263F","#14A187","#EA6A42","#9561D8")):
            x=17+i*29
            if color==self.engine.color:
                self.palette.create_oval(x-13,4,x+13,30,outline=color,width=1)
            item=self.palette.create_oval(x-9,8,x+9,26,fill=color,outline=color)
            self.palette.tag_bind(item,"<Button-1>",lambda event,c=color:self.change_style(color=c))

    def _space(self,event):
        if not isinstance(self.root.focus_get(),(ttk.Combobox,ttk.Button)):
            self.pause()

    def configure(self):
        self.engine.reset_tracking()
        if self.worker:
            self.worker.clear_samples()
        self.hint.set(self.instruction())
        self.dirty=True

    def redraw(self):
        self.dirty=True

    def instruction(self):
        return ("Chụm ngón cái + ngón trỏ 5 frame để hạ bút. Thả ra để nhấc bút. Vẽ trong khung xanh." if self.pen_mode.get()=="Chụm để vẽ" else
                "Giơ ngón trỏ để vẽ; giơ thêm ngón giữa để nhấc bút. Giữ đầu ngón trỏ trong khung xanh.")

    def change_style(self,color=None,width=None):
        self.engine.change_style(color,width)
        self._palette()
        self.dirty=True

    def undo(self):
        self.engine.undo()
        self.dirty=True

    def clear(self):
        self.engine.clear()
        self.dirty=True

    def save(self,format):
        if not self.engine.all_strokes():
            self.hint.set("Chưa có nét vẽ để lưu. Bật camera hoặc vẽ thử bằng chuột trước.")
            return
        was_paused=self.paused
        self.paused=True
        self.mouse_pressed=False
        self.engine.lift()
        if self.worker:
            self.worker.pause(True)
            self.worker.clear_samples()
        try:
            path=filedialog.asksaveasfilename(parent=self.root,title="Lưu nét robot đã vẽ",defaultextension="."+format,
                                             initialfile="ban_ve_robot."+format,filetypes=[(format.upper(),"*."+format)])
            if path:
                (export_svg if format=="svg" else export_png)(self.engine,path)
                self.hint.set("Đã lưu bản vẽ: "+Path(path).name)
        except (OSError,ValueError) as error:
            self.hint.set("Không lưu được bản vẽ: "+str(error))
        finally:
            self.paused=was_paused
            self.engine.reset_tracking()
            if self.worker:
                self.worker.clear_samples()
                self.worker.pause(was_paused)
            self.dirty=True

    def connect(self):
        self.start("camera",int(self.camera.get().split()[-1]))

    def open_video(self):
        if self.worker and self.worker.thread.is_alive():
            self.hint.set("Ngắt nguồn hiện tại trước khi mở video khác.")
            return
        path=filedialog.askopenfilename(parent=self.root,title="Chọn video bàn tay vẽ",filetypes=[("Video","*.mp4 *.mov *.avi *.mkv"),("Tất cả","*.*")])
        if path:
            self.start("video",path)

    def start(self,kind,source=0):
        if self.closing or (self.worker and self.worker.thread.is_alive()):
            self.hint.set("Ngắt nguồn hiện tại trước khi chọn nguồn mới.")
            return
        self.worker=None
        self.engine=DrawingEngine() if kind=="demo" else self.user_engine
        self.engine.reset_tracking()
        self.engine.width=int(self.pen_size.get().split()[0])
        self._palette()
        self.mode=kind
        self.stopping=self.paused=self.stale=False
        self.packet=None
        self.last_input=0
        self.mouse_uv=None
        self.mouse_pressed=False
        self.status_dot.configure(fg=BLUE)
        self.notice.configure(bg="#F4F7FD",fg="#566985")
        self.source_label.set({"camera":f"Camera {source} · Trực tiếp","video":Path(str(source)).name,"demo":"MINH HỌA · Điểm tay tạo sẵn","mouse":"CHUỘT THỬ · Không dùng camera","image":"ẢNH KIỂM TRA · 1 frame"}[kind])
        self.start_button.configure(state="disabled" if kind!="mouse" else "normal")
        self.pause_button.configure(text="Tạm dừng",state="normal")
        self.stop_button.configure(state="normal")
        if kind=="mouse":
            self.packet=Frame(np.full((480,640,3),(253,250,247),np.uint8),HandObservation(),0,time.monotonic())
            self.status.set("Chế độ vẽ bằng chuột")
            self.hint.set("Giữ chuột trái trong vùng vẽ để hạ bút. Thả để nhấc bút. Robot theo nét như khi dùng camera.")
            self.fps.set("CHUỘT THỬ")
        else:
            self.worker=HandWorker(kind,source)
            self.worker.start()
            self.status.set("Đang khởi tạo…")
            self.hint.set("Đang kết nối. Mô hình bàn tay chạy riêng để giao diện luôn phản hồi.")
        self.dirty=True

    def pause(self):
        if self.mode=="idle" or self.stopping:
            return
        if self.mode!="mouse" and (not self.worker or not self.worker.thread.is_alive()):
            return
        self.paused=not self.paused
        self.engine.reset_tracking()
        self.mouse_pressed=False
        if self.worker:
            self.worker.pause(self.paused)
            self.worker.clear_samples()
        self.pause_button.configure(text="Tiếp tục" if self.paused else "Tạm dừng")
        self.status.set("Tạm dừng · Bút đã nhấc" if self.paused else "Đang nhận dạng" if self.mode=="camera" else "Minh họa" if self.mode=="demo" else "Đang vẽ")
        self.status_dot.configure(fg="#D49B36" if self.paused else "#169B7C")
        self.dirty=True

    def stop(self):
        self.engine.reset_tracking()
        self.mouse_pressed=False
        self.stopping=True
        if self.worker and self.worker.thread.is_alive():
            self.worker.stop()
            self.stopping=True
            self.stop_button.configure(state="disabled")
            self.pause_button.configure(state="disabled")
            self.status.set("Đang ngắt kết nối…")
        else:
            self._ended()
        self.dirty=True

    def _ended(self):
        self.engine.reset_tracking()
        self.pause_button.configure(state="disabled",text="Tạm dừng")
        self.stop_button.configure(state="disabled")
        self.start_button.configure(state="normal")
        if self.stopping or self.mode=="mouse":
            self.status.set("Đã ngắt nguồn · Đã giữ bản vẽ")
        self.status_dot.configure(fg="#A8B5C7")

    def _mouse_position(self,event):
        w,h=self.camera_canvas.winfo_width(),self.camera_canvas.winfo_height()
        scale=min(w/640,h/480)
        ox,oy=(w-640*scale)/2,(h-480*scale)/2
        rx,ry,rw,rh=drawing_roi(640,480)
        u=((event.x-ox)/scale-rx)/rw
        v=((event.y-oy)/scale-ry)/rh
        return (u,v) if 0<=u<=1 and 0<=v<=1 else None

    def _mouse_move(self,event):
        if self.mode=="mouse":
            self.mouse_uv=self._mouse_position(event)

    def _mouse_down(self,event):
        if self.mode=="mouse":
            self.mouse_uv=self._mouse_position(event)
            self.mouse_pressed=True

    def _mouse_release(self,event):
        if self.mode=="mouse":
            self.mouse_pressed=False
            self.engine.lift()
            self.dirty=True

    def _mouse_leave(self,event):
        if self.mode=="mouse":
            self.mouse_uv=None
            self.engine.lift()
            self.dirty=True

    def _paint(self):
        cw,ch=max(40,self.camera_canvas.winfo_width()),max(40,self.camera_canvas.winfo_height())
        rw,rh=max(40,self.robot_canvas.winfo_width()),max(40,self.robot_canvas.winfo_height())
        image=camera_image(self.packet,self.engine,(cw,ch),self.mirror.get(),self.skeleton.get() and not self.stale,self.mode)
        self._put_image(self.camera_canvas,image,"camera")
        self._put_image(self.robot_canvas,self.renderer.render(self.engine,(rw,rh),self.show_robot.get()),"robot")
        e=self.engine
        values=("Đang vẽ" if e.pen_down else "Đang hạ bút" if e.frames else "Nhấc bút",
                f"{math.degrees(e.robot.shoulder):.1f}°",f"{180-abs(math.degrees(e.robot.elbow)):.1f}°",e.shape.label)
        for widget,value in zip(self.metric_values,values):
            widget.configure(text=value)
        self.count_label.configure(text=f"{len(e.strokes)+(e.current is not None)} nét")
        self.live_label.configure(text=e.status)
        self.confirm_label.configure(text=f"Hạ bút  {e.frames} / {e.required}")
        self.progress.configure(value=e.frames,maximum=e.required)
        self.tracking.set("Theo dõi đầu ngón trỏ" if e.valid else "Giữ ngón tay trong vùng vẽ")
        self.dirty=False

    def _poll(self):
        self.poll_id=None
        if self.closing:
            if not self.worker or not self.worker.thread.is_alive():
                self.root.destroy()
                return
            self.poll_id=self.root.after(30,self._poll)
            return
        now=time.monotonic()
        if self.mode=="mouse" and not self.paused and not self.stopping and now-self.last_input>1/30:
            self.engine.update_pointer(self.mouse_uv,self.mouse_pressed,now,"mouse")
            self.last_input=now
            self.dirty=True
        if self.worker:
            while True:
                try:
                    sample=self.worker.samples.get_nowait()
                except Empty:
                    break
                if self.paused or self.stopping:
                    continue
                if sample.discontinuity:
                    self.engine.reset_tracking()
                if self.mode=="demo":
                    self.engine.update_pointer(sample.manual_uv,sample.manual_pressed,sample.time,"demo")
                else:
                    self.engine.update(sample.observation,sample.time,self.mirror.get(),"pinch" if self.pen_mode.get()=="Chụm để vẽ" else "index")
                self.dirty=True
            while True:
                try:
                    kind,value=self.worker.messages.get_nowait()
                except Empty:
                    break
                if kind=="status":
                    self.hint.set(value)
                elif kind=="ready":
                    self.status.set("Minh họa chuyển động" if value=="demo" else "Ảnh kiểm tra" if value=="image" else "Robot sẵn sàng vẽ theo")
                    self.status_dot.configure(fg=BLUE if value=="demo" else "#169B7C")
                    self.hint.set("Chuyển động tạo sẵn để xem robot vẽ, không phải kết quả nhận dạng webcam. Bản vẽ thật của bạn được giữ riêng." if value=="demo" else
                                  self.instruction())
                elif kind=="error":
                    self.engine.reset_tracking()
                    self.stale=True
                    self.status.set("Chưa kết nối được")
                    self.status_dot.configure(fg="#CB5C42")
                    self.hint.set(value)
                    self.notice.configure(bg="#FFF5F1",fg="#A54D35")
                elif kind=="ended":
                    self._ended()
                    if not self.stopping and value=="video" and not self.stale:
                        self.status.set("Đã phát hết video")
                    self.dirty=True
            packet=self.worker.latest()
            if packet is not None:
                self.packet=packet
                self.fps.set("ẢNH TĨNH" if self.mode=="image" else f"{packet.fps:.0f} FPS")
                self.dirty=True
                if self.stale and not self.stopping and self.worker.thread.is_alive():
                    self.stale=False
                    self.status.set("Robot sẵn sàng vẽ theo")
            if (self.worker.thread.is_alive() and self.packet and self.mode in ("camera","video") and not self.paused
                    and not self.stale and not self.stopping and now-self.packet.captured_at>.6):
                self.engine.reset_tracking()
                self.stale=True
                self.status.set("Đang chờ khung hình · Đã nhấc bút")
                self.dirty=True
        if self.dirty and now-self.last_paint>.035:
            self._paint()
            self.last_paint=now
        self.poll_id=self.root.after(12,self._poll)
