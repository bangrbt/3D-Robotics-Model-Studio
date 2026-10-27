"""Live Air Drawing -> robot pen. Default launcher for Motion Studio."""
from pathlib import Path
import argparse
import json
from queue import Empty
import sys
import time


def main():
    parser=argparse.ArgumentParser(description="Webcam fingertip drawing and real-time robot pen following")
    source=parser.add_mutually_exclusive_group()
    source.add_argument("--demo",action="store_true")
    source.add_argument("--mouse",action="store_true")
    source.add_argument("--video",type=Path)
    source.add_argument("--image",type=Path)
    parser.add_argument("--camera",type=int,default=0)
    parser.add_argument("--start",action="store_true")
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--max-frames",type=int,default=300)
    parser.add_argument("--download-model",action="store_true")
    parser.add_argument("--snapshot",type=Path)
    parser.add_argument("--exit-after",type=float)
    args=parser.parse_args()
    if args.download_model:
        from drawing_robot.vision import ensure_model
        print(ensure_model())
        return
    kind="demo" if args.demo else "mouse" if args.mouse else "video" if args.video else "image" if args.image else "camera"
    source=args.video or args.image or args.camera
    if args.headless:
        from drawing_robot.core import DrawingEngine
        from drawing_robot.worker import HandWorker
        if kind=="mouse" or args.max_frames<1:
            parser.error("Headless mode needs camera/video/image/demo and --max-frames > 0")
        engine=DrawingEngine()
        worker=HandWorker(kind,source)
        worker.start()
        count,error=0,None
        try:
            while True:
                while True:
                    try:
                        sample=worker.samples.get_nowait()
                    except Empty:
                        break
                    if sample.discontinuity:
                        engine.reset_tracking()
                    if kind=="demo":
                        engine.update_pointer(sample.manual_uv,sample.manual_pressed,sample.time,"demo")
                    else:
                        engine.update(sample.observation,sample.time)
                    count+=1
                worker.latest()
                if count>=args.max_frames or (not worker.thread.is_alive() and worker.samples.empty()):
                    break
                time.sleep(.005)
        finally:
            worker.stop()
            worker.thread.join(10)
        while not worker.messages.empty():
            message,value=worker.messages.get_nowait()
            if message=="error":
                error=value
        if error:
            raise RuntimeError(error)
        engine.lift()
        print(json.dumps({"frames":count,"strokes":len(engine.strokes),"shape":engine.shape.key,"tracked_pointer":engine.valid}))
        return
    import tkinter as tk
    from drawing_robot.desktop import DrawingDesktop
    root=tk.Tk()
    app=DrawingDesktop(root,kind if args.demo or args.mouse or args.video or args.image or args.start else None,source)
    if args.snapshot:
        root.after(4500,lambda:app.snapshot(args.snapshot))
    if args.exit_after is not None:
        root.after(max(500,int(args.exit_after*1000)),app.close)
    root.mainloop()


if __name__=="__main__":
    try:
        main()
    except (RuntimeError,OSError,ValueError) as error:
        print(f"LOI: {error}",file=sys.stderr)
        sys.exit(1)
