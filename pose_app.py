"""Human Pose to Robot Motion. Launch the white desktop UI with run.bat."""
from pathlib import Path
import argparse
import json
import sys


def main():
    parser=argparse.ArgumentParser(description="Human Pose -> robot 2D; shoulder/elbow/wrist angles, Raise/Lower/Wave")
    source=parser.add_mutually_exclusive_group()
    source.add_argument("--demo",action="store_true")
    source.add_argument("--video",type=Path)
    source.add_argument("--image",type=Path)
    parser.add_argument("--camera",type=int,default=0)
    parser.add_argument("--start",action="store_true",help="Open camera automatically")
    parser.add_argument("--download-model",action="store_true")
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--max-frames",type=int,default=300)
    parser.add_argument("--arm",choices=("auto","left","right"),default="auto")
    parser.add_argument("--snapshot",type=Path,help="Save an image of this application window only")
    parser.add_argument("--exit-after",type=float,help="Close after N seconds; for UI checks")
    args=parser.parse_args()
    if args.download_model:
        from pose_robot.vision import ensure_model
        print(ensure_model())
        return
    kind="demo" if args.demo else "video" if args.video else "image" if args.image else "camera"
    source=args.video or args.image or args.camera
    if args.headless:
        from pose_robot.worker import PoseWorker
        from queue import Empty
        import time
        if args.max_frames<1:
            parser.error("--max-frames must be positive")
        worker=PoseWorker(kind,source,args.arm)
        worker.start()
        count=0
        error=None
        latest=None
        try:
            while True:
                while True:
                    try:
                        message,value=worker.messages.get_nowait()
                    except Empty:
                        break
                    if message=="error":
                        error=value
                    elif message=="action":
                        print(json.dumps({"event":value[1],"side":value[0]}))
                packet=worker.latest()
                if packet:
                    count+=1
                    latest=packet.state
                    if count>=args.max_frames:
                        break
                if not worker.thread.is_alive() and worker.frames.empty():
                    break
                time.sleep(.005)
        finally:
            worker.stop()
            worker.thread.join(timeout=10)
        while not worker.messages.empty():
            message,value=worker.messages.get_nowait()
            if message=="error":
                error=value
            elif message=="action":
                print(json.dumps({"event":value[1],"side":value[0]}))
        if error:
            raise RuntimeError(error)
        print(json.dumps({"frames":count,"valid":bool(latest and latest.valid),"angle":latest.angle if latest else None,
                          "side":latest.side if latest else None,"action":latest.action if latest else None}))
        return
    import tkinter as tk
    from pose_robot.desktop import PoseDesktop
    root=tk.Tk()
    initial=kind if args.demo or args.video or args.image or args.start else None
    app=PoseDesktop(root,initial,source)
    app.arm.set({"auto":"Tự động","left":"Tay trái","right":"Tay phải"}[args.arm])
    if args.snapshot:
        root.after(2400,lambda:app.snapshot(args.snapshot))
    if args.exit_after is not None:
        root.after(max(300,int(args.exit_after*1000)),app.close)
    root.mainloop()


if __name__=="__main__":
    try:
        main()
    except (RuntimeError,OSError,ValueError) as error:
        print(f"LOI: {error}",file=sys.stderr)
        sys.exit(1)
