from pathlib import Path
from tempfile import TemporaryDirectory
from queue import Empty
from unittest.mock import Mock,patch
from xml.etree import ElementTree
import subprocess
import sys
import time
import tkinter as tk
import unittest

import cv2
import numpy as np
from PIL import Image

from drawing_robot.core import DrawingEngine,HandObservation
from drawing_robot.export import export_png,export_svg
from drawing_robot.graphics import PaperRenderer,camera_image
from drawing_robot.vision import MODEL,create_detector,detect
from drawing_robot.worker import Frame,HandWorker,Sample


def make_drawing():
    engine=DrawingEngine()
    for i in range(20):
        engine.update_pointer((.2+i*.012,.3),True,i/30)
    engine.update_pointer((.44,.3),False,.7)
    engine.change_style("#EA6A42",5)
    for i in range(20):
        engine.update_pointer((.5+i*.01,.7),True,.8+i/30)
    engine.lift()
    return engine


class DrawingRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(MODEL.is_file(),"Hand model required")
    def test_hand_model_rejects_blank_frames(self):
        engine=DrawingEngine()
        with create_detector() as model:
            for i in range(6):
                observation=detect(model,np.zeros((480,640,3),np.uint8),i*33)
                self.assertFalse(observation.landmarks)
                engine.update(observation,i/30)
        self.assertFalse(engine.all_strokes())

    @unittest.skipUnless((MODEL.parent.parent/"artifacts"/"thumb_up.jpg").is_file(),"Optional official hand sample")
    def test_real_hand_sample_has_21_landmarks_and_one_image_never_starts_ink(self):
        image=cv2.imread(str(MODEL.parent.parent/"artifacts"/"thumb_up.jpg"))
        with create_detector() as model:
            observation=detect(model,image,0)
        self.assertEqual(len(observation.landmarks),21)
        self.assertEqual(len(observation.world_landmarks),21)
        engine=DrawingEngine()
        engine.update(observation,0)
        self.assertFalse(engine.pen_down)
        self.assertFalse(engine.all_strokes())

    def test_svg_and_png_preserve_separate_colored_strokes(self):
        engine=make_drawing()
        self.assertEqual(len(engine.strokes),2)
        with TemporaryDirectory() as directory:
            svg=Path(directory)/"drawing.svg"
            png=Path(directory)/"drawing.png"
            export_svg(engine,svg)
            export_png(engine,png)
            document=ElementTree.parse(svg)
            lines=document.findall("{http://www.w3.org/2000/svg}polyline")
            self.assertEqual(len(lines),2)
            self.assertEqual([p.attrib["stroke"] for p in lines],["#285CE6","#EA6A42"])
            with Image.open(png) as image:
                self.assertEqual(image.size,(1400,1000))
                self.assertEqual(image.getpixel((0,0)),(255,255,255))

    def test_renderer_resizes_and_invalidates_cache_on_undo(self):
        engine=make_drawing()
        renderer=PaperRenderer()
        before=np.asarray(renderer.render(engine,(640,400),False))
        engine.undo()
        after=np.asarray(renderer.render(engine,(640,400),False))
        self.assertGreater(np.count_nonzero(before!=after),100)
        for size in ((40,40),(700,420),(480,600)):
            self.assertEqual(renderer.render(engine,size,True).size,size)
            packet=Frame(np.zeros((480,640,3),np.uint8),HandObservation(),30,time.monotonic())
            self.assertEqual(camera_image(packet,engine,size,source="demo").size,size)

    def test_sample_queue_overflow_marks_a_discontinuity(self):
        worker=HandWorker("demo")
        for i in range(95):
            sample=Sample(HandObservation(),i/30,time.monotonic(),(.5,.5),True)
            worker._publish(sample,Frame(None,HandObservation(),30,time.monotonic()))
        first=worker.samples.get_nowait()
        self.assertTrue(first.discontinuity)
        self.assertLessEqual(worker.samples.qsize(),90)
        self.assertEqual(worker.frames.qsize(),1)

    def test_demo_worker_pauses_and_stops_cleanly(self):
        worker=HandWorker("demo")
        worker.start()
        time.sleep(.23)
        self.assertGreater(worker.samples.qsize(),3)
        worker.pause()
        time.sleep(.08)
        count=worker.samples.qsize()
        time.sleep(.08)
        self.assertEqual(count,worker.samples.qsize())
        worker.stop()
        worker.thread.join(2)
        self.assertFalse(worker.thread.is_alive())

    def test_camera_error_releases_model_and_signals_end(self):
        model=Mock()
        with patch("drawing_robot.worker.ensure_model",return_value=MODEL),patch("drawing_robot.worker.create_detector",return_value=model),patch("drawing_robot.worker.open_camera",side_effect=RuntimeError("Camera unavailable")):
            worker=HandWorker("camera")
            worker.start()
            worker.thread.join(2)
        messages=[]
        while not worker.messages.empty():
            messages.append(worker.messages.get_nowait())
        self.assertIn(("error","Camera unavailable"),messages)
        self.assertIn(("ended","camera"),messages)
        model.close.assert_called_once()

    @unittest.skipUnless(MODEL.is_file(),"Hand model required")
    def test_video_cli_does_not_draw_on_blank_frames(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/"blank.avi"
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),30,(160,120))
            self.assertTrue(writer.isOpened())
            for i in range(10):
                writer.write(np.full((120,160,3),i*3,np.uint8))
            writer.release()
            result=subprocess.run([sys.executable,str(MODEL.parent.parent/"drawing_app.py"),"--video",str(path),"--headless"],capture_output=True,text=True,timeout=25)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('"strokes": 0',result.stdout)


class DrawingDesktopTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root=tk.Tk()
        except tk.TclError as error:
            self.skipTest(str(error))
        self.root.withdraw()
        self.errors=[]
        self.root.report_callback_exception=lambda *error:self.errors.append(error)
        from drawing_robot.desktop import DrawingDesktop
        self.app=DrawingDesktop(self.root)
        self.root.update_idletasks()

    def tearDown(self):
        if hasattr(self,"app") and self.app.worker:
            self.app.worker.stop()
            self.app.worker.thread.join(2)
        if hasattr(self,"root"):
            self.root.destroy()
        self.assertFalse(getattr(self,"errors",[]),"Unexpected native UI callback exception")

    def pump(self,seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            self.root.update()
            time.sleep(.008)

    def test_mouse_draw_and_stop_cannot_keep_producing_ink(self):
        app=self.app
        app.start("mouse")
        app.mouse_uv=(.3,.3)
        app.mouse_pressed=True
        self.pump(.4)
        self.assertTrue(app.engine.pen_down)
        app.stop()
        count=len(app.engine.strokes)
        app.mouse_pressed=True
        self.pump(.3)
        self.assertFalse(app.engine.pen_down)
        self.assertEqual(len(app.engine.strokes),count)
        self.assertEqual(self.root.cget("bg"),"white")

    def test_demo_does_not_destroy_the_users_drawing(self):
        app=self.app
        app.user_engine=make_drawing()
        app.engine=app.user_engine
        app.start("demo")
        self.pump(.15)
        self.assertIsNot(app.engine,app.user_engine)
        app.stop()
        app.worker.thread.join(2)
        app.start("mouse")
        self.assertIs(app.engine,app.user_engine)
        self.assertEqual(len(app.engine.strokes),2)

    def test_export_dialog_lifts_pen_and_resumes_without_a_bridge(self):
        app=self.app
        app.start("mouse")
        app.engine=make_drawing()
        with TemporaryDirectory() as directory:
            target=Path(directory)/"test.svg"
            with patch("drawing_robot.desktop.filedialog.asksaveasfilename",return_value=str(target)):
                app.save("svg")
            self.assertTrue(target.is_file())
            self.assertFalse(app.engine.pen_down)
            self.assertFalse(app.paused)


if __name__=="__main__":
    unittest.main()
